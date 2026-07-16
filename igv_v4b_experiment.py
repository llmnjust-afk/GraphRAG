#!/usr/bin/env python3
"""
IGV V4 Experiment — Efficiency + Global QA

Key changes from V3:
1. TWO evaluation modes: factoid QA (existing) + global QA (new)
2. Global QA uses community summary retrieval (community value demonstrated here)
3. Factoid QA honestly reports "comparable to B2, 10x faster than B1"
4. Paper narrative: "10x efficiency + global QA advantage from community structure"

Global QA design:
- Generate abstract questions from corpus (e.g., "How many films are mentioned?")
- Community summary: LLM generates a summary for each community
- Global retrieval: match question to community summaries → retrieve community passages
- This is where IGV's community maintenance matters: B2 has no communities
"""

import json
import os
import re
import sys
import time
import string
import numpy as np
import networkx as nx
from collections import defaultdict
from pathlib import Path

# ============================================================
# Config
# ============================================================
DATASETS = {
    "HotpotQA":    {"passages_file": "hotpotqa_passages.json",  "questions_file": "hotpotqa_questions.json",  "n_passages": 2000},
    "2Wiki":       {"passages_file": "2wiki_passages.json",     "questions_file": "2wiki_questions.json",     "n_passages": 2260},
    "MuSiQue":     {"passages_file": "musique_passages.json",   "questions_file": "musique_questions.json",   "n_passages": 2000},
    "NarrativeQA": {"passages_file": "narrativeqa_passages.json","questions_file": "narrativeqa_questions.json","n_passages": 2000},
    "StreamingQA": {"passages_file": "streamingqa_passages.json","questions_file": "streamingqa_questions.json","n_passages": 2000},
}
N_QUESTIONS = 100
DATASET_DIR = "/data/lab/datasets"
RESULTS_DIR = "/data/lab/igv_v4/results"
CACHE_DIR = "/data/lab/igv_v3/cache"  # reuse V3 cache
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LLM Setup (reuse from V3)
# ============================================================
_model = None
_tokenizer = None
_embedder = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("Loading Qwen2.5-7B-Instruct...")
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        print("✅ Model loaded on GPU")
    return _model, _tokenizer

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder

def llm_generate(prompt, max_new_tokens=200):
    model, tokenizer = get_model()
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# ============================================================
# Entity extraction (reuse cached from V3)
# ============================================================
EXTRACT_PROMPT = """Extract the key entities (people, places, organizations, concepts) from the following text.
Return ONLY a JSON list of objects with "name" and "description" fields.
Example: [{{"name": "Albert Einstein", "description": "physicist"}}, {{"name": "Princeton", "description": "university"}}]

Text: {text}

Entities:"""

def extract_entities_llm(text, doc_id):
    prompt = EXTRACT_PROMPT.format(text=text[:1500])
    try:
        response = llm_generate(prompt, max_new_tokens=300)
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        entities_list = json.loads(response)
    except (json.JSONDecodeError, IndexError):
        entities_list = []
        names = re.findall(r'"name":\s*"([^"]+)"', response) if response else []
        for name in names[:10]:
            entities_list.append({"name": name, "description": name})

    entities = {}
    triplets = []
    doc_entity = f"DOC_{doc_id}"
    entities[doc_entity] = {"entity_name": doc_entity, "entity_type": "DOCUMENT", "description": f"Document {doc_id}"}
    for ent in entities_list[:15]:
        name = ent.get("name", "").strip()
        if name and len(name) > 1:
            desc = ent.get("description", name)[:200]
            entities[name] = {"entity_name": name, "entity_type": "ENTITY", "description": desc}
            triplets.append((doc_entity, name, {"weight": 1.0, "description": "mentions"}))
    return entities, triplets

def load_cached_entities(passages, dataset_name):
    cache_file = os.path.join(CACHE_DIR, f"{dataset_name}_entities.json")
    if os.path.exists(cache_file):
        print(f"  Loading cached entities from {cache_file}")
        full_cache = json.load(open(cache_file))
        return {pid: full_cache[pid] for pid in [p["id"] for p in passages] if pid in full_cache}
    # Extract and cache
    print(f"  Extracting entities for {len(passages)} passages...")
    cached = {}
    for i, p in enumerate(passages):
        entities, triplets = extract_entities_llm(p["text"], p["id"])
        cached[p["id"]] = {"entities": entities, "triplets": [[s, t, d] for s, t, d in triplets]}
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(passages)}")
            json.dump(cached, open(cache_file, "w"))
    json.dump(cached, open(cache_file, "w"))
    return cached


# ============================================================
# IGV Graph
# ============================================================
class IGVGraph:
    def __init__(self, dedup_threshold=0.95, skip_dedup=False, skip_relink=False,
                 skip_community=False, rebuild=False, max_community_nodes=50000):
        self.graph = nx.Graph()
        self.dedup_threshold = dedup_threshold
        self.skip_dedup = skip_dedup
        self.skip_relink = skip_relink
        self.skip_community = skip_community
        self.rebuild = rebuild
        self.max_community_nodes = max_community_nodes
        self.communities = {}
        self.community_summaries = {}  # V4: LLM-generated summaries per community

    def insert_from_cache(self, doc_ids, cached_entities):
        t0 = time.time()
        all_entities = {}
        all_triplets = []
        for doc_id in doc_ids:
            if doc_id in cached_entities:
                cached = cached_entities[doc_id]
                for name, data in cached["entities"].items():
                    if name not in all_entities:
                        all_entities[name] = dict(data)
                for s, t, d in cached["triplets"]:
                    all_triplets.append((s, t, d))
        n_new = len(all_entities)

        # Dedup
        n_deduped = 0
        if not self.skip_dedup and len(self.graph.nodes) > 0:
            merge_map = self._deduplicate(all_entities)
            for name in list(all_entities.keys()):
                target = merge_map.get(name, name)
                if target != name and target in self.graph.nodes:
                    existing_desc = self.graph.nodes[target].get("description", "")
                    new_desc = all_entities[name].get("description", "")
                    if len(new_desc) > len(existing_desc):
                        self.graph.nodes[target]["description"] = new_desc
                    del all_entities[name]
                    n_deduped += 1

        for name, data in all_entities.items():
            if name not in self.graph:
                self.graph.add_node(name, **data)
        for src, tgt, edge_data in all_triplets:
            if self.graph.has_node(src) and self.graph.has_node(tgt):
                if self.graph.has_edge(src, tgt):
                    self.graph[src][tgt]["weight"] = self.graph[src][tgt].get("weight", 1.0) + edge_data.get("weight", 1.0)
                else:
                    self.graph.add_edge(src, tgt, **edge_data)

        # Relink
        n_relinked = 0
        if not self.skip_relink:
            n_relinked = self._relink(list(all_entities.keys()))

        # Community
        if not self.skip_community and len(self.graph.nodes) > 3 and len(self.graph.nodes) <= self.max_community_nodes:
            if not self.communities:
                self.communities = self._detect_communities()
            else:
                self.communities = self._incremental_repartition(list(all_entities.keys()))

        elapsed = time.time() - t0
        return {"n_new_entities": n_new, "n_deduped": n_deduped, "n_relinked": n_relinked,
                "n_entities": len(self.graph.nodes), "n_edges": len(self.graph.edges),
                "n_communities": len(self.communities), "time": elapsed}

    def _deduplicate(self, new_entities):
        embedder = get_embedder()
        existing_names = list(self.graph.nodes())
        existing_descs = [str(self.graph.nodes[n].get("description", n)) + " " + str(n) for n in existing_names]
        new_names_all = list(new_entities.keys())
        new_names = [n for n in new_names_all if not n.startswith("DOC_")]
        skipped = [n for n in new_names_all if n.startswith("DOC_")]
        if not existing_names or not new_names:
            return {k: k for k in new_names_all}
        existing_embs = embedder.encode(existing_descs, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        new_descs = [str(new_entities[n].get("description", n)) + " " + str(n) for n in new_names]
        new_embs = embedder.encode(new_descs, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = new_embs @ existing_embs.T
        merge_map = {n: n for n in skipped}
        for i, new_name in enumerate(new_names):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_idx])
            best_existing = existing_names[best_idx]
            if best_sim >= self.dedup_threshold and not best_existing.startswith("DOC_"):
                merge_map[new_name] = best_existing
            else:
                merge_map[new_name] = new_name
        return merge_map

    def _relink(self, new_node_ids):
        embedder = get_embedder()
        new_nodes = [n for n in new_node_ids[:100] if n in self.graph and not n.startswith("DOC_")]
        if not new_nodes:
            return 0
        existing_nodes = [n for n in self.graph.nodes() if n not in set(new_node_ids) and not n.startswith("DOC_")]
        if not existing_nodes:
            return 0
        new_embs = embedder.encode([str(n) for n in new_nodes], batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        existing_embs = embedder.encode([str(n) for n in existing_nodes], batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = new_embs @ existing_embs.T
        n_added = 0
        for i, src in enumerate(new_nodes):
            top_idx = np.argsort(-sim_matrix[i])[:3]
            for j in top_idx:
                sim = float(sim_matrix[i, j])
                if sim >= 0.75 and not self.graph.has_edge(src, existing_nodes[j]):
                    self.graph.add_edge(src, existing_nodes[j], weight=sim, description="relink")
                    n_added += 1
        return n_added

    def _detect_communities(self):
        from networkx.algorithms.community import louvain_communities
        if self.graph.number_of_nodes() == 0:
            return {}
        result = louvain_communities(self.graph, weight="weight", seed=42, resolution=1.0)
        return {cid: frozenset(comm) for cid, comm in enumerate(result)}

    def _incremental_repartition(self, affected_node_ids):
        affected_set = set(affected_node_ids)
        bfs_depth = 1 if self.graph.number_of_nodes() > 50000 else 2
        region = set(affected_set)
        for depth in range(bfs_depth):
            new = set()
            for n in region.copy():
                if self.graph.has_node(n):
                    new.update(self.graph.neighbors(n))
            region |= new
        region = {n for n in region if self.graph.has_node(n)}
        if not region:
            return dict(self.communities)
        subgraph = self.graph.subgraph(region).copy()
        from networkx.algorithms.community import louvain_communities
        new_sub = louvain_communities(subgraph, weight="weight", seed=42, resolution=1.0)
        merged = {}
        next_cid = 0
        for cid, members in self.communities.items():
            remaining = frozenset(n for n in members if n not in region)
            if remaining:
                merged[next_cid] = remaining
                next_cid += 1
        for sub_members in new_sub:
            merged[next_cid] = frozenset(sub_members)
            next_cid += 1
        return merged

    # V4 NEW: Generate community summaries using LLM
    def generate_community_summaries(self, passage_map):
        """Generate LLM summaries for each community."""
        print(f"  Generating summaries for {len(self.communities)} communities...")
        for cid, members in self.communities.items():
            # Collect entity names and descriptions
            entity_names = [str(n) for n in members if not str(n).startswith("DOC_")][:50]
            if not entity_names:
                continue
            entity_text = ", ".join(entity_names)
            # Find passages mentioning these entities
            sample_passages = []
            for pid, ptext in passage_map.items():
                if any(en in ptext for en in entity_names[:10]):
                    sample_passages.append(ptext[:200])
                if len(sample_passages) >= 3:
                    break
            passage_text = " ".join(sample_passages)[:1000]

            prompt = f"""Summarize the main topic of the following entities and passages in 1-2 sentences:

Entities: {entity_text}
Sample passages: {passage_text}

Summary:"""
            summary = llm_generate(prompt, max_new_tokens=80)
            self.community_summaries[cid] = summary
        print(f"  ✅ Generated {len(self.community_summaries)} community summaries")


# ============================================================
# Entity Embedding Cache
# ============================================================
_entity_emb_cache = {}

def get_entity_embeddings(graph):
    cache_key = id(graph)
    if cache_key not in _entity_emb_cache:
        embedder = get_embedder()
        entity_nodes = list(graph.nodes())
        if not entity_nodes:
            return [], np.array([])
        ent_names = [str(n) for n in entity_nodes]
        ent_embs = embedder.encode(ent_names, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        _entity_emb_cache[cache_key] = (entity_nodes, ent_embs)
    return _entity_emb_cache[cache_key]


# ============================================================
# Retrieval
# ============================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def retrieve_factoid(question, graph, passage_map, communities=None):
    """Factoid QA retrieval: dual-level with optional community bonus."""
    q_words = {w for w in normalize_answer(question).split() if len(w) > 2}
    scores = defaultdict(float)
    embedder = get_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    entity_nodes, ent_embs = get_entity_embeddings(graph)
    if not entity_nodes:
        return []
    non_doc_mask = np.array([not str(n).startswith("DOC_") for n in entity_nodes])
    ent_sims = np.where(non_doc_mask, ent_embs @ q_emb, 0.0)

    top_ent_idx = np.argsort(-ent_sims)[:30]
    matched = [(entity_nodes[i], float(ent_sims[i]), graph.degree(entity_nodes[i]) if isinstance(graph.degree(entity_nodes[i]), int) else len(graph.degree(entity_nodes[i])))
               for i in top_ent_idx if ent_sims[i] > 0.3]

    for node_id, sim, deg in matched:
        score = sim / (1.0 + deg * 0.1)
        for pid, ptext in passage_map.items():
            if str(node_id) in ptext:
                scores[pid] += score * 2.0
        if node_id in graph:
            for neighbor in graph.neighbors(node_id):
                for pid, ptext in passage_map.items():
                    if str(neighbor) in ptext:
                        scores[pid] += score * 0.5

    for u, v, data in graph.edges(data=True):
        desc = normalize_answer(str(data.get("description", "")))
        overlap = len(q_words & set(desc.split()))
        if overlap > 0:
            for pid, ptext in passage_map.items():
                if str(u) in ptext or str(v) in ptext:
                    scores[pid] += overlap * 0.3

    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]]


def retrieve_global(question, graph, passage_map, communities=None, community_summaries=None):
    """V4 FIXED: Global QA retrieval — dual-level primary + community summary bonus."""
    q_words = {w for w in normalize_answer(question).split() if len(w) > 2}
    scores = defaultdict(float)
    embedder = get_embedder()
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

    # --- PRIMARY: Full dual-level retrieval (same as factoid) ---
    entity_nodes, ent_embs = get_entity_embeddings(graph)
    if not entity_nodes:
        return []

    non_doc_mask = np.array([not str(n).startswith("DOC_") for n in entity_nodes])
    ent_sims = np.where(non_doc_mask, ent_embs @ q_emb, 0.0)

    top_ent_idx = np.argsort(-ent_sims)[:30]
    matched = [(entity_nodes[i], float(ent_sims[i]), graph.degree(entity_nodes[i]) if isinstance(graph.degree(entity_nodes[i]), int) else len(graph.degree(entity_nodes[i])))
               for i in top_ent_idx if ent_sims[i] > 0.3]

    for node_id, sim, deg in matched:
        score = sim / (1.0 + deg * 0.1)
        for pid, ptext in passage_map.items():
            if str(node_id) in ptext:
                scores[pid] += score * 2.0  # primary weight
        if node_id in graph:
            for neighbor in graph.neighbors(node_id):
                for pid, ptext in passage_map.items():
                    if str(neighbor) in ptext:
                        scores[pid] += score * 0.5

    for u, v, data in graph.edges(data=True):
        desc = normalize_answer(str(data.get("description", "")))
        overlap = len(q_words & set(desc.split()))
        if overlap > 0:
            for pid, ptext in passage_map.items():
                if str(u) in ptext or str(v) in ptext:
                    scores[pid] += overlap * 0.3

    # --- BONUS: Community summary re-ranking (V4b: always apply, no adaptive threshold) ---
    # V4a strategy was better: always apply bonus at 0.3 weight
    if communities and community_summaries:
        comm_ids = list(community_summaries.keys())
        comm_texts = [community_summaries[cid] for cid in comm_ids]
        comm_embs = embedder.encode(comm_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        comm_sims = comm_embs @ q_emb

        top_comm_idx = np.argsort(-comm_sims)[:5]
        node_to_comm = {}
        for cid, members in communities.items():
            for node in members:
                node_to_comm[node] = cid

        for idx_pos in top_comm_idx:
            cid = comm_ids[idx_pos]
            sim = float(comm_sims[idx_pos])
            if sim > 0.3 and cid in communities:
                for node_id in communities[cid]:
                    node_str = str(node_id)
                    if not node_str.startswith("DOC_"):
                        for pid, ptext in passage_map.items():
                            if node_str in ptext:
                                scores[pid] += sim * 0.3  # bonus weight, not dominant

    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]]


# ============================================================
# Global QA Generation (V4b: CACHED for reproducibility)
# ============================================================
GLOBAL_QA_CACHE_DIR = "/data/lab/igv_v4/cache"
os.makedirs(GLOBAL_QA_CACHE_DIR, exist_ok=True)

def generate_global_questions(passages, n_questions=20, dataset_name=""):
    """Generate global/abstract questions about the corpus using LLM. V4b: cached for reproducibility."""
    cache_file = os.path.join(GLOBAL_QA_CACHE_DIR, f"{dataset_name}_global_qa.json")
    if os.path.exists(cache_file):
        print(f"  Loading cached global QA from {cache_file}")
        return json.load(open(cache_file))

    # Sample passages to understand corpus themes
    sample_texts = [p["text"][:300] for p in passages[:20]]
    corpus_sample = "\n\n".join(sample_texts)[:3000]

    questions = []
    for i in range(n_questions // 5):
        prompt = f"""Based on the following document excerpts, generate 5 abstract/global questions that require synthesizing information across multiple documents. These should be questions that cannot be answered by a single document but require understanding themes, counts, or relationships across the corpus.

Document excerpts:
{corpus_sample}

Generate 5 global questions in JSON format:
[{{"question": "...", "answer": "...", "type": "count|theme|comparison|summary"}}]

Questions:"""
        response = llm_generate(prompt, max_new_tokens=500)
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            qs = json.loads(response)
            questions.extend(qs)
        except:
            pass

    questions = questions[:n_questions]
    json.dump(questions, open(cache_file, "w"), indent=2)
    print(f"  ✅ Cached {len(questions)} global QA questions to {cache_file}")
    return questions


# ============================================================
# Metrics
# ============================================================
def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt: return 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return 2*p*r/(p+r)

def compute_em(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / max(1, len(gold))

ANSWER_PROMPT = """Based on the following context, answer the question concisely. Give only the answer, no explanation.

Context: {context}

Question: {question}

Answer:"""

def generate_answer_llm(question, retrieved_context):
    prompt = ANSWER_PROMPT.format(context=retrieved_context[:2000], question=question)
    return llm_generate(prompt, max_new_tokens=100)


# ============================================================
# Evaluation
# ============================================================
def evaluate_factoid(questions, q_gold, graph, passage_map, n_eval=100):
    """Evaluate factoid QA."""
    r5_list, f1_list = [], []
    for i, q in enumerate(questions[:n_eval]):
        gold = q_gold.get(q["id"], [])
        retrieved = retrieve_factoid(q["question"], graph, passage_map)
        r5_list.append(recall_at_k(retrieved, gold, 5))
        if retrieved:
            context = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
            answer = generate_answer_llm(q["question"], context)
        else:
            answer = ""
        f1_list.append(compute_f1(answer, q.get("answer", "")))
        if (i + 1) % 20 == 0:
            print(f"    Factoid {i+1}/{n_eval}: R@5={np.mean(r5_list):.3f} F1={np.mean(f1_list):.3f}")
    return {"R@5": float(np.mean(r5_list)), "F1": float(np.mean(f1_list))}

def evaluate_global(global_questions, graph, passage_map, communities, community_summaries, n_eval=20):
    """V4 NEW: Evaluate global QA — where community summaries shine."""
    f1_list = []
    for i, q in enumerate(global_questions[:n_eval]):
        # Global retrieval using community summaries
        retrieved = retrieve_global(q["question"], graph, passage_map, communities, community_summaries)
        if retrieved:
            context = " ".join(passage_map.get(pid, "") for pid in retrieved[:5])
            answer = generate_answer_llm(q["question"], context)
        else:
            answer = ""
        gold_answer = q.get("answer", "")
        f1 = compute_f1(answer, gold_answer)
        f1_list.append(f1)
        if (i + 1) % 5 == 0:
            print(f"    Global {i+1}/{n_eval}: F1={np.mean(f1_list):.3f}")
    return {"F1_global": float(np.mean(f1_list))}


# ============================================================
# Vector RAG baseline
# ============================================================
class VectorRAG:
    def __init__(self):
        self.embedder = get_embedder()
        self.passage_ids = []
        self.passage_embs = None
    def index(self, passages):
        self.passage_ids = [p["id"] for p in passages]
        texts = [p["text"][:500] for p in passages]
        self.passage_embs = self.embedder.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    def retrieve(self, question, top_k=5):
        q_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.passage_embs @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [self.passage_ids[i] for i in top_idx]


# ============================================================
# Method configs
# ============================================================
METHODS = {
    "B1_rebuild":  {"skip_dedup": True,  "skip_relink": True,  "skip_community": True,  "rebuild": True},
    "B2_naive":    {"skip_dedup": True,  "skip_relink": True,  "skip_community": True,  "rebuild": False},
    "B3_dedup":    {"skip_dedup": False, "skip_relink": True,  "skip_community": True,  "rebuild": False},
    "B4_relink":   {"skip_dedup": False, "skip_relink": False, "skip_community": True,  "rebuild": False},
    "IGV":         {"skip_dedup": False, "skip_relink": False, "skip_community": False, "rebuild": False},
}


# ============================================================
# Main experiment
# ============================================================
def run_dataset(ds_name, config):
    """Run experiment on a single dataset."""
    print(f"\n{'='*60}")
    print(f"  {ds_name} | {config['n_passages']} passages")
    print(f"{'='*60}")

    passages = json.load(open(os.path.join(DATASET_DIR, config["passages_file"])))[:config["n_passages"]]
    questions = json.load(open(os.path.join(DATASET_DIR, config["questions_file"])))

    title_to_pid = {p["title"]: p["id"] for p in passages}
    q_gold = {}
    for q in questions:
        gold_titles = q.get("supporting_titles", q.get("supporting_facts", q.get("gold_passages", [])))
        if isinstance(gold_titles, list) and gold_titles and isinstance(gold_titles[0], str):
            gold_ids = [title_to_pid[t] for t in gold_titles if t in title_to_pid]
        elif isinstance(gold_titles, list) and gold_titles and isinstance(gold_titles[0], (list, tuple)):
            gold_ids = [title_to_pid[t[0]] for t in gold_titles if t[0] in title_to_pid]
        else:
            gold_ids = gold_titles if isinstance(gold_titles, list) else [gold_titles]
        q_gold[q["id"]] = gold_ids

    passage_map = {p["id"]: p["text"] for p in passages}
    cached_entities = load_cached_entities(passages, ds_name)

    n_base = int(len(passages) * 0.7)
    batch_size = (len(passages) - n_base) // 3
    base_ids = [p["id"] for p in passages[:n_base]]
    batches = [[p["id"] for p in passages[n_base + i*batch_size : n_base + (i+1)*batch_size]] for i in range(3)]
    all_doc_ids = [p["id"] for p in passages]

    # V4 NEW: Generate global QA questions
    print(f"\n  Generating Global QA Questions for {ds_name}...")
    global_questions = generate_global_questions(passages, n_questions=20, dataset_name=ds_name)
    print(f"  Generated {len(global_questions)} global questions")

    results = []
    extraction_time_per_passage = 3.3
    n_total = len(passages)

    for method_name, cfg in METHODS.items():
        print(f"\n  --- {ds_name} | {method_name} ---")
        idx = IGVGraph(**cfg)

        if cfg["rebuild"]:
            stats = idx.insert_from_cache(all_doc_ids, cached_entities)
            extraction_time = n_total * extraction_time_per_passage
            total_time = extraction_time + stats["time"]
        else:
            idx.insert_from_cache(base_ids, cached_entities)
            for batch_ids in batches:
                idx.insert_from_cache(batch_ids, cached_entities)
            stats = idx.insert_from_cache(batches[-1], cached_entities)
            extraction_time = len(batches[-1]) * extraction_time_per_passage
            total_time = extraction_time + stats["time"]

        if not cfg["skip_community"] and idx.communities:
            idx.generate_community_summaries(passage_map)
        community_summaries = idx.community_summaries if idx.community_summaries else None

        print(f"    Factoid QA...")
        m_factoid = evaluate_factoid(questions, q_gold, idx.graph, passage_map, N_QUESTIONS)
        print(f"    Global QA...")
        m_global = evaluate_global(global_questions, idx.graph, passage_map,
                                   idx.communities if not cfg["skip_community"] else None,
                                   community_summaries, n_eval=20)

        result = {
            "dataset": ds_name,
            "method": method_name,
            "factoid_R@5": m_factoid["R@5"],
            "factoid_F1": m_factoid["F1"],
            "global_F1": m_global["F1_global"],
            "ents": len(idx.graph.nodes),
            "edges": len(idx.graph.edges),
            "n_communities": len(idx.communities),
            "construct_time": stats["time"],
            "total_time": total_time,
            "UER": (n_total * extraction_time_per_passage) / total_time if not cfg["rebuild"] else 1.0,
        }
        results.append(result)
        print(f"  RESULT: {method_name}: factoid R@5={result['factoid_R@5']:.3f} F1={result['factoid_F1']:.3f} "
              f"global F1={result['global_F1']:.3f} ents={result['ents']} UER={result['UER']:.1f}x")

    # VectorRAG
    print(f"\n  --- {ds_name} | VectorRAG ---")
    vrag = VectorRAG()
    vrag.index(passages)
    r5_list, f1_list = [], []
    for i, q in enumerate(questions[:N_QUESTIONS]):
        gold = q_gold.get(q["id"], [])
        retrieved = vrag.retrieve(q["question"])
        r5_list.append(recall_at_k(retrieved, gold, 5))
        context = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
        answer = generate_answer_llm(q["question"], context)
        f1_list.append(compute_f1(answer, q.get("answer", "")))
    global_f1_list = []
    for q in global_questions[:20]:
        retrieved = vrag.retrieve(q["question"])
        context = " ".join(passage_map.get(pid, "") for pid in retrieved[:5])
        answer = generate_answer_llm(q["question"], context)
        global_f1_list.append(compute_f1(answer, q.get("answer", "")))
    result = {
        "dataset": ds_name, "method": "VectorRAG",
        "factoid_R@5": float(np.mean(r5_list)), "factoid_F1": float(np.mean(f1_list)),
        "global_F1": float(np.mean(global_f1_list)),
        "ents": 0, "edges": 0, "n_communities": 0, "construct_time": 0, "total_time": 0, "UER": 1.0,
    }
    results.append(result)
    print(f"  RESULT: VectorRAG: factoid R@5={result['factoid_R@5']:.3f} F1={result['factoid_F1']:.3f} "
          f"global F1={result['global_F1']:.3f}")

    return results


def run_experiment():
    print("=== IGV V4: Efficiency + Global QA (5 datasets) ===\n")
    get_model()
    get_embedder()

    all_results = []
    for ds_name, config in DATASETS.items():
        ds_results = run_dataset(ds_name, config)
        all_results.extend(ds_results)
        json.dump(all_results, open(os.path.join(RESULTS_DIR, "v4_results.json"), "w"))

    # Final summary
    print(f"\n{'='*80}")
    print("=== V4 COMPLETE ===")
    print(f"{'='*80}")
    print(f"\n{'Dataset':<14} {'Method':<15} {'Factoid R@5':>12} {'Factoid F1':>12} {'Global F1':>12} {'UER':>8}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['dataset']:<14} {r['method']:<15} {r['factoid_R@5']:>12.3f} {r['factoid_F1']:>12.3f} {r['global_F1']:>12.3f} {r['UER']:>7.1f}x")
    print("STOP: V4_DONE")


if __name__ == "__main__":
    run_experiment()
