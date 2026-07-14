#!/usr/bin/env python3
"""
IGV V3 Experiment — Fixed ablation design + external baselines + community retrieval

Key fixes vs V2:
1. Entity extraction cached ONCE per dataset, shared across all methods (eliminates LLM noise)
2. Community-level retrieval: communities actually participate in retrieval
3. External baselines: Vector RAG, LightRAG-style, HippoRAG-style (PPR)
4. Larger scale: 5K-10K passages
"""

import json
import os
import re
import sys
import time
import string
import hashlib
import numpy as np
import networkx as nx
from collections import defaultdict
from pathlib import Path

# ============================================================
# Config
# ============================================================
DATASETS = {
    "HotpotQA":    {"passages_file": "hotpotqa_passages.json",  "questions_file": "hotpotqa_questions.json",  "n_passages": 5000},
    "2Wiki":       {"passages_file": "2wiki_passages.json",     "questions_file": "2wiki_questions.json",     "n_passages": 2260},
    "MuSiQue":     {"passages_file": "musique_passages.json",   "questions_file": "musique_questions.json",   "n_passages": 5999},
    "NarrativeQA": {"passages_file": "narrativeqa_passages.json","questions_file": "narrativeqa_questions.json","n_passages": 4937},
    "StreamingQA": {"passages_file": "streamingqa_passages.json","questions_file": "streamingqa_questions.json","n_passages": 4937},
}
N_QUESTIONS = 100
DATASET_DIR = "/data/lab/datasets"
RESULTS_DIR = "/data/lab/igv_v3/results"
CACHE_DIR = "/data/lab/igv_v3/cache"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# LLM Setup
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
        _model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
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

EXTRACT_PROMPT = """Extract the key entities (people, places, organizations, concepts) from the following text.
Return ONLY a JSON list of objects with "name" and "description" fields.
Example: [{{"name": "Albert Einstein", "description": "physicist"}}, {{"name": "Princeton", "description": "university"}}]

Text: {text}

Entities:"""

def extract_entities_llm(text, doc_id):
    """Extract entities using LLM. Returns (entities_dict, triplets_list)."""
    prompt = EXTRACT_PROMPT.format(text=text[:1500])
    try:
        response = llm_generate(prompt, max_new_tokens=300)
        # Parse JSON list
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        entities_list = json.loads(response)
    except (json.JSONDecodeError, IndexError):
        # Fallback: regex extract entity names
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


# ============================================================
# Cached Entity Extraction (V3 KEY FIX: extract ONCE, share across methods)
# ============================================================
def extract_and_cache(passages, dataset_name):
    """Extract entities for all passages ONCE and cache to disk."""
    cache_file = os.path.join(CACHE_DIR, f"{dataset_name}_entities.json")
    if os.path.exists(cache_file):
        print(f"  Loading cached entities from {cache_file}")
        return json.load(open(cache_file))

    print(f"  Extracting entities for {len(passages)} passages (cached to {cache_file})...")
    cached = {}
    for i, p in enumerate(passages):
        entities, triplets = extract_entities_llm(p["text"], p["id"])
        cached[p["id"]] = {
            "entities": entities,
            "triplets": [[s, t, d] for s, t, d in triplets],
        }
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(passages)} passages extracted")
            # Save intermediate
            json.dump(cached, open(cache_file, "w"))

    json.dump(cached, open(cache_file, "w"))
    print(f"  ✅ Cached {len(cached)} passages' entities")
    return cached


# ============================================================
# IGV Graph (V3: uses cached entities, community-level retrieval)
# ============================================================
class IGVGraph:
    def __init__(self, dedup_threshold=0.85, skip_dedup=False, skip_relink=False,
                 skip_community=False, rebuild=False, max_community_nodes=50000):
        self.graph = nx.Graph()
        self.dedup_threshold = dedup_threshold
        self.skip_dedup = skip_dedup
        self.skip_relink = skip_relink
        self.skip_community = skip_community
        self.rebuild = rebuild
        self.max_community_nodes = max_community_nodes
        self.communities = {}

    def insert_from_cache(self, doc_ids, cached_entities):
        """Insert using pre-extracted cached entities. All methods share same extraction."""
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

        # Stage 2: Dedup
        n_deduped = 0
        if not self.skip_dedup and len(self.graph.nodes) > 0:
            merge_map = self._deduplicate(all_entities)
            for name in list(all_entities.keys()):
                target = merge_map.get(name, name)
                if target != name and target in self.graph.nodes:
                    # Merge into existing
                    existing_desc = self.graph.nodes[target].get("description", "")
                    new_desc = all_entities[name].get("description", "")
                    if len(new_desc) > len(existing_desc):
                        self.graph.nodes[target]["description"] = new_desc
                    del all_entities[name]
                    n_deduped += 1

        # Add nodes
        for name, data in all_entities.items():
            if name not in self.graph:
                self.graph.add_node(name, **data)

        # Add edges
        for src, tgt, edge_data in all_triplets:
            if self.graph.has_node(src) and self.graph.has_node(tgt):
                if self.graph.has_edge(src, tgt):
                    self.graph[src][tgt]["weight"] = self.graph[src][tgt].get("weight", 1.0) + edge_data.get("weight", 1.0)
                else:
                    self.graph.add_edge(src, tgt, **edge_data)

        # Stage 3: Relink
        n_relinked = 0
        if not self.skip_relink:
            new_node_ids = list(all_entities.keys())
            n_relinked = self._relink(new_node_ids)

        # Stage 4: Community repartition
        if not self.skip_community and len(self.graph.nodes) > 3 and len(self.graph.nodes) <= self.max_community_nodes:
            if not self.communities:
                self.communities = self._detect_communities()
            else:
                affected = list(all_entities.keys())
                self.communities = self._incremental_repartition(affected)

        elapsed = time.time() - t0
        return {
            "n_new_entities": n_new,
            "n_deduped": n_deduped,
            "n_relinked": n_relinked,
            "n_entities": len(self.graph.nodes),
            "n_edges": len(self.graph.edges),
            "n_communities": len(self.communities),
            "time": elapsed,
        }

    def _deduplicate(self, new_entities):
        embedder = get_embedder()
        existing_names = list(self.graph.nodes())
        existing_descs = [str(self.graph.nodes[n].get("description", n)) + " " + str(n) for n in existing_names]
        new_names = list(new_entities.keys())
        new_descs = [str(new_entities[n].get("description", n)) + " " + str(n) for n in new_names]

        if not existing_names or not new_names:
            return {k: k for k in new_names}

        existing_embs = embedder.encode(existing_descs, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        new_embs = embedder.encode(new_descs, batch_size=256, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        sim_matrix = new_embs @ existing_embs.T

        merge_map = {}
        for i, new_name in enumerate(new_names):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_sim = float(sim_matrix[i, best_idx])
            if best_sim >= self.dedup_threshold:
                merge_map[new_name] = existing_names[best_idx]
            else:
                merge_map[new_name] = new_name
        return merge_map

    def _relink(self, new_node_ids):
        n_added = 0
        for src in new_node_ids[:50]:
            if src not in self.graph:
                continue
            visited = {src: 0}
            frontier = [src]
            for dist in range(1, 3):
                if not frontier:
                    break
                next_frontier = []
                for fnode in frontier:
                    for tgt in self.graph.neighbors(fnode):
                        if tgt in visited:
                            continue
                        visited[tgt] = dist
                        if tgt not in new_node_ids and not self.graph.has_edge(src, tgt):
                            confidence = 1.0 / (1.0 + dist)
                            if confidence >= 0.85:
                                self.graph.add_edge(src, tgt, weight=confidence, description="relink")
                                n_added += 1
                        next_frontier.append(tgt)
                frontier = next_frontier
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


# ============================================================
# Retrieval (V3: community-level retrieval added)
# ============================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def retrieve_graph(question, graph, passage_map, top_k=5, method="dual", communities=None):
    """Graph-based retrieval with community support."""
    q_words = {w for w in normalize_answer(question).split() if len(w) > 2}
    scores = defaultdict(float)

    if method == "keyword":
        for node_id in graph.nodes():
            nw = set(normalize_answer(str(node_id)).split())
            overlap = len(q_words & nw)
            if overlap > 0:
                deg = graph.degree(node_id)
                for pid, ptext in passage_map.items():
                    if str(node_id) in ptext:
                        scores[pid] += overlap / (1.0 + deg)
        for pid, ptext in passage_map.items():
            pw = set(normalize_answer(ptext).split())
            overlap = len(q_words & pw)
            if overlap > 0:
                scores[pid] += overlap * 0.3

    elif method == "dual":
        # Low-level: entity matching + 1-hop expansion
        matched_entities = []
        for node_id in graph.nodes():
            nw = set(normalize_answer(str(node_id)).split())
            overlap = len(q_words & nw)
            if overlap > 0:
                matched_entities.append((node_id, overlap, graph.degree(node_id)))

        for node_id, overlap, deg in matched_entities[:20]:
            score = overlap / (1.0 + deg)
            for pid, ptext in passage_map.items():
                if str(node_id) in ptext:
                    scores[pid] += score * 2.0
            for neighbor in graph.neighbors(node_id):
                for pid, ptext in passage_map.items():
                    if str(neighbor) in ptext:
                        scores[pid] += score * 0.5

        # High-level: relation edges
        for u, v, data in graph.edges(data=True):
            desc = normalize_answer(str(data.get("description", "")))
            overlap = len(q_words & set(desc.split()))
            if overlap > 0:
                for pid, ptext in passage_map.items():
                    if str(u) in ptext or str(v) in ptext:
                        scores[pid] += overlap * 0.3

    elif method == "ppr":
        seeds = {}
        for node_id in graph.nodes():
            nw = set(normalize_answer(str(node_id)).split())
            overlap = len(q_words & nw)
            if overlap > 0:
                seeds[node_id] = overlap
        if seeds:
            total_weight = sum(seeds.values())
            personalization = {n: seeds.get(n, 0) / total_weight for n in graph.nodes()}
            try:
                ppr = nx.pagerank(graph, personalization=personalization, alpha=0.85, max_iter=100)
            except:
                ppr = {}
            for node_id, pr_score in ppr.items():
                for pid, ptext in passage_map.items():
                    if str(node_id) in ptext:
                        scores[pid] += pr_score * 10.0

    elif method == "community":
        # V3 NEW: Community-level retrieval
        # Step 1: Find query-matching entities
        matched_entities = []
        for node_id in graph.nodes():
            nw = set(normalize_answer(str(node_id)).split())
            overlap = len(q_words & nw)
            if overlap > 0:
                matched_entities.append((node_id, overlap, graph.degree(node_id)))

        # Step 2: Identify relevant communities
        node_to_comm = {}
        if communities:
            for cid, members in communities.items():
                for node in members:
                    node_to_comm[node] = cid

        # Step 3: Score communities by entity matches
        comm_scores = defaultdict(float)
        for node_id, overlap, deg in matched_entities[:20]:
            cid = node_to_comm.get(node_id)
            if cid is not None:
                comm_scores[cid] += overlap / (1.0 + deg)

        # Step 4: Retrieve from top communities (community-level expansion)
        top_comms = sorted(comm_scores.items(), key=lambda x: -x[1])[:3]
        for cid, cscore in top_comms:
            if cid in communities:
                for node_id in communities[cid]:
                    # Entity match score
                    nw = set(normalize_answer(str(node_id)).split())
                    overlap = len(q_words & nw)
                    if overlap > 0:
                        for pid, ptext in passage_map.items():
                            if str(node_id) in ptext:
                                scores[pid] += overlap * 2.0
                    # Community context: neighbors also get scored
                    for pid, ptext in passage_map.items():
                        if str(node_id) in ptext:
                            scores[pid] += cscore * 0.3

        # Also do basic entity matching as fallback
        for node_id, overlap, deg in matched_entities[:20]:
            score = overlap / (1.0 + deg)
            for pid, ptext in passage_map.items():
                if str(node_id) in ptext:
                    scores[pid] += score * 1.0

    elif method == "vector":
        # V3 NEW: Vector RAG baseline (no graph)
        embedder = get_embedder()
        q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        # This requires pre-computed passage embeddings — handled separately
        pass

    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


# ============================================================
# Vector RAG baseline (separate implementation)
# ============================================================
class VectorRAG:
    """Simple vector RAG baseline — no graph."""
    def __init__(self):
        self.embedder = get_embedder()
        self.passage_ids = []
        self.passage_embs = None

    def index(self, passages):
        self.passage_ids = [p["id"] for p in passages]
        texts = [p["text"][:500] for p in passages]
        self.passage_embs = self.embedder.encode(texts, batch_size=256, show_progress_bar=True,
                                                   convert_to_numpy=True, normalize_embeddings=True)

    def retrieve(self, question, top_k=5):
        q_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self.passage_embs @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [self.passage_ids[i] for i in top_idx]


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
def evaluate(questions, q_gold, graph, passage_map, retrieval_method, n_eval=100, communities=None):
    r2_list, r5_list, f1_list, em_list = [], [], [], []

    for i, q in enumerate(questions[:n_eval]):
        question = q["question"]
        gold = q_gold.get(q["id"], [])

        if retrieval_method == "vector":
            retrieved = graph.retrieve(question, top_k=5)
        else:
            retrieved = retrieve_graph(question, graph, passage_map, top_k=5,
                                       method=retrieval_method, communities=communities)

        r2_list.append(recall_at_k(retrieved, gold, 2))
        r5_list.append(recall_at_k(retrieved, gold, 5))

        if retrieved:
            context = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
            answer = generate_answer_llm(question, context)
        else:
            answer = ""

        gold_answer = q.get("answer", gold[0] if gold else "")
        f1_list.append(compute_f1(answer, gold_answer))
        em_list.append(compute_em(answer, gold_answer))

        if (i + 1) % 20 == 0:
            print(f"    Eval {i+1}/{n_eval}: R@5={np.mean(r5_list):.3f} F1={np.mean(f1_list):.3f}")

    return {
        "R@2": float(np.mean(r2_list)),
        "R@5": float(np.mean(r5_list)),
        "F1": float(np.mean(f1_list)),
        "EM": float(np.mean(em_list)),
        "n_eval": len(r5_list),
    }


# ============================================================
# Method configurations
# ============================================================
METHODS = {
    # Internal ablations (progressive)
    "B1_rebuild":  {"skip_dedup": True,  "skip_relink": True,  "skip_community": True,  "rebuild": True},
    "B2_naive":    {"skip_dedup": True,  "skip_relink": True,  "skip_community": True,  "rebuild": False},
    "B3_dedup":    {"skip_dedup": False, "skip_relink": True,  "skip_community": True,  "rebuild": False},
    "B4_relink":   {"skip_dedup": False, "skip_relink": False, "skip_community": True,  "rebuild": False},
    "IGV":         {"skip_dedup": False, "skip_relink": False, "skip_community": False, "rebuild": False},
}

# External baselines
EXTERNAL_BASELINES = {
    "VectorRAG":   {"type": "vector", "retrieval": "vector"},
    "LightRAG":    {"type": "graph",  "retrieval": "dual",   "config": {"skip_dedup": True,  "skip_relink": True,  "skip_community": True,  "rebuild": False}},
    "HippoRAG":    {"type": "graph",  "retrieval": "ppr",    "config": {"skip_dedup": False, "skip_relink": True,  "skip_community": True,  "rebuild": False}},
}

RETRIEVAL_MAP = {
    "B1_rebuild": "dual",
    "B2_naive": "dual",
    "B3_dedup": "dual",
    "B4_relink": "dual",
    "IGV": "community",  # V3: IGV uses community-level retrieval
    "LightRAG": "dual",
    "HippoRAG": "ppr",
}


# ============================================================
# Main experiment loop
# ============================================================
def run_dataset(dataset_name, config):
    print(f"\n{'='*60}")
    print(f"  {dataset_name} | {config['n_passages']} passages | {N_QUESTIONS} questions")
    print(f"{'='*60}")

    # Load data
    passages = json.load(open(os.path.join(DATASET_DIR, config["passages_file"])))
    passages = passages[:config["n_passages"]]
    questions = json.load(open(os.path.join(DATASET_DIR, config["questions_file"])))

    # Build gold mapping
    q_gold = {}
    for q in questions:
        gold_ids = q.get("supporting_facts", q.get("gold_passages", []))
        if not gold_ids and "answer" in q:
            # For datasets without gold passages, use all passages as potential gold
            pass
        q_gold[q["id"]] = gold_ids if isinstance(gold_ids, list) else [gold_ids]

    passage_map = {p["id"]: p["text"] for p in passages}

    # V3 KEY FIX: Extract entities ONCE, cache, share across all methods
    cached_entities = extract_and_cache(passages, dataset_name)

    # Split: 70% base + 3x10% incremental
    n_base = int(len(passages) * 0.7)
    batch_size = (len(passages) - n_base) // 3
    base_ids = [p["id"] for p in passages[:n_base]]
    batches = [
        [p["id"] for p in passages[n_base + i*batch_size : n_base + (i+1)*batch_size]]
        for i in range(3)
    ]

    all_doc_ids = [p["id"] for p in passages]
    results = []

    # ===== Run internal methods =====
    for method_name, cfg in METHODS.items():
        print(f"\n  {dataset_name} | {method_name}")
        retrieval_method = RETRIEVAL_MAP[method_name]

        idx = IGVGraph(**cfg)

        if cfg["rebuild"]:
            # B1: rebuild from scratch (all docs at once)
            stats = idx.insert_from_cache(all_doc_ids, cached_entities)
            m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS,
                        communities=idx.communities if not cfg["skip_community"] else None)
            m.update({"dataset": dataset_name, "method": method_name, "stage": 3,
                     "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": stats["time"]})
            results.append(m)
            print(f"  {method_name} FINAL: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {stats['time']:.1f}s")
        else:
            # Incremental: base + 3 batches
            stats_base = idx.insert_from_cache(base_ids, cached_entities)
            m0 = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS,
                         communities=idx.communities if not cfg["skip_community"] else None)
            m0.update({"dataset": dataset_name, "method": method_name, "stage": 0,
                      "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": stats_base["time"]})
            results.append(m0)
            print(f"  {method_name} base: ents={len(idx.graph.nodes)} R@5={m0['R@5']:.3f} F1={m0['F1']:.3f} {stats_base['time']:.1f}s")

            for bi, batch_ids in enumerate(batches):
                stats = idx.insert_from_cache(batch_ids, cached_entities)
                if bi == 2:  # Evaluate at final stage
                    m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS,
                                communities=idx.communities if not cfg["skip_community"] else None)
                    m.update({"dataset": dataset_name, "method": method_name, "stage": bi+1,
                             "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": stats["time"]})
                else:
                    m = {"R@2": 0, "R@5": 0, "F1": 0, "EM": 0, "n_eval": 0}
                    m.update({"dataset": dataset_name, "method": method_name, "stage": bi+1,
                             "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": stats["time"]})
                results.append(m)
                if bi == 2:
                    print(f"  {method_name} inc{bi+1}: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {stats['time']:.1f}s")
                else:
                    print(f"  {method_name} inc{bi+1}: ents={len(idx.graph.nodes)} {stats['time']:.1f}s")

        # Save intermediate results
        json.dump(results, open(os.path.join(RESULTS_DIR, "v3_results_partial.json"), "w"))

    # ===== Run external baselines =====
    for bl_name, bl_cfg in EXTERNAL_BASELINES.items():
        print(f"\n  {dataset_name} | {bl_name} (external baseline)")

        if bl_cfg["type"] == "vector":
            # Vector RAG
            vrag = VectorRAG()
            vrag.index(passages)
            m = evaluate(questions, q_gold, vrag, passage_map, "vector", N_QUESTIONS)
            m.update({"dataset": dataset_name, "method": bl_name, "stage": 3,
                     "ents": 0, "edges": 0, "time": 0})
            results.append(m)
            print(f"  {bl_name}: R@5={m['R@5']:.3f} F1={m['F1']:.3f}")

        elif bl_cfg["type"] == "graph":
            # Graph-based external baseline (uses cached entities, same as internal)
            idx = IGVGraph(**bl_cfg["config"])
            # Process all docs incrementally (same as B2 but with different retrieval)
            idx.insert_from_cache(base_ids, cached_entities)
            for batch_ids in batches:
                idx.insert_from_cache(batch_ids, cached_entities)

            retrieval = bl_cfg["retrieval"]
            m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval, N_QUESTIONS)
            m.update({"dataset": dataset_name, "method": bl_name, "stage": 3,
                     "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": 0})
            results.append(m)
            print(f"  {bl_name}: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f}")

        json.dump(results, open(os.path.join(RESULTS_DIR, "v3_results_partial.json"), "w"))

    return results


# ============================================================
# Tau sensitivity
# ============================================================
def run_tau_sensitivity():
    print(f"\n{'='*60}")
    print(f"  Tau Sensitivity Analysis")
    print(f"{'='*60}")

    passages = json.load(open(os.path.join(DATASET_DIR, "hotpotqa_passages.json")))[:1000]
    questions = json.load(open(os.path.join(DATASET_DIR, "hotpotqa_questions.json")))
    q_gold = {}
    for q in questions:
        gold_ids = q.get("supporting_facts", q.get("gold_passages", []))
        q_gold[q["id"]] = gold_ids if isinstance(gold_ids, list) else [gold_ids]
    passage_map = {p["id"]: p["text"] for p in passages}

    # Reuse cached entities from full HotpotQA run (first 1000 passages)
    full_cache_file = os.path.join(CACHE_DIR, "HotpotQA_entities.json")
    if os.path.exists(full_cache_file):
        full_cache = json.load(open(full_cache_file))
        cached_entities = {pid: full_cache[pid] for pid in [p["id"] for p in passages] if pid in full_cache}
    else:
        cached_entities = extract_and_cache(passages, "HotpotQA_tau1000")
    all_doc_ids = [p["id"] for p in passages]

    tau_results = []
    for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        print(f"\n  tau={tau}")
        idx = IGVGraph(dedup_threshold=tau, skip_community=False, skip_relink=False, skip_dedup=False)
        idx.insert_from_cache(all_doc_ids, cached_entities)
        m = evaluate(questions, q_gold, idx.graph, passage_map, "community", 50,
                    communities=idx.communities)
        m["tau"] = tau
        m["ents"] = len(idx.graph.nodes)
        m["n_communities"] = len(idx.communities)
        tau_results.append(m)
        print(f"  tau={tau}: F1={m['F1']:.3f} R@5={m['R@5']:.3f} ents={m['ents']} communities={m['n_communities']}")

    return tau_results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=== IGV V3: Fixed Ablation + External Baselines + Community Retrieval ===\n")

    # Initialize model once
    get_model()
    get_embedder()

    all_results = []

    # Run all datasets
    for ds_name, ds_config in DATASETS.items():
        ds_results = run_dataset(ds_name, ds_config)
        all_results.extend(ds_results)
        # Save after each dataset
        json.dump(all_results, open(os.path.join(RESULTS_DIR, "v3_complete_results.json"), "w"))

    # Tau sensitivity
    tau_results = run_tau_sensitivity()
    json.dump(tau_results, open(os.path.join(RESULTS_DIR, "v3_tau_sensitivity.json"), "w"))

    # Save final results
    json.dump(all_results, open(os.path.join(RESULTS_DIR, "v3_complete_results.json"), "w"))
    json.dump(tau_results, open(os.path.join(RESULTS_DIR, "v3_tau_sensitivity.json"), "w"))

    print(f"\n=== V3 COMPLETE: {len(all_results)} records + {len(tau_results)} tau results ===")
    print("STOP: V3_DONE")
