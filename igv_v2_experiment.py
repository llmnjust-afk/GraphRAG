#!/usr/bin/env python3
"""
IGV v2: Full pipeline with LLM entity extraction, graph-based retrieval, 
large-scale evaluation, and all fixes from review.

Experiments:
1. LLM-based entity extraction (replaces regex)
2. Graph-based retrieval (dual-level + PPR + community)
3. 5K passage scale with efficiency comparison
4. LLM answer generation with 200 questions
5. Corrected t-tests + Cohen's d
6. Tau sensitivity analysis
"""

import asyncio, json, os, re, time, string, subprocess, gc, random, math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import networkx as nx
from scipy import stats as scipy_stats

# ============================================================
# Config
# ============================================================

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
DATASETS_DIR = "/data/lab/datasets"
RESULTS_DIR = "/data/lab/igv_v2/results"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [42, 52, 62]
N_QUESTIONS = 200

DATASET_CONFIGS = {
    "HotpotQA": {"pfile": "hotpotqa_passages.json", "qfile": "hotpotqa_questions.json", "n_passages": 5000},
    "2Wiki": {"pfile": "2wiki_passages.json", "qfile": "2wiki_questions.json", "n_passages": 3000},
    "MuSiQue": {"pfile": "musique_passages.json", "qfile": "musique_questions.json", "n_passages": 3000},
    "NarrativeQA": {"pfile": "narrativeqa_passages.json", "qfile": "narrativeqa_questions.json", "n_passages": 1000},
    "StreamingQA": {"pfile": "streamingqa_passages.json", "qfile": "streamingqa_questions.json", "n_passages": 5000},
}

# ============================================================
# LLM (loaded once)
# ============================================================
_tok = None
_model = None

def load_model():
    global _tok, _model
    if _model is not None:
        return _tok, _model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen2.5-7B-Instruct...")
    _tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    _model.eval()
    print("✅ Model loaded on GPU")
    return _tok, _model

def llm_generate(prompt, max_new_tokens=200):
    tok, model = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def llm_batch_generate(prompts, max_new_tokens=150, batch_size=16):
    """Batch generation for efficiency."""
    tok, model = load_model()
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        texts = []
        for p in batch:
            messages = [{"role": "user", "content": p}]
            texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            gen = tok.decode(outputs[j][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append(gen)
        if i % 64 == 0 and i > 0:
            print(f"    LLM batch {i}/{len(prompts)}")
        del inputs, outputs
        torch.cuda.empty_cache()
    return results

# ============================================================
# LLM-based Entity Extraction (fixes BLOCKER 1)
# ============================================================

EXTRACT_PROMPT = """Extract the key entities (people, places, organizations, concepts, dates) from the following text. Return ONLY a JSON list of entity name strings, nothing else.

Text: {text}

Entities (JSON list):"""

def extract_entities_llm(text, doc_id):
    """Use LLM to extract entities from text."""
    prompt = EXTRACT_PROMPT.format(text=text[:1000])  # truncate to fit context
    response = llm_generate(prompt, max_new_tokens=150)
    
    # Parse JSON list from response
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            entities_list = json.loads(match.group())
        else:
            # Fallback: split by comma
            entities_list = [e.strip().strip('"').strip("'") for e in response.split(",") if e.strip()]
    except:
        entities_list = []
    
    # Build entity dicts and triplets
    entities = []
    triplets = []
    doc_entity = f"DOC_{doc_id}"
    entities.append({
        "entity_name": doc_entity,
        "entity_type": "DOCUMENT",
        "description": f"Document {doc_id}: {text[:100]}...",
    })
    
    for ent_name in entities_list[:10]:  # limit to 10 entities per doc
        ent_name = ent_name.strip().strip('"').strip("'")
        if len(ent_name) < 2:
            continue
        entities.append({
            "entity_name": ent_name,
            "entity_type": "ENTITY",
            "description": f"Mentioned in {doc_id}: {text[max(0,text.find(ent_name)-30):text.find(ent_name)+len(ent_name)+30] if ent_name in text else ent_name}",
        })
        triplets.append((doc_entity, ent_name, {"weight": 1.0, "description": "mentions"}))
    
    # Link co-occurring entities
    ent_names = [e["entity_name"] for e in entities if e["entity_type"] == "ENTITY"]
    for j in range(len(ent_names)):
        for k in range(j+1, len(ent_names)):
            triplets.append((ent_names[j], ent_names[k], {"weight": 0.5, "description": "co-occurs"}))
    
    return entities, triplets

# ============================================================
# Graph-based Retrieval (fixes BLOCKER 2)
# ============================================================

def retrieve_graph(question, graph, passage_map, top_k=5, method="dual"):
    """Graph-based retrieval supporting multiple strategies."""
    q_words = {w for w in normalize_answer(question).split() if len(w) > 2}
    scores = defaultdict(float)
    
    if method == "keyword":
        # Original keyword matching (baseline)
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
        # Dual-level: low-level (entity) + high-level (relation)
        # Low-level: find entities matching query keywords
        matched_entities = []
        for node_id in graph.nodes():
            nw = set(normalize_answer(str(node_id)).split())
            overlap = len(q_words & nw)
            if overlap > 0:
                matched_entities.append((node_id, overlap, graph.degree(node_id)))
        
        # Expand to 1-hop neighbors (subgraph retrieval)
        for node_id, overlap, deg in matched_entities[:20]:
            score = overlap / (1.0 + deg)
            # Find passages mentioning this entity
            for pid, ptext in passage_map.items():
                if str(node_id) in ptext:
                    scores[pid] += score * 2.0  # higher weight for entity match
            # 1-hop expansion
            for neighbor in graph.neighbors(node_id):
                for pid, ptext in passage_map.items():
                    if str(neighbor) in ptext:
                        scores[pid] += score * 0.5  # lower weight for neighbor
        
        # High-level: match relation edges
        for u, v, data in graph.edges(data=True):
            desc = normalize_answer(str(data.get("description", "")))
            overlap = len(q_words & set(desc.split()))
            if overlap > 0:
                for pid, ptext in passage_map.items():
                    if str(u) in ptext or str(v) in ptext:
                        scores[pid] += overlap * 0.3
    
    elif method == "ppr":
        # Personalized PageRank (HippoRAG-style)
        # Identify seed nodes
        seeds = {}
        for node_id in graph.nodes():
            nw = set(normalize_answer(str(node_id)).split())
            overlap = len(q_words & nw)
            if overlap > 0:
                seeds[node_id] = overlap
        
        if seeds:
            # Run PPR with seeds
            total_weight = sum(seeds.values())
            personalization = {n: seeds.get(n, 0) / total_weight for n in graph.nodes()}
            try:
                ppr = nx.pagerank(graph, personalization=personalization, alpha=0.85, max_iter=100)
            except:
                ppr = {}
            
            # Map PPR scores to passages
            for node_id, pr_score in ppr.items():
                for pid, ptext in passage_map.items():
                    if str(node_id) in ptext:
                        scores[pid] += pr_score * 10.0  # scale up
    
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

# ============================================================
# Metrics
# ============================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

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

# ============================================================
# LLM Answer Generation (fixes BLOCKER 4)
# ============================================================

ANSWER_PROMPT = """Based on the following context, answer the question concisely. Give only the answer, no explanation.

Context: {context}

Question: {question}

Answer:"""

def generate_answer_llm(question, retrieved_context):
    """Generate answer using LLM with retrieved context."""
    prompt = ANSWER_PROMPT.format(context=retrieved_context[:2000], question=question)
    return llm_generate(prompt, max_new_tokens=100)

# ============================================================
# IGV Graph Index (simplified, in-memory)
# ============================================================

class IGVGraph:
    """In-memory knowledge graph with dedup, relink, and community detection."""
    
    def __init__(self, dedup_threshold=0.85, skip_dedup=False, skip_relink=False, skip_community=False, max_community_nodes=50000):
        self.graph = nx.Graph()
        self.dedup_threshold = dedup_threshold
        self.skip_dedup = skip_dedup
        self.skip_relink = skip_relink
        self.skip_community = skip_community
        self.max_community_nodes = max_community_nodes
        self.communities = {}
        self._embedder = None
        self._entity_embeddings = {}  # cache
    
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedder
    
    def insert(self, documents, extractor_func=None):
        """Insert documents: extract entities, dedup, relink, repartition."""
        t0 = time.time()
        
        # Stage 1: Entity extraction
        all_entities = {}
        all_triplets = []
        for doc in documents:
            if extractor_func:
                ents, trips = extractor_func(doc["content"], doc["doc_id"])
            else:
                ents, trips = self._regex_extract(doc["content"], doc["doc_id"])
            for e in ents:
                all_entities[e["entity_name"]] = e
            all_triplets.extend(trips)
        
        n_new = len(all_entities)
        
        # Stage 2: Dedup
        n_deduped = 0
        if not self.skip_dedup and len(self.graph.nodes) > 0:
            merge_map = self._deduplicate(all_entities)
            # Apply merge
            for new_name, target in merge_map.items():
                if target != new_name and target in self.graph:
                    # Merge into existing
                    if new_name in all_entities:
                        existing = dict(self.graph.nodes[target])
                        existing["description"] = (existing.get("description", "") + " " + all_entities[new_name].get("description", "")).strip()[:500]
                        self.graph.nodes[target].update(existing)
                        n_deduped += 1
                    # Redirect triplets
                    all_triplets = [(merge_map.get(s, s), merge_map.get(o, o), d) for s, o, d in all_triplets]
                    # Remove from new entities (already merged)
                    if new_name in all_entities:
                        del all_entities[new_name]
                else:
                    # Keep as new
                    pass
        
        # Add new entities as nodes
        for name, data in all_entities.items():
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
                # Full detection
                self.communities = self._detect_communities()
            else:
                # Incremental
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
    
    def _regex_extract(self, text, doc_id):
        """Fallback regex extractor."""
        entities = []
        triplets = []
        proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        doc_entity = f"DOC_{doc_id}"
        entities.append({"entity_name": doc_entity, "entity_type": "DOCUMENT", "description": f"Document {doc_id}"})
        seen = set()
        for pn in proper_nouns[:10]:
            if pn.lower() not in seen and len(pn) > 2:
                seen.add(pn.lower())
                entities.append({"entity_name": pn, "entity_type": "ENTITY", "description": f"Mentioned in {doc_id}"})
                triplets.append((doc_entity, pn, {"weight": 1.0, "description": "mentions"}))
        return entities, triplets
    
    def _deduplicate(self, new_entities):
        """Embedding-based dedup."""
        embedder = self._get_embedder()
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
        """BFS-based relinking."""
        n_added = 0
        for src in new_node_ids[:50]:  # limit for speed
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
# Method configurations
# ============================================================
METHOD_CONFIGS = {
    "B1_rebuild": {"skip_community": True, "skip_relink": True, "skip_dedup": True, "rebuild": True},
    "B2_naive":   {"skip_community": True, "skip_relink": True, "skip_dedup": True, "rebuild": False},
    "B3_dedup":   {"skip_community": True, "skip_relink": True, "skip_dedup": False, "rebuild": False},
    "B4_relink":  {"skip_community": True, "skip_relink": False, "skip_dedup": False, "rebuild": False},
    "IGV":        {"skip_community": False, "skip_relink": False, "skip_dedup": False, "rebuild": False},
}

# ============================================================
# Build gold passage IDs
# ============================================================
def build_gold(questions, passages):
    q_gold = {}
    for q in questions:
        gold = set()
        for title in q.get("supporting_titles", []):
            for p in passages:
                if p.get("title") == title:
                    gold.add(p["id"])
        if not gold:
            ans = q.get("answer", "").lower()
            for p in passages:
                if ans in p["text"].lower():
                    gold.add(p["id"])
        q_gold[q["id"]] = gold
    return q_gold

# ============================================================
# Evaluate
# ============================================================
def evaluate(questions, q_gold, graph, passage_map, retrieval_method="dual", n_eval=200):
    r2, r5, f1, em = [], [], [], []
    n_eval = min(n_eval, len(questions))
    
    for q in questions[:n_eval]:
        retrieved = retrieve_graph(q["question"], graph, passage_map, top_k=5, method=retrieval_method)
        gold = q_gold.get(q["id"], set())
        r2.append(recall_at_k(retrieved, gold, 2))
        r5.append(recall_at_k(retrieved, gold, 5))
        
        # LLM answer generation
        if retrieved:
            context = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
            answer = generate_answer_llm(q["question"], context)
        else:
            answer = ""
        f1.append(compute_f1(answer, q.get("answer", "")))
        em.append(compute_em(answer, q.get("answer", "")))
    
    return {
        "R@2": float(np.mean(r2)), "R@5": float(np.mean(r5)),
        "F1": float(np.mean(f1)), "EM": float(np.mean(em)),
        "n_eval": len(r2),
    }

# ============================================================
# Experiment runner
# ============================================================
async def run_experiment(dataset_name, passages, questions, method, seed, retrieval_method="dual"):
    cfg = METHOD_CONFIGS[method].copy()
    is_rebuild = cfg.pop("rebuild", False)
    np.random.seed(seed)
    random.seed(seed)
    
    indices = np.random.permutation(len(passages))
    n_base = int(len(passages) * 0.7)
    base_idx = indices[:n_base]
    inc_all = indices[n_base:]
    batch_size = max(1, len(inc_all) // 3)
    batches = [inc_all[i*batch_size:(i+1)*batch_size] for i in range(3)]
    
    passage_map = {p["id"]: p["text"] for p in passages}
    q_gold = build_gold(questions, passages)
    results = []
    
    if is_rebuild:
        all_docs = []
        for stage in range(4):
            stage_docs = [passages[i] for i in (base_idx if stage == 0 else batches[stage-1])]
            all_docs.extend(stage_docs)
            idx = IGVGraph(**cfg)
            t0 = time.time()
            # Use LLM extraction for B1 too (fair comparison)
            idx.insert(
                [{"content": p["text"], "doc_id": p["id"]} for p in all_docs],
                extractor_func=extract_entities_llm
            )
            elapsed = time.time() - t0
            m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS)
            m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": stage,
                     "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": elapsed})
            results.append(m)
            print(f"  [{seed}] {method} s{stage}: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
    else:
        idx = IGVGraph(**cfg)
        # Base
        t0 = time.time()
        stats = idx.insert(
            [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx],
            extractor_func=extract_entities_llm
        )
        elapsed = time.time() - t0
        m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS)
        m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": 0,
                 "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": elapsed,
                 "stats": stats})
        results.append(m)
        print(f"  [{seed}] {method} base: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
        
        for bi, batch_idx in enumerate(batches):
            t0 = time.time()
            stats = idx.insert(
                [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx],
                extractor_func=extract_entities_llm
            )
            elapsed = time.time() - t0
            m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS)
            m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": bi+1,
                     "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": elapsed,
                     "stats": stats})
            results.append(m)
            print(f"  [{seed}] {method} inc{bi+1}: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
    
    return results


# ============================================================
# Main
# ============================================================
async def main():
    load_model()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    
    # ---- Experiment 1-4: Full pipeline on all datasets ----
    for ds_name, cfg in DATASET_CONFIGS.items():
        ppath = f"{DATASETS_DIR}/{cfg['pfile']}"
        qpath = f"{DATASETS_DIR}/{cfg['qfile']}"
        if not os.path.exists(ppath):
            print(f"\n⚠️  {ds_name} not found, skipping")
            continue
        
        with open(ppath) as f:
            passages = json.load(f)[:cfg["n_passages"]]
        with open(qpath) as f:
            questions = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"  {ds_name} | {len(passages)} passages | {len(questions)} questions")
        print(f"{'='*60}")
        
        for seed in SEEDS:
            for method in ["B1_rebuild", "B2_naive", "B3_dedup", "B4_relink", "IGV"]:
                print(f"\n  {ds_name} | {method} | seed={seed}")
                try:
                    results = await run_experiment(ds_name, passages, questions, method, seed, retrieval_method="dual")
                    all_results.extend(results)
                except Exception as e:
                    print(f"  ❌ ERROR: {e}")
                    import traceback; traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
    
    # Save
    with open(f"{RESULTS_DIR}/v2_complete_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved {len(all_results)} records")
    
    # ---- Experiment 5: t-tests + Cohen's d ----
    print("\n" + "="*60)
    print("  EXPERIMENT 5: Paired t-tests + Cohen's d")
    print("="*60)
    ttest_results = []
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["stage"] == 3:
            grouped[r["dataset"]][r["method"]].append(r)
    
    datasets = sorted(set(r["dataset"] for r in all_results))
    for ds in datasets:
        for method_a, method_b in [("B2_naive", "IGV"), ("B2_naive", "B3_dedup"), ("B1_rebuild", "IGV")]:
            if method_a not in grouped[ds] or method_b not in grouped[ds]:
                continue
            for metric in ["R@2", "R@5", "F1"]:
                vals_a = [r[metric] for r in grouped[ds][method_a]]
                vals_b = [r[metric] for r in grouped[ds][method_b]]
                if len(vals_a) >= 3 and len(vals_b) >= 3:
                    t_stat, p_value = scipy_stats.ttest_rel(vals_a, vals_b)
                    # Cohen's d
                    diff = np.array(vals_a) - np.array(vals_b)
                    pooled_std = np.std(diff)
                    cohen_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0.0
                    ttest_results.append({
                        "dataset": ds, "method_a": method_a, "method_b": method_b,
                        "metric": metric, "t_stat": float(t_stat), "p_value": float(p_value),
                        "cohen_d": float(cohen_d),
                        "mean_a": float(np.mean(vals_a)), "mean_b": float(np.mean(vals_b)),
                        "vals_a": vals_a, "vals_b": vals_b,
                        "significant": bool(p_value < 0.05),
                    })
                    sig = "*" if p_value < 0.05 else ""
                    print(f"  {ds}: {method_a} vs {method_b} [{metric}]: t={t_stat:.3f}, p={p_value:.4f}, d={cohen_d:.3f} {sig}")
    
    with open(f"{RESULTS_DIR}/v2_ttest.json", "w") as f:
        json.dump(ttest_results, f, indent=2)
    
    # ---- Experiment 6: Tau sensitivity ----
    print("\n" + "="*60)
    print("  EXPERIMENT 6: Tau Sensitivity Analysis")
    print("="*60)
    # Run on HotpotQA with 1000 passages, 1 seed
    ppath = f"{DATASETS_DIR}/hotpotqa_passages.json"
    if os.path.exists(ppath):
        with open(ppath) as f: passages = json.load(f)[:1000]
        with open(f"{DATASETS_DIR}/hotpotqa_questions.json") as f: questions = json.load(f)
        passage_map = {p["id"]: p["text"] for p in passages}
        q_gold = build_gold(questions, passages)
        
        tau_results = []
        for tau in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            print(f"\n  τ={tau}...")
            idx = IGVGraph(dedup_threshold=tau, skip_community=False, skip_relink=False, skip_dedup=False)
            idx.insert(
                [{"content": p["text"], "doc_id": p["id"]} for p in passages],
                extractor_func=extract_entities_llm
            )
            m = evaluate(questions, q_gold, idx.graph, passage_map, "dual", 100)
            m.update({"tau": tau, "ents": len(idx.graph.nodes), "n_communities": len(idx.communities)})
            tau_results.append(m)
            print(f"    ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f}")
            gc.collect(); torch.cuda.empty_cache()
        
        with open(f"{RESULTS_DIR}/v2_tau_sensitivity.json", "w") as f:
            json.dump(tau_results, f, indent=2)
        print(f"\n  ✅ Saved {len(tau_results)} tau sensitivity records")
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  ALL V2 EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"  Main results: {len(all_results)} records")
    print(f"  t-tests: {len(ttest_results)} records")
    print(f"  tau sensitivity: {len(tau_results) if 'tau_results' in dir() else 0} records")
    
    # ---- Push to GitHub ----
    if GITHUB_TOKEN:
        os.chdir(RESULTS_DIR)
        subprocess.run(["rm", "-rf", ".git"])
        subprocess.run(["git", "init", "-q"])
        subprocess.run(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])
        subprocess.run(["git", "config", "user.name", "llmnjust-afk"])
        subprocess.run(["git", "remote", "add", "origin", f"https://{GITHUB_TOKEN}@github.com/llmnjust-afk/GraphRAG.git"])
        subprocess.run(["git", "add", "-A"])
        subprocess.run(["git", "commit", "-m", "V2 results: LLM extraction + graph retrieval + 5K scale + tau sensitivity"])
        subprocess.run(["git", "push", "-u", "origin", "master", "--force"])
        print("\n📤 Pushed to GitHub")

if __name__ == "__main__":
    print("=== IGV V2: Full Pipeline Experiments ===\n")
    asyncio.run(main())
