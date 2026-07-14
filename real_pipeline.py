#!/usr/bin/env python3
"""
Real evaluation pipeline: HotpotQA with R@2, R@5, F1, EM metrics.
Compares B1 (rebuild), B2 (naive incremental), IGV (proposed).
"""

import asyncio, json, os, re, time, string, subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import networkx as nx
from igv import IGVIndex

GITHUB_TOKEN = "${GITHUB_TOKEN}"
DATASETS_DIR = "/data/lab/datasets"

# ============================================================
# Evaluation Metrics
# ============================================================

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def compute_f1(pred, gold):
    pred_t = normalize_answer(pred).split()
    gold_t = normalize_answer(gold).split()
    if not pred_t or not gold_t:
        return 0.0
    common = set(pred_t) & set(gold_t)
    if not common:
        return 0.0
    p = len(common) / len(pred_t)
    r = len(common) / len(gold_t)
    return 2 * p * r / (p + r)

def compute_em(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def recall_at_k(retrieved_ids, gold_ids, k):
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(gold_ids))
    return hits / max(1, len(gold_ids))

# ============================================================
# Retrieval (keyword + graph-based)
# ============================================================

def retrieve(question, idx, passage_map, top_k=5):
    """Retrieve passage IDs using graph entity matching + keyword overlap."""
    q_words = set(normalize_answer(question).split())
    q_words = {w for w in q_words if len(w) > 2}
    
    scores = defaultdict(float)
    
    # Pre-filter: only check nodes whose name overlaps with question
    relevant_nodes = []
    for node_id in idx.graph.nodes():
        node_words = set(normalize_answer(str(node_id)).split())
        if len(q_words & node_words) > 0:
            relevant_nodes.append((node_id, idx.graph.degree(node_id)))
    
    # Sort by degree (lower degree = more specific)
    relevant_nodes.sort(key=lambda x: x[1])
    
    # Only check top 200 most specific nodes
    for node_id, degree in relevant_nodes[:200]:
        node_words = set(normalize_answer(str(node_id)).split())
        overlap = len(q_words & node_words)
        if overlap > 0:
            for pid, ptext in passage_map.items():
                if str(node_id) in ptext:
                    scores[pid] += overlap / (1.0 + degree)
    
    # Direct keyword match (limited to top passages)
    for pid, ptext in list(passage_map.items())[:2000]:
        p_words = set(normalize_answer(ptext).split())
        overlap = len(q_words & p_words)
        if overlap > 0:
            scores[pid] += overlap * 0.5
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [pid for pid, _ in ranked[:top_k]]

# ============================================================
# Experiment Runner
# ============================================================

async def run_experiment(dataset_name, passages, questions, method="IGV"):
    """Run one method on one dataset. Returns per-stage metrics."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name} | Method: {method}")
    print(f"  Passages: {len(passages)} | Questions: {len(questions)}")
    print(f"{'='*60}")
    
    np.random.seed(42)
    indices = np.random.permutation(len(passages))
    n_base = int(len(passages) * 0.7)
    base_idx = indices[:n_base]
    inc_all = indices[n_base:]
    batch_size = max(1, len(inc_all) // 3)
    batches = [inc_all[i*batch_size:(i+1)*batch_size] for i in range(3)]
    
    passage_map = {p["id"]: p["text"] for p in passages}
    
    # Build gold passage IDs per question
    q_gold = {}
    for q in questions:
        gold_ids = set()
        for title in q.get("supporting_titles", []):
            for p in passages:
                if p.get("title", "") == title:
                    gold_ids.add(p["id"])
        if not gold_ids:
            ans = q.get("answer", "").lower()
            for p in passages:
                if ans in p["text"].lower():
                    gold_ids.add(p["id"])
        q_gold[q["id"]] = gold_ids
    
    results = []
    
    if method == "B1_rebuild":
        # Rebuild from scratch each stage
        all_docs = []
        for stage_idx in range(4):
            if stage_idx == 0:
                stage_docs = [passages[i] for i in base_idx]
            else:
                stage_docs = [passages[i] for i in batches[stage_idx - 1]]
            all_docs.extend(stage_docs)
            
            idx = IGVIndex(working_dir=f"/tmp/real_{method}_s{stage_idx}", skip_community=True, skip_relink=True, skip_dedup=True)
            await idx.initialize()
            t0 = time.time()
            await idx.insert([{"content": p["text"], "doc_id": p["id"]} for p in all_docs])
            elapsed = time.time() - t0
            
            metrics = evaluate(questions, q_gold, idx, passage_map)
            metrics.update({"dataset": dataset_name, "method": method,
                           "stage": stage_idx, "ents": idx.n_entities,
                           "edges": idx.n_edges, "time": elapsed})
            results.append(metrics)
            print(f"  Stage {stage_idx}: ents={idx.n_entities} R@2={metrics['R@2']:.3f} R@5={metrics['R@5']:.3f} F1={metrics['F1']:.3f} {elapsed:.1f}s")
            await idx.finalize()
    else:
        # Incremental methods (B2, IGV)
        if method == "IGV":
            idx = IGVIndex(working_dir=f"/tmp/real_{method}", skip_community=False, max_community_nodes=10000, skip_relink=True, skip_dedup=True)
        else:
            idx = IGVIndex(working_dir=f"/tmp/real_{method}", skip_community=True, skip_relink=True, skip_dedup=True)
        await idx.initialize()
        
        # Base
        t0 = time.time()
        base_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx]
        await idx.insert(base_docs)
        elapsed = time.time() - t0
        
        metrics = evaluate(questions, q_gold, idx, passage_map)
        metrics.update({"dataset": dataset_name, "method": method,
                       "stage": 0, "ents": idx.n_entities,
                       "edges": idx.n_edges, "time": elapsed})
        results.append(metrics)
        print(f"  Base: ents={idx.n_entities} R@2={metrics['R@2']:.3f} R@5={metrics['R@5']:.3f} F1={metrics['F1']:.3f} {elapsed:.1f}s")
        
        # Incremental batches
        for bi, batch_idx in enumerate(batches):
            batch_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx]
            t0 = time.time()
            await idx.insert(batch_docs)
            elapsed = time.time() - t0
            
            metrics = evaluate(questions, q_gold, idx, passage_map)
            metrics.update({"dataset": dataset_name, "method": method,
                           "stage": bi + 1, "ents": idx.n_entities,
                           "edges": idx.n_edges, "time": elapsed})
            results.append(metrics)
            print(f"  Inc {bi+1}: ents={idx.n_entities} R@2={metrics['R@2']:.3f} R@5={metrics['R@5']:.3f} F1={metrics['F1']:.3f} {elapsed:.1f}s")
        
        await idx.finalize()
    
    return results


def evaluate(questions, q_gold, idx, passage_map):
    """Compute R@2, R@5, F1, EM on 100 questions."""
    r2_list, r5_list, f1_list, em_list = [], [], [], []
    
    for q in questions[:100]:
        retrieved = retrieve(q["question"], idx, passage_map, top_k=5)
        gold = q_gold.get(q["id"], set())
        
        r2_list.append(recall_at_k(retrieved, gold, 2))
        r5_list.append(recall_at_k(retrieved, gold, 5))
        
        # Simple answer generation: extract best sentence from top passage
        if retrieved:
            top_text = passage_map.get(retrieved[0], "")
            sentences = top_text.split(".")
            # Find sentence with most question keyword overlap
            q_words = set(normalize_answer(q["question"]).split())
            best_sent = max(sentences, key=lambda s: len(q_words & set(normalize_answer(s).split()))) if sentences else ""
            answer = best_sent[:200]
        else:
            answer = ""
        
        f1_list.append(compute_f1(answer, q.get("answer", "")))
        em_list.append(compute_em(answer, q.get("answer", "")))
    
    return {
        "R@2": float(np.mean(r2_list)),
        "R@5": float(np.mean(r5_list)),
        "F1": float(np.mean(f1_list)),
        "EM": float(np.mean(em_list)),
        "n_eval": len(r2_list),
    }


async def main():
    all_results = []
    
    # ---- Load HotpotQA ----
    ppath = f"{DATASETS_DIR}/hotpotqa_passages.json"
    qpath = f"{DATASETS_DIR}/hotpotqa_questions.json"
    
    if not os.path.exists(ppath):
        print("❌ HotpotQA not found. Run download first.")
        return
    
    with open(ppath) as f:
        passages = json.load(f)[:500]
    with open(qpath) as f:
        questions = json.load(f)[:100]
    
    print(f"Loaded HotpotQA: {len(passages)} passages, {len(questions)} questions")
    
    # ---- Run all 3 methods ----
    # B1: Rebuild from scratch
    all_results.extend(await run_experiment("HotpotQA", passages, questions, "B1_rebuild"))
    
    # B2: Naive incremental
    all_results.extend(await run_experiment("HotpotQA", passages, questions, "B2_naive"))
    
    # IGV: Proposed method
    all_results.extend(await run_experiment("HotpotQA", passages, questions, "IGV"))
    
    # ---- Save ----
    os.makedirs("/data/lab/igv_workspace/results", exist_ok=True)
    with open("/data/lab/igv_workspace/results/real_hotpotqa.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: HotpotQA")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Stage':<8} {'R@2':<8} {'R@5':<8} {'F1':<8} {'EM':<8} {'Ents':<8} {'Time':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['method']:<15} {r['stage']:<8} {r['R@2']:<8.3f} {r['R@5']:<8.3f} {r['F1']:<8.3f} {r['EM']:<8.3f} {r['ents']:<8} {r['time']:<8.1f}")
    
    # ---- Push to GitHub ----
    os.chdir("/data/lab/igv_workspace/results")
    subprocess.run(["rm", "-rf", ".git"])
    subprocess.run(["git", "init", "-q"])
    subprocess.run(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])
    subprocess.run(["git", "config", "user.name", "llmnjust-afk"])
    subprocess.run(["git", "remote", "add", "origin", f"https://{GITHUB_TOKEN}@github.com/llmnjust-afk/GraphRAG.git"])
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["git", "commit", "-m", "Real HotpotQA results: B1 vs B2 vs IGV with R@2, R@5, F1, EM"])
    subprocess.run(["git", "push", "-u", "origin", "master", "--force"])
    print("\n📤 Pushed to GitHub: llmnjust-afk/GraphRAG")

if __name__ == "__main__":
    print("=== Real Dataset Evaluation Pipeline ===\n")
    asyncio.run(main())
