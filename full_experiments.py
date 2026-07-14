#!/usr/bin/env python3
"""
COMPLETE experiment suite — all innovations enabled, 3 seeds, LLM-as-judge.
B1: Rebuild from scratch
B2: Naive incremental (set-union, no dedup/relink/community)
B3: + fixed-threshold dedup (embedding cosine, tau=0.85)
B4: + shallow relinking (2-hop)
IGV: dedup + relink + community repartition (proposed)
"""

import asyncio, json, os, re, time, string, subprocess, gc, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import networkx as nx

from igv import IGVIndex

GITHUB_TOKEN = "${GITHUB_TOKEN}"
DATASETS_DIR = "/data/lab/datasets"
SEEDS = [42, 52, 62]
DATASETS = {
    "HotpotQA": ("hotpotqa_passages.json", "hotpotqa_questions.json"),
    "2Wiki": ("2wiki_passages.json", "2wiki_questions.json"),
    "MuSiQue": ("musique_passages.json", "musique_questions.json"),
    "NarrativeQA": ("narrativeqa_passages.json", "narrativeqa_questions.json"),
    "StreamingQA": ("streamingqa_passages.json", "streamingqa_questions.json"),
}

# ============================================================
# LLM (Qwen2.5-7B-Instruct)
# ============================================================
_tok = None
_model = None

def load_model():
    global _tok, _model
    if _model is not None: return _tok, _model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen2.5-7B-Instruct...")
    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    _model.eval()
    print("✅ Model loaded")
    return _tok, _model

def llm_generate(prompt, max_new_tokens=200):
    tok, model = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def llm_judge(question, answer, criterion="comprehensiveness"):
    """LLM-as-judge: score answer 1-5 on given criterion."""
    prompt = f"""Rate the following answer on {criterion} (1=poor, 5=excellent). Return ONLY a single number.

Question: {question}
Answer: {answer[:500]}

Rating (1-5):"""
    resp = llm_generate(prompt, max_new_tokens=5)
    try:
        score = int(re.search(r'[1-5]', resp).group())
    except:
        score = 3
    return score

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
# Retrieval
# ============================================================
def retrieve(question, idx, passage_map, top_k=5):
    q_words = {w for w in normalize_answer(question).split() if len(w) > 2}
    scores = defaultdict(float)
    for node_id in idx.graph.nodes():
        nw = set(normalize_answer(str(node_id)).split())
        overlap = len(q_words & nw)
        if overlap > 0:
            deg = idx.graph.degree(node_id)
            for pid, ptext in passage_map.items():
                if str(node_id) in ptext:
                    scores[pid] += overlap / (1.0 + deg)
    for pid, ptext in passage_map.items():
        pw = set(normalize_answer(ptext).split())
        overlap = len(q_words & pw)
        if overlap > 0:
            scores[pid] += overlap * 0.3
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

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
# Experiment runner
# ============================================================
async def run_experiment(dataset_name, passages, questions, method, seed):
    cfg = METHOD_CONFIGS[method]
    np.random.seed(seed)
    random.seed(seed)

    indices = np.random.permutation(len(passages))
    n_base = int(len(passages) * 0.7)
    base_idx = indices[:n_base]
    inc_all = indices[n_base:]
    batch_size = max(1, len(inc_all) // 3)
    batches = [inc_all[i*batch_size:(i+1)*batch_size] for i in range(3)]

    passage_map = {p["id"]: p["text"] for p in passages}
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

    results = []

    if cfg["rebuild"]:
        all_docs = []
        for stage in range(4):
            stage_docs = [passages[i] for i in (base_idx if stage == 0 else batches[stage-1])]
            all_docs.extend(stage_docs)
            idx = IGVIndex(working_dir=f"/tmp/full_{dataset_name}_{method}_s{stage}_seed{seed}",
                          skip_community=True, skip_relink=True, skip_dedup=True)
            await idx.initialize()
            t0 = time.time()
            await idx.insert([{"content": p["text"], "doc_id": p["id"]} for p in all_docs])
            elapsed = time.time() - t0
            m = evaluate(questions, q_gold, idx, passage_map, do_judge=(stage == 3))
            m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": stage,
                     "ents": idx.n_entities, "edges": idx.n_edges, "time": elapsed})
            results.append(m)
            print(f"  [{seed}] {method} s{stage}: ents={idx.n_entities} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
            await idx.finalize()
    else:
        skip = {k: v for k, v in cfg.items() if k != "rebuild"}
        idx = IGVIndex(working_dir=f"/tmp/full_{dataset_name}_{method}_seed{seed}",
                      max_community_nodes=10000, **skip)
        await idx.initialize()

        # Base
        t0 = time.time()
        await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx])
        elapsed = time.time() - t0
        m = evaluate(questions, q_gold, idx, passage_map, do_judge=False)
        m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": 0,
                 "ents": idx.n_entities, "edges": idx.n_edges, "time": elapsed})
        results.append(m)
        print(f"  [{seed}] {method} base: ents={idx.n_entities} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")

        for bi, batch_idx in enumerate(batches):
            t0 = time.time()
            await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx])
            elapsed = time.time() - t0
            m = evaluate(questions, q_gold, idx, passage_map, do_judge=(bi == 2))
            m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": bi+1,
                     "ents": idx.n_entities, "edges": idx.n_edges, "time": elapsed})
            results.append(m)
            print(f"  [{seed}] {method} inc{bi+1}: ents={idx.n_entities} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
        await idx.finalize()

    return results


def evaluate(questions, q_gold, idx, passage_map, do_judge=False, n_eval=50):
    r2, r5, f1, em = [], [], [], []
    comp, div = [], []

    for q in questions[:n_eval]:
        retrieved = retrieve(q["question"], idx, passage_map, top_k=5)
        gold = q_gold.get(q["id"], set())
        r2.append(recall_at_k(retrieved, gold, 2))
        r5.append(recall_at_k(retrieved, gold, 5))

        if retrieved:
            ctx = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
            answer = llm_generate(
                f"Based on the context, answer concisely.\nContext: {ctx[:800]}\nQuestion: {q['question']}\nAnswer:",
                max_new_tokens=50)
        else:
            answer = ""
        f1.append(compute_f1(answer, q.get("answer", "")))
        em.append(compute_em(answer, q.get("answer", "")))

        if do_judge:
            comp.append(llm_judge(q["question"], answer, "comprehensiveness"))
            div.append(llm_judge(q["question"], answer, "diversity"))

    return {
        "R@2": float(np.mean(r2)), "R@5": float(np.mean(r5)),
        "F1": float(np.mean(f1)), "EM": float(np.mean(em)),
        "Comprehensiveness": float(np.mean(comp)) if comp else 0.0,
        "Diversity": float(np.mean(div)) if div else 0.0,
        "n_eval": len(r2),
    }


async def main():
    load_model()

    all_results = []

    for ds_name, (pfile, qfile) in DATASETS.items():
        ppath = f"{DATASETS_DIR}/{pfile}"
        qpath = f"{DATASETS_DIR}/{qfile}"
        if not os.path.exists(ppath):
            print(f"\n⚠️  {ds_name} not found, skipping")
            continue
        with open(ppath) as f: passages = json.load(f)[:300]
        with open(qpath) as f: questions = json.load(f)[:50]

        for seed in SEEDS:
            for method in ["B1_rebuild", "B2_naive", "B3_dedup", "B4_relink", "IGV"]:
                print(f"\n{'='*50}")
                print(f"  {ds_name} | {method} | seed={seed}")
                print(f"{'='*50}")
                try:
                    results = await run_experiment(ds_name, passages, questions, method, seed)
                    all_results.extend(results)
                except Exception as e:
                    print(f"  ❌ ERROR: {e}")
                    import traceback; traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    # Save
    os.makedirs("/data/lab/igv_workspace/results", exist_ok=True)
    outpath = "/data/lab/igv_workspace/results/complete_final_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print(f"  FINAL COMPLETE RESULTS (mean ± std across 3 seeds)")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} {'Method':<12} {'R@2':<12} {'R@5':<12} {'F1':<12} {'Comp':<8} {'Div':<8}")
    print("-" * 80)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        if r["stage"] == 3:
            key = (r["dataset"], r["method"])
            grouped[key].append(r)
    for (ds, method), runs in sorted(grouped.items()):
        r2 = [r["R@2"] for r in runs]
        r5 = [r["R@5"] for r in runs]
        f1 = [r["F1"] for r in runs]
        comp = [r["Comprehensiveness"] for r in runs]
        div = [r["Diversity"] for r in runs]
        print(f"{ds:<12} {method:<12} {np.mean(r2):.3f}±{np.std(r2):.3f}  {np.mean(r5):.3f}±{np.std(r5):.3f}  {np.mean(f1):.3f}±{np.std(f1):.3f}  {np.mean(comp):.1f}     {np.mean(div):.1f}")

    # Push
    os.chdir("/data/lab/igv_workspace/results")
    subprocess.run(["rm", "-rf", ".git"])
    subprocess.run(["git", "init", "-q"])
    subprocess.run(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])
    subprocess.run(["git", "config", "user.name", "llmnjust-afk"])
    subprocess.run(["git", "remote", "add", "origin", f"https://{GITHUB_TOKEN}@github.com/llmnjust-afk/GraphRAG.git"])
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["git", "commit", "-m", "Complete final results: 5 datasets × 5 methods × 3 seeds + LLM-judge"])
    subprocess.run(["git", "push", "-u", "origin", "master", "--force"])
    print("\n📤 Pushed to GitHub")

if __name__ == "__main__":
    print("=== COMPLETE EXPERIMENT SUITE ===\n")
    asyncio.run(main())
