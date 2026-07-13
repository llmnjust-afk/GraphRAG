#!/usr/bin/env python3
"""
Supplementary experiments:
1. Incremental-specific metrics (UER, GQD, SF, FR) on real datasets
2. Long-term streaming ablation (12 batches) on real data
3. Paired t-test for statistical significance
"""

import asyncio, json, os, re, time, string, subprocess, gc, random
from collections import defaultdict
from scipy import stats as scipy_stats
import numpy as np
import torch

from igv import IGVIndex
from igv.community import detect_communities, incremental_repartition

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
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
    if _model is not None:
        return _tok, _model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading Qwen2.5-7B-Instruct...")
    _tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    _model.eval()
    print("Model loaded")
    return _tok, _model

def llm_generate(prompt, max_new_tokens=200):
    tok, model = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ============================================================
# Evaluation Metrics
# ============================================================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt:
        return 0.0
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    p, r = len(common) / len(pt), len(common) / len(gt)
    return 2 * p * r / (p + r)

def compute_em(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / max(1, len(gold))


# ============================================================
# Retrieval (keyword + graph-based)
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
# Experiment 1: Incremental-specific metrics (UER, GQD, SF, FR)
# ============================================================

def evaluate_r5(questions, idx, passage_map, n_eval=50):
    """Compute R@5 on n_eval questions."""
    r5_list = []
    for q in questions[:n_eval]:
        q_words = {w for w in normalize_answer(q["question"]).split() if len(w) > 2}
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
        # Gold passages: those containing the answer
        gold = set()
        ans = q.get("answer", "").lower()
        for pid, ptext in passage_map.items():
            if ans in ptext.lower():
                gold.add(pid)
        retrieved = [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:5]]
        r5_list.append(recall_at_k(retrieved, gold, 5))
    return float(np.mean(r5_list))


async def run_incremental_metrics(dataset_name, passages, questions, seed):
    """Compute UER, GQD, SF, FR on real data."""
    print(f"\n  [Inc Metrics] {dataset_name} seed={seed}")
    np.random.seed(seed)
    random.seed(seed)

    indices = np.random.permutation(len(passages))
    n_base = int(len(passages) * 0.7)
    base_idx = indices[:n_base]
    inc_all = indices[n_base:]
    batch_size = max(1, len(inc_all) // 3)
    batches = [inc_all[i * batch_size:(i + 1) * batch_size] for i in range(3)]

    passage_map = {p["id"]: p["text"] for p in passages}
    results = []

    for method in ["B1_rebuild", "B2_naive", "IGV"]:
        cfg = METHOD_CONFIGS[method].copy()
        is_rebuild = cfg.pop("rebuild", False)

        if is_rebuild:
            all_docs = []
            total_llm_calls_rebuild = 0
            for stage in range(4):
                stage_docs = [passages[i] for i in (base_idx if stage == 0 else batches[stage - 1])]
                all_docs.extend(stage_docs)
                total_llm_calls_rebuild += len(stage_docs)
                idx = IGVIndex(
                    working_dir=f"/tmp/inc_met_{dataset_name}_{method}_s{stage}_seed{seed}",
                    skip_community=True, skip_relink=True, skip_dedup=True
                )
                await idx.initialize()
                t0 = time.time()
                await idx.insert([{"content": p["text"], "doc_id": p["id"]} for p in all_docs])
                elapsed = time.time() - t0
                r5 = evaluate_r5(questions, idx, passage_map)
                results.append({
                    "dataset": dataset_name, "method": method, "seed": seed,
                    "stage": stage, "ents": idx.n_entities, "edges": idx.n_edges,
                    "update_time": elapsed, "llm_calls": total_llm_calls_rebuild,
                    "R@5": r5,
                })
                await idx.finalize()
        else:
            skip = {k: v for k, v in cfg.items()}
            idx = IGVIndex(
                working_dir=f"/tmp/inc_met_{dataset_name}_{method}_seed{seed}",
                max_community_nodes=10000, **skip
            )
            await idx.initialize()

            total_llm_calls = 0
            prev_ents = 0
            prev_r5 = 0.0

            # Base
            base_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx]
            total_llm_calls += len(base_docs)
            t0 = time.time()
            await idx.insert(base_docs)
            elapsed = time.time() - t0
            r5_base = evaluate_r5(questions, idx, passage_map)

            results.append({
                "dataset": dataset_name, "method": method, "seed": seed,
                "stage": 0, "ents": idx.n_entities, "edges": idx.n_edges,
                "update_time": elapsed, "llm_calls": total_llm_calls,
                "R@5": r5_base,
            })
            prev_ents = idx.n_entities
            prev_r5 = r5_base

            for bi, batch_idx in enumerate(batches):
                batch_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx]
                total_llm_calls += len(batch_docs)
                t0 = time.time()
                await idx.insert(batch_docs)
                elapsed = time.time() - t0
                r5 = evaluate_r5(questions, idx, passage_map)

                # UER = total_rebuild_calls / incremental_calls
                rebuild_calls = len(base_idx) + sum(len(b) for b in batches[:bi + 1])
                uer = rebuild_calls / max(1, total_llm_calls)

                # SF = new_ents / total_ents
                new_ents = idx.n_entities - prev_ents
                sf = new_ents / max(1, idx.n_entities)

                # FR = max(0, prev_r5 - r5) / prev_r5
                fr = max(0, prev_r5 - r5) / max(0.001, prev_r5) if prev_r5 > 0 else 0.0

                results.append({
                    "dataset": dataset_name, "method": method, "seed": seed,
                    "stage": bi + 1, "ents": idx.n_entities, "edges": idx.n_edges,
                    "update_time": elapsed, "llm_calls": total_llm_calls,
                    "R@5": r5, "UER": uer, "SF": sf, "FR": fr,
                    "new_ents": new_ents,
                })
                prev_ents = idx.n_entities
                prev_r5 = r5
            await idx.finalize()

    # Compute GQD post-hoc: (R@5_rebuild - R@5_method) / R@5_rebuild
    b1_r5 = {r["stage"]: r.get("R@5", 0) for r in results if r["method"] == "B1_rebuild"}
    for r in results:
        if r["method"] in ("B2_naive", "IGV") and r["stage"] in b1_r5:
            r5_rebuild = b1_r5[r["stage"]]
            r5_method = r.get("R@5", 0)
            r["GQD"] = (r5_rebuild - r5_method) / max(0.001, r5_rebuild) if r5_rebuild > 0 else 0.0

    return results


# ============================================================
# Experiment 2: Long-term streaming (12 batches)
# ============================================================

async def run_longterm_streaming(dataset_name, passages, questions, seed=42):
    """Run 12 incremental batches and track degradation."""
    print(f"\n  [Long-term] {dataset_name} seed={seed}")
    np.random.seed(seed)
    random.seed(seed)

    indices = np.random.permutation(len(passages))
    n_base = int(len(passages) * 0.5)  # 50% base, 50% split into 12 batches
    base_idx = indices[:n_base]
    inc_all = indices[n_base:]
    n_batches = min(12, len(inc_all))
    batch_size = max(1, len(inc_all) // n_batches)
    batches = [inc_all[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]

    passage_map = {p["id"]: p["text"] for p in passages}
    results = []

    for method in ["B2_naive", "IGV"]:
        cfg = METHOD_CONFIGS[method].copy()
        cfg.pop("rebuild", None)
        idx = IGVIndex(
            working_dir=f"/tmp/longterm_{dataset_name}_{method}_seed{seed}",
            max_community_nodes=10000, **cfg
        )
        await idx.initialize()

        # Base
        t0 = time.time()
        await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx])
        elapsed = time.time() - t0
        r5 = evaluate_r5(questions, idx, passage_map)
        results.append({
            "dataset": dataset_name, "method": method, "seed": seed,
            "stage": 0, "batch": 0, "ents": idx.n_entities,
            "edges": idx.n_edges, "update_time": elapsed, "R@5": r5,
        })
        print(f"    {method} base: ents={idx.n_entities} R@5={r5:.3f}")

        prev_r5 = r5
        for bi, batch_idx in enumerate(batches):
            if len(batch_idx) == 0:
                continue
            t0 = time.time()
            await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx])
            elapsed = time.time() - t0
            r5 = evaluate_r5(questions, idx, passage_map)

            # Degradation: GQD = (prev_r5 - r5) / prev_r5
            gqd = (prev_r5 - r5) / max(0.001, prev_r5) if prev_r5 > 0 else 0.0

            results.append({
                "dataset": dataset_name, "method": method, "seed": seed,
                "stage": bi + 1, "batch": bi + 1, "ents": idx.n_entities,
                "edges": idx.n_edges, "update_time": elapsed, "R@5": r5,
                "GQD": gqd,
            })
            print(f"    {method} batch {bi+1}: ents={idx.n_entities} R@5={r5:.3f} GQD={gqd:.4f}")
            prev_r5 = r5

        await idx.finalize()

    return results


# ============================================================
# Experiment 3: Paired t-test
# ============================================================

def run_paired_ttest():
    """Run paired t-test between methods on R@5 and F1."""
    print("\n  === Paired t-tests ===")

    with open("/data/lab/igv_workspace/results/complete_final_results.json") as f:
        complete = json.load(f)

    grouped = defaultdict(lambda: defaultdict(list))
    for r in complete:
        if r["stage"] == 3:
            grouped[r["dataset"]][r["method"]].append(r)

    ttest_results = []
    datasets = sorted(set(r["dataset"] for r in complete))
    methods_to_compare = [("B2_naive", "IGV"), ("B2_naive", "B3_dedup"), ("B1_rebuild", "IGV")]

    for ds in datasets:
        for method_a, method_b in methods_to_compare:
            if method_a not in grouped[ds] or method_b not in grouped[ds]:
                continue
            for metric in ["R@2", "R@5", "F1"]:
                vals_a = [r[metric] for r in grouped[ds][method_a]]
                vals_b = [r[metric] for r in grouped[ds][method_b]]
                if len(vals_a) >= 3 and len(vals_b) >= 3:
                    t_stat, p_value = scipy_stats.ttest_rel(vals_a, vals_b)
                    entry = {
                        "dataset": ds, "method_a": method_a, "method_b": method_b,
                        "metric": metric, "t_stat": float(t_stat),
                        "p_value": float(p_value),
                        "mean_a": float(np.mean(vals_a)),
                        "mean_b": float(np.mean(vals_b)),
                        "significant": bool(p_value < 0.05),
                    }
                    ttest_results.append(entry)
                    sig = "*" if p_value < 0.05 else ""
                    print(f"    {ds}: {method_a} vs {method_b} [{metric}]: t={t_stat:.3f}, p={p_value:.4f} {sig}")

    return ttest_results


# ============================================================
# Main
# ============================================================

async def main():
    load_model()

    # ---- Experiment 1: Incremental metrics ----
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Incremental-Specific Metrics")
    print("=" * 60)
    inc_metrics = []
    for ds_name, (pfile, qfile) in DATASETS.items():
        ppath = f"{DATASETS_DIR}/{pfile}"
        if not os.path.exists(ppath):
            continue
        with open(ppath) as f:
            passages = json.load(f)[:300]
        with open(f"{DATASETS_DIR}/{qfile}") as f:
            questions = json.load(f)[:50]
        for seed in SEEDS:
            results = await run_incremental_metrics(ds_name, passages, questions, seed)
            inc_metrics.extend(results)
            gc.collect()
            torch.cuda.empty_cache()

    with open("/data/lab/igv_workspace/results/incremental_metrics.json", "w") as f:
        json.dump(inc_metrics, f, indent=2)
    print(f"\n  Saved {len(inc_metrics)} incremental metric records")

    # ---- Experiment 2: Long-term streaming ----
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Long-term Streaming (12 batches)")
    print("=" * 60)
    longterm = []
    for ds_name, (pfile, qfile) in [
        ("HotpotQA", ("hotpotqa_passages.json", "hotpotqa_questions.json")),
        ("2Wiki", ("2wiki_passages.json", "2wiki_questions.json")),
        ("MuSiQue", ("musique_passages.json", "musique_questions.json")),
    ]:
        ppath = f"{DATASETS_DIR}/{pfile}"
        if not os.path.exists(ppath):
            continue
        with open(ppath) as f:
            passages = json.load(f)[:300]
        with open(f"{DATASETS_DIR}/{qfile}") as f:
            questions = json.load(f)[:50]
        results = await run_longterm_streaming(ds_name, passages, questions, seed=42)
        longterm.extend(results)
        gc.collect()
        torch.cuda.empty_cache()

    with open("/data/lab/igv_workspace/results/longterm_streaming.json", "w") as f:
        json.dump(longterm, f, indent=2)
    print(f"\n  Saved {len(longterm)} long-term streaming records")

    # ---- Experiment 3: Paired t-test ----
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Statistical Significance (Paired t-test)")
    print("=" * 60)
    ttest = run_paired_ttest()

    with open("/data/lab/igv_workspace/results/paired_ttest.json", "w") as f:
        json.dump(ttest, f, indent=2)
    print(f"\n  Saved {len(ttest)} t-test records")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  SUPPLEMENTARY EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"  Incremental metrics: {len(inc_metrics)} records")
    print(f"  Long-term streaming: {len(longterm)} records")
    print(f"  Paired t-tests: {len(ttest)} records")

    # ---- Push to GitHub ----
    if GITHUB_TOKEN:
        os.chdir("/data/lab/igv_workspace/results")
        subprocess.run(["rm", "-rf", ".git"])
        subprocess.run(["git", "init", "-q"])
        subprocess.run(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])
        subprocess.run(["git", "config", "user.name", "llmnjust-afk"])
        subprocess.run(["git", "remote", "add", "origin", f"https://{GITHUB_TOKEN}@github.com/llmnjust-afk/GraphRAG.git"])
        subprocess.run(["git", "add", "-A"])
        subprocess.run(["git", "commit", "-m", "Supplementary results: incremental metrics + long-term streaming + t-tests"])
        subprocess.run(["git", "push", "-u", "origin", "master", "--force"])
        print("\nPushed to GitHub: llmnjust-afk/GraphRAG")


if __name__ == "__main__":
    print("=== Supplementary Experiments ===\n")
    asyncio.run(main())
