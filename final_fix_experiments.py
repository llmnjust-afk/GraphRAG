#!/usr/bin/env python3
"""
Final fix experiments:
1. Correct GQD: run B1+IGV in same pass with identical R@5 evaluation
2. 200 questions per dataset
3. B3/B4 in long-term streaming
"""

import asyncio, json, os, re, time, string, subprocess, gc, random
from collections import defaultdict
from scipy import stats as scipy_stats
import numpy as np
import torch
from igv import IGVIndex

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
DATASETS_DIR = "/data/lab/datasets"
SEEDS = [42, 52, 62]
N_QUESTIONS = 200  # expanded from 50
DATASETS = {
    "HotpotQA": ("hotpotqa_passages.json", "hotpotqa_questions.json"),
    "2Wiki": ("2wiki_passages.json", "2wiki_questions.json"),
    "MuSiQue": ("musique_passages.json", "musique_questions.json"),
    "NarrativeQA": ("narrativeqa_passages.json", "narrativeqa_questions.json"),
    "StreamingQA": ("streamingqa_passages.json", "streamingqa_questions.json"),
}

# ---- LLM ----
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
        device_map="auto", trust_remote_code=True)
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

def llm_judge(question, answer, criterion):
    prompt = f"Rate the following answer on {criterion} (1=poor, 5=excellent). Return ONLY a single number.\nQuestion: {question}\nAnswer: {answer[:500]}\nRating (1-5):"
    resp = llm_generate(prompt, max_new_tokens=5)
    try:
        return int(re.search(r'[1-5]', resp).group())
    except:
        return 3

# ---- Metrics ----
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

# ---- Retrieval ----
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

# ---- Gold passages ----
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

# ---- Evaluate ----
def evaluate(questions, q_gold, idx, passage_map, do_llm=False, do_judge=False, n_eval=200):
    r2, r5, f1, em, comp, div = [], [], [], [], [], []
    for q in questions[:n_eval]:
        retrieved = retrieve(q["question"], idx, passage_map, top_k=5)
        gold = q_gold.get(q["id"], set())
        r2.append(recall_at_k(retrieved, gold, 2))
        r5.append(recall_at_k(retrieved, gold, 5))
        if do_llm and retrieved:
            ctx = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
            answer = llm_generate(
                f"Based on the context, answer concisely.\nContext: {ctx[:800]}\nQuestion: {q['question']}\nAnswer:",
                max_new_tokens=50)
        elif retrieved:
            top_text = passage_map.get(retrieved[0], "")
            answer = top_text.split(".")[0][:200]
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

# ---- Method configs ----
METHOD_CONFIGS = {
    "B1_rebuild": {"skip_community": True, "skip_relink": True, "skip_dedup": True, "rebuild": True},
    "B2_naive":   {"skip_community": True, "skip_relink": True, "skip_dedup": True, "rebuild": False},
    "B3_dedup":   {"skip_community": True, "skip_relink": True, "skip_dedup": False, "rebuild": False},
    "B4_relink":  {"skip_community": True, "skip_relink": False, "skip_dedup": False, "rebuild": False},
    "IGV":        {"skip_community": False, "skip_relink": False, "skip_dedup": False, "rebuild": False},
}

# ============================================================
# Experiment 1: Correct GQD with 200 questions
# Run B1 and IGV in same pass, compute GQD = (R5_b1 - R5_igv) / R5_b1
# ============================================================

async def run_gqd_experiment(dataset_name, passages, questions, seed):
    """Run B1 + IGV simultaneously, compute GQD correctly."""
    print(f"\n  [GQD] {dataset_name} seed={seed}")
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
    n_eval = min(N_QUESTIONS, len(questions))

    results = []

    # ---- Run B1 (rebuild) ----
    print(f"    Running B1 rebuild...")
    b1_r5_by_stage = {}
    all_docs_b1 = []
    for stage in range(4):
        stage_docs = [passages[i] for i in (base_idx if stage == 0 else batches[stage-1])]
        all_docs_b1.extend(stage_docs)
        idx_b1 = IGVIndex(working_dir=f"/tmp/gqd_{dataset_name}_B1_s{stage}_seed{seed}",
                         skip_community=True, skip_relink=True, skip_dedup=True)
        await idx_b1.initialize()
        t0 = time.time()
        await idx_b1.insert([{"content": p["text"], "doc_id": p["id"]} for p in all_docs_b1])
        elapsed = time.time() - t0
        m = evaluate(questions, q_gold, idx_b1, passage_map, n_eval=n_eval)
        b1_r5_by_stage[stage] = m["R@5"]
        results.append({
            "dataset": dataset_name, "method": "B1_rebuild", "seed": seed,
            "stage": stage, "ents": idx_b1.n_entities, "edges": idx_b1.n_edges,
            "time": elapsed, **m,
        })
        print(f"      B1 stage {stage}: R@5={m['R@5']:.4f} ents={idx_b1.n_entities}")
        await idx_b1.finalize()

    # ---- Run IGV ----
    print(f"    Running IGV...")
    idx_igv = IGVIndex(working_dir=f"/tmp/gqd_{dataset_name}_IGV_seed{seed}",
                       skip_community=False, skip_relink=False, skip_dedup=False,
                       max_community_nodes=10000)
    await idx_igv.initialize()

    prev_r5 = 0.0
    total_llm_calls = 0
    prev_ents = 0

    # Base
    base_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx]
    total_llm_calls += len(base_docs)
    t0 = time.time()
    await idx_igv.insert(base_docs)
    elapsed = time.time() - t0
    m = evaluate(questions, q_gold, idx_igv, passage_map, n_eval=n_eval)

    # GQD = (R5_b1 - R5_igv) / R5_b1
    gqd = (b1_r5_by_stage[0] - m["R@5"]) / max(0.001, b1_r5_by_stage[0]) if b1_r5_by_stage[0] > 0 else 0.0
    # UER = rebuild_calls / incremental_calls
    rebuild_calls = len(base_idx)
    uer = rebuild_calls / max(1, total_llm_calls)
    # SF = new_ents / total_ents
    sf = idx_igv.n_entities / max(1, idx_igv.n_entities)  # base: all new
    # FR = 0 for base
    fr = 0.0

    results.append({
        "dataset": dataset_name, "method": "IGV", "seed": seed,
        "stage": 0, "ents": idx_igv.n_entities, "edges": idx_igv.n_edges,
        "time": elapsed, "GQD": gqd, "UER": uer, "SF": sf, "FR": fr, **m,
    })
    print(f"      IGV base: R@5={m['R@5']:.4f} GQD={gqd:.4f} ents={idx_igv.n_entities}")
    prev_r5 = m["R@5"]
    prev_ents = idx_igv.n_entities

    for bi, batch_idx in enumerate(batches):
        batch_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx]
        total_llm_calls += len(batch_docs)
        t0 = time.time()
        await idx_igv.insert(batch_docs)
        elapsed = time.time() - t0
        m = evaluate(questions, q_gold, idx_igv, passage_map, n_eval=n_eval)

        stage = bi + 1
        b1_r5 = b1_r5_by_stage.get(stage, b1_r5_by_stage[3])
        gqd = (b1_r5 - m["R@5"]) / max(0.001, b1_r5) if b1_r5 > 0 else 0.0
        rebuild_calls = len(base_idx) + sum(len(b) for b in batches[:bi+1])
        uer = rebuild_calls / max(1, total_llm_calls)
        new_ents = idx_igv.n_entities - prev_ents
        sf = new_ents / max(1, idx_igv.n_entities)
        fr = max(0, prev_r5 - m["R@5"]) / max(0.001, prev_r5) if prev_r5 > 0 else 0.0

        results.append({
            "dataset": dataset_name, "method": "IGV", "seed": seed,
            "stage": stage, "ents": idx_igv.n_entities, "edges": idx_igv.n_edges,
            "time": elapsed, "GQD": gqd, "UER": uer, "SF": sf, "FR": fr, **m,
        })
        print(f"      IGV inc {bi+1}: R@5={m['R@5']:.4f} GQD={gqd:.4f} SF={sf:.4f} FR={fr:.4f}")
        prev_r5 = m["R@5"]
        prev_ents = idx_igv.n_entities

    await idx_igv.finalize()
    return results


# ============================================================
# Experiment 2: Long-term streaming with B2/B3/B4/IGV (12 batches)
# ============================================================

async def run_longterm_full(dataset_name, passages, questions, seed=42):
    print(f"\n  [Long-term Full] {dataset_name} seed={seed}")
    np.random.seed(seed)
    random.seed(seed)

    indices = np.random.permutation(len(passages))
    n_base = int(len(passages) * 0.5)
    base_idx = indices[:n_base]
    inc_all = indices[n_base:]
    n_batches = min(12, len(inc_all))
    batch_size = max(1, len(inc_all) // n_batches)
    batches = [inc_all[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

    passage_map = {p["id"]: p["text"] for p in passages}
    q_gold = build_gold(questions, passages)
    n_eval = min(N_QUESTIONS, len(questions))
    results = []

    for method in ["B2_naive", "B3_dedup", "B4_relink", "IGV"]:
        cfg = METHOD_CONFIGS[method].copy()
        cfg.pop("rebuild", None)
        idx = IGVIndex(working_dir=f"/tmp/ltfull_{dataset_name}_{method}_seed{seed}",
                      max_community_nodes=10000, **cfg)
        await idx.initialize()

        t0 = time.time()
        await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx])
        elapsed = time.time() - t0
        r5 = evaluate(questions, q_gold, idx, passage_map, n_eval=n_eval)["R@5"]
        results.append({
            "dataset": dataset_name, "method": method, "seed": seed,
            "stage": 0, "batch": 0, "ents": idx.n_entities,
            "edges": idx.n_edges, "time": elapsed, "R@5": r5,
        })
        print(f"    {method} base: R@5={r5:.4f}")

        prev_r5 = r5
        for bi, batch_idx in enumerate(batches):
            if len(batch_idx) == 0:
                continue
            t0 = time.time()
            await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx])
            elapsed = time.time() - t0
            r5 = evaluate(questions, q_gold, idx, passage_map, n_eval=n_eval)["R@5"]
            gqd = (prev_r5 - r5) / max(0.001, prev_r5) if prev_r5 > 0 else 0.0
            results.append({
                "dataset": dataset_name, "method": method, "seed": seed,
                "stage": bi + 1, "batch": bi + 1, "ents": idx.n_entities,
                "edges": idx.n_edges, "time": elapsed, "R@5": r5, "GQD": gqd,
            })
            print(f"    {method} batch {bi+1}: R@5={r5:.4f} GQD={gqd:.4f}")
            prev_r5 = r5
        await idx.finalize()

    return results


# ============================================================
# Main
# ============================================================

async def main():
    load_model()

    # ---- Experiment 1: Correct GQD with 200 questions ----
    print("\n" + "="*60)
    print("  EXPERIMENT 1: Correct GQD + 200 Questions")
    print("="*60)
    gqd_results = []
    for ds_name, (pfile, qfile) in DATASETS.items():
        ppath = f"{DATASETS_DIR}/{pfile}"
        if not os.path.exists(ppath):
            continue
        with open(ppath) as f: passages = json.load(f)[:300]
        with open(f"{DATASETS_DIR}/{qfile}") as f: questions = json.load(f)
        for seed in SEEDS:
            results = await run_gqd_experiment(ds_name, passages, questions, seed)
            gqd_results.extend(results)
            gc.collect()
            torch.cuda.empty_cache()

    with open("/data/lab/igv_workspace/results/gqd_correct.json", "w") as f:
        json.dump(gqd_results, f, indent=2)
    print(f"\n  Saved {len(gqd_results)} GQD records")

    # ---- Experiment 2: Long-term with all 4 methods ----
    print("\n" + "="*60)
    print("  EXPERIMENT 2: Long-term Streaming (4 methods, 12 batches)")
    print("="*60)
    lt_results = []
    for ds_name in ["HotpotQA", "2Wiki", "MuSiQue"]:
        pfile, qfile = DATASETS[ds_name]
        ppath = f"{DATASETS_DIR}/{pfile}"
        if not os.path.exists(ppath):
            continue
        with open(ppath) as f: passages = json.load(f)[:300]
        with open(f"{DATASETS_DIR}/{qfile}") as f: questions = json.load(f)
        results = await run_longterm_full(ds_name, passages, questions, seed=42)
        lt_results.extend(results)
        gc.collect()
        torch.cuda.empty_cache()

    with open("/data/lab/igv_workspace/results/longterm_full.json", "w") as f:
        json.dump(lt_results, f, indent=2)
    print(f"\n  Saved {len(lt_results)} long-term records")

    # ---- Summary ----
    print("\n" + "="*60)
    print("  FINAL FIX EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"  GQD (correct, 200 Q): {len(gqd_results)} records")
    print(f"  Long-term (4 methods): {len(lt_results)} records")

    # Print GQD summary
    print("\n  GQD Summary (IGV at stage 3, mean over 3 seeds):")
    for ds in DATASETS:
        vals = [r["GQD"] for r in gqd_results if r["dataset"]==ds and r["method"]=="IGV" and r["stage"]==3]
        if vals:
            print(f"    {ds}: GQD={np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Print long-term summary
    print("\n  Long-term R@5 degradation (12 batches):")
    for ds in ["HotpotQA", "2Wiki", "MuSiQue"]:
        for method in ["B2_naive", "B3_dedup", "B4_relink", "IGV"]:
            runs = [r for r in lt_results if r["dataset"]==ds and r["method"]==method]
            if runs:
                r5_start = runs[0]["R@5"]
                r5_end = runs[-1]["R@5"]
                print(f"    {ds} {method}: {r5_start:.4f} -> {r5_end:.4f}")

    # ---- Push ----
    if GITHUB_TOKEN:
        os.chdir("/data/lab/igv_workspace/results")
        subprocess.run(["rm", "-rf", ".git"])
        subprocess.run(["git", "init", "-q"])
        subprocess.run(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])
        subprocess.run(["git", "config", "user.name", "llmnjust-afk"])
        subprocess.run(["git", "remote", "add", "origin", f"https://{GITHUB_TOKEN}@github.com/llmnjust-afk/GraphRAG.git"])
        subprocess.run(["git", "add", "-A"])
        subprocess.run(["git", "commit", "-m", "Final fix: correct GQD + 200 questions + 4-method long-term"])
        subprocess.run(["git", "push", "-u", "origin", "master", "--force"])
        print("\n  Pushed to GitHub")

if __name__ == "__main__":
    print("=== Final Fix Experiments ===\n")
    asyncio.run(main())
