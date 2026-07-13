#!/usr/bin/env python3
"""
Full LLM-enhanced pipeline:
- Qwen2.5-7B-Instruct for entity extraction + answer generation
- IGV incremental graph index
- 5 datasets: HotpotQA, 2Wiki, MuSiQue, NarrativeQA, StreamingQA
- Metrics: R@2, R@5, F1, EM, entity count, update time
"""

import asyncio, json, os, re, time, string, subprocess, gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM

from igv import IGVIndex

# ============================================================
# Config
# ============================================================

GITHUB_TOKEN = "${GITHUB_TOKEN}"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASETS_DIR = "/data/lab/datasets"
DATASETS = {
    "HotpotQA": ("hotpotqa_passages.json", "hotpotqa_questions.json"),
    "2Wiki": ("2wiki_passages.json", "2wiki_questions.json"),
    "MuSiQue": ("musique_passages.json", "musique_questions.json"),
    "NarrativeQA": ("narrativeqa_passages.json", "narrativeqa_questions.json"),
    "StreamingQA": ("streamingqa_passages.json", "streamingqa_questions.json"),
}

# ============================================================
# LLM Setup (loaded once)
# ============================================================

_tokenizer = None
_model = None

def load_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    print("Loading Qwen2.5-7B-Instruct...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    _model.eval()
    print(f"✅ Model loaded on GPU")
    return _tokenizer, _model

def llm_generate(prompt: str, max_new_tokens: int = 200) -> str:
    """Generate text with the LLM."""
    tok, model = load_model()
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
    return tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def llm_batch_generate(prompts: list[str], max_new_tokens: int = 200, batch_size: int = 8) -> list[str]:
    """Batch generate for efficiency."""
    tok, model = load_model()
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        messages_list = [[{"role": "user", "content": p}] for p in batch]
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True, pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            gen = tok.decode(outputs[j][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append(gen)
        if i % 32 == 0:
            print(f"    LLM batch {i}/{len(prompts)}")
    return results

# ============================================================
# LLM Entity Extraction
# ============================================================

def extract_entities_llm(text: str, doc_id: str) -> tuple[list, list]:
    """Use LLM to extract entities and relations from text."""
    prompt = f"""Extract the key entities (people, places, organizations, concepts) from the following text. Return ONLY a JSON list of entity names, nothing else.

Text: {text[:500]}

Entities (JSON list):"""

    response = llm_generate(prompt, max_new_tokens=150)
    
    # Parse JSON list from response
    try:
        # Find JSON array in response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            entities_list = json.loads(match.group())
        else:
            entities_list = [e.strip().strip('"') for e in response.split(",") if e.strip()]
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
            "description": f"Mentioned in {doc_id}",
        })
        triplets.append((doc_entity, ent_name, {"weight": 1.0, "description": "mentions"}))
    
    # Link co-occurring entities
    ent_names = [e["entity_name"] for e in entities if e["entity_type"] == "ENTITY"]
    for j in range(len(ent_names)):
        for k in range(j+1, len(ent_names)):
            triplets.append((ent_names[j], ent_names[k], {"weight": 0.5, "description": "co-occurs"}))
    
    return entities, triplets

# ============================================================
# LLM Answer Generation
# ============================================================

def generate_answer_llm(question: str, context: str) -> str:
    """Use LLM to generate an answer given context."""
    prompt = f"""Based on the following context, answer the question concisely.

Context: {context[:1000]}

Question: {question}

Answer:"""
    return llm_generate(prompt, max_new_tokens=50)

# ============================================================
# Evaluation Metrics
# ============================================================

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())

def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt: return 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return 2 * p * r / (p + r)

def compute_em(pred, gold):
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0

def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & set(gold)) / max(1, len(gold))

# ============================================================
# Retrieval (graph-based + keyword)
# ============================================================

def retrieve(question, idx, passage_map, top_k=5):
    q_words = {w for w in normalize_answer(question).split() if len(w) > 2}
    scores = defaultdict(float)
    
    # Graph entity matching
    for node_id in idx.graph.nodes():
        nw = set(normalize_answer(str(node_id)).split())
        overlap = len(q_words & nw)
        if overlap > 0:
            deg = idx.graph.degree(node_id)
            for pid, ptext in passage_map.items():
                if str(node_id) in ptext:
                    scores[pid] += overlap / (1.0 + deg)
    
    # Keyword matching
    for pid, ptext in passage_map.items():
        pw = set(normalize_answer(ptext).split())
        overlap = len(q_words & pw)
        if overlap > 0:
            scores[pid] += overlap * 0.3
    
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]

# ============================================================
# Experiment Runner
# ============================================================

async def run_experiment(dataset_name, passages, questions, method="IGV"):
    print(f"\n{'='*60}")
    print(f"  {dataset_name} | {method}")
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
    
    # Gold passage IDs
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
    is_incremental = method in ("B2_naive", "IGV")
    
    if method == "B1_rebuild":
        all_docs = []
        for stage in range(4):
            stage_docs = [passages[i] for i in (base_idx if stage == 0 else batches[stage-1])]
            all_docs.extend(stage_docs)
            idx = IGVIndex(working_dir=f"/tmp/llm_{dataset_name}_{method}_s{stage}",
                          skip_community=True, skip_relink=True, skip_dedup=True)
            await idx.initialize()
            t0 = time.time()
            await idx.insert([{"content": p["text"], "doc_id": p["id"]} for p in all_docs])
            elapsed = time.time() - t0
            m = evaluate(questions, q_gold, idx, passage_map, use_llm=(stage == 3))
            m.update({"dataset": dataset_name, "method": method, "stage": stage,
                     "ents": idx.n_entities, "edges": idx.n_edges, "time": elapsed})
            results.append(m)
            print(f"  Stage {stage}: ents={idx.n_entities} R@2={m['R@2']:.3f} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
            await idx.finalize()
    else:
        skip_flags = {"skip_community": method != "IGV", "skip_relink": True, "skip_dedup": True}
        idx = IGVIndex(working_dir=f"/tmp/llm_{dataset_name}_{method}", max_community_nodes=10000, **skip_flags)
        await idx.initialize()
        
        # Base
        t0 = time.time()
        await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in base_idx])
        elapsed = time.time() - t0
        m = evaluate(questions, q_gold, idx, passage_map, use_llm=False)
        m.update({"dataset": dataset_name, "method": method, "stage": 0,
                 "ents": idx.n_entities, "edges": idx.n_edges, "time": elapsed})
        results.append(m)
        print(f"  Base: ents={idx.n_entities} R@2={m['R@2']:.3f} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
        
        for bi, batch_idx in enumerate(batches):
            t0 = time.time()
            await idx.insert([{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx])
            elapsed = time.time() - t0
            m = evaluate(questions, q_gold, idx, passage_map, use_llm=(bi == 2))
            m.update({"dataset": dataset_name, "method": method, "stage": bi+1,
                     "ents": idx.n_entities, "edges": idx.n_edges, "time": elapsed})
            results.append(m)
            print(f"  Inc {bi+1}: ents={idx.n_entities} R@2={m['R@2']:.3f} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")
        
        await idx.finalize()
    
    return results


def evaluate(questions, q_gold, idx, passage_map, use_llm=False, n_eval=50):
    """Evaluate retrieval + generation on n_eval questions."""
    r2_list, r5_list, f1_list, em_list = [], [], [], []
    
    for q in questions[:n_eval]:
        retrieved = retrieve(q["question"], idx, passage_map, top_k=5)
        gold = q_gold.get(q["id"], set())
        
        r2_list.append(recall_at_k(retrieved, gold, 2))
        r5_list.append(recall_at_k(retrieved, gold, 5))
        
        if use_llm and retrieved:
            context = " ".join(passage_map.get(pid, "") for pid in retrieved[:3])
            answer = generate_answer_llm(q["question"], context)
        elif retrieved:
            top_text = passage_map.get(retrieved[0], "")
            answer = top_text.split(".")[0][:200]
        else:
            answer = ""
        
        f1_list.append(compute_f1(answer, q.get("answer", "")))
        em_list.append(compute_em(answer, q.get("answer", "")))
    
    return {
        "R@2": float(np.mean(r2_list)), "R@5": float(np.mean(r5_list)),
        "F1": float(np.mean(f1_list)), "EM": float(np.mean(em_list)),
        "n_eval": len(r2_list),
    }


# ============================================================
# Main
# ============================================================

async def main():
    load_model()  # Load LLM once
    
    all_results = []
    
    for ds_name, (pfile, qfile) in DATASETS.items():
        ppath = f"{DATASETS_DIR}/{pfile}"
        qpath = f"{DATASETS_DIR}/{qfile}"
        if not os.path.exists(ppath):
            print(f"\n⚠️  {ds_name} not found, skipping")
            continue
        
        with open(ppath) as f: passages = json.load(f)
        with open(qpath) as f: questions = json.load(f)
        
        # Limit to 300 passages for feasibility
        passages = passages[:300]
        questions = questions[:50]
        
        for method in ["B1_rebuild", "B2_naive", "IGV"]:
            results = await run_experiment(ds_name, passages, questions, method)
            all_results.extend(results)
            gc.collect()
            torch.cuda.empty_cache()
    
    # Save
    os.makedirs("/data/lab/igv_workspace/results", exist_ok=True)
    with open("/data/lab/igv_workspace/results/final_llm_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Method':<12} {'Stage':<6} {'R@2':<8} {'R@5':<8} {'F1':<8} {'EM':<8} {'Ents':<8}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['dataset']:<15} {r['method']:<12} {r['stage']:<6} {r['R@2']:<8.3f} {r['R@5']:<8.3f} {r['F1']:<8.3f} {r['EM']:<8.3f} {r['ents']:<8}")
    
    # Push
    os.chdir("/data/lab/igv_workspace/results")
    subprocess.run(["rm", "-rf", ".git"])
    subprocess.run(["git", "init", "-q"])
    subprocess.run(["git", "config", "user.email", "llmnjust-afk@users.noreply.github.com"])
    subprocess.run(["git", "config", "user.name", "llmnjust-afk"])
    subprocess.run(["git", "remote", "add", "origin", f"https://{GITHUB_TOKEN}@github.com/llmnjust-afk/GraphRAG.git"])
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["git", "commit", "-m", "Final LLM-enhanced results: 5 datasets × 3 methods"])
    subprocess.run(["git", "push", "-u", "origin", "master", "--force"])
    print("\n📤 Pushed to GitHub")

if __name__ == "__main__":
    print("=== LLM-Enhanced Full Pipeline ===\n")
    asyncio.run(main())
