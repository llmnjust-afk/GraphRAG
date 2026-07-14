# Patch: B1 only rebuilds at stage 3 (not every stage)
# Also reduce n_passages for faster turnaround
# Also reduce n_questions to 100 for evaluation speed

with open("igv_v2_experiment.py", "r") as f:
    content = f.read()

# 1. Reduce passage counts for speed
content = content.replace('"n_passages": 5000', '"n_passages": 2000')
content = content.replace('"n_passages": 3000', '"n_passages": 1500')
content = content.replace('"n_passages": 1000', '"n_passages": 1000')

# 2. Reduce questions to 100
content = content.replace("N_QUESTIONS = 200", "N_QUESTIONS = 100")

# 3. B1 rebuild: only evaluate at stage 3, skip stages 0-2 evaluation
# Replace the B1 rebuild loop to only do stage 3
old_b1 = """    if is_rebuild:
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
            print(f"  [{seed}] {method} s{stage}: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")"""

new_b1 = """    if is_rebuild:
        # B1: only rebuild at stage 3 (final) for efficiency
        all_docs = [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in range(len(passages))]
        idx = IGVGraph(**cfg)
        t0 = time.time()
        idx.insert(all_docs, extractor_func=extract_entities_llm)
        elapsed = time.time() - t0
        m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS)
        m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": 3,
                 "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": elapsed})
        results.append(m)
        print(f"  [{seed}] {method} FINAL: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")"""

content = content.replace(old_b1, new_b1)

# 4. For incremental methods: only evaluate at stage 0 and stage 3 (skip 1,2)
old_inc_eval = """        for bi, batch_idx in enumerate(batches):
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
            print(f"  [{seed}] {method} inc{bi+1}: ents={len(idx.graph.nodes)} R@5={m['R@5']:.3f} F1={m['F1']:.3f} {elapsed:.1f}s")"""

new_inc_eval = """        for bi, batch_idx in enumerate(batches):
            t0 = time.time()
            stats = idx.insert(
                [{"content": passages[i]["text"], "doc_id": passages[i]["id"]} for i in batch_idx],
                extractor_func=extract_entities_llm
            )
            elapsed = time.time() - t0
            # Only evaluate at last batch (stage 3) to save time
            if bi == 2:
                m = evaluate(questions, q_gold, idx.graph, passage_map, retrieval_method, N_QUESTIONS)
            else:
                m = {"R@2": 0, "R@5": 0, "F1": 0, "EM": 0, "n_eval": 0}
            m.update({"dataset": dataset_name, "method": method, "seed": seed, "stage": bi+1,
                     "ents": len(idx.graph.nodes), "edges": len(idx.graph.edges), "time": elapsed,
                     "stats": stats})
            results.append(m)
            print(f"  [{seed}] {method} inc{bi+1}: ents={len(idx.graph.nodes)} {elapsed:.1f}s" + (f" R@5={m['R@5']:.3f} F1={m['F1']:.3f}" if bi==2 else ""))"""

content = content.replace(old_inc_eval, new_inc_eval)

# 5. Tau sensitivity: reduce to 500 passages
content = content.replace("passages = json.load(f)[:1000]", "passages = json.load(f)[:500]")
content = content.replace("m = evaluate(questions, q_gold, idx.graph, passage_map, \"dual\", 100)", "m = evaluate(questions, q_gold, idx.graph, passage_map, \"dual\", 50)")

with open("igv_v2_experiment.py", "w") as f:
    f.write(content)

# Calculate new estimate
print("✅ Optimized: B1 only at stage 3, eval only at stage 0+3, 100 questions, smaller passages")
print()

# New estimate
total_h = 0
configs = [("HotpotQA", 2000), ("2Wiki", 1500), ("MuSiQue", 1500), ("NarrativeQA", 1000), ("StreamingQA", 2000)]
for name, n in configs:
    n_base = int(n * 0.7)
    batch = (n - n_base) // 3
    # B1: only 1 rebuild of all n docs
    b1_docs = n
    # B2/B3/B4/IGV: base + 3 batches = n docs each
    other_docs = n * 4
    # Eval: 100 questions * 2 stages (0 and 3) * 5 methods = 1000 calls
    eval_calls = 100 * 2 * 5
    total_s = (b1_docs + other_docs) * 2 + eval_calls * 2
    h = total_s / 3600
    total_h += h
    print(f"  {name} ({n}): {h:.1f}h")

# Tau: 500 passages * 6 values = 3000 docs + 50 questions * 6 = 300 calls
tau_s = 3000 * 2 + 300 * 2
total_h += tau_s / 3600
print(f"  Tau: {tau_s/3600:.1f}h")
print(f"  Total: {total_h:.1f}h = {total_h/24:.1f} days")
