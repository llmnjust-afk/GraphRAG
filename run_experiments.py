#!/usr/bin/env python3
"""IGV Experiment Runner — runs all baselines on all datasets, logs all metrics."""

import asyncio
import json
import os
import time
from pathlib import Path

from igv import IGVIndex
from igv.community import detect_communities


# ---------------------------------------------------------------------------
# Dummy document generator (replace with real dataset loading)
# ---------------------------------------------------------------------------

def generate_dummy_docs(n: int, prefix: str = "doc") -> list[dict[str, str]]:
    return [
        {"content": f"This is document {prefix}_{i}. It contains unique terms about graph machine learning entity extraction indexing systems.", "doc_id": f"{prefix}_{i}"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Baseline orchestrator
# ---------------------------------------------------------------------------

async def run_baseline(
    name: str,
    base_docs: list[dict[str, str]],
    inc_batches: list[list[dict[str, str]]],
    *,
    out_dir: str = "results",
):
    """Run a single baseline (Rebuild-from-scratch or incremental).

    Returns list of per-stage metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = []

    # Initialize
    idx = IGVIndex(working_dir=f"{out_dir}/{name}_idx")
    await idx.initialize()

    # Insert base
    t0 = time.monotonic()
    stats = await idx.insert(base_docs)
    total_time = time.monotonic() - t0
    results.append({
        "stage": "base",
        "baseline": name,
        "n_entities": idx.n_entities,
        "n_edges": idx.n_edges,
        "n_communities": idx.n_communities,
        "update_time_sec": total_time,
        "n_docs": len(base_docs),
        "stats": stats.to_dict() if hasattr(stats, "to_dict") else {},
    })

    # Insert incremental batches
    for batch_idx, batch in enumerate(inc_batches):
        t0 = time.monotonic()
        stats = await idx.insert(batch)
        total_time = time.monotonic() - t0
        results.append({
            "stage": f"inc_{batch_idx + 1}",
            "baseline": name,
            "n_entities": idx.n_entities,
            "n_edges": idx.n_edges,
            "n_communities": idx.n_communities,
            "update_time_sec": total_time,
            "n_docs": len(batch),
            "stats": stats.to_dict() if hasattr(stats, "to_dict") else {},
        })

    await idx.finalize()
    return results


async def run_rebuild_baseline(
    base_docs: list[dict[str, str]],
    inc_batches: list[list[dict[str, str]]],
    *,
    out_dir: str = "results",
):
    """Rebuild-from-scratch baseline — re-inserts ALL docs each time."""
    os.makedirs(out_dir, exist_ok=True)
    results = []

    all_docs = list(base_docs)
    for batch_idx, batch in enumerate(inc_batches):
        all_docs.extend(batch)
        idx = IGVIndex(working_dir=f"{out_dir}/rebuild_stage_{batch_idx}")
        await idx.initialize()
        t0 = time.monotonic()
        await idx.insert(all_docs)
        total_time = time.monotonic() - t0
        results.append({
            "stage": f"rebuild_stage_{batch_idx}",
            "baseline": "Rebuild",
            "n_entities": idx.n_entities,
            "n_edges": idx.n_edges,
            "n_communities": idx.n_communities,
            "update_time_sec": total_time,
            "n_docs": len(all_docs),
        })
        await idx.finalize()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=== IGV Experiment Runner ===\n")

    # Generate synthetic data (3 batches of incremental updates)
    np.random.seed(42)
    base = generate_dummy_docs(50, "base")
    inc1 = generate_dummy_docs(10, "inc1")
    inc2 = generate_dummy_docs(10, "inc2")
    inc3 = generate_dummy_docs(10, "inc3")

    print(f"Base documents: {len(base)}")
    print(f"Incremental batches: {len(inc1)}, {len(inc2)}, {len(inc3)}\n")

    # ---- Run baseline: Rebuild from scratch ----
    print("--- Running REBUILD baseline ---")
    rebuild_results = await run_rebuild_baseline(base, [inc1, inc2, inc3])

    # ---- Run baseline: Incremental (IGV) ----
    print("--- Running INCREMENTAL (IGV) ---")
    incremental_results = await run_baseline("IGV", base, [inc1, inc2, inc3])

    # ---- Compute UER ----
    print("\n=== Final Metrics ===")
    for r_rebuild, r_inc in zip(rebuild_results, incremental_results[1:]):
        print(
            f"Stage {r_rebuild['stage']}: "
            f"Rebuild={r_rebuild['update_time_sec']:.2f}s "
            f"IGV={r_inc['update_time_sec']:.2f}s "
            f"UER={r_rebuild['update_time_sec'] / max(0.001, r_inc['update_time_sec']):.1f}x"
        )

if __name__ == "__main__":
    asyncio.run(main())
