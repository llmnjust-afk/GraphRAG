"""Integration tests for IGV — validates core, dedup, community, relinking, and index.

Usage:  python -m pytest test_igv.py -v
Or just: python test_igv.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# ---- let's just test the pure functions directly (no async framework needed) ----

# ---------------------------------------------------------------------------
# Test: dedup.py
# ---------------------------------------------------------------------------

def test_dedup_entities_exact_match():
    from igv.dedup import deduplicate_entities

    existing = {
        "Alice": {"entity_name": "Alice", "description": "Alice is a researcher at MSR."},
        "Bob": {"entity_name": "Bob", "description": "Bob is a professor at Stanford."}
    }
    new = {
        "Alice": {"entity_name": "Alice", "description": "Alice works at MSR."},
        "Charlie": {"entity_name": "Charlie", "description": "Charlie is a PhD student."}
    }

    mapping = deduplicate_entities(existing, new)
    # Alice should map to existing Alice (exact match)
    assert mapping["Alice"] == "Alice", f"Expected Alice→Alice, got {mapping}"
    # Charlie should stay as new (no conflict)
    assert mapping["Charlie"] == "Charlie", f"Expected Charlie→Charlie, got {mapping}"


def test_dedup_entities_empty_existing():
    from igv.dedup import deduplicate_entities

    new = {"X": {"description": "test"}}
    mapping = deduplicate_entities({}, new)
    assert mapping == {"X": "X"}


def test_dedup_entities_similar_descriptions():
    from igv.dedup import deduplicate_entities, simhash_hit, _simhash_fingerprint, _token_ngrams

    # Verify simhashes detect similarity
    t1 = "graph neural network retrieval augmented generation"
    t2 = "graph neural network retrieval augmented generation system"
    fp1 = _simhash_fingerprint(_token_ngrams(t1))
    fp2 = _simhash_fingerprint(_token_ngrams(t2))
    # These should collide (3-gram overlap is high)
    assert simhash_hit(fp1, fp2, max_bit_diff=4), f"Simhashes should collide: {bin(fp1)} vs {bin(fp2)}"


def test_merge_descriptions():
    from igv.dedup import merge_entity_descriptions
    assert merge_entity_descriptions("short", "a longer description here") == "a longer description here"
    assert merge_entity_descriptions("", "only new") == "only new"
    assert merge_entity_descriptions("old", "") == "old"


# ---------------------------------------------------------------------------
# Test: community.py
# ---------------------------------------------------------------------------

def test_detect_communities():
    import networkx as nx
    from igv.community import detect_communities

    # Build a known 2-clique graph
    g = nx.Graph()
    g.add_edges_from([
        (1, 2), (2, 3), (1, 3),  # triangle 1
        (4, 5), (5, 6), (4, 6),  # triangle 2
    ])
    g.add_edge(3, 4)  # bridge

    communities = detect_communities(g)
    assert len(communities) >= 1, "Should find at least 1 community"
    # All nodes should be assigned
    all_nodes = set()
    for members in communities.values():
        all_nodes |= members
    expected = set(g.nodes())
    assert all_nodes == expected, f"Missing nodes: {expected - all_nodes}"


def test_incremental_repartition():
    import networkx as nx
    from igv.community import detect_communities, incremental_repartition

    g = nx.Graph()
    g.add_edges_from([
        (1, 2), (2, 3), (1, 3),
        (4, 5), (5, 6), (4, 6),
    ])
    g.add_edge(3, 4)
    old_part = detect_communities(g)
    n_before = len(old_part)

    # Add a new node that should create a new community
    g.add_node(7)
    g.add_edge(7, 1)
    new_part = incremental_repartition(g, [7], old_part)
    assert len(new_part) >= 1, f"Communities empty: {len(new_part)}"


# ---------------------------------------------------------------------------
# Test: relinking.py
# ---------------------------------------------------------------------------

def test_relink_candidates():
    import networkx as nx
    from igv.relinking import relink_candidates, compute_candidate_pairs

    g = nx.Graph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    # new nodes
    g.add_node("new1")
    g.add_node("new2")

    # "new1" is 1 hop from C
    g.add_edge("new1", "C")
    # "new2" has no neighbours — should produce no candidates

    candidates = compute_candidate_pairs(g, ["new1", "new2"], hops=2)
    # new1 should have at least B as a candidate (B is 2 hops from new1)
    for src, tgt, meta in candidates:
        assert src in ("new1", "new2")
        assert tgt not in ("new1", "new2")


# ---------------------------------------------------------------------------
# Test: incremental_index.py (main IGV)
# ---------------------------------------------------------------------------

async def test_igv_full_pipeline():
    from igv import IGVIndex

    with tempfile.TemporaryDirectory() as tmp:
        idx = IGVIndex(working_dir=tmp)
        await idx.initialize()

        # Insert base
        base = [{"content": f"Document about machine learning and knowledge graph {i}.", "doc_id": f"base_{i}"} for i in range(20)]
        stats = await idx.insert(base)
        assert idx.n_entities > 0, "No entities after base insert"
        print(f"Base: {stats.to_dict()}")

        # Insert incremental batch
        inc = [{"content": f"New document about graph neural networks and retrieval augmentation {i}.", "doc_id": f"inc_{i}"} for i in range(5)]
        stats_inc = await idx.insert(inc)
        assert idx.n_entities >= stats.n_entities_before, "Entities decreased!"
        print(f"Incremental: {stats_inc.to_dict()}")

        # Query
        results = await idx.query("knowledge graph learning")
        assert len(results) >= 1, "No query results"

        await idx.finalize()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Running IGV Integration Tests ===\n")

    # add current dir to path
    sys.path.insert(0, str(Path(__file__).parent))

    # Run sync tests
    tests = [
        test_dedup_entities_exact_match,
        test_dedup_entities_empty_existing,
        test_dedup_entities_similar_descriptions,
        test_merge_descriptions,
        test_detect_communities,
        test_incremental_repartition,
        test_relink_candidates,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")

    # Run async test
    print(f"\n  Running test_igv_full_pipeline...")
    try:
        asyncio.run(test_igv_full_pipeline())
        print(f"  ✓ test_igv_full_pipeline")
        passed += 1
    except Exception as e:
        print(f"  ✗ test_igv_full_pipeline: {e}")

    print(f"\n{'PASSED' if passed == len(tests) + 1 else 'SOME FAILED'} ({passed}/{len(tests) + 1})")
