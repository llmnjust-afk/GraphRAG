"""IGV Incremental Relinking — connect new subgraph to existing graph via multi-hop expansion.

Ported from the experimental plan: after dedup & merge, find candidate
edges between new and existing nodes via 2-hop neighbourhood expansion.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .dedup import deduplicate_entities as _dedup_core


def compute_candidate_pairs(
    graph: nx.Graph,
    new_node_ids: list[str],
    *,
    hops: int = 2,
    max_candidates: int = 5000,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Generate (new_node, existing_node, meta) candidates for re-linking.

    Semantics:
      For each new node n, BFS-expand `hops` steps into existing graph.
      Any existing node reachable within these hops is a candidate target.

    Returns
    -------
    candidates : list[tuple[str, str, dict]]
        (source, target, {"distance": int, "common_neighbor_count": int})
    """
    if not new_node_ids:
        return []

    candidates: list[tuple[str, str, dict]] = []
    seen_pairs: set[tuple[str, str]] = set()
    existing_nodes = set(graph.nodes()) - set(new_node_ids)

    for src in new_node_ids:
        if src not in graph:
            continue
        # BFS outwards with depth limit
        visited: dict[str, int] = {src: 0}
        frontier = [src]
        for dist in range(1, hops + 1):
            if not frontier:
                break
            next_frontier: list[str] = []
            for fnode in frontier:
                for tgt in graph.neighbors(fnode):
                    if tgt in visited:
                        continue
                    visited[tgt] = dist
                    if tgt in existing_nodes:
                        pair = (src, tgt)
                        if pair not in seen_pairs:
                            # Count common neighbours for tie-breaking
                            common = len(
                                set(graph.neighbors(src))
                                & set(graph.neighbors(tgt))
                            )
                            candidates.append(
                                (src, tgt, {"distance": dist, "common_neighbor_count": common})
                            )
                            seen_pairs.add(pair)
                    next_frontier.append(tgt)
            frontier = next_frontier
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    # Sort: shorter distance first, higher common neighbours first
    candidates.sort(
        key=lambda x: (x[2]["distance"], -x[2]["common_neighbor_count"])
    )
    return candidates[:max_candidates]


def relink_candidates(
    graph: nx.Graph,
    new_node_ids: list[str],
    *,
    hops: int = 2,
    max_edges: int = 500,
    threshold: float = 0.85,
) -> int:
    """Compute candidate pairs and add high-confidence edges.

    Returns
    -------
    n_added : int
        Number of new edges added.
    """
    candidates = compute_candidate_pairs(graph, new_node_ids, hops=hops)
    n_added = 0
    for src, tgt, meta in candidates:
        if n_added >= max_edges:
            break
        # Simhash-based confidence (simplification — production would use
        # entity embedding similarity)
        confidence = 1.0 / (1.0 + meta["distance"])
        if confidence >= threshold:
            graph.add_edge(
                src, tgt, weight=confidence, relink="incremental", **meta
            )
            n_added += 1
    return n_added
