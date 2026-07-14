"""IGV Incremental Index — main orchestrator for LightRAG + incremental updates.

Usage::

  from igv import IGVIndex
  idx = IGVIndex(working_dir="./my_graph")
  await idx.initialize()
  await idx.insert(new_docs)        # incremental
  results = await idx.query("question?")
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .community import detect_communities, incremental_repartition, compute_module_delta
from .dedup import deduplicate_entities, merge_entity_descriptions
from .relinking import relink_candidates


@dataclass
class IncrementalStats:
    """Per-update metrics for the paper's incremental indices."""

    n_docs_inserted: int = 0
    n_entities_before: int = 0
    n_edges_before: int = 0
    n_entities_after: int = 0
    n_edges_after: int = 0
    n_deduped: int = 0
    n_relinked: int = 0
    n_communities_before: int = 0
    n_communities_after: int = 0
    update_time_sec: float = 0.0
    # computed indices
    uer: float = 0.0  # update efficiency ratio
    node_migration_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_docs": self.n_docs_inserted,
            "entities_before": self.n_entities_before,
            "entities_after": self.n_entities_after,
            "edges_before": self.n_edges_before,
            "edges_after": self.n_edges_after,
            "n_deduped": self.n_deduped,
            "n_relinked": self.n_relinked,
            "communities_before": self.n_communities_before,
            "communities_after": self.n_communities_after,
            "update_time_sec": round(self.update_time_sec, 2),
            "UER": round(self.uer, 3),
            "node_migration_rate": round(self.node_migration_rate, 4),
        }


@dataclass
class IGVIndex:
    """Incremental Graph Versioner — drop-in wrapper on a networkx Graph.

    Integrates: LightRAG-style LLM extraction → dedup → relink → community repartition.
    """

    working_dir: str = "./igv_index"
    # Louvain params
    community_resolution: float = 1.0
    community_seed: int = 42
    # Relink params
    relink_hops: int = 2
    relink_max_edges: int = 500
    # Dedup
    dedup_threshold: float = 0.85
    skip_community: bool = False
    max_community_nodes: int = 5000
    skip_relink: bool = False
    skip_dedup: bool = False
    # Community repartition trigger
    node_migration_threshold: float = 0.05  # 5%

    # --- runtime state (not serialized) ---
    _graph: nx.Graph | None = field(init=False, default=None)
    _communities: dict[int, frozenset] = field(init=False, default_factory=dict)
    _doc_counter: int = field(init=False, default=0)

    # ---------- lifecycle ----------

    async def initialize(self) -> None:
        """Load or create the graph."""
        os.makedirs(self.working_dir, exist_ok=True)
        path = self._graph_path()
        if os.path.exists(path):
            self._graph = nx.read_graphml(path)
        else:
            self._graph = nx.Graph()
        # Ensure graph is always writable
        if self._graph is None:
            self._graph = nx.Graph()

    async def finalize(self) -> None:
        """Persist graph to disk."""
        if self._graph is not None:
            nx.write_graphml(self._graph, self._graph_path())

    # ---------- public API ----------

    async def insert(
        self,
        documents: list[dict[str, str]],
        *,
        rebuild: bool = False,
    ) -> IncrementalStats:
        """Insert documents into the index.

        `documents` format: [{"content": "text...", "doc_id": "doc_1"}, ...]

        Returns incremental statistics for logging and metric computation.
        """
        t0 = time.monotonic()
        stats = IncrementalStats()
        stats.n_docs_inserted = len(documents)
        stats.n_entities_before = len(self._graph.nodes) if self._graph else 0
        stats.n_edges_before = len(self._graph.edges) if self._graph else 0

        # 1. Simulate entity extraction (production: LightRAG's extract_entities)
        new_triplets, new_entities = await self._python_extract_entities(documents)

        # 2. Dedup + merge
        existing_entities = {
            n: dict(self._graph.nodes[n]) for n in self._graph.nodes
        }
        if self.skip_dedup:
            merge_map = {k: k for k in new_entities}
        else:
            existing_entities = {
                n: dict(self._graph.nodes[n]) for n in self._graph.nodes
            }
            merge_map = deduplicate_entities(
                existing_entities,
                new_entities,
            )

        # ----- INSERT NODES (merge or add) -----
        new_node_ids: list[str] = []
        for name, data in new_entities.items():
            target = merge_map.get(name, name)
            if target in existing_entities:
                # Merge descriptions (in production: LLM summary of both)
                existing_entities[target]["description"] = (
                    merge_entity_descriptions(
                        existing_entities[target].get("description", ""),
                        data.get("description", ""),
                    )
                )
                stats.n_deduped += 1
            else:
                self._graph.add_node(target, **data)
                existing_entities[target] = data
            new_node_ids.append(target)

        # ----- INSERT EDGES -----
        for src, tgt, edge_data in new_triplets:
            if src in existing_entities and tgt in existing_entities:
                if self._graph.has_edge(src, tgt):
                    # Increment weight
                    self._graph[src][tgt]["weight"] = (
                        self._graph[src][tgt].get("weight", 1.0) + edge_data.get("weight", 1.0)
                    )
                else:
                    self._graph.add_edge(src, tgt, **edge_data)

        # 3. Relink new nodes to existing graph
        n_added = 0
        if not self.skip_relink:
            n_added = relink_candidates(
                self._graph,
                new_node_ids,
                hops=self.relink_hops,
                max_edges=self.relink_max_edges,
            )
        stats.n_relinked = n_added

        # 4. Incremental community repartition
        old_communities = dict(self._communities)
        new_partition = {}
        stats.node_migration_rate = 0.0
        if len(self._graph.nodes) > 3 and not self.skip_community and len(self._graph.nodes) <= self.max_community_nodes:
            if not self._communities:
                # First insertion — full community detection
                new_partition = detect_communities(
                    self._graph,
                    seed=self.community_seed,
                    resolution=self.community_resolution,
                )
            else:
                # Incremental: only re-partition affected region
                new_partition = incremental_repartition(
                    self._graph,
                    new_node_ids,
                    self._communities,
                    seed=self.community_seed,
                    resolution=self.community_resolution,
                )
                delta = compute_module_delta(
                    self._graph, old_communities, new_partition
                )
                stats.node_migration_rate = delta["node_migration_rate"]

            stats.n_communities_before = len(self._communities)
            stats.n_communities_after = len(new_partition)
            self._communities = new_partition

        # ---- finalize ----
        stats.n_entities_after = len(self._graph.nodes)
        stats.n_edges_after = len(self._graph.edges)
        stats.update_time_sec = time.monotonic() - t0

        # UER (Update Efficiency Ratio) — simplified as entities touched per doc
        stats.uer = (
            abs(stats.n_entities_after - stats.n_entities_before)
            / max(1, stats.n_docs_inserted)
        )

        self._doc_counter += len(documents)
        return stats

    async def query(self, question: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """Minimal keyword-based retrieval (replace with LightRAG query in production)."""
        results = []
        for node_id, data in list(self._graph.nodes(data=True))[:top_k * 50]:
            desc = str(data.get("description", data.get("entity_name", node_id)))
            if any(word in desc.lower() for word in question.lower().split()):
                results.append({"node": node_id, "description": desc[:500]})
            if len(results) >= top_k:
                break
        return results[:top_k]

    @property
    def graph(self) -> nx.Graph:
        if self._graph is None:
            raise RuntimeError("Index not initialized. Call await index.initialize()")
        return self._graph

    @property
    def n_entities(self) -> int:
        return len(self._graph.nodes) if self._graph else 0

    @property
    def n_edges(self) -> int:
        return len(self._graph.edges) if self._graph else 0

    @property
    def n_communities(self) -> int:
        return len(self._communities)

    # ---------- helpers ----------

    def _graph_path(self) -> str:
        return os.path.join(self.working_dir, "graph.graphml")

    async def _python_extract_entities(
        self, documents: list[dict[str, str]]
    ) -> tuple[list[tuple[str, str, dict]], dict[str, dict]]:
        """Lightweight entity extractor using proper noun detection.
        
        Extracts capitalized phrases (likely named entities) instead of every word.
        This creates far fewer entities (~3-5 per passage vs ~100).
        """
        import re
        triplets: list[tuple[str, str, dict]] = []
        entities: dict[str, dict] = {}
        
        for i, doc in enumerate(documents):
            text = doc.get("content", "")
            eid = doc.get("doc_id", f"doc_{self._doc_counter + i}")
            
            # Extract proper nouns: sequences of Capitalized words
            # Also extract the document title/id as an entity
            proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
            
            # Create document entity
            doc_entity = f"DOC_{eid}"
            entities[doc_entity] = {
                "entity_name": doc_entity,
                "entity_type": "DOCUMENT",
                "description": f"Document {eid}",
            }
            
            # Create entity per unique proper noun (limit to 5 per doc)
            seen = set()
            for pn in proper_nouns[:10]:
                if pn.lower() not in seen and len(pn) > 2:
                    seen.add(pn.lower())
                    ent_id = pn
                    entities[ent_id] = {
                        "entity_name": ent_id,
                        "entity_type": "ENTITY",
                        "description": f"Named entity mentioned in {eid}",
                    }
                    triplets.append((doc_entity, ent_id, {"weight": 1.0, "description": "mentions"}))
            
            # Also link co-occurring entities (entities in same document)
            ent_list = [e for e in entities if e != doc_entity and f"in {eid}" in entities[e].get("description", "")]
            for j in range(len(ent_list)):
                for k in range(j+1, len(ent_list)):
                    triplets.append((ent_list[j], ent_list[k], {"weight": 0.5, "description": "co-occurs_in"}))
        
        return triplets, entities
