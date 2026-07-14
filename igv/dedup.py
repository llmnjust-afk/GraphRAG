"""IGV entity deduplication — fast embedding-based approach.

Uses sentence-transformers batch encoding + numpy cosine similarity.
Replaces the slow simhash approach (66s for 200×2000 → <0.5s).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

# Lazy-load embedding model (singleton)
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


def deduplicate_entities(
    existing_entities: dict[str, dict[str, Any]],
    new_entities: dict[str, dict[str, Any]],
    *,
    sim_threshold: float = 0.85,
    exact_match: bool = True,
) -> dict[str, str]:
    """Resolve new entities against existing ones via embedding cosine similarity.

    Returns new_entity_name → existing_entity_name (merge) or → self (keep).
    """
    mapping: dict[str, str] = {}
    if not new_entities:
        return mapping
    if not existing_entities:
        return {k: k for k in new_entities}

    # ---- Step 1: exact name match (fast path) ----
    unmatched_new: dict[str, dict] = {}
    for new_name, new_ent in new_entities.items():
        if exact_match and new_name in existing_entities:
            mapping[new_name] = new_name
        else:
            unmatched_new[new_name] = new_ent

    if not unmatched_new:
        return mapping

    # ---- Step 2: embedding-based similarity for remaining ----
    embedder = _get_embedder()

    # Batch encode existing entities
    existing_names = list(existing_entities.keys())
    existing_descs = [
        str(existing_entities[n].get("description", "")) + " " + str(n)
        for n in existing_names
    ]
    existing_embs = embedder.encode(
        existing_descs, batch_size=256, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )  # shape: (N, 384), L2-normalized

    # Batch encode new entities
    new_names = list(unmatched_new.keys())
    new_descs = [
        str(unmatched_new[n].get("description", "")) + " " + str(n)
        for n in new_names
    ]
    new_embs = embedder.encode(
        new_descs, batch_size=256, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )  # shape: (M, 384)

    # ---- Step 3: cosine similarity (vectorized) ----
    # Since embeddings are L2-normalized, dot product = cosine similarity
    sim_matrix = new_embs @ existing_embs.T  # shape: (M, N)

    for i, new_name in enumerate(new_names):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_idx])
        if best_sim >= sim_threshold:
            mapping[new_name] = existing_names[best_idx]
        else:
            mapping[new_name] = new_name

    return mapping


def merge_entity_descriptions(existing_desc: str, new_desc: str) -> str:
    """Keep the longer description (more informative)."""
    e = existing_desc.strip()
    n = new_desc.strip()
    if not e:
        return n
    if not n:
        return e
    if e == n:
        return e
    return e if len(e) > len(n) else n


def normalize_entity_name(name: str) -> str:
    return " ".join(name.strip().lower().split())
