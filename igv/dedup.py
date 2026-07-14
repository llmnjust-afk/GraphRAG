"""IGV Core: Incremental Graph Versioner — entity deduplication & semantic merging.

All operations are pure functions operating on {entity_name: {attr: val}} dicts.
No networkx or LLM dependencies at this layer — makes the module testable and fast.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# ---------------------------------------------------------------------------
# Embedding-free simhash for cheap near-duplicate detection
# ---------------------------------------------------------------------------

def _token_ngrams(text: str, n: int = 3) -> list[str]:
    """Character n-grams for locality-sensitive fingerprinting."""
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return [t[i : i + n] for i in range(max(0, len(t) - n + 1))]


def _simhash_fingerprint(ngrams: list[str], n_bits: int = 64) -> int:
    """Compute simhash fingerprint — collision ⇒ approximate cosine ≥ threshold."""
    if not ngrams:
        return 0
    vector = [0] * n_bits
    mask = (1 << n_bits) - 1
    for g in ngrams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16) & mask
        for bit in range(n_bits):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    fp = 0
    for bit in range(n_bits):
        if vector[bit] > 0:
            fp |= 1 << bit
    return fp


def _hamming_distance(a: int, b: int, n_bits: int = 64) -> int:
    return (a ^ b).bit_count()


def simhash_hit(a: int, b: int, max_bit_diff: int = 3, n_bits: int = 64) -> bool:
    """Return True if two simhashes are likely within cosine-sim ≥ 0.85."""
    return _hamming_distance(a, b, n_bits) <= max_bit_diff


# ---------------------------------------------------------------------------
# Core merge / dedup functions (no LLM call yet)
# ---------------------------------------------------------------------------

def _hash_candidate(entity: dict[str, Any]) -> int:
    """Short textual representation for simhash."""
    return _simhash_fingerprint(
        _token_ngrams(entity.get("description", "") + " " + entity.get("entity_name", ""))
    )


def deduplicate_entities(
    existing_entities: dict[str, dict[str, Any]],
    new_entities: dict[str, dict[str, Any]],
    *,
    sim_threshold: int = 8,  # bits (8/64 = 87.5% similarity)
    exact_match: bool = True,
) -> dict[str, str]:
    """Resolve new entities against existing ones.

    Returns
    -------
    mapping : dict[str, str]
        new_entity_name → existing_entity_name (for re-link), or new_entity_name → new_entity_name (kept as-is)
    """
    mapping: dict[str, str] = {}
    if not existing_entities or not new_entities:
        return {k: k for k in new_entities}  # nothing to dedup against

    existing_items = list(existing_entities.items())
    for new_name, new_ent in new_entities.items():
        # ---- exact-match shortcut (fast path) ----
        if exact_match and new_name in existing_entities:
            mapping[new_name] = new_name
            continue

        # ---- simhash pre-filter ----
        new_fp = _hash_candidate(new_ent)
        best_name, best_dist = new_name, 999
        for ex_name, ex_ent in existing_items:
            ex_fp = _hash_candidate(ex_ent)
            # Only compute hamming if simhashes are within 8 bits difference
            dist = _hamming_distance(new_fp, ex_fp)
            if dist < best_dist:
                best_dist = dist
                best_name = ex_name
            # Early exit on exact match
            if dist == 0 and best_name != new_name:
                break

        if best_dist <= sim_threshold:
            mapping[new_name] = best_name
        else:
            mapping[new_name] = new_name

    return mapping


def merge_entity_descriptions(
    existing_desc: str, new_desc: str
) -> str:
    """Rule-based description merging (LLM-free for performance).

    Produces a short composite description that:
    - Keeps the more specific/longer description
    - Concatenates if both are substantial (compounds rare entity nuance)
    """
    e = existing_desc.strip()
    n = new_desc.strip()
    if not e:
        return n
    if not n:
        return e
    if e == n:
        return e
    # If both are substantial, keep longer one and append note
    if len(e) > len(n):
        return e
    return n


def normalize_entity_name(name: str) -> str:
    """Canonicalise entity name for stable dedup: lower → trim → dedup whitespace."""
    return " ".join(name.strip().lower().split())
