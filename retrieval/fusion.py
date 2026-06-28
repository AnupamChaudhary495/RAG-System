"""Reciprocal Rank Fusion (RRF) — pure Python, zero Qdrant imports.

Merges ranked result lists from multiple retrieval methods using the RRF
formula so that documents appearing highly in any list are promoted, and
documents appearing in multiple lists receive an additional boost.

Formula:
    RRF(d) = Σ  1 / (k + rank_r(d))
             r∈R

where k=60 (standard smoothing constant), rank_r(d) is the 1-based rank
of document d in result list r, and R is the set of all result lists.
"""

from __future__ import annotations

from dataclasses import dataclass

# RRF smoothing constant — intentionally hardcoded per the RAG literature.
_K = 60


@dataclass
class RRFResult:
    """A single candidate after Reciprocal Rank Fusion."""

    chunk_id: str
    """UUID of the chunk (matches Phase 1 / Phase 2 chunk_id)."""

    text: str
    """Chunk text extracted from the Qdrant payload."""

    rrf_score: float
    """Combined RRF score — higher is better, no upper bound."""

    payload: dict
    """Full Qdrant payload dict (text + all metadata fields)."""


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, dict, float]],
) -> list[RRFResult]:
    """Merge ranked search result lists using Reciprocal Rank Fusion.

    Each result list is a sequence of (chunk_id, payload, score) tuples
    already ordered by descending relevance score.  The original scores are
    ignored — only the rank positions matter.

    Dense payload takes priority over sparse when a document appears in
    both lists (first result_list wins for any given chunk_id).

    Args:
        *result_lists: Two or more ranked (chunk_id, payload, score) lists.
                       Passing a single list or empty lists is handled
                       gracefully — the formula still applies.

    Returns:
        Deduplicated list of RRFResult objects sorted by descending
        rrf_score.  Contains at most as many entries as the union of all
        unique chunk_ids across the input lists.
    """
    rrf_scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}      # first-seen payload wins

    for result_list in result_lists:
        for rank_0, (chunk_id, payload, _score) in enumerate(result_list):
            rank_1 = rank_0 + 1         # convert to 1-based rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (_K + rank_1)
            if chunk_id not in payloads:
                payloads[chunk_id] = payload

    return sorted(
        [
            RRFResult(
                chunk_id=cid,
                text=payloads[cid].get("text", ""),
                rrf_score=score,
                payload=payloads[cid],
            )
            for cid, score in rrf_scores.items()
        ],
        key=lambda r: r.rrf_score,
        reverse=True,
    )
