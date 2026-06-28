"""Qdrant dense vector search.

Accepts an injected QdrantClient — does not instantiate one internally,
making it independently testable with a mock client.
"""

from __future__ import annotations

from qdrant_client import QdrantClient

SearchHit = tuple[str, dict, float]


def dense_search(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    top_k: int = 50,
) -> list[SearchHit]:
    result = client.query_points(
        collection_name=collection_name,
        query=dense_vector,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    return [
        (str(hit.id), hit.payload or {}, hit.score)
        for hit in result.points
    ]
