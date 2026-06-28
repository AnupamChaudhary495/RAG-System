"""Qdrant sparse vector search.

Accepts an injected QdrantClient — does not instantiate one internally,
making it independently testable with a mock client.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

SearchHit = tuple[str, dict, float]


def sparse_search(
    client: QdrantClient,
    collection_name: str,
    sparse_vector: dict[int, float],
    top_k: int = 50,
) -> list[SearchHit]:
    indices = list(sparse_vector.keys())
    values = [sparse_vector[i] for i in indices]

    result = client.query_points(
        collection_name=collection_name,
        query=SparseVector(indices=indices, values=values),
        using="sparse",
        limit=top_k,
        with_payload=True,
    )
    return [
        (str(hit.id), hit.payload or {}, hit.score)
        for hit in result.points
    ]
