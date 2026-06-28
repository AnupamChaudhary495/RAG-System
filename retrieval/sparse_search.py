"""Qdrant sparse vector search.

Accepts an injected QdrantClient — does not instantiate one internally,
making it independently testable with a mock client.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedSparseVector, SparseVector


# Type alias for a search hit: (chunk_id, payload, score)
SearchHit = tuple[str, dict, float]


def sparse_search(
    client: QdrantClient,
    collection_name: str,
    sparse_vector: dict[int, float],
    top_k: int = 50,
) -> list[SearchHit]:
    """Search the Qdrant collection using the sparse (lexical) vector index.

    Args:
        client:          Connected QdrantClient instance.
        collection_name: Name of the target Qdrant collection.
        sparse_vector:   Query sparse embedding as {token_id: weight}.
        top_k:           Number of candidates to retrieve.

    Returns:
        List of (chunk_id, payload, score) tuples ordered by descending
        sparse similarity score, length ≤ top_k.
    """
    indices = list(sparse_vector.keys())
    values = [sparse_vector[i] for i in indices]

    hits = client.search(
        collection_name=collection_name,
        query_vector=NamedSparseVector(
            name="sparse",
            vector=SparseVector(indices=indices, values=values),
        ),
        limit=top_k,
        with_payload=True,
    )
    return [
        (str(hit.id), hit.payload or {}, hit.score)
        for hit in hits
    ]
