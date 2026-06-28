"""Hybrid retrieval engine for the Enterprise RAG system.

Phase 3: dense + sparse search → Reciprocal Rank Fusion → cross-encoder reranking.

Typical usage::

    from retrieval.retriever import Retriever

    r = Retriever()
    results = r.retrieve("what is reciprocal rank fusion?", top_k=5)
    for res in results:
        print(res.source_filename, res.reranker_score, res.text[:80])
"""

from retrieval.retriever import Retriever, RetrievalResult

__all__ = ["Retriever", "RetrievalResult"]
