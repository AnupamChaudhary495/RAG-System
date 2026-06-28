"""Cross-encoder reranking using BAAI/bge-reranker-v2-m3.

Wraps FlagEmbedding's FlagReranker.  Scores are raw logits (unbounded) —
higher is more relevant.  The model is loaded once at instantiation and
reused across calls.
"""

from __future__ import annotations

from FlagEmbedding import FlagReranker

from retrieval.fusion import RRFResult

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Loads bge-reranker-v2-m3 once and scores (query, text) pairs.

    Args:
        model_name: HuggingFace model identifier.
        use_fp16:   Load weights in float16 to halve memory usage.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        use_fp16: bool = True,
    ) -> None:
        print(f"[reranker] Loading {model_name} (fp16={use_fp16}) …")
        self._model = FlagReranker(model_name, use_fp16=use_fp16)
        print("[reranker] Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: list[RRFResult],
        top_k: int,
    ) -> list[tuple[RRFResult, float]]:
        """Score each candidate against the query and return the top_k.

        Args:
            query:      The user query string.
            candidates: RRF-fused candidate list (order does not matter here).
            top_k:      Number of results to return after reranking.

        Returns:
            List of (RRFResult, reranker_score) tuples sorted by descending
            score, length = min(top_k, len(candidates)).
        """
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        scores: list[float] = self._model.compute_score(pairs, normalize=False)

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]
