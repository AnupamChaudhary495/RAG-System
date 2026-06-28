"""Cross-encoder reranking using BAAI/bge-reranker-v2-m3.

Wraps FlagEmbedding's FlagReranker.  Scores are raw logits (unbounded) —
higher is more relevant.  The model is loaded once at instantiation and
reused across calls.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from retrieval.fusion import RRFResult

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        use_fp16: bool = True,
    ) -> None:
        print(f"[reranker] Loading {model_name} (fp16={use_fp16}) …")
        self._model = CrossEncoder(model_name)
        print("[reranker] Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: list[RRFResult],
        top_k: int,
    ) -> list[tuple[RRFResult, float]]:
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]
