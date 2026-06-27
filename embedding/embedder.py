"""BGE-M3 embedding model — dense and sparse vectors.

The model is loaded once at instantiation (use_fp16=True to halve VRAM
usage).  Call encode_batch() with a list of strings; it returns dense
float32 vectors (1024-dim) and sparse dicts {token_id: weight} for each
input.
"""

from __future__ import annotations

from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel

BATCH_SIZE = 32
MODEL_NAME = "BAAI/bge-m3"
DENSE_DIM = 1024


@dataclass
class EncodedBatch:
    """Output of a single encode_batch() call."""

    dense: list[list[float]]
    """Dense float32 vectors, each of length DENSE_DIM (1024)."""

    sparse: list[dict[int, float]]
    """Sparse vectors as {token_id (int): weight (float)} dicts."""


class BGE3Embedder:
    """Loads BAAI/bge-m3 once and exposes batched dense+sparse encoding.

    Args:
        model_name: HuggingFace model identifier (default BAAI/bge-m3).
        use_fp16:   Load weights in float16 to reduce memory usage.
        batch_size: Number of texts per model.encode() call.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        use_fp16: bool = True,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        print(f"[embedder] Loading {model_name} (fp16={use_fp16}) …")
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.batch_size = batch_size
        print("[embedder] Model ready.")

    def encode_batch(self, texts: list[str]) -> EncodedBatch:
        """Encode a list of texts and return dense + sparse vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            EncodedBatch with parallel dense and sparse lists.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")

        output = self._model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs: list[list[float]] = [
            vec.tolist() for vec in output["dense_vecs"]
        ]

        # FlagEmbedding returns sparse weights keyed by string token IDs.
        # Qdrant requires integer indices — cast them here.
        sparse_vecs: list[dict[int, float]] = [
            {int(token_id): float(weight) for token_id, weight in sv.items()}
            for sv in output["lexical_weights"]
        ]

        return EncodedBatch(dense=dense_vecs, sparse=sparse_vecs)
