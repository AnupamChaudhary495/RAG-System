"""Orchestration: read Phase 1 JSON → embed → upsert to Qdrant.

Entry points
------------
Python API:
    from embedding.ingest_vectors import ingest_vectors
    ingest_vectors("./output/chunks.json")

CLI:
    uv run python -m embedding.ingest_vectors ./output/chunks.json [http://localhost:6333]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from embedding.embedder import BGE3Embedder
from embedding.vector_store import (
    COLLECTION_NAME,
    REQUIRED_PAYLOAD_KEYS,
    build_client,
    build_points,
    ensure_collection,
    upsert_points,
)

BATCH_SIZE = 32


def _load_chunks(path: Path) -> list[dict]:
    """Load and return the chunk list from a Phase 1 JSON file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _validate_chunks(chunks: list[dict]) -> None:
    """Ensure every chunk carries all required keys.

    Args:
        chunks: List of chunk dicts.

    Raises:
        ValueError: On the first chunk missing a required key, including its
                    chunk_id for easy debugging.
    """
    for chunk in chunks:
        missing = REQUIRED_PAYLOAD_KEYS - chunk.keys()
        if missing:
            cid = chunk.get("chunk_id", "<unknown>")
            raise ValueError(
                f"Chunk '{cid}' is missing required keys: {sorted(missing)}"
            )


def ingest_vectors(
    chunks_json_path: str,
    qdrant_url: str = "http://localhost:6333",
) -> int:
    """Read Phase 1 chunks, embed with BGE-M3, and upsert into Qdrant.

    Args:
        chunks_json_path: Path to the JSON file produced by Phase 1.
        qdrant_url:       URL of the running Qdrant instance.

    Returns:
        Total number of points successfully upserted.

    Raises:
        FileNotFoundError: If chunks_json_path does not exist.
        ValueError:        If any chunk is missing a required key.
    """
    json_path = Path(chunks_json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_json_path}")

    # ── 1. Load & validate ────────────────────────────────────────────────
    print(f"[ingest] Loading chunks from {json_path} …")
    chunks = _load_chunks(json_path)
    print(f"[ingest] {len(chunks)} chunks loaded.")
    _validate_chunks(chunks)

    # ── 2. Prepare Qdrant ─────────────────────────────────────────────────
    print(f"[ingest] Connecting to Qdrant at {qdrant_url} …")
    client = build_client(qdrant_url)
    ensure_collection(client)

    # ── 3. Load model ─────────────────────────────────────────────────────
    embedder = BGE3Embedder(batch_size=BATCH_SIZE)

    # ── 4. Embed + upsert in batches ──────────────────────────────────────
    total_batches = math.ceil(len(chunks) / BATCH_SIZE)
    total_upserted = 0

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(chunks))
        batch = chunks[start:end]

        print(f"[ingest] Embedding batch {batch_idx + 1}/{total_batches} "
              f"(chunks {start}–{end - 1}) …")
        encoded = embedder.encode_batch([c["text"] for c in batch])

        points = build_points(batch, encoded.dense, encoded.sparse)
        upsert_points(client, points)
        total_upserted += len(points)

    # ── 5. Summary ────────────────────────────────────────────────────────
    print()
    print(f"  Total chunks embedded : {len(chunks)}")
    print(f"  Total points upserted : {total_upserted}")
    print(f"  Collection            : {COLLECTION_NAME}")
    print(f"  Qdrant URL            : {qdrant_url}")

    return total_upserted


# ── CLI entry point ────────────────────────────────────────────────────────

def _cli() -> None:
    """Run ingest_vectors from the command line.

    Usage:
        python -m embedding.ingest_vectors <chunks_json_path> [qdrant_url]
    """
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m embedding.ingest_vectors <chunks_json_path> [qdrant_url]")
        sys.exit(1)

    path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) == 3 else "http://localhost:6333"
    ingest_vectors(path, url)


if __name__ == "__main__":
    _cli()
