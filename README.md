# Enterprise RAG System

A production-grade Retrieval-Augmented Generation system built in pure Python,
managed with [uv](https://docs.astral.sh/uv/).

---

## Architecture

```
PDF files
   │
   ▼
Phase 1 — Ingestion Pipeline (ingestion/)
   │  PyMuPDF / Unstructured.io → sanitize → chunk (512 tok, 50 tok overlap)
   │  Output: output/chunks.json
   ▼
Phase 2 — Embedding & Vector Store (embedding/)
   │  BGE-M3 dense (1024-dim) + sparse vectors → Qdrant "rag_chunks" collection
   ▼
Qdrant  (http://localhost:6333)
```

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Docker](https://www.docker.com/) — for Qdrant

---

## Phase 0 — Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

| Flag | Purpose |
|---|---|
| `-p 6333:6333` | REST API and web UI (used by `qdrant-client` and `curl` queries) |
| `-p 6334:6334` | gRPC API (optional — exposed for high-throughput clients) |
| `-v $(pwd)/qdrant_storage:/qdrant/storage` | Mounts a local directory so vector data persists when the container restarts |
| `qdrant/qdrant` | Official Qdrant Docker image (pulls `latest` tag) |

---

## Phase 1 — Document Ingestion

```bash
# Install dependencies
uv sync

# Run the pipeline (replace ./pdfs with your PDF directory)
uv run python -m ingestion.pipeline ./pdfs ./output/chunks.json
```

Output: `./output/chunks.json` — a JSON array of sanitized, token-counted
text chunks with full provenance metadata.

---

## Phase 2 — Embedding & Vector Storage

```bash
# Run the embedding pipeline
uv run python -m embedding.ingest_vectors ./output/chunks.json

# With a custom Qdrant URL
uv run python -m embedding.ingest_vectors ./output/chunks.json http://localhost:6333
```

---

## Verify Upsert Success

```bash
curl http://localhost:6333/collections/rag_chunks
```

A successful response looks like:

```json
{
  "result": {
    "status": "green",
    "vectors_count": 1234,
    "points_count": 1234,
    "config": {
      "params": {
        "vectors": { "dense": { "size": 1024, "distance": "Cosine" } },
        "sparse_vectors": { "sparse": {} }
      }
    }
  }
}
```

---

## Run Tests

```bash
uv run pytest tests/ -v
```

---

## Project Structure

```
RAG-System/
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py        # ingest_directory() entry point
│   ├── parsers.py         # PyMuPDF + Unstructured.io
│   ├── sanitizer.py       # regex sanitization pipeline
│   ├── chunker.py         # recursive char split + sliding window
│   └── metadata.py        # chunk metadata extraction
├── embedding/
│   ├── __init__.py
│   ├── embedder.py        # BGE-M3 dense + sparse encoding
│   ├── vector_store.py    # Qdrant collection + upsert
│   └── ingest_vectors.py  # orchestration entry point
├── tests/
│   ├── test_pipeline.py   # Phase 1 unit tests (31 tests)
│   └── test_embedding.py  # Phase 2 unit tests (17 tests)
├── research/              # Phase 0 design documentation
├── pyproject.toml
└── uv.lock
```
