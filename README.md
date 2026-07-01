# RAG System

A production-grade Retrieval-Augmented Generation system built in Python with a Next.js frontend. Uses local LLMs via Ollama — no OpenAI API key required.

---

## Architecture

```
User Query (Next.js frontend)
        │
        ▼
FastAPI — SSE streaming endpoint (port 8000)
        │
        ▼
LangGraph Agentic Pipeline
   ├── Router node      — classifies query: retrieve or generate
   ├── Retriever node   — hybrid search (dense + sparse + RRF + rerank)
   ├── Rewriter node    — rewrites query if confidence too low (retry loop)
   └── Generator node   — synthesises answer with citations via llama3.2
        │
        ├── BGE-M3 (BAAI/bge-m3)
        │   Dense (1024-dim cosine) + Sparse vectors → Qdrant
        │
        └── BGE Reranker (BAAI/bge-reranker-v2-m3)
            Cross-encoder reranks top-61 RRF candidates → top-5 to LLM
        │
        ▼
Redis — session memory (conversation history, TTL 24h)
        │
        ▼
Next.js frontend — streams tokens, renders citations
```

---

## Models

| Role | Model | Runtime |
|---|---|---|
| Router & Generator | `llama3.2` | Ollama (local) |
| Embedder | `BAAI/bge-m3` | HuggingFace (in-process) |
| Reranker | `BAAI/bge-reranker-v2-m3` | HuggingFace (in-process) |

---

## 🚀 One-click start (Windows)

**Double-click [`RAG-Assistant.bat`](RAG-Assistant.bat)** in the project root.

That single launcher does everything, end to end:

1. Installs any missing dependencies — `uv`, Node.js, Ollama, and the Qdrant + Redis binaries (downloaded automatically)
2. Installs Python (`uv sync`) and frontend (`npm install`) packages
3. Pulls the `llama3.2` model and creates a default `.env`
4. Builds the frontend, starts **Redis + Qdrant + Ollama + backend + frontend**
5. Ingests the knowledge base (`research/*.md`) if the vector store is empty
6. Opens the app in **its own window** (Chrome/Edge `--app` mode)

Press **Enter** in the launcher window to shut everything down. If you closed it
without stopping, run [`scripts/stop.ps1`](scripts/stop.ps1).

```powershell
# Equivalent manual invocation / useful switches:
powershell -ExecutionPolicy Bypass -File scripts\launch.ps1
#   -SkipInstall   skip dependency install/build (fast restart)
#   -CheckOnly     report tool + service status, then exit
```

> First launch downloads models (~2–3 GB) and builds the frontend, so it takes a
> few minutes. Subsequent launches are fast. The first query also warms the
> in-process BGE-M3 / reranker models.

### Install as a desktop app (PWA)

The frontend is a Progressive Web App. In Chrome/Edge, click the **install icon**
in the address bar (or the **Install app** button in the sidebar) to install
"RAG Assistant" as a standalone windowed app.

---

## Prerequisites (handled automatically by the launcher)

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [Ollama](https://ollama.com/) — local LLM server
- [Qdrant](https://qdrant.tech/) — vector database
- [Redis](https://redis.io/) — session store
- Node.js 18+ — for the frontend

---

## Manual start (any OS)

<details>
<summary>Step-by-step, if you prefer to run each piece yourself</summary>

```bash
# 1. Python deps
uv sync

# 2. Frontend deps
cd frontend && npm install && cd ..

# 3. Model + env
ollama pull llama3.2
cp .env.example .env      # then set Ollama values (see Environment Variables)

# 4. Services (each in its own terminal)
.services/redis/redis-server.exe
.services/qdrant/qdrant.exe --config-path .services/qdrant_config.yaml
ollama serve              # skip if Ollama already runs as a service

# 5. Ingest the knowledge base
uv run python ingest_markdown.py

# 6. Backend + frontend
uv run uvicorn api.main:app --port 8000
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

</details>

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (Redis status) |
| `POST` | `/chat/stream` | SSE streaming chat |
| `GET` | `/session/{id}/history` | Retrieve conversation history |
| `DELETE` | `/session/{id}` | Clear session |
| `GET` | `/chunks` | Fetch source chunks by ID |

### Example request

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "session_id": "my-session"}'
```

SSE event types: `token`, `metadata`, `error`, `done`.

---

## Run Tests

```bash
uv run pytest tests/ -v
```

---

## Project Structure

```
RAG-System/
├── ingestion/              # Phase 1 — document parsing & chunking
│   ├── pipeline.py
│   ├── parsers.py          # PyMuPDF + Unstructured.io fallback
│   ├── sanitizer.py
│   ├── chunker.py
│   └── metadata.py
├── embedding/              # Phase 2 — BGE-M3 embedding & Qdrant upsert
│   ├── embedder.py
│   ├── vector_store.py
│   └── ingest_vectors.py
├── retrieval/              # Phase 3 — hybrid retrieval engine
│   ├── dense_search.py     # Qdrant dense vector search
│   ├── sparse_search.py    # Qdrant sparse vector search
│   ├── fusion.py           # Reciprocal Rank Fusion
│   ├── reranker.py         # BGE cross-encoder reranker
│   └── retriever.py        # Orchestrates full retrieval pipeline
├── orchestration/          # Phase 4 — LangGraph agentic pipeline
│   ├── state.py            # RAGState TypedDict
│   ├── app.py              # LangGraph graph builder
│   └── nodes/
│       ├── router.py
│       ├── retriever_node.py
│       ├── generator.py
│       └── rewriter.py
├── api/                    # Phase 5 — FastAPI SSE backend
│   ├── main.py
│   ├── session.py          # Redis session memory
│   ├── schemas.py
│   └── streaming.py
├── frontend/               # Phase 5 — Next.js 14 chat UI
│   ├── app/
│   ├── components/
│   └── lib/
├── research/               # Source markdown documents (knowledge base)
├── ingest_markdown.py      # One-shot ingestion script for research/
├── ocr_ingest.py           # GPT-4o Vision OCR for image-based PDFs
├── .env.example
├── pyproject.toml
└── uv.lock
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `ollama` | Set to your OpenAI key to use cloud models |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | LLM endpoint (Ollama or OpenAI) |
| `ROUTER_MODEL` | `llama3.2` | Model for query classification |
| `GENERATOR_MODEL` | `llama3.2` | Model for answer generation |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection string |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS allowed origins |
