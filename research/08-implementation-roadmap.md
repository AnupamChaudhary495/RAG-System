# Implementation Roadmap: Production RAG System

A reference implementation plan for a containerized, production-grade RAG system. The sequence progresses from data ingestion through hybrid retrieval, LLM orchestration, API serving, and observability/evaluation.

---

## Phase 1: Data ETL and Ingestion Pipeline

**Objective:** Produce clean, semantically chunked, metadata-enriched document chunks from raw source files.

**Tasks:**
- Configure a Python environment (Poetry or `uv`) and install `PyMuPDF` (text extraction) and `Unstructured.io` (table/image-aware parsing).
- Implement a sanitization pipeline using regular expressions to remove boilerplate (headers, footers, page numbers) and normalize whitespace and hyphenation artifacts.
- Implement Recursive Character Chunking (512 tokens, 50-token overlap) as the baseline chunking strategy.
- Extract and attach structural metadata to each chunk: `source_filename`, `page_number`, `timestamp`, `section_heading`.

**Deliverable:** A pipeline that accepts a directory of PDFs and outputs a JSON array of metadata-enriched, sanitized text chunks.

---

## Phase 2: Vector Mathematics and Storage

**Objective:** Embed the processed chunks and populate a queryable vector store.

**Tasks:**
- Deploy the BGE-M3 embedding model locally via HuggingFace SentenceTransformers. BGE-M3 produces both dense and sparse vectors, supporting hybrid search without a second model.
- Deploy Qdrant locally via Docker:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```
- Implement an upsert pipeline that embeds each chunk (dense + sparse vectors), and writes the vector payload including all metadata fields to a named Qdrant collection.

**Deliverable:** A populated Qdrant collection with dense vectors, sparse vectors, and metadata payloads per chunk.

---

## Phase 3: Hybrid Retrieval Engine

**Objective:** Implement a multi-stage retrieval pipeline combining dense, sparse, and reranking stages.

**Tasks:**
- Implement dense search via Qdrant's vector similarity API.
- Implement sparse search via Qdrant's native sparse vector API (BM25-compatible).
- Implement Reciprocal Rank Fusion (RRF) to merge dense and sparse result sets into a single ranked list.
- Integrate a cross-encoder reranker (e.g., BGE-Reranker-v2) to rescore the top 50 fused results and return the top 5 by relevance score.

**Deliverable:** A `Retriever` class that accepts a string query and returns the 5 highest-relevance chunks with scores and metadata.

---

## Phase 4: LLM Orchestration and Agentic Routing

**Objective:** Connect the retrieval engine to a generative LLM via a stateful orchestration graph.

**Tasks:**
- Implement a LangGraph state machine with two primary nodes:
  - **Router Node:** A low-latency LLM call (`gpt-4o-mini`) that classifies the query as requiring vector retrieval or as conversational/general (no retrieval needed).
  - **Generator Node:** Constructs a prompt from the query, conversation history, and the 5 retrieved chunks. Constrains the LLM to output structured JSON including the answer and an array of source chunk IDs used.
- Implement an agentic fallback loop: if the Generator Node evaluates retrieved context as insufficient (low coverage), the agent rewrites the query (Query Expansion) and re-executes retrieval before generating a final answer.

**Deliverable:** A compiled LangGraph application that accepts a query, routes it, retrieves context, and returns a grounded, citation-tagged JSON response.

---

## Phase 5: API Layer, Frontend, and Session Memory

**Objective:** Expose the orchestration graph as a production API and manage conversational state.

**Tasks:**
- Wrap the LangGraph application in a FastAPI backend with an asynchronous streaming endpoint (`POST /chat/stream`).
- Deploy Redis (Docker) to store per-session conversation history. Inject the last $n$ messages (typically 5–10) into the LangGraph state on each request to maintain conversational context.
- Implement a frontend (Next.js + TailwindCSS) with a streaming chat interface that renders Markdown and resolves source chunk IDs from JSON responses into inline citation UI components.

**Deliverable:** A full-stack web application with streaming chat, session memory, and verified inline citations.

---

## Phase 6: Observability, Evaluation, and Deployment

**Objective:** Instrument the pipeline for performance monitoring and validate retrieval/generation quality against a labeled evaluation set.

**Tasks:**
- Integrate Langfuse into the FastAPI backend to trace all LLM calls, recording token counts, latency, and cost per pipeline stage.
- Construct a golden evaluation dataset of 50 queries with expected answers covering representative query types.
- Run automated evaluation using Ragas or DeepEval against the golden dataset, measuring:
  - **Context Precision** and **Context Recall** (retrieval layer)
  - **Faithfulness** and **Answer Relevancy** (generation layer)
- Write a `docker-compose.yml` orchestrating all services (Next.js frontend, FastAPI backend, Qdrant, Redis) for single-command local deployment.

**Deliverable:** A containerized, fully instrumented system with CI-integrated evaluation scripts and documented baseline metrics.

---

## Advanced Patterns

The following patterns address production reliability and citation accuracy requirements beyond the baseline implementation:

**Agentic Self-Correction (LangGraph):** The retrieval loop described in Phase 4 can be extended with a formal quality gate. After retrieval, a dedicated evaluation step scores context coverage. Below a threshold score, the agent executes query expansion (generating alternative phrasings) and re-queries before proceeding to generation. This prevents low-recall retrievals from propagating to the generative layer.

**Deterministic Citation Verification:** Rather than instructing the LLM to reproduce citation text (which is prone to hallucination), the LLM outputs only the IDs of the chunks it used. The application layer looks up those IDs in the vector store or a document metadata index and renders the verified source, page, and passage as a UI citation component. This decouples citation accuracy from LLM instruction-following reliability.

---

## Key Takeaways

- The ingestion and retrieval pipeline should be built and validated independently before connecting the generative layer; retrieval quality gates the entire system.
- BGE-M3's simultaneous dense and sparse vector output eliminates the need for a separate BM25 index in Qdrant-based hybrid search pipelines.
- Agentic fallback loops (query expansion + re-retrieval) improve recall for ambiguous or poorly phrased queries without requiring the user to reformulate.
- Deterministic chunk ID citation is more reliable than LLM-generated citations and should be the standard approach for knowledge-sensitive production systems.
- Evaluation against a labeled golden dataset with Ragas/DeepEval metrics (Faithfulness, Context Recall) is the only reliable method to quantify system performance and detect regressions after pipeline changes.
