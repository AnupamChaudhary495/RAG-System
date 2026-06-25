# Retrieval Techniques

## 1. Retrieval Pipeline Design

Production retrieval is a multi-stage funnel, not a single similarity search. The pipeline is designed to maximize recall in early stages (retrieving all potentially relevant chunks) and then optimize for precision in later stages (eliminating irrelevant chunks before LLM context injection). A failure in retrieval cannot be compensated by the generative model.

---

## 2. Dense Search (Semantic Similarity Search)

Dense search maps both queries and documents into a shared continuous embedding space and retrieves documents by geometric proximity.

**Mechanism:** The query is encoded into a vector using the same embedding model used during ingestion. ANN algorithms (typically HNSW) identify the closest document vectors in latent space.

**Strengths:** Captures semantic intent and paraphrase. A query for "cats" retrieves documents discussing "felines" because their embeddings occupy nearby regions in vector space.

**Limitations:** Poor performance on exact lexical matches. Queries containing specific identifiers — error codes (e.g., `ERR-992-B`), SKUs, employee IDs, medical acronyms — may produce embeddings that cluster generically with similar-type terms, failing to surface the exact match. Dense search encodes meaning, not character sequences.

---

## 3. Sparse Search (BM25)

Sparse search uses term-frequency-based scoring from classical Information Retrieval to address the exact-match limitation of dense search.

**Mechanism:** Documents are represented as sparse vectors of term weights. BM25 (Best Matching 25) is the industry standard, extending TF-IDF with document length normalization and term frequency saturation:

$$\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$: term frequency of query term $q_i$ in document $D$
- $|D|$: document length in tokens
- $\text{avgdl}$: average document length across the corpus
- $k_1$: term frequency saturation parameter (typically 1.2–2.0); prevents infinitely long documents from scoring arbitrarily high
- $b$: length normalization parameter (typically 0.75)

**Strengths:** Guarantees exact string recall. Documents containing the precise query term receive a strongly boosted relevance score regardless of semantic distance.

**Limitations:** No semantic generalization. A BM25 search for "cardiac arrest" does not retrieve documents about "heart failure" unless the exact term appears.

---

## 4. Hybrid Search

Hybrid search combines dense and sparse retrieval simultaneously, capturing both semantic relevance and exact-match precision. The results of each method must be fused into a single ranked list despite their score distributions being on incompatible scales (cosine similarity vs. BM25 scores).

### Reciprocal Rank Fusion (RRF)

RRF ignores raw scores entirely and fuses based on ranking position:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

Where $k$ (typically 60) is a smoothing constant that dampens the advantage of top-ranked documents and reduces sensitivity to outliers. Documents appearing in the top ranks of both dense and sparse results receive the highest fused scores. Documents appearing in only one result set are penalized.

RRF is robust to score distribution differences and requires no calibration.

### Alpha Fusion (Convex Combination)

When both retrieval scores are normalized to $[0, 1]$:

$$\text{Score} = \alpha \cdot \text{Dense\_Score} + (1 - \alpha) \cdot \text{Sparse\_Score}$$

The weight $\alpha$ is a tunable parameter. A value of $\alpha = 0.7$ (70% semantic, 30% lexical) is a common baseline for enterprise workloads but should be tuned against a domain-specific evaluation set. Alpha fusion requires careful normalization to prevent one score distribution from dominating.

---

## 5. Two-Stage Retrieval: Reranking

Dense, sparse, and hybrid retrieval all rely on **bi-encoders**: the query and document are encoded independently, and similarity is computed from the resulting vectors. This is computationally efficient but loses cross-attention between query and document tokens.

**Cross-Encoders** address this limitation. A cross-encoder processes the query and document chunk jointly within the same transformer context window:

```
[CLS] <query tokens> [SEP] <document chunk tokens> [SEP]
```

Self-attention heads can compute token-level interactions between query and document simultaneously, producing a calibrated relevance score in $[0, 1]$. This is substantially more accurate than bi-encoder similarity but computationally infeasible at corpus scale.

### Two-Stage Retrieval Pipeline

| Stage | Method | Retrieve | Purpose |
|---|---|---|---|
| Stage 1 | Hybrid search (bi-encoder) | Top $k=50$ chunks | High recall at low latency |
| Stage 2 | Cross-encoder reranker | Top 5 reranked chunks | High precision for LLM context |

The reranker operates only on the small candidate set from Stage 1, making the compute cost tractable. The final top-ranked chunks are injected into the LLM prompt.

---

## 6. Metadata Filtering and Self-Querying Retrievers

Vector similarity is agnostic to document metadata. A query for "Q3 revenue numbers" may retrieve semantically similar documents from multiple years because their embeddings are geometrically close.

**Hard Filtering:** Metadata pre-filters constrain the ANN search to a specific document subset before similarity computation (e.g., `date > 2025-01-01 AND department = Finance`). Implemented via BitSet intersection as described in Section 4.

**Self-Querying Retrievers:** A routing LLM parses the natural language query and extracts structured metadata filters automatically:

- *Input query:* "Show me HR policies about remote work from this year."
- *Router output:* `{"semantic_query": "remote work policies", "filter": {"department": "HR", "year": 2026}}`

The generated filter is passed to the vector database as a pre-filter, ensuring similarity search operates only over the semantically valid document subset. This approach handles temporal, categorical, and access-scoped queries without requiring users to formulate structured queries manually.

---

## Key Takeaways

- Dense search provides semantic generalization; sparse search (BM25) provides exact lexical recall — neither alone is sufficient for production workloads.
- Hybrid search combining both with RRF is the production standard; RRF is preferred over alpha fusion when score normalization is impractical.
- Two-stage retrieval (bi-encoder for recall, cross-encoder for precision) enables high-accuracy context selection at scale: the cross-encoder's superior relevance estimation is applied only to the small candidate set produced by the fast first stage.
- BM25 parameters $k_1$ and $b$ control term frequency saturation and length normalization; default values (1.2, 0.75) are reasonable baselines but domain-specific tuning improves performance.
- Metadata pre-filtering is required for temporally scoped, categorically scoped, or multi-tenant retrieval; self-querying retrievers automate filter extraction from natural language queries.
