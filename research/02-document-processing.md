# Document Processing

## 1. The Criticality of Data Ingestion

The quality of a RAG system is directly bounded by its document processing pipeline. A production ingestion pipeline is an ETL (Extract, Transform, Load) system requiring structural awareness, metadata enrichment, and mathematically sound chunking — not a simple text extraction script.

---

## 2. PDF Parsing and Document Structuring

PDFs are rendering formats that describe where to draw glyphs on a canvas using absolute coordinates. They do not encode document structure (headings, tables, columns) in a machine-readable form, which makes parsing non-trivial.

* **Naive Text Extraction (Anti-Pattern):** Libraries like PyPDF2 read glyphs in left-to-right, top-to-bottom order by coordinate. Multi-column layouts are interleaved, table structures are destroyed, and hierarchical context is lost.
* **High-Speed Production Extraction:** For high-volume, text-heavy documents, `PyMuPDF` (`fitz`) provides fast C-binding-based extraction with partial bounding-box awareness, suitable for documents without complex layouts.
* **Layout-Aware / Vision Parsing:** For complex enterprise documents (financial reports, research papers with tables and figures), pure text extraction is insufficient. Architectures using `LayoutLMv3` or document intelligence parsers (e.g., LlamaParse, Unstructured.io) apply computer vision to identify bounding boxes around structural elements. Tables are converted deterministically into Markdown prior to embedding, preserving row-column semantics for downstream LLM interpretation.

---

## 3. Text Cleaning and Sanitization

Raw extracted text must be sanitized before embedding to prevent pollution of the vector space.

* **Boilerplate Removal:** Headers, footers, and pagination artifacts (e.g., "Page 12 of 50") must be stripped. Repeated boilerplate across chunks causes the embedding model to artificially cluster semantically unrelated chunks by shared non-content tokens.
* **Regex Sanitization:** Consecutive whitespace, non-printable control characters, and PDF line-break hyphenations (e.g., "com-\nputer" → "computer") must be normalized before embedding.
* **PII Masking:** In regulated environments, tools such as Microsoft Presidio are deployed inline in the ingestion pipeline to detect and redact Personally Identifiable Information via Named Entity Recognition (NER) before text is written to the vector store.

---

## 4. Chunking Strategies

Chunking determines how documents are divided into indexable units. The strategy directly controls retrieval recall.

### A. Fixed-Size and Recursive Character Chunking

* **Fixed-Size Chunking:** Divides text by token count (e.g., 512 tokens). Computationally inexpensive but frequently bisects sentences or concepts across chunk boundaries.
* **Recursive Character Chunking:** The standard baseline. Splits text using a priority-ordered list of separators (`\n\n`, then `\n`, then ` `) applied recursively until chunks fall within the target size. Preserves paragraph and sentence boundaries where possible.
* **Overlap:** To prevent orphaned context at chunk boundaries, a sliding window is used. For chunk size $C$ and overlap $O$, chunk $n+1$ begins at position $C - O$ tokens into chunk $n$:

$$\text{chunk}_{n+1}\text{\_start} = n \cdot (C - O)$$

### B. Semantic Chunking

Groups text by thematic boundaries rather than fixed token counts.

**Mechanism:** Individual sentences are embedded and cosine similarity is computed between adjacent sentence pairs:

$$\text{sim}(S_i, S_{i+1}) = \frac{\vec{v}_i \cdot \vec{v}_{i+1}}{\|\vec{v}_i\| \|\vec{v}_{i+1}\|}$$

**Thresholding:** A chunk boundary is inserted when the similarity drops below a dynamically computed threshold (e.g., the 95th percentile of similarity drops within the document). This produces variable-length chunks that preserve semantic coherence at the cost of predictable sizing.

### C. Hierarchical (Parent-Child) Chunking

Designed to balance retrieval precision with context generation breadth.

* **Structure:** Documents are divided into large "Parent" chunks (e.g., 2,000 tokens) and then subdivided into small "Child" chunks (e.g., 250 tokens).
* **Indexing:** Only Child chunks are embedded and indexed in the vector database. Each Child stores a metadata pointer to its Parent.
* **Retrieval:** Dense search retrieves relevant Child chunks. The system then fetches the corresponding Parent chunk via the metadata pointer and provides the Parent — not the Child — to the LLM context window. This gives the LLM broader surrounding context while keeping the retrieval step precise.

---

## 5. Metadata Management and Database Filtering

Storing unstructured chunks without structured metadata limits the retrieval engine to pure vector similarity. Production systems attach metadata fields (e.g., `source`, `author`, `timestamp`, `department`, `access_level`, `page_number`) to every chunk, enabling pre-filtering before similarity math is applied.

### Pre-Filtering vs. Post-Filtering

When executing a filtered vector search (e.g., "retrieve HR policy chunks where `year = 2026`") against an HNSW index, the query execution strategy significantly affects recall:

* **Post-Filtering:** The HNSW graph traversal runs unconstrained to find the top $k$ nearest neighbors, and the metadata filter is applied to the result set. If most of the $k$ results fail the filter, the effective recall collapses — e.g., requesting $k=10$ may return only 2 usable chunks.
* **Pre-Filtering:** A scalar index is scanned first to construct a BitSet of all document IDs satisfying the filter. HNSW traversal then ignores any node not present in the BitSet. This guarantees that the returned $k$ chunks all satisfy the filter, at the cost of a more constrained graph traversal that may increase query time.

Pre-filtering is the production standard for multi-tenant and access-controlled deployments, despite its higher computational overhead during graph traversal.

---

## Key Takeaways

- PDF parsing fidelity determines the upper bound of chunk quality; layout-aware parsers are required for structured documents.
- Text sanitization prevents embedding space pollution from boilerplate and formatting artifacts.
- Recursive character chunking is the standard baseline; semantic and hierarchical chunking address its limitations for semantically dense or context-sensitive documents.
- Chunk overlap prevents context loss at boundaries; the overlap size is a tunable parameter balancing context retention against index redundancy.
- Metadata pre-filtering is essential for multi-tenant, access-controlled, or time-scoped retrieval and must be supported natively by the vector database.
