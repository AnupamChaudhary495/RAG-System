# Vector Databases

## 1. Role and Motivation

A vector database executes **Approximate Nearest Neighbor (ANN)** search over high-dimensional continuous vector spaces. This is fundamentally distinct from relational databases (exact row matching) or document stores (exact field matching).

Exact nearest neighbor search via brute-force distance computation has complexity $\mathcal{O}(N \times D)$, where $N$ is the number of vectors and $D$ is dimensionality. For a corpus of 10 million vectors at 3,072 dimensions, a single query requires approximately $3 \times 10^{10}$ floating-point operations — on the order of minutes per query. Vector databases solve this through specialized approximate indexing structures that trade a small, controlled amount of recall for orders-of-magnitude latency improvements.

---

## 2. Core Indexing Algorithms

### Flat Index (Exact k-NN)

Computes exact distances between the query and every stored vector. Guarantees 100% recall but does not scale beyond small corpora (tens of thousands of vectors) due to linear time complexity.

### IVF (Inverted File Index)

Applies k-means clustering to partition vectors into Voronoi cells. At query time, the query vector is compared only to cell centroids, and full distance computation is performed only within the nearest cells. Reduces the search space significantly but introduces boundary artifacts: vectors near cell boundaries may belong to the correct cell's neighbor, causing recall loss. Mitigated by probing multiple nearby cells (`nprobe` parameter), at the cost of increased query time.

### HNSW (Hierarchical Navigable Small World)

The current production standard for high-performance RAG. Constructs a multi-layered proximity graph:

- **Upper layers:** Sparse long-range connections for coarse navigation. Traversal enters the graph here and rapidly moves toward the query region.
- **Lower layers:** Dense short-range connections for fine-grained local search.

At query time, the search descends through layers, refining the candidate set at each level, until it identifies approximate nearest neighbors at the base layer. HNSW provides the best empirical tradeoff between query latency and recall accuracy, at the cost of high memory usage for the graph structure.

---

## 3. Memory Footprint and Quantization

Raw vector storage is a significant memory cost. For `text-embedding-ada-002` / `text-embedding-3-small` (1,536 dimensions, 32-bit float):

$$1536 \times 4 \text{ bytes} = 6{,}144 \text{ bytes} \approx 6.14 \text{ KB per vector}$$

For 10 million vectors:

$$10^7 \times 6{,}144 \text{ bytes} \approx 61.4 \text{ GB}$$

The HNSW graph overhead can approximately double the memory footprint. Production deployments use vector quantization to reduce memory requirements:

| Technique | Mechanism | Memory Reduction | Recall Impact |
|---|---|---|---|
| **Scalar Quantization (SQ8)** | Maps 32-bit floats to 8-bit integers via linear rescaling | ~75% | Minimal |
| **Product Quantization (PQ)** | Splits vectors into sub-vectors; represents each by its nearest cluster centroid | >90% | Moderate (lossy) |
| **Binary Quantization** | Thresholds each dimension to a single bit | ~97% | Higher loss; requires model support |

SQ8 is the standard starting point for memory reduction. PQ is used for very large corpora where SQ8 remains insufficient.

---

## 4. Vector Database Comparison

| Database | Implementation | Core Strengths | Limitations | Production Readiness |
|---|---|---|---|---|
| **Qdrant** | Rust, cloud/local | High throughput; payload-based pre-filtering via BitSets; native sparse vector support; binary quantization | Smaller community than Pinecone | Production-ready |
| **Pinecone** | Proprietary, serverless | Fully managed; strong developer experience; no infrastructure management | Closed-source; high cost at scale; vendor lock-in | Production-ready |
| **Weaviate** | Go, cloud/local | Unified object+vector storage; built-in vectorization modules; GraphQL API | GraphQL API increases integration complexity | Production-ready |
| **Milvus** | C++/Go, cloud/local | Highly scalable distributed architecture; multiple index types; strong Kubernetes support | Higher operational complexity | Production-ready (large-scale) |
| **FAISS** | C++ library (Meta) | Fastest available similarity search primitives; extensive index type support; open-source | Not a database — no CRUD, no metadata filtering, no client/server layer; requires heavy custom wrapping for production use | Specialized / library use |
| **ChromaDB** | Python/TypeScript | In-process or local-disk deployment; minimal setup | No mature distributed scaling; limited concurrency for production workloads | Development / prototyping |

---

## 5. Architectural Trade-offs

### Pre-Filtering vs. Post-Filtering

In multi-tenant deployments, metadata filters (e.g., `tenant_id`, `access_level`, `date_range`) must be applied during retrieval. The execution strategy determines recall behavior:

* **Post-Filtering:** HNSW graph traversal retrieves the top $k$ nearest neighbors without filter constraints. The metadata filter is applied to the result set. If the majority of retrieved vectors fail the filter, effective recall collapses. Requesting $k=10$ may yield 1–2 usable results.
* **Pre-Filtering:** A scalar index is scanned to produce a BitSet of all IDs satisfying the filter. HNSW traversal is constrained to nodes within the BitSet. Guarantees that returned results satisfy the filter; degrades gracefully when the filtered subset is small. Hardware-accelerated BitSet operations (as in Qdrant) minimize the overhead of the constraint.

Pre-filtering is required for access-controlled and multi-tenant deployments. The vector database must support it natively with hardware-level optimization.

### Storage Tiering

Keeping all vectors in RAM is cost-prohibitive at scale. Modern databases (Qdrant, Milvus) support storage tiering:

- **HNSW graph index:** Retained in RAM for fast traversal.
- **Vector payloads:** Memory-mapped from SSD (mmap), streamed on demand.

This reduces hosting costs at the cost of a latency increase for the payload fetch. Suitable when the majority of query latency is dominated by the HNSW traversal rather than the payload load.

---

## Key Takeaways

- Exact k-NN search does not scale; ANN indexing structures (IVF, HNSW) provide sub-millisecond retrieval at the cost of a small, controllable recall loss.
- HNSW is the production standard: it offers the best latency/recall tradeoff but requires significant memory for the graph structure.
- Vector quantization (SQ8, PQ) reduces memory footprint substantially; SQ8 is typically preferred as the first optimization due to minimal recall degradation.
- Pre-filtering via BitSets is the correct approach for multi-tenant and access-controlled deployments; post-filtering causes recall collapse when filter selectivity is high.
- FAISS is a high-performance similarity search library, not a database — it lacks CRUD, metadata filtering, and client/server architecture, and requires significant wrapping for production use.
- Storage tiering (RAM index + SSD payloads) is a cost-latency tradeoff available in enterprise-grade databases for large-scale deployments.
