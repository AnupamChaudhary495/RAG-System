# Embeddings

## 1. The Semantic Latent Space

Embedding models transform text into high-dimensional numerical vectors where semantically similar content occupies proximate regions of vector space. This enables similarity-based retrieval as an alternative to exact lexical matching.

Embedding models project text into a continuous vector space, typically ranging from 384 to 3,072 dimensions. Geometric distance in this space encodes semantic relationship. Polysemy is resolved contextually: "bank" in a financial context produces a vector far from "bank" in a geographical context, because the surrounding tokens shift the model's contextual representation.

---

## 2. Distance Metrics

Once text is represented as vectors, retrieval requires a mathematical measure of similarity between the query vector and document vectors.

### Cosine Similarity

Measures the angle between two vectors, independent of their magnitude:

$$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \cdot \sqrt{\sum_{i=1}^{n} B_i^2}}$$

Cosine similarity is effective for text because document length (which affects vector magnitude) should not disproportionately influence relevance ranking.

### Dot Product (Inner Product)

$$\text{Dot Product}(\mathbf{A}, \mathbf{B}) = \sum_{i=1}^{n} A_i B_i$$

When vectors are normalized to unit length ($\|\mathbf{v}\| = 1$), the dot product is mathematically equivalent to cosine similarity. Computing the dot product is faster on CPU/GPU hardware than computing the magnitude terms required for cosine similarity. Normalizing vectors during ingestion and using dot product at query time is a standard production optimization.

### Euclidean Distance ($L_2$ Norm)

Measures straight-line distance between vector endpoints. Sensitive to vector magnitude, making it less suitable for semantic text retrieval compared to cosine or dot product metrics. Commonly used in image embedding spaces.

---

## 3. Embedding Generation: Transformer Architecture

Encoder-only transformers (e.g., BERT and its derivatives) produce a distinct contextualized vector for each input token. Representing a full text chunk as a single vector requires aggregating these token-level representations via a pooling operation:

* **Mean Pooling:** Averages all token vectors across the sequence. The most common approach for sentence and passage embedding.
* **CLS Token:** Some models are trained to encode a special `[CLS]` token at the start of the sequence as an aggregate sequence representation. Used natively by models trained with classification objectives.

---

## 4. Embedding Model Comparison

| Model | Dimensions | Type | Key Characteristics |
|---|---|---|---|
| `text-embedding-3-large` (OpenAI) | Up to 3,072 | Proprietary API | Matryoshka Representation Learning (MRL) support; vectors can be truncated to lower dimensions with graceful performance degradation |
| `text-embedding-3-small` (OpenAI) | Up to 1,536 | Proprietary API | Lower cost; suitable for high-volume, lower-complexity retrieval |
| BGE-M3 (BAAI) | 1,024 | Open-weight, local | Multi-lingual, multi-granularity, multi-function; generates both dense and sparse (lexical) vectors simultaneously — well-suited for hybrid search pipelines |
| Nomic-Embed | 768 | Open-weight, local | Competitive performance at lower dimensionality; suitable for resource-constrained deployments |

**Matryoshka Representation Learning (MRL):** MRL trains the embedding model such that the first $d$ dimensions of a $D$-dimensional vector constitute a meaningful $d$-dimensional embedding. This allows truncation (e.g., from 3,072 to 256 dimensions) at query time, trading a modest precision loss for significant memory and computation savings in large-scale vector stores.

**Local model deployment:** Running open-weight models via HuggingFace Transformers or ONNX runtime eliminates data egress to external APIs — a strict requirement in regulated industries (healthcare, finance, defense).

---

## 5. Embedding Evaluation

Model selection must be grounded in quantitative benchmarks rather than anecdotal performance.

### Benchmarks

* **MTEB (Massive Text Embedding Benchmark):** The standard industry leaderboard. Evaluates models across eight task categories: classification, clustering, pair classification, reranking, retrieval, semantic textual similarity (STS), summarization, and bitext mining.
* **BEIR (Benchmarking Information Retrieval):** A heterogeneous benchmark suite for zero-shot retrieval evaluation across diverse domain-specific corpora.

### Metrics

* **NDCG (Normalized Discounted Cumulative Gain):** Evaluates not just whether a relevant chunk was retrieved, but its position in the ranked list. Retrieval at rank 1 scores higher than retrieval at rank 9 via logarithmic discounting:

$$\text{DCG}_p = \sum_{i=1}^{p} \frac{rel_i}{\log_2(i+1)}$$

NDCG is then normalized against the ideal ranking (IDCG) to produce a score in $[0, 1]$.

* **MRR (Mean Reciprocal Rank):** Averages the reciprocal rank of the first relevant result across queries. Useful for single-answer retrieval tasks.

---

## Key Takeaways

- Embeddings project text into a continuous vector space where geometric proximity encodes semantic similarity, enabling intent-based retrieval beyond exact keyword matching.
- Dot product on pre-normalized unit vectors is computationally equivalent to cosine similarity and faster in practice; normalizing vectors at ingestion time is the standard approach.
- Mean pooling is the dominant aggregation strategy for producing chunk-level embeddings from token-level transformer outputs.
- MRL-trained models (e.g., `text-embedding-3-large`) allow dimension truncation with controlled precision loss, enabling memory-cost tradeoffs in large-scale deployments.
- BGE-M3 produces both dense and sparse vectors simultaneously, making it particularly well-suited for hybrid search pipelines.
- Model selection should be validated against MTEB/BEIR benchmarks using NDCG or MRR; performance on general benchmarks may not transfer to domain-specific corpora without targeted evaluation.
