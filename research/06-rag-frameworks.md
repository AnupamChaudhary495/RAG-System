# RAG Orchestration Frameworks

## 1. Role of Orchestration Frameworks

A RAG pipeline involves a graph of coordinated operations: document parsing, chunking, async embedding API calls, vector database writes and reads, prompt construction, LLM inference, and memory management. Orchestration frameworks provide the connective tissue between these components, handling the asynchronous execution, type routing, error propagation, and component lifecycle that would otherwise require significant custom infrastructure.

---

## 2. LlamaIndex

LlamaIndex was designed specifically for data ingestion and retrieval-augmented generation.

**Core Philosophy:** Data-centric. Optimized for document ingestion, indexing, and retrieval. The LLM is treated as a consumer of a structured data layer.

**Internal Architecture:**
- Introduces `Document` (raw text container) and `Node` (atomic indexed chunk) as first-class primitives.
- Hierarchical relationships between nodes are represented natively, enabling parent-child chunking patterns.
- Specialized `NodeParser` implementations (e.g., `SentenceWindowNodeParser`, `HierarchicalNodeParser`) handle chunking strategies with metadata propagation.
- Query engines and retrievers are composable, allowing hybrid retrieval, reranking, and routing to be assembled declaratively.

**Strengths:** Best-in-class tooling for complex document ingestion, advanced retrieval strategies (hierarchical, semantic, hybrid), and multi-document indexing.

**Limitations:** Less suited for building multi-tool agentic workflows with complex state machines or cyclical logic.

---

## 3. LangChain / LangGraph

LangChain is a general-purpose LLM application framework; LangGraph is its stateful agent runtime.

**Core Philosophy:** Agent-centric. The LLM is the reasoning engine; RAG retrieval is one tool among many that the agent can invoke.

**Internal Architecture:**
- Modern LangChain is built around LCEL (LangChain Expression Language), a declarative composition protocol using the `Runnable` interface. Chains can be streamed, batched, and executed asynchronously with a consistent API.
- LangGraph models agent workflows as directed or cyclical state machines. Nodes represent processing steps; edges encode conditional routing logic. This enables agent loops, retry mechanisms, and multi-step reasoning flows that linear pipelines cannot express.

**Strengths:** Multi-tool agents, complex conditional flows, cyclical reasoning loops (e.g., self-correcting retrieval), and conversational memory management.

**Limitations:** Heavy abstraction layers increase debugging complexity. Component interfaces change frequently across versions, requiring ongoing maintenance.

---

## 4. Haystack (by deepset)

Haystack is a modular pipeline framework oriented toward enterprise NLP and strict production deployments.

**Core Philosophy:** Pipeline-oriented. Data flow is modeled as an explicit Directed Acyclic Graph (DAG). Unlike heavily abstracted frameworks, Haystack requires developers to explicitly declare component inputs, outputs, and connections.

**Internal Architecture (Haystack 2.0):**
- Every processing step is a `Component` with typed input and output ports.
- Pipelines explicitly define the graph topology — which component's output flows into which component's input.
- This explicitness makes distributed tracing, logging, and error isolation straightforward compared to frameworks that hide data flow behind magic abstractions.

**Strengths:** Strong observability, testability, and control over data flow. Well-suited for regulated environments requiring audit trails and deterministic pipeline behavior.

**Limitations:** More boilerplate required compared to higher-level frameworks; less extensive ecosystem of pre-built integrations.

---

## 5. Framework Comparison

| Framework | Core Paradigm | Optimal Use Case | Key Trade-off |
|---|---|---|---|
| **LlamaIndex** | Data-centric ingestion and retrieval | Complex document pipelines, advanced RAG retrieval strategies | Strong on data layer; weaker for multi-tool agent workflows |
| **LangChain / LangGraph** | Agent-centric tool orchestration | Multi-tool agents, cyclical reasoning, conversational state | Broad capability; abstraction complexity increases debugging cost |
| **Haystack** | Explicit DAG pipeline | Enterprise NLP, regulated environments, strict observability requirements | Maximum control and auditability; more verbose to configure |

---

## 6. Combining Frameworks

These frameworks are not mutually exclusive. In a microservice architecture, the data ingestion pipeline and the agent layer can be decoupled:

- **LlamaIndex** handles document ingestion, vector indexing, and hybrid retrieval logic. The resulting query engine is exposed as a standalone HTTP endpoint.
- **LangChain/LangGraph** builds the conversational agent, manages session memory, performs routing decisions, and calls the LlamaIndex query endpoint as an external tool.

This separation of concerns isolates the performance-critical retrieval path from the agent orchestration logic, allowing each to be scaled, optimized, and tested independently.

---

## Key Takeaways

- LlamaIndex excels at data ingestion and retrieval; LangChain/LangGraph excels at multi-tool agent orchestration — they address different layers of the RAG stack.
- Haystack's explicit DAG architecture trades development speed for operational transparency, making it well-suited for regulated or high-observability environments.
- Combining LlamaIndex (retrieval service) with LangGraph (agent runtime) as decoupled microservices is a common production pattern that allows each component to be scaled and maintained independently.
- LCEL's `Runnable` protocol enables consistent streaming, batching, and async execution across LangChain components, reducing the boilerplate of building asynchronous pipelines.
