# LLM Zoomcamp — Learning Journal

A hands-on follow-along of the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp), building a Retrieval-Augmented Generation (RAG) system from scratch — starting with basic keyword search and evolving to semantic vector search, agentic loops, and orchestrated multi-agent workflows.

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| `minsearch` | Lightweight in-memory text & vector search |
| `sqlitesearch` | Persistent vector search via SQLite + IVF indexing |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |
| `groq` | LLM inference (Llama 3.1 / 3.3) |
| `openai` | OpenAI-compatible LLM client |
| `google-generativeai` | Gemini LLM support |
| `python-dotenv` | API key management via `.env` |
| `uv` | Fast Python package/project manager |
| Jupyter | Interactive notebooks for experimentation |
| `toyaikit` | Educational agent framework with chat UI, tool runners, and LLM clients |
| Kestra | Open-source workflow orchestration platform |
| Docker Compose | Local Kestra + Postgres stack |
| Tavily | Web search API for live retrieval in agents |

---

## Module 1 — Intro to RAG (`/Intro`)

### What we built
A complete RAG pipeline from scratch using **keyword-based full-text search**.

### Key concepts covered

- **Data ingestion** (`ingest.py`): Fetched FAQ data from the DataTalks.Club API across multiple courses, parsed and flattened into a list of documents with fields: `id`, `course`, `section`, `question`, `answer`.

- **Index building**: Used `minsearch.Index` with `text_fields=['question', 'section', 'answer']` and `keyword_fields=['course']` for filtered full-text search.

- **RAG pipeline** (`rag_helper.py`): Built a reusable `RAGBase` class encapsulating:
  - `search()` — keyword search with field boosting (`question: 2.0`, `section: 0.5`) and course filtering
  - `build_context()` — formats top results into a readable context block
  - `build_prompt()` — injects question + context into a prompt template
  - `llm()` — calls the LLM with a system instruction + user prompt
  - `rag()` — orchestrates the full pipeline end-to-end

- **Multiple LLM backends**: Experimented with Groq (Llama 3.1 8B Instant), OpenAI, and Google Gemini via their respective Python SDKs.

- **Persistent storage** (`persistant_rag_ingest.ipynb`, `persistant_rag.ipynb`): Moved from in-memory indexing to a **SQLite-backed** store (`faq.db`) using `sqlitesearch`, so documents don't need to be re-fetched and re-indexed on every run. A separate ingest step populates the DB once; the RAG pipeline then queries it directly.

### Files
| File | Description |
|---|---|
| `notebook.ipynb` | Main exploration notebook — building the RAG pipeline step by step |
| `ingest.py` | Fetches FAQ data from the API and builds a `minsearch` index |
| `rag_helper.py` | `RAGBase` class with search, prompt building, and LLM call logic |
| `rag_ingest.ipynb` | One-time ingestion into SQLite |
| `persistant_rag_ingest.ipynb` | Persistent ingest pipeline using `sqlitesearch` |
| `persistant_rag.ipynb` | RAG pipeline reading from the persistent SQLite DB |

---

## Module 1.5 — Agents (`/Intro/agents`)

### What we built
Extended the RAG system with an **agentic loop** — the LLM can now autonomously decide when and how to call tools, rather than following a fixed pipeline.

### Key concepts covered

- **Function calling**: Defined a `search_faq` tool in the OpenAI tool schema format and passed it to the Groq API, letting the LLM decide when to invoke it based on the user's question.

- **Single tool call**: Observed the model returning a `tool_calls` response instead of a plain text answer, then manually executing the function and feeding results back into the message history.

- **Agentic loop** (`agent_loop()`): Built a `while True` loop that:
  1. Sends messages + tools to the LLM
  2. If the model returns tool calls → executes them and appends results to the message history
  3. If the model returns a text answer → breaks the loop and returns it
  This allows the agent to make **multiple sequential searches** to refine its answer.

- **Prompt engineering for agents**: Crafted system instructions telling the agent to make multiple searches, expand keywords based on results, stay on-topic, and only answer from FAQ data.

- **Off-topic handling**: Demonstrated that with a stricter system prompt the agent correctly declines to answer unrelated questions (e.g. *"what's queen gambit?"*).

- **`toyaikit` chat interface**: Replaced the raw `agent_loop()` with `toyaikit`'s higher-level abstractions:
  - `IPythonChatInterface` — renders an interactive chat UI directly inside Jupyter
  - `OpenAIChatCompletionsClient` — wraps the Groq client (pointed at `https://api.groq.com/openai/v1`) using the `chat.completions` API
  - `OpenAIChatCompletionsRunner` — manages the tool-calling loop, message history, and callbacks declaratively
  - `DisplayingRunnerCallback` — streams assistant responses and tool calls into the chat UI in real time
  - Key gotcha: `OpenAIClient` calls `client.responses.create()` which Groq does not support — must use `OpenAIChatCompletionsClient` instead

- **Model selection for tool calling**: Smaller models (e.g. `llama-3.1-8b-instant`) generate malformed function call syntax and fail with `tool_use_failed`. `llama-3.3-70b-versatile` is required for reliable tool use with `toyaikit`.

### Files
| File | Description |
|---|---|
| `agents/agent.ipynb` | Step-by-step notebook — function calling, single tool call, full agentic loop, `agent_loop()` helper, and `toyaikit` chat interface |

---

## Module 2 — Vector Search (`/Vector Search`)

### What we built
Upgraded the RAG system to use **semantic vector search** — finding documents by *meaning* rather than keyword overlap.

### Key concepts covered

- **Embeddings & similarity** (`vector_search.ipynb`):
  - Loaded `sentence-transformers` model `all-MiniLM-L6-v2` to encode text into 384-dimensional vectors
  - Explored **dot product** as a similarity metric between query and document vectors
  - Verified that semantically similar questions (e.g. *"can I still join?"* vs *"can I still enroll?"*) produce high dot-product scores, while unrelated questions score near zero

- **Batch encoding**: Encoded all 1,368 FAQ documents in batches of 50, producing a full embedding matrix `X` of shape `(1368, 384)`.

- **Brute-force vector search**: Used `numpy` matrix multiplication (`X.dot(query_vector)`) to score all documents, then retrieved top-5 using `np.argsort`.

- **`minsearch.VectorSearch`**: Replaced the manual numpy search with `minsearch`'s built-in `VectorSearch` class, supporting `filter_dict` for course-scoped retrieval — clean and in-memory.

- **`RAGVector` class**: Subclassed `RAGBase` to override the `search()` method — instead of keyword search, it:
  1. Encodes the query string into a vector using the embedder
  2. Queries `VectorSearch` with that vector + course filter
  This made the switch from keyword to vector RAG a minimal, clean change.

- **Persistent vector index** (`vector_search_persistant.ipynb`): Used `sqlitesearch.VectorSearchIndex` with `mode='ivf'` (Inverted File Index) to persist vectors to `faq_vectors2.db` — enabling fast approximate nearest-neighbour search without re-encoding on every run.

- **pgvector** (`vector_search_pgvector.ipynb`): Started exploring PostgreSQL with the `pgvector` extension as a production-grade vector store alternative.

### Files
| File | Description |
|---|---|
| `vector_search.ipynb` | Main notebook — embeddings, similarity, vector RAG pipeline |
| `ingest.py` | Same FAQ data loader; also builds a keyword index (reused from Intro) |
| `rag_helper.py` | Updated `RAGBase` with softer LLM instructions; uses `llama-3.3-70b-versatile` |
| `vector_search_persistant.ipynb` | Persistent vector index using `sqlitesearch` + IVF mode |
| `vector_search_pgvector.ipynb` | Exploration of pgvector as a vector store |
| `faq_vectors2.db` | Persisted vector database (SQLite) |

---

## Module 3 — AI Orchestration with Kestra (`/Orchestration`)

### What we built
Moved from standalone Python scripts to **declarative AI workflows** using [Kestra](https://kestra.io/) — an open-source orchestration platform. Progressively built from a plain LLM chat, through RAG workflows, to autonomous multi-agent systems.

### Key concepts covered

- **Context engineering**: Demonstrated why generic LLM responses fail without grounding — running the same question with and without RAG (`1_chat_without_rag.yaml` vs `2_chat_with_rag.yaml`) to make the difference concrete.

- **RAG in Kestra** (`2_chat_with_rag.yaml`): Used `io.kestra.plugin.ai.rag.IngestDocument` to embed Kestra release docs into a KV store, then queried with `io.kestra.plugin.ai.rag.ChatCompletion` — all declaratively in YAML, no Python code.

- **Web search RAG** (`3_rag_with_websearch.yaml`): Attached Tavily web search as a content retriever, letting the agent pull live data before answering.

- **AI Copilot**: Used Kestra's built-in Copilot to generate and refine flows by describing inputs and goals, rather than writing YAML manually.

- **Simple agent** (`4_simple_agent.yaml`): Built a two-agent chain — a multilingual summariser followed by a brevity agent — with `pluginDefaults` to avoid repeating provider config, and token usage logging for cost monitoring.

- **Web research agent** (`5_web_research_agent.yaml`): A single agent with Tavily as a tool, capable of autonomously searching the web to answer questions.

- **Multi-agent system** (`6_multi_agent_research.yaml`): Composed a main Analyst agent that delegates to a Research agent (as a tool). The research agent uses Tavily to gather live data; the analyst synthesises it into structured JSON. Demonstrates modularity and separation of concerns across agents.

- **Infrastructure** (`docker-compose.yml`): Ran Kestra `v1.3.21` + Postgres locally via Docker Compose. Gemini 2.5 Flash used as the AI Copilot backend; secrets injected via environment variables.

### Files
| File | Description |
|---|---|
| `docker-compose.yml` | Kestra + Postgres stack for local orchestration |
| `flows/1_chat_without_rag.yaml` | Plain LLM chat — shows hallucination without context |
| `flows/2_chat_with_rag.yaml` | RAG flow — ingests docs and queries with grounded context |
| `flows/3_rag_with_websearch.yaml` | RAG + Tavily web search as a live content retriever |
| `flows/4_simple_agent.yaml` | Two-agent chain with token usage logging |
| `flows/5_web_research_agent.yaml` | Single agent with autonomous web search via Tavily |
| `flows/6_multi_agent_research.yaml` | Multi-agent system: Analyst delegates to Research agent |
| `lessons/01-intro.md` → `09-next-steps.md` | Structured lesson notes covering theory and best practices |

---

## Progression Summary

```
Intro              Agents             Vector Search      Orchestration
────────────────────────────────────────────────────────────────────────────
Keyword search  →  Tool-calling    →  Semantic search →  Declarative YAML
In-memory index →  Agentic loop   →  Persistent DB   →  Kestra flows
Single LLM call →  Multi-step LLM →  Same LLM        →  Multi-agent system
RAGBase class   →  agent_loop()   →  RAGVector class →  Kestra AI plugins
Exact matching  →  Autonomous FAQ →  Meaning-based   →  Live web + RAG
```

## Setup

```bash
# Install dependencies with uv
uv sync

# Activate virtual environment
.venv\Scripts\activate   # Windows

# Add your API keys to .env
# GROQ_API_KEY=...
# OPENAI_API_KEY=...
# GOOGLE_API_KEY=...

# Launch Jupyter
jupyter notebook
```

### Module 3 — Kestra (Docker)

```bash
# Add Kestra secrets to Orchestration/.env
# SECRET_GEMINI_API_KEY=...
# SECRET_TAVILY_API_KEY=...
# SECRET_OPENAI_API_KEY=...

# Start Kestra + Postgres
docker compose up -d

# Open Kestra UI
# http://localhost:8080  (admin@kestra.io / Admin1234!)
```