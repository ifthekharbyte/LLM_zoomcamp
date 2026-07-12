# LLM Zoomcamp — Learning Journal

A hands-on follow-along of the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp), building a Retrieval-Augmented Generation (RAG) system from scratch — starting with basic keyword search and evolving to semantic vector search.

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

## Progression Summary

```
Intro                          Vector Search
─────────────────────────────────────────────────────────────
Keyword search (minsearch)  →  Semantic search (embeddings)
In-memory index             →  Persistent SQLite vector DB
Single LLM (Groq)           →  Same LLM, smarter retrieval
RAGBase class               →  RAGVector subclass
Exact term matching         →  Meaning-based similarity
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