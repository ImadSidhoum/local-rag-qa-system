# Technical Explanation

## 1) Corpus Selection and Justification

The corpus is the open-access technical PDF **"Attention Is All You Need"** (arXiv:1706.03762). It is a strong fit because:
- It is freely and legally accessible.
- It is compact (~15 pages), matching the requested small-corpus scope.
- It contains dense technical content suitable for retrieval-grounded QA (definitions, formulas, architecture details, training setup).

The file is not committed directly; instead, `data/download_corpus.sh` provides deterministic legal download for reproducibility.

## 2) RAG Architecture

Orchestration stack:
- `LangGraph`: query flow graph (`rewrite -> retrieve -> gate -> generate`)
- `LangChain`: prompt templates + Ollama chat model invocation
- `Langfuse`: self-hosted observability stack in Docker Compose + optional tracing callbacks
- `uv`: Python dependency installation/runtime command wrapper in backend workflows

Configuration:
- Runtime settings are centralized in root `.env` (loaded by Docker Compose via `env_file`).

### Ingestion

- Parser: LangChain `PyPDFLoader` for page-wise extraction with standardized `Document` objects.
- Metadata captured per chunk:
  - `source` (filename)
  - `page`
  - `chunk_id`
  - `chunk_index`
  - chunk `text`

### Chunking Strategy

Implemented with LangChain `RecursiveCharacterTextSplitter` in `backend/app/chunking.py`:
- Character-based chunks with configurable overlap.
- Defaults: `chunk_size=900`, `chunk_overlap=150`.
- Splitter separators prioritize paragraph/sentence/space boundaries.

Reasoning:
- 900 chars keeps enough local context for equations/definitions without overloading retrieval.
- 150 overlap preserves continuity for split concepts.

### Embeddings

Model: `sentence-transformers/all-MiniLM-L6-v2`.

Why:
- Open-source and lightweight.
- Good retrieval quality for short technical passages.
- Fast CPU inference and stable local operation.

The embedding pipeline uses LangChain `HuggingFaceEmbeddings` and is deterministic:
- fixed random seed,
- sorted file order,
- stable metadata/chunk-id generation.

### Vector Store

Store: local persistent Chroma via LangChain `Chroma` wrapper.

Why:
- Fully local, open-source, simple persistence.
- Good developer ergonomics for metadata + vector queries.
- Easy reset/rebuild flow for idempotent ingestion.

Persistence is backed by Docker named volume `chroma_data`.
Telemetry is hard-disabled with a no-op Chroma telemetry client to avoid noisy runtime warnings.

### Retrieval

- Default top-k retrieval (`TOP_K=4`) via LangChain retriever interface.
- Optional MMR reranking (`USE_MMR=true`) with configurable `MMR_CANDIDATES` and `MMR_LAMBDA`.
- Similarity threshold (`MIN_SIMILARITY`) triggers a safe abstention response.
- Retrieval, gating, and generation are executed by a compiled LangGraph state machine for deterministic flow.
- If memory is enabled, recent session turns are included in retrieval query expansion to improve follow-up resolution.
- Graph nodes are `rewrite -> retrieve -> gate -> generate`.

### Generation

Runtime: `Ollama` service in Docker.

Default model: `llama3.2:1b` (configurable), chosen for good local speed/memory on Apple Silicon laptops, with fallback (`llama3.2:1b`).

Prompt design enforces:
- use context only,
- abstain with exact text if unsupported,
- include structured citations: `[source=<filename> page=<page> chunk=<chunk_id>]`.

### Observability (Langfuse)

- Langfuse is deployed as part of `docker-compose` with:
  - `langfuse-web`
  - `langfuse-worker`
  - `langfuse-postgres`
  - `langfuse-clickhouse`
  - `langfuse-minio`
  - `langfuse-redis`
- Backend tracing is wired via LangChain callback integration.
- Tracing is disabled by default; enable with:
  - `LANGFUSE_ENABLED=true`
  - `LANGFUSE_HOST=http://langfuse-web:3000` (Docker internal URL)
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`

### Conversation Memory

- Session memory is supported via `session_id` on `/query`.
- Memory controls:
  - `MEMORY_ENABLED` (default `true`)
  - `MEMORY_MAX_TURNS` (default `6`)
- Memory is in-process (resets when backend restarts).

### Query Rewrite

- Controls:
  - `QUERY_REWRITE_ENABLED` (default `true`)
  - `QUERY_REWRITE_MAX_TOKENS` (default `96`)
- Rewriter uses conversation history to resolve references (e.g., `its`, `that`) before vector search.

## 3) API and Frontend

### FastAPI Endpoints

- `GET /health`
- `POST /ingest` (idempotent, optional `force`)
- `POST /query` → `{answer, sources[], session_id}` with scores + excerpts
- `GET /config` (safe runtime knobs)
- `POST /eval/run` + `GET /eval/status/{job_id}` + `GET /eval/results/{job_id}` for async evaluation jobs
- `GET /eval/artifact/{job_id}/{csv|markdown|jsonl}` for report files

### React Frontend

- Question text area
- Submit/loading/error states
- Rendered answer with model name
- Expandable source sections showing page/chunk provenance
- Evaluation panel with run button, progress polling, aggregate metrics, and artifact links

## 4) Evaluation Methodology

Evaluation dataset: `scripts/eval_dataset.json` with 9 samples:
- standalone factual QA
- follow-up multi-turn QA (`conversation_id`) for memory/rewrite behavior
- one explicit abstention sample (`expect_idk=true`)

Runner: `scripts/evaluate.py`.
The runner calls the same FastAPI endpoints used by the app (`/health`, `/config`, `/query`, optional `/ingest`) and can replay multi-turn sessions via `conversation_id`.
The web UI also triggers asynchronous backend evaluation jobs (`/eval/run`, `/eval/status/{job_id}`), reusing the same dataset and metrics logic.

Metrics:
1. `cosine_similarity`: embedding similarity between expected and generated answers.
2. `citation_coverage`: expected source-page recall.
3. `source_precision` / `source_f1`: grounding quality on returned citations.
4. `answer_has_citation`: citation format presence in final answer.
5. Optional per-sample metrics:
   - `required_term_coverage`
   - `rewrite_term_coverage`
   - `idk_correct`

Artifacts:
- `docs/evaluation_results.csv`
- `docs/evaluation_results.md`
- `docs/evaluation_predictions.jsonl`
- UI/API job artifacts are persisted under `data/eval/<job_id>/`

Langfuse integration:
- Evaluation runs can optionally log per-sample traces and metric scores to Langfuse.
- This is useful for comparing regressions and debugging low-scoring samples from the Langfuse UI.
- CSV/Markdown artifacts remain the deterministic offline record.
- Langfuse score logging is currently implemented in the script runner (`scripts/evaluate.py --langfuse`).

## 5) Results Summary

Current run summary is in `docs/evaluation_results.md`.

Interpretation guidance:
- High cosine + high citation coverage indicates both answer quality and grounding.
- High cosine + low coverage suggests semantically good answer but weak retrieval attribution.
- Missing inline citations indicate prompt adherence issues.

## 6) Limits and Improvements

Known limitations:
- Single-paper corpus; broader corpora need better document routing and compression.
- No second-stage reranker by default.
- No advanced factuality metric (e.g., RAGAS) due local simplicity and runtime cost.
- Evaluation jobs are serialized (one active job at a time) and currently have no cancel endpoint.

Improvements with more time:
- Add reranker (`bge-reranker`) to improve top-k precision.
- Add hybrid retrieval (BM25 + dense embeddings).
- Add async batch ingest and richer extraction fallback (`pdfplumber` when OCR-like PDFs fail).
- Add regression test suite around retrieval/citation behavior.
