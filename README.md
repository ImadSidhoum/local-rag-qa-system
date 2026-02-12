# Local RAG QA System (100% Open Source)

End-to-end local RAG project using:
- `FastAPI` backend
- `React + TypeScript + Vite` frontend
- `LangChain` for ingestion, chunking, embeddings, retrieval, prompts, and LLM runtime abstraction
- `LangGraph` for query orchestration graph
- `Sentence-Transformers` embeddings
- `ChromaDB` local persistent vector store
- `Ollama` local open-source LLM runtime
- `Langfuse` self-hosted stack (web + worker + db/storage) with optional tracing

This repository answers technical questions over a small PDF corpus and returns grounded answers with explicit source citations (`source + page + chunk_id`).
It also supports session memory so follow-up questions can reference previous turns.

## 1) Project Structure

```text
backend/                # FastAPI app + RAG pipeline
frontend/               # React TypeScript UI
data/                   # corpus + download script
scripts/                # ingest/query/evaluation utilities
docs/                   # explanation + evaluation results
docker-compose.yml      # one-command local stack
```

## 2) Corpus Choice

Corpus used: **"Attention Is All You Need"** (Vaswani et al., 2017), open-access PDF from arXiv.

- URL: `https://arxiv.org/pdf/1706.03762.pdf`
- Size: small technical PDF (~15 pages)
- Legal access: open-access preprint hosted by arXiv

Download corpus:

```bash
bash data/download_corpus.sh
```

## 3) Quickstart (5-minute path)

Prerequisites:
- Docker + Docker Compose

Configuration is loaded from root `.env` (copy from `.env.example` first).
Backend dependency installation inside the container uses `uv`.

Commands:

```bash
cp .env.example .env
bash data/download_corpus.sh
docker compose up --build -d
```

Health check:

```bash
curl http://localhost:8000/health
```

Open:
- Frontend: `http://localhost:5173`
- Langfuse UI: `http://localhost:3000`

Then ingest directly from frontend using the `Ingest Corpus` button.
You can also run evaluation directly from frontend using the `Run Evaluation` button.

## 4) Query the System

### API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What scaling factor is used in scaled dot-product attention?", "session_id":"demo-session-1"}'
```

### CLI helper

```bash
python3 scripts/query.py "How many heads are used in the base Transformer?" --session-id demo-session-1
```

### Frontend UI

Open `http://localhost:5173`

UI features:
- ingest corpus directly from UI (`Ingest Corpus` button)
- run full evaluation from UI (`Run Evaluation` button)
- ask question
- persistent conversation memory via session id
- `New Conversation` button to start a fresh session
- loading/error states
- generated answer
- expandable cited sources with page/chunk metadata
- evaluation progress, aggregate metrics, and artifact links (`csv` / `markdown` / `jsonl`)

## 5) Ingestion API

Idempotent ingest endpoint:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

Force rebuild:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

## 6) Evaluation

Dataset file:
- `scripts/eval_dataset.json` (9 samples, including follow-up and abstention cases)

Run evaluation:

```bash
uv run --with-requirements backend/requirements.txt python scripts/evaluate.py \
  --backend-url http://localhost:8000 \
  --dataset scripts/eval_dataset.json \
  --auto-ingest \
  --output-dir docs
```

Outputs:
- `docs/evaluation_results.csv`
- `docs/evaluation_results.md`
- `docs/evaluation_predictions.jsonl`

Metrics implemented:
- `cosine_similarity` (expected vs generated answer embedding similarity)
- `citation_coverage` (source-page recall against expected sources)
- `source_precision` and `source_f1` (grounding quality)
- `answer_has_citation` (binary presence of citation format in answer)
- `required_term_coverage` (optional per-sample key-term coverage)
- `rewrite_term_coverage` (optional follow-up rewrite quality)
- `idk_correct` (optional abstention correctness)

Session-aware eval:
- Add `conversation_id` in dataset rows to evaluate memory + rewrite over turns.
- Add `expected_rewrite_contains` for follow-up rows that should be rewritten before retrieval.

Backend evaluation job endpoints (used by UI button):
- `POST /eval/run`
- `GET /eval/status/{job_id}`
- `GET /eval/results/{job_id}`
- `GET /eval/artifact/{job_id}/{csv|markdown|jsonl}`

### Run Evaluation from UI

Open `http://localhost:5173` and click `Run Evaluation`.

The UI will:
- create an async job on backend,
- poll progress,
- show aggregate metrics when done,
- expose artifact download links.

Notes:
- only one evaluation job runs at a time (second launch returns `409`)
- backend stores artifacts under `data/eval/<job_id>/`
- UI-triggered eval focuses on local artifacts and does not push eval scores to Langfuse

### Run Evaluation via API

Start job:

```bash
curl -X POST http://localhost:8000/eval/run \
  -H "Content-Type: application/json" \
  -d '{"auto_ingest": true}'
```

Check status:

```bash
curl http://localhost:8000/eval/status/<job_id>
```

Fetch results:

```bash
curl http://localhost:8000/eval/results/<job_id>
```

Download artifact:

```bash
curl -L http://localhost:8000/eval/artifact/<job_id>/markdown
```

## 7) Configuration

Backend config template:
- `backend/.env.example`

Frontend config template:
- `frontend/.env.example`

Compose/runtime config template:
- `.env.example` (copy to `.env`)

Main adjustable knobs:
- chunk size/overlap
- top-k retrieval
- optional MMR retrieval
- evaluation dataset path (`EVAL_DATASET_PATH`) and output directory (`EVAL_OUTPUT_DIR`)
- Chroma telemetry mode (`CHROMA_PRODUCT_TELEMETRY_IMPL`, default no-op)
- temperature/max tokens
- query rewrite (follow-up disambiguation before retrieval)
- memory toggle/window size
- Ollama model and fallback model
- Langfuse tracing flags/keys

## 8) Langfuse (Self-Hosted + Optional Tracing)

Langfuse services are included in `docker compose` and launch with the default stack.

To enable backend tracing:
1. Open `http://localhost:3000` and create/login to Langfuse.
2. Create a project and copy the project public/secret API keys.
3. In `.env`, set the tracing variables:

```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

4. Rebuild backend:

```bash
docker compose up -d --build backend
```

Notes:
- In Docker, backend should use `LANGFUSE_HOST=http://langfuse-web:3000` (default in `.env.example`).
- If running backend outside Docker, set `LANGFUSE_HOST=http://localhost:3000`.

Optional: log evaluation runs to Langfuse:

```bash
uv run --with-requirements backend/requirements.txt python scripts/evaluate.py \
  --backend-url http://localhost:8000 \
  --dataset scripts/eval_dataset.json \
  --langfuse \
  --langfuse-host http://localhost:3000 \
  --langfuse-public-key "$LANGFUSE_PUBLIC_KEY" \
  --langfuse-secret-key "$LANGFUSE_SECRET_KEY"
```

This Langfuse logging path is provided by `scripts/evaluate.py` (`--langfuse`).

## 9) Conversation Memory

Memory is session-based and in-process (kept while backend container is running).

Required `.env` vars:

```bash
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=6
```

How it works:
- Frontend sends a `session_id` with each `/query`.
- Backend includes recent turns in prompt/retrieval to resolve follow-up references.
- Click `New Conversation` in UI to start a fresh memory thread.
- API clients can control memory by reusing/changing `session_id`.

## 10) Query Rewriting Before Retrieval

Follow-up questions are rewritten to standalone form before embedding/retrieval.

Example:
- Q1: `What's a self-attention layer?`
- Q2: `What's its complexity?`
- Rewritten retrieval query: `What is the complexity of a self-attention layer?`

Control via `.env`:

```bash
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MAX_TOKENS=96
```

The rewritten query is returned in API response as `rewritten_question`.

## 11) Troubleshooting

### Ollama model is missing

The backend can auto-pull (`OLLAMA_AUTO_PULL=true`). If disabled, pull manually:

```bash
docker compose exec ollama ollama pull llama3.2:1b
```

### Apple Silicon / low-RAM machine

Use a smaller model in `.env`:
- `OLLAMA_MODEL=llama3.2:1b`

Then restart backend:

```bash
docker compose up -d --build backend
```

### No indexed chunks / query returns 409

Run ingestion first:

Use the frontend `Ingest Corpus` button, or run:

```bash
python3 scripts/ingest.py --backend-url http://localhost:8000
```

### Evaluation start returns 409

An evaluation job is already running. Wait for completion, then relaunch.

Check current job:

```bash
curl http://localhost:8000/eval/status/<job_id>
```

### Evaluation is slow

This is expected on CPU + small local model (`llama3.2:1b`), especially multi-turn samples.
For quicker runs, reduce dataset size or switch to a faster model if your machine supports it.

### Langfuse services are not up

```bash
docker compose ps
docker compose logs --tail=120 langfuse-web langfuse-worker
```

### Full reset

```bash
docker compose down -v
docker compose up --build -d
```

## 12) Git Milestones Used

Conventional commits were used for milestones:
- `chore: init repo`
- `feat(backend): add ingestion pipeline`
- `feat(backend): add retrieval and generation`
- `feat(frontend): add query UI`
- `feat(eval): add evaluation harness`
- `docs: add explanation and README`

## 13) Stop Services

```bash
docker compose down
```

## 14) Local Backend (uv)

If you want to run backend outside Docker:

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```
