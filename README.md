# Local RAG QA System (100% Open Source)

End-to-end local RAG project using:
- `FastAPI` backend
- `React + TypeScript + Vite` frontend
- `LangChain` for prompt + LLM runtime abstraction
- `LangGraph` for query orchestration graph
- `Sentence-Transformers` embeddings
- `ChromaDB` local persistent vector store
- `Ollama` local open-source LLM runtime
- `Langfuse` (optional) for local/self-hosted tracing

This repository answers technical questions over a small PDF corpus and returns grounded answers with explicit source citations (`source + page + chunk_id`).

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

Backend dependency installation inside the container uses `uv`.

Commands:

```bash
bash data/download_corpus.sh
docker compose up --build -d
python3 scripts/ingest.py --backend-url http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## 4) Query the System

### API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What scaling factor is used in scaled dot-product attention?"}'
```

### CLI helper

```bash
python3 scripts/query.py "How many heads are used in the base Transformer?"
```

### Frontend UI

Open `http://localhost:5173`

UI features:
- ask question
- loading/error states
- generated answer
- expandable cited sources with page/chunk metadata

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
- `scripts/eval_dataset.json` (6 QA samples with expected answers + expected sources)

Run evaluation:

```bash
python3 scripts/evaluate.py \
  --backend-url http://localhost:8000 \
  --dataset scripts/eval_dataset.json \
  --output-dir docs
```

Outputs:
- `docs/evaluation_results.csv`
- `docs/evaluation_results.md`

Metrics implemented:
- `cosine_similarity` (expected vs generated answer embedding similarity)
- `citation_coverage` (expected source-page overlap with returned sources)
- `answer_has_citation` (binary presence of citation format in answer)

## 7) Configuration

Backend config template:
- `backend/.env.example`

Frontend config template:
- `frontend/.env.example`

Main adjustable knobs:
- chunk size/overlap
- top-k retrieval
- optional MMR retrieval
- temperature/max tokens
- Ollama model and fallback model
- Langfuse tracing flags/keys

## 8) Langfuse Tracing (Optional)

Langfuse is integrated via LangChain callbacks and is disabled by default.

Set environment variables (for your local/self-hosted Langfuse instance):

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=your_public_key
export LANGFUSE_SECRET_KEY=your_secret_key
docker compose up -d --build backend
```

## 9) Troubleshooting

### Ollama model is missing

The backend can auto-pull (`OLLAMA_AUTO_PULL=true`). If disabled, pull manually:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

### Low-RAM machine

Use a smaller model in `docker-compose.yml`:
- `OLLAMA_MODEL=llama3.2:1b`

Then restart backend:

```bash
docker compose up -d --build backend
```

### No indexed chunks / query returns 409

Run ingestion first:

```bash
python3 scripts/ingest.py --backend-url http://localhost:8000
```

### Full reset

```bash
docker compose down -v
docker compose up --build -d
```

## 10) Git Milestones Used

Conventional commits were used for milestones:
- `chore: init repo`
- `feat(backend): add ingestion pipeline`
- `feat(backend): add retrieval and generation`
- `feat(frontend): add query UI`
- `feat(eval): add evaluation harness`
- `docs: add explanation and README`

## 11) Stop Services

```bash
docker compose down
```

## 12) Local Backend (uv)

If you want to run backend outside Docker:

```bash
cd backend
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```
