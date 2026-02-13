#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE=".env"
if [[ ! -f "$ENV_FILE" ]]; then
  ENV_FILE=".env.example"
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing both .env and .env.example at project root."
  exit 1
fi

EMBEDDING_MODEL="${EMBEDDING_MODEL:-$(grep -E '^EMBEDDING_MODEL=' "$ENV_FILE" | cut -d= -f2-)}"
OLLAMA_MODEL="${OLLAMA_MODEL:-$(grep -E '^OLLAMA_MODEL=' "$ENV_FILE" | cut -d= -f2-)}"

if [[ -z "$EMBEDDING_MODEL" ]]; then
  echo "EMBEDDING_MODEL is empty. Set it in .env."
  exit 1
fi

if [[ -z "$OLLAMA_MODEL" ]]; then
  echo "OLLAMA_MODEL is empty. Set it in .env."
  exit 1
fi

echo "[1/2] Pulling Ollama model with progress: $OLLAMA_MODEL"
docker compose up -d ollama
docker compose exec ollama ollama pull "$OLLAMA_MODEL"

echo "[2/2] Downloading embedding model with progress: $EMBEDDING_MODEL"
docker compose build backend
docker compose run --rm --no-deps \
  -e EMBEDDING_MODEL="$EMBEDDING_MODEL" \
  --entrypoint sh backend -lc '
set -e
if command -v hf >/dev/null 2>&1; then
  hf download "$EMBEDDING_MODEL"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$EMBEDDING_MODEL"
else
  python - <<'"'"'PY'"'"'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["EMBEDDING_MODEL"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
snapshot_download(repo_id=repo_id, token=token)
print("download complete")
PY
fi
'

echo "Warmup completed."
