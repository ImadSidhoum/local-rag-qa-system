#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS_DIR="${SCRIPT_DIR}/corpus"
mkdir -p "${CORPUS_DIR}"

PDF_URL="https://arxiv.org/pdf/1706.03762.pdf"
OUT_FILE="${CORPUS_DIR}/attention_is_all_you_need.pdf"

if [[ -f "${OUT_FILE}" ]]; then
  echo "Corpus already exists: ${OUT_FILE}"
  exit 0
fi

echo "Downloading corpus PDF from ${PDF_URL} ..."
curl -L --fail --retry 3 --retry-delay 2 "${PDF_URL}" -o "${OUT_FILE}"

echo "Downloaded: ${OUT_FILE}"
