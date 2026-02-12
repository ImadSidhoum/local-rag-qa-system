#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from backend_api import BackendAPIError, BackendClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG backend")
    parser.add_argument("question")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--session-id", default=None, help="Optional session ID for multi-turn memory")
    args = parser.parse_args()

    client = BackendClient(args.backend_url, timeout_seconds=args.timeout_seconds)
    try:
        payload = client.query(question=args.question, session_id=args.session_id)
    except BackendAPIError as exc:
        raise SystemExit(f"Query failed: {exc}") from exc
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
