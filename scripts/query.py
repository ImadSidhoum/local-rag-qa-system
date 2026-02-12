#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG backend")
    parser.add_argument("question")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--session-id", default=None, help="Optional session ID for multi-turn memory")
    args = parser.parse_args()

    response = requests.post(
        f"{args.backend_url.rstrip('/')}/query",
        json={"question": args.question, "session_id": args.session_id},
        timeout=600,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
