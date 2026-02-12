#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from backend_api import BackendAPIError, BackendClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Trigger backend ingestion")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--force", action="store_true", help="Force re-index")
    args = parser.parse_args()

    client = BackendClient(args.backend_url, timeout_seconds=args.timeout_seconds)
    try:
        payload = client.ingest(force=args.force)
    except BackendAPIError as exc:
        raise SystemExit(f"Ingestion failed: {exc}") from exc
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
