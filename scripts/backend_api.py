#!/usr/bin/env python3
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BackendAPIError(RuntimeError):
    pass


def _read_error_detail(body: str, fallback: str) -> str:
    try:
        payload = json.loads(body)
    except Exception:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if detail:
        return str(detail)
    return fallback


class BackendClient:
    def __init__(self, backend_url: str, timeout_seconds: int = 600) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = b""
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self.backend_url}{path}",
            data=body if payload is not None else None,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="ignore")
            raise BackendAPIError(
                _read_error_detail(response_body, f"{method} {path} failed ({exc.code})")
            ) from exc
        except URLError as exc:
            raise BackendAPIError(f"{method} {path} failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            raise BackendAPIError(f"{method} {path} returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise BackendAPIError(f"{method} {path} returned non-object JSON payload")
        return data

    def _get(self, path: str) -> dict[str, Any]:
        return self._request_json(method="GET", path=path, payload=None)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json(method="POST", path=path, payload=payload)

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def config(self) -> dict[str, Any]:
        return self._get("/config")

    def ingest(self, force: bool = False) -> dict[str, Any]:
        return self._post("/ingest", {"force": force})

    def query(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        return self._post("/query", {"question": question, "session_id": session_id})
