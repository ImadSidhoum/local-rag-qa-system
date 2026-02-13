from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 240) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def is_ready(self) -> bool:
        try:
            response = requests.get(self._url("/api/tags"), timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def wait_until_ready(self, attempts: int = 30, sleep_seconds: int = 2) -> bool:
        for _ in range(attempts):
            if self.is_ready():
                return True
            time.sleep(sleep_seconds)
        return False

    def list_models(self) -> set[str]:
        response = requests.get(self._url("/api/tags"), timeout=10)
        response.raise_for_status()
        payload = response.json()
        return {model["name"] for model in payload.get("models", [])}

    def pull_model(self, model_name: str) -> None:
        logger.info("Pulling Ollama model: %s", model_name)
        response = requests.post(
            self._url("/api/pull"),
            json={"name": model_name, "stream": False},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def ensure_model(self, primary_model: str, auto_pull: bool) -> str:
        installed_models = self.list_models()

        if primary_model in installed_models:
            return primary_model

        if auto_pull:
            self.pull_model(primary_model)
            return primary_model

        raise RuntimeError(
            f"Model '{primary_model}' is not available and auto-pull is disabled. "
            f"Install it with: ollama pull {primary_model}"
        )

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = requests.post(
            self._url("/api/chat"),
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        return content.strip()
