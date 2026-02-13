from __future__ import annotations

import logging
import os
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def resolve_hf_token(explicit_token: str | None = None) -> str:
    token = (
        explicit_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()
    return token


def _apply_hf_token_to_env(token: str) -> None:
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def load_hf_embeddings(
    model_name: str,
    batch_size: int,
    hf_token: str | None = None,
) -> tuple[HuggingFaceEmbeddings, str]:
    resolved_hf_token = resolve_hf_token(hf_token)
    _apply_hf_token_to_env(resolved_hf_token)

    model_kwargs: dict[str, Any] = {"device": "cpu"}
    if resolved_hf_token:
        model_kwargs["token"] = resolved_hf_token

    logger.info("Loading embedding model: %s", model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": batch_size,
        },
    )
    return embeddings, model_name


def load_sentence_transformer(
    model_name: str,
    hf_token: str | None = None,
) -> tuple[SentenceTransformer, str]:
    resolved_hf_token = resolve_hf_token(hf_token)
    _apply_hf_token_to_env(resolved_hf_token)

    logger.info("Loading sentence-transformer model: %s", model_name)
    model = SentenceTransformer(
        model_name_or_path=model_name,
        token=resolved_hf_token or None,
    )
    return model, model_name


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        hf_token: str | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.model, self.model_name = load_hf_embeddings(
            model_name=model_name,
            batch_size=batch_size,
            hf_token=hf_token,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text)
