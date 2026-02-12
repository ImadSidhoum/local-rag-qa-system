from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Force rebuilding the vector index")


class IngestResponse(BaseModel):
    status: str
    message: str
    documents: int = 0
    pages: int = 0
    chunks: int = 0
    skipped: bool = False


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, description="Natural language question")


class SourceItem(BaseModel):
    source: str
    page: int
    chunk_id: str
    score: float
    text_excerpt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    model: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    ollama_ready: bool


class ConfigResponse(BaseModel):
    config: dict[str, Any]
