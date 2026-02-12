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
    session_id: str | None = Field(
        default=None,
        description="Optional conversation session identifier for multi-turn memory",
    )


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
    session_id: str | None = None
    rewritten_question: str | None = None


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    ollama_ready: bool


class ConfigResponse(BaseModel):
    config: dict[str, Any]


class EvalRunRequest(BaseModel):
    dataset_path: str | None = Field(
        default=None,
        description="Optional dataset path on backend filesystem",
    )
    auto_ingest: bool = Field(default=True, description="Ingest automatically if index is empty")
    force_ingest: bool = Field(default=False, description="Force index rebuild before evaluation")
    embedding_model: str | None = Field(
        default=None,
        description="Optional embedding model for cosine metric",
    )
    default_session_id: str | None = Field(
        default=None,
        description="Fallback session id when sample has no session/conversation id",
    )
    session_prefix: str = Field(default="eval", description="Prefix for generated conversation sessions")


class EvalRunResponse(BaseModel):
    job_id: str
    status: str
    message: str


class EvalStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    processed: int = 0
    total: int = 0
    progress: float = 0.0
    current_sample_id: str | None = None
    message: str | None = None
    error: str | None = None
    summary: dict[str, float | None] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class EvalResultsResponse(BaseModel):
    job_id: str
    status: str
    summary: dict[str, float | None] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    rows: list[dict[str, Any]] = Field(default_factory=list)
