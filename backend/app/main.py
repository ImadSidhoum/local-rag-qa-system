from __future__ import annotations

import logging
import random

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.embeddings import EmbeddingModel
from app.logging_utils import configure_logging
from app.ollama_client import OllamaClient
from app.rag_service import RagService
from app.schemas import ConfigResponse, HealthResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceItem
from app.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_service() -> RagService:
    service = getattr(app.state, "rag_service", None)
    if service is None:
        raise RuntimeError("RAG service is not initialized")
    return service


@app.on_event("startup")
def startup() -> None:
    configure_logging(settings.log_level)
    random.seed(settings.random_seed)
    np.random.seed(settings.random_seed)

    embedding_model = EmbeddingModel(settings.embedding_model, settings.batch_size)
    vector_store = ChromaVectorStore(
        settings.chroma_dir,
        anonymized_telemetry=settings.chroma_anonymized_telemetry,
    )
    ollama_client = OllamaClient(settings.ollama_base_url, settings.ollama_timeout_seconds)
    app.state.rag_service = RagService(settings, embedding_model, vector_store, ollama_client)

    logger.info("Application started with data dir=%s", settings.data_dir)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = get_rag_service()
    ollama_ready = service.ollama_client.is_ready()
    return HealthResponse(status="ok", indexed_chunks=service.vector_store.count(), ollama_ready=ollama_ready)


@app.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    exposed = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "use_mmr": settings.use_mmr,
        "ollama_model": settings.ollama_model,
        "ollama_fallback_model": settings.ollama_fallback_model,
        "temperature": settings.gen_temperature,
        "max_tokens": settings.gen_max_tokens,
        "memory_enabled": settings.memory_enabled,
        "memory_max_turns": settings.memory_max_turns,
        "langfuse_enabled": settings.langfuse_enabled,
        "langfuse_host": settings.langfuse_host,
    }
    return ConfigResponse(config=exposed)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    service = get_rag_service()
    try:
        summary = service.ingest(force=payload.force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        status=summary.status,
        message=summary.message,
        documents=summary.documents,
        pages=summary.pages,
        chunks=summary.chunks,
        skipped=summary.skipped,
    )


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    service = get_rag_service()
    try:
        result = service.query(payload.question, payload.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    sources = [
        SourceItem(
            source=chunk.source,
            page=chunk.page,
            chunk_id=chunk.chunk_id,
            score=round(chunk.score, 4),
            text_excerpt=(chunk.text[:280] + "...") if len(chunk.text) > 280 else chunk.text,
        )
        for chunk in result.sources
    ]

    return QueryResponse(answer=result.answer, model=result.model, sources=sources, session_id=result.session_id)
