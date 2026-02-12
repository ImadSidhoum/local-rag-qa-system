from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunking import TextChunk, chunk_pages
from app.config import Settings
from app.embeddings import EmbeddingModel
from app.ollama_client import OllamaClient
from app.pdf_ingestion import PageDocument, load_corpus_pages
from app.vector_store import ChromaVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a strict Retrieval-Augmented QA assistant.
Rules:
1) Use only the provided context.
2) If the answer is not in context, respond exactly: I don't know based on the provided documents.
3) Cite factual statements using this exact format: [source=<filename> page=<page> chunk=<chunk_id>]
4) Never invent sources or page numbers.
5) Keep the answer concise and technical.
""".strip()


@dataclass
class IngestSummary:
    status: str
    message: str
    documents: int
    pages: int
    chunks: int
    skipped: bool


@dataclass
class QueryResult:
    answer: str
    model: str
    sources: list[RetrievedChunk]


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        ollama_client: OllamaClient,
    ) -> None:
        self.settings = settings
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.ollama_client = ollama_client

    def _manifest_fingerprint(self, pdf_files: list[Path]) -> str:
        fingerprint_payload = {
            "files": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime": int(path.stat().st_mtime),
                }
                for path in pdf_files
            ],
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "embedding_model": self.settings.embedding_model,
        }
        raw = json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _read_manifest(self) -> dict[str, str] | None:
        if not self.settings.index_manifest_path.exists():
            return None
        return json.loads(self.settings.index_manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, payload: dict[str, str | int]) -> None:
        self.settings.index_manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def ingest(self, force: bool = False) -> IngestSummary:
        pdf_files, pages = load_corpus_pages(self.settings.corpus_dir)

        fingerprint = self._manifest_fingerprint(pdf_files)
        manifest = self._read_manifest()
        existing_count = self.vector_store.count()

        if (
            not force
            and manifest is not None
            and manifest.get("fingerprint") == fingerprint
            and existing_count > 0
        ):
            return IngestSummary(
                status="ok",
                message="Index already up to date",
                documents=len(pdf_files),
                pages=len(pages),
                chunks=existing_count,
                skipped=True,
            )

        chunks = chunk_pages(pages, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap)
        if not chunks:
            raise ValueError("No chunks generated from corpus")

        logger.info("Embedding %s chunks with model %s", len(chunks), self.settings.embedding_model)
        embeddings = self.embedding_model.embed_texts([chunk.text for chunk in chunks])

        self.vector_store.reset_collection()
        self.vector_store.upsert_chunks(chunks, embeddings)

        self._write_manifest(
            {
                "fingerprint": fingerprint,
                "documents": len(pdf_files),
                "pages": len(pages),
                "chunks": len(chunks),
                "embedding_model": self.settings.embedding_model,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            }
        )

        return IngestSummary(
            status="ok",
            message="Index built successfully",
            documents=len(pdf_files),
            pages=len(pages),
            chunks=len(chunks),
            skipped=False,
        )

    def _apply_mmr(self, query_embedding: list[float], candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        top_k = min(self.settings.top_k, len(candidates))
        if top_k <= 0:
            return []

        matrix = np.array([candidate.embedding for candidate in candidates], dtype=np.float32)
        query = np.array(query_embedding, dtype=np.float32)

        query_sim = matrix @ query
        selected_indexes: list[int] = []

        for _ in range(top_k):
            if not selected_indexes:
                selected_indexes.append(int(np.argmax(query_sim)))
                continue

            selected_matrix = matrix[selected_indexes]
            diversity_penalty = np.max(matrix @ selected_matrix.T, axis=1)
            mmr_scores = self.settings.mmr_lambda * query_sim - (1.0 - self.settings.mmr_lambda) * diversity_penalty
            mmr_scores[selected_indexes] = -np.inf
            selected_indexes.append(int(np.argmax(mmr_scores)))

        selected = [candidates[index] for index in selected_indexes]
        for item in selected:
            item.embedding = None
        return selected

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        lines: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            lines.append(
                f"[{idx}] source={chunk.source} page={chunk.page} chunk={chunk.chunk_id} score={chunk.score:.3f}\n"
                f"{chunk.text}"
            )
        return "\n\n".join(lines)

    def _ensure_citations(self, answer: str, chunks: list[RetrievedChunk]) -> str:
        has_citation = bool(re.search(r"\[source=.*?page=.*?chunk=.*?\]", answer))
        if has_citation:
            return answer

        if not chunks:
            return answer

        appended = "; ".join(
            f"[source={chunk.source} page={chunk.page} chunk={chunk.chunk_id}]" for chunk in chunks[:2]
        )
        return f"{answer}\n\nCitations: {appended}"

    def query(self, question: str) -> QueryResult:
        if self.vector_store.count() == 0:
            raise RuntimeError("Vector index is empty. Call /ingest first.")

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        query_embedding = self.embedding_model.embed_query(question)

        candidate_count = self.settings.top_k
        include_embeddings = False
        if self.settings.use_mmr:
            candidate_count = max(self.settings.mmr_candidates, self.settings.top_k)
            include_embeddings = True

        candidates = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=candidate_count,
            include_embeddings=include_embeddings,
        )

        if self.settings.use_mmr:
            selected = self._apply_mmr(query_embedding, candidates)
        else:
            selected = candidates[: self.settings.top_k]

        if not selected:
            return QueryResult(
                answer="I don't know based on the provided documents.",
                model=self.settings.ollama_model,
                sources=[],
            )

        best_score = max(chunk.score for chunk in selected)
        if best_score < self.settings.min_similarity:
            return QueryResult(
                answer="I don't know based on the provided documents.",
                model=self.settings.ollama_model,
                sources=selected,
            )

        context = self._format_context(selected)
        user_prompt = (
            "Answer the question using only the context below.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n"
        )

        if not self.ollama_client.wait_until_ready(attempts=20, sleep_seconds=2):
            raise RuntimeError("Ollama service is not ready")

        model_name = self.ollama_client.ensure_model(
            primary_model=self.settings.ollama_model,
            fallback_model=self.settings.ollama_fallback_model,
            auto_pull=self.settings.ollama_auto_pull,
        )

        raw_answer = self.ollama_client.chat(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.settings.gen_temperature,
            max_tokens=self.settings.gen_max_tokens,
        )

        answer = self._ensure_citations(raw_answer, selected)
        return QueryResult(answer=answer, model=model_name, sources=selected)
