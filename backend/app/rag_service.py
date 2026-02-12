from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.chunking import chunk_pages
from app.config import Settings
from app.embeddings import EmbeddingModel
from app.langfuse_utils import build_langfuse_callback
from app.ollama_client import OllamaClient
from app.pdf_ingestion import load_corpus_pages
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

IDK_ANSWER = "I don't know based on the provided documents."


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


class QueryState(TypedDict, total=False):
    question: str
    selected: list[RetrievedChunk]
    context: str
    answer: str
    model: str


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
        self.langfuse_callback = build_langfuse_callback(settings)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Answer the question using only the context below.\n\n"
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n",
                ),
            ]
        )
        self.query_graph = self._build_query_graph()

    def _build_query_graph(self) -> Any:
        graph = StateGraph(QueryState)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("gate", self._node_gate)
        graph.add_node("generate", self._node_generate)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "gate")
        graph.add_conditional_edges(
            "gate",
            self._route_after_gate,
            {
                "generate": "generate",
                "end": END,
            },
        )
        graph.add_edge("generate", END)
        return graph.compile()

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

    def _to_documents(self, chunks: list[RetrievedChunk]) -> list[Document]:
        return [
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "score": round(chunk.score, 4),
                },
            )
            for chunk in chunks
        ]

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        documents = self._to_documents(chunks)
        lines: list[str] = []
        for idx, document in enumerate(documents, start=1):
            lines.append(
                f"[{idx}] source={document.metadata['source']} page={document.metadata['page']} "
                f"chunk={document.metadata['chunk_id']} score={document.metadata['score']:.3f}\n"
                f"{document.page_content}"
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

    def _node_retrieve(self, state: QueryState) -> dict[str, Any]:
        question = state["question"]
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

        return {"selected": selected}

    def _node_gate(self, state: QueryState) -> dict[str, Any]:
        selected = state.get("selected", [])
        if not selected:
            return {
                "answer": IDK_ANSWER,
                "model": self.settings.ollama_model,
            }

        best_score = max(chunk.score for chunk in selected)
        if best_score < self.settings.min_similarity:
            return {
                "answer": IDK_ANSWER,
                "model": self.settings.ollama_model,
            }

        context = self._format_context(selected)
        return {"context": context}

    def _route_after_gate(self, state: QueryState) -> str:
        if state.get("context"):
            return "generate"
        return "end"

    def _invoke_llm(self, model_name: str, question: str, context: str) -> str:
        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=model_name,
            temperature=self.settings.gen_temperature,
            num_predict=self.settings.gen_max_tokens,
        )
        messages = self.prompt.format_messages(question=question, context=context)

        invoke_config: dict[str, Any] = {
            "metadata": {
                "component": "rag_query",
                "question_length": len(question),
            }
        }
        if self.langfuse_callback is not None:
            invoke_config["callbacks"] = [self.langfuse_callback]

        response = llm.invoke(messages, config=invoke_config)
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content)

    def _node_generate(self, state: QueryState) -> dict[str, Any]:
        if not self.ollama_client.wait_until_ready(attempts=20, sleep_seconds=2):
            raise RuntimeError("Ollama service is not ready")

        model_name = self.ollama_client.ensure_model(
            primary_model=self.settings.ollama_model,
            fallback_model=self.settings.ollama_fallback_model,
            auto_pull=self.settings.ollama_auto_pull,
        )

        question = state["question"]
        context = state.get("context", "")
        raw_answer = self._invoke_llm(model_name=model_name, question=question, context=context)
        return {
            "answer": raw_answer,
            "model": model_name,
        }

    def query(self, question: str) -> QueryResult:
        if self.vector_store.count() == 0:
            raise RuntimeError("Vector index is empty. Call /ingest first.")

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        result_state = self.query_graph.invoke({"question": question})
        selected = result_state.get("selected", [])
        raw_answer = result_state.get("answer", IDK_ANSWER)

        if raw_answer == IDK_ANSWER:
            answer = raw_answer
        else:
            answer = self._ensure_citations(raw_answer, selected)
        model_name = result_state.get("model", self.settings.ollama_model)
        return QueryResult(answer=answer, model=model_name, sources=selected)
