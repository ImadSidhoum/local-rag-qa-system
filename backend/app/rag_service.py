from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from app.chunking import split_documents
from app.config import Settings
from app.embeddings import load_hf_embeddings, resolve_hf_token
from app.langfuse_utils import build_langfuse_callback
from app.ollama_client import OllamaClient
from app.pdf_ingestion import load_corpus_documents

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a strict Retrieval-Augmented QA assistant.
Rules:
1) Use only the provided context.
2) Use conversation history only to resolve references (e.g., "it", "that"), not as a factual source.
3) If the answer is not in context, respond exactly: I don't know based on the provided documents.
4) Cite factual statements using this exact format: [source=<filename> page=<page> chunk=<chunk_id>]
5) Never invent sources or page numbers.
6) Keep the answer concise and technical.
""".strip()

IDK_ANSWER = "I don't know based on the provided documents."
EMPTY_HISTORY = "(none)"
REWRITE_SYSTEM_PROMPT = """
You rewrite follow-up questions into standalone questions for vector retrieval.
Rules:
1) Keep intent unchanged.
2) Resolve pronouns/references using conversation history.
3) Keep it concise and specific.
4) If question is already standalone, return it unchanged.
5) Output only the rewritten question text.
""".strip()


@dataclass
class RetrievedChunk:
    source: str
    page: int
    chunk_id: str
    chunk_index: int
    text: str
    score: float


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
    session_id: str | None = None
    rewritten_question: str | None = None


class QueryState(TypedDict, total=False):
    question: str
    session_id: str
    history: str
    rewritten_question: str
    selected: list[RetrievedChunk]
    context: str
    answer: str
    model: str


class RagService:
    def __init__(
        self,
        settings: Settings,
        ollama_client: OllamaClient,
    ) -> None:
        self.settings = settings
        self.ollama_client = ollama_client
        self.langfuse_callback = build_langfuse_callback(settings)
        self._memory_lock = Lock()
        self._conversation_memory: dict[str, deque[tuple[str, str]]] = {}

        self.embedding_model, self.embedding_model_name = load_hf_embeddings(
            model_name=self.settings.embedding_model,
            batch_size=self.settings.batch_size,
            hf_token=resolve_hf_token(self.settings.hf_token or self.settings.huggingface_hub_token),
        )
        self.vectorstore = self._build_vectorstore()

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Conversation history (may be empty):\n{history}\n\n"
                    "Answer the question using only the context below.\n\n"
                    "Question:\n{question}\n\n"
                    "Context:\n{context}\n",
                ),
            ]
        )
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REWRITE_SYSTEM_PROMPT),
                (
                    "human",
                    "Conversation history:\n{history}\n\nCurrent question:\n{question}\n\n"
                    "Return rewritten standalone question:",
                ),
            ]
        )

        self.query_graph = self._build_query_graph()

    def _build_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=self.settings.chroma_collection_name,
            persist_directory=str(self.settings.chroma_dir),
            embedding_function=self.embedding_model,
            client_settings=ChromaSettings(
                anonymized_telemetry=self.settings.chroma_anonymized_telemetry,
                chroma_product_telemetry_impl=self.settings.chroma_product_telemetry_impl,
                chroma_telemetry_impl=self.settings.chroma_telemetry_impl,
            ),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _reset_vectorstore(self) -> None:
        try:
            self.vectorstore.delete_collection()
        except Exception:
            pass
        self.vectorstore = self._build_vectorstore()

    def indexed_chunk_count(self) -> int:
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return 0
        try:
            return int(collection.count())
        except Exception:
            return 0

    def _build_query_graph(self) -> Any:
        graph = StateGraph(QueryState)
        graph.add_node("rewrite", self._node_rewrite)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("gate", self._node_gate)
        graph.add_node("generate", self._node_generate)

        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "retrieve")
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
        payload = {
            "files": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "mtime": int(path.stat().st_mtime),
                }
                for path in pdf_files
            ],
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "collection_name": self.settings.chroma_collection_name,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _read_manifest(self) -> dict[str, str] | None:
        if not self.settings.index_manifest_path.exists():
            return None
        return json.loads(self.settings.index_manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, payload: dict[str, str | int]) -> None:
        self.settings.index_manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _list_pdf_files(self) -> list[Path]:
        pdf_files = sorted(path for path in self.settings.corpus_dir.glob("*.pdf") if path.is_file())
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in corpus directory: {self.settings.corpus_dir}")
        return pdf_files

    def _load_documents(self) -> tuple[list[Path], list[Document]]:
        return load_corpus_documents(self.settings.corpus_dir)

    def _index_documents(self, chunks: list[Document]) -> None:
        self._reset_vectorstore()

        total = len(chunks)
        batch_size = max(1, self.settings.batch_size)
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            ids = [str(doc.metadata["chunk_id"]) for doc in batch]
            self.vectorstore.add_documents(batch, ids=ids)
            logger.info("Indexed %s/%s chunks", min(start + batch_size, total), total)

    def ingest(self, force: bool = False) -> IngestSummary:
        pdf_files = self._list_pdf_files()
        fingerprint = self._manifest_fingerprint(pdf_files)
        manifest = self._read_manifest()
        existing_count = self.indexed_chunk_count()

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
                pages=int(manifest.get("pages", 0)),
                chunks=existing_count,
                skipped=True,
            )

        _, page_documents = self._load_documents()
        chunks = split_documents(
            page_documents,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No chunks generated from corpus")
        self._index_documents(chunks)

        self._write_manifest(
            {
                "fingerprint": fingerprint,
                "documents": len(pdf_files),
                "pages": len(page_documents),
                "chunks": len(chunks),
                "embedding_model": self.embedding_model_name,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
            }
        )

        return IngestSummary(
            status="ok",
            message="Index built successfully",
            documents=len(pdf_files),
            pages=len(page_documents),
            chunks=len(chunks),
            skipped=False,
        )

    def _build_retriever(self):
        if self.settings.use_mmr:
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.settings.top_k,
                    "fetch_k": max(self.settings.mmr_candidates, self.settings.top_k),
                    "lambda_mult": self.settings.mmr_lambda,
                },
            )
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.settings.top_k},
        )

    def _score_documents(self, query: str) -> dict[str, float]:
        score_limit = max(self.settings.top_k, self.settings.mmr_candidates)
        scored_pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=score_limit)

        score_map: dict[str, float] = {}
        for document, score in scored_pairs:
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            clamped_score = max(0.0, min(1.0, float(score)))
            previous = score_map.get(chunk_id)
            score_map[chunk_id] = clamped_score if previous is None else max(previous, clamped_score)

        return score_map

    def _documents_to_chunks(self, documents: list[Document], score_map: dict[str, float]) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for doc in documents:
            source = str(doc.metadata.get("source", "unknown.pdf"))
            page = int(doc.metadata.get("page", 1))
            chunk_id = str(doc.metadata.get("chunk_id", f"{source}-p{page}-c0"))
            chunk_index = int(doc.metadata.get("chunk_index", 0))
            score = score_map.get(chunk_id, 0.0)
            chunks.append(
                RetrievedChunk(
                    source=source,
                    page=page,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    text=doc.page_content,
                    score=score,
                )
            )
        return chunks

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

    def _normalize_session_id(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        trimmed = session_id.strip()
        if not trimmed:
            return None
        return trimmed[:128]

    def _get_history_text(self, session_id: str | None) -> str:
        if not self.settings.memory_enabled or not session_id:
            return EMPTY_HISTORY

        with self._memory_lock:
            history = list(self._conversation_memory.get(session_id, deque()))

        if not history:
            return EMPTY_HISTORY

        window = max(1, self.settings.memory_max_turns)
        lines: list[str] = []
        for index, (question, answer) in enumerate(history[-window:], start=1):
            lines.append(f"Turn {index} user: {question}\nTurn {index} assistant: {answer}")
        return "\n\n".join(lines)

    def _append_memory(self, session_id: str | None, question: str, answer: str) -> None:
        if not self.settings.memory_enabled or not session_id:
            return

        max_turns = max(1, self.settings.memory_max_turns)
        with self._memory_lock:
            session_memory = self._conversation_memory.get(session_id)
            if session_memory is None:
                session_memory = deque(maxlen=max_turns)
                self._conversation_memory[session_id] = session_memory
            session_memory.append((question, answer))

    def _sanitize_rewritten_question(self, candidate: str, fallback: str) -> str:
        cleaned = candidate.strip().strip("\"'`")
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return fallback
        if len(cleaned) > 500:
            return cleaned[:500].rstrip()
        return cleaned

    def _rewrite_question_with_llm(self, model_name: str, history: str, question: str, session_id: str | None) -> str:
        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=model_name,
            temperature=0.0,
            num_predict=max(16, self.settings.query_rewrite_max_tokens),
        )
        messages = self.rewrite_prompt.format_messages(history=history, question=question)

        metadata: dict[str, Any] = {
            "component": "query_rewrite",
            "history_used": history != EMPTY_HISTORY,
        }
        if session_id:
            metadata["session_id"] = session_id

        invoke_config: dict[str, Any] = {"metadata": metadata}
        if self.langfuse_callback is not None:
            invoke_config["callbacks"] = [self.langfuse_callback]

        response = llm.invoke(messages, config=invoke_config)
        content = getattr(response, "content", "")
        rewritten = content if isinstance(content, str) else str(content)
        return self._sanitize_rewritten_question(rewritten, fallback=question)

    def _node_rewrite(self, state: QueryState) -> dict[str, Any]:
        question = state["question"]
        history = state.get("history", EMPTY_HISTORY)
        session_id = state.get("session_id")

        if not self.settings.query_rewrite_enabled or history == EMPTY_HISTORY:
            return {"rewritten_question": question}

        if not self.ollama_client.wait_until_ready(attempts=5, sleep_seconds=1):
            logger.warning("Skipping rewrite because Ollama is not ready")
            return {"rewritten_question": question}

        try:
            model_name = self.ollama_client.ensure_model(
                primary_model=self.settings.ollama_model,
                auto_pull=self.settings.ollama_auto_pull,
            )
            rewritten = self._rewrite_question_with_llm(model_name, history, question, session_id)
            return {"rewritten_question": rewritten, "model": model_name}
        except Exception as exc:  # pragma: no cover
            logger.warning("Query rewrite failed, using original question: %s", exc)
            return {"rewritten_question": question}

    def _node_retrieve(self, state: QueryState) -> dict[str, Any]:
        retrieval_query = state.get("rewritten_question", state["question"])
        retriever = self._build_retriever()
        documents = retriever.invoke(retrieval_query)
        score_map = self._score_documents(retrieval_query)
        selected = self._documents_to_chunks(documents, score_map)
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

    def _invoke_llm(
        self,
        model_name: str,
        question: str,
        context: str,
        history: str,
        session_id: str | None,
        rewritten_question: str,
    ) -> str:
        llm = ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=model_name,
            temperature=self.settings.gen_temperature,
            num_predict=self.settings.gen_max_tokens,
        )
        messages = self.answer_prompt.format_messages(question=question, context=context, history=history)

        metadata: dict[str, Any] = {
            "component": "rag_query",
            "question_length": len(question),
            "has_history": history != EMPTY_HISTORY,
            "rewritten_question": rewritten_question,
        }
        if session_id:
            metadata["session_id"] = session_id

        invoke_config: dict[str, Any] = {"metadata": metadata}
        if self.langfuse_callback is not None:
            invoke_config["callbacks"] = [self.langfuse_callback]

        response = llm.invoke(messages, config=invoke_config)
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content)

    def _node_generate(self, state: QueryState) -> dict[str, Any]:
        if not self.ollama_client.wait_until_ready(attempts=20, sleep_seconds=2):
            raise RuntimeError("Ollama service is not ready")

        model_name = state.get("model")
        if not model_name:
            model_name = self.ollama_client.ensure_model(
                primary_model=self.settings.ollama_model,
                auto_pull=self.settings.ollama_auto_pull,
            )

        question = state["question"]
        context = state.get("context", "")
        history = state.get("history", EMPTY_HISTORY)
        session_id = state.get("session_id")
        rewritten_question = state.get("rewritten_question", question)

        raw_answer = self._invoke_llm(
            model_name=model_name,
            question=question,
            context=context,
            history=history,
            session_id=session_id,
            rewritten_question=rewritten_question,
        )
        return {
            "answer": raw_answer,
            "model": model_name,
        }

    def query(self, question: str, session_id: str | None = None) -> QueryResult:
        if self.indexed_chunk_count() == 0:
            raise RuntimeError("Vector index is empty. Call /ingest first.")

        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        normalized_session_id = self._normalize_session_id(session_id)
        history = self._get_history_text(normalized_session_id)

        result_state = self.query_graph.invoke(
            {
                "question": question,
                "session_id": normalized_session_id,
                "history": history,
            }
        )

        selected = result_state.get("selected", [])
        raw_answer = result_state.get("answer", IDK_ANSWER)

        if raw_answer == IDK_ANSWER:
            answer = raw_answer
        else:
            answer = self._ensure_citations(raw_answer, selected)

        self._append_memory(normalized_session_id, question, answer)

        rewritten_question = result_state.get("rewritten_question", question)
        model_name = result_state.get("model", self.settings.ollama_model)
        return QueryResult(
            answer=answer,
            model=model_name,
            sources=selected,
            session_id=normalized_session_id,
            rewritten_question=rewritten_question,
        )
