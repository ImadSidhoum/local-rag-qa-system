from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings import EmbeddingModel


@dataclass
class RetrievedChunk:
    source: str
    page: int
    chunk_id: str
    chunk_index: int
    text: str
    score: float | None = None


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        embedding_model: EmbeddingModel,
        collection_name: str = "rag_chunks",
        anonymized_telemetry: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model.model
        self.client_settings = ChromaSettings(anonymized_telemetry=anonymized_telemetry)
        self.store = Chroma(
            collection_name=collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding_model,
            client_settings=self.client_settings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.store.delete_collection()
        except Exception:
            pass
        self.store = self.store.__class__(
            collection_name=self.collection_name,
            persist_directory=str(self.persist_dir),
            embedding_function=self.embedding_model,
            client_settings=self.client_settings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        collection = getattr(self.store, "_collection", None)
        if collection is None:
            return 0
        return int(collection.count())

    def upsert_chunks(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        self.store.add_documents(chunks, ids=ids)

    def query(self, question: str, n_results: int, use_mmr: bool = False) -> list[RetrievedChunk]:
        if use_mmr:
            retriever = self.store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": n_results, "fetch_k": max(2 * n_results, n_results)},
            )
            documents = retriever.invoke(question)
            score_map: dict[str, float] = {}
        else:
            scored = self.store.similarity_search_with_relevance_scores(question, k=n_results)
            documents = [doc for doc, _ in scored]
            score_map = {str(doc.metadata.get("chunk_id", "")): float(score) for doc, score in scored}

        retrieved: list[RetrievedChunk] = []
        for doc in documents:
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            retrieved.append(
                RetrievedChunk(
                    source=str(doc.metadata.get("source", "unknown.pdf")),
                    page=int(doc.metadata.get("page", 1)),
                    chunk_id=chunk_id,
                    chunk_index=int(doc.metadata.get("chunk_index", 0)),
                    text=doc.page_content,
                    score=score_map.get(chunk_id),
                )
            )
        return retrieved
