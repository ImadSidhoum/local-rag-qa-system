from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import TextChunk


@dataclass
class RetrievedChunk:
    source: str
    page: int
    chunk_id: str
    chunk_index: int
    text: str
    score: float
    embedding: list[float] | None = None


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "rag_chunks",
        anonymized_telemetry: bool = False,
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=anonymized_telemetry),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        include_embeddings: bool,
    ) -> list[RetrievedChunk]:
        include_fields = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include_fields.append("embeddings")

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=include_fields,
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        embeddings = (result.get("embeddings") or [[]])[0] if include_embeddings else []

        retrieved: list[RetrievedChunk] = []
        for index, doc in enumerate(documents):
            metadata = metadatas[index]
            distance = float(distances[index])
            score = max(0.0, 1.0 - distance)
            embedding = embeddings[index] if include_embeddings else None

            retrieved.append(
                RetrievedChunk(
                    source=str(metadata["source"]),
                    page=int(metadata["page"]),
                    chunk_id=str(metadata["chunk_id"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=doc,
                    score=score,
                    embedding=embedding,
                )
            )

        return retrieved
