from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int,
    overlap: int,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    per_source_counter: dict[str, int] = {}

    for chunk in chunks:
        raw_source = str(chunk.metadata.get("source", "unknown.pdf"))
        source = Path(raw_source).name
        page = int(chunk.metadata.get("page", 1))

        chunk_index = per_source_counter.get(source, 0)
        chunk_id = f"{source}-p{page}-c{chunk_index}"
        chunk.metadata["source"] = source
        chunk.metadata["page"] = page
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = chunk_id
        per_source_counter[source] = chunk_index + 1

    return chunks
