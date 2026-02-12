from __future__ import annotations

from dataclasses import dataclass

from app.pdf_ingestion import PageDocument


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source: str
    page: int
    chunk_index: int
    text: str


def _split_text_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(text_length, start + chunk_size)

        # Prefer ending on a natural boundary when possible.
        if end < text_length:
            window = text[start:end]
            boundary_candidates = [window.rfind("\n\n"), window.rfind(". "), window.rfind(" ")]
            boundary = max(boundary_candidates)
            if boundary > chunk_size // 2:
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


def chunk_pages(pages: list[PageDocument], chunk_size: int, overlap: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    per_source_counter: dict[str, int] = {}

    for page_doc in pages:
        source_counter = per_source_counter.setdefault(page_doc.source, 0)
        page_chunks = _split_text_with_overlap(page_doc.text, chunk_size=chunk_size, overlap=overlap)

        for chunk_text in page_chunks:
            chunk_index = source_counter
            chunk_id = f"{page_doc.source}-p{page_doc.page}-c{chunk_index}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source=page_doc.source,
                    page=page_doc.page,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )
            source_counter += 1

        per_source_counter[page_doc.source] = source_counter

    return chunks
