from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def list_pdf_files(corpus_dir: Path) -> list[Path]:
    return sorted(path for path in corpus_dir.glob("*.pdf") if path.is_file())


def load_corpus_documents(corpus_dir: Path) -> tuple[list[Path], list[Document]]:
    pdf_files = list_pdf_files(corpus_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in corpus directory: {corpus_dir}")

    documents: list[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        loaded_pages = loader.load()

        for page_doc in loaded_pages:
            text = page_doc.page_content.strip()
            if not text:
                continue

            page_index = int(page_doc.metadata.get("page", 0))
            page_doc.metadata["source"] = pdf_path.name
            page_doc.metadata["page"] = page_index + 1
            documents.append(page_doc)

        logger.info("Parsed %s pages with text from %s", len(loaded_pages), pdf_path.name)

    if not documents:
        raise ValueError("PDF files were found, but no extractable text was detected")

    return pdf_files, documents
