from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageDocument:
    source: str
    page: int
    text: str


def _normalize_text(text: str) -> str:
    collapsed = re.sub(r"[\t\r\f\v]+", " ", text)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    collapsed = re.sub(r"[ ]{2,}", " ", collapsed)
    return collapsed.strip()


def list_pdf_files(corpus_dir: Path) -> list[Path]:
    return sorted(path for path in corpus_dir.glob("*.pdf") if path.is_file())


def extract_pages_from_pdf(pdf_path: Path) -> list[PageDocument]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageDocument] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _normalize_text(raw_text)
        if not text:
            continue

        pages.append(
            PageDocument(
                source=pdf_path.name,
                page=page_number,
                text=text,
            )
        )

    logger.info("Parsed %s pages with text from %s", len(pages), pdf_path.name)
    return pages


def load_corpus_pages(corpus_dir: Path) -> tuple[list[Path], list[PageDocument]]:
    pdf_files = list_pdf_files(corpus_dir)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in corpus directory: {corpus_dir}")

    all_pages: list[PageDocument] = []
    for pdf_path in pdf_files:
        all_pages.extend(extract_pages_from_pdf(pdf_path))

    if not all_pages:
        raise ValueError("PDF files were found, but no extractable text was detected")

    return pdf_files, all_pages
