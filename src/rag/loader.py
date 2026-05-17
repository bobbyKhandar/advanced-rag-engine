import logging
from typing import Optional

import fitz
import pymupdf4llm
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def get_page_count(file_path: str) -> int:
    with fitz.open(file_path) as doc:
        return doc.page_count


def load_pdf_to_markdown(
    file_path: str,
    skip_start: int = 0,
    skip_end: int = 0,
) -> str:
    total = get_page_count(file_path)
    end = total - skip_end
    if skip_start >= end:
        return ""
    pages = list(range(skip_start, end))
    return pymupdf4llm.to_markdown(file_path, pages=pages)


def load_and_chunk(
    file_path: str,
    chunker,
    skip_start: int = 0,
    skip_end: int = 0,
) -> tuple[list[Document], list[Document]]:
    md = load_pdf_to_markdown(file_path, skip_start, skip_end)
    return chunker.split_text(md)
