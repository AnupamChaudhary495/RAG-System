"""PDF parsing with PyMuPDF (primary) and Unstructured.io (fallback).

PyMuPDF handles standard text-heavy PDFs efficiently. Unstructured.io is
used automatically when PyMuPDF output is low-confidence (sparse text or
many empty pages), and converts detected tables to Markdown.
"""

import re
from pathlib import Path
from typing import TypedDict


class PageData(TypedDict):
    """Text content and page number for a single parsed PDF page."""
    page_number: int
    text: str


# Thresholds for low-confidence PyMuPDF output
_MIN_CHARS_PER_PAGE = 100
_MAX_EMPTY_PAGE_RATIO = 0.30


def _is_low_confidence(pages: list[PageData]) -> bool:
    """Return True if PyMuPDF output is too sparse to be reliable."""
    if not pages:
        return True
    total_chars = sum(len(p["text"]) for p in pages)
    avg_chars = total_chars / len(pages)
    empty_count = sum(1 for p in pages if not p["text"].strip())
    empty_ratio = empty_count / len(pages)
    return avg_chars < _MIN_CHARS_PER_PAGE or empty_ratio > _MAX_EMPTY_PAGE_RATIO


def _parse_with_pymupdf(pdf_path: Path) -> list[PageData]:
    """Extract text page-by-page using PyMuPDF."""
    import fitz  # PyMuPDF

    pages: list[PageData] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text: str = page.get_text("text")
            pages.append({"page_number": page_num, "text": text})
    return pages


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML <table> string to a Markdown table."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    md_rows: list[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        clean = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
        md_rows.append("| " + " | ".join(clean) + " |")
        if i == 0:
            md_rows.append("|" + "|".join(["---"] * len(clean)) + "|")
    return "\n".join(md_rows)


def _element_to_text(element) -> str:
    """Convert an Unstructured element to text, rendering tables as Markdown."""
    try:
        from unstructured.documents.elements import Table

        if isinstance(element, Table):
            html = getattr(element.metadata, "text_as_html", None)
            if html:
                return _html_table_to_markdown(html)
    except ImportError:
        pass
    return element.text or ""


def _parse_with_unstructured(pdf_path: Path) -> list[PageData]:
    """Extract text using Unstructured.io with high-resolution strategy.

    Tables are converted to Markdown; all other elements are returned as plain text.
    """
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        include_page_breaks=True,
    )

    page_texts: dict[int, list[str]] = {}
    current_page = 1

    for element in elements:
        page_num: int = getattr(element.metadata, "page_number", None) or current_page
        current_page = page_num
        page_texts.setdefault(page_num, [])
        text = _element_to_text(element)
        if text.strip():
            page_texts[page_num].append(text)

    return [
        {"page_number": pnum, "text": "\n\n".join(texts)}
        for pnum, texts in sorted(page_texts.items())
    ]


def parse_pdf(pdf_path: Path) -> tuple[list[PageData], str]:
    """Parse a PDF, falling back to Unstructured.io when PyMuPDF output is sparse.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Tuple of (pages, parser_name) where parser_name is "pymupdf" or "unstructured".
    """
    pages = _parse_with_pymupdf(pdf_path)
    if _is_low_confidence(pages):
        pages = _parse_with_unstructured(pdf_path)
        return pages, "unstructured"
    return pages, "pymupdf"
