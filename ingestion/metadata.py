"""Metadata extraction and attachment for document chunks.

Each chunk receives a metadata dict with provenance, position, and structural
context (nearest section heading) derived from the surrounding document text.
"""

import uuid
import re
from datetime import datetime
from typing import Optional


# Maximum word count for a line to be treated as a section heading
_HEADING_MAX_WORDS = 8


def _detect_section_heading(full_text: str, chunk_start_char: int) -> Optional[str]:
    """Scan backwards from chunk_start_char to find the nearest section heading.

    A line is treated as a heading if it meets any of:
      - ALL CAPS with ≤ _HEADING_MAX_WORDS words
      - Title Case with ≤ _HEADING_MAX_WORDS words
      - Ends with a colon ":"

    Args:
        full_text: The complete document text.
        chunk_start_char: Character offset of the chunk's first character.

    Returns:
        The heading string, or None if no heading is found.
    """
    preceding = full_text[:chunk_start_char]
    lines = preceding.splitlines()

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            continue
        words = stripped.split()
        if stripped.isupper() and len(words) <= _HEADING_MAX_WORDS:
            return stripped
        if stripped.istitle() and len(words) <= _HEADING_MAX_WORDS:
            return stripped
        if stripped.endswith(":"):
            return stripped

    return None


def _find_page_number(
    chunk_start_char: int,
    page_boundaries: list[tuple[int, int, int]],
) -> Optional[int]:
    """Return the page number for a chunk's starting character position.

    Args:
        chunk_start_char: Character offset of the chunk in the full document text.
        page_boundaries: List of (start_char, end_char, page_number) tuples.

    Returns:
        The page number, or None if the position falls between boundaries.
    """
    for start, end, page_num in page_boundaries:
        if start <= chunk_start_char < end:
            return page_num
    return None


def build_chunk_metadata(
    chunk_index: int,
    chunk_start_char: int,
    full_text: str,
    source_filename: str,
    page_boundaries: list[tuple[int, int, int]],
    ingest_time: datetime,
    token_count: int,
) -> dict:
    """Build the metadata dictionary for a single chunk.

    Args:
        chunk_index: Zero-based position of this chunk within the document.
        chunk_start_char: Character offset of the chunk in the full document text.
        full_text: Complete document text (used for heading detection).
        source_filename: Original PDF filename (basename only).
        page_boundaries: List of (start_char, end_char, page_number) tuples.
        ingest_time: UTC datetime when the ingestion run started.
        token_count: Number of cl100k_base tokens in this chunk.

    Returns:
        Metadata dict with all required fields.
    """
    return {
        "chunk_id": str(uuid.uuid4()),
        "source_filename": source_filename,
        "page_number": _find_page_number(chunk_start_char, page_boundaries),
        "timestamp": ingest_time.isoformat(),
        "section_heading": _detect_section_heading(full_text, chunk_start_char),
        "chunk_index": chunk_index,
        "token_count": token_count,
    }
