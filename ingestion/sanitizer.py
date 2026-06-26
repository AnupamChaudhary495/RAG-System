"""Text sanitization pipeline for extracted PDF content.

Transformations applied in order:
  1. Strip boilerplate (headers/footers appearing on >40% of pages)
  2. Remove page number artifacts
  3. Repair hyphenated line breaks
  4. Strip non-printable control characters (preserving \\n and \\t)
  5. Normalize consecutive whitespace
"""

import re
from collections import Counter

from ingestion.parsers import PageData


def _detect_boilerplate(pages: list[PageData]) -> set[str]:
    """Identify lines that appear on more than 40% of pages (headers/footers)."""
    if not pages:
        return set()

    threshold = 0.4 * len(pages)
    line_counts: Counter[str] = Counter()

    for page in pages:
        # Count each unique non-empty line once per page
        unique_lines = {line.strip() for line in page["text"].splitlines() if line.strip()}
        line_counts.update(unique_lines)

    return {line for line, count in line_counts.items() if count > threshold}


def _strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    """Remove lines identified as boilerplate from text."""
    if not boilerplate:
        return text
    filtered = [line for line in text.splitlines() if line.strip() not in boilerplate]
    return "\n".join(filtered)


def _remove_page_numbers(text: str) -> str:
    """Remove common page number artifact patterns."""
    # "Page 12 of 50" / "page 12 of 50"
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
    # "- 12 -" style markers
    text = re.sub(r"-\s*\d+\s*-", "", text)
    # Standalone digit(s) on their own line (possibly surrounded by whitespace)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text


def _repair_hyphenated_breaks(text: str) -> str:
    """Join words broken across lines with a hyphen: 'com-\\nuter' → 'computer'."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _strip_control_chars(text: str) -> str:
    """Remove non-printable control characters, preserving \\n (0x0A) and \\t (0x09)."""
    # Keep: tab (09), newline (0A), printable ASCII (20-7E), extended Latin (80-FF)
    return re.sub(r"[^\x09\x0A\x20-\x7E\x80-\xFF]", "", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and excess blank lines."""
    # 2+ horizontal spaces → single space (tabs and newlines preserved)
    text = re.sub(r"[^\S\n\t]{2,}", " ", text)
    # 3+ consecutive newlines → 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def sanitize_pages(pages: list[PageData]) -> list[PageData]:
    """Apply the full sanitization pipeline to a list of parsed pages.

    Args:
        pages: List of PageData dicts from the PDF parser.

    Returns:
        Sanitized list of PageData dicts with the same structure.
    """
    boilerplate = _detect_boilerplate(pages)

    sanitized: list[PageData] = []
    for page in pages:
        text = page["text"]
        text = _strip_boilerplate(text, boilerplate)
        text = _remove_page_numbers(text)
        text = _repair_hyphenated_breaks(text)
        text = _strip_control_chars(text)
        text = _normalize_whitespace(text)
        sanitized.append({"page_number": page["page_number"], "text": text})

    return sanitized
