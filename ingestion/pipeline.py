"""Main entry point for the document ingestion pipeline.

Usage:
    from ingestion.pipeline import ingest_directory

    chunks = ingest_directory("./pdfs", "./output/chunks.json")

Or from the terminal:
    uv run python -m ingestion.pipeline ./pdfs ./output/chunks.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ingestion.chunker import chunk_text, count_tokens
from ingestion.metadata import build_chunk_metadata
from ingestion.parsers import PageData, parse_pdf
from ingestion.sanitizer import sanitize_pages


def _build_full_text_and_boundaries(
    pages: list[PageData],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate page texts and build character-offset page boundary map.

    Returns:
        (full_text, page_boundaries) where page_boundaries is a list of
        (start_char, end_char, page_number) tuples.
    """
    parts: list[str] = []
    boundaries: list[tuple[int, int, int]] = []
    offset = 0

    for page in pages:
        text = page["text"]
        start = offset
        end = offset + len(text)
        boundaries.append((start, end, page["page_number"]))
        parts.append(text)
        offset = end + 1  # +1 accounts for the "\n" join separator

    full_text = "\n".join(parts)
    return full_text, boundaries


def _locate_chunk(chunk: str, full_text: str, search_from: int) -> int:
    """Find the character position of a chunk in full_text, searching forward.

    Uses the first 80 characters of the chunk as a search key (long enough
    to be unique, short enough to be efficient).

    Returns the character index, or 0 as a safe fallback.
    """
    key = chunk[:80]
    pos = full_text.find(key, search_from)
    return pos if pos != -1 else search_from


def ingest_directory(pdf_dir: str, output_path: str) -> list[dict]:
    """Ingest all PDFs in a directory and write enriched chunks to a JSON file.

    For each PDF:
      1. Parse with PyMuPDF (fallback to Unstructured.io for low-confidence output)
      2. Apply the full sanitization pipeline
      3. Chunk with recursive character splitting + 50-token overlap
      4. Attach metadata (chunk_id, page, heading, timestamps, token count)

    Args:
        pdf_dir: Path to the directory containing PDF files.
        output_path: File path for the output JSON array.

    Returns:
        List of chunk dicts, each containing "text" plus all metadata fields.

    Raises:
        FileNotFoundError: If pdf_dir does not exist.
        NotADirectoryError: If pdf_dir is a file, not a directory.
        ValueError: If pdf_dir contains no PDF files.
    """
    source_dir = Path(pdf_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {pdf_dir}")

    pdf_files = sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    all_chunks: list[dict] = []
    ingest_time = datetime.now(timezone.utc)

    for pdf_path in pdf_files:
        print(f"\n[→] {pdf_path.name}")

        # 1. Parse
        pages, parser_used = parse_pdf(pdf_path)
        print(f"    parser   : {parser_used}")
        print(f"    pages    : {len(pages)}")

        # 2. Sanitize
        pages = sanitize_pages(pages)

        # 3. Build concatenated text + page boundary map
        full_text, page_boundaries = _build_full_text_and_boundaries(pages)

        # 4. Chunk
        chunks = chunk_text(full_text)
        print(f"    chunks   : {len(chunks)}")

        # 5. Locate each chunk and build metadata
        search_from = 0
        for idx, chunk in enumerate(chunks):
            chunk_start = _locate_chunk(chunk, full_text, search_from)
            token_count = count_tokens(chunk)

            metadata = build_chunk_metadata(
                chunk_index=idx,
                chunk_start_char=chunk_start,
                full_text=full_text,
                source_filename=pdf_path.name,
                page_boundaries=page_boundaries,
                ingest_time=ingest_time,
                token_count=token_count,
            )

            all_chunks.append({"text": chunk, **metadata})
            # Advance past the start of this chunk; next chunk starts later
            # (overlap means search_from stays behind chunk end intentionally)
            search_from = chunk_start + 1

    # 6. Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, indent=2, ensure_ascii=False)

    print(f"\n[✓] {len(all_chunks)} chunks written → {output_path}")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Run the pipeline from the command line.

    Usage: python -m ingestion.pipeline <pdf_dir> <output_path>
    """
    if len(sys.argv) != 3:
        print("Usage: python -m ingestion.pipeline <pdf_dir> <output_path>")
        sys.exit(1)
    ingest_directory(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    _cli()
