"""Document upload + ingestion for the API.

Turns user-uploaded files (PDF / Markdown / plain text) into embedded chunks in
the Qdrant collection so they become immediately queryable — no manual copying
into a directory required. Also lists and deletes ingested documents.

The BGE-M3 embedder is lazily loaded once and reused across uploads.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.http import models as qmodels

from embedding.embedder import BGE3Embedder
from embedding.vector_store import (
    COLLECTION_NAME,
    build_client,
    build_points,
    ensure_collection,
    upsert_points,
)
from ingestion.chunker import chunk_text
from ingestion.parsers import parse_pdf

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}
_BATCH = 32

_embedder: BGE3Embedder | None = None


def _get_embedder() -> BGE3Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BGE3Embedder()
    return _embedder


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunk_id(filename: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}-{index}"))


def _mk_chunk(
    filename: str, heading: str, page: int | None, index: int, text: str
) -> dict:
    return {
        "chunk_id": _chunk_id(filename, index),
        "source_filename": filename,
        "page_number": page,
        "section_heading": heading,
        "chunk_index": index,
        "token_count": len(text.split()),
        "text": text,
        "timestamp": _now(),
    }


def _file_to_chunks(filename: str, data: bytes) -> list[dict]:
    """Parse a single file into embeddable chunk dicts."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    chunks: list[dict] = []
    idx = 0

    if ext == ".pdf":
        # parse_pdf needs a real file path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(data)
            tmp = Path(tf.name)
        try:
            pages, _parser = parse_pdf(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        for page in pages:
            for piece in chunk_text(page["text"]):
                chunks.append(
                    _mk_chunk(filename, stem, page["page_number"], idx, piece)
                )
                idx += 1
    else:
        text = data.decode("utf-8", errors="replace")
        # Split on Markdown headings so each chunk keeps a meaningful heading.
        sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
        sections = [s for s in sections if s.strip()] or [text]
        for section in sections:
            m = re.match(r"^(#{1,3}) (.+)", section)
            heading = m.group(2).strip() if m else stem
            for piece in chunk_text(section):
                chunks.append(_mk_chunk(filename, heading, None, idx, piece))
                idx += 1

    return chunks


def _delete_by_filename(client, filename: str) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_filename",
                        match=qmodels.MatchValue(value=filename),
                    )
                ]
            )
        ),
        wait=True,
    )


def ingest_uploaded_files(
    files: list[tuple[str, bytes]], qdrant_url: str
) -> dict:
    """Parse, embed and upsert uploaded files. Re-uploading replaces a file.

    Args:
        files: List of (filename, raw_bytes) tuples.
        qdrant_url: Qdrant connection URL.

    Returns:
        {"added": [{"filename", "chunks"}], "errors": [{"filename", "error"}]}
    """
    client = build_client(qdrant_url)
    ensure_collection(client)
    embedder = _get_embedder()

    added: list[dict] = []
    errors: list[dict] = []

    for filename, data in files:
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            errors.append(
                {"filename": filename, "error": "Unsupported file type"}
            )
            continue
        try:
            chunks = _file_to_chunks(filename, data)
            if not chunks:
                errors.append(
                    {"filename": filename, "error": "No readable text found"}
                )
                continue

            # Replace any previous version of this document.
            _delete_by_filename(client, filename)

            for i in range(0, len(chunks), _BATCH):
                batch = chunks[i : i + _BATCH]
                encoded = embedder.encode_batch([c["text"] for c in batch])
                points = build_points(batch, encoded.dense, encoded.sparse)
                upsert_points(client, points)

            added.append({"filename": filename, "chunks": len(chunks)})
        except Exception as exc:  # noqa: BLE001 - report per-file, keep going
            errors.append({"filename": filename, "error": str(exc)})

    return {"added": added, "errors": errors}


def list_documents(qdrant_url: str) -> list[dict]:
    """Return ingested documents as [{filename, chunks}], sorted by name."""
    client = build_client(qdrant_url)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        return []

    counts: dict[str, int] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["source_filename"],
            limit=256,
            offset=offset,
        )
        for p in points:
            name = (p.payload or {}).get("source_filename", "unknown")
            counts[name] = counts.get(name, 0) + 1
        if offset is None:
            break

    return [
        {"filename": name, "chunks": count}
        for name, count in sorted(counts.items())
    ]


def delete_document(filename: str, qdrant_url: str) -> None:
    """Remove every chunk belonging to a document."""
    client = build_client(qdrant_url)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        _delete_by_filename(client, filename)
