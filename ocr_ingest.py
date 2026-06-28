"""OCR ingestion for image-based PDFs using GPT-4o Vision.

Renders each PDF page → base64 image → GPT-4o extracts full text → chunks → Qdrant.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

from ingestion.chunker import chunk_text
from embedding.ingest_vectors import ingest_vectors

client = OpenAI()

_OCR_PROMPT = (
    "You are an OCR assistant. Extract ALL text visible in this lecture slide image "
    "exactly as written — including headings, bullet points, code, formulas, and diagrams described in words. "
    "Preserve structure with newlines. Output plain text only, no commentary."
)


def page_to_base64(page: fitz.Page, dpi: int = 150) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("png")).decode()


def ocr_page(page: fitz.Page) -> str:
    b64 = page_to_base64(page)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ],
        }],
        max_tokens=2000,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def ingest_pdf(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    chunks = []
    print(f"  [{pdf_path.name}] {len(doc)} pages", flush=True)

    for page_num, page in enumerate(doc, start=1):
        print(f"    page {page_num}/{len(doc)} → OCR...", end=" ", flush=True)
        text = ocr_page(page)
        print(f"{len(text)} chars", flush=True)

        if not text.strip():
            continue

        text_chunks = chunk_text(text)
        for chunk_i, chunk_text_val in enumerate(text_chunks):
            chunks.append({
                "chunk_id": f"{pdf_path.stem}-p{page_num}-c{chunk_i}",
                "source_filename": pdf_path.name,
                "page_number": page_num,
                "section_heading": pdf_path.stem,
                "chunk_index": chunk_i,
                "token_count": len(chunk_text_val.split()),
                "text": chunk_text_val,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    doc.close()
    return chunks


def main():
    input_dir = Path(__file__).parent / "input"
    pdf_files = sorted(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs\n")

    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        chunks = ingest_pdf(pdf_path)
        print(f"  → {len(chunks)} chunks\n")
        all_chunks.extend(chunks)

    out_path = Path(__file__).parent / "output" / "chunks.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} total chunks to {out_path}\n")

    print("Running embedding pipeline (BGE-M3)...")
    ingest_vectors(str(out_path))
    print("\nDone — Qdrant is ready!")


if __name__ == "__main__":
    main()
