"""One-off script: ingest research/*.md files into Qdrant via the embedding pipeline."""

import json
import re
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Reuse the chunker and embedding pipeline
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.chunker import chunk_text
from embedding.ingest_vectors import ingest_vectors


def md_to_chunks(md_path: Path, doc_index: int) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")

    # Split into logical sections by heading
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    for sec_i, section in enumerate(sections):
        heading_match = re.match(r"^(#{1,3}) (.+)", section)
        heading = heading_match.group(2).strip() if heading_match else md_path.stem

        token_chunks = chunk_text(section)
        for chunk_i, chunk_text_val in enumerate(token_chunks):
            slug = f"{md_path.stem}-s{sec_i}-c{chunk_i}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, slug))
            chunks.append({
                "chunk_id": chunk_id,
                "source_filename": md_path.name,
                "page_number": sec_i + 1,
                "section_heading": heading,
                "chunk_index": chunk_i,
                "token_count": len(chunk_text_val.split()),
                "text": chunk_text_val,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return chunks


def main():
    research_dir = Path(__file__).parent / "research"
    md_files = sorted(research_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    all_chunks = []
    for i, md_file in enumerate(md_files):
        chunks = md_to_chunks(md_file, i)
        print(f"  {md_file.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    out_path = Path(__file__).parent / "output" / "chunks.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_chunks)} chunks to {out_path}")
    return out_path


if __name__ == "__main__":
    chunks_path = main()
    print("\nRunning embedding pipeline...")
    ingest_vectors(str(chunks_path))
    print("\nDone! Qdrant collection is ready.")
