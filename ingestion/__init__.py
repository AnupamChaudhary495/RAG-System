"""Ingestion pipeline for the Enterprise RAG system.

Accepts a directory of PDF files and outputs sanitized, metadata-enriched
text chunks ready for embedding.
"""

from ingestion.pipeline import ingest_directory

__all__ = ["ingest_directory"]
