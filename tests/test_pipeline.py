"""Unit tests for the RAG ingestion pipeline.

Covers:
  - Sanitizer: page number removal, hyphenated line break repair
  - Chunker: token size limits, exact 50-token overlap
  - Metadata: required key presence
  - Pipeline: error handling for missing directory
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import tiktoken

# ---------------------------------------------------------------------------
# Sanitizer tests
# ---------------------------------------------------------------------------

from ingestion.sanitizer import (
    _remove_page_numbers,
    _repair_hyphenated_breaks,
    _detect_boilerplate,
    _strip_boilerplate,
    sanitize_pages,
)
from ingestion.parsers import PageData


class TestSanitizerPageNumbers:
    def test_removes_page_n_of_m(self):
        text = "Some content\nPage 12 of 50\nMore content"
        result = _remove_page_numbers(text)
        assert "Page 12 of 50" not in result
        assert "Some content" in result

    def test_removes_page_lowercase(self):
        text = "Introduction\npage 3 of 10\nBody text"
        result = _remove_page_numbers(text)
        assert "page 3 of 10" not in result

    def test_removes_dash_number_dash(self):
        text = "Header\n- 12 -\nParagraph"
        result = _remove_page_numbers(text)
        assert "- 12 -" not in result

    def test_removes_standalone_digit_line(self):
        text = "Chapter One\n\n42\n\nThis is the text."
        result = _remove_page_numbers(text)
        lines = [l.strip() for l in result.splitlines()]
        assert "42" not in lines

    def test_preserves_inline_numbers(self):
        # Digits embedded in sentences must NOT be stripped
        text = "The system handles 42 requests per second."
        result = _remove_page_numbers(text)
        assert "42" in result


class TestSanitizerHyphenatedBreaks:
    def test_repairs_simple_hyphenated_break(self):
        result = _repair_hyphenated_breaks("com-\nputer")
        assert result == "computer"

    def test_repairs_mid_sentence(self):
        text = "The re-\nsult was unexpected in the ex-\nperiment."
        result = _repair_hyphenated_breaks(text)
        assert "result" in result
        assert "experiment" in result
        assert "-\n" not in result

    def test_preserves_legitimate_hyphens(self):
        # Hyphens not followed by newlines should be left alone
        text = "well-known algorithm"
        result = _repair_hyphenated_breaks(text)
        assert result == "well-known algorithm"


class TestBoilerplateDetection:
    def _make_pages(self, texts: list[str]) -> list[PageData]:
        return [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)]

    def test_detects_repeated_header(self):
        footer = "ACME Corporation — Confidential"
        pages = self._make_pages([
            f"Content A\n{footer}",
            f"Content B\n{footer}",
            f"Content C\n{footer}",
            f"Content D\n{footer}",
            "Content E",
        ])
        boilerplate = _detect_boilerplate(pages)
        assert footer in boilerplate

    def test_does_not_flag_rare_lines(self):
        pages = self._make_pages([
            "Unique sentence one.",
            "Unique sentence two.",
            "Common line\nUnique three.",
            "Unique four.",
            "Unique five.",
        ])
        boilerplate = _detect_boilerplate(pages)
        assert "Common line" not in boilerplate  # only on 1/5 = 20% of pages

    def test_strips_detected_boilerplate(self):
        footer = "FOOTER TEXT"
        pages = self._make_pages([f"Real content\n{footer}"] * 5)
        boilerplate = _detect_boilerplate(pages)
        sanitized = sanitize_pages(pages)
        for page in sanitized:
            assert footer not in page["text"]


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

from ingestion.chunker import chunk_text, count_tokens, CHUNK_SIZE, OVERLAP, _encoding


class TestChunkerTokenLimits:
    def _long_text(self, n_words: int = 5000) -> str:
        word = "retrieval "
        return (word * n_words).strip()

    def test_all_chunks_within_size_limit(self):
        text = self._long_text(5000)
        chunks = chunk_text(text)
        assert chunks, "chunker produced no chunks"
        for i, chunk in enumerate(chunks):
            tokens = count_tokens(chunk)
            assert tokens <= CHUNK_SIZE, (
                f"Chunk {i} has {tokens} tokens, exceeds limit of {CHUNK_SIZE}"
            )

    def test_produces_multiple_chunks_for_long_text(self):
        text = self._long_text(5000)
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n   ") == []

    def test_short_text_returns_single_chunk(self):
        text = "This is a short document."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestChunkerOverlap:
    def _token_ids(self, text: str) -> list[int]:
        return _encoding.encode(text)

    def test_consecutive_chunks_overlap_exactly_50_tokens(self):
        # Create text long enough to produce at least 3 chunks
        word = "embedding "
        text = (word * 6000).strip()
        chunks = chunk_text(text)

        assert len(chunks) >= 3, "Need at least 3 chunks to test overlap"

        for i in range(len(chunks) - 1):
            ids_a = self._token_ids(chunks[i])
            ids_b = self._token_ids(chunks[i + 1])

            # chunk[i+1] starts at position (i+1)*(CHUNK_SIZE-OVERLAP) in the stream,
            # so the last OVERLAP tokens of chunk[i] equal the first OVERLAP tokens of chunk[i+1]
            # (unless chunk[i] is shorter than CHUNK_SIZE — last chunk edge case).
            if len(ids_a) == CHUNK_SIZE:
                overlap_from_a = ids_a[-OVERLAP:]
                overlap_from_b = ids_b[:OVERLAP]
                assert overlap_from_a == overlap_from_b, (
                    f"Chunks {i} and {i+1} do not share exactly {OVERLAP} overlap tokens"
                )

    def test_overlap_step_formula(self):
        """chunk[n+1] starts at n*(CHUNK_SIZE-OVERLAP) tokens into the document."""
        step = CHUNK_SIZE - OVERLAP
        word = "vector "
        text = (word * 6000).strip()
        all_tokens = self._token_ids(text)
        # Re-produce what chunk_text does internally to verify the formula
        chunks = chunk_text(text)

        for n, chunk in enumerate(chunks):
            expected_start = n * step
            if expected_start >= len(all_tokens):
                break
            expected_end = min(expected_start + CHUNK_SIZE, len(all_tokens))
            expected_tokens = all_tokens[expected_start:expected_end]
            actual_tokens = self._token_ids(chunk)
            assert actual_tokens == expected_tokens, (
                f"Chunk {n} does not match sliding window formula"
            )


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

from ingestion.metadata import build_chunk_metadata

REQUIRED_METADATA_KEYS = {
    "chunk_id",
    "source_filename",
    "page_number",
    "timestamp",
    "section_heading",
    "chunk_index",
    "token_count",
}


class TestMetadata:
    def _sample_metadata(self, **overrides) -> dict:
        defaults = dict(
            chunk_index=0,
            chunk_start_char=0,
            full_text="Introduction:\nThis is the content.",
            source_filename="sample.pdf",
            page_boundaries=[(0, 100, 1)],
            ingest_time=datetime.now(timezone.utc),
            token_count=42,
        )
        defaults.update(overrides)
        return build_chunk_metadata(**defaults)

    def test_all_required_keys_present(self):
        meta = self._sample_metadata()
        missing = REQUIRED_METADATA_KEYS - meta.keys()
        assert not missing, f"Metadata missing keys: {missing}"

    def test_chunk_id_is_uuid4_string(self):
        import uuid
        meta = self._sample_metadata()
        # Should not raise
        parsed = uuid.UUID(meta["chunk_id"])
        assert parsed.version == 4

    def test_timestamp_is_iso8601(self):
        from datetime import datetime
        meta = self._sample_metadata()
        # Should not raise; isoformat includes timezone offset
        dt = datetime.fromisoformat(meta["timestamp"])
        assert dt.tzinfo is not None

    def test_chunk_index_matches_input(self):
        meta = self._sample_metadata(chunk_index=7)
        assert meta["chunk_index"] == 7

    def test_token_count_matches_input(self):
        meta = self._sample_metadata(token_count=123)
        assert meta["token_count"] == 123

    def test_page_number_resolved_from_boundaries(self):
        boundaries = [(0, 200, 1), (201, 400, 2)]
        meta = self._sample_metadata(chunk_start_char=250, page_boundaries=boundaries)
        assert meta["page_number"] == 2

    def test_page_number_none_for_out_of_range(self):
        boundaries = [(0, 100, 1)]
        meta = self._sample_metadata(chunk_start_char=500, page_boundaries=boundaries)
        assert meta["page_number"] is None

    def test_section_heading_detected_all_caps(self):
        full_text = "INTRODUCTION\nSome paragraph text follows."
        meta = self._sample_metadata(
            full_text=full_text,
            chunk_start_char=full_text.index("Some"),
        )
        assert meta["section_heading"] == "INTRODUCTION"

    def test_section_heading_detected_title_case(self):
        full_text = "Getting Started\nSome content here."
        meta = self._sample_metadata(
            full_text=full_text,
            chunk_start_char=full_text.index("Some"),
        )
        assert meta["section_heading"] == "Getting Started"

    def test_section_heading_detected_colon(self):
        full_text = "Prerequisites:\nInstall Python 3.11."
        meta = self._sample_metadata(
            full_text=full_text,
            chunk_start_char=full_text.index("Install"),
        )
        assert meta["section_heading"] == "Prerequisites:"

    def test_section_heading_none_when_absent(self):
        full_text = "just lowercase text here nothing special."
        meta = self._sample_metadata(
            full_text=full_text,
            chunk_start_char=len(full_text),
        )
        assert meta["section_heading"] is None


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

from ingestion.pipeline import ingest_directory


class TestPipelineErrors:
    def test_raises_file_not_found_for_missing_directory(self):
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            ingest_directory("/nonexistent/path/does/not/exist", "/tmp/out.json")

    def test_raises_not_a_directory_for_file_path(self, tmp_path: Path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        with pytest.raises(NotADirectoryError):
            ingest_directory(str(file_path), str(tmp_path / "out.json"))

    def test_raises_value_error_for_empty_directory(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No PDF files found"):
            ingest_directory(str(tmp_path), str(tmp_path / "out.json"))
