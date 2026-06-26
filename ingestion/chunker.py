"""Recursive character chunking with token-based size control and exact overlap.

Algorithm:
  Phase 1 — Recursive split: break the text at natural separator boundaries
             (double-newline > newline > sentence > word) so no chunk exceeds
             CHUNK_SIZE tokens when possible.
  Phase 2 — Sliding window: re-encode the full token sequence and apply a
             sliding window of width CHUNK_SIZE and step (CHUNK_SIZE - OVERLAP).
             This guarantees chunk[n+1] starts exactly at position
             n * (CHUNK_SIZE - OVERLAP) tokens into the document, giving a
             provably exact OVERLAP-token overlap between consecutive chunks.
"""

import tiktoken

CHUNK_SIZE = 512
OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " "]

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in text."""
    return len(_encoding.encode(text))


# ---------------------------------------------------------------------------
# Phase 1 helpers — recursive character splitting
# ---------------------------------------------------------------------------

def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split text by separator, re-attaching the separator to each segment's tail."""
    if not separator:
        return list(text)
    parts = text.split(separator)
    result: list[str] = []
    for i, part in enumerate(parts):
        segment = part + separator if i < len(parts) - 1 else part
        if segment:
            result.append(segment)
    return result


def _force_split_tokens(text: str, chunk_size: int) -> list[str]:
    """Last-resort split at exact token boundaries when no separator fits."""
    tokens = _encoding.encode(text)
    return [
        _encoding.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text using separator priority until each piece ≤ chunk_size.

    Separators are tried in order; the first one found in the text is used.
    Pieces that are still too large are split with the next separator in the list,
    falling back to token-level splitting when no separator remains.
    """
    if count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    # Choose the highest-priority separator present in the text
    chosen_sep: str = ""
    remaining_seps: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            chosen_sep = sep
            remaining_seps = separators[i + 1 :]
            break
    else:
        return _force_split_tokens(text, chunk_size)

    splits = _split_by_separator(text, chosen_sep)
    final: list[str] = []

    for split in splits:
        if count_tokens(split) <= chunk_size:
            final.append(split)
        elif remaining_seps:
            final.extend(_recursive_split(split, remaining_seps, chunk_size))
        else:
            final.extend(_force_split_tokens(split, chunk_size))

    return final


# ---------------------------------------------------------------------------
# Phase 2 — sliding window with exact overlap
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of at most CHUNK_SIZE tokens.

    Uses recursive character splitting (respecting separator priority) followed
    by a token-level sliding window that enforces exactly OVERLAP tokens of
    overlap between consecutive chunks:

        chunk[n] covers tokens[n*step : n*step + CHUNK_SIZE]
        where step = CHUNK_SIZE - OVERLAP = 462

    Args:
        text: The full document text to chunk.

    Returns:
        Ordered list of chunk strings, each ≤ CHUNK_SIZE tokens.
        Empty or whitespace-only chunks are excluded.
    """
    if not text.strip():
        return []

    # Phase 1: recursive split preserves separator boundaries in the token stream
    splits = _recursive_split(text, SEPARATORS, CHUNK_SIZE)
    if not splits:
        return []

    # Phase 2: encode the reconstructed text as ONE unit so BPE merges are
    # identical to encoding the original string (piece-wise encoding produces
    # different token sequences for context-sensitive BPE merges like
    # "vector " + "vector " vs "vector vector").
    all_tokens: list[int] = _encoding.encode("".join(splits))

    step = CHUNK_SIZE - OVERLAP  # 462 tokens per step
    chunks: list[str] = []
    n = 0

    while True:
        start = n * step
        if start >= len(all_tokens):
            break
        end = min(start + CHUNK_SIZE, len(all_tokens))
        chunk = _encoding.decode(all_tokens[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(all_tokens):
            break
        n += 1

    return chunks
