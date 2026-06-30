"""SSE formatting helpers."""

from __future__ import annotations

import json


def format_sse(data: dict) -> str:
    """Encode a dict as a Server-Sent Event data line."""
    return f"data: {json.dumps(data)}\n\n"
