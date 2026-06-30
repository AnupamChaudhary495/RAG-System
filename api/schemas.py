"""Pydantic request/response schemas for the RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: str = Field(..., min_length=1, max_length=128)
