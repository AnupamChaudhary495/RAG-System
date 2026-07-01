"""FastAPI application — SSE streaming chat endpoint + session management."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

load_dotenv(Path(__file__).parent.parent / ".env")

from api import documents, session
from api.schemas import ChatRequest
from api.streaming import format_sse
from observability.langfuse_handler import make_langfuse_config
from orchestration.app import build_app


def _qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333")


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = aioredis.from_url(redis_url, protocol=2)
    session.set_redis_client(redis_client)
    yield
    await redis_client.aclose()


app = FastAPI(title="RAG System API", version="1.0.0", lifespan=lifespan)

_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    try:
        await session._redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "error"
    return {"status": "ok", "redis": redis_status}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            history = await session.get_history(request.session_id)

            graph = build_app()
            initial_state = {
                "query": request.query,
                "original_query": request.query,
                "conversation_history": history[-10:],
                "router_decision": "",
                "retrieved_chunks": [],
                "rewritten_query": None,
                "answer": "",
                "source_chunk_ids": [],
                "confidence_score": 0.0,
                "retry_count": 0,
            }

            langfuse_config = make_langfuse_config(
                session_id=request.session_id,
                query=request.query,
            )

            full_answer = ""
            final_state = None

            async for event in graph.astream_events(
                initial_state, version="v2", config=langfuse_config
            ):
                kind = event.get("event")
                name = event.get("name")

                if kind == "on_chat_model_stream" and name == "generator":
                    token = event["data"]["chunk"].content
                    if token:
                        full_answer += token
                        yield format_sse({"type": "token", "content": token})

                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event["data"].get("output")

            if final_state:
                yield format_sse({
                    "type": "metadata",
                    "source_chunk_ids": final_state.get("source_chunk_ids", []),
                    "confidence_score": final_state.get("confidence_score", 0.0),
                    "retry_count": final_state.get("retry_count", 0),
                    "router_decision": final_state.get("router_decision", ""),
                })
                await session.append_turns(
                    request.session_id, request.query, full_answer
                )

            yield format_sse({"type": "done"})

        except Exception as exc:
            yield format_sse({"type": "error", "message": str(exc)})
            yield format_sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await session.clear_session(session_id)
    return {"deleted": session_id}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    history = await session.get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.get("/chunks")
async def get_chunks(ids: str = ""):
    if not ids:
        return []
    chunk_ids = [i.strip() for i in ids.split(",") if i.strip()]
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant = QdrantClient(url=qdrant_url)
    points = qdrant.retrieve(
        collection_name="rag_chunks",
        ids=chunk_ids,
        with_payload=True,
    )
    return [
        {
            "chunk_id": str(p.id),
            "source_filename": p.payload.get("source_filename", ""),
            "page_number": p.payload.get("page_number"),
            "section_heading": p.payload.get("section_heading"),
            "token_count": p.payload.get("token_count", 0),
        }
        for p in points
    ]


@app.post("/documents")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Ingest uploaded files (PDF / Markdown / text) into the knowledge base."""
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        payloads.append((f.filename or "upload", data))
    if not payloads:
        raise HTTPException(status_code=400, detail="No files provided")

    result = await asyncio.to_thread(
        documents.ingest_uploaded_files, payloads, _qdrant_url()
    )
    return result


@app.get("/documents")
async def get_documents():
    """List ingested documents with their chunk counts."""
    return await asyncio.to_thread(documents.list_documents, _qdrant_url())


@app.delete("/documents/{filename}")
async def remove_document(filename: str):
    """Remove every chunk of a document from the knowledge base."""
    await asyncio.to_thread(documents.delete_document, filename, _qdrant_url())
    return {"deleted": filename}
