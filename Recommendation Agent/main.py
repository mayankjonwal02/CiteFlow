from __future__ import annotations

"""CITEFLOW application with durable document, citation, chat, and library APIs."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from utils.chat_service import send_chat_message
from utils.edit_service import run_edit
from utils.library_service import import_bibtex, import_doi, import_pdf_bytes, search_library
from utils.metrics import metrics
from utils.models import (
    BibtexImportRequest,
    ChatMessageRequest,
    CitationStyleUpdateRequest,
    DocumentManifestResponse,
    DOIImportRequest,
    EditRequest,
    LibrarySearchRequest,
    SuggestionAcceptRequest,
    SuggestionRequest,
)
from utils.store import store
from utils.suggestion_service import (
    accept_suggestion,
    citation_caches,
    get_document_manifest,
    get_document_references,
    initialized_documents,
    request_suggestion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("citeflow.main")

active_sessions: dict[str, WebSocket] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  CITEFLOW - AI Writing Assistant")
    logger.info("=" * 60)
    store.initialize()
    logger.info("✅ SQLite DB: %s", os.getenv("CITEFLOW_DB_PATH", "./data/citeflow.db"))
    logger.info("✅ Qdrant URL: %s", os.getenv("QDRANT_URL", "http://qdrant-digitrix:6333"))
    logger.info("✅ SearXNG URL: %s", os.getenv("SEARXNG_URL", "http://searxng-digitrix:8080"))
    logger.info("✅ Firecrawl URL: %s", os.getenv("FIRECRAWLER_URL", "http://firecrawl-api-digitrix:3002"))
    yield
    active_sessions.clear()
    citation_caches.clear()
    initialized_documents.clear()


app = FastAPI(
    title="CITEFLOW",
    description="AI-powered academic editor backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "CITEFLOW",
        "status": "running",
        "version": "2.0.0",
        "description": "AI academic writing backend with suggestions, references, chat, edits, and library ingestion",
        "websocket_endpoint": "ws://<host>:8000/suggest/{document_id}",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "initialized_sessions": len(initialized_documents),
        "services": {
            "qdrant": os.getenv("QDRANT_URL", "http://qdrant-digitrix:6333"),
            "searxng": os.getenv("SEARXNG_URL", "http://searxng-digitrix:8080"),
            "firecrawl": os.getenv("FIRECRAWLER_URL", "http://firecrawl-api-digitrix:3002"),
            "sqlite": os.getenv("CITEFLOW_DB_PATH", "./data/citeflow.db"),
        },
    }


@app.get("/metrics")
async def get_metrics():
    return {"metrics": metrics.snapshot()}


@app.post("/suggestions/accept")
async def accept_suggestion_endpoint(payload: SuggestionAcceptRequest):
    try:
        return accept_suggestion(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/documents/{document_id}/references")
async def get_references(document_id: str):
    return get_document_references(document_id)


@app.patch("/documents/{document_id}/citation-style")
async def update_citation_style(document_id: str, payload: CitationStyleUpdateRequest):
    store.update_citation_style(document_id, payload.citationStyle.value)
    return get_document_references(document_id)


@app.get("/documents/{document_id}/manifest")
async def document_manifest(document_id: str) -> DocumentManifestResponse:
    return DocumentManifestResponse.model_validate(get_document_manifest(document_id))


@app.post("/documents/{document_id}/chat/messages")
async def chat_message(document_id: str, payload: ChatMessageRequest):
    return await send_chat_message(document_id, payload)


@app.get("/documents/{document_id}/chat/threads/{thread_id}")
async def get_chat_thread(document_id: str, thread_id: str):
    thread = store.get_chat_thread(thread_id)
    if not thread or thread["document_id"] != document_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread": thread, "messages": store.get_chat_messages(thread_id)}


@app.post("/documents/{document_id}/edits")
async def edit_document(document_id: str, payload: EditRequest):
    return await run_edit(document_id, payload)


@app.post("/library/import/doi")
async def import_library_doi(payload: DOIImportRequest):
    return await import_doi(payload)


@app.post("/library/import/bibtex")
async def import_library_bibtex(payload: BibtexImportRequest):
    return await import_bibtex(payload)


@app.post("/library/import/pdf")
async def import_library_pdf(file: UploadFile = File(...), document_id: Optional[str] = None):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    return await import_pdf_bytes(file.filename or "upload.pdf", data, document_id=document_id)


@app.post("/library/search")
async def search_library_endpoint(payload: LibrarySearchRequest):
    return await search_library(payload)


@app.websocket("/suggest/{document_id}")
async def suggest_websocket(websocket: WebSocket, document_id: str):
    await websocket.accept()
    active_sessions[document_id] = websocket
    citation_caches.setdefault(document_id, {})

    await websocket.send_text(
        json.dumps(
            {
                "status": "connected",
                "message": f"Connected to CITEFLOW for document: {document_id}",
                "document_id": document_id,
            }
        )
    )

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON payload"}))
                continue

            paragraph_text = data.get("paragraphText", data.get("content", ""))
            if not paragraph_text:
                await websocket.send_text(json.dumps({"error": "Missing required field: 'paragraphText' or 'content'"}))
                continue

            request = SuggestionRequest(
                documentId=document_id,
                blockId=data.get("blockId", "paragraph-1"),
                paragraphText=paragraph_text,
                title=data.get("title", ""),
                heading=data.get("heading", ""),
                cursorContext=data.get("cursorContext", ""),
                citationStyle=data.get("citationStyle", "APA"),
                retryOfSuggestionId=data.get("retryOfSuggestionId"),
            )

            try:
                result = await request_suggestion(request)
                await websocket.send_text(json.dumps(result.model_dump(mode="json")))
            except Exception as exc:
                logger.error("Suggestion processing failed for %s: %s", document_id, exc, exc_info=True)
                await websocket.send_text(
                    json.dumps(
                        {
                            "error": f"Processing error: {exc}",
                            "suggestion": "",
                            "citations": [],
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", document_id)
    finally:
        active_sessions.pop(document_id, None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
