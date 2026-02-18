"""
CITEFLOW - AI-powered document suggestions with mandatory citations.

Main FastAPI application with WebSocket support for real-time suggestions.

System Design:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    WebSocket Session Lifecycle                       │
  │                                                                      │
  │  CONNECT  ws://host:8000/suggest/{doc_id}                           │
  │     │                                                                │
  │     ▼                                                                │
  │  MESSAGE #1  ─── RESEARCH PATH (slow, ~15-30s) ──────────────────▶  │
  │     │         SearXNG → Firecrawl → Qdrant(store) → Qdrant(query)  │
  │     │         → GPT-4o → suggestion + citations                     │
  │     │                                                                │
  │     │         ✅ doc_id marked as "initialized"                     │
  │     ▼                                                                │
  │  MESSAGE #2+ ─── FAST PATH (fast, ~2-4s) ────────────────────────▶  │
  │     │         Qdrant(query) → GPT-4o → suggestion + citations      │
  │     │         No web search, no scraping — uses stored vectors     │
  │     ▼                                                                │
  │  DISCONNECT                                                          │
  │     │                                                                │
  │     ▼                                                                │
  │  CLEANUP  ── Delete doc_id collection from Qdrant ──────────────▶  │
  └──────────────────────────────────────────────────────────────────────┘
"""

import os
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from utils.agent import get_suggestion_with_research, get_suggestion_fast
from utils.qdrant_ops import delete_collection

# ─── Logging Configuration ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("citeflow.main")

# ─── Session Tracking ───────────────────────────────────────────────────────

# Tracks active WebSocket connections: doc_id → WebSocket
active_sessions: dict[str, WebSocket] = {}

# Tracks which doc_ids have completed their first research call
# (their Qdrant collection is populated and ready for fast queries)
initialized_sessions: set[str] = set()


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logger.info("=" * 60)
    logger.info("  CITEFLOW - AI Writing Assistant with Citations")
    logger.info("=" * 60)
    logger.info("Starting up...")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        logger.warning("⚠️  OPENAI_API_KEY is not set or is a placeholder!")
    else:
        logger.info("✅ OpenAI API key configured")

    logger.info(f"✅ Qdrant URL: {os.getenv('QDRANT_URL', 'http://qdrant-digitrix:6333')}")
    logger.info(f"✅ SearXNG URL: {os.getenv('SEARXNG_URL', 'http://searxng-digitrix:8080')}")
    logger.info(f"✅ Firecrawl URL: {os.getenv('FIRECRAWLER_URL', 'http://firecrawl-api-digitrix:3002')}")

    yield

    # Cleanup on shutdown: delete all session collections from Qdrant
    logger.info("Shutting down... Cleaning up all session data.")
    for doc_id in list(active_sessions.keys()):
        _cleanup_session(doc_id)
    for doc_id in list(initialized_sessions):
        _cleanup_session(doc_id)
    active_sessions.clear()
    initialized_sessions.clear()


def _cleanup_session(doc_id: str):
    """Delete a doc_id's Qdrant collection."""
    try:
        collection_name = f"doc_{doc_id}"
        delete_collection(collection_name)
        logger.info(f"🗑️  Cleaned up Qdrant collection: {collection_name}")
    except Exception as e:
        logger.error(f"Cleanup error for {doc_id}: {e}")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="CITEFLOW",
    description="AI-powered document suggestions with mandatory citations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "CITEFLOW",
        "status": "running",
        "version": "1.0.0",
        "description": "AI-powered document suggestions with mandatory citations",
        "websocket_endpoint": "ws://<host>:8000/suggest/{document_id}",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "initialized_sessions": len(initialized_sessions),
        "services": {
            "qdrant": os.getenv("QDRANT_URL", "http://qdrant-digitrix:6333"),
            "searxng": os.getenv("SEARXNG_URL", "http://searxng-digitrix:8080"),
            "firecrawl": os.getenv("FIRECRAWLER_URL", "http://firecrawl-api-digitrix:3002"),
        },
    }


# ─── WebSocket Endpoint ─────────────────────────────────────────────────────

@app.websocket("/suggest/{document_id}")
async def suggest_websocket(websocket: WebSocket, document_id: str):
    """
    WebSocket endpoint for real-time document suggestions.

    Connect:  ws://host:8000/suggest/{document_id}

    Send JSON:
        {"title": "...", "heading": "...", "content": "..."}

    Receive JSON:
        {"suggestion": "...", "citations": ["url1", "url2"]}

    Lifecycle:
        - 1st message  → full research (search + scrape + store + query)  ~15-30s
        - 2nd+ message → fast query (Qdrant only)                        ~2-4s
        - disconnect    → Qdrant collection for this doc_id is deleted
    """
    await websocket.accept()
    active_sessions[document_id] = websocket

    logger.info(f"🔌 WebSocket connected: document_id={document_id}")

    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "status": "connected",
        "message": f"Connected to CITEFLOW for document: {document_id}",
        "document_id": document_id,
    }))

    try:
        while True:
            # ── Receive message ──────────────────────────────────────────
            raw_data = await websocket.receive_text()
            logger.info(f"📥 Received request for document: {document_id}")

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format. Expected: {title, heading, content}"
                }))
                continue

            title = data.get("title", "")
            heading = data.get("heading", "")
            content = data.get("content", "")

            if not content:
                await websocket.send_text(json.dumps({
                    "error": "Missing required field: 'content'"
                }))
                continue

            # ── Route: research path vs fast path ────────────────────────
            is_first_call = document_id not in initialized_sessions

            if is_first_call:
                logger.info(
                    f"🔬 RESEARCH PATH (1st call) for doc={document_id} | "
                    f"title='{title}', heading='{heading}', "
                    f"content='{content[:60]}...'"
                )
            else:
                logger.info(
                    f"⚡ FAST PATH (subsequent call) for doc={document_id} | "
                    f"heading='{heading}', content='{content[:60]}...'"
                )

            # ── Process ──────────────────────────────────────────────────
            try:
                if is_first_call:
                    result = await get_suggestion_with_research(
                        document_id=document_id,
                        title=title,
                        heading=heading,
                        content=content,
                    )
                    # Mark this doc_id as initialized
                    initialized_sessions.add(document_id)
                    logger.info(f"✅ doc={document_id} initialized — future calls use FAST PATH")
                else:
                    result = await get_suggestion_fast(
                        document_id=document_id,
                        title=title,
                        heading=heading,
                        content=content,
                    )

                # Send the suggestion
                await websocket.send_text(json.dumps(result))

                path_label = "RESEARCH" if is_first_call else "FAST"
                logger.info(
                    f"📤 [{path_label}] Sent suggestion for doc={document_id} "
                    f"with {len(result.get('citations', []))} citations"
                )

            except Exception as e:
                logger.error(f"Error processing suggestion: {e}", exc_info=True)
                await websocket.send_text(json.dumps({
                    "error": f"Processing error: {str(e)}",
                    "suggestion": "",
                    "citations": [],
                }))

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: document_id={document_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {document_id}: {e}", exc_info=True)
    finally:
        # ── Cleanup on disconnect ────────────────────────────────────────
        active_sessions.pop(document_id, None)
        initialized_sessions.discard(document_id)
        _cleanup_session(document_id)
        logger.info(f"🧹 Session fully cleaned up for doc={document_id}")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
