from __future__ import annotations

import hashlib
import logging
import time
import uuid

from utils.agent import get_suggestion_fast, get_suggestion_with_research
from utils.metrics import metrics
from utils.models import (
    ReferencesResponse,
    SuggestionAcceptRequest,
    SuggestionAcceptResponse,
    SuggestionRequest,
    SuggestionResponse,
)
from utils.reference_formatter import build_citation_key, coerce_citation_record, format_references
from utils.store import store

logger = logging.getLogger("citeflow.suggestions")

citation_caches: dict[str, dict[str, dict]] = {}
initialized_documents: set[str] = set()


def ensure_state(document_id: str):
    citation_caches.setdefault(document_id, {})


def build_dedupe_key(request: SuggestionRequest) -> str:
    raw = "|".join(
        [
            request.documentId,
            request.blockId,
            request.title.strip(),
            request.heading.strip(),
            request.paragraphText.strip(),
            request.cursorContext.strip(),
            request.citationStyle.value,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


async def request_suggestion(request: SuggestionRequest) -> SuggestionResponse:
    store.ensure_document(request.documentId, request.title, request.citationStyle.value)
    ensure_state(request.documentId)

    dedupe_key = build_dedupe_key(request)
    if not request.retryOfSuggestionId:
        cached = store.get_latest_suggestion_by_dedupe(request.documentId, dedupe_key)
        if cached and cached.get("text"):
            metrics.increment("suggestions.duplicate_hits")
            return SuggestionResponse(
                suggestionId=cached["suggestion_id"],
                text=cached["text"],
                citations=[coerce_citation_record(c) for c in cached.get("citations_json", [])],
                status="duplicate",
                dedupeKey=dedupe_key,
                latencyMs=0,
                debug={"source": "dedupe-cache"},
            )

    started = time.perf_counter()
    is_first_call = request.documentId not in initialized_documents
    session_cache = citation_caches[request.documentId]

    if is_first_call:
        result = await get_suggestion_with_research(
            document_id=request.documentId,
            title=request.title,
            heading=request.heading,
            content=request.paragraphText or request.cursorContext,
            citation_cache=session_cache,
        )
        initialized_documents.add(request.documentId)
    else:
        result = await get_suggestion_fast(
            document_id=request.documentId,
            title=request.title,
            heading=request.heading,
            content=request.paragraphText or request.cursorContext,
            citation_cache=session_cache,
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    suggestion_id = f"sg_{uuid.uuid4().hex}"
    response = SuggestionResponse(
        suggestionId=suggestion_id,
        text=result.get("suggestion", ""),
        citations=[coerce_citation_record(c) for c in result.get("citations", [])],
        status="generated",
        dedupeKey=dedupe_key,
        latencyMs=elapsed_ms,
        debug={"path": "research" if is_first_call else "fast"},
    )

    store.save_suggestion(
        {
            "suggestion_id": suggestion_id,
            "document_id": request.documentId,
            "block_id": request.blockId,
            "paragraph_text": request.paragraphText,
            "title": request.title,
            "heading": request.heading,
            "cursor_context": request.cursorContext,
            "citation_style": request.citationStyle.value,
            "retry_of_suggestion_id": request.retryOfSuggestionId,
            "dedupe_key": dedupe_key,
            "status": response.status,
            "text": response.text,
            "response_json": response.model_dump(mode="json"),
            "citations": [citation.model_dump(mode="json") for citation in response.citations],
        }
    )
    metrics.increment("suggestions.generated")
    return response


def accept_suggestion(payload: SuggestionAcceptRequest) -> SuggestionAcceptResponse:
    suggestion = store.get_suggestion(payload.suggestionId)
    if not suggestion:
        raise ValueError(f"Unknown suggestion: {payload.suggestionId}")

    store.ensure_document(payload.documentId)
    store.upsert_block(payload.documentId, payload.blockId, payload.acceptedText)
    citations = suggestion.get("citations_json", [])
    for idx, citation in enumerate(citations, start=1):
        citation_key = build_citation_key(citation)
        store.upsert_document_citation(payload.documentId, citation_key, citation)
        store.link_block_citation(
            payload.documentId,
            payload.blockId,
            citation_key,
            payload.suggestionId,
            idx,
        )

    store.mark_suggestion_accepted(payload.suggestionId)
    metrics.increment("suggestions.accepted")

    references = get_document_references(payload.documentId)
    return SuggestionAcceptResponse(
        suggestionId=payload.suggestionId,
        documentId=payload.documentId,
        blockId=payload.blockId,
        acceptedText=payload.acceptedText,
        attachedCitationCount=len(citations),
        referencesVisible=references.visible,
    )


def get_document_references(document_id: str) -> ReferencesResponse:
    document = store.get_document(document_id) or {"citation_style": "APA"}
    style = document["citation_style"]
    citations = store.get_document_citations(document_id)
    entries = format_references(citations, style)
    return ReferencesResponse(
        documentId=document_id,
        citationStyle=style,
        visible=bool(entries),
        entries=entries,
    )


def get_document_manifest(document_id: str):
    document = store.get_document(document_id) or {"citation_style": "APA"}
    return {
        "documentId": document_id,
        "citationStyle": document["citation_style"],
        "blocks": store.get_blocks(document_id),
        "references": get_document_references(document_id).model_dump(mode="json"),
    }
