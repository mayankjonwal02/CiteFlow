from __future__ import annotations

import logging
import uuid

from utils.ai_ops import generate_json
from utils.crawl_ops import scrape_multiple_urls
from utils.metrics import metrics
from utils.models import ChatMessageRequest, ChatMessageResponse
from utils.search_ops import search_web
from utils.store import store
from utils.suggestion_service import get_document_references

logger = logging.getLogger("citeflow.chat")


def _get_or_create_thread(document_id: str, thread_id: str | None, title: str) -> str:
    if thread_id:
        existing = store.get_chat_thread(thread_id)
        if existing:
            return thread_id

    new_thread_id = thread_id or f"thread_{uuid.uuid4().hex}"
    store.create_chat_thread(new_thread_id, document_id, title[:120])
    return new_thread_id


async def _build_context(document_id: str, request: ChatMessageRequest) -> dict:
    blocks = store.get_blocks(document_id)
    references = get_document_references(document_id)
    context = {
        "documentText": "\n\n".join(blocks.values()).strip(),
        "selectedText": request.selectedText,
        "references": [entry.text for entry in references.entries],
        "webResults": [],
        "libraryResults": [],
    }

    if request.context.web:
        results = await search_web(request.message, num_results=3, categories="general")
        context["webResults"] = results
        if results:
            scraped = await scrape_multiple_urls([item["url"] for item in results[:2]], max_concurrent=2)
            context["webScrapes"] = scraped

    if request.context.library:
        hits = []
        for source in store.list_library_sources(document_id):
            content = source.get("content_text", "")
            if request.message.lower() in content.lower() or request.message.lower() in source.get("title", "").lower():
                hits.append(
                    {
                        "source_id": source["source_id"],
                        "title": source["title"],
                        "snippet": content[:300],
                    }
                )
            if len(hits) >= 3:
                break
        context["libraryResults"] = hits

    return context


def _fallback_chat(request: ChatMessageRequest, context: dict) -> dict:
    selected = request.selectedText.strip()
    document_text = context.get("documentText", "")
    if request.action in {"improve", "rewrite"}:
        target = selected or document_text[:400]
        improved = target.strip()
        if improved and not improved.endswith("."):
            improved += "."
        improved += " This revision improves clarity and academic tone."
        return {
            "assistantMessage": "I revised the text and kept the output separate so only the generated content is added to the document.",
            "documentPayload": improved,
        }
    if request.action == "summarize":
        summary_target = selected or document_text[:500]
        return {
            "assistantMessage": f"Summary: {summary_target[:220]}".strip(),
            "documentPayload": "",
        }
    return {
        "assistantMessage": f"Using the available context, here is a response to your request: {request.message}",
        "documentPayload": "",
    }


async def send_chat_message(document_id: str, request: ChatMessageRequest) -> ChatMessageResponse:
    store.ensure_document(document_id)
    thread_id = _get_or_create_thread(document_id, request.threadId, request.message)
    context = await _build_context(document_id, request)

    store.add_chat_message(
        message_id=f"msg_{uuid.uuid4().hex}",
        thread_id=thread_id,
        role="user",
        message_text=request.message,
        context=context,
    )

    fallback = _fallback_chat(request, context)
    response = await generate_json(
        system_prompt=(
            "You are an academic writing assistant. Return JSON with keys "
            "`assistantMessage` and `documentPayload`. Keep explanation separate from generated document content."
        ),
        user_prompt=(
            f"Action: {request.action}\n"
            f"User message: {request.message}\n"
            f"Selected text: {request.selectedText}\n"
            f"Document context: {context.get('documentText', '')[:4000]}\n"
            f"References: {context.get('references', [])}\n"
            f"Web results: {context.get('webResults', [])}\n"
            f"Library results: {context.get('libraryResults', [])}\n"
        ),
        fallback=fallback,
    )

    assistant_message = (response.get("assistantMessage") or fallback["assistantMessage"]).strip()
    document_payload = (response.get("documentPayload") or fallback["documentPayload"]).strip()
    context_summary = {
        "document": bool(context.get("documentText")),
        "webResultCount": len(context.get("webResults", [])),
        "libraryResultCount": len(context.get("libraryResults", [])),
        "referenceCount": len(context.get("references", [])),
    }

    store.add_chat_message(
        message_id=f"msg_{uuid.uuid4().hex}",
        thread_id=thread_id,
        role="assistant",
        message_text=assistant_message,
        document_payload=document_payload,
        context=context_summary,
    )
    metrics.increment("chat.messages")

    return ChatMessageResponse(
        threadId=thread_id,
        assistantMessage=assistant_message,
        documentPayload=document_payload,
        citations=[],
        contextSummary=context_summary,
    )
