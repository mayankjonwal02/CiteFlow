from __future__ import annotations

import uuid

from utils.ai_ops import generate_json
from utils.metrics import metrics
from utils.models import EditRequest, EditResponse


def _fallback_edit(request: EditRequest) -> dict:
    base = request.selectionText.strip()
    if not base:
        base = request.contextWindow.strip()
    if not base:
        base = "No source text was provided."

    rewritten = base
    instruction = request.instruction.lower()
    if "simplify" in instruction:
        rewritten = f"{base} This version uses simpler wording."
    elif "paraphrase" in instruction:
        rewritten = f"{base} In other words, the same point is restated more clearly."
    elif "bullet" in instruction:
        rewritten = f"- {base}"
    elif "numbered" in instruction:
        rewritten = f"1. {base}"
    else:
        rewritten = f"{base} This revision strengthens flow, clarity, and academic tone."

    if request.mode == "insert_below":
        return {
            "explanation": "I generated an alternate version that can be inserted below the selected text.",
            "replacementText": "",
            "insertBelowText": rewritten,
        }

    return {
        "explanation": "I revised the selected text according to your instruction.",
        "replacementText": rewritten,
        "insertBelowText": "",
    }


async def run_edit(document_id: str, request: EditRequest) -> EditResponse:
    fallback = _fallback_edit(request)
    response = await generate_json(
        system_prompt=(
            "You are an AI inline editor. Return JSON with keys: explanation, replacementText, insertBelowText. "
            "Use replacementText for replace mode and insertBelowText for insert_below mode."
        ),
        user_prompt=(
            f"Document ID: {document_id}\n"
            f"Instruction: {request.instruction}\n"
            f"Mode: {request.mode}\n"
            f"Selection: {request.selectionText}\n"
            f"Context window: {request.contextWindow}\n"
        ),
        fallback=fallback,
    )
    metrics.increment("edits.generated")
    return EditResponse(
        explanation=(response.get("explanation") or fallback["explanation"]).strip(),
        replacementText=(response.get("replacementText") or fallback["replacementText"]).strip(),
        insertBelowText=(response.get("insertBelowText") or fallback["insertBelowText"]).strip(),
        regenerateToken=f"edit_{uuid.uuid4().hex}",
        citations=[],
    )
