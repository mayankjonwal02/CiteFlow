from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger("citeflow.ai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CITEFLOW_CHAT_MODEL", "gpt-4o-mini")

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI | None:
    global _client
    if not OPENAI_API_KEY:
        return None
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def generate_json(
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    client = get_client()
    if client is None:
        return fallback

    try:
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            return json.loads(text[json_start:json_end])
    except Exception as exc:
        logger.error("JSON generation failed: %s", exc)

    return fallback


async def generate_text(system_prompt: str, user_prompt: str, fallback: str) -> str:
    client = get_client()
    if client is None:
        return fallback

    try:
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip() or fallback
    except Exception as exc:
        logger.error("Text generation failed: %s", exc)
        return fallback
