"""
Web search operations with SearXNG (default) and opt-in Parallel Search MCP.
"""

import os
import asyncio
import json
import uuid
from datetime import timedelta
import logging
import httpx

logger = logging.getLogger("citeflow.search")

SEARCH_PROVIDER = os.getenv("CITEFLOW_SEARCH_PROVIDER", "searxng").strip().lower() or "searxng"
PARALLEL_TIMEOUT_SECONDS = 30.0
PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
PARALLEL_USER_AGENT = "CiteFlow/ParallelSearch (+https://github.com/mayankjonwal02/CiteFlow)"
# Direct callers share a stable identifier; the research agent supplies one per run.
_PARALLEL_SESSION_ID = uuid.uuid4().hex

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng-digitrix:8080")


async def _search_web_searxng(query: str, num_results: int = 5, categories: str = "general") -> list[dict]:
    """
    Search the web using SearXNG meta-search engine.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return.
        categories: SearXNG search categories (e.g., 'general', 'news', 'science').

    Returns:
        A list of dicts with keys: title, url, snippet.
    """
    search_url = f"{SEARXNG_URL}/search"
    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "pageno": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })

        logger.info(f"SearXNG returned {len(results)} results for query: '{query}'")
        return results

    except httpx.HTTPStatusError as e:
        logger.error(f"SearXNG HTTP error: {e.response.status_code} - {e.response.text}")
        return []
    except httpx.ConnectError:
        logger.error(f"Cannot connect to SearXNG at {SEARXNG_URL}")
        return []
    except Exception as e:
        logger.error(f"SearXNG search error: {e}")
        return []


async def _search_web_parallel(query: str, num_results: int, session_id: str) -> list[dict]:
    """Call Parallel's anonymous MCP endpoint through Streamable HTTP."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        # Bound the whole operation, including handshake, tool discovery and cleanup.
        async with asyncio.timeout(PARALLEL_TIMEOUT_SECONDS):
            async with streamablehttp_client(
                PARALLEL_MCP_URL,
                headers={"User-Agent": PARALLEL_USER_AGENT},
                timeout=PARALLEL_TIMEOUT_SECONDS,
                sse_read_timeout=PARALLEL_TIMEOUT_SECONDS,
            ) as (read, write, _):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=PARALLEL_TIMEOUT_SECONDS)) as session:
                    await session.initialize()
                    result = await session.call_tool("web_search", {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": session_id,
                    })
                    if result.isError:
                        logger.warning("Parallel Search MCP returned a tool error")
                        return []
                    data = result.structuredContent
                    if data is None:
                        # MCP also permits JSON encoded in a text content block.
                        data = json.loads(next(block.text for block in result.content if block.type == "text"))

        results = [{
            "title": item.get("title") or "",
            "url": item["url"],
            "snippet": "\n\n".join(item.get("excerpts", [])),
        } for item in data["results"][:num_results]]
        logger.info("Parallel returned %s results", len(results))
        return results
    except Exception as e:
        logger.error("Parallel search error: %s", e)
        return []


async def search_web(
    query: str, num_results: int = 5, categories: str = "general", *, session_id: str | None = None,
) -> list[dict]:
    """Return title/url/snippet results from the configured search provider.

    CITEFLOW_SEARCH_PROVIDER defaults to searxng. Parallel supports general web
    search only; SearXNG-specific categories are rejected rather than ignored.
    Provider errors return an empty list, with no cross-provider fallback.
    """
    if num_results <= 0:
        return []
    if SEARCH_PROVIDER == "parallel":
        if categories != "general":
            logger.error("Parallel does not support SearXNG categories; use categories='general'")
            return []
        return await _search_web_parallel(query, num_results, session_id or _PARALLEL_SESSION_ID)
    if SEARCH_PROVIDER == "searxng":
        return await _search_web_searxng(query, num_results, categories)
    logger.error("Unknown CITEFLOW_SEARCH_PROVIDER: %s", SEARCH_PROVIDER)
    return []
