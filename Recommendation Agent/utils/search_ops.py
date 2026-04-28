"""
Search Operations
Searches the web using Tavily (when TAVILY_API_KEY is set) or a self-hosted SearXNG meta-search engine.
"""

import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger("citeflow.search")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng-digitrix:8080")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


async def _search_web_tavily(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily Search API.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return.

    Returns:
        A list of dicts with keys: title, url, snippet.
    """
    from tavily import AsyncTavilyClient

    try:
        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        response = await client.search(
            query=query,
            max_results=num_results,
            search_depth="advanced",
            topic="general",
        )

        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })

        logger.info(f"Tavily returned {len(results)} results for query: '{query}'")
        return results

    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


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


async def search_web(query: str, num_results: int = 5, categories: str = "general") -> list[dict]:
    """
    Search the web for information.

    Uses Tavily Search when TAVILY_API_KEY is set, otherwise falls back to SearXNG.

    Args:
        query: The search query string.
        num_results: Maximum number of results to return.
        categories: SearXNG search categories (ignored when using Tavily).

    Returns:
        A list of dicts with keys: title, url, snippet.
    """
    if TAVILY_API_KEY:
        return await _search_web_tavily(query, num_results)
    return await _search_web_searxng(query, num_results, categories)
