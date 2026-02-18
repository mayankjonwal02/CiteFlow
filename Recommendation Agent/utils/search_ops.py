"""
SearXNG Search Operations
Searches the web using a self-hosted SearXNG meta-search engine.
"""

import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger("citeflow.search")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng-digitrix:8080")


async def search_web(query: str, num_results: int = 5, categories: str = "general") -> list[dict]:
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
