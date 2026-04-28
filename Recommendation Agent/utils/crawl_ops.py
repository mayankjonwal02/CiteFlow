"""
Web Scraping Operations
Extracts content from web pages using Tavily Extract (preferred) or Firecrawl API (fallback).
"""

import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger("citeflow.crawl")

FIRECRAWLER_URL = os.getenv("FIRECRAWLER_URL", "http://firecrawl-api-digitrix:3002")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def _truncate_content(content: str, url: str) -> str:
    """Truncate content to avoid token limits and log the result."""
    max_chars = 8000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Content truncated...]"
    logger.info(f"Successfully scraped {url} ({len(content)} chars)")
    return content


async def _scrape_url_tavily(url: str) -> Optional[str]:
    """
    Attempt to scrape a URL using Tavily Extract.

    Returns:
        Extracted content as a string, or None on failure/empty result.
    """
    from tavily import AsyncTavilyClient

    try:
        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        response = await client.extract(urls=[url])
        results = response.get("results", [])
        if results and results[0].get("raw_content"):
            return results[0]["raw_content"]
        logger.warning(f"Tavily Extract returned no content for {url}")
        return None
    except Exception as e:
        logger.error(f"Tavily Extract error for {url}: {e}")
        return None


async def _scrape_url_firecrawl(url: str) -> Optional[str]:
    """
    Scrape a URL using the self-hosted Firecrawl HTTP API.

    Returns:
        Extracted markdown content, or None on failure.
    """
    scrape_endpoint = f"{FIRECRAWLER_URL}/v1/scrape"
    payload = {
        "url": url,
        "formats": ["markdown"],
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(scrape_endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        if data.get("success"):
            content = data.get("data", {}).get("markdown", "")
            if content:
                return content
            else:
                logger.warning(f"No markdown content extracted from {url}")
                return None
        else:
            logger.warning(f"Firecrawl scrape unsuccessful for {url}: {data}")
            return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Firecrawl HTTP error for {url}: {e.response.status_code}")
        return None
    except httpx.ConnectError:
        logger.error(f"Cannot connect to Firecrawl at {FIRECRAWLER_URL}")
        return None
    except Exception as e:
        logger.error(f"Firecrawl scrape error for {url}: {e}")
        return None


async def scrape_url(url: str) -> Optional[str]:
    """
    Scrape a web page and extract its main content.

    When TAVILY_API_KEY is set, Tavily Extract is tried first.
    Falls back to the self-hosted Firecrawl API on failure or empty result.

    Args:
        url: The URL to scrape.

    Returns:
        Extracted markdown/text content, or None if scraping fails.
    """
    content = None

    if TAVILY_API_KEY:
        content = await _scrape_url_tavily(url)

    if not content:
        content = await _scrape_url_firecrawl(url)

    if content:
        return _truncate_content(content, url)

    return None


async def scrape_multiple_urls(urls: list[str], max_concurrent: int = 3) -> dict[str, Optional[str]]:
    """
    Scrape multiple URLs concurrently.

    Args:
        urls: List of URLs to scrape.
        max_concurrent: Maximum number of concurrent scrapes.

    Returns:
        Dict mapping URL to extracted content (or None if failed).
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def _scrape_with_semaphore(url: str):
        async with semaphore:
            results[url] = await scrape_url(url)

    tasks = [_scrape_with_semaphore(url) for url in urls]
    await asyncio.gather(*tasks, return_exceptions=True)

    return results
