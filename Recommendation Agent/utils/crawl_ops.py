"""
Firecrawl Web Scraping Operations
Extracts content from web pages using Firecrawl API.
"""

import os
import logging
import httpx
from typing import Optional

logger = logging.getLogger("citeflow.crawl")

FIRECRAWLER_URL = os.getenv("FIRECRAWLER_URL", "http://firecrawl-api-digitrix:3002")


async def scrape_url(url: str) -> Optional[str]:
    """
    Scrape a web page and extract its main content using Firecrawl.

    Args:
        url: The URL to scrape.

    Returns:
        Extracted markdown/text content, or None if scraping fails.
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
                # Truncate very long content to avoid token limits
                max_chars = 8000
                if len(content) > max_chars:
                    content = content[:max_chars] + "\n\n[Content truncated...]"
                logger.info(f"Successfully scraped {url} ({len(content)} chars)")
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
