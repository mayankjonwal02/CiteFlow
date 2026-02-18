"""
OpenAI Embeddings
Generates vector embeddings for text using OpenAI's embedding models.
"""

import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger("citeflow.embeddings")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    """Get or create a singleton AsyncOpenAI client."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    client = get_openai_client()

    # Clean and truncate text
    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * EMBEDDING_DIMENSION

    # Truncate to ~8000 tokens worth of text (~32000 chars)
    max_chars = 32000
    if len(text) > max_chars:
        text = text[:max_chars]

    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding for text ({len(text)} chars)")
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        return [0.0] * EMBEDDING_DIMENSION


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors.
    """
    client = get_openai_client()

    # Clean texts
    cleaned = []
    for t in texts:
        t = t.replace("\n", " ").strip()
        if not t:
            t = "empty"
        if len(t) > 32000:
            t = t[:32000]
        cleaned.append(t)

    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=cleaned,
        )
        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Generated {len(embeddings)} embeddings in batch")
        return embeddings
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]
