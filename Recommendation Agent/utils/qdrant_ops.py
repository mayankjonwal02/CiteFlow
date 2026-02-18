"""
Qdrant Vector Database Operations
Manages collections, insertion, and semantic search in Qdrant.
"""

import os
import uuid
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.embeddings import get_embedding, get_embeddings_batch, EMBEDDING_DIMENSION

logger = logging.getLogger("citeflow.qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant-digitrix:6333")

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create a singleton Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, timeout=30)
    return _client


def ensure_collection(collection_name: str) -> bool:
    """
    Ensure a Qdrant collection exists. Create if it doesn't.

    Args:
        collection_name: Name of the collection.

    Returns:
        True if collection exists or was created.
    """
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        existing = [c.name for c in collections]

        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            logger.debug(f"Collection already exists: {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Error ensuring collection '{collection_name}': {e}")
        return False


async def store_documents(
    collection_name: str,
    texts: list[str],
    urls: list[str],
    metadata: list[dict] | None = None,
) -> bool:
    """
    Store documents with embeddings in Qdrant.

    Args:
        collection_name: Target collection name.
        texts: List of text content to embed and store.
        urls: List of source URLs corresponding to each text.
        metadata: Optional additional metadata for each document.

    Returns:
        True if storage was successful.
    """
    if not texts:
        return True

    client = get_qdrant_client()
    ensure_collection(collection_name)

    try:
        # Generate embeddings for all texts
        embeddings = await get_embeddings_batch(texts)

        # Prepare points
        points = []
        for i, (text, url, embedding) in enumerate(zip(texts, urls, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": text,
                "url": url,
                "source": url,
            }
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)

        logger.info(f"Stored {len(points)} documents in collection '{collection_name}'")
        return True

    except Exception as e:
        logger.error(f"Error storing documents in '{collection_name}': {e}")
        return False


async def search_qdrant(
    collection_name: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantic search in a Qdrant collection.

    Args:
        collection_name: The collection to search.
        query: The search query text.
        top_k: Number of top results to return.

    Returns:
        List of dicts with keys: text, url, score.
    """
    client = get_qdrant_client()

    try:
        # Ensure collection exists
        collections = client.get_collections().collections
        existing = [c.name for c in collections]
        if collection_name not in existing:
            logger.warning(f"Collection '{collection_name}' does not exist for search")
            return []

        # Generate query embedding
        query_embedding = await get_embedding(query)

        # Search
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in search_results.points:
            results.append({
                "text": point.payload.get("text", ""),
                "url": point.payload.get("url", ""),
                "source": point.payload.get("source", point.payload.get("url", "")),
                "score": point.score,
            })

        logger.info(
            f"Qdrant search in '{collection_name}' returned {len(results)} results for: '{query[:50]}...'"
        )
        return results

    except Exception as e:
        logger.error(f"Qdrant search error in '{collection_name}': {e}")
        return []


def delete_collection(collection_name: str) -> bool:
    """
    Delete a Qdrant collection (for cleanup).

    Args:
        collection_name: Name of the collection to delete.

    Returns:
        True if deletion was successful.
    """
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted Qdrant collection: {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection '{collection_name}': {e}")
        return False
