"""
Dense Similarity — Embedding-based vector search.

Wraps Qdrant vector operations for semantic similarity search.
Uses the active encoder (via ModelRouter) to embed queries
and retrieves nearest neighbours from the vector DB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from app.models.base_encoder import BaseEncoder
from app.db.qdrant_client import QdrantManager

logger = structlog.get_logger(__name__)


class DenseSearch:
    """
    Semantic search using dense (embedding) vectors stored in Qdrant.
    """

    def __init__(self, qdrant: QdrantManager):
        self._qdrant = qdrant

    def search(
        self,
        query: str,
        encoder: BaseEncoder,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query and search Qdrant for nearest vectors.

        Args:
            query:           Text to search for.
            encoder:         The encoder to use for embedding.
            top_k:           Number of results to return.
            score_threshold: Minimum similarity score (optional).

        Returns:
            List of dicts with keys: id, score, payload.
        """
        logger.info(
            "dense_search",
            query_len=len(query),
            model=encoder.model_name,
            top_k=top_k,
        )

        # Embed the query using the active model
        query_vector = encoder.encode(query)

        # Search Qdrant
        results = self._qdrant.search(
            vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        logger.info("dense_search_complete", num_results=len(results))
        return results

    def index(
        self,
        text: str,
        encoder: BaseEncoder,
        payload: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Embed text and store in Qdrant.

        Args:
            text:    The text to embed and store.
            encoder: The encoder to use.
            payload: Optional metadata to store alongside the vector.
            doc_id:  Optional document ID; auto-generated if not provided.

        Returns:
            The document ID.
        """
        vector = encoder.encode(text)
        full_payload = {"text": text, **(payload or {})}

        return self._qdrant.upsert(
            vector=vector,
            payload=full_payload,
            doc_id=doc_id,
        )

    def index_batch(
        self,
        texts: List[str],
        encoder: BaseEncoder,
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Batch embed and store multiple texts."""
        vectors = encoder.encode_batch(texts)
        if payloads is None:
            payloads = [{"text": t} for t in texts]
        else:
            payloads = [{**p, "text": t} for t, p in zip(texts, payloads)]

        return self._qdrant.upsert_batch(vectors=vectors, payloads=payloads)
