"""
Qdrant Client — Vector database operations.

Wraps the Qdrant Python client for vector CRUD operations.
Gracefully degrades to in-memory mode if Qdrant server is unavailable.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.config import get_settings

logger = structlog.get_logger(__name__)


class QdrantManager:
    """
    Manages vector storage and retrieval in Qdrant.

    Falls back to in-memory Qdrant if the external server is unreachable
    (useful for local development without Docker).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        vector_dim: int | None = None,
    ):
        settings = get_settings()
        self._host = host or settings.qdrant_host
        self._port = port or settings.qdrant_port
        self._collection = collection_name or settings.qdrant_collection
        self._vector_dim = vector_dim or settings.student_dim  # default to active model

        self._client = self._connect()
        self._ensure_collection()

    def _connect(self) -> QdrantClient:
        """Connect to Qdrant server or fall back to in-memory mode."""
        try:
            client = QdrantClient(host=self._host, port=self._port, timeout=5)
            # Test connectivity
            client.get_collections()
            logger.info(
                "qdrant_connected",
                host=self._host,
                port=self._port,
            )
            return client
        except Exception as exc:
            logger.warning(
                "qdrant_unavailable",
                error=str(exc),
                msg="Using in-memory Qdrant for development",
            )
            return QdrantClient(":memory:")

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "collection_created",
                name=self._collection,
                dim=self._vector_dim,
            )

    def recreate_collection(self, vector_dim: int) -> None:
        """
        Drop and recreate the collection with a new vector dimension.

        Useful when switching between models with different embedding sizes.
        """
        self._vector_dim = vector_dim
        self._client.delete_collection(self._collection)
        self._ensure_collection()
        logger.info("collection_recreated", dim=vector_dim)

    # ── Write ───────────────────────────────────────────────

    def upsert(
        self,
        vector: List[float],
        payload: Dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        """
        Insert or update a single vector with payload.

        Returns the document ID.
        """
        point_id = doc_id or str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.debug("vector_upserted", id=point_id)
        return point_id

    def upsert_batch(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        doc_ids: List[str] | None = None,
    ) -> List[str]:
        """Insert or update multiple vectors in a single batch."""
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            PointStruct(id=did, vector=vec, payload=pay)
            for did, vec, pay in zip(doc_ids, vectors, payloads)
        ]

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.info("batch_upserted", count=len(points))
        return doc_ids

    # ── Read ────────────────────────────────────────────────

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest vectors by cosine similarity.

        Returns list of dicts: { id, score, payload }.
        Uses query_points() API (qdrant-client >= 1.10).
        """
        from qdrant_client.models import models

        results = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in results.points
        ]

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        info = self._client.get_collection(self._collection)
        return info.points_count or 0
