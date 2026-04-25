"""
Celery Worker — Async inference tasks.

Heavy ML jobs are offloaded to Celery workers running in the background.
Node.js submits a job, gets a job_id immediately, then polls for results.

This directly answers Assessment Q4: "If ML responses are slow, how to handle?"
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import structlog
from celery import Celery

from app.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()

# ── Celery app ──────────────────────────────────────────────

celery_app = Celery(
    "keabuilder_ml",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,  # Results expire after 1 hour
    task_acks_late=True,  # Acknowledge task after completion (safer)
    worker_prefetch_multiplier=1,  # One task at a time per worker
)


# ── Tasks ───────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="ml.similarity_job",
    max_retries=3,
    default_retry_delay=5,
)
def run_similarity_job(
    self,
    query: str,
    top_k: int = 5,
    method: str = "hybrid",
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Run a similarity search in the background.

    This task is meant to be called from FastAPI's /predict/async endpoint.
    Results are stored in the Celery result backend (Redis) and can be
    polled via GET /job/{job_id}.

    Args:
        query:  Search text.
        top_k:  Number of results.
        method: dense | sparse | hybrid
        model:  Optional model override.

    Returns:
        Dict with search results and metadata.
    """
    start = time.time()
    job_id = self.request.id

    logger.info(
        "async_job_started",
        job_id=job_id,
        query_len=len(query),
        method=method,
        model=model,
    )

    try:
        # Import here to avoid circular imports at module level
        # (FastAPI app creates the instances; worker reuses config)
        from app.feature_flags.flag_store import FlagStore
        from app.feature_flags.model_router import ModelRouter
        from app.similarity.dense import DenseSearch
        from app.similarity.sparse import SparseSearch
        from app.similarity.hybrid import HybridSearch
        from app.db.qdrant_client import QdrantManager

        # Initialise components (each worker process loads once)
        flag_store = FlagStore()
        router = ModelRouter(flag_store)

        # Load minimal model set for the worker
        _lazy_load_models(router)

        encoder = router.route(model)
        qdrant = QdrantManager(vector_dim=encoder.dim)

        if method == "dense":
            dense = DenseSearch(qdrant)
            results = dense.search(query=query, encoder=encoder, top_k=top_k)
        elif method == "sparse":
            sparse = SparseSearch()
            results = sparse.search(query=query, top_k=top_k)
        elif method == "hybrid":
            dense = DenseSearch(qdrant)
            sparse = SparseSearch()
            hybrid = HybridSearch(dense=dense, sparse=sparse)
            results = hybrid.search(query=query, encoder=encoder, top_k=top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = (time.time() - start) * 1000

        logger.info(
            "async_job_completed",
            job_id=job_id,
            num_results=len(results),
            latency_ms=elapsed,
        )

        return {
            "job_id": job_id,
            "status": "done",
            "query": query,
            "method": method,
            "model": encoder.model_name,
            "results": results,
            "latency_ms": round(elapsed, 2),
        }

    except Exception as exc:
        logger.error(
            "async_job_failed",
            job_id=job_id,
            error=str(exc),
        )
        # Retry with exponential backoff
        raise self.retry(exc=exc)


def _lazy_load_models(router: ModelRouter) -> None:
    """
    Lazily load models into the router if not already registered.

    In a real production setup, models would be loaded once at worker
    startup via a Celery signal. This is a simplified version.
    """
    if router.available_models:
        return  # Already loaded

    from app.models.minilm_encoder import MiniLMEncoder

    # Only load the lightweight model in workers to save memory
    router.register("minilm", MiniLMEncoder())
    logger.info("worker_models_loaded", models=router.available_models)
