"""
API Models — Pydantic request/response schemas.

Every endpoint uses typed schemas — no raw dicts.
This provides automatic validation, serialization,
and OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────


class SimilarityMethodEnum(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ── Request Schemas ──────────────────────────────────────────


class EmbedRequest(BaseModel):
    """Request to embed and store text."""

    text: str = Field(..., min_length=1, max_length=10_000, description="Text to embed")
    model: Optional[str] = Field(
        None, description="Model override (student/minilm/teacher). Uses default if omitted."
    )
    payload: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata to store with the vector"
    )


class EmbedBatchRequest(BaseModel):
    """Batch embed request."""

    texts: List[str] = Field(..., min_length=1, max_length=100)
    model: Optional[str] = None
    payloads: Optional[List[Dict[str, Any]]] = None


class SimilarityRequest(BaseModel):
    """Request for similarity search."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    method: SimilarityMethodEnum = Field(
        SimilarityMethodEnum.DENSE, description="Retrieval strategy"
    )
    model: Optional[str] = Field(
        None, description="Model override for dense/hybrid search"
    )
    score_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class AsyncPredictRequest(BaseModel):
    """Request for async heavy inference job."""

    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(5, ge=1, le=100)
    method: SimilarityMethodEnum = SimilarityMethodEnum.HYBRID
    model: Optional[str] = None


# ── Response Schemas ─────────────────────────────────────────


class MatchResult(BaseModel):
    """A single similarity match."""

    id: str
    score: float
    payload: Dict[str, Any] = Field(default_factory=dict)
    rank: Optional[int] = None
    # Hybrid-specific fields
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None


class EmbedResponse(BaseModel):
    """Response after embedding text."""

    id: str
    model: str
    dim: int
    vector: Optional[List[float]] = Field(
        None, description="Returned only if include_vector=true"
    )


class SimilarityResponse(BaseModel):
    """Response for similarity search."""

    query: str
    method: str
    model: str
    results: List[MatchResult]
    latency_ms: float
    total_indexed: int = 0


class AsyncJobResponse(BaseModel):
    """Response when an async job is enqueued."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    message: str = "Job enqueued. Poll GET /job/{job_id} for results."


class JobStatusResponse(BaseModel):
    """Response when polling a job's status."""

    job_id: str
    status: JobStatus
    result: Optional[SimilarityResponse] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str = "ok"
    environment: str
    models_loaded: List[str]
    qdrant_connected: bool
    redis_connected: bool
    uptime_seconds: float


class FlagConfigResponse(BaseModel):
    """Current feature flag configuration."""

    default_model: str
    models: Dict[str, Dict[str, Any]]
