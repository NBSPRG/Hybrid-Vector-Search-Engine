"""
KeaBuilder ML Service — FastAPI Application.

Production-grade ML microservice that integrates with KeaBuilder's
Node.js backend via HTTP. Serves as a sidecar service.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings, ModelName
from app.schemas.api_models import (
    AsyncJobResponse,
    AsyncPredictRequest,
    EmbedRequest,
    EmbedResponse,
    FlagConfigResponse,
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    MatchResult,
    SimilarityRequest,
    SimilarityResponse,
)
from app.feature_flags.flag_store import FlagStore
from app.feature_flags.model_router import ModelRouter
from app.db.qdrant_client import QdrantManager
from app.similarity.dense import DenseSearch
from app.similarity.sparse import SparseSearch
from app.similarity.hybrid import HybridSearch

logger = structlog.get_logger(__name__)

# ── Global state (initialised in lifespan) ──────────────────

_state: Dict[str, Any] = {}
_start_time: float = 0.0


# ── Lifespan — load models once at startup ──────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all heavy resources (models, DB connections) at startup.
    Release them on shutdown. Models are NOT reloaded per request.
    """
    global _start_time
    _start_time = time.time()

    settings = get_settings()
    logger.info("startup_begin", env=settings.app_env)

    # 1. Feature flag store (Redis or in-memory fallback)
    flag_store = FlagStore()

    # 2. Model router
    router = ModelRouter(flag_store)

    # 3. Load encoder models
    #    (Imports here to avoid slow module-level loading)
    from app.models.minilm_encoder import MiniLMEncoder

    # Always load MiniLM (lightweight, good baseline)
    minilm = MiniLMEncoder()
    router.register("minilm", minilm)

    # Conditionally load heavier models based on environment
    if settings.app_env != "development":
        from app.models.teacher_encoder import TeacherEncoder
        teacher = TeacherEncoder()
        router.register("teacher", teacher)

        from app.models.student_encoder import StudentEncoder
        student = StudentEncoder(teacher_encoder=teacher)
        router.register("student", student)
    else:
        # In dev mode, student uses random projections (no teacher dependency)
        from app.models.student_encoder import StudentEncoder
        student = StudentEncoder()
        router.register("student", student)

    logger.info("models_loaded", models=router.available_models)

    # 4. Vector DB — use the default model's dimension
    default_model = flag_store.get_active_model()
    default_encoder = router.get_encoder(default_model)
    qdrant = QdrantManager(vector_dim=default_encoder.dim)

    # 5. Search engines
    dense_search = DenseSearch(qdrant)
    sparse_search = SparseSearch()
    hybrid_search = HybridSearch(dense=dense_search, sparse=sparse_search)

    # Store in global state
    _state.update(
        {
            "settings": settings,
            "flag_store": flag_store,
            "router": router,
            "qdrant": qdrant,
            "dense": dense_search,
            "sparse": sparse_search,
            "hybrid": hybrid_search,
        }
    )

    logger.info("startup_complete", models=router.available_models)

    yield  # ← App is running

    # Shutdown cleanup
    logger.info("shutdown")


# ── FastAPI app ─────────────────────────────────────────────

app = FastAPI(
    title="KeaBuilder ML Service",
    description=(
        "Production ML microservice for KeaBuilder. "
        "Provides embedding, similarity search, and async inference. "
        "Integrates with Node.js backend via HTTP."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow KeaBuilder Node.js backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request ID + latency tracking ───────────────

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Add request ID and measure latency for every request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.time()

    # Bind request context for structured logging
    structlog.contextvars.bind_contextvars(request_id=request_id)

    response = await call_next(request)

    elapsed_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request_complete",
        method=request.method,
        path=str(request.url.path),
        status=response.status_code,
        latency_ms=round(elapsed_ms, 2),
    )

    structlog.contextvars.unbind_contextvars("request_id")
    return response


# ──────────────────────────────────────────────────────────────
#  ENDPOINTS
# ──────────────────────────────────────────────────────────────


@app.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed_text(request: EmbedRequest):
    """
    Embed text and store in Qdrant vector database.

    Uses the active model (or per-request override) to generate
    an embedding vector, then stores it with optional metadata.
    """
    router: ModelRouter = _state["router"]
    dense: DenseSearch = _state["dense"]

    try:
        encoder = router.route(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    doc_id = dense.index(
        text=request.text,
        encoder=encoder,
        payload=request.payload,
    )

    return EmbedResponse(
        id=doc_id,
        model=encoder.model_name,
        dim=encoder.dim,
    )


@app.post("/similarity", response_model=SimilarityResponse, tags=["Search"])
async def similarity_search(request: SimilarityRequest):
    """
    Find top-k similar inputs using dense, sparse, or hybrid retrieval.

    This is the synchronous endpoint for fast queries (< 200ms).
    For heavy jobs, use POST /predict/async instead.
    """
    router: ModelRouter = _state["router"]
    start = time.time()

    try:
        encoder = router.route(request.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if request.method == "dense":
        results = _state["dense"].search(
            query=request.query,
            encoder=encoder,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    elif request.method == "sparse":
        results = _state["sparse"].search(
            query=request.query,
            top_k=request.top_k,
        )
    elif request.method == "hybrid":
        results = _state["hybrid"].search(
            query=request.query,
            encoder=encoder,
            top_k=request.top_k,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

    elapsed_ms = (time.time() - start) * 1000

    return SimilarityResponse(
        query=request.query,
        method=request.method,
        model=encoder.model_name,
        results=[MatchResult(**r) for r in results],
        latency_ms=round(elapsed_ms, 2),
        total_indexed=_state["qdrant"].count(),
    )


@app.post("/predict/async", response_model=AsyncJobResponse, tags=["Async"])
async def predict_async(request: AsyncPredictRequest):
    """
    Enqueue a heavy ML job for background processing.

    Returns a job_id immediately. Use GET /job/{job_id} to poll results.
    """
    from app.worker.tasks import run_similarity_job

    try:
        task = run_similarity_job.delay(
            query=request.query,
            top_k=request.top_k,
            method=request.method.value,
            model=request.model,
        )

        return AsyncJobResponse(
            job_id=task.id,
            status=JobStatus.PENDING,
        )
    except Exception as exc:
        logger.error("async_enqueue_failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Failed to enqueue job. Is Celery/Redis running?",
        )


@app.get("/job/{job_id}", response_model=JobStatusResponse, tags=["Async"])
async def get_job_status(job_id: str):
    """
    Poll the status of an async job.

    Frontend flow:
      1. POST /predict/async → get job_id
      2. GET /job/{job_id} every 1s
      3. When status=done, render results
    """
    from celery.result import AsyncResult
    from app.worker.tasks import celery_app

    result = AsyncResult(job_id, app=celery_app)

    if result.state == "PENDING":
        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)
    elif result.state == "STARTED":
        return JobStatusResponse(job_id=job_id, status=JobStatus.RUNNING)
    elif result.state == "SUCCESS":
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.DONE,
            result=result.result,
        )
    elif result.state == "FAILURE":
        return JobStatusResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(result.info),
        )
    else:
        return JobStatusResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check():
    """
    Liveness probe for Docker / Kubernetes.

    Returns system status including loaded models and service connectivity.
    """
    router: ModelRouter = _state["router"]
    flag_store: FlagStore = _state["flag_store"]

    # Check Redis connectivity
    redis_ok = False
    try:
        config = flag_store.get_config()
        redis_ok = flag_store._client is not None
    except Exception:
        pass

    # Check Qdrant connectivity
    qdrant_ok = False
    try:
        _state["qdrant"].count()
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        environment=_state["settings"].app_env,
        models_loaded=router.available_models,
        qdrant_connected=qdrant_ok,
        redis_connected=redis_ok,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/flags", response_model=FlagConfigResponse, tags=["Feature Flags"])
async def get_flags():
    """Return current feature flag configuration."""
    config = _state["flag_store"].get_config()
    return FlagConfigResponse(
        default_model=config.get("default_model", "student"),
        models=config.get("models", {}),
    )


@app.put("/flags/active-model/{model_name}", tags=["Feature Flags"])
async def set_active_model(model_name: str):
    """
    Change the global default model.
    """
    flag_store: FlagStore = _state["flag_store"]
    router: ModelRouter = _state["router"]

    if model_name not in router.available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not loaded. Available: {router.available_models}",
        )

    flag_store.set_active_model(model_name)
    return {"message": f"Active model changed to '{model_name}'", "model": model_name}


# ── Entrypoint ──────────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level,
    )
