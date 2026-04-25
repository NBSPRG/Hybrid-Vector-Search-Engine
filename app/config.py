"""
KeaBuilder ML Service — Configuration.

All settings loaded from environment variables via Pydantic Settings.
No hardcoded secrets. See .env.example for defaults.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field


class ModelName(str, Enum):
    """Available model identifiers for feature-flag routing."""

    STUDENT = "student"
    MINILM = "minilm"
    TEACHER = "teacher"


class SimilarityMethod(str, Enum):
    """Retrieval strategy selector."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """
    Central configuration for the ML service.

    Values are read from environment variables (or a `.env` file).
    """

    # ── Service ──────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    # ── Active model (global feature flag default) ───────────
    active_model: ModelName = ModelName.STUDENT

    # ── Redis ────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Qdrant ───────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "keabuilder_leads"

    # ── Model paths / names ──────────────────────────────────
    student_model_path: str = "artifacts/student_model.pt"
    teacher_model_name: str = "all-mpnet-base-v2"
    minilm_model_name: str = "all-MiniLM-L6-v2"

    # ── Model dimensions (must match trained weights) ────────
    student_dim: int = 128
    minilm_dim: int = 384
    teacher_dim: int = 768

    # ── Celery ───────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Similarity defaults ──────────────────────────────────
    default_top_k: int = 5
    default_similarity_method: SimilarityMethod = SimilarityMethod.DENSE
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor — settings are parsed once and cached."""
    return Settings()
