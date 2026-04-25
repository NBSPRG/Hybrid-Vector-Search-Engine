"""
Teacher Encoder — all-mpnet-base-v2.

The largest, most accurate model. Used as:
  1. Ground truth during knowledge distillation training
  2. Production model when accuracy matters more than latency
  3. Shadow model for quality comparisons

768-dimensional embeddings, ~420 MB.
"""

from __future__ import annotations

import structlog
from typing import List

from sentence_transformers import SentenceTransformer

from app.models.base_encoder import BaseEncoder
from app.config import get_settings

logger = structlog.get_logger(__name__)


class TeacherEncoder(BaseEncoder):
    """Wraps sentence-transformers all-mpnet-base-v2."""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        settings = get_settings()
        self._model_name = model_name or settings.teacher_model_name
        self._device = device
        self._dim = settings.teacher_dim

        logger.info(
            "loading_teacher_model",
            model=self._model_name,
            device=self._device,
        )
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("teacher_model_loaded", model=self._model_name)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, text: str) -> List[float]:
        """Encode a single text using the teacher model."""
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Batched encoding — sentence-transformers handles
        internal batching and GPU parallelism efficiently.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()
