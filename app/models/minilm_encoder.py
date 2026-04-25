"""
MiniLM Encoder — all-MiniLM-L6-v2.

A balanced mid-size model. Good accuracy-to-speed ratio.
Useful as a baseline and for A/B comparison against the student.

384-dimensional embeddings, ~80 MB.
"""

from __future__ import annotations

import structlog
from typing import List

from sentence_transformers import SentenceTransformer

from app.models.base_encoder import BaseEncoder
from app.config import get_settings

logger = structlog.get_logger(__name__)


class MiniLMEncoder(BaseEncoder):
    """Wraps sentence-transformers all-MiniLM-L6-v2."""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        settings = get_settings()
        self._model_name = model_name or settings.minilm_model_name
        self._device = device
        self._dim = settings.minilm_dim

        logger.info(
            "loading_minilm_model",
            model=self._model_name,
            device=self._device,
        )
        self._model = SentenceTransformer(self._model_name, device=self._device)
        logger.info("minilm_model_loaded", model=self._model_name)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, text: str) -> List[float]:
        """Encode a single text using the MiniLM model."""
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
        internal batching efficiently.
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return embeddings.tolist()
