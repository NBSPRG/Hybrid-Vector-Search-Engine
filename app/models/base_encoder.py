"""
Base Encoder — Abstract Interface.

Every ML model in the system implements this contract.
This is the KEY architectural decision: uniform encode() API
means feature flags can swap models without breaking anything upstream.

Implements: Strategy Pattern
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseEncoder(ABC):
    """
    Abstract base for all text encoders.

    Guarantees:
        - encode()       → single text  → single vector
        - encode_batch() → list of text → list of vectors
        - dim            → embedding dimensionality (e.g. 128, 384, 768)
        - model_name     → human-readable identifier for logging/metrics
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of output embeddings."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a human-readable model identifier."""
        ...

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """
        Encode a single text string into a fixed-size embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats with length == self.dim.
        """
        ...

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts. Default: sequential calls.

        Subclasses SHOULD override this with batched inference
        for better GPU utilisation.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        return [self.encode(t) for t in texts]

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} dim={self.dim} model={self.model_name}>"
