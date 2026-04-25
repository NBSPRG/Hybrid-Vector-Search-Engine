"""
Model Router — Routes requests to the correct encoder.

Simple routing: returns the model set as default_model in the flag store,
or the per-request override if specified. No traffic splitting.
"""

from __future__ import annotations

from typing import Dict, Optional

import structlog

from app.models.base_encoder import BaseEncoder
from app.feature_flags.flag_store import FlagStore, ModelStatus

logger = structlog.get_logger(__name__)


class ModelRouter:
    """
    Resolves which encoder instance should handle a request.

    Uses FlagStore to determine the default model, and supports
    per-request overrides via the `model` query parameter.
    """

    def __init__(self, flag_store: FlagStore):
        self._flag_store = flag_store
        self._registry: Dict[str, BaseEncoder] = {}

    # ── Registry management ─────────────────────────────────

    def register(self, name: str, encoder: BaseEncoder) -> None:
        """Register a loaded encoder under a name (e.g. 'student')."""
        self._registry[name] = encoder
        logger.info("encoder_registered", name=name, encoder=repr(encoder))

    def get_encoder(self, name: str) -> Optional[BaseEncoder]:
        """Get an encoder by name from the registry."""
        return self._registry.get(name)

    @property
    def available_models(self) -> list[str]:
        """Return names of all loaded models."""
        return list(self._registry.keys())

    # ── Routing ─────────────────────────────────────────────

    def route(self, request_model: Optional[str] = None) -> BaseEncoder:
        """
        Return the correct encoder for this request.

        Priority:
          1. Per-request override (request_model param)
          2. Default model from flag store

        Args:
            request_model: Optional per-request model override.

        Returns:
            The selected BaseEncoder instance.

        Raises:
            ValueError: If the model is not available.
        """
        # Determine which model to use
        if request_model:
            selected_name = request_model
            logger.debug("per_request_model_selected", model=selected_name)
        else:
            selected_name = self._flag_store.get_active_model()
            logger.debug("default_model_selected", model=selected_name)

        # Check lifecycle status
        status = self._flag_store.get_model_status(selected_name)
        if status == ModelStatus.DEPRECATED:
            logger.warning("deprecated_model_requested", model=selected_name)
            raise ValueError(
                f"Model '{selected_name}' is deprecated. "
                f"Migrate to '{self._flag_store.get_active_model()}'."
            )

        # Look up the encoder
        encoder = self._registry.get(selected_name)
        if encoder is None:
            logger.error(
                "encoder_not_in_registry",
                model=selected_name,
                available=self.available_models,
            )
            raise ValueError(
                f"Model '{selected_name}' is not loaded. "
                f"Available: {self.available_models}"
            )

        logger.info(
            "request_routed",
            model=selected_name,
            status=status,
            dim=encoder.dim,
        )
        return encoder
