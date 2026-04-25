"""
Flag Store — Redis-backed feature flag persistence.

Reads and writes model configuration flags from Redis.
Changing a flag value instantly swaps models in production
without redeployment.

Flag schema stored in Redis as a JSON hash under key: "ml:flags"
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, Optional

import redis
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Redis key namespace
CONFIG_KEY = "ml:model_config"


class ModelStatus(str, Enum):
    """Deployment lifecycle stage for a model."""

    ACTIVE = "active"          # Receives production traffic
    AVAILABLE = "available"    # Loaded and ready, but not the default
    SHADOW = "shadow"          # Not loaded in dev mode
    DEPRECATED = "deprecated"  # Still loads, returns 410


# Default flag configuration — student is the default model
DEFAULT_MODEL_CONFIG = {
    "models": {
        "student": {
            "status": ModelStatus.AVAILABLE,
            "dim": 128,
        },
        "minilm": {
            "status": ModelStatus.ACTIVE,
            "dim": 384,
        },
        "teacher": {
            "status": ModelStatus.SHADOW,
            "dim": 768,
        },
    },
    "default_model": "minilm",
}


class FlagStore:
    """
    Reads and writes feature flags from Redis.

    Falls back to in-memory defaults if Redis is unavailable
    (graceful degradation for local development).
    """

    def __init__(self, redis_url: str | None = None):
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._client: Optional[redis.Redis] = None
        self._local_config: Dict[str, Any] = DEFAULT_MODEL_CONFIG.copy()
        self._connect()

    def _connect(self) -> None:
        """Attempt Redis connection; fall back to local config if unavailable."""
        try:
            self._client = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._client.ping()
            logger.info("redis_connected", url=self._redis_url)
            # Initialise flags if they don't exist yet
            if not self._client.exists(CONFIG_KEY):
                self._push_config(DEFAULT_MODEL_CONFIG)
                logger.info("flags_initialised_with_defaults")
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning(
                "redis_unavailable",
                error=str(exc),
                msg="Falling back to in-memory flags",
            )
            self._client = None

    # ── Read ────────────────────────────────────────────────

    def get_active_model(self) -> str:
        """Return the current default model name."""
        config = self.get_config()
        return config.get("default_model", "minilm")

    def get_model_status(self, model_name: str) -> str:
        """Return the lifecycle status of a specific model."""
        config = self.get_config()
        model_info = config.get("models", {}).get(model_name, {})
        return model_info.get("status", ModelStatus.DEPRECATED)

    def get_config(self) -> Dict[str, Any]:
        """Return the full flag configuration."""
        if self._client:
            try:
                raw = self._client.get(CONFIG_KEY)
                if raw:
                    return json.loads(raw)
            except (redis.ConnectionError, json.JSONDecodeError) as exc:
                logger.warning("flag_read_error", error=str(exc))
        return self._local_config

    # ── Write ───────────────────────────────────────────────

    def set_active_model(self, model_name: str) -> None:
        """Change the global default model."""
        config = self.get_config()
        # Set new model as active, old active model as available
        old_active = config.get("default_model")
        if old_active and old_active in config.get("models", {}):
            config["models"][old_active]["status"] = ModelStatus.AVAILABLE
        if model_name in config.get("models", {}):
            config["models"][model_name]["status"] = ModelStatus.ACTIVE
        config["default_model"] = model_name
        self._push_config(config)
        logger.info("active_model_changed", model=model_name)

    def set_model_status(self, model_name: str, status: ModelStatus) -> None:
        """Update the lifecycle status of a model."""
        config = self.get_config()
        if model_name in config.get("models", {}):
            config["models"][model_name]["status"] = status
            self._push_config(config)
            logger.info(
                "model_status_changed",
                model=model_name,
                status=status,
            )

    def _push_config(self, config: Dict[str, Any]) -> None:
        """Write full config to Redis (or update local fallback)."""
        self._local_config = config
        if self._client:
            try:
                self._client.set(CONFIG_KEY, json.dumps(config, default=str))
            except redis.ConnectionError as exc:
                logger.warning("flag_write_error", error=str(exc))
