"""
Student Encoder — Distilled MLP with residual connections.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import structlog
import torch
import torch.nn as nn

from app.models.base_encoder import BaseEncoder
from app.config import get_settings

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────
#  Student MLP — must match training script exactly
# ──────────────────────────────────────────────────────────────

class StudentMLP(nn.Module):
    """
    Improved student: 768 → 512 → 256 → 128 with residual connection.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )
        # Residual projection: 512 → 256 for skip connection
        self.skip_proj = nn.Linear(512, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(x)           # 768 → 512
        h2 = self.layer2(h1)          # 512 → 256
        h2 = h2 + self.skip_proj(h1)  # residual connection
        out = self.layer3(h2)         # 256 → 128
        return out


# ──────────────────────────────────────────────────────────────
#  Encoder wrapper (implements BaseEncoder contract)
# ──────────────────────────────────────────────────────────────

class StudentEncoder(BaseEncoder):
    """
    Loads the distilled student_model.pt and exposes the standard encode() interface.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "cpu",
        teacher_encoder=None,
    ):
        settings = get_settings()
        self._model_path = model_path or settings.student_model_path
        self._device = device
        self._dim = settings.student_dim
        self._teacher = teacher_encoder  # optional — for full pipeline

        # Initialise model with matching architecture
        self._model = StudentMLP(
            input_dim=settings.teacher_dim,
            output_dim=settings.student_dim,
        )

        if Path(self._model_path).exists():
            logger.info("loading_student_model", path=self._model_path)
            state = torch.load(
                self._model_path,
                map_location=self._device,
                weights_only=False,  # checkpoint has numpy metadata from Colab
            )
            # Support both raw state_dict and wrapped checkpoint
            if isinstance(state, dict) and "model_state_dict" in state:
                self._model.load_state_dict(state["model_state_dict"])
            else:
                self._model.load_state_dict(state)
            logger.info("student_model_loaded", path=self._model_path)
        else:
            logger.warning(
                "student_model_not_found",
                path=self._model_path,
                msg="Using randomly initialised weights — train the model first!",
            )

        self._model.to(self._device)
        self._model.eval()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "student-mlp"

    def encode(self, text: str) -> List[float]:
        """
        Encode text → 128-dim student embedding.

        Pipeline: text → teacher.encode() → StudentMLP → 128-dim
        """
        if self._teacher is None:
            # Fallback: random projection (for testing without teacher)
            logger.debug("student_encode_no_teacher", text_len=len(text))
            dummy_input = torch.randn(1, 768, device=self._device)
        else:
            teacher_emb = self._teacher.encode(text)
            dummy_input = torch.tensor(
                [teacher_emb], dtype=torch.float32, device=self._device
            )

        with torch.no_grad():
            output = self._model(dummy_input)
            # L2 normalise for cosine similarity
            output = torch.nn.functional.normalize(output, p=2, dim=-1)

        return output.squeeze(0).cpu().tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Batched encoding through teacher → student pipeline."""
        if self._teacher is None:
            return [self.encode(t) for t in texts]

        # Batch through teacher first
        teacher_embs = self._teacher.encode_batch(texts)
        teacher_tensor = torch.tensor(
            teacher_embs, dtype=torch.float32, device=self._device
        )

        with torch.no_grad():
            outputs = self._model(teacher_tensor)
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)

        return outputs.cpu().tolist()
