"""Numpy fallback backend for smoke/integration without heavyweight runtime deps."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class NumpyLinearBackend:
    """Simple linear backend: action = obs_last @ W + b.

    If model file is absent, fallback policy keeps heading and tracks current altitude/speed.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.weights: np.ndarray | None = None
        self.bias: np.ndarray | None = None

        if self.model_path is None:
            return
        if not self.model_path.exists():
            return

        payload: Any = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.weights = np.asarray(payload.get("weights"), dtype=np.float32)
        self.bias = np.asarray(payload.get("bias"), dtype=np.float32)
        if self.weights.ndim != 2 or self.bias.ndim != 1:
            raise ValueError(f"Invalid numpy_linear model payload in {self.model_path}")

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        obs_seq = np.asarray(obs_seq, dtype=np.float32)
        if obs_seq.ndim != 2:
            raise ValueError(f"obs_seq must be rank-2 [T,D], got shape={obs_seq.shape}")
        last = obs_seq[-1]

        if self.weights is not None and self.bias is not None:
            if last.shape[0] != self.weights.shape[0]:
                raise ValueError(
                    f"obs dim mismatch: obs={last.shape[0]} weights={self.weights.shape[0]}"
                )
            return (last @ self.weights + self.bias).astype(np.float32)

        # Heuristic fallback aligned with action schema.
        vx, vy, vz = float(last[3]), float(last[4]), float(last[5])
        alt = float(last[2])
        spd = math.sqrt(vx * vx + vy * vy + vz * vz)
        return np.asarray([0.0, alt, max(spd, 120.0)], dtype=np.float32)
