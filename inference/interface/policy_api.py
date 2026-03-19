"""Inference runtime protocol definitions."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PolicyBackend(Protocol):
    """Backend contract: consumes an observation sequence and returns one action vector."""

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        """Predict [dpsi_rad, alt_sp_m, spd_sp_mps] from obs sequence [T, D]."""


class PolicyRuntimeAPI(Protocol):
    """Policy runtime contract used by sim-side adapters."""

    seq_len: int

    def predict(self, obs_seq: np.ndarray) -> np.ndarray:
        """Predict one physical action vector from obs sequence."""
