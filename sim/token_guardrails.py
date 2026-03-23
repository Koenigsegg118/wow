#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guardrails and fallback logic for token policy bridge.

Provides configurable safety checks on decoded action chunks and per-tick
actions before they are sent to AFSIM.  Any failed check triggers a fallback
action instead of the model output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class FallbackStrategy(Enum):
    NEUTRAL = "neutral"       # dpsi=0, dalt=0, dspd=0
    HOLD_LAST = "hold_last"   # reuse last valid action


@dataclass
class GuardrailConfig:
    max_abs_dpsi: float = 0.35        # rad (~20 deg per tick)
    max_abs_dalt: float = 400.0       # m
    max_abs_dspd: float = 40.0        # m/s
    conf_threshold: float = 0.35
    chunk_jump_dpsi: float = 0.5      # max inter-step dpsi jump within chunk
    chunk_jump_dalt: float = 600.0
    chunk_jump_dspd: float = 60.0
    fallback: FallbackStrategy = FallbackStrategy.HOLD_LAST


NEUTRAL_ACTION = np.zeros(3, dtype=np.float32)


@dataclass
class GuardrailResult:
    passed: bool
    reason: Optional[str] = None
    details: Optional[str] = None


class TokenGuardrails:
    """Stateless checker — call ``check_chunk`` after decoding, then
    ``check_tick`` on each per-tick action."""

    def __init__(self, cfg: GuardrailConfig | None = None):
        self.cfg = cfg or GuardrailConfig()

    # ── chunk-level checks (called once per new token) ────────

    def check_chunk(
        self,
        chunk: np.ndarray,
        confidence: float,
    ) -> GuardrailResult:
        """Validate a full [token_steps, 3] action chunk.

        Checks (in order):
        1. NaN / Inf in chunk or confidence
        2. Confidence below threshold
        3. Chunk smoothness (inter-step jumps)
        """
        if not np.isfinite(chunk).all() or not math.isfinite(confidence):
            return GuardrailResult(False, "nan_inf",
                                   "chunk or confidence contains NaN/Inf")

        if confidence < self.cfg.conf_threshold:
            return GuardrailResult(
                False, "low_confidence",
                f"confidence={confidence:.4f} < {self.cfg.conf_threshold}")

        for i in range(1, chunk.shape[0]):
            d = np.abs(chunk[i] - chunk[i - 1])
            if d[0] > self.cfg.chunk_jump_dpsi:
                return GuardrailResult(
                    False, "chunk_jump",
                    f"step {i-1}→{i} dpsi jump={d[0]:.4f}")
            if d[1] > self.cfg.chunk_jump_dalt:
                return GuardrailResult(
                    False, "chunk_jump",
                    f"step {i-1}→{i} dalt jump={d[1]:.1f}")
            if d[2] > self.cfg.chunk_jump_dspd:
                return GuardrailResult(
                    False, "chunk_jump",
                    f"step {i-1}→{i} dspd jump={d[2]:.2f}")

        return GuardrailResult(True)

    # ── tick-level checks (called every 0.5 s) ────────────────

    def check_tick(self, action: np.ndarray) -> GuardrailResult:
        """Validate a single per-tick action [dpsi, dalt, dspd]."""
        if not np.isfinite(action).all():
            return GuardrailResult(False, "nan_inf",
                                   "tick action contains NaN/Inf")

        dpsi, dalt, dspd = float(action[0]), float(action[1]), float(action[2])

        if abs(dpsi) > self.cfg.max_abs_dpsi:
            return GuardrailResult(
                False, "magnitude",
                f"|dpsi|={abs(dpsi):.4f} > {self.cfg.max_abs_dpsi}")
        if abs(dalt) > self.cfg.max_abs_dalt:
            return GuardrailResult(
                False, "magnitude",
                f"|dalt|={abs(dalt):.1f} > {self.cfg.max_abs_dalt}")
        if abs(dspd) > self.cfg.max_abs_dspd:
            return GuardrailResult(
                False, "magnitude",
                f"|dspd|={abs(dspd):.2f} > {self.cfg.max_abs_dspd}")

        return GuardrailResult(True)

    # ── fallback resolution ───────────────────────────────────

    def resolve_fallback(
        self,
        last_valid: np.ndarray | None,
    ) -> np.ndarray:
        """Return fallback action based on configured strategy."""
        if (self.cfg.fallback == FallbackStrategy.HOLD_LAST
                and last_valid is not None):
            return last_valid.copy()
        return NEUTRAL_ACTION.copy()
