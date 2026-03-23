#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only token policy shadow module.

Designed to be embedded inside ``bc_policy_socket_server.py``.  It receives
the same AFSIM state that the primary controller sees, runs the frozen
token policy chain in parallel, and logs what the token policy *would*
have commanded — without writing anything back to AFSIM.

ACTION SEMANTICS
-----------------
The VQ-VAE chunk [4,3] contains 2s-lookahead DELTAS (not sequential actions).
The 4 rows are overlapping windows describing the same trajectory segment.
Default: only chunk[0] is used (receding-horizon).

The shadow computes what TurnToRelativeHeading command the token bridge
WOULD send, using the corrected horizon-fraction mapping:
    psi_dot_cmd = dpsi / H_action_sec
    turn_deg    = degrees(psi_dot_cmd * dt)

Usage (from bc_policy_socket_server CLI)::

    python -m sim.bc_policy_socket_server ... \\
        --token_shadow \\
        --token_policy_ckpt checkpoints/onestep_token_bc_t4_cb64_h4/best.pt \\
        --token_vq_ckpt    checkpoints/vqvae_clean_t4_cb64/best.pt \\
        --token_vocab      datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json

The primary controller's action frame is **never modified**.
"""

from __future__ import annotations

import datetime
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


EARTH_R = 6378137.0
GRAVITY = 9.81


def _latlon_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    lat0, lon0 = math.radians(lat0_deg), math.radians(lon0_deg)
    x_e = (lon - lon0) * math.cos(0.5 * (lat + lat0)) * EARTH_R
    y_n = (lat - lat0) * EARTH_R
    return x_e, y_n


def _unwrap(curr_rad, prev_rad):
    if prev_rad is None:
        return curr_rad
    d = curr_rad - prev_rad
    if d > math.pi:
        curr_rad -= 2.0 * math.pi
    elif d < -math.pi:
        curr_rad += 2.0 * math.pi
    return curr_rad


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


# ── per-platform shadow state ─────────────────────────────────

@dataclass
class _ShadowPlatformState:
    prev_token: int = -1
    chunk_buffer: Optional[np.ndarray] = None
    chunk_index: int = 4
    ref_lat: Optional[float] = None
    ref_lon: Optional[float] = None
    prev_track_angle_u: Optional[float] = None
    total_predictions: int = 0
    obs_ring: list = field(default_factory=list)


# ── main shadow class ─────────────────────────────────────────

class TokenShadow:
    """Read-only token policy observer.

    Call ``observe()`` once per control tick, passing the same
    ``sim_data`` and ``sim_time`` that the primary controller receives,
    plus the ``primary_action`` that the primary controller decided to
    send.  The shadow logs everything but never returns an action that
    should be sent to AFSIM.
    """

    H_ACTION_SEC = 2.0  # from training contract: 4 steps × 0.5s

    def __init__(
        self,
        *,
        runtime,
        control_indices: list[int],
        log_path: str = "logs/token_shadow.jsonl",
        guardrails=None,
    ):
        self.runtime = runtime
        self.control_indices = control_indices
        self.guardrails = guardrails
        self._history_len: int = getattr(runtime, "obs_hist_len", 4)
        self._prev_sim_time: Optional[float] = None
        self._measured_dt: float = 0.2

        self.states: dict[int, _ShadowPlatformState] = {}
        for idx in control_indices:
            st = _ShadowPlatformState()
            st.prev_token = runtime.bos_token
            self.states[idx] = st

        self._contract_version: str = getattr(
            runtime, "CONTRACT_VERSION", "onestep_h4_current_anchor_ego_v1")

        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._log_path, "w", encoding="utf-8")

        print(f"[SHADOW] Token shadow active, logging to {log_path}",
              flush=True)
        print(f"[SHADOW] Observing indices: {control_indices}", flush=True)
        print(f"[SHADOW] Contract: {self._contract_version}, "
              f"history_len={self._history_len}, "
              f"H_action_sec={self.H_ACTION_SEC}", flush=True)

    # ── observation builder ───────────────────────────────────

    @staticmethod
    def _state_of(sim_data, idx):
        b = 8 * idx
        return {
            "live": float(sim_data[b + 0]),
            "lat": float(sim_data[b + 1]),
            "lon": float(sim_data[b + 2]),
            "alt_m": float(sim_data[b + 3]),
            "velN_mps": float(sim_data[b + 4]),
            "velE_mps": float(sim_data[b + 5]),
            "velD_mps": float(sim_data[b + 6]),
            "heading_deg": float(sim_data[b + 7]),
        }

    @staticmethod
    def _build_obs(st: dict, ps: _ShadowPlatformState) -> np.ndarray:
        lat, lon = st["lat"], st["lon"]
        if ps.ref_lat is None:
            ps.ref_lat, ps.ref_lon = lat, lon

        x_e, y_n = _latlon_to_enu(lat, lon, ps.ref_lat, ps.ref_lon)
        z_u = st["alt_m"]
        vx_e = st["velE_mps"]
        vy_n = st["velN_mps"]
        vz_u = -st["velD_mps"]

        track_raw = math.atan2(vx_e, vy_n)
        track_u = _unwrap(track_raw, ps.prev_track_angle_u)
        ps.prev_track_angle_u = track_u

        ground_speed = math.sqrt(vx_e * vx_e + vy_n * vy_n)

        return np.array(
            [x_e, y_n, z_u, vx_e, vy_n, vz_u, track_u, ground_speed],
            dtype=np.float32,
        )

    # ── history helper ────────────────────────────────────────

    def _build_obs_hist(self, ps: _ShadowPlatformState) -> np.ndarray:
        H = self._history_len
        ring = ps.obs_ring
        n = len(ring)
        if n >= H:
            return np.stack(ring[-H:], axis=0)
        pad_obs = ring[0]
        rows = [pad_obs] * (H - n) + list(ring)
        return np.stack(rows, axis=0)

    # ── main entry point ──────────────────────────────────────

    def observe(
        self,
        sim_data: np.ndarray,
        sim_time: float,
        primary_action: np.ndarray,
    ) -> None:
        """Observe one tick.  Does NOT modify ``primary_action``."""
        dt = self._measured_dt
        if self._prev_sim_time is not None and sim_time > self._prev_sim_time:
            dt = sim_time - self._prev_sim_time
            self._measured_dt = dt
        self._prev_sim_time = sim_time

        for idx in self.control_indices:
            if 8 * (idx + 1) > len(sim_data):
                continue
            raw_st = self._state_of(sim_data, idx)
            if raw_st["live"] <= 0:
                continue

            ps = self.states[idx]
            obs = self._build_obs(raw_st, ps)

            current_alt = raw_st["alt_m"]
            current_spd = math.sqrt(
                raw_st["velN_mps"] ** 2 + raw_st["velE_mps"] ** 2)

            is_new_token = False
            dense_token = -1
            raw_token = -1
            confidence = 0.0
            decoded_chunk = None

            if ps.chunk_index >= self.runtime.token_steps:
                ps.obs_ring.append(obs.copy())
                if len(ps.obs_ring) > self._history_len:
                    ps.obs_ring = ps.obs_ring[-self._history_len:]
                obs_hist = self._build_obs_hist(ps)

                dense_token, decoded_chunk, confidence = \
                    self.runtime.predict_and_decode(obs_hist, ps.prev_token)
                raw_token = self.runtime.dense_to_raw[dense_token]
                is_new_token = True
                ps.total_predictions += 1
                ps.chunk_buffer = decoded_chunk.copy()
                ps.prev_token = dense_token
                ps.chunk_index = 0

            # receding-horizon: always use chunk[0]
            tick_action = ps.chunk_buffer[0].copy()
            ps.chunk_index += 1

            dpsi_raw = float(tick_action[0])
            dalt_raw = float(tick_action[1])
            dspd_raw = float(tick_action[2])

            # corrected horizon-fraction mapping
            psi_dot = dpsi_raw / self.H_ACTION_SEC
            suggested_turn_deg = math.degrees(psi_dot * dt)
            suggested_alt = _clamp(current_alt + dalt_raw, 200.0, 20000.0)
            suggested_spd = _clamp(current_spd + dspd_raw, 120.0, 650.0)

            n_req = math.sqrt(1.0 + (current_spd * abs(psi_dot) / GRAVITY) ** 2)
            n_cmd = min(n_req, 7.0)

            # primary controller's action
            base = 8 * idx
            actual_turn = float(primary_action[base + 0])
            actual_alt = float(primary_action[base + 1])
            actual_spd = float(primary_action[base + 2])

            delta_turn = suggested_turn_deg - actual_turn
            delta_alt = suggested_alt - actual_alt
            delta_spd = suggested_spd - actual_spd

            guardrail_would_trigger = False
            guardrail_reason = None
            if self.guardrails is not None:
                if is_new_token:
                    cr = self.guardrails.check_chunk(decoded_chunk, confidence)
                    if not cr.passed:
                        guardrail_would_trigger = True
                        guardrail_reason = cr.reason
                if not guardrail_would_trigger:
                    tr = self.guardrails.check_tick(tick_action)
                    if not tr.passed:
                        guardrail_would_trigger = True
                        guardrail_reason = tr.reason

            # ── log ───────────────────────────────────────────
            history_ready = len(ps.obs_ring) >= self._history_len
            record = {
                "contract_version": self._contract_version,
                "sim_time": sim_time,
                "platform_idx": idx,
                "chunk_index": ps.chunk_index - 1,
                "exec_mode": "first_row",
                "H_action_sec": self.H_ACTION_SEC,
                "dt_measured": round(dt, 6),
                "history_len": self._history_len,
                "history_ready": history_ready,
                "history_count": len(ps.obs_ring),
                "psi_dot_cmd_deg_s": round(math.degrees(psi_dot), 4),
                "n_req": round(n_req, 4),
                "n_cmd": round(n_cmd, 4),
            }

            if is_new_token:
                record.update({
                    "event": "new_token",
                    "ts": datetime.datetime.now().isoformat(),
                    "is_token_boundary": True,
                    "token_decision_index": ps.total_predictions,
                    "obs_hist_shape": [self._history_len, 8],
                    "obs_anchor": obs,
                    "prev_token_in": ps.prev_token,
                    "predicted_dense_token": dense_token,
                    "predicted_raw_token": raw_token,
                    "confidence": round(confidence, 6),
                    "decoded_chunk": decoded_chunk,
                })
            else:
                record.update({
                    "event": "tick",
                    "is_token_boundary": False,
                })

            record.update({
                "token_suggested_action": [
                    suggested_turn_deg, suggested_alt, suggested_spd],
                "primary_actual_action": [
                    actual_turn, actual_alt, actual_spd],
                "delta_turn_deg": round(delta_turn, 4),
                "delta_alt_m": round(delta_alt, 2),
                "delta_spd_mps": round(delta_spd, 3),
                "turn_rate_cmd_deg_s": round(math.degrees(psi_dot), 4),
                "token_action_3d": [dpsi_raw, dalt_raw, dspd_raw],
                "guardrail_would_trigger": guardrail_would_trigger,
                "guardrail_reason": guardrail_reason,
            })

            self._fh.write(json.dumps(record, default=_json_default) + "\n")

        self._fh.flush()

    # ── lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        self._fh.close()
        for idx, ps in self.states.items():
            print(f"[SHADOW] idx={idx}: predictions={ps.total_predictions}",
                  flush=True)

    def summary(self) -> dict:
        return {
            idx: {"predictions": ps.total_predictions}
            for idx, ps in self.states.items()
        }
