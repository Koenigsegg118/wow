#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone token policy bridge server for AFSIM.

Replaces the continuous BC policy with the frozen discrete token policy
chain (OneStepTokenBC -> VQ-VAE decode -> 4-step action chunk) while
keeping the existing AFSIM socket protocol unchanged.

AFSIM SCENARIOS (TCP client -> this server, default localhost:65432)
------------------------------------------------------------------
Same SAC_PROCESSOR socket protocol as ``bc_policy_socket_server``.
Typical scenes under ``build/demos/air_to_air/``:

- ``scenarios/2v2_p6dof_token.txt`` (header documents this bridge; may need
  a top-level entry with ``SCNRIO=2v2_p6dof_token``).
- ``2v2_model_only.txt`` -> ``scenarios/2v2_p6dof_model_only.txt``
- ``2v2_kinematic_model_only.txt`` -> ``scenarios/2v2_kinematic_model_only.txt``
- ``2v2_bc_close.txt`` / ``2v2_tspi_blue.txt`` when control_indices match
  ``mControlledPlatforms``.

See ``docs/MIGRATION_HANDOFF.md`` §4.1 for the full mapping table.

ACTION SEMANTICS (from training contract)
------------------------------------------
The VQ-VAE chunk [4, 3] contains [dpsi_rad, dalt_m, dspd_mps].
ALL THREE are 2s-lookahead **DELTAS**, NOT absolute setpoints:

    dpsi_rad[i]  = heading[t+i+K]  - heading[t+i]     (K=4, dt_data=0.5s -> 2.0s)
    dalt_m[i]    = altitude[t+i+K] - altitude[t+i]
    dspd_mps[i]  = speed[t+i+K]    - speed[t+i]

The 4 rows are overlapping 2s-lookahead windows from consecutive steps,
describing the SAME trajectory segment.  They are NOT 4 independent
sequential actions.

EXECUTION MODES
---------------
first_row (DEFAULT):  Receding-horizon — use only chunk[0].
    Most semantically correct: the policy predicts at the current state,
    and chunk[0] is the action for the current decision point.
    Rows 1-3 are anchored at future states the aircraft hasn't reached yet.

reduce_mean:  Average all 4 rows into a single action.
    Treats the 4 overlapping lookahead descriptions as redundant
    views of the same trajectory and averages them.

reduce_median:  Median of all 4 rows.
    More robust to outlier rows than mean.

replay_4rows:  Legacy mode — sequentially play chunk[0..3].
    ONLY for A/B ablation experiments.  Semantically incorrect because
    rows 1-3 are lookahead labels for future states, not actions to
    execute at the current state.

CONTROL MAPPING
---------------
dpsi_rad is converted to a per-tick heading command:
    psi_dot_cmd = dpsi / H_action_sec                  (rad/s)
    turn_deg    = degrees(psi_dot_cmd * dt)             (per-tick command)

dalt/dspd set absolute targets at the token boundary and are latched:
    alt_target  = current_alt_at_boundary + dalt_m
    spd_target  = current_spd_at_boundary + dspd_mps

G-LOAD MODES
-------------
fixed:    TurnToRelativeHeading(turn_deg, G_FIXED * 9.81)
dynamic:  n_req = sqrt(1 + (V * psi_dot_cmd / g)^2)
          n_cmd = clamp(n_req, 1.0, G_MAX_CFG)
          TurnToRelativeHeading(turn_deg, n_cmd * 9.81)

LATCH MECHANISM
---------------
Between token boundaries, alt/spd setpoints are held constant.
Heading uses per-tick fractions (inherently non-latchable for relative commands).
Optional thresholds control when a new token's intent overwrites the latch:
    heading_update_thresh_deg, alt_update_thresh_m, speed_update_thresh_mps
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import socket
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from sim.token_guardrails import (
    FallbackStrategy,
    GuardrailConfig,
    GuardrailResult,
    TokenGuardrails,
)

EARTH_R = 6378137.0
GRAVITY = 9.81


class Mode(Enum):
    DRYRUN = "dryrun"
    LIMITED = "limited"
    FULL = "full"


class ExecMode(Enum):
    FIRST_ROW = "first_row"
    REDUCE_MEAN = "reduce_mean"
    REDUCE_MEDIAN = "reduce_median"
    REPLAY_4ROWS = "replay_4rows"


class GMode(Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"


# ── reuse the AFSIM socket protocol from bc_policy_socket_server ──

class StateReceiver:
    """Parse AFSIM text frames: ``simTime 640 v0..v639``."""

    def __init__(self):
        self._text_buf = ""
        self._tokens: list[str] = []

    def _feed(self, conn) -> None:
        data = conn.recv(8192)
        if not data:
            raise EOFError("socket closed")
        s = data.decode("ascii", errors="ignore")
        self._text_buf += s
        if self._text_buf and (not self._text_buf[-1].isspace()):
            parts = self._text_buf.split()
            if parts:
                self._text_buf = parts[-1]
                self._tokens.extend(parts[:-1])
            return
        parts = self._text_buf.split()
        self._text_buf = ""
        self._tokens.extend(parts)

    def recv_frame(self, conn):
        while True:
            while len(self._tokens) < 2:
                self._feed(conn)
            state_len = int(float(self._tokens[1]))
            need = 2 + state_len
            while len(self._tokens) < need:
                self._feed(conn)
            frame = self._tokens[:need]
            del self._tokens[:need]
            sim_time = float(frame[0])
            vals = np.asarray(frame[2: 2 + state_len], dtype=np.float32)
            return sim_time, vals


def send_status_data(conn, action_640: np.ndarray) -> None:
    action_640 = np.asarray(action_640, dtype=np.float32)
    assert action_640.size == 640
    conn.sendall(b"STATUS")
    conn.sendall(action_640.astype("<f4", copy=False).tobytes())


# ── helpers ───────────────────────────────────────────────────

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


def log(msg: str) -> None:
    print(msg, flush=True)


def _reduce_chunk(chunk: np.ndarray, exec_mode: ExecMode) -> np.ndarray:
    """Reduce a [4,3] chunk to a single [3] action vector."""
    if exec_mode == ExecMode.FIRST_ROW:
        return chunk[0].copy()
    elif exec_mode == ExecMode.REDUCE_MEAN:
        return chunk.mean(axis=0).copy()
    elif exec_mode == ExecMode.REDUCE_MEDIAN:
        return np.median(chunk, axis=0).copy()
    elif exec_mode == ExecMode.REPLAY_4ROWS:
        raise ValueError("_reduce_chunk should not be called for replay_4rows")
    raise ValueError(f"Unknown exec_mode: {exec_mode}")


def compute_dynamic_g(speed_mps: float, psi_dot_rad_s: float,
                      g_max: float) -> tuple[float, float]:
    """Compute required and commanded G-load for a coordinated turn.

    Returns (n_req, n_cmd) where n_cmd = clamp(n_req, 1.0, g_max).
    """
    n_req = math.sqrt(1.0 + (speed_mps * psi_dot_rad_s / GRAVITY) ** 2)
    n_cmd = _clamp(n_req, 1.0, g_max)
    return n_req, n_cmd


def max_turn_rate_at_g(speed_mps: float, n_load: float) -> float:
    """Theoretical maximum steady-state turn rate (rad/s) at given G and speed."""
    if n_load <= 1.0 or speed_mps <= 1.0:
        return 0.0
    return GRAVITY * math.sqrt(n_load ** 2 - 1.0) / speed_mps


# ── per-platform state ────────────────────────────────────────

@dataclass
class TokenPlatformState:
    prev_token: int = -1
    chunk_buffer: Optional[np.ndarray] = None   # [4, 3] raw decoded chunk
    chunk_index: int = 4
    last_valid_action: Optional[np.ndarray] = None
    ref_lat: Optional[float] = None
    ref_lon: Optional[float] = None
    prev_track_angle_u: Optional[float] = None
    total_predictions: int = 0
    total_fallbacks: int = 0
    obs_ring: list = field(default_factory=list)

    # latched setpoints (computed at token boundary, held between)
    latched_alt_target: Optional[float] = None
    latched_spd_target: Optional[float] = None
    latched_dpsi_rad: float = 0.0    # the dpsi used for per-tick heading commands
    latched_psi_dot_rad_s: float = 0.0
    latched_n_cmd: float = 1.0
    latch_is_active: bool = False

    # per-channel telemetry counters
    heading_task_reset_count: int = 0
    alt_task_reset_count: int = 0
    spd_task_reset_count: int = 0
    heading_latch_hold_count: int = 0
    alt_latch_hold_count: int = 0
    spd_latch_hold_count: int = 0
    g_saturation_count: int = 0
    prev_heading_deg: float = 0.0


# ── structured logger ─────────────────────────────────────────

class StructuredLogger:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


# ── main server ───────────────────────────────────────────────

class TokenBridgeServer:

    def __init__(
        self,
        *,
        runtime,
        guardrails,
        mode: Mode = Mode.DRYRUN,
        exec_mode: ExecMode = ExecMode.FIRST_ROW,
        g_mode: GMode = GMode.DYNAMIC,
        g_fixed: float = 3.0,
        g_max: float = 7.0,
        h_action_sec: float = 2.0,
        latch_enabled: bool = True,
        heading_update_thresh_deg: float = 3.0,
        alt_update_thresh_m: float = 50.0,
        speed_update_thresh_mps: float = 10.0,
        control_indices: list[int],
        host: str = "localhost",
        port: int = 65432,
        log_path: str = "logs/token_bridge_runtime.jsonl",
    ):
        self.runtime = runtime
        self.guardrails = guardrails
        self.mode = mode
        self.exec_mode = exec_mode
        self.g_mode = g_mode
        self.g_fixed = g_fixed
        self.g_max = g_max
        self.h_action_sec = h_action_sec
        self.latch_enabled = latch_enabled
        self.heading_update_thresh_deg = heading_update_thresh_deg
        self.alt_update_thresh_m = alt_update_thresh_m
        self.speed_update_thresh_mps = speed_update_thresh_mps
        self.control_indices = control_indices
        self.host = host
        self.port = int(port)
        self.receiver = StateReceiver()
        self._frames_seen = 0
        self._prev_sim_time: Optional[float] = None
        self._measured_dt: float = 0.2

        self._history_len: int = getattr(runtime, "obs_hist_len", 4)

        self.states: dict[int, TokenPlatformState] = {}
        for idx in control_indices:
            st = TokenPlatformState()
            st.prev_token = runtime.bos_token
            self.states[idx] = st

        self._contract_version: str = getattr(
            runtime, "CONTRACT_VERSION", "onestep_h4_current_anchor_ego_v1")

        self.logger = StructuredLogger(log_path)
        log(f"[TOKEN] Mode: {mode.value}")
        log(f"[TOKEN] Exec mode: {exec_mode.value}")
        log(f"[TOKEN] G mode: {g_mode.value} "
            f"(fixed={g_fixed:.1f}G, max={g_max:.1f}G)")
        log(f"[TOKEN] H_action_sec: {h_action_sec:.2f}s")
        log(f"[TOKEN] Latch: {'enabled' if latch_enabled else 'disabled'} "
            f"(hdg_thresh={heading_update_thresh_deg}°, "
            f"alt_thresh={alt_update_thresh_m}m, "
            f"spd_thresh={speed_update_thresh_mps}m/s)")
        log(f"[TOKEN] Contract: {self._contract_version}")
        log(f"[TOKEN] Control indices: {control_indices}")
        log(f"[TOKEN] Log file: {log_path}")
        log(f"[TOKEN] Runtime: {runtime.info()}")

    # ── observation construction ──────────────────────────────

    def _state_of(self, sim_data, idx):
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

    def _build_obs(self, st: dict, ps: TokenPlatformState) -> np.ndarray:
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

    # ── history helper ──────────────────────────────────────────

    def _build_obs_hist(self, ps: TokenPlatformState) -> np.ndarray:
        H = self._history_len
        ring = ps.obs_ring
        n = len(ring)
        if n >= H:
            return np.stack(ring[-H:], axis=0)
        pad_obs = ring[0]
        rows = [pad_obs] * (H - n) + list(ring)
        return np.stack(rows, axis=0)

    # ── latch update logic ────────────────────────────────────

    def _per_channel_latch_check(
        self, ps: TokenPlatformState, new_dpsi: float,
        new_alt_target: float, new_spd_target: float,
    ) -> tuple[bool, bool, bool]:
        """Per-channel latch: returns (update_hdg, update_alt, update_spd).

        Each channel updates independently so that e.g. a speed threshold
        breach does not force the altitude setpoint to be recomputed from
        the aircraft's new (partially-converged) position.
        """
        if not self.latch_enabled or not ps.latch_is_active:
            return True, True, True

        dpsi_diff_deg = abs(math.degrees(new_dpsi) - math.degrees(ps.latched_dpsi_rad))
        update_hdg = dpsi_diff_deg > self.heading_update_thresh_deg

        update_alt = (ps.latched_alt_target is None or
                      abs(new_alt_target - ps.latched_alt_target) > self.alt_update_thresh_m)

        update_spd = (ps.latched_spd_target is None or
                      abs(new_spd_target - ps.latched_spd_target) > self.speed_update_thresh_mps)

        return update_hdg, update_alt, update_spd

    # ── action frame construction ─────────────────────────────

    def build_action_frame(
        self, sim_data: np.ndarray, sim_time: float,
    ) -> np.ndarray:
        dt = self._measured_dt
        if self._prev_sim_time is not None and sim_time > self._prev_sim_time:
            dt = sim_time - self._prev_sim_time
            self._measured_dt = dt
        self._prev_sim_time = sim_time

        action = np.zeros(640, dtype=np.float32)

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
            is_task_reset = False
            is_latched = False
            chunk_result: Optional[GuardrailResult] = None
            dense_token = -1
            raw_token = -1
            confidence = 0.0
            decoded_chunk: Optional[np.ndarray] = None

            # --- determine effective token_steps for chunk exhaustion ---
            effective_steps = self.runtime.token_steps
            if self.exec_mode != ExecMode.REPLAY_4ROWS:
                # non-replay modes use a single reduced action for all ticks
                # but we still re-predict every token_steps frames
                effective_steps = self.runtime.token_steps

            # --- predict new token if chunk exhausted ---
            if ps.chunk_index >= effective_steps:
                ps.obs_ring.append(obs.copy())
                if len(ps.obs_ring) > self._history_len:
                    ps.obs_ring = ps.obs_ring[-self._history_len:]
                obs_hist = self._build_obs_hist(ps)

                dense_token, decoded_chunk, confidence = \
                    self.runtime.predict_and_decode(obs_hist, ps.prev_token)
                raw_token = self.runtime.dense_to_raw[dense_token]
                is_new_token = True
                ps.total_predictions += 1

                if self.mode == Mode.LIMITED:
                    chunk_result = self.guardrails.check_chunk(
                        decoded_chunk, confidence)

                if chunk_result is not None and not chunk_result.passed:
                    fb = self.guardrails.resolve_fallback(ps.last_valid_action)
                    ps.chunk_buffer = np.tile(fb, (self.runtime.token_steps, 1))
                    ps.total_fallbacks += 1
                else:
                    ps.chunk_buffer = decoded_chunk.copy()
                    ps.prev_token = dense_token

                ps.chunk_index = 0

                # --- compute the effective single action from chunk ---
                if self.exec_mode == ExecMode.REPLAY_4ROWS:
                    effective_action = ps.chunk_buffer[0].copy()
                else:
                    effective_action = _reduce_chunk(ps.chunk_buffer, self.exec_mode)

                dpsi_raw = float(effective_action[0])
                dalt_raw = float(effective_action[1])
                dspd_raw = float(effective_action[2])

                new_alt_target = _clamp(current_alt + dalt_raw, 200.0, 20000.0)
                new_spd_target = _clamp(current_spd + dspd_raw, 120.0, 650.0)

                upd_hdg, upd_alt, upd_spd = self._per_channel_latch_check(
                    ps, dpsi_raw, new_alt_target, new_spd_target)

                if upd_hdg or upd_alt or upd_spd:
                    is_task_reset = True
                    if upd_hdg:
                        ps.latched_dpsi_rad = dpsi_raw
                        ps.latched_psi_dot_rad_s = dpsi_raw / self.h_action_sec
                        if self.g_mode == GMode.DYNAMIC:
                            n_req, n_cmd = compute_dynamic_g(
                                current_spd, abs(ps.latched_psi_dot_rad_s), self.g_max)
                            ps.latched_n_cmd = n_cmd
                        else:
                            ps.latched_n_cmd = self.g_fixed
                        ps.heading_task_reset_count += 1
                    else:
                        ps.heading_latch_hold_count += 1
                    if upd_alt:
                        ps.latched_alt_target = new_alt_target
                        ps.alt_task_reset_count += 1
                    else:
                        ps.alt_latch_hold_count += 1
                    if upd_spd:
                        ps.latched_spd_target = new_spd_target
                        ps.spd_task_reset_count += 1
                    else:
                        ps.spd_latch_hold_count += 1
                    ps.latch_is_active = True
                else:
                    is_latched = True
                    ps.heading_latch_hold_count += 1
                    ps.alt_latch_hold_count += 1
                    ps.spd_latch_hold_count += 1

            # --- per-tick action output ---
            if self.exec_mode == ExecMode.REPLAY_4ROWS:
                # legacy: play chunk rows sequentially
                tick_action = ps.chunk_buffer[min(ps.chunk_index, 3)].copy()
                dpsi_this_tick = float(tick_action[0])
                dalt_this_tick = float(tick_action[1])
                dspd_this_tick = float(tick_action[2])

                psi_dot = dpsi_this_tick / self.h_action_sec
                turn_deg = math.degrees(psi_dot * dt)
                alt_cmd = _clamp(current_alt + dalt_this_tick, 200.0, 20000.0)
                spd_cmd = _clamp(current_spd + dspd_this_tick, 120.0, 650.0)

                if self.g_mode == GMode.DYNAMIC:
                    n_req, n_cmd = compute_dynamic_g(
                        current_spd, abs(psi_dot), self.g_max)
                else:
                    n_req = 1.0
                    n_cmd = self.g_fixed
            else:
                # first_row / reduce_mean / reduce_median: use latched values
                psi_dot = ps.latched_psi_dot_rad_s
                turn_deg = math.degrees(psi_dot * dt)
                alt_cmd = ps.latched_alt_target if ps.latched_alt_target is not None else current_alt
                spd_cmd = ps.latched_spd_target if ps.latched_spd_target is not None else current_spd
                n_req = math.sqrt(1.0 + (current_spd * abs(psi_dot) / GRAVITY) ** 2)
                n_cmd = ps.latched_n_cmd

            if n_req > n_cmd + 0.01:
                ps.g_saturation_count += 1

            turn_rate_measured_deg_s = 0.0
            if dt > 0.001:
                hdg_now = raw_st["heading_deg"]
                turn_rate_measured_deg_s = (hdg_now - ps.prev_heading_deg) / dt
                ps.prev_heading_deg = hdg_now

            tick_action_vec = np.array([
                ps.latched_dpsi_rad if self.exec_mode != ExecMode.REPLAY_4ROWS
                else float(ps.chunk_buffer[min(ps.chunk_index, 3), 0]),
                float(alt_cmd - current_alt),
                float(spd_cmd - current_spd),
            ], dtype=np.float32)

            tick_result: Optional[GuardrailResult] = None
            if self.mode == Mode.LIMITED:
                tick_result = self.guardrails.check_tick(tick_action_vec)
                if not tick_result.passed:
                    turn_deg = 0.0
                    alt_cmd = current_alt
                    spd_cmd = max(120.0, current_spd)
                    ps.total_fallbacks += 1

            fallback_triggered = False
            fallback_reason: Optional[str] = None
            if chunk_result is not None and not chunk_result.passed:
                fallback_triggered = True
                fallback_reason = chunk_result.reason
            if tick_result is not None and not tick_result.passed:
                fallback_triggered = True
                fallback_reason = tick_result.reason

            if not fallback_triggered:
                ps.last_valid_action = tick_action_vec.copy()

            g_cmd_mps2 = n_cmd * GRAVITY

            if self.mode == Mode.DRYRUN:
                sent_turn = 0.0
                sent_alt = current_alt
                sent_spd = current_spd
                sent_g = 3.0 * GRAVITY
            else:
                sent_turn = turn_deg
                sent_alt = alt_cmd
                sent_spd = spd_cmd
                sent_g = g_cmd_mps2

            base = 8 * idx
            action[base + 0] = float(sent_turn)
            action[base + 1] = float(sent_alt)
            action[base + 2] = float(sent_spd)
            action[base + 3] = float(sent_g)

            ps.chunk_index += 1

            # ── structured log ────────────────────────────────
            history_ready = len(ps.obs_ring) >= self._history_len
            psi_dot_deg_s = math.degrees(psi_dot)
            theory_turn_rate = max_turn_rate_at_g(current_spd, n_cmd)

            total_ticks = ps.heading_task_reset_count + ps.heading_latch_hold_count
            g_sat_ratio = (ps.g_saturation_count / max(total_ticks, 1))

            base_record = {
                "contract_version": self._contract_version,
                "sim_time": sim_time,
                "platform_idx": idx,
                "mode": self.mode.value,
                "exec_mode": self.exec_mode.value,
                "g_mode": self.g_mode.value,
                "H_action_sec": self.h_action_sec,
                "dt_measured": round(dt, 6),
                "chunk_index": ps.chunk_index - 1,
                "history_len": self._history_len,
                "history_ready": history_ready,
                "history_count": len(ps.obs_ring),
                "turn_rate_cmd_deg_s": round(psi_dot_deg_s, 4),
                "turn_rate_measured_deg_s": round(turn_rate_measured_deg_s, 4),
                "turn_rate_theory_deg_s": round(math.degrees(theory_turn_rate), 4),
                "n_req": round(n_req, 4),
                "n_cmd": round(n_cmd, 4),
                "is_task_reset": is_task_reset,
                "is_latched": is_latched,
                "current_speed_mps": round(current_spd, 2),
                "current_alt_m": round(current_alt, 2),
                "heading_task_reset_count": ps.heading_task_reset_count,
                "alt_task_reset_count": ps.alt_task_reset_count,
                "spd_task_reset_count": ps.spd_task_reset_count,
                "heading_latch_hold_count": ps.heading_latch_hold_count,
                "alt_latch_hold_count": ps.alt_latch_hold_count,
                "spd_latch_hold_count": ps.spd_latch_hold_count,
                "g_saturation_count": ps.g_saturation_count,
                "g_saturation_ratio": round(g_sat_ratio, 4),
                "token_action_3d": tick_action_vec.tolist(),
                "runtime_control_4d": [sent_turn, sent_alt, sent_spd, sent_g],
            }

            if is_new_token:
                base_record.update({
                    "event": "new_token",
                    "ts": datetime.datetime.now().isoformat(),
                    "is_token_boundary": True,
                    "token_decision_index": ps.total_predictions,
                    "obs_hist_shape": [self._history_len, 8],
                    "obs_anchor": obs,
                    "prev_token_in": ps.prev_token
                                     if not fallback_triggered
                                     else "fallback",
                    "predicted_dense_token": dense_token,
                    "predicted_raw_token": raw_token,
                    "confidence": round(confidence, 6),
                    "decoded_chunk": decoded_chunk,
                    "guardrail_passed": not fallback_triggered,
                    "fallback_triggered": fallback_triggered,
                    "fallback_reason": fallback_reason,
                })
            else:
                base_record.update({
                    "event": "tick",
                    "is_token_boundary": False,
                    "fallback_triggered": fallback_triggered,
                    "fallback_reason": fallback_reason,
                })

            self.logger.write(base_record)

        return action

    # ── main loop ─────────────────────────────────────────────

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        log(f"[TOKEN] Server listening on {self.host}:{self.port}")
        log("[TOKEN] Waiting for AFSIM connection ...")
        conn, addr = server.accept()
        log(f"[TOKEN] Connected from {addr}")
        try:
            while True:
                sim_time, sim_data = self.receiver.recv_frame(conn)
                self._frames_seen += 1
                action = self.build_action_frame(sim_data, sim_time)
                send_status_data(conn, action)
                if self._frames_seen <= 3:
                    log(f"[TOKEN] Frame ok: t={sim_time:.2f}, "
                        f"state_len={len(sim_data)}")
                if self._frames_seen % 20 == 0:
                    idx0 = self.control_indices[0] if self.control_indices else 0
                    ps = self.states.get(idx0)
                    b = 8 * idx0
                    log(f"[TOKEN] t={sim_time:.2f} "
                        f"turn={action[b]:.2f} alt={action[b+1]:.1f} "
                        f"spd={action[b+2]:.1f} "
                        f"G_cmd={action[b+3]/GRAVITY:.1f} "
                        f"pred={ps.total_predictions if ps else '?'} "
                        f"fb={ps.total_fallbacks if ps else '?'}")
        except (KeyboardInterrupt, EOFError):
            log("[TOKEN] Stopped.")
        finally:
            conn.close()
            server.close()
            self.logger.close()
            log("[TOKEN] Server shut down.")

    # ── self-test (no socket needed) ──────────────────────────

    def self_test(self) -> None:
        log("[TOKEN] Running self-test ...")
        sim_data = np.zeros(640, dtype=np.float32)
        idx = self.control_indices[0] if self.control_indices else 0
        b = 8 * idx
        sim_data[b + 0] = 1       # live
        sim_data[b + 1] = 63.0    # lat
        sim_data[b + 2] = 12.0    # lon
        sim_data[b + 3] = 5000.0  # alt_m
        sim_data[b + 4] = 220.0   # velN_mps
        sim_data[b + 5] = 0.0     # velE_mps
        sim_data[b + 6] = 0.0     # velD_mps
        sim_data[b + 7] = 0.0     # heading_deg

        token_steps = self.runtime.token_steps
        for tick in range(token_steps * 3):
            action = self.build_action_frame(sim_data, sim_time=0.2 * tick)
            t = action[b + 0]
            a = action[b + 1]
            s = action[b + 2]
            g = action[b + 3] / GRAVITY
            ps = self.states[idx]
            log(f"  tick={tick:2d}  turn={t:+8.3f}  alt={a:9.1f}  "
                f"spd={s:7.1f}  G={g:.1f}  chunk_idx={ps.chunk_index-1}  "
                f"prev_tok={ps.prev_token}")

        ps = self.states[idx]
        log(f"[TOKEN] Self-test complete: "
            f"predictions={ps.total_predictions}, "
            f"fallbacks={ps.total_fallbacks}")


# ── CLI ───────────────────────────────────────────────────────

def _parse_indices(s: str) -> list[int]:
    return [int(t.strip()) for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Token policy bridge server for AFSIM")
    ap.add_argument("--policy_ckpt", required=True,
                    help="OneStepTokenBC checkpoint (.pt)")
    ap.add_argument("--vq_ckpt", required=True,
                    help="VQ-VAE checkpoint (.pt)")
    ap.add_argument("--vocab", required=True,
                    help="Vocab JSON (dense<->raw mapping)")
    ap.add_argument("--mode", default="dryrun",
                    choices=["dryrun", "limited", "full"])
    ap.add_argument("--exec_mode", default="first_row",
                    choices=["first_row", "reduce_mean", "reduce_median",
                             "replay_4rows"],
                    help="Chunk execution strategy (default: first_row)")
    ap.add_argument("--g_mode", default="dynamic",
                    choices=["fixed", "dynamic"],
                    help="G-load mode (default: dynamic)")
    ap.add_argument("--g_fixed", type=float, default=3.0,
                    help="Fixed G-load value when g_mode=fixed (default: 3.0)")
    ap.add_argument("--g_max", type=float, default=7.0,
                    help="Maximum G-load cap for dynamic mode (default: 7.0)")
    ap.add_argument("--h_action_sec", type=float, default=2.0,
                    help="Action lookahead horizon in seconds "
                         "(from training contract, default: 2.0)")
    ap.add_argument("--latch", default="enabled",
                    choices=["enabled", "disabled"],
                    help="Latch mechanism for alt/spd setpoints "
                         "(default: enabled)")
    ap.add_argument("--heading_update_thresh", type=float, default=3.0,
                    help="Heading threshold for latch update (deg, default: 3.0)")
    ap.add_argument("--alt_update_thresh", type=float, default=50.0,
                    help="Altitude threshold for latch update (m, default: 50.0)")
    ap.add_argument("--speed_update_thresh", type=float, default=10.0,
                    help="Speed threshold for latch update (m/s, default: 10.0)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=65432)
    ap.add_argument("--control_indices", default="0,1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_path",
                    default="logs/token_bridge_runtime.jsonl")
    ap.add_argument("--self_test", action="store_true")
    # guardrail params
    ap.add_argument("--max_abs_dpsi", type=float, default=0.35)
    ap.add_argument("--max_abs_dalt", type=float, default=400.0)
    ap.add_argument("--max_abs_dspd", type=float, default=40.0)
    ap.add_argument("--conf_threshold", type=float, default=0.35)
    ap.add_argument("--fallback", default="hold_last",
                    choices=["neutral", "hold_last"])
    args = ap.parse_args()

    from sim.token_policy_runtime import TokenPolicyRuntime, TokenRuntimeConfig

    runtime = TokenPolicyRuntime(TokenRuntimeConfig(
        policy_ckpt=args.policy_ckpt,
        vq_ckpt=args.vq_ckpt,
        vocab_json=args.vocab,
        device=args.device,
    ))

    guardrails = TokenGuardrails(GuardrailConfig(
        max_abs_dpsi=args.max_abs_dpsi,
        max_abs_dalt=args.max_abs_dalt,
        max_abs_dspd=args.max_abs_dspd,
        conf_threshold=args.conf_threshold,
        fallback=FallbackStrategy(args.fallback),
    ))

    server = TokenBridgeServer(
        runtime=runtime,
        guardrails=guardrails,
        mode=Mode(args.mode),
        exec_mode=ExecMode(args.exec_mode),
        g_mode=GMode(args.g_mode),
        g_fixed=args.g_fixed,
        g_max=args.g_max,
        h_action_sec=args.h_action_sec,
        latch_enabled=(args.latch == "enabled"),
        heading_update_thresh_deg=args.heading_update_thresh,
        alt_update_thresh_m=args.alt_update_thresh,
        speed_update_thresh_mps=args.speed_update_thresh,
        control_indices=_parse_indices(args.control_indices),
        host=args.host,
        port=args.port,
        log_path=args.log_path,
    )

    if args.self_test:
        server.self_test()
    else:
        server.run()


if __name__ == "__main__":
    main()
