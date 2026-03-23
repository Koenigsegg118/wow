#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BC policy socket server for AFSIM 2v2_p6dof_bc scenario.

Input frame from AFSIM (text tokens):
  simTime 640 [80 platforms * 8 values]
Per-platform state values:
  [live, lat, lon, alt_m, velN_mps, velE_mps, velD_mps, heading_deg]

Output frame to AFSIM (binary):
  "STATUS" + 640 * float32
Per-platform action fields used by 2v2_p6dof_bc:
  [0]=turn_deg, [1]=alt_sp_m, [2]=spd_sp_mps

Observation construction:
  track_angle_rad_unwrapped is computed from ``atan2(vx_east, vy_north)``
  with continuous unwrapping.  AFSIM's raw ``heading_deg`` (body heading)
  is kept only as a diagnostic quantity and is never fed to the model.

Action construction:
  ``dpsi_rad`` from the model is interpreted as the track-angle delta from
  now to the artifact lookahead horizon.  The server uses
  ``horizon_steps / control_frequency_hz`` to convert that lookahead delta into
  the per-tick ``TurnToRelativeHeading`` command sent to AFSIM.

Runtime contract mode:
  Default is **strict** — only fully-conformant 0.2.0 artifacts are accepted.
  Pass ``--allow_legacy_artifact`` to enter a compatibility path that accepts
  older artifacts / legacy .pt checkpoints with warnings.

Model/runtime boundary:
  sim -> inference -> shared
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from inference.runtime import PolicyRuntime, REQUIRED_ACTION_FIELD_ORDER
from shared.schema import SCHEMA_VERSION
from sim.afsim_policy_bridge import SmoothSetpointController


EARTH_R = 6378137.0  # meters

SUPPORTED_FIELD_ORDER = [
    "x_e_m", "y_n_m", "z_u_m",
    "vx_e_mps", "vy_n_mps", "vz_u_mps",
    "track_angle_rad_unwrapped", "ground_speed_mps",
]

_DEFAULT_CONTROL_FREQ_HZ = 2.0
_HEADING_DIAG_THRESHOLD_DEG = 5.0
_HEADING_DIAG_LOG_INTERVAL = 100  # log every N frames per track
_DT_DEVIATION_THRESHOLD = 0.2     # 20 %


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# AFSIM socket protocol
# ---------------------------------------------------------------------------

class StateReceiver:
    """Parse AFSIM text frames: simTime 640 v0..v639."""

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
            vals = np.asarray(frame[2 : 2 + state_len], dtype=np.float32)
            return sim_time, vals


def send_status_data(connection, action_640_f32: np.ndarray) -> None:
    action_640_f32 = np.asarray(action_640_f32, dtype=np.float32)
    if action_640_f32.size != 640:
        raise ValueError(f"action size must be 640 float32, got {action_640_f32.size}")
    connection.sendall(b"STATUS")
    payload = action_640_f32.astype("<f4", copy=False).tobytes()
    connection.sendall(payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latlon_to_enu_m(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x_e = (lon - lon0) * math.cos(0.5 * (lat + lat0)) * EARTH_R
    y_n = (lat - lat0) * EARTH_R
    return x_e, y_n


def unwrap_with_prev(curr_rad: float, prev_unwrapped_rad: float | None) -> float:
    if prev_unwrapped_rad is None:
        return curr_rad
    d = curr_rad - prev_unwrapped_rad
    if d > math.pi:
        curr_rad -= 2.0 * math.pi
    elif d < -math.pi:
        curr_rad += 2.0 * math.pi
    return curr_rad


def _wrap_pi_diff(a: float, b: float) -> float:
    """Signed shortest-arc difference a - b in (-pi, pi]."""
    d = a - b
    return d - 2.0 * math.pi * round(d / (2.0 * math.pi))


# ---------------------------------------------------------------------------
# Per-platform state tracker
# ---------------------------------------------------------------------------

@dataclass
class PlatformTrack:
    ref_lat: float | None = None
    ref_lon: float | None = None
    prev_track_angle_u: float | None = None
    obs_hist: deque | None = None
    ctrl: SmoothSetpointController | None = None
    diag_frame_count: int = 0

    def ensure(self, seq_len: int) -> None:
        if self.obs_hist is None:
            self.obs_hist = deque(maxlen=seq_len)
        if self.ctrl is None:
            self.ctrl = SmoothSetpointController()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class BCPolicyServer:
    def __init__(
        self,
        *,
        artifact_dir: str | None = None,
        legacy_model_path: str | None = None,
        seq_len: int | None = None,
        control_indices=None,
        host="localhost",
        port=65432,
        strict_dt: bool = False,
        allow_legacy_artifact: bool = False,
    ):
        self.control_indices = control_indices or [0, 1, 2, 3]
        self.host = host
        self.port = int(port)
        self.receiver = StateReceiver()
        self._frames_seen = 0
        self._strict_dt = strict_dt
        self._prev_sim_time: float | None = None
        self._dt_warn_count = 0
        self._allow_legacy = allow_legacy_artifact

        if allow_legacy_artifact:
            log("[BC] MODE: legacy-compatible (--allow_legacy_artifact)")
        else:
            log("[BC] MODE: strict (default)")

        # ---- load policy runtime ----
        if artifact_dir and Path(artifact_dir).exists():
            self.policy = PolicyRuntime.from_artifact_dir(
                artifact_dir, allow_legacy=allow_legacy_artifact,
            )
            log(f"[BC] Runtime source: artifact={artifact_dir}")
        elif legacy_model_path:
            self.policy = PolicyRuntime.from_legacy_checkpoint(
                legacy_model_path,
                seq_len=(seq_len or 20),
                allow_legacy=allow_legacy_artifact,
            )
            log(f"[BC] Runtime source: legacy_checkpoint={legacy_model_path}")
        else:
            raise FileNotFoundError(
                "No valid policy source found. Provide --artifact directory "
                "or --model legacy checkpoint (requires --allow_legacy_artifact)."
            )

        # ---- validate contract ----
        self._validate_field_order()
        self._validate_action_field_order()
        self._validate_semantic_conventions()

        # ---- seq_len ----
        self.seq_len = int(self.policy.seq_len)
        if seq_len is not None and int(seq_len) != self.seq_len:
            raise ValueError(
                f"CLI --seq_len={seq_len} conflicts with artifact "
                f"seq_len={self.seq_len}.  Remove --seq_len to use the "
                f"artifact value."
            )
        log(f"[BC] seq_len={self.seq_len} (from artifact)")

        # ---- expected dt / control frequency ----
        freq = self.policy.control_frequency_hz
        if freq is not None:
            self._control_frequency_hz = float(freq)
            self._expected_dt = 1.0 / self._control_frequency_hz
            log(f"[BC] Control frequency from artifact: {self._control_frequency_hz} Hz  "
                f"(expected dt={self._expected_dt:.4f}s)")
        else:
            self._control_frequency_hz = _DEFAULT_CONTROL_FREQ_HZ
            self._expected_dt = 1.0 / self._control_frequency_hz
            log(f"[BC] WARNING: artifact missing control_frequency_hz. "
                f"Defaulting to {self._control_frequency_hz} Hz "
                f"(dt={self._expected_dt:.4f}s)")

        # ---- horizon_steps ----
        hs = self.policy.horizon_steps
        if hs is not None:
            self._horizon_steps = int(hs)
            log(f"[BC] Horizon steps from artifact: {self._horizon_steps}")
        else:
            self._horizon_steps = 1
            log("[BC] WARNING: artifact missing horizon_steps. Assuming 1.")

        self._lookahead_horizon_s = (
            float(self._horizon_steps) / float(self._control_frequency_hz)
        )
        log(f"[BC] dpsi lookahead horizon: {self._lookahead_horizon_s:.4f}s")

        log(f"[BC] Control indices: {self.control_indices}")
        self.tracks: dict[int, PlatformTrack] = {
            idx: PlatformTrack() for idx in self.control_indices
        }

    # ------------------------------------------------------------------
    # Contract validation
    # ------------------------------------------------------------------

    def _validate_field_order(self) -> None:
        fo = self.policy.field_order
        if fo is None:
            raise ValueError(
                "PolicyRuntime has no field_order.  "
                "Cannot validate observation construction."
            )
        if fo != SUPPORTED_FIELD_ORDER:
            raise ValueError(
                f"obs field_order mismatch!\n"
                f"  artifact :  {fo}\n"
                f"  supported:  {SUPPORTED_FIELD_ORDER}\n"
                f"Cannot safely construct observations."
            )
        if self.policy.obs_dim != len(SUPPORTED_FIELD_ORDER):
            raise ValueError(
                f"obs_dim mismatch: artifact={self.policy.obs_dim}, "
                f"supported={len(SUPPORTED_FIELD_ORDER)}"
            )
        log(f"[BC] Contract OK: obs field_order validated, "
            f"obs_dim={self.policy.obs_dim}")

    def _validate_action_field_order(self) -> None:
        afo = self.policy.action_field_order
        if afo is None:
            raise ValueError(
                "PolicyRuntime has no action_field_order.  "
                "Cannot validate action mapping."
            )
        if afo != REQUIRED_ACTION_FIELD_ORDER:
            raise ValueError(
                f"action_field_order mismatch!\n"
                f"  artifact :  {afo}\n"
                f"  required :  {REQUIRED_ACTION_FIELD_ORDER}\n"
                f"Cannot safely map model output to AFSIM commands."
            )
        log("[BC] Contract OK: action_field_order validated")

    def _validate_semantic_conventions(self) -> None:
        obs_spec = self.policy.policy_meta.get("obs_spec", {}) or {}
        act_spec = self.policy.policy_meta.get("action_spec", {}) or {}

        track_angle_convention = obs_spec.get("track_angle_convention")
        if track_angle_convention not in (None, "atan2(vx_east, vy_north)_unwrapped"):
            raise ValueError(
                "Unsupported obs_spec.track_angle_convention="
                f"{track_angle_convention!r}. Runtime expects "
                "'atan2(vx_east, vy_north)_unwrapped'."
            )

        ground_speed_convention = obs_spec.get("ground_speed_convention")
        if ground_speed_convention not in (None, "horizontal_2d"):
            raise ValueError(
                "Unsupported obs_spec.ground_speed_convention="
                f"{ground_speed_convention!r}. Runtime expects 'horizontal_2d'."
            )

        angle_reference = act_spec.get("angle_reference")
        if angle_reference is None:
            if not self._allow_legacy:
                raise ValueError(
                    "Artifact missing required action_spec.angle_reference. "
                    "Strict runtime needs this to confirm dpsi_rad is relative "
                    "to track_angle."
                )
            log("[BC] WARNING: artifact missing angle_reference. Assuming track_angle.")
        elif angle_reference != "track_angle":
            raise ValueError(
                f"Unsupported action_spec.angle_reference={angle_reference!r}. "
                "Runtime expects 'track_angle'."
            )

        log("[BC] Contract OK: semantic conventions validated")

    # ------------------------------------------------------------------
    # dt monitoring
    # ------------------------------------------------------------------

    def _compute_and_check_dt(self, sim_time: float) -> float:
        """Return actual dt; warn or fail on frequency mismatch."""
        if self._prev_sim_time is None:
            self._prev_sim_time = sim_time
            return self._expected_dt

        actual_dt = sim_time - self._prev_sim_time
        self._prev_sim_time = sim_time

        if actual_dt <= 0.0:
            log(f"[BC] WARNING: non-positive dt={actual_dt:.6f}s at t={sim_time:.4f}. "
                f"Using expected_dt={self._expected_dt:.4f}s instead.")
            return self._expected_dt

        rel_err = abs(actual_dt - self._expected_dt) / self._expected_dt
        if rel_err > _DT_DEVIATION_THRESHOLD:
            msg = (
                f"[BC] dt deviation: actual={actual_dt:.4f}s vs "
                f"expected={self._expected_dt:.4f}s (err={rel_err*100:.1f}%)"
            )
            if self._strict_dt:
                raise RuntimeError(f"FATAL {msg}  (strict_dt mode)")
            if self._dt_warn_count < 5:
                log(f"WARNING {msg}")
            elif self._dt_warn_count == 5:
                log("[BC] WARNING: further dt-deviation warnings suppressed")
            self._dt_warn_count += 1

        return actual_dt

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

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

    def _build_obs(self, st: dict, track: PlatformTrack) -> np.ndarray:
        lat = st["lat"]
        lon = st["lon"]
        if track.ref_lat is None or track.ref_lon is None:
            track.ref_lat = lat
            track.ref_lon = lon

        x_e, y_n = latlon_to_enu_m(lat, lon, track.ref_lat, track.ref_lon)
        z_u = st["alt_m"]
        vx_e = st["velE_mps"]
        vy_n = st["velN_mps"]
        vz_u = -st["velD_mps"]

        track_angle_raw = math.atan2(vx_e, vy_n)
        track_angle_u = unwrap_with_prev(track_angle_raw, track.prev_track_angle_u)
        track.prev_track_angle_u = track_angle_u

        ground_speed = math.sqrt(vx_e * vx_e + vy_n * vy_n)

        track.diag_frame_count += 1
        if track.diag_frame_count % _HEADING_DIAG_LOG_INTERVAL == 1:
            body_heading_rad = math.radians(st["heading_deg"])
            dev = abs(math.degrees(_wrap_pi_diff(track_angle_raw, body_heading_rad)))
            if dev > _HEADING_DIAG_THRESHOLD_DEG:
                log(f"[BC] DIAG: body_hdg={st['heading_deg']:.1f}\u00b0 "
                    f"track_angle={math.degrees(track_angle_raw):.1f}\u00b0 "
                    f"delta={dev:.1f}\u00b0")

        return np.array(
            [x_e, y_n, z_u, vx_e, vy_n, vz_u, track_angle_u, ground_speed],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Inference + action mapping
    # ------------------------------------------------------------------

    def _predict_action_phys(self, obs_seq_np):
        pred = self.policy.predict(obs_seq_np)
        return float(pred[0]), float(pred[1]), float(pred[2])

    def build_action_frame(
        self, sim_data: np.ndarray, sim_time: float | None = None,
    ) -> np.ndarray:
        dt = (
            self._compute_and_check_dt(sim_time)
            if sim_time is not None
            else self._expected_dt
        )

        action = np.zeros(640, dtype=np.float32)
        for idx in self.control_indices:
            if 8 * (idx + 1) > len(sim_data):
                continue
            st = self._state_of(sim_data, idx)
            if st["live"] <= 0:
                continue

            track = self.tracks[idx]
            track.ensure(self.seq_len)
            if track.ctrl.prev is None:
                gs = math.sqrt(st["velN_mps"] ** 2 + st["velE_mps"] ** 2)
                track.ctrl.reset(alt_m=st["alt_m"], spd_mps=max(gs, 120.0))

            obs = self._build_obs(st, track)
            track.obs_hist.append(obs)
            if len(track.obs_hist) < self.seq_len:
                continue

            obs_seq = np.stack(list(track.obs_hist), axis=0)
            assert obs_seq.shape[0] == self.seq_len, (
                f"BUG: obs_seq window {obs_seq.shape[0]} != "
                f"artifact seq_len {self.seq_len}"
            )

            dpsi_rad_lookahead, alt_sp_m, spd_sp_mps = self._predict_action_phys(obs_seq)
            _cmd, (turn_deg, alt_cmd, spd_cmd) = track.ctrl.step(
                dpsi_rad=dpsi_rad_lookahead,
                alt_sp_m=alt_sp_m,
                spd_sp_mps=spd_sp_mps,
                alt_m=st["alt_m"],
                spd_mps=math.sqrt(st["velN_mps"] ** 2 + st["velE_mps"] ** 2),
                dt=dt,
                lookahead_horizon_s=self._lookahead_horizon_s,
            )

            base = 8 * idx
            action[base + 0] = float(turn_deg)
            action[base + 1] = float(alt_cmd)
            action[base + 2] = float(spd_cmd)
            action[base + 3] = 3.0 * 9.81   # BC policy uses fixed 3G
        return action

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, *, token_shadow=None):
        """Main loop.

        Args:
            token_shadow: optional ``TokenShadow`` instance.  When
                provided, each tick's state and primary action are
                forwarded to the shadow for read-only observation
                and logging.  The primary action is never modified.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)
        log(f"[BC] Socket Server listening on {self.host}:{self.port}")
        log("[BC] Waiting for AFSIM connection ...")
        conn, addr = server.accept()
        log(f"[BC] Connected from {addr}")
        try:
            while True:
                sim_time, sim_data = self.receiver.recv_frame(conn)
                self._frames_seen += 1
                action = self.build_action_frame(sim_data, sim_time=sim_time)
                if token_shadow is not None:
                    token_shadow.observe(sim_data, sim_time, action)
                send_status_data(conn, action)
                if self._frames_seen <= 3:
                    log(f"[BC] First frames ok: t={sim_time:.2f}, "
                        f"state_len={len(sim_data)}")
                if self._frames_seen % 20 == 0:
                    t0 = action[0]
                    a0 = action[1]
                    s0 = action[2]
                    log(f"[BC] t={sim_time:.2f} idx0 turn={t0:.2f} "
                        f"alt={a0:.1f} spd={s0:.1f}")
        except KeyboardInterrupt:
            log("[BC] Stopped by user.")
        finally:
            if token_shadow is not None:
                token_shadow.close()
            conn.close()
            server.close()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_indices(s):
    out = []
    for t in (s or "").split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def self_test(
    *,
    artifact_dir: str | None,
    model_path: str | None,
    seq_len: int | None,
    allow_legacy_artifact: bool = False,
):
    log("[BC] Running self-test ...")
    srv = BCPolicyServer(
        artifact_dir=artifact_dir,
        legacy_model_path=model_path,
        seq_len=seq_len,
        control_indices=[0],
        host="localhost",
        port=65432,
        allow_legacy_artifact=allow_legacy_artifact,
    )
    sim_data = np.zeros(640, dtype=np.float32)
    sim_data[0] = 1       # live
    sim_data[1] = 63.0    # lat
    sim_data[2] = 12.0    # lon
    sim_data[3] = 5000.0  # alt_m
    sim_data[4] = 220.0   # velN_mps
    sim_data[5] = 0.0     # velE_mps
    sim_data[6] = 0.0     # velD_mps
    sim_data[7] = 0.0     # heading_deg
    for _ in range(srv.seq_len):
        action = srv.build_action_frame(sim_data)
    log(f"[SELF_TEST] action idx0: turn={action[0]:.6f} "
        f"alt={action[1]:.3f} spd={action[2]:.3f}")


def main():
    default_artifact = _ROOT_DIR / "artifacts" / "policy_demo" / f"v{SCHEMA_VERSION}"
    legacy_default_model = _ROOT_DIR / "bc_transformer_smooth.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(default_artifact))
    ap.add_argument("--model", default="",
                    help="Legacy checkpoint path (.pt), deprecated.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=65432)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--control_indices", default="0,1,2,3")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--strict_dt", action="store_true",
                    help="Abort on control-frequency mismatch instead of warning.")
    ap.add_argument("--allow_legacy_artifact", action="store_true",
                    help="Accept older schema / legacy .pt checkpoints with "
                    "warnings.  Default is strict mode that rejects any "
                    "artifact missing required 0.2.0 metadata.")
    # ── token shadow (read-only parallel observer) ──
    ap.add_argument("--token_shadow", action="store_true",
                    help="Enable token policy shadow mode: token policy "
                    "runs in parallel, logs suggested actions, but never "
                    "affects the primary controller output.")
    ap.add_argument("--token_policy_ckpt", default="",
                    help="OneStepTokenBC checkpoint for shadow mode.")
    ap.add_argument("--token_vq_ckpt", default="",
                    help="VQ-VAE checkpoint for shadow mode.")
    ap.add_argument("--token_vocab", default="",
                    help="Vocab JSON for shadow mode.")
    ap.add_argument("--token_shadow_indices", default="",
                    help="Comma-separated platform indices for shadow "
                    "(default: same as --control_indices).")
    ap.add_argument("--token_shadow_log",
                    default="logs/token_shadow.jsonl",
                    help="JSONL log path for shadow output.")
    ap.add_argument("--token_device", default="cpu")
    args = ap.parse_args()

    artifact_path = Path(args.artifact) if args.artifact else None
    model_path = args.model if args.model else None

    if artifact_path is not None and artifact_path.exists():
        pass
    else:
        if model_path:
            if not args.allow_legacy_artifact:
                raise SystemExit(
                    "Legacy --model path requires --allow_legacy_artifact.  "
                    "Migrate to artifact-based deployment."
                )
            log("[BC] WARNING: using legacy --model path; migrate to --artifact.")
        elif legacy_default_model.exists():
            if not args.allow_legacy_artifact:
                raise SystemExit(
                    f"Artifact not found: {artifact_path}.  "
                    f"Legacy checkpoint exists at {legacy_default_model} but "
                    f"strict mode is active.  Pass --allow_legacy_artifact to "
                    f"use it, or provide a valid --artifact directory."
                )
            model_path = str(legacy_default_model)
            log("[BC] WARNING: artifact not found; falling back to legacy checkpoint.")
        else:
            raise SystemExit(
                f"Artifact not found: {artifact_path}. "
                "Provide --artifact with a valid policy directory "
                "or --model legacy checkpoint (with --allow_legacy_artifact)."
            )

    if args.self_test:
        self_test(
            artifact_dir=(
                str(artifact_path)
                if artifact_path is not None and artifact_path.exists()
                else None
            ),
            model_path=model_path,
            seq_len=args.seq_len,
            allow_legacy_artifact=args.allow_legacy_artifact,
        )
        return

    server = BCPolicyServer(
        artifact_dir=(
            str(artifact_path)
            if artifact_path is not None and artifact_path.exists()
            else None
        ),
        legacy_model_path=model_path,
        seq_len=args.seq_len,
        control_indices=_parse_indices(args.control_indices),
        host=args.host,
        port=args.port,
        strict_dt=args.strict_dt,
        allow_legacy_artifact=args.allow_legacy_artifact,
    )

    # ── optional token shadow ──
    token_shadow = None
    if args.token_shadow:
        from sim.token_policy_runtime import TokenPolicyRuntime, TokenRuntimeConfig
        from sim.token_shadow import TokenShadow
        from sim.token_guardrails import TokenGuardrails

        shadow_indices = (
            _parse_indices(args.token_shadow_indices)
            if args.token_shadow_indices
            else _parse_indices(args.control_indices)
        )

        tok_defaults = {
            "policy_ckpt": str(_ROOT_DIR / "checkpoints"
                               / "onestep_token_bc_t4_cb64_h4" / "best.pt"),
            "vq_ckpt": str(_ROOT_DIR / "checkpoints"
                           / "vqvae_clean_t4_cb64" / "best.pt"),
            "vocab": str(_ROOT_DIR / "datasets"
                         / "dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json"),
        }
        tok_runtime = TokenPolicyRuntime(TokenRuntimeConfig(
            policy_ckpt=args.token_policy_ckpt or tok_defaults["policy_ckpt"],
            vq_ckpt=args.token_vq_ckpt or tok_defaults["vq_ckpt"],
            vocab_json=args.token_vocab or tok_defaults["vocab"],
            device=args.token_device,
        ))
        token_shadow = TokenShadow(
            runtime=tok_runtime,
            control_indices=shadow_indices,
            log_path=args.token_shadow_log,
            guardrails=TokenGuardrails(),
        )
        log("[BC] Token shadow module loaded and active.")

    server.run(token_shadow=token_shadow)


if __name__ == "__main__":
    main()
