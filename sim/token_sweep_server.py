#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token sweep server — test VQ-VAE tokens in AFSIM with configurable execution.

Connects to AFSIM via the standard SAC_PROCESSOR socket protocol and
tests dense tokens by decoding them through VQ-VAE and sending the
resulting actions to AFSIM.

EXPERIMENT MODES (--exec_mode × --g_mode)
------------------------------------------
6 combinations:
    first_row   + fixed     first_row   + dynamic
    replay_4rows + fixed    replay_4rows + dynamic
    reduce_median + fixed   reduce_median + dynamic

ACTION SEMANTICS
-----------------
All VQ-VAE action outputs are 2s-lookahead DELTAS:
    dpsi_rad[i]  = heading[t+i+K] - heading[t+i]   (K=4 steps, 2.0s horizon)
    dalt_m[i]    = altitude[t+i+K] - altitude[t+i]
    dspd_mps[i]  = speed[t+i+K]    - speed[t+i]

The 4 rows are overlapping windows.  See token_bridge_server.py for details.

Usage (from wow/):
    python -m sim.token_sweep_server ^
        --vq_ckpt  checkpoints/vqvae_clean_t4_cb64/best.pt ^
        --vocab    datasets/dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json ^
        --exec_mode first_row --g_mode dynamic ^
        --tokens 29,14,15,25

Then launch AFSIM with the model-only scenario:
    cd build\\demos\\air_to_air
    afsim 2v2_model_only.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import socket
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from training.vq.vqvae_model import ActionChunkVQVAE

EARTH_R = 6378137.0
GRAVITY = 9.81


class ExecMode(Enum):
    FIRST_ROW = "first_row"
    REDUCE_MEAN = "reduce_mean"
    REDUCE_MEDIAN = "reduce_median"
    REPLAY_4ROWS = "replay_4rows"


class GMode(Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"


# ── AFSIM socket protocol ────────────────────────────────────

class StateReceiver:
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


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def log(msg: str) -> None:
    print(msg, flush=True)


def _reduce_chunk(chunk: np.ndarray, exec_mode: ExecMode) -> np.ndarray:
    if exec_mode == ExecMode.FIRST_ROW:
        return chunk[0].copy()
    elif exec_mode == ExecMode.REDUCE_MEAN:
        return chunk.mean(axis=0).copy()
    elif exec_mode == ExecMode.REDUCE_MEDIAN:
        return np.median(chunk, axis=0).copy()
    raise ValueError(f"Cannot reduce for {exec_mode}")


def compute_dynamic_g(speed_mps: float, psi_dot_abs: float,
                      g_max: float) -> tuple[float, float]:
    n_req = math.sqrt(1.0 + (speed_mps * psi_dot_abs / GRAVITY) ** 2)
    n_cmd = _clamp(n_req, 1.0, g_max)
    return n_req, n_cmd


def max_turn_rate_at_g(speed_mps: float, n_load: float) -> float:
    if n_load <= 1.0 or speed_mps <= 1.0:
        return 0.0
    return GRAVITY * math.sqrt(n_load ** 2 - 1.0) / speed_mps


# ── VQ-VAE loader ────────────────────────────────────────────

def load_decoder(vq_ckpt_path: str, vocab_json_path: str, device: str = "cpu"):
    dev = torch.device(device)
    vq_ckpt = torch.load(vq_ckpt_path, map_location=dev, weights_only=False)
    va = vq_ckpt["args"]
    token_steps = int(va["token_steps"])
    codebook_size = int(va["codebook_size"])

    vqvae = ActionChunkVQVAE(
        token_steps=token_steps,
        codebook_size=codebook_size,
        latent_dim=va.get("latent_dim", 32),
    )
    vqvae.load_state_dict(vq_ckpt["model_state_dict"])
    vqvae.to(dev).eval()

    normalize_action = bool(va.get("normalize_action", False))
    action_stats = vq_ckpt.get("action_stats", {})
    if normalize_action and action_stats:
        fields = action_stats["fields"]
        act_mean = np.array([action_stats["mean"][k] for k in fields], dtype=np.float32)
        act_std = np.array([action_stats["std"][k] for k in fields], dtype=np.float32)
    else:
        act_mean = np.zeros(3, dtype=np.float32)
        act_std = np.ones(3, dtype=np.float32)

    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    dense_to_raw = {int(k): int(v) for k, v in vocab_data["dense_id_to_raw_code"].items()}

    return vqvae, dense_to_raw, token_steps, act_mean, act_std, normalize_action, dev


@torch.no_grad()
def decode_token(vqvae, raw_code: int, token_steps: int,
                 act_mean, act_std, normalize_action, device) -> np.ndarray:
    raw_t = torch.tensor([raw_code], dtype=torch.long, device=device)
    z_q = vqvae.vq.embedding(raw_t)
    chunk_flat = vqvae.decoder(z_q)
    chunk = chunk_flat.cpu().numpy().reshape(token_steps, 3)
    if normalize_action:
        chunk = chunk * act_std[None, :] + act_mean[None, :]
    return chunk.astype(np.float32)


# ── JSONL logger ──────────────────────────────────────────────

class SweepLogger:
    def __init__(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(p, "w", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_default) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return str(obj)


# ── sweep server ──────────────────────────────────────────────

class TokenSweepServer:

    def __init__(
        self,
        *,
        vqvae,
        dense_to_raw: dict,
        token_steps: int,
        act_mean,
        act_std,
        normalize_action: bool,
        device,
        control_indices: list[int],
        exec_mode: ExecMode = ExecMode.FIRST_ROW,
        g_mode: GMode = GMode.DYNAMIC,
        g_fixed: float = 3.0,
        g_max: float = 7.0,
        h_action_sec: float = 2.0,
        host: str = "localhost",
        port: int = 65432,
        log_path: str = "logs/token_sweep.jsonl",
        settle_ticks: int = 8,
        test_tokens: Optional[list[int]] = None,
    ):
        self.vqvae = vqvae
        self.dense_to_raw = dense_to_raw
        self.token_steps = token_steps
        self.act_mean = act_mean
        self.act_std = act_std
        self.normalize_action = normalize_action
        self.device = device
        self.control_indices = control_indices
        self.exec_mode = exec_mode
        self.g_mode = g_mode
        self.g_fixed = g_fixed
        self.g_max = g_max
        self.h_action_sec = h_action_sec
        self.host = host
        self.port = port
        self.settle_ticks = settle_ticks

        self.receiver = StateReceiver()
        self.logger = SweepLogger(log_path)

        self.single_token: Optional[int] = None

        if test_tokens is not None:
            self.sorted_tokens = sorted(test_tokens)
        else:
            self.sorted_tokens = sorted(dense_to_raw.keys())
        self.vocab_size = len(self.sorted_tokens)

        self._all_chunks: dict[int, np.ndarray] = {}
        for did, raw in dense_to_raw.items():
            self._all_chunks[did] = decode_token(
                vqvae, raw, token_steps, act_mean, act_std, normalize_action, device)

        log(f"[SWEEP] Decoded {len(self._all_chunks)} tokens, "
            f"token_steps={token_steps}")
        log(f"[SWEEP] Exec mode: {exec_mode.value}")
        log(f"[SWEEP] G mode: {g_mode.value} "
            f"(fixed={g_fixed:.1f}G, max={g_max:.1f}G)")
        log(f"[SWEEP] H_action_sec: {h_action_sec:.2f}s")
        log(f"[SWEEP] Testing tokens: {self.sorted_tokens}")
        log(f"[SWEEP] Settle ticks: {settle_ticks}")
        log(f"[SWEEP] Control indices: {control_indices}")
        log(f"[SWEEP] Log: {log_path}")

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

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        log(f"[SWEEP] Listening on {self.host}:{self.port}")
        log("[SWEEP] Waiting for AFSIM ...")
        conn, addr = server.accept()
        log(f"[SWEEP] Connected from {addr}")

        token_idx = 0
        tick_in_phase = 0
        phase = "action"
        frames = 0
        done = False
        prev_sim_time: Optional[float] = None
        measured_dt = 0.2

        if self.single_token is not None:
            current_dense = self.single_token
        else:
            current_dense = self.sorted_tokens[0]
        current_chunk = self._all_chunks[current_dense]

        # compute the effective action for this token
        if self.exec_mode == ExecMode.REPLAY_4ROWS:
            effective_action = None
        else:
            effective_action = _reduce_chunk(current_chunk, self.exec_mode)

        # latched setpoints (computed at token start, held for duration)
        latched_alt_target: dict[int, float] = {}
        latched_spd_target: dict[int, float] = {}
        is_task_reset = False

        log(f"[SWEEP] Starting T{current_dense} (1/{self.vocab_size})")

        try:
            while not done:
                sim_time, sim_data = self.receiver.recv_frame(conn)
                frames += 1

                if prev_sim_time is not None and sim_time > prev_sim_time:
                    measured_dt = sim_time - prev_sim_time
                prev_sim_time = sim_time
                dt = measured_dt

                action = np.zeros(640, dtype=np.float32)

                for idx in self.control_indices:
                    if 8 * (idx + 1) > len(sim_data):
                        continue
                    st = self._state_of(sim_data, idx)
                    if st["live"] <= 0:
                        continue

                    current_alt = st["alt_m"]
                    current_spd = math.sqrt(
                        st["velN_mps"] ** 2 + st["velE_mps"] ** 2)

                    if phase == "action":
                        if self.exec_mode == ExecMode.REPLAY_4ROWS:
                            row = current_chunk[min(tick_in_phase, 3)]
                            dpsi_raw = float(row[0])
                            dalt_raw = float(row[1])
                            dspd_raw = float(row[2])
                        else:
                            dpsi_raw = float(effective_action[0])
                            dalt_raw = float(effective_action[1])
                            dspd_raw = float(effective_action[2])

                        psi_dot = dpsi_raw / self.h_action_sec
                        turn_deg = math.degrees(psi_dot * dt)

                        # latch alt/spd at token start
                        if tick_in_phase == 0 or self.exec_mode == ExecMode.REPLAY_4ROWS:
                            alt_target = _clamp(current_alt + dalt_raw, 200.0, 20000.0)
                            spd_target = _clamp(current_spd + dspd_raw, 120.0, 650.0)
                            latched_alt_target[idx] = alt_target
                            latched_spd_target[idx] = spd_target
                            is_task_reset = True
                        else:
                            alt_target = latched_alt_target.get(idx, current_alt)
                            spd_target = latched_spd_target.get(idx, current_spd)
                            is_task_reset = False

                        alt_cmd = alt_target
                        spd_cmd = spd_target

                        if self.g_mode == GMode.DYNAMIC:
                            n_req, n_cmd = compute_dynamic_g(
                                current_spd, abs(psi_dot), self.g_max)
                        else:
                            n_req = 1.0
                            n_cmd = self.g_fixed

                        g_cmd_mps2 = n_cmd * GRAVITY
                        event = "token_action"
                    else:
                        turn_deg = 0.0
                        alt_cmd = current_alt
                        spd_cmd = max(120.0, current_spd)
                        g_cmd_mps2 = 3.0 * GRAVITY
                        dpsi_raw = 0.0
                        dalt_raw = 0.0
                        dspd_raw = 0.0
                        psi_dot = 0.0
                        n_req = 1.0
                        n_cmd = 1.0
                        is_task_reset = False
                        event = "settle"

                    base = 8 * idx
                    action[base + 0] = float(turn_deg)
                    action[base + 1] = float(alt_cmd)
                    action[base + 2] = float(spd_cmd)
                    action[base + 3] = float(g_cmd_mps2)

                    psi_dot_deg_s = math.degrees(psi_dot)
                    theory_rate = max_turn_rate_at_g(current_spd, n_cmd)

                    self.logger.write({
                        "sim_time": sim_time,
                        "event": event,
                        "dense_token": current_dense,
                        "raw_token": self.dense_to_raw[current_dense],
                        "token_idx": token_idx,
                        "tick_in_phase": tick_in_phase,
                        "phase": phase,
                        "platform_idx": idx,
                        "exec_mode": self.exec_mode.value,
                        "g_mode": self.g_mode.value,
                        "H_action_sec": self.h_action_sec,
                        "dt_measured": round(dt, 6),
                        "token_action_3d": [dpsi_raw, dalt_raw, dspd_raw],
                        "runtime_control_4d": [turn_deg, alt_cmd, spd_cmd, g_cmd_mps2],
                        "turn_rate_cmd_deg_s": round(psi_dot_deg_s, 4),
                        "vz_cmd_mps": round(dalt_raw / self.h_action_sec, 4),
                        "ax_cmd_mps2": round(dspd_raw / self.h_action_sec, 4),
                        "n_req": round(n_req, 4),
                        "n_cmd": round(n_cmd, 4),
                        "turn_rate_theory_deg_s": round(math.degrees(theory_rate), 4),
                        "is_task_reset": is_task_reset,
                        "is_latched": not is_task_reset and phase == "action",
                        "afsim_state": {
                            "alt_m": current_alt,
                            "speed_mps": current_spd,
                            "heading_deg": st["heading_deg"],
                            "lat": st["lat"],
                            "lon": st["lon"],
                        },
                    })

                send_status_data(conn, action)

                tick_in_phase += 1

                if phase == "action" and tick_in_phase >= self.token_steps:
                    if self.single_token is not None:
                        tick_in_phase = 0
                    elif self.settle_ticks > 0:
                        phase = "settle"
                        tick_in_phase = 0
                    else:
                        token_idx += 1
                        tick_in_phase = 0
                        if token_idx >= self.vocab_size:
                            done = True
                        else:
                            current_dense = self.sorted_tokens[token_idx]
                            current_chunk = self._all_chunks[current_dense]
                            if self.exec_mode != ExecMode.REPLAY_4ROWS:
                                effective_action = _reduce_chunk(
                                    current_chunk, self.exec_mode)
                            log(f"[SWEEP] t={sim_time:.1f}s  "
                                f"Starting T{current_dense} "
                                f"({token_idx+1}/{self.vocab_size})")

                elif phase == "settle" and tick_in_phase >= self.settle_ticks:
                    token_idx += 1
                    tick_in_phase = 0
                    phase = "action"
                    if token_idx >= self.vocab_size:
                        done = True
                    else:
                        current_dense = self.sorted_tokens[token_idx]
                        current_chunk = self._all_chunks[current_dense]
                        if self.exec_mode != ExecMode.REPLAY_4ROWS:
                            effective_action = _reduce_chunk(
                                current_chunk, self.exec_mode)
                        log(f"[SWEEP] t={sim_time:.1f}s  "
                            f"Starting T{current_dense} "
                            f"({token_idx+1}/{self.vocab_size})")

                if frames <= 4 or frames % 20 == 0:
                    idx0 = self.control_indices[0]
                    st0 = self._state_of(sim_data, idx0)
                    g_disp = action[8*idx0+3] / GRAVITY
                    log(f"[SWEEP] t={sim_time:>7.1f}s  "
                        f"T{current_dense:>2d} {phase:>6s} tick={tick_in_phase}  "
                        f"alt={st0['alt_m']:.0f}m  "
                        f"hdg={st0['heading_deg']:.1f}\u00b0  "
                        f"turn={action[8*idx0]:.2f}\u00b0  "
                        f"G={g_disp:.1f}")

            log(f"[SWEEP] All {self.vocab_size} tokens tested! "
                f"Total frames: {frames}")

        except (KeyboardInterrupt, EOFError):
            log(f"[SWEEP] Stopped at token T{current_dense} "
                f"({token_idx+1}/{self.vocab_size})")
        finally:
            conn.close()
            server.close()
            self.logger.close()
            log("[SWEEP] Server shut down.")


# ── CLI ───────────────────────────────────────────────────────

def main():
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    ap = argparse.ArgumentParser(description="Token sweep server for AFSIM")
    ap.add_argument("--vq_ckpt", required=True,
                    help="VQ-VAE checkpoint (.pt)")
    ap.add_argument("--vocab", required=True,
                    help="Vocab JSON (dense<->raw mapping)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=65432)
    ap.add_argument("--control_indices", default="0,1",
                    help="Comma-separated platform indices to control")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_path", default="logs/token_sweep.jsonl")
    ap.add_argument("--settle_ticks", type=int, default=8,
                    help="Neutral ticks between tokens for stabilization")
    ap.add_argument("--single_token", type=int, default=None,
                    help="Repeat a single dense token ID forever")
    ap.add_argument("--tokens", type=str, default=None,
                    help="Comma-separated dense token IDs to test "
                         "(default: all vocab tokens)")
    ap.add_argument("--exec_mode", default="first_row",
                    choices=["first_row", "reduce_mean", "reduce_median",
                             "replay_4rows"],
                    help="Chunk execution strategy (default: first_row)")
    ap.add_argument("--g_mode", default="dynamic",
                    choices=["fixed", "dynamic"],
                    help="G-load mode (default: dynamic)")
    ap.add_argument("--g_fixed", type=float, default=3.0,
                    help="Fixed G-load when g_mode=fixed (default: 3.0)")
    ap.add_argument("--g_max", type=float, default=7.0,
                    help="Maximum G-load cap (default: 7.0)")
    ap.add_argument("--h_action_sec", type=float, default=2.0,
                    help="Action lookahead horizon in seconds (default: 2.0)")
    args = ap.parse_args()

    indices = [int(t.strip()) for t in args.control_indices.split(",") if t.strip()]
    test_tokens = None
    if args.tokens:
        test_tokens = [int(t.strip()) for t in args.tokens.split(",") if t.strip()]

    vqvae, dense_to_raw, token_steps, act_mean, act_std, norm, dev = \
        load_decoder(args.vq_ckpt, args.vocab, args.device)

    srv = TokenSweepServer(
        vqvae=vqvae,
        dense_to_raw=dense_to_raw,
        token_steps=token_steps,
        act_mean=act_mean,
        act_std=act_std,
        normalize_action=norm,
        device=dev,
        control_indices=indices,
        exec_mode=ExecMode(args.exec_mode),
        g_mode=GMode(args.g_mode),
        g_fixed=args.g_fixed,
        g_max=args.g_max,
        h_action_sec=args.h_action_sec,
        host=args.host,
        port=args.port,
        log_path=args.log_path,
        settle_ticks=args.settle_ticks,
        test_tokens=test_tokens,
    )

    if args.single_token is not None:
        if args.single_token not in dense_to_raw:
            log(f"[SWEEP] ERROR: token {args.single_token} not in vocab "
                f"(valid: {sorted(dense_to_raw.keys())})")
            sys.exit(1)
        srv.single_token = args.single_token

    srv.run()


if __name__ == "__main__":
    main()
