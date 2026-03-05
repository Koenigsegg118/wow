#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BC Transformer socket server for AFSIM 2v2_p6dof_bc scenario.

Input frame from AFSIM (text tokens):
  simTime 640 [80 platforms * 8 values]
Per-platform state values:
  [live, lat, lon, alt_m, velN_mps, velE_mps, velD_mps, heading_deg]

Output frame to AFSIM (binary):
  "STATUS" + 640 * float32
Per-platform action fields used by 2v2_p6dof_bc:
  [0]=turn_deg, [1]=alt_sp_m, [2]=spd_sp_mps
"""

import argparse
import math
import socket
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from afsim_policy_bridge import SmoothSetpointController
from training.train_bc_transformer_smooth import TransformerBC


EARTH_R = 6378137.0  # meters


def log(msg: str) -> None:
    print(msg, flush=True)


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


def latlon_to_enu_m(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x_e = (lon - lon0) * math.cos(0.5 * (lat + lat0)) * EARTH_R
    y_n = (lat - lat0) * EARTH_R
    return x_e, y_n


def unwrap_with_prev(curr_rad, prev_unwrapped_rad):
    if prev_unwrapped_rad is None:
        return curr_rad
    d = curr_rad - prev_unwrapped_rad
    if d > math.pi:
        curr_rad -= 2.0 * math.pi
    elif d < -math.pi:
        curr_rad += 2.0 * math.pi
    return curr_rad


@dataclass
class PlatformTrack:
    ref_lat: float | None = None
    ref_lon: float | None = None
    prev_heading_u: float | None = None
    obs_hist: deque | None = None
    ctrl: SmoothSetpointController | None = None

    def ensure(self, seq_len):
        if self.obs_hist is None:
            self.obs_hist = deque(maxlen=seq_len)
        if self.ctrl is None:
            self.ctrl = SmoothSetpointController()


class BCPolicyServer:
    def __init__(self, model_path, seq_len=20, control_indices=None, host="localhost", port=65432):
        self.model_path = model_path
        self.seq_len = int(seq_len)
        self.control_indices = control_indices or [0, 1, 2, 3]
        self.host = host
        self.port = int(port)
        self.receiver = StateReceiver()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._frames_seen = 0

        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Model file not found: {model_p}")
        log(f"[BC] Booting with model: {model_p}")
        log(f"[BC] Device: {self.device}")
        log(f"[BC] Control indices: {self.control_indices}, seq_len={self.seq_len}")

        self.model, self.obs_mean, self.obs_std, self.act_mean, self.act_std = self._load_model(model_path)
        self.model.eval()
        log("[BC] Model and normalizers loaded.")

        self.tracks = {idx: PlatformTrack() for idx in self.control_indices}

    def _load_model(self, model_path):
        log("[BC] Loading checkpoint ...")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        args = ckpt["args"]
        log(
            f"[BC] Checkpoint dims: obs={len(ckpt['obs_mean'])}, act={len(ckpt['act_mean'])}, "
            f"d_model={args.get('d_model', 128)}, layers={args.get('layers', 4)}, heads={args.get('heads', 4)}"
        )
        model = TransformerBC(
            obs_dim=8,
            act_dim=3,
            d_model=int(args.get("d_model", 128)),
            nhead=int(args.get("heads", 4)),
            num_layers=int(args.get("layers", 4)),
            dropout=float(args.get("dropout", 0.1)),
        ).to(self.device)
        model.load_state_dict(ckpt["model"], strict=True)

        obs_mean = torch.tensor(ckpt["obs_mean"], dtype=torch.float32, device=self.device)
        obs_std = torch.tensor(ckpt["obs_std"], dtype=torch.float32, device=self.device)
        act_mean = torch.tensor(ckpt["act_mean"], dtype=torch.float32, device=self.device)
        act_std = torch.tensor(ckpt["act_std"], dtype=torch.float32, device=self.device)
        return model, obs_mean, obs_std, act_mean, act_std

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

    def _build_obs(self, st, track):
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
        heading_now = math.radians(st["heading_deg"])
        heading_u = unwrap_with_prev(heading_now, track.prev_heading_u)
        track.prev_heading_u = heading_u
        ground_speed = math.sqrt(vx_e * vx_e + vy_n * vy_n)

        return np.array([x_e, y_n, z_u, vx_e, vy_n, vz_u, heading_u, ground_speed], dtype=np.float32)

    def _predict_action_phys(self, obs_seq_np):
        obs_t = torch.tensor(obs_seq_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_n = (obs_t - self.obs_mean.view(1, 1, -1)) / self.obs_std.view(1, 1, -1)
        with torch.no_grad():
            pred_n = self.model(obs_n)
            pred = pred_n * self.act_std.view(1, 1, -1) + self.act_mean.view(1, 1, -1)
        # take latest step
        dpsi_rad, alt_sp_m, spd_sp_mps = pred[0, -1, :].detach().cpu().numpy().tolist()
        return float(dpsi_rad), float(alt_sp_m), float(spd_sp_mps)

    def build_action_frame(self, sim_data):
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
            dpsi_rad, alt_sp_m, spd_sp_mps = self._predict_action_phys(obs_seq)
            _cmd, (turn_deg, alt_cmd, spd_cmd) = track.ctrl.step(
                dpsi_rad=dpsi_rad,
                alt_sp_m=alt_sp_m,
                spd_sp_mps=spd_sp_mps,
                alt_m=st["alt_m"],
                spd_mps=math.sqrt(st["velN_mps"] ** 2 + st["velE_mps"] ** 2),
            )

            base = 8 * idx
            action[base + 0] = float(turn_deg)
            action[base + 1] = float(alt_cmd)
            action[base + 2] = float(spd_cmd)
        return action

    def run(self):
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
                action = self.build_action_frame(sim_data)
                send_status_data(conn, action)
                if self._frames_seen <= 3:
                    log(f"[BC] First frames ok: t={sim_time:.2f}, state_len={len(sim_data)}")
                if int(sim_time) % 10 == 0:
                    t0 = action[0]
                    a0 = action[1]
                    s0 = action[2]
                    log(f"[BC] t={sim_time:.2f} idx0 turn={t0:.2f} alt={a0:.1f} spd={s0:.1f}")
        except KeyboardInterrupt:
            log("[BC] Stopped by user.")
        finally:
            conn.close()
            server.close()


def _parse_indices(s):
    out = []
    for t in (s or "").split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def self_test(model_path, seq_len):
    log("[BC] Running self-test ...")
    srv = BCPolicyServer(model_path=model_path, seq_len=seq_len, control_indices=[0], host="localhost", port=65432)
    sim_data = np.zeros(640, dtype=np.float32)
    # Fake one platform state with northward motion and heading=0.
    sim_data[0] = 1
    sim_data[1] = 63.0
    sim_data[2] = 12.0
    sim_data[3] = 5000.0
    sim_data[4] = 220.0
    sim_data[5] = 0.0
    sim_data[6] = 0.0
    sim_data[7] = 0.0
    for _ in range(seq_len):
        action = srv.build_action_frame(sim_data)
    log(f"[SELF_TEST] action idx0: {action[0]:.6f} {action[1]:.3f} {action[2]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:\\AF2.9\\afsim-2.9.0-win64\\wow\\bc_transformer_smooth.pt")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=65432)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--control_indices", default="0,1,2,3")
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(model_path=args.model, seq_len=args.seq_len)
        return

    server = BCPolicyServer(
        model_path=args.model,
        seq_len=args.seq_len,
        control_indices=_parse_indices(args.control_indices),
        host=args.host,
        port=args.port,
    )
    server.run()


if __name__ == "__main__":
    main()
