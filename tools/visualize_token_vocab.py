#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode all VQ-VAE tokens and visualize their flight trajectories.

For each of the 30 dense tokens, this script:
  1. Decodes the token via VQ-VAE → action chunk [4, 3]
  2. Simulates a simple kinematic trajectory (4 steps × 0.5s)
  3. Plots the results as a grid of mini-trajectories + summary charts

Usage:
    python -m tools.visualize_token_vocab
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.vq.vqvae_model import ActionChunkVQVAE


def load_vqvae_and_vocab(
    vq_ckpt_path: str,
    vocab_json_path: str,
    device: str = "cpu",
):
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
    token_stats = {s["dense_id"]: s for s in vocab_data.get("token_stats", [])}

    return vqvae, dense_to_raw, token_stats, token_steps, act_mean, act_std, normalize_action, dev


@torch.no_grad()
def decode_all_tokens(vqvae, dense_to_raw, token_steps, act_mean, act_std, normalize_action, device):
    results = {}
    for dense_id, raw_code in sorted(dense_to_raw.items()):
        raw_t = torch.tensor([raw_code], dtype=torch.long, device=device)
        z_q = vqvae.vq.embedding(raw_t)
        chunk_flat = vqvae.decoder(z_q)
        chunk = chunk_flat.cpu().numpy().reshape(token_steps, 3)
        if normalize_action:
            chunk = chunk * act_std[None, :] + act_mean[None, :]
        results[dense_id] = chunk.astype(np.float32)
    return results


def simulate_trajectory(chunk, init_heading=0.0, init_alt=5000.0, init_speed=200.0,
                        dt_step=0.5, lookahead_steps=4):
    """Kinematic simulation with correct 2s-lookahead dpsi semantics.

    Each dpsi value = heading change over `lookahead_steps * dt_step` seconds
    (= 2.0s), NOT a per-step increment.  The implied turn rate is
    dpsi / (lookahead_steps * dt_step).  We simulate 8 sub-steps (= 2s
    beyond the chunk) to show the full trajectory the token describes.
    """
    avg_dpsi = float(np.mean(chunk[:, 0]))
    avg_dalt = float(np.mean(chunk[:, 1]))
    avg_dspd = float(np.mean(chunk[:, 2]))

    lookahead_sec = lookahead_steps * dt_step   # 2.0s
    turn_rate = avg_dpsi / lookahead_sec         # rad/s
    alt_rate = avg_dalt / lookahead_sec           # m/s
    spd_rate = avg_dspd / lookahead_sec           # m/s²

    heading = init_heading
    x, y = 0.0, 0.0
    alt = init_alt
    speed = init_speed

    n_sim_steps = 8
    sim_dt = dt_step

    traj_x = [x]
    traj_y = [y]
    traj_alt = [alt]
    traj_spd = [speed]
    traj_hdg = [heading]

    for _ in range(n_sim_steps):
        heading += turn_rate * sim_dt
        speed = max(50.0, speed + spd_rate * sim_dt)
        alt += alt_rate * sim_dt

        x += speed * math.sin(heading) * sim_dt
        y += speed * math.cos(heading) * sim_dt

        traj_x.append(x)
        traj_y.append(y)
        traj_alt.append(alt)
        traj_spd.append(speed)
        traj_hdg.append(heading)

    step0_dpsi_deg = math.degrees(float(chunk[0, 0]))
    step0_dalt = float(chunk[0, 1])
    step0_dspd = float(chunk[0, 2])

    return {
        "x": np.array(traj_x),
        "y": np.array(traj_y),
        "alt": np.array(traj_alt),
        "spd": np.array(traj_spd),
        "hdg": np.array(traj_hdg),
        "step0_dpsi_deg": step0_dpsi_deg,
        "step0_dalt": step0_dalt,
        "step0_dspd": step0_dspd,
        "turn_rate_deg_s": math.degrees(turn_rate),
    }


def main():
    vq_ckpt = str(ROOT / "checkpoints" / "vqvae_clean_t4_cb64" / "best.pt")
    vocab_json = str(ROOT / "datasets" / "dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json")
    out_path = str(ROOT / "logs" / "token_vocab_visual.png")

    print("Loading VQ-VAE and vocab ...")
    vqvae, dense_to_raw, token_stats, token_steps, act_mean, act_std, norm, dev = \
        load_vqvae_and_vocab(vq_ckpt, vocab_json)
    print(f"  token_steps={token_steps}, vocab_size={len(dense_to_raw)}")

    print("Decoding all tokens ...")
    chunks = decode_all_tokens(vqvae, dense_to_raw, token_steps, act_mean, act_std, norm, dev)

    print("Simulating trajectories ...")
    trajs = {}
    for did, chunk in chunks.items():
        trajs[did] = simulate_trajectory(chunk)

    n_tokens = len(chunks)
    ncols = 6
    nrows = 5

    fig = plt.figure(figsize=(24, 22), facecolor="white")
    fig.suptitle("Token Vocabulary — Flight Trajectories (top-down view)\n"
                 "dpsi = 2s lookahead heading delta (NOT per-step increment).  "
                 "Simulated 4s at 200 m/s.",
                 fontsize=16, fontweight="bold", y=0.98)

    gs_top = fig.add_gridspec(nrows, ncols,
                              left=0.04, right=0.96, top=0.93, bottom=0.32,
                              hspace=0.45, wspace=0.35)

    x_all = np.concatenate([t["x"] for t in trajs.values()])
    y_all = np.concatenate([t["y"] for t in trajs.values()])
    margin = 20
    x_range = max(abs(x_all.min()), abs(x_all.max()), 5) + margin
    y_min = min(y_all.min(), 0) - margin
    y_max = max(y_all.max(), 5) + margin

    sorted_ids = sorted(chunks.keys())
    for i, did in enumerate(sorted_ids):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs_top[row, col])
        t = trajs[did]
        stat = token_stats.get(did, {})
        ratio = stat.get("ratio", 0) * 100

        color = "tab:blue"
        if abs(t["step0_dpsi_deg"]) > 2:
            color = "tab:red" if t["step0_dpsi_deg"] > 0 else "tab:green"
        if abs(t["step0_dalt"]) > 50:
            color = "tab:orange" if t["step0_dalt"] > 0 else "tab:purple"

        ax.plot(t["x"], t["y"], "-o", color=color, markersize=3, linewidth=1.8)
        ax.plot(0, 0, "ks", markersize=5)
        ax.arrow(t["x"][-2], t["y"][-2],
                 t["x"][-1] - t["x"][-2], t["y"][-1] - t["y"][-2],
                 head_width=3, head_length=2, fc=color, ec=color)

        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)
        ax.tick_params(labelsize=6)

        title = f"T{did}"
        ax.set_title(title, fontsize=9, fontweight="bold")

        info = (f"dpsi={t['step0_dpsi_deg']:+.1f}\u00b0/2s\n"
                f"dalt={t['step0_dalt']:+.1f}m/2s\n"
                f"dspd={t['step0_dspd']:+.1f}m\u00b7s\u207b\u00b9/2s\n"
                f"rate={t['turn_rate_deg_s']:+.1f}\u00b0/s\n"
                f"freq={ratio:.1f}%")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=6,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7))

    gs_bot = fig.add_gridspec(1, 3,
                              left=0.06, right=0.96, top=0.26, bottom=0.04,
                              wspace=0.3)

    labels = [f"T{d}" for d in sorted_ids]
    x_pos = np.arange(n_tokens)

    ax_hdg = fig.add_subplot(gs_bot[0, 0])
    vals = [trajs[d]["step0_dpsi_deg"] for d in sorted_ids]
    colors_hdg = ["tab:red" if v > 0.5 else "tab:green" if v < -0.5 else "tab:gray"
                  for v in vals]
    ax_hdg.bar(x_pos, vals, color=colors_hdg, width=0.7)
    ax_hdg.set_xticks(x_pos)
    ax_hdg.set_xticklabels(labels, fontsize=6, rotation=45)
    ax_hdg.set_ylabel("dpsi (\u00b0 / 2s lookahead)", fontsize=9)
    ax_hdg.set_title("Heading Delta per Token (2s lookahead)", fontsize=10, fontweight="bold")
    ax_hdg.axhline(0, color="black", linewidth=0.5)
    ax_hdg.grid(axis="y", alpha=0.3)

    ax_alt = fig.add_subplot(gs_bot[0, 1])
    vals_alt = [trajs[d]["step0_dalt"] for d in sorted_ids]
    colors_alt = ["tab:orange" if v > 5 else "tab:purple" if v < -5 else "tab:gray"
                  for v in vals_alt]
    ax_alt.bar(x_pos, vals_alt, color=colors_alt, width=0.7)
    ax_alt.set_xticks(x_pos)
    ax_alt.set_xticklabels(labels, fontsize=6, rotation=45)
    ax_alt.set_ylabel("dalt (m / 2s lookahead)", fontsize=9)
    ax_alt.set_title("Altitude Delta per Token (2s lookahead)", fontsize=10, fontweight="bold")
    ax_alt.axhline(0, color="black", linewidth=0.5)
    ax_alt.grid(axis="y", alpha=0.3)

    ax_spd = fig.add_subplot(gs_bot[0, 2])
    vals_spd = [trajs[d]["step0_dspd"] for d in sorted_ids]
    colors_spd = ["tab:blue" if v > 0.5 else "tab:cyan" if v < -0.5 else "tab:gray"
                  for v in vals_spd]
    ax_spd.bar(x_pos, vals_spd, color=colors_spd, width=0.7)
    ax_spd.set_xticks(x_pos)
    ax_spd.set_xticklabels(labels, fontsize=6, rotation=45)
    ax_spd.set_ylabel("dspd (m/s / 2s lookahead)", fontsize=9)
    ax_spd.set_title("Speed Delta per Token (2s lookahead)", fontsize=10, fontweight="bold")
    ax_spd.axhline(0, color="black", linewidth=0.5)
    ax_spd.grid(axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to: {out_path}")

    print("\n=== Token Vocabulary Summary (per-step = 2s lookahead, NOT sum) ===")
    print(f"{'Token':>6} {'raw':>4} {'dpsi°/2s':>9} {'rate°/s':>8} {'dalt_m/2s':>10} {'dspd/2s':>9} {'freq%':>7}  Description")
    print("-" * 85)
    for did in sorted_ids:
        t = trajs[did]
        stat = token_stats.get(did, {})
        raw = dense_to_raw[did]
        ratio = stat.get("ratio", 0) * 100

        dh = t["step0_dpsi_deg"]
        da = t["step0_dalt"]
        ds = t["step0_dspd"]
        rate = t["turn_rate_deg_s"]
        parts = []
        if abs(dh) > 10:
            parts.append("right turn" if dh > 0 else "left turn")
        elif abs(dh) > 2:
            parts.append("slight right" if dh > 0 else "slight left")
        else:
            parts.append("straight")

        if abs(da) > 50:
            parts.append("climb" if da > 0 else "descend")
        elif abs(da) > 5:
            parts.append("slight climb" if da > 0 else "slight descend")
        else:
            parts.append("level")

        if abs(ds) > 5:
            parts.append("accel" if ds > 0 else "decel")
        elif abs(ds) > 1:
            parts.append("slight accel" if ds > 0 else "slight decel")

        desc = ", ".join(parts)
        print(f"  T{did:>3d}  r{raw:>2d}  {dh:>+8.1f}  {rate:>+7.1f}  {da:>+9.1f}  {ds:>+8.1f}  {ratio:>6.1f}%  {desc}")


if __name__ == "__main__":
    main()
