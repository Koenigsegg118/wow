#!/usr/bin/env python3
"""Generate paper-quality token vocabulary figures.

Outputs:
  1. vqvae_token_traj.png   — 2×5 grid of representative token trajectories
  2. vqvae_token_bars.png   — 3 separate bar charts (heading / altitude / speed)

Usage:
    python -m tools.plot_token_vocab_paper
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.vq.vqvae_model import ActionChunkVQVAE

PAPER_FIG_DIR = ROOT.parent / "paper" / "latex-template" / "figures"

# ── Representative tokens: 10 selected by semantic category ──
# Based on actual decoded data; use English labels (CJK fonts unreliable in matplotlib)
SELECTED_TOKENS = [
    (19, "Steady (39.7%)"),         # top-1: near-straight, slight climb
    (8,  "Climb (+73m)"),           # straight + climb, 21%
    (16, "Descend (-95m)"),         # straight + descend, 6.8%
    (26, "Strong climb (+202m)"),   # straight + strong climb, 6%
    (6,  "Slight R turn"),          # right turn +6.6°, 2.7%
    (5,  "Slight L turn"),          # left turn -7.6°, 3.2%
    (13, "Dive + accel"),           # dive -260m + accel, 2.1%
    (14, "L turn + decel"),         # left -16.4° + decel, 1.1%
    (24, "Sharp R turn (+43°)"),    # hard right +43.5°, 0.1%
    (28, "Sharp L turn (-43°)"),    # hard left -42.7°, 0.2%
]


def load_vqvae_and_vocab(vq_ckpt_path, vocab_json_path, device="cpu"):
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


def simulate_trajectory(chunk, init_heading=0.0, init_speed=200.0, dt_step=0.5, lookahead_steps=4):
    avg_dpsi = float(np.mean(chunk[:, 0]))
    avg_dalt = float(np.mean(chunk[:, 1]))
    avg_dspd = float(np.mean(chunk[:, 2]))

    lookahead_sec = lookahead_steps * dt_step
    turn_rate = avg_dpsi / lookahead_sec
    alt_rate = avg_dalt / lookahead_sec
    spd_rate = avg_dspd / lookahead_sec

    heading = init_heading
    x, y = 0.0, 0.0
    speed = init_speed

    traj_x, traj_y = [x], [y]
    for _ in range(8):
        heading += turn_rate * dt_step
        speed = max(50.0, speed + spd_rate * dt_step)
        x += speed * math.sin(heading) * dt_step
        y += speed * math.cos(heading) * dt_step
        traj_x.append(x)
        traj_y.append(y)

    return {
        "x": np.array(traj_x), "y": np.array(traj_y),
        "step0_dpsi_deg": math.degrees(float(chunk[0, 0])),
        "step0_dalt": float(chunk[0, 1]),
        "step0_dspd": float(chunk[0, 2]),
    }


def get_traj_color(t):
    if abs(t["step0_dpsi_deg"]) > 2:
        return "tab:red" if t["step0_dpsi_deg"] > 0 else "tab:green"
    if abs(t["step0_dalt"]) > 50:
        return "tab:orange" if t["step0_dalt"] > 0 else "tab:purple"
    return "tab:blue"


def plot_selected_trajectories(chunks, trajs, token_stats, dense_to_raw):
    """2 rows × 5 cols grid of representative token trajectories."""
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    axes = axes.flatten()

    # compute shared axis range from selected tokens only
    sel_ids = [did for did, _ in SELECTED_TOKENS]
    x_all = np.concatenate([trajs[d]["x"] for d in sel_ids])
    y_all = np.concatenate([trajs[d]["y"] for d in sel_ids])
    margin = 30
    x_range = max(abs(x_all.min()), abs(x_all.max()), 5) + margin
    y_min = min(y_all.min(), 0) - margin
    y_max = max(y_all.max(), 5) + margin

    for i, (did, cat_label) in enumerate(SELECTED_TOKENS):
        ax = axes[i]
        t = trajs[did]
        color = get_traj_color(t)

        ax.plot(t["x"], t["y"], "-o", color=color, markersize=3, linewidth=2.0)
        ax.plot(0, 0, "ks", markersize=5)
        if len(t["x"]) >= 2:
            ax.annotate("", xy=(t["x"][-1], t["y"][-1]),
                        xytext=(t["x"][-2], t["y"][-2]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

        ax.set_xlim(-x_range, x_range)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.3)
        ax.axvline(0, color="gray", lw=0.3)
        ax.tick_params(labelsize=6)

        stat = token_stats.get(did, {})
        ratio = stat.get("ratio", 0) * 100
        ax.set_title(f"T{did}  ({ratio:.1f}%)\n{cat_label}", fontsize=8, fontweight="bold")

        info = (f"$\\Delta\\psi$={t['step0_dpsi_deg']:+.1f}\u00b0\n"
                f"$\\Delta h$={t['step0_dalt']:+.0f} m\n"
                f"$\\Delta v$={t['step0_dspd']:+.1f} m/s")
        ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=6,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="wheat", alpha=0.75))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Representative Token Trajectories (top-down, 4s at 200 m/s)",
                 fontsize=11, fontweight="bold")

    out = PAPER_FIG_DIR / "vqvae_token_traj.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved -> {out}")
    plt.close(fig)


def plot_bar_charts(chunks, trajs, token_stats, dense_to_raw):
    """3 separate bar charts, one per action channel."""
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

    sorted_ids = sorted(chunks.keys())
    n = len(sorted_ids)
    labels = [f"T{d}" for d in sorted_ids]
    x_pos = np.arange(n)

    # ── (a) heading ──
    fig_h, ax_h = plt.subplots(figsize=(8, 2.8))
    vals_h = [trajs[d]["step0_dpsi_deg"] for d in sorted_ids]
    colors_h = ["tab:red" if v > 0.5 else "tab:green" if v < -0.5 else "tab:gray" for v in vals_h]
    ax_h.bar(x_pos, vals_h, color=colors_h, width=0.7)
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax_h.set_ylabel("$\\Delta\\psi$ (\u00b0 / 2s)", fontsize=10)
    ax_h.set_title("(a) Heading Delta per Token", fontsize=11, fontweight="bold")
    ax_h.axhline(0, color="black", lw=0.5)
    ax_h.grid(axis="y", alpha=0.3)
    fig_h.tight_layout()
    out_h = PAPER_FIG_DIR / "vqvae_bar_heading.png"
    fig_h.savefig(out_h, bbox_inches="tight", dpi=200)
    print(f"Saved -> {out_h}")
    plt.close(fig_h)

    # ── (b) altitude ──
    fig_a, ax_a = plt.subplots(figsize=(8, 2.8))
    vals_a = [trajs[d]["step0_dalt"] for d in sorted_ids]
    colors_a = ["tab:orange" if v > 5 else "tab:purple" if v < -5 else "tab:gray" for v in vals_a]
    ax_a.bar(x_pos, vals_a, color=colors_a, width=0.7)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax_a.set_ylabel("$\\Delta h$ (m / 2s)", fontsize=10)
    ax_a.set_title("(b) Altitude Delta per Token", fontsize=11, fontweight="bold")
    ax_a.axhline(0, color="black", lw=0.5)
    ax_a.grid(axis="y", alpha=0.3)
    fig_a.tight_layout()
    out_a = PAPER_FIG_DIR / "vqvae_bar_altitude.png"
    fig_a.savefig(out_a, bbox_inches="tight", dpi=200)
    print(f"Saved -> {out_a}")
    plt.close(fig_a)

    # ── (c) speed ──
    fig_s, ax_s = plt.subplots(figsize=(8, 2.8))
    vals_s = [trajs[d]["step0_dspd"] for d in sorted_ids]
    colors_s = ["tab:blue" if v > 0.5 else "tab:cyan" if v < -0.5 else "tab:gray" for v in vals_s]
    ax_s.bar(x_pos, vals_s, color=colors_s, width=0.7)
    ax_s.set_xticks(x_pos)
    ax_s.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax_s.set_ylabel("$\\Delta v$ (m/s / 2s)", fontsize=10)
    ax_s.set_title("(c) Speed Delta per Token", fontsize=11, fontweight="bold")
    ax_s.axhline(0, color="black", lw=0.5)
    ax_s.grid(axis="y", alpha=0.3)
    fig_s.tight_layout()
    out_s = PAPER_FIG_DIR / "vqvae_bar_speed.png"
    fig_s.savefig(out_s, bbox_inches="tight", dpi=200)
    print(f"Saved -> {out_s}")
    plt.close(fig_s)


def main():
    vq_ckpt = str(ROOT / "checkpoints" / "vqvae_clean_t4_cb64" / "best.pt")
    vocab_json = str(ROOT / "datasets" / "dt2hz_H2s_vqclean_t4_cb64_tok.vocab.json")

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading VQ-VAE and vocab ...")
    vqvae, dense_to_raw, token_stats, token_steps, act_mean, act_std, norm, dev = \
        load_vqvae_and_vocab(vq_ckpt, vocab_json)
    print(f"  token_steps={token_steps}, vocab_size={len(dense_to_raw)}")

    # Validate selected tokens exist
    valid_ids = set(dense_to_raw.keys())
    for did, label in SELECTED_TOKENS:
        if did not in valid_ids:
            print(f"  WARNING: T{did} ({label}) not in vocab, will skip")

    print("Decoding all tokens ...")
    chunks = decode_all_tokens(vqvae, dense_to_raw, token_steps, act_mean, act_std, norm, dev)

    print("Simulating trajectories ...")
    trajs = {}
    for did, chunk in chunks.items():
        trajs[did] = simulate_trajectory(chunk)

    # Print all tokens so we can verify selection
    print("\n=== All tokens summary ===")
    for did in sorted(chunks.keys()):
        t = trajs[did]
        stat = token_stats.get(did, {})
        ratio = stat.get("ratio", 0) * 100
        print(f"  T{did:>2d}  dpsi={t['step0_dpsi_deg']:>+7.1f}°  "
              f"dalt={t['step0_dalt']:>+8.1f}m  "
              f"dspd={t['step0_dspd']:>+7.1f}m/s  "
              f"freq={ratio:>5.1f}%")

    print("\nGenerating figures ...")
    plot_selected_trajectories(chunks, trajs, token_stats, dense_to_raw)
    plot_bar_charts(chunks, trajs, token_stats, dense_to_raw)
    print("Done!")


if __name__ == "__main__":
    main()
