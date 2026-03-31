#!/usr/bin/env python
"""Plot VQ-VAE training loss curves for E0–E4 experiments.

Generates a 2-row figure for thesis:
  Row 1: val_recon_mse vs epoch
  Row 2: val_vq_loss vs epoch

Usage (from wow/ root):
    python tools/plot_vqvae_loss_curves.py
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── experiment configs ──────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "E0: baseline\n(t=2, cb=128, stride=1)",
        "short": "E0",
        "path": "checkpoints/vqvae_clean/metrics.json",
        "color": "#6b7280",  # gray
        "ls": "--",
    },
    {
        "name": "E1: dedup+downsample\n(t=2, cb=128)",
        "short": "E1",
        "path": "checkpoints/vqvae_clean_dedup/metrics.json",
        "color": "#ef4444",  # red
        "ls": "--",
    },
    {
        "name": "E2: single_random\n(t=2, cb=128)",
        "short": "E2",
        "path": "checkpoints/vqvae_clean_r1/metrics.json",
        "color": "#f59e0b",  # amber
        "ls": "-.",
    },
    {
        "name": "E3: single_random\n(t=4, cb=128)",
        "short": "E3",
        "path": "checkpoints/vqvae_clean_t4/metrics.json",
        "color": "#3b82f6",  # blue
        "ls": "-.",
    },
    {
        "name": "E4: single_random\n(t=4, cb=64) ★",
        "short": "E4",
        "path": "checkpoints/vqvae_clean_t4_cb64/metrics.json",
        "color": "#10b981",  # green
        "ls": "-",
    },
]

ROOT = pathlib.Path(__file__).resolve().parent.parent  # wow/
OUT_DIR = ROOT / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)


def load_history(path: str) -> list[dict]:
    full = ROOT / path
    with open(full) as f:
        data = json.load(f)
    return data["history"]


def main():
    # ── matplotlib 中文支持 ──
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "figure.dpi": 200,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for exp in EXPERIMENTS:
        hist = load_history(exp["path"])
        epochs = [h["epoch"] for h in hist]
        val_recon = [h["val_recon_mse"] for h in hist]
        val_vq = [h["val_vq_loss"] for h in hist]

        lw = 2.0 if "E4" in exp["short"] else 1.2

        axes[0].plot(epochs, val_recon, label=exp["short"],
                     color=exp["color"], ls=exp["ls"], lw=lw)
        axes[1].plot(epochs, val_vq, label=exp["short"],
                     color=exp["color"], ls=exp["ls"], lw=lw)

    # ── subplot 0: recon MSE ──
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Recon MSE")
    axes[0].set_title("(a) Reconstruction Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # ── subplot 1: VQ loss ──
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation VQ Loss")
    axes[1].set_title("(b) VQ Commitment Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.suptitle("VQ-VAE Training Curves: E0–E4", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = OUT_DIR / "vqvae_loss_curves.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved → {out_path}")

    # ── 也保存到 paper figures 目录 ──
    paper_fig = ROOT.parent / "paper" / "latex-template" / "figures" / "vqvae_loss_curves.png"
    fig.savefig(paper_fig, bbox_inches="tight", dpi=200)
    print(f"Saved → {paper_fig}")

    plt.close(fig)


if __name__ == "__main__":
    main()
