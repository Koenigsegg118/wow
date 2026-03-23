#!/usr/bin/env python
"""Unified token inspection & visualisation for VQ-VAE action chunks.

Two modes:
  summary  – codebook usage stats, per-token action profiles, gallery
  animate  – 3D rollout animation for a single token_id

Automatically detects action semantics from the dataset / checkpoint:
  - setpoint: [dpsi_rad, alt_sp_m, spd_sp_mps]   (BC main-line)
  - delta:    [dpsi_rad, dalt_sp_m, dspd_sp_mps]  (VQ clean)

Usage (from repo root):
  python -m training.vq.visualize_tokens --mode summary  --data ... --ckpt ...
  python -m training.vq.visualize_tokens --mode animate  --data ... --ckpt ... --token_id 12
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import warnings
from collections import defaultdict

import numpy as np
import torch

# Fallback defaults (BC main-line)
_DEFAULT_ACT_FIELDS = ("dpsi_rad", "alt_sp_m", "spd_sp_mps")
_DEFAULT_ACT_UNITS = ("rad", "m", "m/s")

_KNOWN_SEMANTICS = {
    ("dpsi_rad", "alt_sp_m", "spd_sp_mps"):   "setpoint",
    ("dpsi_rad", "dalt_sp_m", "dspd_sp_mps"): "delta",
}

_UNITS_MAP = {
    "dpsi_rad": "rad",
    "alt_sp_m": "m",   "dalt_sp_m": "m",
    "spd_sp_mps": "m/s", "dspd_sp_mps": "m/s",
}


# ═════════════════════════════════════════════════════════════════════
#  Action semantics inference
# ═════════════════════════════════════════════════════════════════════

def infer_action_semantics(
    data_path: str,
    ckpt_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Detect action field names and semantics type.

    Priority: NPZ meta > checkpoint metadata > fallback.

    Returns
    -------
    act_fields : tuple of 3 field names
    act_units  : tuple of 3 unit strings
    semantics  : "setpoint" | "delta"
    """
    act_fields = None

    # 1. try NPZ meta
    try:
        with np.load(data_path, allow_pickle=True) as npz:
            if "meta" in npz:
                raw = npz["meta"]
                if isinstance(raw, np.ndarray):
                    meta_str = raw.item() if raw.ndim == 0 else bytes(raw).decode("utf-8")
                else:
                    meta_str = str(raw)
                meta = json.loads(meta_str)
                if "action_semantics" in meta:
                    act_fields = tuple(meta["action_semantics"])
    except Exception:
        pass

    # 2. try checkpoint
    if act_fields is None:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_action_stats = ckpt.get("action_stats", {})
            if "fields" in ckpt_action_stats:
                act_fields = tuple(ckpt_action_stats["fields"])
        except Exception:
            pass

    # 3. fallback
    if act_fields is None:
        warnings.warn(
            "Could not detect action semantics from data or checkpoint. "
            "Falling back to setpoint semantics: "
            f"{_DEFAULT_ACT_FIELDS}",
            stacklevel=2,
        )
        act_fields = _DEFAULT_ACT_FIELDS

    act_units = tuple(_UNITS_MAP.get(f, "?") for f in act_fields)
    semantics = _KNOWN_SEMANTICS.get(act_fields, "setpoint")

    return act_fields, act_units, semantics


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VQ-VAE token inspection & visualisation")

    # common
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="outputs/token_vis")
    p.add_argument("--mode", type=str, required=True,
                   choices=["summary", "animate"])
    p.add_argument("--seed", type=int, default=42)

    # summary
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_samples_per_token", type=int, default=50)
    p.add_argument("--save_gallery", type=_str2bool, default=True)
    p.add_argument("--save_csv", type=_str2bool, default=True)

    # animate
    p.add_argument("--token_id", type=int, default=None)
    p.add_argument("--anim_source", type=str, default="mean",
                   choices=["mean", "sample"])
    p.add_argument("--sample_index", type=int, default=None)
    p.add_argument("--overlay_samples", type=_str2bool, default=False)
    p.add_argument("--num_overlay_samples", type=int, default=5)
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--show_2d_views", type=_str2bool, default=False)
    p.add_argument("--show_action_panel", type=_str2bool, default=False)

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  Loading helpers
# ═════════════════════════════════════════════════════════════════════

def load_model_and_stats(ckpt_path: str, device: torch.device):
    """Return (model, action_stats_dict, training_args_dict)."""
    from training.vq.vqvae_model import ActionChunkVQVAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = ActionChunkVQVAE(
        token_steps=args["token_steps"],
        codebook_size=args["codebook_size"],
        latent_dim=args.get("latent_dim", 32),
        beta=args.get("beta", 0.25),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    action_stats = ckpt.get("action_stats", None)
    return model, action_stats, args


def extract_chunks_and_token_ids(
    model,
    data_path: str,
    action_stats: dict | None,
    training_args: dict,
    device: torch.device,
):
    """Return (raw_chunks [M, T, 3], token_ids [M]) both as numpy arrays.

    raw_chunks are always in **original physical units**.
    """
    from training.vq.chunk_dataset import ActionChunkDataset

    normalize = training_args.get("normalize_action", False)
    token_steps = training_args["token_steps"]

    ds = ActionChunkDataset(data_path, token_steps=token_steps,
                            normalize=normalize,
                            chunk_extract_mode="strided_all",
                            chunk_stride=1,
                            static_keep_ratio=1.0,
                            light_keep_ratio=1.0)

    # Encode in batches
    all_ids = []
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = ds.chunks[i:i + batch_size].to(device)
            _, _, indices = model(batch)
            all_ids.append(indices.cpu())
    token_ids = torch.cat(all_ids).numpy()  # [M]

    # De-normalise chunks to physical units
    raw_chunks = ds.denormalize(ds.chunks).numpy()  # [M, T, 3]

    return raw_chunks, token_ids


# ═════════════════════════════════════════════════════════════════════
#  Token statistics
# ═════════════════════════════════════════════════════════════════════

def summarize_tokens(raw_chunks: np.ndarray, token_ids: np.ndarray,
                     codebook_size: int):
    """Build per-token stats dict keyed by token_id."""
    M = len(token_ids)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, tid in enumerate(token_ids):
        groups[int(tid)].append(idx)

    stats = {}
    for tid in range(codebook_size):
        idxs = groups.get(tid, [])
        count = len(idxs)
        if count == 0:
            stats[tid] = {"count": 0, "usage_ratio": 0.0,
                          "mean": None, "std": None}
            continue
        samples = raw_chunks[idxs]  # [count, T, 3]
        stats[tid] = {
            "count": count,
            "usage_ratio": count / M,
            "mean": samples.mean(axis=0),   # [T, 3]
            "std": samples.std(axis=0),      # [T, 3]
        }
    return stats, groups


# ═════════════════════════════════════════════════════════════════════
#  Simplified kinematic rollout
# ═════════════════════════════════════════════════════════════════════

# Shared rollout hyper-parameters
ROLLOUT_DT = 0.5          # seconds per step
ROLLOUT_X0 = 0.0          # initial east position (m)
ROLLOUT_Y0 = 0.0          # initial north position (m)
ROLLOUT_Z0 = 5000.0       # initial altitude (m)
ROLLOUT_PSI0 = 0.0        # initial heading (rad, 0=North, CW+)
ROLLOUT_V0 = 300.0        # initial speed (m/s)

# Setpoint rollout parameters
ROLLOUT_ALPHA_V = 0.3     # speed 1st-order approach rate per step
ROLLOUT_ALPHA_H = 0.3     # altitude 1st-order approach rate per step
ROLLOUT_MAX_CLIMB = 160.0 # max climb/descent rate (m/s)

# Delta rollout parameters
ROLLOUT_GAMMA_V = 1.0     # fraction of dspd applied per step
ROLLOUT_GAMMA_H = 1.0     # fraction of dalt applied per step


def rollout_setpoint_chunk(
    chunk: np.ndarray,
    *,
    dt: float = ROLLOUT_DT,
    x0: float = ROLLOUT_X0, y0: float = ROLLOUT_Y0,
    z0: float = ROLLOUT_Z0, psi0: float = ROLLOUT_PSI0,
    v0: float = ROLLOUT_V0,
    alpha_v: float = ROLLOUT_ALPHA_V,
    alpha_h: float = ROLLOUT_ALPHA_H,
    max_climb: float = ROLLOUT_MAX_CLIMB,
) -> np.ndarray:
    """Rollout for setpoint semantics [dpsi_rad, alt_sp_m, spd_sp_mps].

    Returns [T+1, 6]: x_east, y_north, z_up, psi, v, step.
    """
    T = chunk.shape[0]
    traj = np.zeros((T + 1, 6), dtype=np.float64)
    x, y, z, psi, v = x0, y0, z0, psi0, v0
    traj[0] = [x, y, z, psi, v, 0]

    for t in range(T):
        dpsi, alt_sp, spd_sp = chunk[t]
        psi = psi + dpsi
        v = v + alpha_v * (spd_sp - v)
        v = max(v, 1.0)
        dz_desired = alpha_h * (alt_sp - z)
        dz_clamped = np.clip(dz_desired, -max_climb * dt, max_climb * dt)
        z = z + dz_clamped
        x = x + v * math.sin(psi) * dt
        y = y + v * math.cos(psi) * dt
        traj[t + 1] = [x, y, z, psi, v, t + 1]

    return traj


def rollout_delta_chunk(
    chunk: np.ndarray,
    *,
    dt: float = ROLLOUT_DT,
    x0: float = ROLLOUT_X0, y0: float = ROLLOUT_Y0,
    z0: float = ROLLOUT_Z0, psi0: float = ROLLOUT_PSI0,
    v0: float = ROLLOUT_V0,
    gamma_v: float = ROLLOUT_GAMMA_V,
    gamma_h: float = ROLLOUT_GAMMA_H,
) -> np.ndarray:
    """Rollout for delta semantics [dpsi_rad, dalt_sp_m, dspd_sp_mps].

    Returns [T+1, 6]: x_east, y_north, z_up, psi, v, step.
    """
    T = chunk.shape[0]
    traj = np.zeros((T + 1, 6), dtype=np.float64)
    x, y, z, psi, v = x0, y0, z0, psi0, v0
    traj[0] = [x, y, z, psi, v, 0]

    for t in range(T):
        dpsi, dalt, dspd = chunk[t]
        psi = psi + dpsi
        v = v + gamma_v * dspd
        v = max(v, 1.0)
        z = z + gamma_h * dalt
        x = x + v * math.sin(psi) * dt
        y = y + v * math.cos(psi) * dt
        traj[t + 1] = [x, y, z, psi, v, t + 1]

    return traj


def rollout_chunk(chunk: np.ndarray, semantics: str, **kw) -> np.ndarray:
    """Dispatch to the correct rollout based on action semantics."""
    if semantics == "delta":
        return rollout_delta_chunk(chunk, **kw)
    return rollout_setpoint_chunk(chunk, **kw)


# ═════════════════════════════════════════════════════════════════════
#  Summary mode
# ═════════════════════════════════════════════════════════════════════

def run_summary(args, raw_chunks, token_ids, codebook_size,
                act_fields, act_units, semantics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stats, groups = summarize_tokens(raw_chunks, token_ids, codebook_size)
    M = len(token_ids)
    token_steps = raw_chunks.shape[1]

    # ── terminal summary ─────────────────────────────────────────
    active_codes = sum(1 for s in stats.values() if s["count"] > 0)
    counts = np.array([stats[t]["count"] for t in range(codebook_size)],
                      dtype=np.float64)
    probs = counts / counts.sum()
    probs_pos = probs[probs > 0]
    perplexity = float(np.exp(-np.sum(probs_pos * np.log(probs_pos))))

    sorted_tids = sorted(stats.keys(), key=lambda t: stats[t]["count"],
                         reverse=True)
    top10_count = sum(stats[t]["count"] for t in sorted_tids[:10])

    print(f"\n{'='*60}")
    print(f"  VQ-VAE Token Summary")
    print(f"{'='*60}")
    print(f"  Action semantics    : {semantics}  {list(act_fields)}")
    print(f"  Total chunks        : {M:,}")
    print(f"  Codebook size       : {codebook_size}")
    print(f"  Active codes        : {active_codes} / {codebook_size}")
    print(f"  Code usage perplexity: {perplexity:.1f}")
    print(f"  Top-10 token share  : {top10_count/M*100:.1f}%")
    print()
    print(f"  {'tid':>5}  {'count':>8}  {'ratio':>7}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*7}")
    for t in sorted_tids[:args.top_k]:
        s = stats[t]
        print(f"  {t:5d}  {s['count']:8d}  {s['usage_ratio']:6.2%}")
    print()

    # ── bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(12, codebook_size * 0.12), 4))
    ax.bar(range(codebook_size), counts, width=1.0, edgecolor="none")
    ax.set_xlabel("token_id")
    ax.set_ylabel("count")
    ax.set_title(f"Token usage  (active={active_codes}, perplexity={perplexity:.1f})")
    fig.tight_layout()
    fig.savefig(save_dir / "token_usage_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved token_usage_bar.png")

    # ── CSV ──────────────────────────────────────────────────────
    if args.save_csv:
        csv_path = save_dir / "token_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["token_id", "count", "usage_ratio"]
            for step in range(token_steps):
                for field in act_fields:
                    header.append(f"{field}_mean_step{step}")
            for step in range(token_steps):
                for field in act_fields:
                    header.append(f"{field}_std_step{step}")
            w.writerow(header)
            for tid in range(codebook_size):
                s = stats[tid]
                row = [tid, s["count"], f"{s['usage_ratio']:.6f}"]
                if s["mean"] is not None:
                    for step in range(token_steps):
                        for c in range(3):
                            row.append(f"{s['mean'][step, c]:.6f}")
                    for step in range(token_steps):
                        for c in range(3):
                            row.append(f"{s['std'][step, c]:.6f}")
                else:
                    row.extend([""] * (token_steps * 3 * 2))
                w.writerow(row)
        print(f"  Saved token_summary.csv")

    # ── per-token detail plots for top-k ─────────────────────────
    step_axis = np.arange(token_steps)
    top_k_tids = sorted_tids[:args.top_k]

    for tid in top_k_tids:
        s = stats[tid]
        if s["count"] == 0:
            continue
        idxs = groups[tid]
        samples = raw_chunks[idxs]

        n_show = min(len(idxs), args.max_samples_per_token)
        show_idxs = np.random.choice(len(idxs), n_show, replace=False)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for c, (ax, fname, unit) in enumerate(
                zip(axes, act_fields, act_units)):
            for si in show_idxs:
                ax.plot(step_axis, samples[si, :, c],
                        color="steelblue", alpha=0.15, linewidth=0.8)
            mean_c = s["mean"][:, c]
            std_c = s["std"][:, c]
            ax.fill_between(step_axis, mean_c - std_c, mean_c + std_c,
                            color="steelblue", alpha=0.25, label="\u00b11 std")
            ax.plot(step_axis, mean_c, color="darkblue", linewidth=2,
                    label="mean")
            ax.set_xlabel("chunk step")
            ax.set_ylabel(f"{fname} ({unit})")
            ax.set_title(fname)
            ax.legend(fontsize=8)
            ax.set_xticks(step_axis)

        fig.suptitle(
            f"token {tid}   count={s['count']}   "
            f"ratio={s['usage_ratio']:.2%}",
            fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_dir / f"token_{tid:03d}_summary.png", dpi=150)
        plt.close(fig)

    print(f"  Saved {len(top_k_tids)} token detail plots")

    # ── gallery ──────────────────────────────────────────────────
    if args.save_gallery and len(top_k_tids) > 0:
        n = len(top_k_tids)
        fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8),
                                 squeeze=False)
        for col, tid in enumerate(top_k_tids):
            s = stats[tid]
            if s["mean"] is None:
                continue
            for row, (fname, unit) in enumerate(zip(act_fields, act_units)):
                ax = axes[row, col]
                mean_c = s["mean"][:, row]
                std_c = s["std"][:, row]
                ax.fill_between(step_axis, mean_c - std_c, mean_c + std_c,
                                color="steelblue", alpha=0.25)
                ax.plot(step_axis, mean_c, color="darkblue", linewidth=1.5)
                ax.set_xticks(step_axis)
                if col == 0:
                    ax.set_ylabel(f"{fname}\n({unit})", fontsize=8)
                else:
                    ax.set_yticklabels([])
                if row == 0:
                    ax.set_title(f"T{tid} ({s['count']})", fontsize=9)
                if row == 2:
                    ax.set_xlabel("step", fontsize=8)
                ax.tick_params(labelsize=7)

        fig.suptitle("Top-k token gallery (mean \u00b1 1 std)", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_dir / "token_gallery.png", dpi=150)
        plt.close(fig)
        print(f"  Saved token_gallery.png")

    print(f"\n  All outputs in {save_dir}/")


# ═════════════════════════════════════════════════════════════════════
#  Animate mode
# ═════════════════════════════════════════════════════════════════════

def run_animate(args, raw_chunks, token_ids, codebook_size,
                act_fields, act_units, semantics):
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.animation as animation

    if args.token_id is None:
        raise ValueError("--token_id is required for animate mode")

    tid = args.token_id
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect samples for this token
    mask = token_ids == tid
    sample_chunks = raw_chunks[mask]
    count = sample_chunks.shape[0]
    total = len(token_ids)
    ratio = count / total if total > 0 else 0.0

    if count == 0:
        print(f"\n  token_id={tid} has 0 samples in the dataset. "
              f"Choose a different token.")
        return

    # ── pick the action chunk ────────────────────────────────────
    if args.anim_source == "mean":
        chunk = sample_chunks.mean(axis=0)
        source_label = "mean"
    else:
        if args.sample_index is not None:
            si = args.sample_index
            if si >= count:
                print(f"  sample_index={si} out of range (count={count}), "
                      f"clamping to {count - 1}")
                si = count - 1
        else:
            si = np.random.randint(count)
        chunk = sample_chunks[si]
        source_label = f"sample#{si}"

    # ── terminal info ────────────────────────────────────────────
    token_steps = chunk.shape[0]
    sem_label = "delta action" if semantics == "delta" else "setpoint action"
    print(f"\n{'='*60}")
    print(f"  Animate token {tid}  ({sem_label})")
    print(f"{'='*60}")
    print(f"  count       : {count:,}")
    print(f"  usage_ratio : {ratio:.2%}")
    print(f"  source      : {source_label}")
    print(f"  semantics   : {semantics}  {list(act_fields)}")
    print(f"  action chunk (physical units):")
    for t in range(token_steps):
        vals = ", ".join(f"{act_fields[c]}={chunk[t,c]:+.4f}"
                         for c in range(3))
        print(f"    step {t}: {vals}")
    print()

    # ── rollout ──────────────────────────────────────────────────
    traj = rollout_chunk(chunk, semantics)

    overlay_trajs = []
    if args.overlay_samples and count > 1:
        n_overlay = min(args.num_overlay_samples, count)
        oidxs = np.random.choice(count, n_overlay, replace=False)
        for oi in oidxs:
            overlay_trajs.append(rollout_chunk(sample_chunks[oi], semantics))

    # ── decide backend ───────────────────────────────────────────
    saving = args.save_path is not None
    if saving:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")

    # ── build figure ─────────────────────────────────────────────
    show_2d = args.show_2d_views
    show_panel = args.show_action_panel

    if show_2d:
        fig = plt.figure(figsize=(16, 10))
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_yz = fig.add_subplot(2, 2, 4)
        view_axes = [ax_xy, ax_xz, ax_yz]
    else:
        fig = plt.figure(figsize=(10, 8))
        ax3d = fig.add_subplot(111, projection="3d")
        view_axes = []

    title_str = (f"token {tid}  ({source_label}, {semantics})  "
                 f"count={count}  ratio={ratio:.2%}")

    # Compute axis limits from all trajectories for stable view
    all_pts = [traj]
    all_pts.extend(overlay_trajs)
    all_pts_arr = np.concatenate(all_pts, axis=0)
    x_all, y_all, z_all = all_pts_arr[:, 0], all_pts_arr[:, 1], all_pts_arr[:, 2]

    def _pad(lo, hi, frac=0.15):
        span = hi - lo
        if span < 1e-3:
            span = 100.0
        return lo - frac * span, hi + frac * span

    xlim = _pad(x_all.min(), x_all.max())
    ylim = _pad(y_all.min(), y_all.max())
    zlim = _pad(z_all.min(), z_all.max())

    def _setup_3d(ax):
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_zlabel("Alt (m)")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    def _setup_2d():
        if not view_axes:
            return
        ax_xy.set_xlabel("East (m)"); ax_xy.set_ylabel("North (m)")
        ax_xy.set_title("Top view (X-Y)")
        ax_xy.set_xlim(xlim); ax_xy.set_ylim(ylim)
        ax_xz.set_xlabel("East (m)"); ax_xz.set_ylabel("Alt (m)")
        ax_xz.set_title("Side view (X-Z)")
        ax_xz.set_xlim(xlim); ax_xz.set_ylim(zlim)
        ax_yz.set_xlabel("North (m)"); ax_yz.set_ylabel("Alt (m)")
        ax_yz.set_title("Side view (Y-Z)")
        ax_yz.set_xlim(ylim); ax_yz.set_ylim(zlim)

    _setup_3d(ax3d)
    _setup_2d()
    for ot in overlay_trajs:
        ax3d.plot(ot[:, 0], ot[:, 1], ot[:, 2],
                  color="gray", alpha=0.3, linewidth=1)
        if view_axes:
            ax_xy.plot(ot[:, 0], ot[:, 1], color="gray", alpha=0.3, lw=1)
            ax_xz.plot(ot[:, 0], ot[:, 2], color="gray", alpha=0.3, lw=1)
            ax_yz.plot(ot[:, 1], ot[:, 2], color="gray", alpha=0.3, lw=1)

    # Action panel text — uses correct field names
    if show_panel:
        panel_lines = [f"Action chunk ({source_label}, {semantics}):"]
        for t in range(token_steps):
            parts = []
            for c in range(3):
                parts.append(f"{act_fields[c]}={chunk[t,c]:+.4f} {act_units[c]}")
            panel_lines.append(f"  step{t}: {'  '.join(parts)}")
        panel_text = "\n".join(panel_lines)
        fig.text(0.02, 0.02, panel_text, fontsize=8, fontfamily="monospace",
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(title_str, fontsize=12, fontweight="bold")

    # ── static PNG shortcut ──────────────────────────────────────
    save_static_png = saving and pathlib.Path(args.save_path).suffix.lower() == ".png"

    if save_static_png:
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                  color="crimson", linewidth=2.5)
        ax3d.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                  "go", markersize=8, label="start")
        ax3d.plot([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                  "rs", markersize=8, label="end")
        ax3d.legend(fontsize=8)
        if view_axes:
            ax_xy.plot(traj[:, 0], traj[:, 1], color="crimson", lw=2)
            ax_xz.plot(traj[:, 0], traj[:, 2], color="crimson", lw=2)
            ax_yz.plot(traj[:, 1], traj[:, 2], color="crimson", lw=2)
            for ax_v, xi, yi in [(ax_xy, 0, 1), (ax_xz, 0, 2), (ax_yz, 1, 2)]:
                ax_v.plot(traj[0, xi], traj[0, yi], "go", ms=6)
                ax_v.plot(traj[-1, xi], traj[-1, yi], "rs", ms=6)

        fig.tight_layout(rect=[0, 0.06 if show_panel else 0, 1, 0.94])
        sp = pathlib.Path(args.save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(sp), dpi=150)
        print(f"  Saved static image -> {sp}")
        plt.close(fig)
        return

    # ── animation objects ────────────────────────────────────────
    n_frames = traj.shape[0]
    line3d, = ax3d.plot([], [], [], color="crimson", linewidth=2.5)
    start_pt = ax3d.plot([traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                         "go", markersize=8, label="start")[0]
    end_marker, = ax3d.plot([], [], [], "rs", markersize=8, label="end")
    ax3d.legend(fontsize=8)

    lines_2d = []
    end_2d = []
    if view_axes:
        for ax in view_axes:
            ln, = ax.plot([], [], color="crimson", linewidth=2)
            em, = ax.plot([], [], "rs", markersize=6)
            lines_2d.append(ln)
            end_2d.append(em)
        ax_xy.plot(traj[0, 0], traj[0, 1], "go", ms=6)
        ax_xz.plot(traj[0, 0], traj[0, 2], "go", ms=6)
        ax_yz.plot(traj[0, 1], traj[0, 2], "go", ms=6)

    def _update(frame):
        i = frame + 1
        seg = traj[:i + 1]
        line3d.set_data(seg[:, 0], seg[:, 1])
        line3d.set_3d_properties(seg[:, 2])
        end_marker.set_data([seg[-1, 0]], [seg[-1, 1]])
        end_marker.set_3d_properties([seg[-1, 2]])
        artists = [line3d, end_marker]

        if view_axes:
            xy_pairs = [
                (seg[:, 0], seg[:, 1]),
                (seg[:, 0], seg[:, 2]),
                (seg[:, 1], seg[:, 2]),
            ]
            for ln, em, (xs, ys) in zip(lines_2d, end_2d, xy_pairs):
                ln.set_data(xs, ys)
                em.set_data([xs[-1]], [ys[-1]])
                artists.extend([ln, em])

        return artists

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames - 1,
        interval=max(300, int(ROLLOUT_DT * 1000)),
        blit=False, repeat=True)

    fig.tight_layout(rect=[0, 0.06 if show_panel else 0, 1, 0.94])

    if saving:
        sp = pathlib.Path(args.save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        suffix = sp.suffix.lower()
        if suffix == ".gif":
            anim.save(str(sp), writer="pillow", fps=2)
        elif suffix in (".mp4", ".mov"):
            anim.save(str(sp), writer="ffmpeg", fps=2)
        elif suffix == ".html":
            html = anim.to_jshtml()
            sp.write_text(html)
        else:
            anim.save(str(sp), fps=2)
        print(f"  Saved animation -> {sp}")
        plt.close(fig)
    else:
        print("  Showing interactive animation (close window to exit) ...")
        plt.show()


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    # Infer action semantics before loading model
    act_fields, act_units, semantics = infer_action_semantics(
        args.data, args.ckpt)
    print(f"  Action semantics: {semantics}  {list(act_fields)}")

    model, action_stats, training_args = load_model_and_stats(args.ckpt, device)
    codebook_size = training_args["codebook_size"]

    raw_chunks, token_ids = extract_chunks_and_token_ids(
        model, args.data, action_stats, training_args, device)

    if args.mode == "summary":
        run_summary(args, raw_chunks, token_ids, codebook_size,
                    act_fields, act_units, semantics)
    elif args.mode == "animate":
        run_animate(args, raw_chunks, token_ids, codebook_size,
                    act_fields, act_units, semantics)


if __name__ == "__main__":
    main()
