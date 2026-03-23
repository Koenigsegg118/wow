#!/usr/bin/env python
"""Train a VQ-VAE on action chunks extracted from the training NPZ.

Usage (from repo root):
    python -m training.vq.train_vqvae --data datasets/dt2hz_H2s_fighteronly.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from training.vq.chunk_dataset import ActionChunkDataset
from training.vq.vqvae_model import ActionChunkVQVAE

ACT_FIELDS = ("dpsi_rad", "alt_sp_m", "spd_sp_mps")


def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train action-chunk VQ-VAE")
    p.add_argument("--data", type=str, required=True, help="Path to training NPZ")
    p.add_argument("--save_dir", type=str, default="checkpoints/vqvae",
                   help="Directory for checkpoint & metrics")
    p.add_argument("--token_steps", type=int, default=2,
                   help="Time-steps per action chunk (default: 2 → 1 s)")
    p.add_argument("--codebook_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=0.25,
                   help="VQ commitment loss weight")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of data held out for validation")
    p.add_argument("--normalize_action", type=_str2bool, default=True,
                   help="Per-channel z-score normalisation (default: true)")

    # ── chunk extraction ──
    p.add_argument("--chunk_extract_mode", type=str, default="single_fixed",
                   choices=["single_fixed", "single_random", "strided_all"],
                   help="How to extract chunks from source windows")
    p.add_argument("--fixed_offset", type=int, default=0,
                   help="Time offset for single_fixed mode")
    p.add_argument("--chunk_stride", type=int, default=None,
                   help="Stride for strided_all mode (default: token_steps)")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Hard cap on total chunks after all sampling")

    # ── motion-aware sampling ──
    p.add_argument("--lambda_alt", type=float, default=1.0,
                   help="Weight for altitude channel in motion score")
    p.add_argument("--lambda_spd", type=float, default=1.0,
                   help="Weight for speed channel in motion score")
    p.add_argument("--static_keep_ratio", type=float, default=0.25,
                   help="Fraction of static chunks to keep")
    p.add_argument("--light_keep_ratio", type=float, default=0.5,
                   help="Fraction of light-motion chunks to keep")
    p.add_argument("--static_q", type=float, default=0.50,
                   help="Quantile threshold for static bucket boundary")
    p.add_argument("--light_q", type=float, default=0.85,
                   help="Quantile threshold for light bucket boundary")

    return p.parse_args()


# ── metrics helpers ──────────────────────────────────────────────────

def _per_channel_error(x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, dict[str, float]]:
    """Compute per-channel RMSE and MAE.  x, x_hat: [B, token_steps, 3]."""
    diff = x - x_hat
    rmse = diff.pow(2).mean(dim=(0, 1)).sqrt().cpu().tolist()  # [3]
    mae = diff.abs().mean(dim=(0, 1)).cpu().tolist()           # [3]
    return {
        "rmse": {n: round(v, 6) for n, v in zip(ACT_FIELDS, rmse)},
        "mae":  {n: round(v, 6) for n, v in zip(ACT_FIELDS, mae)},
    }


def evaluate(
    model: ActionChunkVQVAE,
    loader: DataLoader,
    device: torch.device,
    ds: ActionChunkDataset,
) -> dict:
    model.eval()
    total_recon = 0.0
    total_vq = 0.0
    all_x = []
    all_xhat = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            x_hat, vq_loss, _ = model(batch)
            recon = F.mse_loss(x_hat, batch)
            total_recon += recon.item() * batch.shape[0]
            total_vq += vq_loss.item() * batch.shape[0]
            all_x.append(batch.cpu())
            all_xhat.append(x_hat.cpu())
            n += batch.shape[0]

    all_x = torch.cat(all_x)
    all_xhat = torch.cat(all_xhat)

    # Normalised-space errors (always available — this is the training loss space)
    norm_err = _per_channel_error(all_x, all_xhat)

    # Raw-space errors (de-normalise first if normalisation was used)
    raw_x = ds.denormalize(all_x)
    raw_xhat = ds.denormalize(all_xhat)
    raw_err = _per_channel_error(raw_x, raw_xhat)

    return {
        "recon_mse": total_recon / n,
        "vq_loss": total_vq / n,
        "norm_rmse": norm_err["rmse"],
        "norm_mae":  norm_err["mae"],
        "raw_rmse":  raw_err["rmse"],
        "raw_mae":   raw_err["mae"],
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- data ----------
    ds = ActionChunkDataset(
        args.data,
        token_steps=args.token_steps,
        normalize=args.normalize_action,
        chunk_extract_mode=args.chunk_extract_mode,
        fixed_offset=args.fixed_offset,
        chunk_stride=args.chunk_stride if args.chunk_stride else args.token_steps,
        lambda_alt=args.lambda_alt,
        lambda_spd=args.lambda_spd,
        static_keep_ratio=args.static_keep_ratio,
        light_keep_ratio=args.light_keep_ratio,
        static_q=args.static_q,
        light_q=args.light_q,
        max_chunks=args.max_chunks,
        seed=args.seed,
    )

    # ── print sampling summary ──
    ss = ds.sampling_stats()
    print(f"\n{'='*60}")
    print(f"  Chunk sampling summary")
    print(f"{'='*60}")
    print(f"  Source windows       : {ss['num_source_windows']:,}")
    print(f"  After extraction     : {ss['num_chunks_after_extract']:,}"
          f"  (mode={args.chunk_extract_mode})")
    if ss.get("motion"):
        mi = ss["motion"]
        print(f"  Motion buckets:")
        print(f"    static       : {mi['static']:>8,}  (score <= {mi['static_threshold']:.4f})"
              f"  → kept {mi['static_kept']:,}  ({args.static_keep_ratio:.0%})")
        print(f"    light        : {mi['light']:>8,}  (score <= {mi['light_threshold']:.4f})"
              f"  → kept {mi['light_kept']:,}  ({args.light_keep_ratio:.0%})")
        print(f"    maneuvering  : {mi['maneuvering']:>8,}"
              f"  → kept {mi['maneuvering_kept']:,}  (100%)")
    print(f"  After motion sample  : {ss['num_chunks_after_motion_sampling']:,}")
    print(f"  Final chunks         : {ss['num_chunks_final']:,}")
    print(f"{'='*60}\n")

    # ── save sampling stats ──
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sampling_stats_full = {
        **ss,
        "chunk_extract_mode": args.chunk_extract_mode,
        "fixed_offset": args.fixed_offset,
        "chunk_stride": args.chunk_stride if args.chunk_stride else args.token_steps,
        "lambda_alt": args.lambda_alt,
        "lambda_spd": args.lambda_spd,
        "static_keep_ratio": args.static_keep_ratio,
        "light_keep_ratio": args.light_keep_ratio,
        "static_q": args.static_q,
        "light_q": args.light_q,
        "max_chunks": args.max_chunks,
        "seed": args.seed,
    }
    with open(save_dir / "sampling_stats.json", "w") as f:
        json.dump(sampling_stats_full, f, indent=2)
    print(f"Saved sampling_stats.json")

    # ── train/val split ──
    n_val = int(len(ds) * args.val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"Chunks: {len(ds)} total, {n_train} train, {n_val} val  "
          f"shape=[{args.token_steps}, 3]  normalize={args.normalize_action}")

    if args.normalize_action:
        m = ds.act_mean.tolist()
        s = ds.act_std.tolist()
        print(f"  mean = [{m[0]:.6f}, {m[1]:.2f}, {m[2]:.2f}]")
        print(f"  std  = [{s[0]:.6f}, {s[1]:.2f}, {s[2]:.2f}]")

    # ---------- model ----------
    model = ActionChunkVQVAE(
        token_steps=args.token_steps,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        beta=args.beta,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Save action normalisation stats (always, even when normalize=False)
    action_stats = {
        "normalize": args.normalize_action,
        "fields": list(ACT_FIELDS),
        "mean": {n: round(v, 8) for n, v in zip(ACT_FIELDS, ds.act_mean.tolist())},
        "std":  {n: round(v, 8) for n, v in zip(ACT_FIELDS, ds.act_std.tolist())},
    }
    with open(save_dir / "action_stats.json", "w") as f:
        json.dump(action_stats, f, indent=2)
    print(f"Saved action_stats.json → {save_dir / 'action_stats.json'}")

    # ---------- train ----------
    best_val_recon = float("inf")
    history: list[dict] = []
    t0 = time.time()

    ckpt_common = {
        "args": vars(args),
        "action_stats": action_stats,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_recon = 0.0
        ep_vq = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            x_hat, vq_loss, _ = model(batch)
            recon_loss = F.mse_loss(x_hat, batch)
            loss = recon_loss + vq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_recon += recon_loss.item() * batch.shape[0]
            ep_vq += vq_loss.item() * batch.shape[0]
            n += batch.shape[0]

        train_recon = ep_recon / n
        train_vq = ep_vq / n

        val = evaluate(model, val_loader, device, ds)
        val_recon = val["recon_mse"]

        improved = val_recon < best_val_recon
        if improved:
            best_val_recon = val_recon
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                **ckpt_common,
            }, save_dir / "best.pt")

        record = {
            "epoch": epoch,
            "train_recon_mse": round(train_recon, 6),
            "train_vq_loss": round(train_vq, 6),
            "val_recon_mse": round(val_recon, 6),
            "val_vq_loss": round(val["vq_loss"], 6),
            # normalised-space
            **{f"val_norm_rmse_{k}": v for k, v in val["norm_rmse"].items()},
            **{f"val_norm_mae_{k}": v for k, v in val["norm_mae"].items()},
            # raw-space (physical units)
            **{f"val_raw_rmse_{k}": v for k, v in val["raw_rmse"].items()},
            **{f"val_raw_mae_{k}": v for k, v in val["raw_mae"].items()},
        }
        history.append(record)

        rr = val["raw_rmse"]
        rm = val["raw_mae"]
        star = " *" if improved else ""
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"train={train_recon:.6f} vq={train_vq:.6f} | "
            f"val={val_recon:.6f} | "
            f"raw_rmse  dpsi={rr['dpsi_rad']:.4f} alt={rr['alt_sp_m']:.2f} spd={rr['spd_sp_mps']:.2f} | "
            f"raw_mae   dpsi={rm['dpsi_rad']:.4f} alt={rm['alt_sp_m']:.2f} spd={rm['spd_sp_mps']:.2f}"
            f"{star}"
        )

    elapsed = time.time() - t0

    # ---------- save ----------
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        **ckpt_common,
    }, save_dir / "last.pt")

    metrics = {
        "args": vars(args),
        "action_stats": action_stats,
        "best_val_recon_mse": round(best_val_recon, 6),
        "total_chunks": len(ds),
        "train_chunks": n_train,
        "val_chunks": n_val,
        "elapsed_s": round(elapsed, 1),
        "device": str(device),
        "history": history,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s.  Best val recon MSE = {best_val_recon:.6f}")
    print(f"Saved to {save_dir}/")


if __name__ == "__main__":
    main()
