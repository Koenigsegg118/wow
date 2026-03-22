#!/usr/bin/env python
"""Expand 5-token windowed dataset into one-step samples for receding-horizon policy.

For each window (N) and each token position (j=0..4), creates one training sample:
  - obs_hist:    [H, 8]  token-boundary observation history with current-step
                          ego-centric transform applied
  - prev_token:  int64   previous token (or vocab_size as <start> for j=0)
  - target_token: int64  ground-truth token at position j
  - motion_score: float32
  - window_idx:  int64   source window index (for traceability)
  - token_pos:   int64   position within window (0..4)
  - split_label: int64   0=train, 1=val, 2=test

Block split is done at window level BEFORE expanding to prevent leakage.

History is sampled at token boundary timesteps (every token_steps=4 raw steps),
then transformed to current-step anchored ego-centric coordinates.

Usage (from repo root):
    python -m training.vq.build_onestep_token_dataset \
        --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
        --out datasets/dt2hz_H2s_vqclean_t4_cb64_onestep_h4.npz \
        --history_len 4
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from training.vq.train_token_bc import block_split
from training.vq.ego_obs_utils import (
    extract_token_boundary_history,
    ego_transform_current_anchor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build one-step token dataset from windowed tokenized NPZ")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--history_len", type=int, default=4,
                   help="Number of token-boundary obs timesteps (>=2 recommended)")
    p.add_argument("--include_prev_token", type=str, default="true",
                   choices=["true", "false"])
    p.add_argument("--train_frac", type=float, default=0.80)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--guard", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    include_prev = args.include_prev_token == "true"
    H = args.history_len

    # ── load source ───────────────────────────────────────────
    raw = np.load(args.data, allow_pickle=True)
    obs_full = raw["obs_full"].astype(np.float32)      # [N, 20, 8]
    token_ids = raw["token_ids"].astype(np.int64)       # [N, 5]
    motion_score = raw["motion_score"].astype(np.float32)  # [N, 5]

    meta_raw = raw["meta"]
    if isinstance(meta_raw, np.ndarray):
        ms = meta_raw.item() if meta_raw.ndim == 0 else bytes(meta_raw).decode()
    else:
        ms = str(meta_raw)
    src_meta = json.loads(ms)

    N, n_tok = token_ids.shape
    token_steps = src_meta["token_steps"]  # 4
    vocab_size = src_meta["active_vocab_size"]  # 30
    obs_dim = obs_full.shape[-1]  # 8

    print(f"Source: {args.data}")
    print(f"  N={N:,}  n_tok={n_tok}  token_steps={token_steps}  vocab={vocab_size}")
    print(f"  history_len={H}  include_prev_token={include_prev}")

    # ── block split at window level ───────────────────────────
    splits = block_split(N, args.train_frac, args.val_frac, args.guard)
    # create per-window split label: 0=train, 1=val, 2=test, -1=guard
    window_split = np.full(N, -1, dtype=np.int64)
    window_split[splits["train"]] = 0
    window_split[splits["val"]] = 1
    window_split[splits["test"]] = 2

    n_used = (window_split >= 0).sum()
    print(f"\nBlock split (window-level):")
    print(f"  train={len(splits['train']):,}  val={len(splits['val']):,}  "
          f"test={len(splits['test']):,}  guard={N - n_used:,}")

    # ── expand to one-step samples ────────────────────────────
    # only include windows that are in a split (not guard)
    valid_mask = window_split >= 0
    valid_indices = np.where(valid_mask)[0]
    M = len(valid_indices) * n_tok

    out_obs_hist = np.zeros((M, H, obs_dim), dtype=np.float32)
    out_prev_token = np.zeros(M, dtype=np.int64)
    out_target_token = np.zeros(M, dtype=np.int64)
    out_motion = np.zeros(M, dtype=np.float32)
    out_window_idx = np.zeros(M, dtype=np.int64)
    out_token_pos = np.zeros(M, dtype=np.int64)
    out_split = np.zeros(M, dtype=np.int64)

    idx = 0
    for wi in valid_indices:
        for j in range(n_tok):
            # extract token-boundary history (global coords)
            obs_hist_raw = extract_token_boundary_history(
                obs_full[wi], j, token_steps, H)  # [H, 8]

            # apply current-step anchored ego transform
            out_obs_hist[idx] = ego_transform_current_anchor(obs_hist_raw)

            # prev token
            if include_prev and j > 0:
                out_prev_token[idx] = token_ids[wi, j - 1]
            else:
                out_prev_token[idx] = vocab_size  # <start> sentinel

            out_target_token[idx] = token_ids[wi, j]
            out_motion[idx] = motion_score[wi, j]
            out_window_idx[idx] = wi
            out_token_pos[idx] = j
            out_split[idx] = window_split[wi]
            idx += 1

    assert idx == M

    # ── compute obs stats from train split ────────────────────
    train_mask = out_split == 0
    obs_mean = out_obs_hist[train_mask].mean(axis=(0, 1))  # [8]
    obs_std = out_obs_hist[train_mask].std(axis=(0, 1))     # [8]
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)

    print(f"\nEgo-transformed obs stats (train):")
    names = ["dx'", "dy'", "z_u", "vx'", "vy'", "vz_u", "heading'", "speed"]
    for i, name in enumerate(names):
        print(f"  {name:10s}  mean={obs_mean[i]:10.2f}  std={obs_std[i]:10.2f}")

    # ── save ──────────────────────────────────────────────────
    out_meta = {
        "source_data": str(pathlib.Path(args.data).resolve()),
        "history_len": H,
        "history_mode": "token_boundary_current_anchor_ego",
        "include_prev_token": include_prev,
        "vocab_size": vocab_size,
        "token_steps": token_steps,
        "n_tok_per_window": n_tok,
        "total_samples": M,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "guard": args.guard,
        "split_counts": {
            "train": int(train_mask.sum()),
            "val": int((out_split == 1).sum()),
            "test": int((out_split == 2).sum()),
        },
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_path),
        obs_hist=out_obs_hist,
        prev_token=out_prev_token,
        target_token=out_target_token,
        motion_score=out_motion,
        window_idx=out_window_idx,
        token_pos=out_token_pos,
        split_label=out_split,
        obs_mean=obs_mean,
        obs_std=obs_std,
        meta=json.dumps(out_meta),
    )

    print(f"\nSaved: {out_path}")
    print(f"  total samples: {M:,}")
    print(f"  train: {out_meta['split_counts']['train']:,}")
    print(f"  val:   {out_meta['split_counts']['val']:,}")
    print(f"  test:  {out_meta['split_counts']['test']:,}")
    print(f"  obs_hist shape: [{H}, {obs_dim}]")
    print(f"  prev_token range: [0, {vocab_size}] (BOS={vocab_size})")

    # also save meta as json
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(out_meta, f, indent=2)
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
