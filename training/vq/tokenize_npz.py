#!/usr/bin/env python
"""Freeze a trained VQ-VAE tokenizer and encode a dataset into token sequences.

Produces a tokenized NPZ ready for downstream sequence modelling (e.g. GPT-style
action-token policy).

Usage (from repo root):
    python -m training.vq.tokenize_npz \
        --data datasets/dt2hz_H2s_vqclean.npz \
        --ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
        --out datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from training.vq.vqvae_model import ActionChunkVQVAE


# ── helpers ──────────────────────────────────────────────────────────

def compute_motion_score(
    chunks: np.ndarray,
    lambda_alt: float = 1.0,
    lambda_spd: float = 1.0,
) -> np.ndarray:
    """Per-chunk motion score.  chunks: [M, token_steps, 3]."""
    abs_vals = np.abs(chunks)
    dpsi = abs_vals[:, :, 0].mean(axis=1)
    dalt = abs_vals[:, :, 1].mean(axis=1) / 100.0
    dspd = abs_vals[:, :, 2].mean(axis=1) / 10.0
    return (dpsi + lambda_alt * dalt + lambda_spd * dspd).astype(np.float32)


def load_tokenizer(ckpt_path: str, device: torch.device):
    """Load frozen VQ-VAE from checkpoint.  Returns (model, action_stats, args)."""
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
    return model, ckpt.get("action_stats", {}), args


def encode_chunks(
    model: ActionChunkVQVAE,
    chunks: np.ndarray,
    normalize: bool,
    act_mean: np.ndarray,
    act_std: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Encode chunks → raw token ids.  Returns [M] int64."""
    # Normalise if the tokenizer was trained with normalisation
    if normalize:
        chunks_input = (chunks - act_mean[None, None, :]) / act_std[None, None, :]
    else:
        chunks_input = chunks

    chunks_t = torch.from_numpy(chunks_input.astype(np.float32))
    all_ids = []
    with torch.no_grad():
        for i in range(0, len(chunks_t), batch_size):
            batch = chunks_t[i : i + batch_size].to(device)
            _, _, indices = model(batch)
            all_ids.append(indices.cpu().numpy())
    return np.concatenate(all_ids).astype(np.int64)


# ── main ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tokenize a VQ clean dataset with a frozen VQ-VAE")
    p.add_argument("--data", type=str, required=True,
                   help="Path to source NPZ (e.g. datasets/dt2hz_H2s_vqclean.npz)")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to VQ-VAE checkpoint (e.g. checkpoints/.../best.pt)")
    p.add_argument("--out", type=str, required=True,
                   help="Output tokenized NPZ path")
    p.add_argument("--lambda_alt", type=float, default=1.0)
    p.add_argument("--lambda_spd", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load tokenizer ───────────────────────────────────────────
    model, action_stats, train_args = load_tokenizer(args.ckpt, device)
    token_steps = train_args["token_steps"]
    codebook_size = train_args["codebook_size"]
    normalize = train_args.get("normalize_action", False)

    act_mean = np.array(
        [action_stats["mean"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None
    act_std = np.array(
        [action_stats["std"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None

    print(f"Tokenizer: token_steps={token_steps}  codebook={codebook_size}"
          f"  normalize={normalize}")
    print(f"Checkpoint: {args.ckpt}")

    # ── load source data ─────────────────────────────────────────
    raw = np.load(args.data, allow_pickle=True)
    actions = raw["action"]    # [N, T, 3]
    obs = raw["obs"]           # [N, T, 8]
    N, T, _ = actions.shape
    assert T % token_steps == 0, (
        f"Window length T={T} not divisible by token_steps={token_steps}")
    n_tok = T // token_steps   # tokens per window

    # Detect action semantics
    action_semantics = ["dpsi_rad", "dalt_sp_m", "dspd_sp_mps"]  # default
    if "meta" in raw:
        try:
            meta_raw = raw["meta"]
            if isinstance(meta_raw, np.ndarray):
                ms = meta_raw.item() if meta_raw.ndim == 0 else bytes(meta_raw).decode()
            else:
                ms = str(meta_raw)
            m = json.loads(ms)
            if "action_semantics" in m:
                action_semantics = m["action_semantics"]
        except Exception:
            pass

    print(f"Source: {args.data}")
    print(f"  N={N:,}  T={T}  tokens_per_window={n_tok}")
    print(f"  action_semantics={action_semantics}")

    # ── deterministic non-overlapping chunk split ────────────────
    # [N, T, 3] → [N, n_tok, token_steps, 3]
    chunks_4d = actions.reshape(N, n_tok, token_steps, 3)
    # flatten to [N*n_tok, token_steps, 3] for encoding
    chunks_flat = chunks_4d.reshape(N * n_tok, token_steps, 3)

    print(f"Chunks: {N*n_tok:,} total ({N:,} × {n_tok})")

    # ── encode ───────────────────────────────────────────────────
    raw_ids_flat = encode_chunks(
        model, chunks_flat, normalize, act_mean, act_std, device)
    raw_ids = raw_ids_flat.reshape(N, n_tok)  # [N, 5]

    # ── active code remap ────────────────────────────────────────
    unique_raw = np.sort(np.unique(raw_ids_flat))
    K = len(unique_raw)
    raw_to_dense = {int(r): int(d) for d, r in enumerate(unique_raw)}
    dense_to_raw = {int(d): int(r) for d, r in enumerate(unique_raw)}

    dense_ids_flat = np.array([raw_to_dense[int(x)] for x in raw_ids_flat],
                              dtype=np.int64)
    dense_ids = dense_ids_flat.reshape(N, n_tok)  # [N, 5]

    print(f"Active codes: {K} / {codebook_size}")
    print(f"Dense vocab: [0, {K-1}]")

    # ── obs at token starts ──────────────────────────────────────
    start_indices = [i * token_steps for i in range(n_tok)]  # [0, 4, 8, 12, 16]
    obs_tok_start = obs[:, start_indices, :]  # [N, 5, 8]

    # ── motion scores ────────────────────────────────────────────
    motion_flat = compute_motion_score(
        chunks_flat, args.lambda_alt, args.lambda_spd)
    motion = motion_flat.reshape(N, n_tok)  # [N, 5]

    # ── token usage stats ────────────────────────────────────────
    raw_counts = {}
    for rid in raw_ids_flat:
        raw_counts[int(rid)] = raw_counts.get(int(rid), 0) + 1
    total = len(raw_ids_flat)

    dense_counts = {}
    for did in dense_ids_flat:
        dense_counts[int(did)] = dense_counts.get(int(did), 0) + 1

    # top-10 raw
    top10_raw = sorted(raw_counts.items(), key=lambda x: -x[1])[:10]
    # top-10 dense
    top10_dense = sorted(dense_counts.items(), key=lambda x: -x[1])[:10]

    print(f"\nTop-10 raw tokens:")
    for rid, cnt in top10_raw:
        print(f"  raw={rid:3d}  dense={raw_to_dense[rid]:2d}"
              f"  count={cnt:>9,}  ratio={cnt/total:.2%}")

    # ── build metadata ───────────────────────────────────────────
    meta = {
        "token_steps": token_steps,
        "tokens_per_window": n_tok,
        "window_length": T,
        "codebook_size_original": codebook_size,
        "active_vocab_size": K,
        "action_semantics": action_semantics,
        "tokenization_mode": "deterministic_nonoverlap",
        "tokenizer_checkpoint": str(pathlib.Path(args.ckpt).resolve()),
        "source_data": str(pathlib.Path(args.data).resolve()),
        "normalize": normalize,
        "lambda_alt": args.lambda_alt,
        "lambda_spd": args.lambda_spd,
        "raw_to_dense_mapping": raw_to_dense,
        "dense_to_raw_mapping": dense_to_raw,
    }

    # ── save NPZ ─────────────────────────────────────────────────
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_path),
        obs_full=obs.astype(np.float32),
        obs_tok_start=obs_tok_start.astype(np.float32),
        token_ids_raw=raw_ids,
        token_ids=dense_ids,
        motion_score=motion,
        active_raw_codes=unique_raw,
        meta=json.dumps(meta),
    )
    print(f"\nSaved: {out_path}")

    # ── save meta.json ───────────────────────────────────────────
    meta_json = {
        **meta,
        "num_windows": N,
        "total_tokens": int(N * n_tok),
        "top10_raw_usage": [
            {"raw_code": rid, "dense_id": raw_to_dense[rid],
             "count": cnt, "ratio": round(cnt / total, 4)}
            for rid, cnt in top10_raw
        ],
        "top10_dense_usage": [
            {"dense_id": did, "raw_code": dense_to_raw[did],
             "count": cnt, "ratio": round(cnt / total, 4)}
            for did, cnt in top10_dense
        ],
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)
    print(f"Saved: {meta_path}")

    # ── save vocab.json ──────────────────────────────────────────
    vocab = {
        "active_vocab_size": K,
        "codebook_size_original": codebook_size,
        "raw_code_to_dense_id": {str(k): v for k, v in raw_to_dense.items()},
        "dense_id_to_raw_code": {str(k): v for k, v in dense_to_raw.items()},
        "token_stats": [
            {
                "dense_id": d,
                "raw_code": int(dense_to_raw[d]),
                "count": dense_counts.get(d, 0),
                "ratio": round(dense_counts.get(d, 0) / total, 6),
            }
            for d in range(K)
        ],
    }
    vocab_path = out_path.with_suffix(".vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved: {vocab_path}")

    # ── summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Tokenization complete")
    print(f"{'='*60}")
    print(f"  Windows           : {N:,}")
    print(f"  Tokens per window : {n_tok}")
    print(f"  Total tokens      : {N*n_tok:,}")
    print(f"  Active vocab (K)  : {K}")
    print(f"  Dense id range    : [0, {K-1}]")
    print(f"  Output files:")
    print(f"    {out_path}")
    print(f"    {meta_path}")
    print(f"    {vocab_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
