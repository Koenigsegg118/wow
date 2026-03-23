#!/usr/bin/env python
"""Autoregressive evaluation and decode-back metrics for Token BC.

Evaluates a trained Token BC model in two modes:
  1. Teacher-forcing (ground-truth previous tokens)
  2. Autoregressive / free-running (model's own predicted tokens)

Then decodes predicted tokens back to continuous actions via frozen VQ-VAE
decoder and computes per-dimension RMSE/MAE against ground-truth actions.

Usage (from repo root):
    python -m training.vq.eval_token_bc_autoreg \
        --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
        --token_bc_ckpt checkpoints/token_bc_t4_cb64/best.pt \
        --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
        --save_dir checkpoints/token_bc_t4_cb64_eval \
        --maneuver_quantile 0.7 --batch 256
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

from training.vq.train_token_bc import (
    TokenBCDataset,
    TokenBCTransformer,
)
from training.vq.vqvae_model import ActionChunkVQVAE


# ── helpers ──────────────────────────────────────────────────────

def load_token_bc(ckpt_path: str, device: torch.device):
    """Load trained Token BC model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = TokenBCTransformer(
        vocab_size=ckpt["vocab_size"],
        obs_dim=8,
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        n_heads=args["n_heads"],
        dropout=0.0,  # no dropout at eval
        n_positions=ckpt["n_tok"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def load_vqvae(ckpt_path: str, device: torch.device):
    """Load frozen VQ-VAE for decode-back."""
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
    return model, ckpt


def decode_tokens_to_actions(
    vqvae: ActionChunkVQVAE,
    raw_token_ids: torch.Tensor,
    normalize: bool,
    act_mean: np.ndarray | None,
    act_std: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    """Decode raw codebook token ids → continuous action chunks.

    Args:
        raw_token_ids: [B, n_tok] int64, raw codebook indices
        normalize: whether tokenizer was trained with normalisation
        act_mean, act_std: action statistics for denormalisation

    Returns:
        actions: [B, n_tok * token_steps, 3] float32
    """
    B, n_tok = raw_token_ids.shape
    token_steps = vqvae.token_steps

    with torch.no_grad():
        # look up codebook embeddings directly
        flat_ids = raw_token_ids.reshape(-1).to(device)  # [B*n_tok]
        z_q = vqvae.vq.embedding(flat_ids)  # [B*n_tok, latent_dim]
        x_hat = vqvae.decoder(z_q)  # [B*n_tok, token_steps*3]
        x_hat = x_hat.reshape(B * n_tok, token_steps, 3).cpu().numpy()

    # denormalise if needed
    if normalize and act_mean is not None:
        x_hat = x_hat * act_std[None, None, :] + act_mean[None, None, :]

    # reshape to [B, n_tok * token_steps, 3]
    x_hat = x_hat.reshape(B, n_tok * token_steps, 3)
    return x_hat.astype(np.float32)


# ── teacher-forcing & autoregressive prediction ──────────────────

@torch.no_grad()
def predict_teacher_forcing(
    model: TokenBCTransformer,
    obs: torch.Tensor,
    tokens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Teacher-forcing prediction. Returns [B, T] predicted token ids."""
    obs, tokens = obs.to(device), tokens.to(device)
    logits = model(obs, tokens)  # [B, T, V]
    return logits.argmax(dim=-1).cpu()  # [B, T]


@torch.no_grad()
def predict_autoregressive(
    model: TokenBCTransformer,
    obs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Free-running autoregressive prediction.

    At each step t, uses the model's own prediction from step t-1.
    Returns [B, T] predicted token ids.
    """
    B, T, obs_dim = obs.shape
    obs = obs.to(device)

    # start with dummy tokens (will be overwritten step by step)
    pred_tokens = torch.zeros(B, T, dtype=torch.long, device=device)

    for t in range(T):
        # build input: at position t, model sees obs[0:t+1] and prev tokens[0:t]
        # but we need full-sequence input for the transformer
        logits = model(obs, pred_tokens)  # [B, T, V]
        pred_t = logits[:, t, :].argmax(dim=-1)  # [B]
        if t < T - 1:
            pred_tokens[:, t] = pred_t
        else:
            pred_tokens[:, t] = pred_t

    return pred_tokens.cpu()  # [B, T]


# ── metrics computation ─────────────────────────────────────────

def compute_token_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    logits_all: np.ndarray | None = None,
) -> dict:
    """Token-level metrics: top-1 accuracy, per-position accuracy, seq exact match."""
    B, T = gt.shape
    match = (pred == gt)

    top1 = match.mean()
    per_pos = [match[:, t].mean() for t in range(T)]
    seq_exact = match.all(axis=1).mean()

    result = {
        "top1_accuracy": float(top1),
        "accuracy_per_position": [float(x) for x in per_pos],
        "sequence_exact_match": float(seq_exact),
        "total_tokens": int(B * T),
        "total_sequences": int(B),
    }

    # top-3/5 from logits if available
    if logits_all is not None:
        logits_t = torch.from_numpy(logits_all).reshape(-1, logits_all.shape[-1])
        gt_flat = torch.from_numpy(gt).reshape(-1)
        for k in [3, 5]:
            _, topk = logits_t.topk(k, dim=-1)
            hit = (topk == gt_flat.unsqueeze(-1)).any(dim=-1)
            result[f"top{k}_accuracy"] = float(hit.float().mean())

    return result


def compute_action_metrics(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
) -> dict:
    """Per-dimension RMSE and MAE for decoded continuous actions.

    Args:
        pred_actions: [B, 20, 3]
        gt_actions:   [B, 20, 3]
    """
    dim_names = ["dpsi_rad", "dalt_sp_m", "dspd_sp_mps"]
    diff = pred_actions - gt_actions  # [B, 20, 3]

    result = {}
    for d, name in enumerate(dim_names):
        per_dim = diff[:, :, d]  # [B, 20]
        rmse = float(np.sqrt((per_dim ** 2).mean()))
        mae = float(np.abs(per_dim).mean())
        result[f"{name}_rmse"] = rmse
        result[f"{name}_mae"] = mae

    # per-step RMSE (averaged across dimensions)
    T = pred_actions.shape[1]
    per_step_rmse = []
    for t in range(T):
        step_diff = diff[:, t, :]  # [B, 3]
        per_step_rmse.append(float(np.sqrt((step_diff ** 2).mean())))
    result["per_step_rmse"] = per_step_rmse

    # overall RMSE
    result["overall_rmse"] = float(np.sqrt((diff ** 2).mean()))
    result["overall_mae"] = float(np.abs(diff).mean())

    return result


def compute_per_token_recall(
    pred: np.ndarray,
    gt: np.ndarray,
    vocab_size: int,
    top_n: int = 10,
) -> list[dict]:
    """Per-token recall for top-N most frequent tokens."""
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    counts = np.bincount(gt_flat, minlength=vocab_size)
    top_ids = np.argsort(-counts)[:top_n]

    rows = []
    for tid in top_ids:
        mask = gt_flat == tid
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        n_correct = int((pred_flat[mask] == tid).sum())
        # what does the model predict instead?
        wrong_mask = mask & (pred_flat != tid)
        if wrong_mask.sum() > 0:
            wrong_preds = pred_flat[wrong_mask]
            wrong_counts = np.bincount(wrong_preds, minlength=vocab_size)
            top_confuse = int(wrong_counts.argmax())
            top_confuse_count = int(wrong_counts[top_confuse])
        else:
            top_confuse = -1
            top_confuse_count = 0

        rows.append({
            "dense_id": int(tid),
            "gt_count": n_total,
            "gt_ratio": round(n_total / len(gt_flat), 4),
            "recall": round(n_correct / n_total, 4),
            "top_confuse_with": top_confuse,
            "top_confuse_count": top_confuse_count,
        })

    return rows


def compute_confusion_top10(
    pred: np.ndarray,
    gt: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Confusion matrix for top-10 most frequent tokens.

    Returns [10, vocab_size] matrix: rows = GT token, cols = predicted token.
    """
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    counts = np.bincount(gt_flat, minlength=vocab_size)
    top_ids = np.argsort(-counts)[:10]

    matrix = np.zeros((10, vocab_size), dtype=np.int64)
    for i, tid in enumerate(top_ids):
        mask = gt_flat == tid
        pred_for_tid = pred_flat[mask]
        matrix[i] = np.bincount(pred_for_tid, minlength=vocab_size)

    return matrix, top_ids


# ── main ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autoregressive evaluation and decode-back metrics for Token BC")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--token_bc_ckpt", type=str, required=True)
    p.add_argument("--vq_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints/token_bc_t4_cb64_eval")
    p.add_argument("--source_data", type=str, default=None,
                   help="Source NPZ with ground-truth actions (auto-detected from meta)")
    p.add_argument("--maneuver_quantile", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── load models ───────────────────────────────────────────
    print("Loading Token BC model...")
    bc_model, bc_ckpt = load_token_bc(args.token_bc_ckpt, device)
    bc_args = bc_ckpt["args"]

    print("Loading VQ-VAE decoder...")
    vqvae, vq_ckpt = load_vqvae(args.vq_ckpt, device)
    vq_args = vq_ckpt["args"]
    vq_action_stats = vq_ckpt.get("action_stats", {})
    normalize = vq_args.get("normalize_action", False)
    token_steps = vq_args["token_steps"]

    act_mean = np.array(
        [vq_action_stats["mean"][k] for k in vq_action_stats["fields"]],
        dtype=np.float32) if normalize else None
    act_std = np.array(
        [vq_action_stats["std"][k] for k in vq_action_stats["fields"]],
        dtype=np.float32) if normalize else None

    # ── load tokenized dataset ────────────────────────────────
    print("Loading tokenized dataset...")
    ds_test = TokenBCDataset(
        args.data, split="test",
        train_frac=bc_args.get("train_frac", 0.8),
        val_frac=bc_args.get("val_frac", 0.1),
        guard=bc_args.get("guard", 100),
    )
    vocab_size = ds_test.vocab_size
    n_tok = ds_test.n_tok

    # load raw metadata for dense→raw mapping
    raw_npz = np.load(args.data, allow_pickle=True)
    meta_raw = raw_npz["meta"]
    if isinstance(meta_raw, np.ndarray):
        ms = meta_raw.item() if meta_raw.ndim == 0 else bytes(meta_raw).decode()
    else:
        ms = str(meta_raw)
    tok_meta = json.loads(ms)
    dense_to_raw = {int(k): int(v) for k, v in tok_meta["dense_to_raw_mapping"].items()}

    # load ground-truth actions from source dataset
    source_path = args.source_data or tok_meta.get("source_data", "")
    source_path_p = pathlib.Path(source_path)
    if not source_path_p.exists():
        # try relative to data dir
        source_path_p = pathlib.Path(args.data).parent / source_path_p.name
    if source_path_p.exists():
        print(f"Loading source actions from: {source_path_p}")
        source_npz = np.load(str(source_path_p), allow_pickle=True)
        gt_actions_all = source_npz["action"].astype(np.float32)  # [N, 20, 3]
    else:
        print(f"WARNING: Source data not found at {source_path}. "
              "Decode-back metrics will be skipped.")
        gt_actions_all = None

    # ── build test loader ─────────────────────────────────────
    loader = torch.utils.data.DataLoader(
        ds_test, batch_size=args.batch, shuffle=False, num_workers=0)

    # maneuver threshold from training set
    ds_train = TokenBCDataset(
        args.data, split="train",
        train_frac=bc_args.get("train_frac", 0.8),
        val_frac=bc_args.get("val_frac", 0.1),
        guard=bc_args.get("guard", 100),
    )
    all_motion_train = ds_train.motion[ds_train.indices].flatten()
    maneuver_threshold = float(np.quantile(all_motion_train, args.maneuver_quantile))

    print(f"\nTest set: {len(ds_test):,} windows")
    print(f"Vocab size: {vocab_size}")
    print(f"Maneuver threshold: {maneuver_threshold:.6f} (quantile={args.maneuver_quantile})")
    print(f"Normalize: {normalize}")

    # ── collect predictions ───────────────────────────────────
    print("\nRunning predictions...")
    all_tf_preds = []    # teacher-forcing
    all_ar_preds = []    # autoregressive
    all_gt = []
    all_motion = []
    all_tf_logits = []
    all_ar_logits = []

    with torch.no_grad():
        for obs, tokens, motion in loader:
            # teacher-forcing
            obs_d, tokens_d = obs.to(device), tokens.to(device)
            tf_logits = bc_model(obs_d, tokens_d)
            tf_preds = tf_logits.argmax(dim=-1).cpu()
            all_tf_preds.append(tf_preds.numpy())
            all_tf_logits.append(tf_logits.cpu().numpy())

            # autoregressive
            ar_preds = predict_autoregressive(bc_model, obs, device)
            # also get logits for top-k: re-run with predicted tokens
            ar_logits = bc_model(obs_d, ar_preds.to(device)).cpu().numpy()
            all_ar_preds.append(ar_preds.numpy())
            all_ar_logits.append(ar_logits)

            all_gt.append(tokens.numpy())
            all_motion.append(motion.numpy())

    tf_preds = np.concatenate(all_tf_preds)     # [N, 5]
    ar_preds = np.concatenate(all_ar_preds)     # [N, 5]
    gt_tokens = np.concatenate(all_gt)          # [N, 5]
    motion = np.concatenate(all_motion)         # [N, 5]
    tf_logits = np.concatenate(all_tf_logits)   # [N, 5, V]
    ar_logits = np.concatenate(all_ar_logits)   # [N, 5, V]

    N_test = len(gt_tokens)

    # ── maneuver mask (per-window: max motion_score > threshold) ──
    motion_flat = motion.flatten()
    maneuver_mask_flat = motion_flat > maneuver_threshold
    # for per-window maneuver: any token in the window is maneuvering
    maneuver_window = motion.max(axis=1) > maneuver_threshold  # [N]

    # maneuver token-level indices
    man_tok_mask = maneuver_mask_flat
    man_win_mask = maneuver_window

    # ── token-level metrics ───────────────────────────────────
    print("\n--- Teacher-Forcing Evaluation ---")
    tf_all = compute_token_metrics(tf_preds, gt_tokens, tf_logits)
    print(f"  top1={tf_all['top1_accuracy']:.4f}  "
          f"top3={tf_all['top3_accuracy']:.4f}  "
          f"top5={tf_all['top5_accuracy']:.4f}  "
          f"seq_exact={tf_all['sequence_exact_match']:.4f}")

    # maneuver subset (token-level)
    man_idx_flat = np.where(man_tok_mask)[0]
    if len(man_idx_flat) > 0:
        tf_man = compute_token_metrics(
            tf_preds.flatten()[man_idx_flat].reshape(-1, 1),
            gt_tokens.flatten()[man_idx_flat].reshape(-1, 1),
            tf_logits.reshape(-1, vocab_size)[man_idx_flat].reshape(-1, 1, vocab_size),
        )
        # seq exact match for maneuver windows
        tf_man["sequence_exact_match"] = float(
            (tf_preds[man_win_mask] == gt_tokens[man_win_mask]).all(axis=1).mean()
        ) if man_win_mask.any() else 0.0
        tf_man["total_maneuver_tokens"] = int(man_idx_flat.shape[0])
        print(f"  maneuver: top1={tf_man['top1_accuracy']:.4f}  "
              f"top3={tf_man['top3_accuracy']:.4f}  "
              f"top5={tf_man['top5_accuracy']:.4f}")
    else:
        tf_man = {"top1_accuracy": 0.0, "total_maneuver_tokens": 0}

    print("\n--- Autoregressive Evaluation ---")
    ar_all = compute_token_metrics(ar_preds, gt_tokens, ar_logits)
    print(f"  top1={ar_all['top1_accuracy']:.4f}  "
          f"top3={ar_all.get('top3_accuracy', 0):.4f}  "
          f"top5={ar_all.get('top5_accuracy', 0):.4f}  "
          f"seq_exact={ar_all['sequence_exact_match']:.4f}")

    if len(man_idx_flat) > 0:
        ar_man = compute_token_metrics(
            ar_preds.flatten()[man_idx_flat].reshape(-1, 1),
            gt_tokens.flatten()[man_idx_flat].reshape(-1, 1),
            ar_logits.reshape(-1, vocab_size)[man_idx_flat].reshape(-1, 1, vocab_size),
        )
        ar_man["sequence_exact_match"] = float(
            (ar_preds[man_win_mask] == gt_tokens[man_win_mask]).all(axis=1).mean()
        ) if man_win_mask.any() else 0.0
        ar_man["total_maneuver_tokens"] = int(man_idx_flat.shape[0])
        print(f"  maneuver: top1={ar_man['top1_accuracy']:.4f}  "
              f"top3={ar_man.get('top3_accuracy', 0):.4f}  "
              f"top5={ar_man.get('top5_accuracy', 0):.4f}")
    else:
        ar_man = {"top1_accuracy": 0.0, "total_maneuver_tokens": 0}

    # ── decode-back continuous action metrics ─────────────────
    decode_tf_metrics = {}
    decode_ar_metrics = {}

    if gt_actions_all is not None:
        gt_actions_test = gt_actions_all[ds_test.indices]  # [N, 20, 3]

        # map dense → raw for VQ-VAE decoder
        def dense_to_raw_ids(dense_ids: np.ndarray) -> torch.Tensor:
            raw = np.vectorize(dense_to_raw.get)(dense_ids)
            return torch.from_numpy(raw.astype(np.int64))

        print("\n--- Decode-Back Metrics (Teacher-Forcing) ---")
        tf_raw = dense_to_raw_ids(tf_preds)
        tf_actions = decode_tokens_to_actions(
            vqvae, tf_raw, normalize, act_mean, act_std, device)
        decode_tf_metrics = compute_action_metrics(tf_actions, gt_actions_test)
        print(f"  dpsi RMSE={decode_tf_metrics['dpsi_rad_rmse']:.4f} rad  "
              f"dalt RMSE={decode_tf_metrics['dalt_sp_m_rmse']:.1f} m  "
              f"dspd RMSE={decode_tf_metrics['dspd_sp_mps_rmse']:.2f} m/s")
        print(f"  overall RMSE={decode_tf_metrics['overall_rmse']:.4f}")

        # maneuver subset decode
        if man_win_mask.any():
            tf_actions_man = tf_actions[man_win_mask]
            gt_actions_man = gt_actions_test[man_win_mask]
            decode_tf_man = compute_action_metrics(tf_actions_man, gt_actions_man)
            decode_tf_metrics["maneuver"] = decode_tf_man
            print(f"  maneuver: dpsi={decode_tf_man['dpsi_rad_rmse']:.4f}  "
                  f"dalt={decode_tf_man['dalt_sp_m_rmse']:.1f}  "
                  f"dspd={decode_tf_man['dspd_sp_mps_rmse']:.2f}")

        print("\n--- Decode-Back Metrics (Autoregressive) ---")
        ar_raw = dense_to_raw_ids(ar_preds)
        ar_actions = decode_tokens_to_actions(
            vqvae, ar_raw, normalize, act_mean, act_std, device)
        decode_ar_metrics = compute_action_metrics(ar_actions, gt_actions_test)
        print(f"  dpsi RMSE={decode_ar_metrics['dpsi_rad_rmse']:.4f} rad  "
              f"dalt RMSE={decode_ar_metrics['dalt_sp_m_rmse']:.1f} m  "
              f"dspd RMSE={decode_ar_metrics['dspd_sp_mps_rmse']:.2f} m/s")
        print(f"  overall RMSE={decode_ar_metrics['overall_rmse']:.4f}")

        if man_win_mask.any():
            ar_actions_man = ar_actions[man_win_mask]
            decode_ar_man = compute_action_metrics(ar_actions_man, gt_actions_man)
            decode_ar_metrics["maneuver"] = decode_ar_man
            print(f"  maneuver: dpsi={decode_ar_man['dpsi_rad_rmse']:.4f}  "
                  f"dalt={decode_ar_man['dalt_sp_m_rmse']:.1f}  "
                  f"dspd={decode_ar_man['dspd_sp_mps_rmse']:.2f}")

    # ── per-token recall ──────────────────────────────────────
    print("\n--- Per-Token Recall (Autoregressive, top-10) ---")
    recall_rows = compute_per_token_recall(ar_preds, gt_tokens, vocab_size)
    print(f"  {'id':>3s}  {'ratio':>6s}  {'recall':>6s}  {'confuse':>8s}")
    for r in recall_rows:
        print(f"  {r['dense_id']:3d}  {r['gt_ratio']:6.2%}  "
              f"{r['recall']:6.2%}  → T{r['top_confuse_with']}"
              f"({r['top_confuse_count']})")

    # ── confusion matrix ──────────────────────────────────────
    conf_matrix, conf_ids = compute_confusion_top10(ar_preds, gt_tokens, vocab_size)

    # ── save outputs ──────────────────────────────────────────
    # eval JSONs
    tf_all["maneuver_quantile"] = args.maneuver_quantile
    tf_all["maneuver_threshold"] = maneuver_threshold
    with open(save_dir / "eval_teacher_all.json", "w") as f:
        json.dump(tf_all, f, indent=2)

    tf_man["maneuver_quantile"] = args.maneuver_quantile
    tf_man["maneuver_threshold"] = maneuver_threshold
    with open(save_dir / "eval_teacher_maneuver.json", "w") as f:
        json.dump(tf_man, f, indent=2)

    ar_all["maneuver_quantile"] = args.maneuver_quantile
    ar_all["maneuver_threshold"] = maneuver_threshold
    with open(save_dir / "eval_autoreg_all.json", "w") as f:
        json.dump(ar_all, f, indent=2)

    ar_man["maneuver_quantile"] = args.maneuver_quantile
    ar_man["maneuver_threshold"] = maneuver_threshold
    with open(save_dir / "eval_autoreg_maneuver.json", "w") as f:
        json.dump(ar_man, f, indent=2)

    # decode metrics
    with open(save_dir / "decode_teacher_metrics.json", "w") as f:
        json.dump(decode_tf_metrics, f, indent=2)
    with open(save_dir / "decode_autoreg_metrics.json", "w") as f:
        json.dump(decode_ar_metrics, f, indent=2)

    # per-token recall CSV
    recall_path = save_dir / "per_token_recall.csv"
    with open(recall_path, "w") as f:
        f.write("dense_id,gt_count,gt_ratio,recall,top_confuse_with,top_confuse_count\n")
        for r in recall_rows:
            f.write(f"{r['dense_id']},{r['gt_count']},{r['gt_ratio']},"
                    f"{r['recall']},{r['top_confuse_with']},{r['top_confuse_count']}\n")

    # confusion matrix CSV
    conf_path = save_dir / "confusion_top10.csv"
    with open(conf_path, "w") as f:
        header = "gt_token," + ",".join(f"pred_{i}" for i in range(vocab_size))
        f.write(header + "\n")
        for i, tid in enumerate(conf_ids):
            row = ",".join(str(x) for x in conf_matrix[i])
            f.write(f"{tid},{row}\n")

    # ── summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Evaluation Summary")
    print(f"{'='*65}")
    print(f"  {'':30s}  {'Teacher':>10s}  {'AutoReg':>10s}")
    print(f"  {'ALL top-1':30s}  "
          f"{tf_all['top1_accuracy']:10.4f}  "
          f"{ar_all['top1_accuracy']:10.4f}")
    print(f"  {'ALL top-3':30s}  "
          f"{tf_all.get('top3_accuracy',0):10.4f}  "
          f"{ar_all.get('top3_accuracy',0):10.4f}")
    print(f"  {'ALL seq exact match':30s}  "
          f"{tf_all['sequence_exact_match']:10.4f}  "
          f"{ar_all['sequence_exact_match']:10.4f}")
    print(f"  {'MANEUVER top-1':30s}  "
          f"{tf_man.get('top1_accuracy',0):10.4f}  "
          f"{ar_man.get('top1_accuracy',0):10.4f}")
    if decode_tf_metrics and decode_ar_metrics:
        print(f"  {'DECODE dpsi RMSE (rad)':30s}  "
              f"{decode_tf_metrics['dpsi_rad_rmse']:10.4f}  "
              f"{decode_ar_metrics['dpsi_rad_rmse']:10.4f}")
        print(f"  {'DECODE dalt RMSE (m)':30s}  "
              f"{decode_tf_metrics['dalt_sp_m_rmse']:10.1f}  "
              f"{decode_ar_metrics['dalt_sp_m_rmse']:10.1f}")
        print(f"  {'DECODE dspd RMSE (m/s)':30s}  "
              f"{decode_tf_metrics['dspd_sp_mps_rmse']:10.2f}  "
              f"{decode_ar_metrics['dspd_sp_mps_rmse']:10.2f}")
    print(f"{'='*65}")

    print(f"\nSaved {len(list(save_dir.glob('*')))} files to {save_dir}/")
    for p in sorted(save_dir.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
