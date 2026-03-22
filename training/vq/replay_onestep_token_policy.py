#!/usr/bin/env python
"""Replay one-step token policy in oracle and self-fed modes.

Two replay modes on the test split of the tokenized dataset:
  A. oracle-prev-token: uses ground-truth prev_token (upper bound)
  B. self-fed-prev-token: uses model's own prediction as prev_token (deployment)

For each mode, computes token accuracy (top-1/3/5, per-position, seq exact match),
decode-back continuous action RMSE/MAE, per-token recall, and drift diagnostics.

Optionally exports bridge-friendly rollout data and sample visualizations.

Usage (from repo root):
    python -m training.vq.replay_onestep_token_policy \
        --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
        --policy_ckpt checkpoints/onestep_token_bc_t4_cb64/best.pt \
        --vq_ckpt checkpoints/vqvae_clean_t4_cb64/best.pt \
        --save_dir checkpoints/onestep_token_bc_t4_cb64_replay \
        --maneuver_quantile 0.7 --batch 512 \
        --num_vis_samples 5 --export_bridge_rollout true
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from training.vq.train_onestep_token_bc import OneStepTokenBC
from training.vq.vqvae_model import ActionChunkVQVAE
from training.vq.train_token_bc import block_split
from training.vq.ego_obs_utils import (
    extract_token_boundary_history,
    ego_transform_current_anchor,
)


# ── loaders ──────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = OneStepTokenBC(
        vocab_size=ckpt["vocab_size"],
        obs_dim=8,
        obs_hist_len=ckpt["obs_hist_len"],
        hidden_dim=a["hidden_dim"],
        n_layers=a["num_layers"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def load_vqvae(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ActionChunkVQVAE(
        token_steps=a["token_steps"],
        codebook_size=a["codebook_size"],
        latent_dim=a.get("latent_dim", 32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def decode_raw_tokens(
    vqvae: ActionChunkVQVAE,
    raw_ids: np.ndarray,
    normalize: bool,
    act_mean: np.ndarray | None,
    act_std: np.ndarray | None,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    """Decode raw codebook ids → continuous action chunks [M, token_steps, 3]."""
    token_steps = vqvae.token_steps
    all_chunks = []
    with torch.no_grad():
        for i in range(0, len(raw_ids), batch_size):
            ids_t = torch.from_numpy(raw_ids[i:i + batch_size].astype(np.int64)).to(device)
            z_q = vqvae.vq.embedding(ids_t)
            x = vqvae.decoder(z_q).reshape(-1, token_steps, 3).cpu().numpy()
            if normalize and act_mean is not None:
                x = x * act_std[None, None, :] + act_mean[None, None, :]
            all_chunks.append(x)
    return np.concatenate(all_chunks).astype(np.float32)


# ── replay ───────────────────────────────────────────────────────

@torch.no_grad()
def replay_windows(
    policy: OneStepTokenBC,
    obs_full: np.ndarray,
    gt_tokens: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    device: torch.device,
    obs_hist_len: int = 4,
    token_steps: int = 4,
):
    """Run oracle and self-fed replay on all windows.

    Args:
        obs_full:   [N, 20, 8]  full window observations (global coords)
        gt_tokens:  [N, 5] dense token ids
        obs_hist_len: history length (token boundary steps)
        token_steps:  raw steps per token (4)

    Returns:
        oracle_preds:  [N, 5] int64
        selffed_preds: [N, 5] int64
        oracle_logits: [N, 5, V] float32
        selffed_logits:[N, 5, V] float32
    """
    N, T_full, obs_dim = obs_full.shape
    n_tok = gt_tokens.shape[1]
    V = policy.vocab_size
    BOS = V  # <start> sentinel
    H = obs_hist_len

    oracle_preds = np.zeros((N, n_tok), dtype=np.int64)
    selffed_preds = np.zeros((N, n_tok), dtype=np.int64)
    oracle_logits = np.zeros((N, n_tok, V), dtype=np.float32)
    selffed_logits = np.zeros((N, n_tok, V), dtype=np.float32)

    # pre-build all obs_hist for all windows and positions
    # [N, n_tok, H, 8] — ego-transformed and normalized
    all_obs = np.zeros((N, n_tok, H, obs_dim), dtype=np.float32)
    for j in range(n_tok):
        # extract token-boundary history for all windows at position j
        hist_raw = np.stack([
            extract_token_boundary_history(obs_full[i], j, token_steps, H)
            for i in range(N)
        ])  # [N, H, 8]
        hist_ego = ego_transform_current_anchor(hist_raw)  # [N, H, 8]
        all_obs[:, j] = (hist_ego - obs_mean) / obs_std

    # process in batches
    batch_size = 4096
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        gt_batch = gt_tokens[start:end]  # [B, 5]
        sf_prev = np.full(B, BOS, dtype=np.int64)

        for j in range(n_tok):
            obs_t = torch.from_numpy(all_obs[start:end, j]).float().to(device)  # [B, H, 8]

            # oracle
            if j == 0:
                or_prev = np.full(B, BOS, dtype=np.int64)
            else:
                or_prev = gt_batch[:, j - 1]
            or_prev_t = torch.from_numpy(or_prev).long().to(device)

            or_log = policy(obs_t, or_prev_t).cpu().numpy()
            or_pred = or_log.argmax(axis=-1)
            oracle_preds[start:end, j] = or_pred
            oracle_logits[start:end, j] = or_log

            # self-fed
            sf_prev_t = torch.from_numpy(sf_prev).long().to(device)
            sf_log = policy(obs_t, sf_prev_t).cpu().numpy()
            sf_pred = sf_log.argmax(axis=-1)
            selffed_preds[start:end, j] = sf_pred
            selffed_logits[start:end, j] = sf_log

            sf_prev = sf_pred.copy()

    return oracle_preds, selffed_preds, oracle_logits, selffed_logits


# ── metrics ──────────────────────────────────────────────────────

def token_metrics(
    preds: np.ndarray,
    gt: np.ndarray,
    logits: np.ndarray,
    motion: np.ndarray,
    man_threshold: float,
    vocab_size: int,
) -> dict:
    """Compute token-level metrics for one replay mode."""
    N, T = gt.shape
    match = (preds == gt)

    # all-token
    top1 = float(match.mean())
    per_pos = [float(match[:, t].mean()) for t in range(T)]
    seq_exact = float(match.all(axis=1).mean())

    # top-k from logits
    logits_flat = torch.from_numpy(logits.reshape(-1, vocab_size))
    gt_flat = torch.from_numpy(gt.reshape(-1))
    topk_acc = {}
    for k in [3, 5]:
        _, topk = logits_flat.topk(k, dim=-1)
        hit = (topk == gt_flat.unsqueeze(-1)).any(dim=-1)
        topk_acc[k] = float(hit.float().mean())

    # maneuver subset (token-level)
    man_mask = motion.flatten() > man_threshold
    man_result = {}
    if man_mask.any():
        man_match = match.flatten()[man_mask]
        man_result["top1"] = float(man_match.mean())
        man_logits = logits_flat[man_mask]
        man_gt = gt_flat[man_mask]
        for k in [3, 5]:
            _, topk = man_logits.topk(k, dim=-1)
            hit = (topk == man_gt.unsqueeze(-1)).any(dim=-1)
            man_result[f"top{k}"] = float(hit.float().mean())
        man_result["total"] = int(man_mask.sum())
    else:
        man_result = {"top1": 0.0, "top3": 0.0, "top5": 0.0, "total": 0}

    # per-token recall (top-10)
    preds_flat = preds.flatten()
    gt_flat_np = gt.flatten()
    counts = np.bincount(gt_flat_np, minlength=vocab_size)
    top_ids = np.argsort(-counts)[:10]
    recall = []
    for tid in top_ids:
        mask = gt_flat_np == tid
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((preds_flat[mask] == tid).sum())
        # top confuse
        wrong = mask & (preds_flat != tid)
        if wrong.sum() > 0:
            wc = np.bincount(preds_flat[wrong], minlength=vocab_size)
            top_c = int(wc.argmax())
        else:
            top_c = -1
        recall.append({
            "dense_id": int(tid), "count": n,
            "recall": round(correct / n, 4),
            "top_confuse": top_c,
        })

    return {
        "top1_accuracy": top1,
        "top3_accuracy": topk_acc[3],
        "top5_accuracy": topk_acc[5],
        "accuracy_per_position": per_pos,
        "sequence_exact_match": seq_exact,
        "total_sequences": int(N),
        "maneuver": man_result,
        "per_token_recall": recall,
    }


def action_metrics(pred_actions: np.ndarray, gt_actions: np.ndarray) -> dict:
    """Per-dim RMSE/MAE for [M, T, 3] action arrays."""
    dim_names = ["dpsi_rad", "dalt_sp_m", "dspd_sp_mps"]
    diff = pred_actions - gt_actions
    result = {}
    for d, name in enumerate(dim_names):
        col = diff[..., d]
        result[f"{name}_rmse"] = float(np.sqrt((col ** 2).mean()))
        result[f"{name}_mae"] = float(np.abs(col).mean())
    result["overall_rmse"] = float(np.sqrt((diff ** 2).mean()))
    return result


# ── visualization ────────────────────────────────────────────────

def plot_sample(
    idx: int,
    gt_tokens: np.ndarray,
    or_tokens: np.ndarray,
    sf_tokens: np.ndarray,
    gt_actions_20: np.ndarray,
    or_actions_20: np.ndarray,
    sf_actions_20: np.ndarray,
    save_path: pathlib.Path,
):
    """Plot GT vs oracle vs self-fed for one sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)

    # token sequence
    ax = axes[0]
    x = np.arange(5)
    ax.plot(x, gt_tokens, "ko-", label="GT", markersize=8)
    ax.plot(x, or_tokens, "bs--", label="Oracle", markersize=6)
    ax.plot(x, sf_tokens, "r^--", label="Self-fed", markersize=6)
    ax.set_ylabel("Token ID")
    ax.set_xlabel("Position")
    ax.set_title(f"Sample {idx}: Token Sequence")
    ax.legend()
    ax.set_xticks(x)

    # action dims
    t = np.arange(20) * 0.5  # seconds
    dim_names = ["dpsi (rad)", "dalt (m)", "dspd (m/s)"]
    for d in range(3):
        ax = axes[d + 1]
        ax.plot(t, gt_actions_20[:, d], "k-", label="GT", linewidth=1.5)
        ax.plot(t, or_actions_20[:, d], "b--", label="Oracle", linewidth=1)
        ax.plot(t, sf_actions_20[:, d], "r--", label="Self-fed", linewidth=1)
        ax.set_ylabel(dim_names[d])
        if d == 2:
            ax.set_xlabel("Time (s)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay one-step token policy (oracle vs self-fed)")
    p.add_argument("--data", required=True)
    p.add_argument("--policy_ckpt", required=True)
    p.add_argument("--vq_ckpt", required=True)
    p.add_argument("--save_dir", default="checkpoints/onestep_token_bc_t4_cb64_replay")
    p.add_argument("--maneuver_quantile", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--num_vis_samples", type=int, default=5)
    p.add_argument("--export_bridge_rollout", type=str, default="true",
                   choices=["true", "false"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── load models ───────────────────────────────────────────
    print("Loading policy...")
    policy, pol_ckpt = load_policy(args.policy_ckpt, device)
    pol_args = pol_ckpt["args"]
    obs_mean = pol_ckpt["obs_mean"]
    obs_std = pol_ckpt["obs_std"]
    vocab_size = pol_ckpt["vocab_size"]
    obs_hist_len = pol_ckpt["obs_hist_len"]

    print("Loading VQ-VAE...")
    vqvae, vq_ckpt = load_vqvae(args.vq_ckpt, device)
    vq_args = vq_ckpt["args"]
    token_steps = vq_args["token_steps"]
    normalize = vq_args.get("normalize_action", False)
    action_stats = vq_ckpt.get("action_stats", {})
    act_mean = np.array(
        [action_stats["mean"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None
    act_std = np.array(
        [action_stats["std"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None

    # ── load tokenized dataset ────────────────────────────────
    print("Loading data...")
    raw = np.load(args.data, allow_pickle=True)
    obs_full = raw["obs_full"].astype(np.float32)              # [N, 20, 8]
    token_ids = raw["token_ids"].astype(np.int64)             # [N, 5]
    token_ids_raw = raw["token_ids_raw"].astype(np.int64)     # [N, 5]
    motion_score = raw["motion_score"].astype(np.float32)     # [N, 5]

    meta_raw = raw["meta"]
    if isinstance(meta_raw, np.ndarray):
        ms = meta_raw.item() if meta_raw.ndim == 0 else bytes(meta_raw).decode()
    else:
        ms = str(meta_raw)
    tok_meta = json.loads(ms)
    dense_to_raw = {int(k): int(v) for k, v in tok_meta["dense_to_raw_mapping"].items()}

    # test split (same as training used)
    N = len(token_ids)
    train_frac = pol_args.get("train_frac", 0.8)
    val_frac = pol_args.get("val_frac", 0.1)
    guard = pol_args.get("guard", 100)

    # use block_split from train_token_bc
    splits = block_split(N, train_frac, val_frac, guard)
    test_idx = splits["test"]
    # also use train split for maneuver threshold
    train_idx = splits["train"]

    obs_full_test = obs_full[test_idx]          # [Nt, 20, 8]
    tok_test = token_ids[test_idx]            # [Nt, 5]
    tok_raw_test = token_ids_raw[test_idx]    # [Nt, 5]
    motion_test = motion_score[test_idx]      # [Nt, 5]

    # maneuver threshold from train
    train_motion = motion_score[train_idx].flatten()
    man_threshold = float(np.quantile(train_motion, args.maneuver_quantile))

    Nt = len(test_idx)
    print(f"\nTest windows: {Nt:,}")
    print(f"Maneuver threshold: {man_threshold:.6f} (q={args.maneuver_quantile})")

    # ── load source actions for GT comparison ─────────────────
    source_path = tok_meta.get("source_data", "")
    source_p = pathlib.Path(source_path)
    if not source_p.exists():
        source_p = pathlib.Path(args.data).parent / source_p.name
    gt_actions_all = None
    if source_p.exists():
        print(f"Loading source actions: {source_p}")
        gt_actions_all = np.load(str(source_p), allow_pickle=True)["action"].astype(np.float32)

    # ── replay ────────────────────────────────────────────────
    print("\nRunning replay...")
    or_preds, sf_preds, or_logits, sf_logits = replay_windows(
        policy, obs_full_test, tok_test, obs_mean, obs_std, device,
        obs_hist_len, token_steps)

    # ── token metrics ─────────────────────────────────────────
    print("\n--- Oracle-prev-token ---")
    or_metrics = token_metrics(or_preds, tok_test, or_logits, motion_test,
                               man_threshold, vocab_size)
    print(f"  top1={or_metrics['top1_accuracy']:.4f}  "
          f"top3={or_metrics['top3_accuracy']:.4f}  "
          f"seq_exact={or_metrics['sequence_exact_match']:.4f}")
    print(f"  maneuver top1={or_metrics['maneuver']['top1']:.4f}")
    print(f"  per-pos: {['%.4f' % x for x in or_metrics['accuracy_per_position']]}")

    print("\n--- Self-fed-prev-token ---")
    sf_metrics = token_metrics(sf_preds, tok_test, sf_logits, motion_test,
                               man_threshold, vocab_size)
    print(f"  top1={sf_metrics['top1_accuracy']:.4f}  "
          f"top3={sf_metrics['top3_accuracy']:.4f}  "
          f"seq_exact={sf_metrics['sequence_exact_match']:.4f}")
    print(f"  maneuver top1={sf_metrics['maneuver']['top1']:.4f}")
    print(f"  per-pos: {['%.4f' % x for x in sf_metrics['accuracy_per_position']]}")

    # ── decode-back ───────────────────────────────────────────
    def map_dense_to_raw(dense_ids):
        return np.vectorize(dense_to_raw.get)(dense_ids).astype(np.int64)

    or_raw = map_dense_to_raw(or_preds)   # [Nt, 5]
    sf_raw = map_dense_to_raw(sf_preds)   # [Nt, 5]

    # decode all predictions to [Nt*5, 4, 3]
    or_chunks = decode_raw_tokens(vqvae, or_raw.flatten(), normalize,
                                  act_mean, act_std, device)
    sf_chunks = decode_raw_tokens(vqvae, sf_raw.flatten(), normalize,
                                  act_mean, act_std, device)
    gt_chunks = decode_raw_tokens(vqvae, tok_raw_test.flatten(), normalize,
                                  act_mean, act_std, device)

    # reshape to [Nt, 20, 3]
    or_actions_20 = or_chunks.reshape(Nt, 5 * token_steps, 3)
    sf_actions_20 = sf_chunks.reshape(Nt, 5 * token_steps, 3)
    gt_actions_decoded = gt_chunks.reshape(Nt, 5 * token_steps, 3)

    # use raw source actions if available, else use decoded GT
    if gt_actions_all is not None:
        gt_actions_ref = gt_actions_all[test_idx]
    else:
        gt_actions_ref = gt_actions_decoded

    print("\n--- Decode-Back (vs raw GT actions) ---")
    or_act_m = action_metrics(or_actions_20, gt_actions_ref)
    sf_act_m = action_metrics(sf_actions_20, gt_actions_ref)

    print(f"  Oracle:   dpsi={or_act_m['dpsi_rad_rmse']:.4f}  "
          f"dalt={or_act_m['dalt_sp_m_rmse']:.1f}  "
          f"dspd={or_act_m['dspd_sp_mps_rmse']:.2f}")
    print(f"  Self-fed: dpsi={sf_act_m['dpsi_rad_rmse']:.4f}  "
          f"dalt={sf_act_m['dalt_sp_m_rmse']:.1f}  "
          f"dspd={sf_act_m['dspd_sp_mps_rmse']:.2f}")

    # maneuver subset decode
    man_win = motion_test.max(axis=1) > man_threshold
    or_act_man = action_metrics(or_actions_20[man_win], gt_actions_ref[man_win]) \
        if man_win.any() else {}
    sf_act_man = action_metrics(sf_actions_20[man_win], gt_actions_ref[man_win]) \
        if man_win.any() else {}

    if or_act_man:
        print(f"  Oracle  (man): dpsi={or_act_man['dpsi_rad_rmse']:.4f}  "
              f"dalt={or_act_man['dalt_sp_m_rmse']:.1f}  "
              f"dspd={or_act_man['dspd_sp_mps_rmse']:.2f}")
        print(f"  Self-fed(man): dpsi={sf_act_man['dpsi_rad_rmse']:.4f}  "
              f"dalt={sf_act_man['dalt_sp_m_rmse']:.1f}  "
              f"dspd={sf_act_man['dspd_sp_mps_rmse']:.2f}")

    # ── drift diagnostics ─────────────────────────────────────
    drift = {
        "top1_drop": or_metrics["top1_accuracy"] - sf_metrics["top1_accuracy"],
        "top3_drop": or_metrics["top3_accuracy"] - sf_metrics["top3_accuracy"],
        "maneuver_top1_drop": or_metrics["maneuver"]["top1"] - sf_metrics["maneuver"]["top1"],
        "dpsi_rmse_increase": sf_act_m["dpsi_rad_rmse"] - or_act_m["dpsi_rad_rmse"],
        "dalt_rmse_increase": sf_act_m["dalt_sp_m_rmse"] - or_act_m["dalt_sp_m_rmse"],
        "dspd_rmse_increase": sf_act_m["dspd_sp_mps_rmse"] - or_act_m["dspd_sp_mps_rmse"],
    }

    print(f"\n--- Drift (self-fed vs oracle) ---")
    print(f"  top1 drop:     {drift['top1_drop']:+.4f}")
    print(f"  man top1 drop: {drift['maneuver_top1_drop']:+.4f}")
    print(f"  dpsi RMSE Δ:   {drift['dpsi_rmse_increase']:+.4f} rad")
    print(f"  dalt RMSE Δ:   {drift['dalt_rmse_increase']:+.1f} m")
    print(f"  dspd RMSE Δ:   {drift['dspd_rmse_increase']:+.2f} m/s")

    # ── save JSONs ────────────────────────────────────────────
    with open(save_dir / "oracle_token_metrics.json", "w") as f:
        json.dump(or_metrics, f, indent=2)
    with open(save_dir / "selffed_token_metrics.json", "w") as f:
        json.dump(sf_metrics, f, indent=2)
    with open(save_dir / "oracle_decode_metrics.json", "w") as f:
        json.dump(or_act_m, f, indent=2)
    with open(save_dir / "selffed_decode_metrics.json", "w") as f:
        json.dump(sf_act_m, f, indent=2)
    if or_act_man:
        with open(save_dir / "oracle_decode_maneuver.json", "w") as f:
            json.dump(or_act_man, f, indent=2)
        with open(save_dir / "selffed_decode_maneuver.json", "w") as f:
            json.dump(sf_act_man, f, indent=2)
    with open(save_dir / "drift_diagnostics.json", "w") as f:
        json.dump(drift, f, indent=2)

    # per-token recall CSV (self-fed)
    with open(save_dir / "per_token_recall.csv", "w") as f:
        f.write("dense_id,count,recall,top_confuse\n")
        for r in sf_metrics["per_token_recall"]:
            f.write(f"{r['dense_id']},{r['count']},{r['recall']},{r['top_confuse']}\n")

    # ── visualizations ────────────────────────────────────────
    if args.num_vis_samples > 0:
        vis_dir = save_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        # pick diverse samples: some static, some maneuvering
        mean_motion = motion_test.mean(axis=1)
        sorted_idx = np.argsort(mean_motion)
        n_vis = min(args.num_vis_samples, Nt)
        # pick from quantiles to get diversity
        vis_indices = sorted_idx[np.linspace(0, Nt - 1, n_vis, dtype=int)]

        print(f"\nGenerating {n_vis} visualizations...")
        for vi, si in enumerate(vis_indices):
            plot_sample(
                idx=int(test_idx[si]),
                gt_tokens=tok_test[si],
                or_tokens=or_preds[si],
                sf_tokens=sf_preds[si],
                gt_actions_20=gt_actions_ref[si],
                or_actions_20=or_actions_20[si],
                sf_actions_20=sf_actions_20[si],
                save_path=vis_dir / f"sample_{vi:02d}.png",
            )
        print(f"  Saved {n_vis} plots to {vis_dir}/")

    # ── bridge rollout export ─────────────────────────────────
    if args.export_bridge_rollout == "true":
        print("\nExporting bridge rollout...")
        # self-fed actions in [Nt, 5, 4, 3] format
        sf_chunks_shaped = sf_chunks.reshape(Nt, 5, token_steps, 3)

        bridge_path = save_dir / "bridge_rollout.npz"
        np.savez(
            str(bridge_path),
            sample_ids=test_idx.astype(np.int64),
            predicted_tokens=sf_preds.astype(np.int64),
            predicted_tokens_raw=sf_raw.astype(np.int64),
            decoded_action_chunks=sf_chunks_shaped.astype(np.float32),
            obs_full=obs_full_test.astype(np.float32),
            gt_tokens=tok_test.astype(np.int64),
        )
        print(f"  Saved: {bridge_path}")
        print(f"    sample_ids: [{Nt}]")
        print(f"    predicted_tokens: [{Nt}, 5]")
        print(f"    decoded_action_chunks: [{Nt}, 5, {token_steps}, 3]")

    # ── summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Replay Summary")
    print(f"{'='*65}")
    print(f"  {'':30s}  {'Oracle':>10s}  {'Self-fed':>10s}  {'Drift':>10s}")
    print(f"  {'ALL top-1':30s}  "
          f"{or_metrics['top1_accuracy']:10.4f}  "
          f"{sf_metrics['top1_accuracy']:10.4f}  "
          f"{drift['top1_drop']:+10.4f}")
    print(f"  {'ALL top-3':30s}  "
          f"{or_metrics['top3_accuracy']:10.4f}  "
          f"{sf_metrics['top3_accuracy']:10.4f}  "
          f"{drift['top3_drop']:+10.4f}")
    print(f"  {'ALL seq exact match':30s}  "
          f"{or_metrics['sequence_exact_match']:10.4f}  "
          f"{sf_metrics['sequence_exact_match']:10.4f}  "
          f"{'':>10s}")
    print(f"  {'MANEUVER top-1':30s}  "
          f"{or_metrics['maneuver']['top1']:10.4f}  "
          f"{sf_metrics['maneuver']['top1']:10.4f}  "
          f"{drift['maneuver_top1_drop']:+10.4f}")
    print(f"  {'DECODE dpsi RMSE (rad)':30s}  "
          f"{or_act_m['dpsi_rad_rmse']:10.4f}  "
          f"{sf_act_m['dpsi_rad_rmse']:10.4f}  "
          f"{drift['dpsi_rmse_increase']:+10.4f}")
    print(f"  {'DECODE dalt RMSE (m)':30s}  "
          f"{or_act_m['dalt_sp_m_rmse']:10.1f}  "
          f"{sf_act_m['dalt_sp_m_rmse']:10.1f}  "
          f"{drift['dalt_rmse_increase']:+10.1f}")
    print(f"  {'DECODE dspd RMSE (m/s)':30s}  "
          f"{or_act_m['dspd_sp_mps_rmse']:10.2f}  "
          f"{sf_act_m['dspd_sp_mps_rmse']:10.2f}  "
          f"{drift['dspd_rmse_increase']:+10.2f}")
    print(f"{'='*65}")

    print(f"\nSaved {len(list(save_dir.glob('*')))} items to {save_dir}/")


if __name__ == "__main__":
    main()
