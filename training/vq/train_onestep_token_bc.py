#!/usr/bin/env python
"""One-step receding-horizon token policy baseline.

Predicts one action token at a time, given current-step anchored ego-centric
observation history and the previous token.  Deployable in a receding-horizon
loop: predict token → decode to 4-step action → execute → repeat.

Usage (from repo root):
    python -m training.vq.train_onestep_token_bc \
        --data datasets/dt2hz_H2s_vqclean_t4_cb64_onestep_h4.npz \
        --save_dir checkpoints/onestep_token_bc_t4_cb64_h4 \
        --epochs 50 --batch 512 --lr 3e-4 \
        --hidden_dim 128 --dropout 0.1 \
        --class_weight_mode inverse_sqrt \
        --maneuver_quantile 0.7
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── dataset ──────────────────────────────────────────────────────

class OneStepDataset(Dataset):
    """One-step token prediction dataset (pre-split via split_label)."""

    def __init__(self, data_path: str, split: str = "train"):
        raw = np.load(data_path, allow_pickle=True)
        self.obs_hist = raw["obs_hist"].astype(np.float32)      # [M, H, 8]
        self.prev_token = raw["prev_token"].astype(np.int64)    # [M]
        self.target_token = raw["target_token"].astype(np.int64)  # [M]
        self.motion = raw["motion_score"].astype(np.float32)    # [M]
        self.token_pos = raw["token_pos"].astype(np.int64)      # [M]
        split_label = raw["split_label"].astype(np.int64)       # [M]

        self.obs_mean = raw["obs_mean"].astype(np.float32)  # [8]
        self.obs_std = raw["obs_std"].astype(np.float32)    # [8]

        meta_raw = raw["meta"]
        if isinstance(meta_raw, np.ndarray):
            ms = meta_raw.item() if meta_raw.ndim == 0 else bytes(meta_raw).decode()
        else:
            ms = str(meta_raw)
        self.meta = json.loads(ms)
        self.vocab_size = self.meta["vocab_size"]
        # support both old ("obs_hist_len") and new ("history_len") meta keys
        self.obs_hist_len = self.meta.get("history_len",
                                          self.meta.get("obs_hist_len", 1))

        # filter by split
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.indices = np.where(split_label == split_id)[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        obs = (self.obs_hist[j] - self.obs_mean) / self.obs_std  # [H, 8]
        return (
            torch.from_numpy(obs),
            torch.tensor(self.prev_token[j], dtype=torch.long),
            torch.tensor(self.target_token[j], dtype=torch.long),
            torch.tensor(self.motion[j], dtype=torch.float32),
        )


# ── class weights ────────────────────────────────────────────────

def compute_class_weights(
    dataset: OneStepDataset,
    mode: str = "inverse_sqrt",
) -> torch.Tensor:
    """Per-class weights from training token frequencies."""
    tokens = dataset.target_token[dataset.indices]
    V = dataset.vocab_size
    counts = np.bincount(tokens, minlength=V).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    if mode == "none":
        weights = np.ones(V, dtype=np.float64)
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
        weights *= V / weights.sum()
    elif mode == "effective_num":
        beta = 0.999
        effective = (1.0 - beta) / (1.0 - np.power(beta, counts))
        weights = effective
        weights *= V / weights.sum()
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")

    return torch.from_numpy(weights).float()


# ── model ────────────────────────────────────────────────────────

class OneStepTokenBC(nn.Module):
    """MLP-based one-step token policy.

    Input: obs_hist [B, H, obs_dim] + prev_token [B]
    Output: logits [B, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        obs_dim: int = 8,
        obs_hist_len: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.obs_hist_len = obs_hist_len

        # obs encoder: flatten history then project
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim * obs_hist_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # prev token embedding (+1 for <start> sentinel)
        self.tok_embed = nn.Embedding(vocab_size + 1, hidden_dim)

        # fusion MLP
        layers = []
        in_dim = hidden_dim * 2  # obs_enc + tok_embed
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, vocab_size))
        self.head = nn.Sequential(*layers)

    def forward(self, obs_hist: torch.Tensor, prev_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_hist:   [B, H, obs_dim]
            prev_token: [B] int64

        Returns:
            logits: [B, vocab_size]
        """
        B = obs_hist.shape[0]
        obs_flat = obs_hist.reshape(B, -1)  # [B, H * obs_dim]
        obs_enc = self.obs_encoder(obs_flat)  # [B, hidden_dim]
        tok_enc = self.tok_embed(prev_token)  # [B, hidden_dim]
        fused = torch.cat([obs_enc, tok_enc], dim=-1)  # [B, hidden_dim * 2]
        return self.head(fused)  # [B, vocab_size]


# ── evaluation ───────────────────────────────────────────────────

def compute_majority_baseline(dataset: OneStepDataset, maneuver_threshold=None):
    """Majority-token baseline accuracy."""
    tokens = dataset.target_token[dataset.indices]
    motion = dataset.motion[dataset.indices]
    V = dataset.vocab_size

    counts = np.bincount(tokens, minlength=V)
    maj_id = int(counts.argmax())
    total = len(tokens)

    result = {
        "majority_token_id": maj_id,
        "majority_accuracy": int(counts[maj_id]) / total,
        "total_tokens": total,
    }

    if maneuver_threshold is not None:
        man_mask = motion > maneuver_threshold
        man_total = int(man_mask.sum())
        if man_total > 0:
            man_tok = tokens[man_mask]
            man_counts = np.bincount(man_tok, minlength=V)
            man_maj_id = int(man_counts.argmax())
            result["maneuver_majority_token_id"] = man_maj_id
            result["maneuver_majority_accuracy"] = int(man_counts[man_maj_id]) / man_total
            result["maneuver_total_tokens"] = man_total
        else:
            result["maneuver_majority_accuracy"] = 0.0
            result["maneuver_total_tokens"] = 0

    return result


@torch.no_grad()
def evaluate(
    model: OneStepTokenBC,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    maneuver_threshold: float,
    n_tok_positions: int = 5,
) -> dict:
    """Evaluate one-step model."""
    model.eval()

    all_logits = []
    all_targets = []
    all_motion = []
    all_pos = []

    for obs_hist, prev_tok, target, motion in loader:
        logits = model(obs_hist.to(device), prev_tok.to(device))
        all_logits.append(logits.cpu())
        all_targets.append(target)
        all_motion.append(motion)

    # need token_pos for per-position accuracy
    # we don't have it from the loader directly, so re-fetch from dataset
    # Actually we can compute it from the accumulation order

    logits_cat = torch.cat(all_logits)      # [M, V]
    targets_cat = torch.cat(all_targets)    # [M]
    motion_cat = torch.cat(all_motion)      # [M]

    total = len(targets_cat)
    preds = logits_cat.argmax(dim=-1)
    match = (preds == targets_cat)

    # top-k accuracy
    result = {"total_tokens": total}
    for k in [1, 3, 5]:
        if k >= vocab_size:
            result[f"top{k}_accuracy"] = 1.0
            continue
        _, topk = logits_cat.topk(k, dim=-1)
        hit = (topk == targets_cat.unsqueeze(-1)).any(dim=-1)
        result[f"top{k}_accuracy"] = float(hit.float().mean())

    # CE loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    result["loss"] = float(criterion(logits_cat, targets_cat)) / max(total, 1)

    # maneuver subset
    man_mask = motion_cat > maneuver_threshold
    n_man = int(man_mask.sum())
    result["maneuver_total_tokens"] = n_man
    if n_man > 0:
        for k in [1, 3, 5]:
            if k >= vocab_size:
                result[f"maneuver_top{k}_accuracy"] = 1.0
                continue
            _, topk = logits_cat[man_mask].topk(k, dim=-1)
            hit = (topk == targets_cat[man_mask].unsqueeze(-1)).any(dim=-1)
            result[f"maneuver_top{k}_accuracy"] = float(hit.float().mean())
        result["maneuver_loss"] = float(
            criterion(logits_cat[man_mask], targets_cat[man_mask])) / n_man
    else:
        for k in [1, 3, 5]:
            result[f"maneuver_top{k}_accuracy"] = 0.0
        result["maneuver_loss"] = 0.0

    # per-token recall (top-10)
    preds_np = preds.numpy()
    targets_np = targets_cat.numpy()
    counts = np.bincount(targets_np, minlength=vocab_size)
    top_ids = np.argsort(-counts)[:10]
    recall_rows = []
    for tid in top_ids:
        mask = targets_np == tid
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        n_correct = int((preds_np[mask] == tid).sum())
        recall_rows.append({
            "dense_id": int(tid),
            "gt_count": n_total,
            "recall": round(n_correct / n_total, 4),
        })
    result["per_token_recall"] = recall_rows

    return result


# ── decode-back ──────────────────────────────────────────────────

def decode_back_eval(
    model: OneStepTokenBC,
    loader: DataLoader,
    device: torch.device,
    vq_ckpt_path: str,
    source_data_path: str,
    dense_to_raw: dict,
    maneuver_threshold: float,
) -> dict:
    """Decode predicted tokens back to continuous actions and compute RMSE."""
    from training.vq.vqvae_model import ActionChunkVQVAE

    # load VQ-VAE
    vq_ckpt = torch.load(vq_ckpt_path, map_location=device, weights_only=False)
    vq_args = vq_ckpt["args"]
    vqvae = ActionChunkVQVAE(
        token_steps=vq_args["token_steps"],
        codebook_size=vq_args["codebook_size"],
        latent_dim=vq_args.get("latent_dim", 32),
    )
    vqvae.load_state_dict(vq_ckpt["model_state_dict"])
    vqvae.to(device).eval()

    normalize = vq_args.get("normalize_action", False)
    action_stats = vq_ckpt.get("action_stats", {})
    act_mean = np.array(
        [action_stats["mean"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None
    act_std = np.array(
        [action_stats["std"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None

    token_steps = vq_args["token_steps"]

    # load source actions for ground truth
    source = np.load(source_data_path, allow_pickle=True)
    gt_actions_all = source["action"].astype(np.float32)  # [N, 20, 3]

    # collect predictions
    model.eval()
    all_preds = []
    all_targets = []
    all_motion = []

    # we also need window_idx and token_pos from the dataset
    # but loader doesn't return them, so we access dataset directly
    dataset = loader.dataset
    window_indices = dataset.target_token  # dummy, we need from raw
    raw_data = np.load(dataset.meta["source_data"].replace("\\", "/"),
                       allow_pickle=True) if False else None

    # simpler approach: collect from loader and use dataset's stored arrays
    with torch.no_grad():
        for obs_hist, prev_tok, target, motion in loader:
            logits = model(obs_hist.to(device), prev_tok.to(device))
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds.numpy())
            all_targets.append(target.numpy())
            all_motion.append(motion.numpy())

    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    motion_np = np.concatenate(all_motion)

    # get window_idx and token_pos for these samples
    indices = dataset.indices
    raw_npz = np.load(dataset.meta.get("_data_path", ""), allow_pickle=True) \
        if "_data_path" in dataset.meta else None

    # Access from the dataset's underlying arrays
    win_idx = np.load(loader.dataset.obs_hist.__array_interface__["data"][0]
                      ) if False else None

    # Actually, let's access window_idx from the stored NPZ directly
    # We stored them in the onestep NPZ
    data_path_attr = getattr(dataset, '_data_path', None)

    # Simplest approach: re-load the NPZ to get window_idx and token_pos
    # The dataset object has these as attributes
    onestep_raw = np.load(
        loader.dataset.obs_hist.__class__.__name__  # this won't work
    ) if False else None

    # Let's just decode the predicted tokens directly and compare with
    # ground-truth tokens decoded. Both are chunk-level comparisons.

    # Map dense → raw
    pred_raw = np.array([dense_to_raw[int(x)] for x in preds_np], dtype=np.int64)
    gt_raw = np.array([dense_to_raw[int(x)] for x in targets_np], dtype=np.int64)

    # Decode via VQ-VAE
    def decode_batch(raw_ids):
        raw_t = torch.from_numpy(raw_ids).to(device)
        with torch.no_grad():
            z_q = vqvae.vq.embedding(raw_t)  # [M, latent]
            x_hat = vqvae.decoder(z_q)  # [M, token_steps*3]
            x_hat = x_hat.reshape(-1, token_steps, 3).cpu().numpy()
        if normalize and act_mean is not None:
            x_hat = x_hat * act_std[None, None, :] + act_mean[None, None, :]
        return x_hat

    pred_actions = decode_batch(pred_raw)  # [M, 4, 3]
    gt_actions = decode_batch(gt_raw)      # [M, 4, 3]

    # per-dim RMSE/MAE
    dim_names = ["dpsi_rad", "dalt_sp_m", "dspd_sp_mps"]
    diff = pred_actions - gt_actions
    result = {}
    for d, name in enumerate(dim_names):
        per_dim = diff[:, :, d]
        result[f"{name}_rmse"] = float(np.sqrt((per_dim ** 2).mean()))
        result[f"{name}_mae"] = float(np.abs(per_dim).mean())

    result["overall_rmse"] = float(np.sqrt((diff ** 2).mean()))

    # maneuver subset
    man_mask = motion_np > maneuver_threshold
    if man_mask.any():
        man_diff = diff[man_mask]
        man_result = {}
        for d, name in enumerate(dim_names):
            per_dim = man_diff[:, :, d]
            man_result[f"{name}_rmse"] = float(np.sqrt((per_dim ** 2).mean()))
            man_result[f"{name}_mae"] = float(np.abs(per_dim).mean())
        man_result["overall_rmse"] = float(np.sqrt((man_diff ** 2).mean()))
        result["maneuver"] = man_result

    return result


# ── simplified decode-back (no source data dependency) ───────────

def decode_back_eval_simple(
    model: OneStepTokenBC,
    loader: DataLoader,
    device: torch.device,
    vq_ckpt_path: str,
    dense_to_raw: dict,
    maneuver_threshold: float,
) -> dict:
    """Decode predicted tokens to actions, compare with GT token decodes.

    This compares decoded(predicted_token) vs decoded(ground_truth_token),
    measuring how much error the token prediction introduces on top of
    the quantisation error.
    """
    from training.vq.vqvae_model import ActionChunkVQVAE

    vq_ckpt = torch.load(vq_ckpt_path, map_location=device, weights_only=False)
    vq_args = vq_ckpt["args"]
    vqvae = ActionChunkVQVAE(
        token_steps=vq_args["token_steps"],
        codebook_size=vq_args["codebook_size"],
        latent_dim=vq_args.get("latent_dim", 32),
    )
    vqvae.load_state_dict(vq_ckpt["model_state_dict"])
    vqvae.to(device).eval()

    normalize = vq_args.get("normalize_action", False)
    action_stats = vq_ckpt.get("action_stats", {})
    act_mean = np.array(
        [action_stats["mean"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None
    act_std = np.array(
        [action_stats["std"][k] for k in action_stats["fields"]],
        dtype=np.float32) if normalize else None
    token_steps = vq_args["token_steps"]

    model.eval()
    all_preds = []
    all_targets = []
    all_motion = []

    with torch.no_grad():
        for obs_hist, prev_tok, target, motion in loader:
            logits = model(obs_hist.to(device), prev_tok.to(device))
            preds = logits.argmax(dim=-1).cpu()
            all_preds.append(preds.numpy())
            all_targets.append(target.numpy())
            all_motion.append(motion.numpy())

    preds_np = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    motion_np = np.concatenate(all_motion)

    pred_raw = np.array([dense_to_raw[int(x)] for x in preds_np], dtype=np.int64)
    gt_raw = np.array([dense_to_raw[int(x)] for x in targets_np], dtype=np.int64)

    def decode_batch(raw_ids, batch_size=8192):
        all_chunks = []
        for i in range(0, len(raw_ids), batch_size):
            batch = torch.from_numpy(raw_ids[i:i + batch_size]).to(device)
            with torch.no_grad():
                z_q = vqvae.vq.embedding(batch)
                x_hat = vqvae.decoder(z_q).reshape(-1, token_steps, 3).cpu().numpy()
            if normalize and act_mean is not None:
                x_hat = x_hat * act_std[None, None, :] + act_mean[None, None, :]
            all_chunks.append(x_hat)
        return np.concatenate(all_chunks)

    pred_actions = decode_batch(pred_raw)  # [M, 4, 3]
    gt_actions = decode_batch(gt_raw)      # [M, 4, 3]

    dim_names = ["dpsi_rad", "dalt_sp_m", "dspd_sp_mps"]
    diff = pred_actions - gt_actions
    result = {}
    for d, name in enumerate(dim_names):
        per_dim = diff[:, :, d]
        result[f"{name}_rmse"] = float(np.sqrt((per_dim ** 2).mean()))
        result[f"{name}_mae"] = float(np.abs(per_dim).mean())
    result["overall_rmse"] = float(np.sqrt((diff ** 2).mean()))

    man_mask = motion_np > maneuver_threshold
    if man_mask.any():
        man_diff = diff[man_mask]
        man_result = {}
        for d, name in enumerate(dim_names):
            per_dim = man_diff[:, :, d]
            man_result[f"{name}_rmse"] = float(np.sqrt((per_dim ** 2).mean()))
            man_result[f"{name}_mae"] = float(np.abs(per_dim).mean())
        man_result["overall_rmse"] = float(np.sqrt((man_diff ** 2).mean()))
        result["maneuver"] = man_result

    return result


# ── training ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-step receding-horizon token policy baseline")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints/onestep_token_bc_t4_cb64")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--class_weight_mode", type=str, default="inverse_sqrt",
                   choices=["none", "inverse_sqrt", "effective_num"])
    p.add_argument("--maneuver_quantile", type=float, default=0.7)
    p.add_argument("--vq_ckpt", type=str, default=None,
                   help="VQ-VAE checkpoint for decode-back eval (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────
    ds_train = OneStepDataset(args.data, "train")
    ds_val = OneStepDataset(args.data, "val")
    ds_test = OneStepDataset(args.data, "test")

    vocab_size = ds_train.vocab_size
    obs_hist_len = ds_train.obs_hist_len
    obs_dim = ds_train.obs_hist.shape[-1]

    print(f"Dataset: {args.data}")
    print(f"  vocab_size={vocab_size}  obs_hist_len={obs_hist_len}  obs_dim={obs_dim}")
    print(f"  train={len(ds_train):,}  val={len(ds_val):,}  test={len(ds_test):,}")

    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ── maneuver threshold ────────────────────────────────────
    train_motion = ds_train.motion[ds_train.indices]
    maneuver_threshold = float(np.quantile(train_motion, args.maneuver_quantile))
    print(f"\nManeuver threshold: {maneuver_threshold:.6f} "
          f"(quantile={args.maneuver_quantile})")

    # ── class weights ─────────────────────────────────────────
    class_weights = compute_class_weights(ds_train, args.class_weight_mode)
    print(f"Class weight mode: {args.class_weight_mode}")
    print(f"  range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")

    cw_path = save_dir / "class_weights.json"
    with open(cw_path, "w") as f:
        json.dump({"mode": args.class_weight_mode, "weights": class_weights.tolist()}, f)

    # ── majority baseline ─────────────────────────────────────
    maj_train = compute_majority_baseline(ds_train, maneuver_threshold)
    maj_val = compute_majority_baseline(ds_val, maneuver_threshold)
    maj_test = compute_majority_baseline(ds_test, maneuver_threshold)

    print(f"\nMajority baseline:")
    print(f"  train: id={maj_train['majority_token_id']}  "
          f"all={maj_train['majority_accuracy']:.4f}  "
          f"man={maj_train.get('maneuver_majority_accuracy', 0):.4f}")
    print(f"  test:  id={maj_test['majority_token_id']}  "
          f"all={maj_test['majority_accuracy']:.4f}  "
          f"man={maj_test.get('maneuver_majority_accuracy', 0):.4f}")

    # ── model ─────────────────────────────────────────────────
    model = OneStepTokenBC(
        vocab_size=vocab_size,
        obs_dim=obs_dim,
        obs_hist_len=obs_hist_len,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: OneStepTokenBC")
    print(f"  hidden_dim={args.hidden_dim}  layers={args.num_layers}  "
          f"obs_hist_len={obs_hist_len}")
    print(f"  parameters={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── training loop ─────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    t_start = time.time()

    history_mode = ds_train.meta.get("history_mode", "raw")

    def _save(path, epoch, val_acc):
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "vocab_size": vocab_size,
            "obs_hist_len": obs_hist_len,
            "history_mode": history_mode,
            "obs_mean": ds_train.obs_mean,
            "obs_std": ds_train.obs_std,
            "epoch": epoch,
            "val_acc": val_acc,
            "maneuver_threshold": maneuver_threshold,
        }, str(path))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for obs_hist, prev_tok, target, motion in loader_train:
            obs_hist = obs_hist.to(device)
            prev_tok = prev_tok.to(device)
            target = target.to(device)

            logits = model(obs_hist, prev_tok)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == target).sum().item()
            epoch_total += target.numel()
            epoch_loss += loss.item() * target.numel()

        scheduler.step()
        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)

        val_metrics = evaluate(model, loader_val, device, vocab_size,
                               maneuver_threshold)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_top1": round(val_metrics["top1_accuracy"], 4),
            "val_top3": round(val_metrics["top3_accuracy"], 4),
            "val_man_top1": round(val_metrics.get("maneuver_top1_accuracy", 0), 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(record)

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"tr_loss={train_loss:.4f}  tr_acc={train_acc:.4f}  "
                  f"val_top1={val_metrics['top1_accuracy']:.4f}  "
                  f"val_top3={val_metrics['top3_accuracy']:.4f}  "
                  f"val_man={val_metrics.get('maneuver_top1_accuracy', 0):.4f}  "
                  f"[{elapsed:.0f}s]")

        _save(save_dir / "last.pt", epoch, val_metrics["top1_accuracy"])

        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["top1_accuracy"]
            patience_counter = 0
            _save(save_dir / "best.pt", epoch, best_val_acc)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    elapsed_total = time.time() - t_start
    print(f"\nTraining complete in {elapsed_total:.0f}s")
    print(f"Best val top-1: {best_val_acc:.4f}")

    # ── test evaluation ───────────────────────────────────────
    ckpt = torch.load(str(save_dir / "best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, loader_test, device, vocab_size,
                            maneuver_threshold)

    lift_all = test_metrics["top1_accuracy"] - maj_test["majority_accuracy"]
    lift_man = (test_metrics.get("maneuver_top1_accuracy", 0)
                - maj_test.get("maneuver_majority_accuracy", 0))

    print(f"\n{'='*60}")
    print(f"  Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  ALL-TOKEN:")
    print(f"    top-1={test_metrics['top1_accuracy']:.4f}  "
          f"top-3={test_metrics['top3_accuracy']:.4f}  "
          f"top-5={test_metrics['top5_accuracy']:.4f}")
    print(f"    majority baseline = {maj_test['majority_accuracy']:.4f}")
    print(f"    lift = {lift_all:+.4f}")
    print(f"  MANEUVER (top {(1-args.maneuver_quantile)*100:.0f}%):")
    print(f"    top-1={test_metrics.get('maneuver_top1_accuracy', 0):.4f}  "
          f"top-3={test_metrics.get('maneuver_top3_accuracy', 0):.4f}  "
          f"top-5={test_metrics.get('maneuver_top5_accuracy', 0):.4f}")
    print(f"    majority baseline = {maj_test.get('maneuver_majority_accuracy', 0):.4f}")
    print(f"    lift = {lift_man:+.4f}")

    # per-token recall
    print(f"\n  Per-Token Recall (top-10):")
    for r in test_metrics.get("per_token_recall", []):
        print(f"    T{r['dense_id']:2d}: recall={r['recall']:.2%}  "
              f"(n={r['gt_count']:,})")
    print(f"{'='*60}")

    # ── save eval JSONs ───────────────────────────────────────
    eval_all = {
        "cross_entropy": test_metrics["loss"],
        "top1_accuracy": test_metrics["top1_accuracy"],
        "top3_accuracy": test_metrics["top3_accuracy"],
        "top5_accuracy": test_metrics["top5_accuracy"],
        "total_tokens": test_metrics["total_tokens"],
        "majority_baseline": maj_test["majority_accuracy"],
        "lift_over_majority": lift_all,
    }
    with open(save_dir / "eval_all.json", "w") as f:
        json.dump(eval_all, f, indent=2)

    eval_man = {
        "top1_accuracy": test_metrics.get("maneuver_top1_accuracy", 0),
        "top3_accuracy": test_metrics.get("maneuver_top3_accuracy", 0),
        "top5_accuracy": test_metrics.get("maneuver_top5_accuracy", 0),
        "total_maneuver_tokens": test_metrics["maneuver_total_tokens"],
        "maneuver_quantile": args.maneuver_quantile,
        "maneuver_threshold": maneuver_threshold,
        "majority_baseline": maj_test.get("maneuver_majority_accuracy", 0),
        "lift_over_majority": lift_man,
    }
    with open(save_dir / "eval_maneuver.json", "w") as f:
        json.dump(eval_man, f, indent=2)

    # per-token recall CSV
    with open(save_dir / "per_token_recall.csv", "w") as f:
        f.write("dense_id,gt_count,recall\n")
        for r in test_metrics.get("per_token_recall", []):
            f.write(f"{r['dense_id']},{r['gt_count']},{r['recall']}\n")

    # ── decode-back (optional) ────────────────────────────────
    decode_metrics = {}
    if args.vq_ckpt:
        # load dense_to_raw mapping from source tokenized dataset
        src_meta_str = ds_train.meta.get("source_data", "")
        if src_meta_str:
            try:
                tok_npz = np.load(src_meta_str, allow_pickle=True)
                tok_meta_raw = tok_npz["meta"]
                if isinstance(tok_meta_raw, np.ndarray):
                    tms = tok_meta_raw.item() if tok_meta_raw.ndim == 0 \
                        else bytes(tok_meta_raw).decode()
                else:
                    tms = str(tok_meta_raw)
                tok_meta = json.loads(tms)
                d2r = {int(k): int(v) for k, v in
                       tok_meta["dense_to_raw_mapping"].items()}
            except Exception as e:
                print(f"WARNING: Could not load dense_to_raw mapping: {e}")
                d2r = None
        else:
            d2r = None

        if d2r:
            print("\nRunning decode-back evaluation...")
            decode_metrics = decode_back_eval_simple(
                model, loader_test, device, args.vq_ckpt,
                d2r, maneuver_threshold)

            print(f"  dpsi RMSE={decode_metrics['dpsi_rad_rmse']:.4f} rad  "
                  f"dalt RMSE={decode_metrics['dalt_sp_m_rmse']:.1f} m  "
                  f"dspd RMSE={decode_metrics['dspd_sp_mps_rmse']:.2f} m/s")
            if "maneuver" in decode_metrics:
                m = decode_metrics["maneuver"]
                print(f"  maneuver: dpsi={m['dpsi_rad_rmse']:.4f}  "
                      f"dalt={m['dalt_sp_m_rmse']:.1f}  "
                      f"dspd={m['dspd_sp_mps_rmse']:.2f}")

            with open(save_dir / "decode_metrics.json", "w") as f:
                json.dump(decode_metrics, f, indent=2)

    # ── metrics.json ──────────────────────────────────────────
    metrics = {
        "dataset": args.data,
        "model_args": vars(args),
        "vocab_size": vocab_size,
        "obs_hist_len": obs_hist_len,
        "class_weight_mode": args.class_weight_mode,
        "maneuver_quantile": args.maneuver_quantile,
        "maneuver_threshold": maneuver_threshold,
        "majority_baseline": {"train": maj_train, "val": maj_val, "test": maj_test},
        "eval_all": eval_all,
        "eval_maneuver": eval_man,
        "decode_metrics": decode_metrics,
        "training": {
            "best_epoch": ckpt["epoch"],
            "best_val_top1": best_val_acc,
            "total_time_s": round(elapsed_total, 1),
            "n_params": n_params,
        },
        "history": history,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved:")
    for p in sorted(save_dir.glob("*")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
