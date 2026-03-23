#!/usr/bin/env python
"""Token BC baseline: next-token prediction on VQ-tokenized action sequences.

Given a tokenized dataset (from tokenize_npz.py), trains a causal Transformer
to predict the next action token conditioned on observation context.

Key design choices:
- Block split with guard bands to prevent window-leakage
- Weighted cross-entropy to handle class imbalance (static-token dominance)
- Majority-token baseline reported alongside model accuracy
- Dual evaluation: all-token + maneuver-subset (by motion_score quantile)

Usage (from repo root):
    python -m training.vq.train_token_bc \
        --data datasets/dt2hz_H2s_vqclean_t4_cb64_tok.npz \
        --save_dir checkpoints/token_bc_t4_cb64 \
        --epochs 50 --batch 256 --lr 3e-4 \
        --hidden_dim 128 --num_layers 2 \
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


# ── dataset with block split ─────────────────────────────────────

def block_split(
    N: int,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    guard: int = 100,
) -> dict[str, np.ndarray]:
    """Sequential block split with guard bands.

    Windows from the same trajectory are contiguous.  stride=5 with T=20
    means adjacent windows overlap by 15 steps.  A guard band of ``guard``
    windows (default 100, = 500 steps = 250s at 2 Hz) eliminates any
    cross-split leakage.

    Returns dict with 'train', 'val', 'test' index arrays.
    """
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    # test gets the remainder after guards
    t0, t1 = 0, n_train
    v0, v1 = n_train + guard, n_train + guard + n_val
    s0, s1 = n_train + guard + n_val + guard, N

    # clamp to valid range
    v0 = min(v0, N)
    v1 = min(v1, N)
    s0 = min(s0, N)

    return {
        "train": np.arange(t0, t1),
        "val": np.arange(v0, v1),
        "test": np.arange(s0, s1),
        "guard_size": guard,
        "split_sizes": {
            "train": t1 - t0,
            "val": v1 - v0,
            "test": s1 - s0,
            "guard_total": 2 * guard,
        },
    }


class TokenBCDataset(Dataset):
    """Next-token prediction dataset.

    Each sample is one window: (obs_seq, token_seq, motion_scores).
    The model sees obs[0:t] + tokens[0:t-1] and predicts tokens[t]
    for t in [0, n_tok).
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        guard: int = 100,
    ):
        raw = np.load(data_path, allow_pickle=True)
        self.obs = raw["obs_tok_start"].astype(np.float32)  # [N, 5, 8]
        self.tokens = raw["token_ids"].astype(np.int64)      # [N, 5]
        self.motion = raw["motion_score"].astype(np.float32)  # [N, 5]

        # metadata
        meta_raw = raw["meta"]
        if isinstance(meta_raw, np.ndarray):
            ms = meta_raw.item() if meta_raw.ndim == 0 else bytes(meta_raw).decode()
        else:
            ms = str(meta_raw)
        self.meta = json.loads(ms)
        self.vocab_size = self.meta["active_vocab_size"]
        self.n_tok = self.tokens.shape[1]

        # compute obs stats from FULL dataset (before split) for normalisation
        self._obs_mean = self.obs.mean(axis=(0, 1))  # [8]
        self._obs_std = self.obs.std(axis=(0, 1))     # [8]
        self._obs_std = np.where(self._obs_std < 1e-6, 1.0, self._obs_std)

        # block split
        N = len(self.obs)
        splits = block_split(N, train_frac, val_frac, guard)
        self.indices = splits[split]
        self.split_info = splits["split_sizes"]

    @property
    def obs_mean(self) -> np.ndarray:
        return self._obs_mean

    @property
    def obs_std(self) -> np.ndarray:
        return self._obs_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        obs = (self.obs[j] - self._obs_mean) / self._obs_std
        return (
            torch.from_numpy(obs),               # [n_tok, 8]
            torch.from_numpy(self.tokens[j]),     # [n_tok]
            torch.from_numpy(self.motion[j]),     # [n_tok]
        )


# ── class weights ────────────────────────────────────────────────

def compute_class_weights(
    dataset: TokenBCDataset,
    vocab_size: int,
    mode: str = "inverse_sqrt",
) -> torch.Tensor:
    """Compute per-class weights from training token frequencies.

    Modes:
      none          — uniform weights (1.0 for all classes)
      inverse_sqrt  — w_c = 1 / sqrt(count_c);  then normalised so mean=1
      effective_num — w_c = (1 - beta) / (1 - beta^n_c), beta=0.999
    """
    all_tokens = dataset.tokens[dataset.indices].flatten()
    counts = np.bincount(all_tokens, minlength=vocab_size).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div by zero for unseen codes

    if mode == "none":
        weights = np.ones(vocab_size, dtype=np.float64)
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
        weights *= vocab_size / weights.sum()  # normalise so mean ≈ 1
    elif mode == "effective_num":
        beta = 0.999
        effective = (1.0 - beta) / (1.0 - np.power(beta, counts))
        weights = effective
        weights *= vocab_size / weights.sum()
    else:
        raise ValueError(f"Unknown class_weight_mode: {mode}")

    return torch.from_numpy(weights).float()


# ── model ────────────────────────────────────────────────────────

class TokenBCTransformer(nn.Module):
    """Causal Transformer for next-token prediction.

    At each position t, the input is: obs_embed(obs[t]) + tok_embed(tok[t-1]).
    For t=0, a learnable <start> embedding replaces tok_embed.
    Output: logits over vocab_size at each position.
    """

    def __init__(
        self,
        vocab_size: int,
        obs_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_positions: int = 5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_positions = n_positions

        # embeddings
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.tok_embed = nn.Embedding(vocab_size, hidden_dim)
        self.start_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.pos_embed = nn.Embedding(n_positions, hidden_dim)

        # transformer
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # output head
        self.head = nn.Linear(hidden_dim, vocab_size)

        # causal mask (upper triangular = masked)
        mask = torch.triu(torch.ones(n_positions, n_positions), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, obs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    [B, T, obs_dim]
            tokens: [B, T] int64 — ground-truth tokens (teacher forcing)

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T, _ = obs.shape

        # obs embedding at each position
        obs_emb = self.obs_proj(obs)  # [B, T, hidden_dim]

        # shifted token embedding: tok[t-1] at position t, <start> at t=0
        tok_emb_all = self.tok_embed(tokens)  # [B, T, hidden_dim]
        tok_shifted = torch.cat([
            self.start_embed.unsqueeze(0).unsqueeze(0).expand(B, 1, -1),
            tok_emb_all[:, :-1, :],
        ], dim=1)  # [B, T, hidden_dim]

        # position embedding
        pos_ids = torch.arange(T, device=obs.device)
        pos_emb = self.pos_embed(pos_ids).unsqueeze(0)  # [1, T, hidden_dim]

        x = obs_emb + tok_shifted + pos_emb  # [B, T, hidden_dim]

        # causal transformer
        x = self.transformer(x, mask=self.causal_mask[:T, :T])

        return self.head(x)  # [B, T, vocab_size]


# ── evaluation ───────────────────────────────────────────────────

def compute_majority_baseline(
    dataset: TokenBCDataset,
    maneuver_threshold: float | None = None,
) -> dict:
    """Compute majority-token baseline accuracy (all-token and maneuver-subset)."""
    all_tokens = dataset.tokens[dataset.indices]  # [M, 5]
    all_motion = dataset.motion[dataset.indices]   # [M, 5]
    flat_tok = all_tokens.flatten()
    flat_mot = all_motion.flatten()

    counts = np.bincount(flat_tok, minlength=dataset.vocab_size)
    majority_id = int(counts.argmax())
    majority_count = int(counts[majority_id])
    total = len(flat_tok)

    result = {
        "majority_token_id": majority_id,
        "majority_count": majority_count,
        "total_tokens": total,
        "majority_accuracy": majority_count / total,
    }

    # maneuver subset majority baseline
    if maneuver_threshold is not None:
        man_mask = flat_mot > maneuver_threshold
        man_total = int(man_mask.sum())
        if man_total > 0:
            man_tok = flat_tok[man_mask]
            man_counts = np.bincount(man_tok, minlength=dataset.vocab_size)
            man_majority_id = int(man_counts.argmax())
            man_majority_count = int(man_counts[man_majority_id])
            result["maneuver_majority_token_id"] = man_majority_id
            result["maneuver_majority_count"] = man_majority_count
            result["maneuver_total_tokens"] = man_total
            result["maneuver_majority_accuracy"] = man_majority_count / man_total
        else:
            result["maneuver_majority_accuracy"] = 0.0
            result["maneuver_total_tokens"] = 0

    return result


def _topk_accuracy(logits_flat: torch.Tensor, targets_flat: torch.Tensor,
                   ks: list[int]) -> dict[int, float]:
    """Compute top-k accuracy for multiple k values."""
    result = {}
    for k in ks:
        if k >= logits_flat.shape[-1]:
            result[k] = 1.0
            continue
        _, topk_preds = logits_flat.topk(k, dim=-1)
        match = (topk_preds == targets_flat.unsqueeze(-1)).any(dim=-1)
        result[k] = match.float().mean().item()
    return result


@torch.no_grad()
def evaluate(
    model: TokenBCTransformer,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
    maneuver_threshold: float,
) -> dict:
    """Evaluate model on a dataloader.

    Returns cross-entropy, top-1/3/5 accuracy for both all-token and maneuver subset.
    """
    model.eval()
    n_tok = model.n_positions

    # collect all predictions
    all_logits = []
    all_targets = []
    all_motion = []

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0

    for obs, tokens, motion in loader:
        obs, tokens = obs.to(device), tokens.to(device)
        logits = model(obs, tokens)  # [B, T, V]
        loss = criterion(logits.reshape(-1, vocab_size), tokens.reshape(-1))
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_targets.append(tokens.cpu())
        all_motion.append(motion)

    logits_cat = torch.cat(all_logits, dim=0)    # [N, T, V]
    targets_cat = torch.cat(all_targets, dim=0)  # [N, T]
    motion_cat = torch.cat(all_motion, dim=0)    # [N, T]

    N_total = targets_cat.numel()

    # flatten for top-k
    logits_flat = logits_cat.reshape(-1, vocab_size)
    targets_flat = targets_cat.reshape(-1)
    motion_flat = motion_cat.reshape(-1)

    # all-token metrics
    topk_all = _topk_accuracy(logits_flat, targets_flat, [1, 3, 5])

    # per-position accuracy
    preds = logits_cat.argmax(dim=-1)
    match = (preds == targets_cat)
    acc_per_pos = [match[:, t].float().mean().item() for t in range(n_tok)]

    # maneuver subset
    man_mask = motion_flat > maneuver_threshold
    n_maneuver = int(man_mask.sum())
    if n_maneuver > 0:
        topk_man = _topk_accuracy(logits_flat[man_mask], targets_flat[man_mask], [1, 3, 5])
        # maneuver CE
        man_loss = criterion(logits_flat[man_mask], targets_flat[man_mask]).item()
    else:
        topk_man = {1: 0.0, 3: 0.0, 5: 0.0}
        man_loss = 0.0

    return {
        "loss": total_loss / max(N_total, 1),
        "top1_accuracy": topk_all[1],
        "top3_accuracy": topk_all[3],
        "top5_accuracy": topk_all[5],
        "accuracy_per_position": acc_per_pos,
        "total_tokens": N_total,
        "maneuver_loss": man_loss / max(n_maneuver, 1),
        "maneuver_top1_accuracy": topk_man[1],
        "maneuver_top3_accuracy": topk_man[3],
        "maneuver_top5_accuracy": topk_man[5],
        "maneuver_total_tokens": n_maneuver,
    }


# ── training ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Token BC baseline: next-token prediction on VQ-tokenized actions")
    # data
    p.add_argument("--data", type=str, required=True,
                   help="Tokenized NPZ (from tokenize_npz.py)")
    p.add_argument("--save_dir", type=str, default="checkpoints/token_bc_t4_cb64")
    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs)")
    # model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # class imbalance
    p.add_argument("--class_weight_mode", type=str, default="inverse_sqrt",
                   choices=["none", "inverse_sqrt", "effective_num"],
                   help="Class weight mode for cross-entropy loss")
    # split
    p.add_argument("--train_frac", type=float, default=0.80)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--guard", type=int, default=100,
                   help="Guard band size (windows) between splits")
    # evaluation
    p.add_argument("--maneuver_quantile", type=float, default=0.7,
                   help="Quantile threshold for maneuver subset (e.g. 0.7 = top 30%%)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────
    ds_train = TokenBCDataset(args.data, "train", args.train_frac, args.val_frac, args.guard)
    ds_val = TokenBCDataset(args.data, "val", args.train_frac, args.val_frac, args.guard)
    ds_test = TokenBCDataset(args.data, "test", args.train_frac, args.val_frac, args.guard)

    vocab_size = ds_train.vocab_size
    n_tok = ds_train.n_tok

    print(f"Dataset: {args.data}")
    print(f"  vocab_size={vocab_size}  tokens_per_window={n_tok}")
    print(f"  train={len(ds_train):,}  val={len(ds_val):,}  test={len(ds_test):,}")
    print(f"  guard_band={args.guard} windows")
    print(f"  split_mode=block_split (no provenance available)")
    print(f"  split_info={ds_train.split_info}")

    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=True)
    loader_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ── maneuver threshold from quantile ──────────────────────
    all_motion_train = ds_train.motion[ds_train.indices].flatten()
    maneuver_threshold = float(np.quantile(all_motion_train, args.maneuver_quantile))
    print(f"\nManeuver threshold: motion_score > {maneuver_threshold:.6f} "
          f"(quantile={args.maneuver_quantile})")

    # ── class weights ─────────────────────────────────────────
    class_weights = compute_class_weights(ds_train, vocab_size, args.class_weight_mode)
    print(f"\nClass weight mode: {args.class_weight_mode}")
    print(f"  weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
    print(f"  weight mean:  {class_weights.mean():.4f}")

    # save class weights
    cw_dict = {
        "mode": args.class_weight_mode,
        "vocab_size": vocab_size,
        "weights": class_weights.tolist(),
    }
    cw_path = save_dir / "class_weights.json"
    with open(cw_path, "w") as f:
        json.dump(cw_dict, f, indent=2)
    print(f"  saved: {cw_path}")

    # ── majority baseline ─────────────────────────────────────
    maj_train = compute_majority_baseline(ds_train, maneuver_threshold)
    maj_val = compute_majority_baseline(ds_val, maneuver_threshold)
    maj_test = compute_majority_baseline(ds_test, maneuver_threshold)

    print(f"\nMajority-token baseline:")
    print(f"  train: token={maj_train['majority_token_id']}  "
          f"all_acc={maj_train['majority_accuracy']:.4f}  "
          f"man_acc={maj_train.get('maneuver_majority_accuracy', 0):.4f}")
    print(f"  val:   token={maj_val['majority_token_id']}  "
          f"all_acc={maj_val['majority_accuracy']:.4f}  "
          f"man_acc={maj_val.get('maneuver_majority_accuracy', 0):.4f}")
    print(f"  test:  token={maj_test['majority_token_id']}  "
          f"all_acc={maj_test['majority_accuracy']:.4f}  "
          f"man_acc={maj_test.get('maneuver_majority_accuracy', 0):.4f}")

    # ── model ─────────────────────────────────────────────────
    model = TokenBCTransformer(
        vocab_size=vocab_size,
        obs_dim=ds_train.obs.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        n_positions=n_tok,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: TokenBCTransformer")
    print(f"  hidden_dim={args.hidden_dim}  layers={args.num_layers}  "
          f"heads={args.n_heads}")
    print(f"  parameters={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── training loop ─────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    t_start = time.time()

    def _save_ckpt(path: pathlib.Path, epoch: int, val_acc: float):
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "vocab_size": vocab_size,
            "n_tok": n_tok,
            "obs_mean": ds_train.obs_mean,
            "obs_std": ds_train.obs_std,
            "epoch": epoch,
            "val_acc": val_acc,
            "class_weight_mode": args.class_weight_mode,
            "maneuver_threshold": maneuver_threshold,
        }, str(path))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for obs, tokens, motion in loader_train:
            obs, tokens = obs.to(device), tokens.to(device)
            logits = model(obs, tokens)  # [B, T, V]

            loss = criterion(logits.reshape(-1, vocab_size), tokens.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == tokens).sum().item()
            epoch_total += tokens.numel()
            epoch_loss += loss.item() * tokens.numel()

        scheduler.step()
        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)

        # validate
        val_metrics = evaluate(model, loader_val, device, vocab_size,
                               maneuver_threshold)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_top1": round(val_metrics["top1_accuracy"], 4),
            "val_top3": round(val_metrics["top3_accuracy"], 4),
            "val_man_top1": round(val_metrics["maneuver_top1_accuracy"], 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(record)

        # progress
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"tr_loss={train_loss:.4f}  tr_acc={train_acc:.4f}  "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"val_top1={val_metrics['top1_accuracy']:.4f}  "
                  f"val_top3={val_metrics['top3_accuracy']:.4f}  "
                  f"val_man={val_metrics['maneuver_top1_accuracy']:.4f}  "
                  f"[{elapsed:.0f}s]")

        # save last.pt every epoch
        _save_ckpt(save_dir / "last.pt", epoch, val_metrics["top1_accuracy"])

        # best checkpoint
        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["top1_accuracy"]
            patience_counter = 0
            _save_ckpt(save_dir / "best.pt", epoch, best_val_acc)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    elapsed_total = time.time() - t_start
    print(f"\nTraining complete in {elapsed_total:.0f}s")
    print(f"Best val top-1 accuracy: {best_val_acc:.4f}")

    # ── final evaluation on test set ──────────────────────────
    ckpt = torch.load(str(save_dir / "best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, loader_test, device, vocab_size,
                            maneuver_threshold)

    # compute lifts
    lift_all = test_metrics["top1_accuracy"] - maj_test["majority_accuracy"]
    lift_man = (test_metrics["maneuver_top1_accuracy"]
                - maj_test.get("maneuver_majority_accuracy", 0.0))

    print(f"\n{'='*60}")
    print(f"  Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  ALL-TOKEN:")
    print(f"    Model  top-1={test_metrics['top1_accuracy']:.4f}  "
          f"top-3={test_metrics['top3_accuracy']:.4f}  "
          f"top-5={test_metrics['top5_accuracy']:.4f}")
    print(f"    Majority baseline    = {maj_test['majority_accuracy']:.4f}")
    print(f"    Lift over majority   = {lift_all:+.4f}")
    print(f"  MANEUVER SUBSET (top {(1-args.maneuver_quantile)*100:.0f}%):")
    print(f"    Model  top-1={test_metrics['maneuver_top1_accuracy']:.4f}  "
          f"top-3={test_metrics['maneuver_top3_accuracy']:.4f}  "
          f"top-5={test_metrics['maneuver_top5_accuracy']:.4f}")
    print(f"    Majority baseline    = "
          f"{maj_test.get('maneuver_majority_accuracy', 0):.4f}")
    print(f"    Lift over majority   = {lift_man:+.4f}")
    print(f"  PER-POSITION top-1    : "
          f"{['%.4f' % x for x in test_metrics['accuracy_per_position']]}")
    print(f"  CE loss               : {test_metrics['loss']:.4f}")
    print(f"{'='*60}")

    # ── save eval_all.json ────────────────────────────────────
    eval_all = {
        "cross_entropy": test_metrics["loss"],
        "top1_accuracy": test_metrics["top1_accuracy"],
        "top3_accuracy": test_metrics["top3_accuracy"],
        "top5_accuracy": test_metrics["top5_accuracy"],
        "accuracy_per_position": test_metrics["accuracy_per_position"],
        "total_tokens": test_metrics["total_tokens"],
        "majority_baseline": {
            "majority_token_id": maj_test["majority_token_id"],
            "majority_accuracy": maj_test["majority_accuracy"],
        },
        "lift_over_majority": lift_all,
    }
    with open(save_dir / "eval_all.json", "w") as f:
        json.dump(eval_all, f, indent=2)

    # ── save eval_maneuver.json ───────────────────────────────
    eval_maneuver = {
        "cross_entropy": test_metrics["maneuver_loss"],
        "top1_accuracy": test_metrics["maneuver_top1_accuracy"],
        "top3_accuracy": test_metrics["maneuver_top3_accuracy"],
        "top5_accuracy": test_metrics["maneuver_top5_accuracy"],
        "total_maneuver_tokens": test_metrics["maneuver_total_tokens"],
        "maneuver_quantile": args.maneuver_quantile,
        "maneuver_threshold": maneuver_threshold,
        "majority_baseline": {
            "majority_token_id": maj_test.get("maneuver_majority_token_id", -1),
            "majority_accuracy": maj_test.get("maneuver_majority_accuracy", 0.0),
        },
        "lift_over_majority": lift_man,
    }
    with open(save_dir / "eval_maneuver.json", "w") as f:
        json.dump(eval_maneuver, f, indent=2)

    # ── save metrics.json ─────────────────────────────────────
    metrics = {
        "dataset": args.data,
        "model_args": vars(args),
        "split_mode": "block_split",
        "provenance_available": False,
        "split_info": ds_train.split_info,
        "guard_band": args.guard,
        "active_vocab_size": vocab_size,
        "class_weight_mode": args.class_weight_mode,
        "maneuver_quantile": args.maneuver_quantile,
        "maneuver_threshold": maneuver_threshold,
        "majority_baseline": {
            "train": maj_train,
            "val": maj_val,
            "test": maj_test,
        },
        "eval_all": eval_all,
        "eval_maneuver": eval_maneuver,
        "training": {
            "best_epoch": ckpt["epoch"],
            "best_val_top1": best_val_acc,
            "total_time_s": round(elapsed_total, 1),
            "n_params": n_params,
            "train_loss_final": history[-1]["train_loss"],
            "val_loss_final": history[-1]["val_loss"],
        },
        "history": history,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved:")
    print(f"  {save_dir / 'best.pt'}")
    print(f"  {save_dir / 'last.pt'}")
    print(f"  {save_dir / 'metrics.json'}")
    print(f"  {save_dir / 'eval_all.json'}")
    print(f"  {save_dir / 'eval_maneuver.json'}")
    print(f"  {save_dir / 'class_weights.json'}")


if __name__ == "__main__":
    main()
