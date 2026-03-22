"""Unit tests for Token BC baseline (next-token prediction)."""

import json
import pathlib

import numpy as np
import pytest
import torch

from training.vq.train_token_bc import (
    block_split,
    TokenBCDataset,
    TokenBCTransformer,
    compute_class_weights,
    compute_majority_baseline,
    evaluate,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_tok_npz(tmp: pathlib.Path, N: int = 500, n_tok: int = 5,
                  vocab_size: int = 30) -> str:
    """Create a minimal fake tokenized NPZ."""
    path = str(tmp / "fake_tok.npz")
    meta = {
        "active_vocab_size": vocab_size,
        "token_steps": 4,
        "tokens_per_window": n_tok,
        "window_length": 20,
    }
    np.savez(
        path,
        obs_tok_start=np.random.randn(N, n_tok, 8).astype(np.float32),
        token_ids=np.random.randint(0, vocab_size, (N, n_tok)).astype(np.int64),
        motion_score=np.random.rand(N, n_tok).astype(np.float32),
        active_raw_codes=np.arange(vocab_size),
        meta=json.dumps(meta),
    )
    return path


# ── block split tests ────────────────────────────────────────────────

class TestBlockSplit:
    def test_no_overlap(self):
        splits = block_split(1000, train_frac=0.8, val_frac=0.1, guard=10)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_guard_band_gap(self):
        splits = block_split(10000, train_frac=0.8, val_frac=0.1, guard=50)
        train_max = splits["train"].max()
        val_min = splits["val"].min()
        test_min = splits["test"].min()
        val_max = splits["val"].max()
        assert val_min - train_max > 1
        assert test_min - val_max > 1

    def test_proportions(self):
        N = 10000
        splits = block_split(N, train_frac=0.8, val_frac=0.1, guard=100)
        assert len(splits["train"]) == 8000
        assert len(splits["val"]) == 1000

    def test_small_dataset_no_crash(self):
        splits = block_split(50, train_frac=0.8, val_frac=0.1, guard=5)
        assert len(splits["train"]) > 0

    def test_indices_in_range(self):
        N = 500
        splits = block_split(N, guard=20)
        for name in ["train", "val", "test"]:
            assert splits[name].min() >= 0
            assert splits[name].max() < N


# ── class weight tests ───────────────────────────────────────────────

class TestClassWeights:
    def test_none_mode_uniform(self, tmp_path):
        npz = _make_tok_npz(tmp_path, N=200)
        ds = TokenBCDataset(npz, split="train", guard=10)
        w = compute_class_weights(ds, ds.vocab_size, "none")
        assert torch.allclose(w, torch.ones(ds.vocab_size))

    def test_inverse_sqrt_higher_for_rare(self, tmp_path):
        """Rare tokens should have higher weight."""
        path = str(tmp_path / "skewed.npz")
        N, n_tok, V = 200, 5, 5
        # token 0 appears 80%, others 5% each
        tokens = np.zeros((N, n_tok), dtype=np.int64)
        tokens[:40] = np.array([1, 2, 3, 4, 0])  # 20% have diverse tokens
        meta = {"active_vocab_size": V, "token_steps": 4,
                "tokens_per_window": n_tok, "window_length": 20}
        np.savez(path,
                 obs_tok_start=np.zeros((N, n_tok, 8), dtype=np.float32),
                 token_ids=tokens,
                 motion_score=np.zeros((N, n_tok), dtype=np.float32),
                 active_raw_codes=np.arange(V),
                 meta=json.dumps(meta))
        ds = TokenBCDataset(path, split="train", guard=5)
        w = compute_class_weights(ds, V, "inverse_sqrt")
        # token 0 is most frequent → lowest weight
        assert w[0] < w[1]

    def test_effective_num_returns_valid(self, tmp_path):
        npz = _make_tok_npz(tmp_path, N=200)
        ds = TokenBCDataset(npz, split="train", guard=10)
        w = compute_class_weights(ds, ds.vocab_size, "effective_num")
        assert (w > 0).all()
        assert w.shape == (ds.vocab_size,)


# ── dataset tests ────────────────────────────────────────────────────

class TestTokenBCDataset:
    def test_shape(self, tmp_path):
        npz = _make_tok_npz(tmp_path, N=200)
        ds = TokenBCDataset(npz, split="train", guard=10)
        obs, tokens, motion = ds[0]
        assert obs.shape == (5, 8)
        assert tokens.shape == (5,)
        assert motion.shape == (5,)

    def test_splits_dont_overlap(self, tmp_path):
        npz = _make_tok_npz(tmp_path, N=500)
        ds_tr = TokenBCDataset(npz, split="train", guard=10)
        ds_va = TokenBCDataset(npz, split="val", guard=10)
        ds_te = TokenBCDataset(npz, split="test", guard=10)
        tr_set = set(ds_tr.indices)
        va_set = set(ds_va.indices)
        te_set = set(ds_te.indices)
        assert len(tr_set & va_set) == 0
        assert len(tr_set & te_set) == 0
        assert len(va_set & te_set) == 0

    def test_obs_normalized(self, tmp_path):
        npz = _make_tok_npz(tmp_path, N=300)
        ds = TokenBCDataset(npz, split="train", guard=10)
        all_obs = torch.stack([ds[i][0] for i in range(min(len(ds), 100))])
        assert all_obs.mean().abs() < 0.3
        assert (all_obs.std() - 1.0).abs() < 0.5


# ── model tests ──────────────────────────────────────────────────────

class TestTokenBCTransformer:
    def test_output_shape(self):
        model = TokenBCTransformer(vocab_size=30, n_positions=5)
        obs = torch.randn(4, 5, 8)
        tokens = torch.randint(0, 30, (4, 5))
        logits = model(obs, tokens)
        assert logits.shape == (4, 5, 30)

    def test_causal_masking(self):
        """Output at position t should not change when future tokens change."""
        model = TokenBCTransformer(vocab_size=30, n_positions=5)
        model.eval()
        obs = torch.randn(1, 5, 8)
        tokens_a = torch.randint(0, 30, (1, 5))
        tokens_b = tokens_a.clone()
        tokens_b[0, 3] = (tokens_a[0, 3] + 1) % 30

        with torch.no_grad():
            out_a = model(obs, tokens_a)
            out_b = model(obs, tokens_b)

        for t in range(3):
            assert torch.allclose(out_a[0, t], out_b[0, t], atol=1e-5), \
                f"Position {t} changed when future token changed"

    def test_training_reduces_loss(self):
        torch.manual_seed(0)
        model = TokenBCTransformer(vocab_size=10, hidden_dim=64, num_layers=2,
                                    n_heads=2, n_positions=5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        obs = torch.randn(32, 5, 8)
        tokens = torch.randint(0, 10, (32, 5))

        model.eval()
        with torch.no_grad():
            logits = model(obs, tokens)
            loss_before = criterion(logits.reshape(-1, 10), tokens.reshape(-1)).item()

        model.train()
        for _ in range(100):
            logits = model(obs, tokens)
            loss = criterion(logits.reshape(-1, 10), tokens.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(obs, tokens)
            loss_after = criterion(logits.reshape(-1, 10), tokens.reshape(-1)).item()

        assert loss_after < loss_before * 0.5


# ── majority baseline tests ─────────────────────────────────────────

class TestMajorityBaseline:
    def test_uniform_tokens(self, tmp_path):
        """All same token → majority acc = 100%."""
        path = str(tmp_path / "uniform.npz")
        N, n_tok, V = 100, 5, 10
        meta = {"active_vocab_size": V, "token_steps": 4,
                "tokens_per_window": n_tok, "window_length": 20}
        np.savez(
            path,
            obs_tok_start=np.zeros((N, n_tok, 8), dtype=np.float32),
            token_ids=np.full((N, n_tok), 3, dtype=np.int64),
            motion_score=np.zeros((N, n_tok), dtype=np.float32),
            active_raw_codes=np.arange(V),
            meta=json.dumps(meta),
        )
        ds = TokenBCDataset(path, split="train", guard=5)
        maj = compute_majority_baseline(ds)
        assert maj["majority_token_id"] == 3
        assert maj["majority_accuracy"] == 1.0

    def test_majority_with_maneuver_subset(self, tmp_path):
        """Maneuver subset should have its own majority baseline."""
        path = str(tmp_path / "maneuver.npz")
        N, n_tok, V = 200, 5, 10
        meta = {"active_vocab_size": V, "token_steps": 4,
                "tokens_per_window": n_tok, "window_length": 20}
        motion = np.random.rand(N, n_tok).astype(np.float32)
        np.savez(
            path,
            obs_tok_start=np.zeros((N, n_tok, 8), dtype=np.float32),
            token_ids=np.random.randint(0, V, (N, n_tok)).astype(np.int64),
            motion_score=motion,
            active_raw_codes=np.arange(V),
            meta=json.dumps(meta),
        )
        ds = TokenBCDataset(path, split="train", guard=5)
        maj = compute_majority_baseline(ds, maneuver_threshold=0.5)
        assert "maneuver_majority_accuracy" in maj
        assert 0.0 < maj["maneuver_majority_accuracy"] <= 1.0

    def test_majority_accuracy_range(self, tmp_path):
        npz = _make_tok_npz(tmp_path, N=500, vocab_size=10)
        ds = TokenBCDataset(npz, split="train", guard=10)
        maj = compute_majority_baseline(ds)
        assert 0.0 < maj["majority_accuracy"] <= 1.0


# ── evaluation tests ─────────────────────────────────────────────────

class TestEvaluation:
    def test_topk_accuracy(self, tmp_path):
        """top-k accuracy: top-1 ≤ top-3 ≤ top-5."""
        npz = _make_tok_npz(tmp_path, N=100, vocab_size=10)
        ds = TokenBCDataset(npz, split="train", guard=5)
        loader = torch.utils.data.DataLoader(ds, batch_size=32)
        model = TokenBCTransformer(vocab_size=10, hidden_dim=32,
                                    num_layers=1, n_heads=2, n_positions=5)
        result = evaluate(model, loader, torch.device("cpu"), 10, 0.5)
        assert result["top1_accuracy"] <= result["top3_accuracy"]
        assert result["top3_accuracy"] <= result["top5_accuracy"]
        assert 0.0 <= result["top1_accuracy"] <= 1.0
