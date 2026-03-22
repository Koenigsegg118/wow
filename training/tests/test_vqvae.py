"""Unit tests for the VQ-VAE action tokeniser pipeline."""

import tempfile
import pathlib

import numpy as np
import pytest
import torch

from training.vq.chunk_dataset import (
    ActionChunkDataset,
    extract_chunks_single_fixed,
    extract_chunks_single_random,
    extract_chunks_strided,
    compute_motion_score,
    bucketize_motion,
    downsample_by_motion_bucket,
)
from training.vq.vqvae_model import ActionChunkVQVAE, VectorQuantizer


# ── helpers ──────────────────────────────────────────────────────────

def _make_npz(tmp: pathlib.Path, N: int = 100, T: int = 20) -> str:
    """Create a minimal fake NPZ with action array."""
    path = str(tmp / "fake.npz")
    action = np.random.randn(N, T, 3).astype(np.float32)
    np.savez(path, action=action, obs=np.zeros((N, T, 8), dtype=np.float32))
    return path


# ── chunk extraction tests ───────────────────────────────────────────

class TestChunkExtraction:
    def test_single_fixed_shape(self):
        actions = np.random.randn(50, 20, 3).astype(np.float32)
        chunks = extract_chunks_single_fixed(actions, token_steps=2, fixed_offset=0)
        assert chunks.shape == (50, 2, 3)

    def test_single_fixed_values(self):
        actions = np.arange(60, dtype=np.float32).reshape(1, 20, 3)
        chunks = extract_chunks_single_fixed(actions, token_steps=3, fixed_offset=5)
        expected = actions[0, 5:8, :]
        np.testing.assert_array_equal(chunks[0], expected)

    def test_single_random_shape(self):
        actions = np.random.randn(50, 20, 3).astype(np.float32)
        rng = np.random.RandomState(42)
        chunks = extract_chunks_single_random(actions, token_steps=2, rng=rng)
        assert chunks.shape == (50, 2, 3)

    def test_single_random_reproducible(self):
        actions = np.random.randn(50, 20, 3).astype(np.float32)
        c1 = extract_chunks_single_random(actions, 2, np.random.RandomState(42))
        c2 = extract_chunks_single_random(actions, 2, np.random.RandomState(42))
        np.testing.assert_array_equal(c1, c2)

    def test_strided_shape(self):
        actions = np.random.randn(10, 20, 3).astype(np.float32)
        chunks = extract_chunks_strided(actions, token_steps=2, chunk_stride=4)
        # starts: 0, 4, 8, 12, 16 → 5 positions
        assert chunks.shape == (10 * 5, 2, 3)

    def test_strided_stride1_matches_old(self):
        """stride=1 should produce same count as old sliding window."""
        actions = np.random.randn(10, 20, 3).astype(np.float32)
        chunks = extract_chunks_strided(actions, token_steps=2, chunk_stride=1)
        assert chunks.shape == (10 * 19, 2, 3)


# ── motion scoring tests ────────────────────────────────────────────

class TestMotionScoring:
    def test_zero_action_has_zero_score(self):
        chunks = np.zeros((5, 2, 3), dtype=np.float32)
        scores = compute_motion_score(chunks)
        np.testing.assert_array_equal(scores, 0.0)

    def test_higher_motion_higher_score(self):
        static = np.zeros((1, 2, 3), dtype=np.float32)
        active = np.ones((1, 2, 3), dtype=np.float32) * 10
        s_static = compute_motion_score(static)[0]
        s_active = compute_motion_score(active)[0]
        assert s_active > s_static

    def test_bucketize_returns_three_labels(self):
        scores = np.linspace(0, 1, 100).astype(np.float32)
        labels, info = bucketize_motion(scores, static_q=0.5, light_q=0.85)
        assert set(np.unique(labels)).issubset({0, 1, 2})
        assert info["static"] + info["light"] + info["maneuvering"] == 100

    def test_downsample_reduces_count(self):
        chunks = np.random.randn(1000, 2, 3).astype(np.float32)
        labels = np.zeros(1000, dtype=np.int32)  # all static
        kept, stats = downsample_by_motion_bucket(
            chunks, labels, static_keep=0.25, light_keep=0.5)
        assert len(kept) == 250  # 25% of 1000


# ── ActionChunkDataset integration tests ─────────────────────────────

class TestActionChunkDataset:
    def test_shape_single_fixed(self, tmp_path):
        npz = _make_npz(tmp_path, N=50, T=20)
        ds = ActionChunkDataset(npz, token_steps=2,
                                chunk_extract_mode="single_fixed")
        # single_fixed: 50 windows → 50 chunks, then motion sampling
        # default static_keep=1.0 when not passed explicitly
        assert ds[0].shape == (2, 3)

    def test_shape_strided_all(self, tmp_path):
        npz = _make_npz(tmp_path, N=10, T=20)
        ds = ActionChunkDataset(npz, token_steps=2,
                                chunk_extract_mode="strided_all",
                                chunk_stride=2,
                                static_keep_ratio=1.0,
                                light_keep_ratio=1.0)
        # stride=2: starts at 0,2,4,...,18 → 10 positions × 10 windows
        assert len(ds) == 10 * 10

    def test_backward_compat_strided_stride1(self, tmp_path):
        """strided_all with stride=1 gives same count as old behavior."""
        npz = _make_npz(tmp_path, N=50, T=20)
        ds = ActionChunkDataset(npz, token_steps=2,
                                chunk_extract_mode="strided_all",
                                chunk_stride=1,
                                static_keep_ratio=1.0,
                                light_keep_ratio=1.0)
        assert len(ds) == 50 * 19

    def test_token_steps_exceeds_T_raises(self, tmp_path):
        npz = _make_npz(tmp_path, N=5, T=20)
        with pytest.raises(AssertionError, match="token_steps=21 exceeds"):
            ActionChunkDataset(npz, token_steps=21)

    def test_values_match_source(self, tmp_path):
        """First chunk of single_fixed offset=0 should match action[0, 0:token_steps]."""
        npz_path = tmp_path / "check.npz"
        action = np.arange(60, dtype=np.float32).reshape(1, 20, 3)
        np.savez(str(npz_path), action=action, obs=np.zeros((1, 20, 8)))
        ds = ActionChunkDataset(str(npz_path), token_steps=3, normalize=False,
                                chunk_extract_mode="single_fixed", fixed_offset=0,
                                static_keep_ratio=1.0, light_keep_ratio=1.0)
        expected = torch.from_numpy(action[0, 0:3, :])
        assert torch.allclose(ds[0], expected)

    def test_motion_sampling_reduces_count(self, tmp_path):
        npz = _make_npz(tmp_path, N=200, T=20)
        ds_full = ActionChunkDataset(npz, token_steps=2,
                                     chunk_extract_mode="single_fixed",
                                     static_keep_ratio=1.0,
                                     light_keep_ratio=1.0)
        ds_sampled = ActionChunkDataset(npz, token_steps=2,
                                        chunk_extract_mode="single_fixed",
                                        static_keep_ratio=0.25,
                                        light_keep_ratio=0.5)
        assert len(ds_sampled) < len(ds_full)

    def test_sampling_stats_available(self, tmp_path):
        npz = _make_npz(tmp_path, N=50, T=20)
        ds = ActionChunkDataset(npz, token_steps=2,
                                chunk_extract_mode="single_fixed",
                                static_keep_ratio=0.5,
                                light_keep_ratio=0.8)
        stats = ds.sampling_stats()
        assert "num_source_windows" in stats
        assert "num_chunks_after_extract" in stats
        assert "motion" in stats

    def test_max_chunks_cap(self, tmp_path):
        npz = _make_npz(tmp_path, N=100, T=20)
        ds = ActionChunkDataset(npz, token_steps=2,
                                chunk_extract_mode="single_fixed",
                                static_keep_ratio=1.0,
                                light_keep_ratio=1.0,
                                max_chunks=30)
        assert len(ds) == 30


# ── normalisation tests ──────────────────────────────────────────────

class TestNormalisation:
    def test_normalized_zero_mean_unit_var(self, tmp_path):
        """After normalisation, chunks should be ~zero-mean, ~unit-var."""
        npz = _make_npz(tmp_path, N=200, T=20)
        ds = ActionChunkDataset(npz, token_steps=2, normalize=True,
                                chunk_extract_mode="single_fixed",
                                static_keep_ratio=1.0, light_keep_ratio=1.0)
        all_chunks = ds.chunks  # [M, 2, 3]
        mean = all_chunks.mean(dim=(0, 1))
        std = all_chunks.std(dim=(0, 1))
        assert torch.allclose(mean, torch.zeros(3), atol=0.05)
        assert torch.allclose(std, torch.ones(3), atol=0.05)

    def test_unnormalized_preserves_values(self, tmp_path):
        """Without normalisation, values match raw data."""
        npz = _make_npz(tmp_path, N=10, T=20)
        ds_raw = ActionChunkDataset(npz, token_steps=2, normalize=False,
                                    chunk_extract_mode="single_fixed",
                                    static_keep_ratio=1.0, light_keep_ratio=1.0)
        ds_norm = ActionChunkDataset(npz, token_steps=2, normalize=True,
                                     chunk_extract_mode="single_fixed",
                                     static_keep_ratio=1.0, light_keep_ratio=1.0)
        assert len(ds_raw) == len(ds_norm)
        assert not torch.allclose(ds_raw[0], ds_norm[0])

    def test_denormalize_roundtrip(self, tmp_path):
        """denormalize(normalized_chunk) ≈ original raw chunk."""
        npz = _make_npz(tmp_path, N=50, T=20)
        ds_raw = ActionChunkDataset(npz, token_steps=2, normalize=False,
                                    chunk_extract_mode="single_fixed",
                                    static_keep_ratio=1.0, light_keep_ratio=1.0)
        ds_norm = ActionChunkDataset(npz, token_steps=2, normalize=True,
                                     chunk_extract_mode="single_fixed",
                                     static_keep_ratio=1.0, light_keep_ratio=1.0)
        recovered = ds_norm.denormalize(ds_norm.chunks)
        assert torch.allclose(recovered, ds_raw.chunks, atol=1e-2)

    def test_denormalize_noop_when_not_normalized(self, tmp_path):
        npz = _make_npz(tmp_path, N=10, T=20)
        ds = ActionChunkDataset(npz, token_steps=2, normalize=False,
                                chunk_extract_mode="single_fixed",
                                static_keep_ratio=1.0, light_keep_ratio=1.0)
        x = ds[0]
        assert torch.equal(ds.denormalize(x), x)

    def test_stats_always_available(self, tmp_path):
        """act_mean / act_std are set even when normalize=False."""
        npz = _make_npz(tmp_path, N=10, T=20)
        ds = ActionChunkDataset(npz, token_steps=2, normalize=False,
                                chunk_extract_mode="single_fixed",
                                static_keep_ratio=1.0, light_keep_ratio=1.0)
        assert ds.act_mean.shape == (3,)
        assert ds.act_std.shape == (3,)
        assert (ds.act_std > 0).all()


# ── VQ-VAE model tests ──────────────────────────────────────────────

class TestVectorQuantizer:
    def test_output_shapes(self):
        vq = VectorQuantizer(codebook_size=16, latent_dim=8)
        z = torch.randn(4, 8)
        z_q, loss, indices = vq(z)
        assert z_q.shape == (4, 8)
        assert loss.shape == ()
        assert indices.shape == (4,)
        assert indices.max() < 16

    def test_straight_through_grad(self):
        vq = VectorQuantizer(codebook_size=16, latent_dim=8)
        z = torch.randn(4, 8, requires_grad=True)
        z_q, loss, _ = vq(z)
        (z_q.sum() + loss).backward()
        assert z.grad is not None


class TestActionChunkVQVAE:
    def test_forward_shape(self):
        model = ActionChunkVQVAE(token_steps=2, codebook_size=32, latent_dim=16)
        x = torch.randn(8, 2, 3)
        x_hat, vq_loss, indices = model(x)
        assert x_hat.shape == (8, 2, 3)
        assert vq_loss.shape == ()
        assert indices.shape == (8,)

    def test_different_token_steps(self):
        for ts in [1, 2, 4, 8]:
            model = ActionChunkVQVAE(token_steps=ts, codebook_size=64)
            x = torch.randn(4, ts, 3)
            x_hat, _, _ = model(x)
            assert x_hat.shape == (4, ts, 3)

    def test_reconstruction_improves(self):
        """A few gradient steps should reduce reconstruction loss."""
        torch.manual_seed(0)
        model = ActionChunkVQVAE(token_steps=2, codebook_size=64, latent_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(64, 2, 3)

        model.eval()
        with torch.no_grad():
            x0, vq0, _ = model(x)
            loss_before = torch.nn.functional.mse_loss(x0, x).item() + vq0.item()

        model.train()
        for _ in range(50):
            x_hat, vq_loss, _ = model(x)
            loss = torch.nn.functional.mse_loss(x_hat, x) + vq_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xf, vqf, _ = model(x)
            loss_after = torch.nn.functional.mse_loss(xf, x).item() + vqf.item()

        assert loss_after < loss_before * 0.5, (
            f"Loss did not decrease enough: {loss_before:.4f} → {loss_after:.4f}"
        )
