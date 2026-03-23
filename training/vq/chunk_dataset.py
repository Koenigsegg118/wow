"""Extract fixed-length action chunks from the training NPZ for VQ-VAE.

Supports three chunk extraction modes to control sample duplication,
and optional motion-aware downsampling to reduce static-chunk dominance.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ── chunk extraction helpers ─────────────────────────────────────────

def extract_chunks_single_fixed(
    actions: np.ndarray, token_steps: int, fixed_offset: int = 0,
) -> np.ndarray:
    """One chunk per source window at a fixed time offset.

    Returns [N, token_steps, 3].
    """
    return actions[:, fixed_offset : fixed_offset + token_steps, :].copy()


def extract_chunks_single_random(
    actions: np.ndarray, token_steps: int, rng: np.random.RandomState,
) -> np.ndarray:
    """One chunk per source window at a random time offset.

    Returns [N, token_steps, 3].
    """
    N, T, C = actions.shape
    max_start = T - token_steps
    offsets = rng.randint(0, max_start + 1, size=N)
    chunks = np.empty((N, token_steps, C), dtype=actions.dtype)
    for i, off in enumerate(offsets):
        chunks[i] = actions[i, off : off + token_steps, :]
    return chunks


def extract_chunks_strided(
    actions: np.ndarray, token_steps: int, chunk_stride: int,
) -> np.ndarray:
    """Multiple chunks per source window with given stride.

    Returns [M, token_steps, 3] where M = N * num_positions.
    """
    N, T, C = actions.shape
    starts = list(range(0, T - token_steps + 1, chunk_stride))
    out = np.empty((N * len(starts), token_steps, C), dtype=actions.dtype)
    idx = 0
    for s in starts:
        out[idx : idx + N] = actions[:, s : s + token_steps, :]
        idx += N
    return out


# ── motion scoring & bucketing ───────────────────────────────────────

def compute_motion_score(
    chunks: np.ndarray,
    lambda_alt: float = 1.0,
    lambda_spd: float = 1.0,
) -> np.ndarray:
    """Per-chunk motion intensity score.

    chunks : [M, token_steps, 3]  — channels are [dpsi, dalt, dspd].

    score = mean(|dpsi|) + lambda_alt * mean(|dalt|/100) + lambda_spd * mean(|dspd|/10)

    Returns [M] float32.
    """
    abs_vals = np.abs(chunks)                             # [M, T, 3]
    dpsi = abs_vals[:, :, 0].mean(axis=1)                 # [M]
    dalt = abs_vals[:, :, 1].mean(axis=1) / 100.0         # [M]
    dspd = abs_vals[:, :, 2].mean(axis=1) / 10.0          # [M]
    return (dpsi + lambda_alt * dalt + lambda_spd * dspd).astype(np.float32)


def bucketize_motion(
    scores: np.ndarray,
    static_q: float = 0.50,
    light_q: float = 0.85,
) -> tuple[np.ndarray, dict]:
    """Assign each chunk to static / light / maneuvering bucket.

    Uses quantile thresholds on *scores*.

    Returns
    -------
    labels : [M] int array  (0=static, 1=light, 2=maneuvering)
    info   : dict with threshold values and counts
    """
    t_static = float(np.percentile(scores, static_q * 100))
    t_light = float(np.percentile(scores, light_q * 100))
    labels = np.full(len(scores), 2, dtype=np.int32)    # maneuvering
    labels[scores <= t_static] = 0                       # static
    labels[(scores > t_static) & (scores <= t_light)] = 1  # light
    counts = {
        "static": int((labels == 0).sum()),
        "light": int((labels == 1).sum()),
        "maneuvering": int((labels == 2).sum()),
    }
    info = {
        "static_threshold": round(t_static, 6),
        "light_threshold": round(t_light, 6),
        "static_quantile": static_q,
        "light_quantile": light_q,
        **counts,
    }
    return labels, info


def downsample_by_motion_bucket(
    chunks: np.ndarray,
    labels: np.ndarray,
    static_keep: float = 0.25,
    light_keep: float = 0.5,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, dict]:
    """Down-sample static and light chunks, keep all maneuvering.

    Returns (kept_chunks [K, T, 3], stats dict).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    keep_mask = np.ones(len(chunks), dtype=bool)
    ratios = {0: static_keep, 1: light_keep, 2: 1.0}
    kept_counts = {}

    for bucket, ratio in ratios.items():
        idxs = np.where(labels == bucket)[0]
        if ratio < 1.0 and len(idxs) > 0:
            n_drop = int(len(idxs) * (1.0 - ratio))
            drop = rng.choice(idxs, size=n_drop, replace=False)
            keep_mask[drop] = False
        name = {0: "static", 1: "light", 2: "maneuvering"}[bucket]
        kept_counts[f"{name}_kept"] = int(keep_mask[labels == bucket].sum())

    return chunks[keep_mask], kept_counts


# ── main dataset ─────────────────────────────────────────────────────

class ActionChunkDataset(Dataset):
    """Action chunk dataset with configurable extraction and sampling.

    Parameters
    ----------
    npz_path : str
        NPZ with ``action`` array [N, T, 3].
    token_steps : int
        Time-steps per chunk (default 2).
    normalize : bool
        Per-channel z-score normalisation.
    chunk_extract_mode : str
        ``"single_fixed"`` | ``"single_random"`` | ``"strided_all"``.
    fixed_offset : int
        For ``single_fixed`` mode.
    chunk_stride : int
        For ``strided_all`` mode.
    lambda_alt, lambda_spd : float
        Motion score weights.
    static_keep_ratio, light_keep_ratio : float
        Fraction of static/light chunks to keep (1.0 = no downsampling).
    static_q, light_q : float
        Quantile boundaries for motion buckets.
    max_chunks : int or None
        Hard cap after all sampling.
    seed : int
        Random seed for reproducibility.
    """

    ACT_FIELDS = ("dpsi_rad", "alt_sp_m", "spd_sp_mps")

    def __init__(
        self,
        npz_path: str,
        token_steps: int = 2,
        normalize: bool = False,
        *,
        chunk_extract_mode: str = "single_fixed",
        fixed_offset: int = 0,
        chunk_stride: int | None = None,
        lambda_alt: float = 1.0,
        lambda_spd: float = 1.0,
        static_keep_ratio: float = 1.0,
        light_keep_ratio: float = 1.0,
        static_q: float = 0.50,
        light_q: float = 0.85,
        max_chunks: int | None = None,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)

        raw = np.load(npz_path)
        actions: np.ndarray = raw["action"]  # [N, T, 3]
        assert actions.ndim == 3 and actions.shape[2] == 3

        N, T, C = actions.shape
        assert token_steps <= T, (
            f"token_steps={token_steps} exceeds sequence length T={T}"
        )

        self._num_source_windows = N

        # ── step 1: extract chunks ───────────────────────────────
        if chunk_extract_mode == "single_fixed":
            assert fixed_offset + token_steps <= T, (
                f"fixed_offset={fixed_offset} + token_steps={token_steps} > T={T}"
            )
            chunks = extract_chunks_single_fixed(actions, token_steps, fixed_offset)
        elif chunk_extract_mode == "single_random":
            chunks = extract_chunks_single_random(actions, token_steps, rng)
        elif chunk_extract_mode == "strided_all":
            stride = chunk_stride if chunk_stride is not None else token_steps
            assert stride >= 1, f"chunk_stride must be >= 1, got {stride}"
            chunks = extract_chunks_strided(actions, token_steps, stride)
        else:
            raise ValueError(
                f"Unknown chunk_extract_mode: {chunk_extract_mode!r}. "
                f"Choose from: single_fixed, single_random, strided_all"
            )

        self._num_chunks_after_extract = len(chunks)

        # ── step 2: motion-aware downsampling ────────────────────
        do_motion_sampling = (static_keep_ratio < 1.0 or light_keep_ratio < 1.0)
        if do_motion_sampling:
            scores = compute_motion_score(chunks, lambda_alt, lambda_spd)
            labels, bucket_info = bucketize_motion(scores, static_q, light_q)
            chunks, kept_counts = downsample_by_motion_bucket(
                chunks, labels, static_keep_ratio, light_keep_ratio, rng)
            self._motion_info = {**bucket_info, **kept_counts}
        else:
            self._motion_info = None

        self._num_chunks_after_motion = len(chunks)

        # ── step 3: max_chunks cap ───────────────────────────────
        if max_chunks is not None and len(chunks) > max_chunks:
            idxs = rng.choice(len(chunks), size=max_chunks, replace=False)
            idxs.sort()
            chunks = chunks[idxs]

        self._num_chunks_final = len(chunks)

        # ── per-channel statistics (on final set, before norm) ───
        self.act_mean = torch.from_numpy(
            chunks.mean(axis=(0, 1)).astype(np.float32))  # [3]
        self.act_std = torch.from_numpy(
            chunks.std(axis=(0, 1)).astype(np.float32))   # [3]
        self.act_std = torch.clamp(self.act_std, min=1e-8)

        self.normalized = normalize
        if normalize:
            mean_np = self.act_mean.numpy()
            std_np = self.act_std.numpy()
            chunks = (chunks - mean_np[None, None, :]) / std_np[None, None, :]

        self.chunks = torch.from_numpy(chunks)  # [M, token_steps, 3]

    # ── public helpers ───────────────────────────────────────────

    def sampling_stats(self) -> dict:
        """Return a dict summarising the sampling pipeline."""
        stats: dict = {
            "num_source_windows": self._num_source_windows,
            "num_chunks_after_extract": self._num_chunks_after_extract,
            "num_chunks_after_motion_sampling": self._num_chunks_after_motion,
            "num_chunks_final": self._num_chunks_final,
        }
        if self._motion_info is not None:
            stats["motion"] = self._motion_info
        return stats

    def __len__(self) -> int:
        return self.chunks.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.chunks[index]  # [token_steps, 3]

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalised tensor back to original action space."""
        if not self.normalized:
            return x
        std = self.act_std.to(x.device)
        mean = self.act_mean.to(x.device)
        return x * std + mean
