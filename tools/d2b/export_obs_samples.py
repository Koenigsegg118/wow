"""Module A part 1: Sample obs-sidecar data from frame buffers or JSONL into JSONL.

Reads 640-value AFSIM frames (or an existing obs JSONL file), calls
``build_obs_sidecar`` for each ego platform at each sampled frame, and
writes the results as one-JSON-per-line to an output file.

Each output line is::

    {"obs": <sidecar dict>, "meta": {"scenario_id": ..., "timestamp_sec": ...,
     "ego_platform_idx": ..., "frame_idx": ...}}

Usage (CLI):
    python -m tools.d2b.export_obs_samples \\
        --source frames --frames-dir path/to/npz \\
        --output obs_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from sim.runtime.afsim_obs_sidecar import SidecarContext, build_obs_sidecar
from tools.d2b.d2b_config import DEFAULT_RANDOM_SEED, DEFAULT_SAMPLE_SIZE

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────


def _make_meta(
    scenario_id: str | None,
    timestamp_sec: float | None,
    ego_platform_idx: int,
    frame_idx: int,
) -> dict:
    """Build a meta dict for one obs sample."""
    return {
        "scenario_id": scenario_id,
        "timestamp_sec": timestamp_sec,
        "ego_platform_idx": ego_platform_idx,
        "frame_idx": frame_idx,
    }


# ── sampling from in-memory frame buffer ─────────────────────────


def sample_from_frame_buffer(
    frame_buffer: list[np.ndarray],
    ego_indices: list[int],
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
    scenario_id: str | None = None,
) -> list[dict]:
    """Sample obs-sidecar data from a 640-value frame buffer.

    For each ego platform in *ego_indices*, we maintain a separate
    ``SidecarContext`` so that history fields accumulate correctly.
    Frames are processed in chronological order; only *n_samples*
    (ego, frame) pairs are emitted, chosen uniformly at random.

    Args:
        frame_buffer: List of 640-element numpy arrays, one per time step.
        ego_indices: Platform indices to treat as ego (e.g. [0, 1, 2, 3]).
        n_samples: Maximum number of output samples.
        seed: Random seed for reproducible sampling.
        scenario_id: Optional scenario identifier written into meta.

    Returns:
        List of ``{"obs": <sidecar dict>, "meta": <meta dict>}`` dicts.
    """
    if not frame_buffer:
        return []

    n_frames = len(frame_buffer)
    n_egos = len(ego_indices)
    total_candidates = n_frames * n_egos

    # Decide which (ego, frame) pairs to keep
    rng = np.random.default_rng(seed)
    if total_candidates <= n_samples:
        selected = set(range(total_candidates))
    else:
        selected = set(
            rng.choice(total_candidates, size=n_samples, replace=False).tolist()
        )

    # Build one SidecarContext per ego so history propagates correctly
    contexts: dict[int, SidecarContext] = {
        idx: SidecarContext() for idx in ego_indices
    }

    results: list[dict] = []

    for frame_idx, frame in enumerate(frame_buffer):
        frame_arr = np.asarray(frame, dtype=np.float64)
        for ego_pos, ego_idx in enumerate(ego_indices):
            flat_id = frame_idx * n_egos + ego_pos
            ctx = contexts[ego_idx]

            # Always call build_obs_sidecar so that context accumulates,
            # but only collect the output if this pair was selected.
            obs = build_obs_sidecar(frame_arr, ego_platform_idx=ego_idx, ctx=ctx)

            if flat_id in selected:
                meta = _make_meta(
                    scenario_id=scenario_id,
                    timestamp_sec=None,  # raw frames have no sim-time
                    ego_platform_idx=ego_idx,
                    frame_idx=frame_idx,
                )
                results.append({"obs": obs, "meta": meta})

    return results


# ── sampling from existing JSONL ────────────────────────────────


def sample_from_jsonl(
    input_path: Path,
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict]:
    """Sample from an existing obs JSONL file.

    Reads all lines, then uniformly sub-samples up to *n_samples* records.

    Args:
        input_path: Path to a JSONL file where each line is a JSON object
                    with at least an ``obs`` key.
        n_samples: Maximum number of records to return.
        seed: Random seed for reproducible sampling.

    Returns:
        List of parsed JSON dicts (each has ``obs`` and ``meta`` keys).
    """
    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) <= n_samples:
        return records

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=n_samples, replace=False)
    return [records[i] for i in sorted(indices)]


# ── main entry: write JSONL ──────────────────────────────────────


def export_obs_samples(
    source: str,
    output_path: Path,
    n_samples: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
    scenario_id: str | None = None,
    ego_indices: list[int] | None = None,
    frames_dir: Path | None = None,
    input_jsonl: Path | None = None,
) -> Path:
    """Main entry: sample obs and write to JSONL.

    Args:
        source: Either ``"frames"`` (sample from .npz frame files) or
                ``"jsonl"`` (sub-sample from an existing JSONL file).
        output_path: Where to write the output JSONL.
        n_samples: Maximum number of output records.
        seed: Random seed.
        scenario_id: Scenario identifier (used when source="frames").
        ego_indices: Ego platform indices (used when source="frames").
                     Defaults to [0, 1, 2, 3].
        frames_dir: Directory containing .npz files with a ``frames`` key
                    (used when source="frames").
        input_jsonl: Path to existing JSONL (used when source="jsonl").

    Returns:
        The *output_path* that was written.
    """
    if ego_indices is None:
        ego_indices = [0, 1, 2, 3]

    if source == "jsonl":
        if input_jsonl is None:
            raise ValueError("source='jsonl' requires input_jsonl path")
        samples = sample_from_jsonl(input_jsonl, n_samples=n_samples, seed=seed)

    elif source == "frames":
        if frames_dir is None:
            raise ValueError("source='frames' requires frames_dir path")
        frame_buffer = _load_frames_from_dir(frames_dir)
        samples = sample_from_frame_buffer(
            frame_buffer,
            ego_indices=ego_indices,
            n_samples=n_samples,
            seed=seed,
            scenario_id=scenario_id,
        )
    else:
        raise ValueError(f"Unknown source type: {source!r}. Use 'frames' or 'jsonl'.")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in samples:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Wrote %d obs samples → %s", len(samples), output_path)
    return output_path


def _load_frames_from_dir(frames_dir: Path) -> list[np.ndarray]:
    """Load 640-value frames from .npz files in a directory.

    Each .npz file is expected to have a ``frames`` key containing a 2-D
    array of shape ``(n_steps, 640)``.  All files are concatenated in
    sorted filename order.
    """
    frames_dir = Path(frames_dir)
    all_frames: list[np.ndarray] = []

    npz_files = sorted(frames_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {frames_dir}")

    for npz_path in npz_files:
        data = np.load(npz_path)
        if "frames" not in data:
            logger.warning("Skipping %s: no 'frames' key", npz_path.name)
            continue
        arr = data["frames"]
        for i in range(arr.shape[0]):
            all_frames.append(arr[i])

    logger.info(
        "Loaded %d frames from %d files in %s",
        len(all_frames), len(npz_files), frames_dir,
    )
    return all_frames


# ── CLI ──────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample obs-sidecar data into JSONL."
    )
    parser.add_argument(
        "--source",
        choices=["frames", "jsonl"],
        required=True,
        help="Data source type: 'frames' for .npz frame files, "
             "'jsonl' for existing obs JSONL.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory with .npz frame files (required if source=frames).",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Input JSONL path (required if source=jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Max number of samples to emit (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Scenario identifier written into meta.",
    )
    parser.add_argument(
        "--ego-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Ego platform indices (default: 0 1 2 3).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    export_obs_samples(
        source=args.source,
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        scenario_id=args.scenario_id,
        ego_indices=args.ego_indices,
        frames_dir=args.frames_dir,
        input_jsonl=args.input_jsonl,
    )
