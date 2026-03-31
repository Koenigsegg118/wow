"""Generate synthetic 2v2 air combat frames and produce obs-sidecar JSONL.

Creates 5 tactical scenarios (head_on, tail_chase, extend, multi_target_crossing,
standoff), each with ~20 frames at dt=0.5s.  Each frame is a 640-value float32
array following the AFSIM wire format (80 platforms x 8 fields).

For every frame the script calls ``build_obs_sidecar()`` twice (once per red ego
platform), yielding 200 samples total.

Usage:
    python -m tools.d2b.generate_synthetic_2v2
    python tools/d2b/generate_synthetic_2v2.py --output datasets/d2b/custom.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the wow directory is on sys.path so ``from sim.runtime...`` works
# even when invoked as a standalone script.
# ---------------------------------------------------------------------------
_WOW_ROOT = Path(__file__).resolve().parents[2]  # wow/
if str(_WOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_WOW_ROOT))

from sim.runtime.afsim_obs_sidecar import SidecarContext, build_obs_sidecar

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PLATFORMS = 80
FIELDS_PER_PLATFORM = 8
FRAME_SIZE = N_PLATFORMS * FIELDS_PER_PLATFORM  # 640
METERS_PER_DEG_LAT = 111_320.0

# Ego platform indices (red side)
EGO_INDICES = [0, 1]


# ---------------------------------------------------------------------------
# Platform initial-condition helper
# ---------------------------------------------------------------------------

@dataclass
class PlatformIC:
    """Initial conditions for a single platform."""
    idx: int
    lat: float          # degrees
    lon: float          # degrees
    alt_m: float        # metres MSL
    velN: float         # m/s  (north)
    velE: float         # m/s  (east)
    velD: float = 0.0   # m/s  (down, positive = descending)
    alive: float = 1.0


def _heading_from_vel(velN: float, velE: float) -> float:
    """Compute heading in degrees (0=N, CW+) from velocity components."""
    hdg = math.degrees(math.atan2(velE, velN))
    if hdg < 0:
        hdg += 360.0
    return hdg


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------

def _build_frame(platforms: list[PlatformIC]) -> np.ndarray:
    """Pack platform states into a 640-element float32 array."""
    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    for p in platforms:
        base = p.idx * FIELDS_PER_PLATFORM
        frame[base + 0] = p.alive
        frame[base + 1] = p.lat
        frame[base + 2] = p.lon
        frame[base + 3] = p.alt_m
        frame[base + 4] = p.velN
        frame[base + 5] = p.velE
        frame[base + 6] = p.velD
        frame[base + 7] = _heading_from_vel(p.velN, p.velE)
    return frame


def _propagate(platforms: list[PlatformIC], dt: float) -> list[PlatformIC]:
    """Advance each platform one step using linear kinematics.

    Updates lat/lon based on velocity; altitude updated via velD.
    Returns a *new* list of PlatformIC (does not mutate inputs).
    """
    out: list[PlatformIC] = []
    for p in platforms:
        lat_rad = math.radians(p.lat)
        new_lat = p.lat + (p.velN * dt) / METERS_PER_DEG_LAT
        cos_lat = math.cos(lat_rad)
        new_lon = p.lon + (p.velE * dt) / (METERS_PER_DEG_LAT * cos_lat) if cos_lat > 1e-9 else p.lon
        new_alt = p.alt_m - p.velD * dt  # velD positive = descending
        out.append(PlatformIC(
            idx=p.idx,
            lat=new_lat,
            lon=new_lon,
            alt_m=new_alt,
            velN=p.velN,
            velE=p.velE,
            velD=p.velD,
            alive=p.alive,
        ))
    return out


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def _scenario_head_on() -> list[PlatformIC]:
    return [
        PlatformIC(idx=0, lat=35.0,  lon=120.0,   alt_m=8000, velN=300,  velE=0),
        PlatformIC(idx=1, lat=35.0,  lon=120.001, alt_m=7800, velN=280,  velE=0),
        PlatformIC(idx=2, lat=35.09, lon=120.0,   alt_m=8500, velN=-320, velE=0),
        PlatformIC(idx=3, lat=35.09, lon=120.001, alt_m=8200, velN=-300, velE=20),
    ]


def _scenario_tail_chase() -> list[PlatformIC]:
    return [
        PlatformIC(idx=0, lat=34.95, lon=120.0,   alt_m=7500, velN=350, velE=0),
        PlatformIC(idx=1, lat=34.95, lon=120.001, alt_m=7400, velN=340, velE=0),
        PlatformIC(idx=2, lat=35.0,  lon=120.0,   alt_m=7000, velN=250, velE=0),
        PlatformIC(idx=3, lat=35.0,  lon=120.001, alt_m=7200, velN=240, velE=0),
    ]


def _scenario_extend() -> list[PlatformIC]:
    return [
        PlatformIC(idx=0, lat=35.0, lon=120.0,   alt_m=8000, velN=-200, velE=-150),
        PlatformIC(idx=1, lat=35.0, lon=120.001, alt_m=7800, velN=-180, velE=-140),
        PlatformIC(idx=2, lat=35.0, lon=120.002, alt_m=8200, velN=100,  velE=50),
        PlatformIC(idx=3, lat=35.0, lon=120.003, alt_m=8000, velN=120,  velE=60),
    ]


def _scenario_multi_target_crossing() -> list[PlatformIC]:
    return [
        PlatformIC(idx=0, lat=35.0,  lon=119.99, alt_m=8000, velN=200, velE=200),
        PlatformIC(idx=1, lat=35.0,  lon=120.01, alt_m=8000, velN=200, velE=-200),
        PlatformIC(idx=2, lat=35.05, lon=120.02, alt_m=8200, velN=-200, velE=-200),
        PlatformIC(idx=3, lat=35.05, lon=119.98, alt_m=8200, velN=-200, velE=200),
    ]


def _scenario_standoff() -> list[PlatformIC]:
    return [
        PlatformIC(idx=0, lat=35.0, lon=120.0,   alt_m=8000, velN=250, velE=0),
        PlatformIC(idx=1, lat=35.0, lon=120.001, alt_m=7800, velN=250, velE=0),
        PlatformIC(idx=2, lat=35.0, lon=120.05,  alt_m=8000, velN=250, velE=0),
        PlatformIC(idx=3, lat=35.0, lon=120.051, alt_m=7800, velN=250, velE=0),
    ]


SCENARIOS: list[tuple[str, callable]] = [
    ("head_on", _scenario_head_on),
    ("tail_chase", _scenario_tail_chase),
    ("extend", _scenario_extend),
    ("multi_target_crossing", _scenario_multi_target_crossing),
    ("standoff", _scenario_standoff),
]


# ---------------------------------------------------------------------------
# Generate frames for one scenario
# ---------------------------------------------------------------------------

def generate_scenario_frames(
    init_fn: callable,
    n_frames: int,
    dt: float,
) -> list[np.ndarray]:
    """Return *n_frames* 640-element arrays for a scenario."""
    platforms = init_fn()
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        frames.append(_build_frame(platforms))
        platforms = _propagate(platforms, dt)
    return frames


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_all(
    output_path: str,
    dt: float = 0.5,
    frames_per_scenario: int = 20,
) -> dict[str, int]:
    """Generate all scenarios, build obs sidecars, write JSONL.

    Returns:
        Dict mapping scenario_id to sample count.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    total = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for scenario_id, init_fn in SCENARIOS:
            frames = generate_scenario_frames(init_fn, frames_per_scenario, dt)
            scenario_count = 0

            # One SidecarContext per ego per scenario (reset between scenarios)
            ctxs: dict[int, SidecarContext] = {
                ego_idx: SidecarContext() for ego_idx in EGO_INDICES
            }

            for frame_idx, frame in enumerate(frames):
                timestamp_sec = frame_idx * dt

                for ego_idx in EGO_INDICES:
                    obs = build_obs_sidecar(
                        frame,
                        ego_platform_idx=ego_idx,
                        ctx=ctxs[ego_idx],
                    )

                    record = {
                        "obs": obs,
                        "meta": {
                            "scenario_id": scenario_id,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "ego_platform_idx": ego_idx,
                            "frame_idx": frame_idx,
                        },
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    scenario_count += 1

            counts[scenario_id] = scenario_count
            total += scenario_count

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 2v2 air combat obs-sidecar JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/d2b/synthetic_2v2_obs.jsonl",
        help="Output JSONL path (default: datasets/d2b/synthetic_2v2_obs.jsonl)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Time step in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--frames-per-scenario",
        type=int,
        default=20,
        help="Number of frames per scenario (default: 20)",
    )
    args = parser.parse_args()

    counts = generate_all(
        output_path=args.output,
        dt=args.dt,
        frames_per_scenario=args.frames_per_scenario,
    )

    total = sum(counts.values())
    print(f"\n=== Synthetic 2v2 generation complete ===")
    print(f"Output: {args.output}")
    print(f"Total samples: {total}")
    print(f"Per-scenario counts:")
    for sid, cnt in counts.items():
        print(f"  {sid}: {cnt}")


if __name__ == "__main__":
    main()
