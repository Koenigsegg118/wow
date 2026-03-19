#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read shared/schema episodes and report smoke statistics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from shared.schema import SCHEMA_VERSION, load_episode


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_episode(path: Path) -> Dict[str, Any]:
    episode = load_episode(path)

    altitudes: List[float] = []
    speeds: List[float] = []
    abs_dpsis: List[float] = []
    frame_state_counts: List[int] = []

    for frame in episode["frames"]:
        states = frame["states"]
        frame_state_counts.append(len(states))
        for st in states:
            altitudes.append(float(st["z_u_m"]))
            speed = math.sqrt(
                float(st["vx_e_mps"]) ** 2 + float(st["vy_n_mps"]) ** 2 + float(st["vz_u_mps"]) ** 2
            )
            speeds.append(speed)

    for action_frame in episode["actions"]:
        for cmd in action_frame["commands"]:
            abs_dpsis.append(abs(float(cmd["dpsi_rad"])))

    frame_times = [float(f["t_s"]) for f in episode["frames"]]
    duration_s = frame_times[-1] - frame_times[0] if len(frame_times) >= 2 else 0.0

    return {
        "path": str(path),
        "schema_version": episode.get("schema_version"),
        "schema_ok": episode.get("schema_version") == SCHEMA_VERSION,
        "episode_id": episode.get("episode_id"),
        "time_step_s": float(episode.get("time_step_s", 0.0)),
        "n_entities": len(episode.get("entities", [])),
        "n_frames": len(episode.get("frames", [])),
        "n_actions": len(episode.get("actions", [])),
        "duration_s": float(duration_s),
        "states_per_frame_mean": float(statistics.mean(frame_state_counts)) if frame_state_counts else 0.0,
        "altitude_m": {
            "min": min(altitudes) if altitudes else 0.0,
            "p50": _percentile(altitudes, 50),
            "p95": _percentile(altitudes, 95),
            "max": max(altitudes) if altitudes else 0.0,
        },
        "speed_mps": {
            "min": min(speeds) if speeds else 0.0,
            "p50": _percentile(speeds, 50),
            "p95": _percentile(speeds, 95),
            "max": max(speeds) if speeds else 0.0,
        },
        "abs_dpsi_rad": {
            "min": min(abs_dpsis) if abs_dpsis else 0.0,
            "p50": _percentile(abs_dpsis, 50),
            "p95": _percentile(abs_dpsis, 95),
            "max": max(abs_dpsis) if abs_dpsis else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-check episode files against shared/schema contract")
    ap.add_argument("--episode", nargs="+", required=True, help="Episode JSON file(s)")
    ap.add_argument("--out_json", default=None, help="Optional output report path")
    args = ap.parse_args()

    reports = [summarize_episode(Path(p)) for p in args.episode]

    for item in reports:
        status = "OK" if item["schema_ok"] else "MISMATCH"
        print(f"[{status}] {item['path']}")
        print(
            "  id={episode_id} entities={n_entities} frames={n_frames} actions={n_actions} "
            "dt={time_step_s:.3f}s duration={duration_s:.1f}s".format(**item)
        )
        print(
            "  altitude(m): min={min:.1f} p50={p50:.1f} p95={p95:.1f} max={max:.1f}".format(
                **item["altitude_m"]
            )
        )
        print(
            "  speed(m/s): min={min:.1f} p50={p50:.1f} p95={p95:.1f} max={max:.1f}".format(
                **item["speed_mps"]
            )
        )
        print(
            "  |dpsi|(rad): min={min:.4f} p50={p50:.4f} p95={p95:.4f} max={max:.4f}".format(
                **item["abs_dpsi_rad"]
            )
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote report: {out_path}")

    if not all(item["schema_ok"] for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
