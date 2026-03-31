"""Commit/press boundary A/B test for D2b dispute resolution.

Compares two boundary modes on the disputed head_on samples:
- guideline_strict: follows annotation_guideline_v1.md as-is
  (no automatic commit_intercept for >8km; GPT decides)
- doctrine_override_v1: allows commit_intercept at 8-15km
  when bearing<30, closure strongly negative, not defensive

This script does NOT re-call GPT. It re-evaluates existing GPT labels
against different boundary criteria and reports which cases would flip.

Usage:
    python -m tools.d2b.run_boundary_ab_test \
        --merged datasets/d2b/merged_canonical_live.jsonl \
        --output reports/commit_press_boundary_ablation.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from typing import Any

from tools.d2b.d2b_config import (
    COMMIT_MAX_BEARING_DEG,
    COMMIT_MIN_RANGE_M,
)


# ── Boundary mode definitions ────────────────────────────────

def _classify_guideline_strict(obs: dict, current_label: str) -> str | None:
    """guideline_strict: no automatic reclassification.

    Under strict mode, the boundary is not overridden at the merge layer.
    Whatever GPT or rule produced is kept as-is.
    Returns None = no change.
    """
    return None  # no override


def _classify_doctrine_override_v1(obs: dict, current_label: str) -> str | None:
    """doctrine_override_v1: reclassify press_attack -> commit_intercept
    when range > 8km and geometry supports stable intercept.

    Conditions for override:
    - current_label == "press_attack"
    - enemy exists
    - range > 8000m
    - |bearing| < COMMIT_MAX_BEARING_DEG (30)
    - closure < -100 (strongly closing)
    - not is_defensive
    - not is_merge

    Returns "commit_intercept" if override, None if no change.
    """
    if current_label != "press_attack":
        return None

    enemy = obs.get("enemy")
    if enemy is None:
        return None

    rng = enemy.get("range_m")
    bearing = enemy.get("bearing_deg")
    closure = enemy.get("closure_mps")

    if rng is None or bearing is None or closure is None:
        return None

    engagement = obs.get("engagement") or {}
    is_merge = engagement.get("is_merge", False)
    is_defensive = engagement.get("is_defensive", False)

    if is_merge or is_defensive:
        return None

    # Doctrine override: >8km + small bearing + strong closure
    if (rng > 8000.0
            and abs(bearing) < COMMIT_MAX_BEARING_DEG
            and closure < -100.0):
        return "commit_intercept"

    return None


BOUNDARY_MODES = {
    "guideline_strict": _classify_guideline_strict,
    "doctrine_override_v1": _classify_doctrine_override_v1,
}


# ── A/B test runner ───────────────────────────────────────────

def run_ab_test(
    merged_path: pathlib.Path,
) -> dict[str, Any]:
    """Run A/B test on dispute samples.

    Returns results dict with per-mode stats and per-sample decisions.
    """
    with open(merged_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Filter to press_attack records (the dispute candidates)
    dispute_candidates = []
    for rec in records:
        state = rec.get("label", {}).get("semantic_state")
        if state == "press_attack":
            dispute_candidates.append(rec)

    results: dict[str, Any] = {
        "total_records": len(records),
        "dispute_candidates": len(dispute_candidates),
        "modes": {},
        "per_sample": [],
    }

    for mode_name, classify_fn in BOUNDARY_MODES.items():
        flipped = 0
        kept = 0
        state_after = Counter()

        for rec in dispute_candidates:
            obs = rec.get("obs", {})
            current_state = rec.get("label", {}).get("semantic_state")
            new_state = classify_fn(obs, current_state)

            if new_state is not None and new_state != current_state:
                flipped += 1
                state_after[new_state] += 1
            else:
                kept += 1
                state_after[current_state] += 1

        results["modes"][mode_name] = {
            "flipped": flipped,
            "kept": kept,
            "state_distribution": dict(state_after),
        }

    # Per-sample detail (for report)
    for rec in dispute_candidates:
        obs = rec.get("obs", {})
        enemy = obs.get("enemy") or {}
        current_state = rec.get("label", {}).get("semantic_state")
        conf = rec.get("quality", {}).get("confidence")
        rationale = rec.get("label", {}).get("rationale_short", "")
        sample_id = rec.get("sample_id", "?")

        sample_info = {
            "sample_id": sample_id,
            "range_m": enemy.get("range_m"),
            "bearing_deg": enemy.get("bearing_deg"),
            "closure_mps": enemy.get("closure_mps"),
            "aspect_deg": enemy.get("aspect_deg"),
            "confidence": conf,
            "current_label": current_state,
            "rationale": (rationale or "")[:80],
            "modes": {},
        }

        for mode_name, classify_fn in BOUNDARY_MODES.items():
            new_state = classify_fn(obs, current_state)
            sample_info["modes"][mode_name] = {
                "result": new_state or current_state,
                "flipped": new_state is not None and new_state != current_state,
            }

        results["per_sample"].append(sample_info)

    return results


# ── Report generation ─────────────────────────────────────────

def generate_report(
    results: dict[str, Any],
    output_path: pathlib.Path,
) -> None:
    """Generate commit_press_boundary_ablation.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Commit / Press Boundary A/B Ablation Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Total merged records: {results['total_records']}")
    lines.append(f"- press_attack dispute candidates: {results['dispute_candidates']}")
    lines.append("")

    lines.append("## Mode Comparison")
    lines.append("")
    lines.append("| Mode | Flipped | Kept | Result Distribution |")
    lines.append("|------|---------|------|---------------------|")
    for mode, stats in results["modes"].items():
        dist_str = ", ".join(f"{k}:{v}" for k, v in stats["state_distribution"].items())
        lines.append(f"| {mode} | {stats['flipped']} | {stats['kept']} | {dist_str} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    gs = results["modes"].get("guideline_strict", {})
    do = results["modes"].get("doctrine_override_v1", {})
    lines.append(f"- **guideline_strict**: No override. All {gs.get('kept', 0)} "
                 f"press_attack labels preserved as-is.")
    lines.append(f"- **doctrine_override_v1**: {do.get('flipped', 0)} records "
                 f"reclassified press_attack -> commit_intercept "
                 f"(range>8km, bearing<30, closure<-100).")
    lines.append("")
    lines.append("Note: doctrine_override_v1 deviates from annotation_guideline_v1.md "
                 "which does not define a hard >8km commit boundary. "
                 "This mode treats >8km as an experience-based bias, not a rule.")
    lines.append("")

    lines.append("## Per-Sample Detail (first 20)")
    lines.append("")
    lines.append("| # | Range(m) | Bearing | Closure | Aspect | GPT Conf | "
                 "guideline_strict | doctrine_override | Flipped? |")
    lines.append("|---|----------|---------|---------|--------|----------|"
                 "-----------------|-------------------|----------|")

    for i, s in enumerate(results["per_sample"][:20]):
        rng = f"{s['range_m']:.0f}" if s['range_m'] else "?"
        brg = f"{s['bearing_deg']:.1f}" if s['bearing_deg'] is not None else "?"
        cls = f"{s['closure_mps']:.0f}" if s['closure_mps'] is not None else "?"
        asp = f"{s['aspect_deg']:.1f}" if s['aspect_deg'] is not None else "?"
        conf = f"{s['confidence']:.2f}" if s['confidence'] is not None else "?"
        gs_r = s["modes"]["guideline_strict"]["result"]
        do_r = s["modes"]["doctrine_override_v1"]["result"]
        flip = "YES" if s["modes"]["doctrine_override_v1"]["flipped"] else ""
        lines.append(
            f"| {i+1} | {rng} | {brg} | {cls} | {asp} | {conf} | "
            f"{gs_r} | {do_r} | {flip} |"
        )

    lines.append("")
    lines.append("## Current Default")
    lines.append("")
    lines.append("Default `LABEL_BOUNDARY_MODE = \"guideline_strict\"` in `d2b_config.py`.")
    lines.append("To switch: change `LABEL_BOUNDARY_MODE` and re-run the pipeline.")
    lines.append("All boundary changes go through config, not hardcoded in merge layer.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report: {output_path}")


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Commit/press boundary A/B test")
    parser.add_argument("--merged", type=str,
                        default="datasets/d2b/merged_canonical_live.jsonl")
    parser.add_argument("--output", type=str,
                        default="reports/commit_press_boundary_ablation.md")
    args = parser.parse_args()

    results = run_ab_test(pathlib.Path(args.merged))
    generate_report(results, pathlib.Path(args.output))

    # Summary
    for mode, stats in results["modes"].items():
        print(f"  {mode}: flipped={stats['flipped']}, kept={stats['kept']}")


if __name__ == "__main__":
    main()
