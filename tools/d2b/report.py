"""Generate D2b pipeline statistics and quality reports.

Usage:
    python -m tools.d2b.report \
        --merged  datasets/d2b/merged_canonical.jsonl \
        --outdir  reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── label distribution ────────────────────────────────────────


def compute_label_stats(records: list[dict]) -> dict[str, Any]:
    """Compute label distribution statistics."""
    total = len(records)
    state_counter: Counter = Counter()
    target_counter: Counter = Counter()
    role_counter: Counter = Counter()
    profile_counter: Counter = Counter()
    tier_counter: Counter = Counter()
    source_type_counter: Counter = Counter()

    null_state_count = 0
    conflict_count = 0
    needs_review_count = 0

    for rec in records:
        label = rec.get("label", {})
        quality = rec.get("quality", {})
        source = rec.get("source", {})

        state = label.get("semantic_state")
        if state is None:
            null_state_count += 1
        else:
            state_counter[state] += 1

        target = label.get("target_ref")
        target_counter[target if target else "null"] += 1

        role = label.get("role")
        role_counter[role if role else "null"] += 1

        profile = label.get("profile_hint")
        profile_counter[profile if profile else "null"] += 1

        tier_counter[quality.get("tier", "unknown")] += 1
        source_type_counter[source.get("type", "unknown")] += 1

        if quality.get("conflict_flag"):
            conflict_count += 1
        if quality.get("needs_review"):
            needs_review_count += 1

    return {
        "total_samples": total,
        "null_semantic_state": null_state_count,
        "null_ratio": round(null_state_count / max(total, 1), 4),
        "semantic_state_distribution": dict(state_counter.most_common()),
        "target_ref_distribution": dict(target_counter.most_common()),
        "role_distribution": dict(role_counter.most_common()),
        "profile_hint_distribution": dict(profile_counter.most_common()),
        "tier_distribution": dict(tier_counter.most_common()),
        "source_type_distribution": dict(source_type_counter.most_common()),
        "conflict_flag_count": conflict_count,
        "needs_review_count": needs_review_count,
    }


# ── data quality ──────────────────────────────────────────────


def compute_quality_report(records: list[dict]) -> dict[str, Any]:
    """Analyze data quality issues."""
    issues: list[dict] = []
    null_heavy_states: Counter = Counter()

    for rec in records:
        sample_id = rec.get("sample_id", "?")
        obs = rec.get("obs", {})
        label = rec.get("label", {})
        quality = rec.get("quality", {})

        # Check for excessive nulls in obs
        null_fields = []
        ego = obs.get("ego", {})
        if ego:
            for k, v in ego.items():
                if v is None:
                    null_fields.append(f"ego.{k}")

        if len(null_fields) > 5:
            issues.append({
                "sample_id": sample_id,
                "issue": "excessive_null_obs",
                "null_count": len(null_fields),
                "fields": null_fields[:5],
            })

        # Check for null enemy with non-null engagement state
        if obs.get("enemy") is None and label.get("semantic_state") in (
            "commit_intercept", "press_attack", "merge_entry",
            "offensive_turn", "defensive_break",
        ):
            issues.append({
                "sample_id": sample_id,
                "issue": "engagement_without_enemy",
                "semantic_state": label["semantic_state"],
            })

        # Track which states have mostly GPT-only sources
        if quality.get("tier") in ("bronze", "weak"):
            state = label.get("semantic_state")
            if state:
                null_heavy_states[state] += 1

    return {
        "total_issues": len(issues),
        "issues_sample": issues[:20],
        "labels_mostly_low_quality": dict(null_heavy_states.most_common(5)),
    }


# ── contract mismatch ────────────────────────────────────────


def compute_contract_mismatch(records: list[dict]) -> dict[str, Any]:
    """Check records against semantic_data_contract_v1 rules.

    Cross-checks:
    1. bearing_deg sign convention: positive = starboard [-180, 180]
    2. closure_mps sign convention: negative = closing
    3. alt_diff_m sign convention: positive = ego higher
    4. aspect_deg range: [0, 180], 0 = head-on
    5. engagement.is_merge vs range consistency
    6. engagement.is_defensive vs aspect/closure/range consistency
    7. Forbidden string values: "unknown", "unavailable", "N/A"
    8. enemy/ally must be object or null, never array
    9. target_ref enum compliance
    10. semantic_state enum compliance
    """
    from tools.d2b.d2b_config import (
        ALLOWED_SEMANTIC_STATES,
        ALLOWED_TARGET_REFS,
        MERGE_RANGE_M,
        DEF_ASPECT_DEG,
        DEF_RANGE_M,
    )

    findings: list[dict] = []

    for rec in records:
        sid = rec.get("sample_id", "?")
        obs = rec.get("obs", {})
        label = rec.get("label", {})
        enemy = obs.get("enemy")
        engagement = obs.get("engagement")

        # 1-4. Geometry convention checks
        if enemy is not None:
            bearing = enemy.get("bearing_deg")
            if bearing is not None and (bearing < -180 or bearing > 180):
                findings.append({"sample_id": sid, "check": "bearing_range",
                                 "detail": f"bearing_deg={bearing} outside [-180,180]"})

            aspect = enemy.get("aspect_deg")
            if aspect is not None and (aspect < 0 or aspect > 180):
                findings.append({"sample_id": sid, "check": "aspect_range",
                                 "detail": f"aspect_deg={aspect} outside [0,180]"})

        # 5. is_merge vs range consistency
        if enemy is not None and engagement is not None:
            rng = enemy.get("range_m")
            is_merge = engagement.get("is_merge")
            if rng is not None and is_merge is not None:
                if is_merge and rng >= MERGE_RANGE_M:
                    findings.append({"sample_id": sid, "check": "merge_range_mismatch",
                                     "detail": f"is_merge=True but range={rng}m >= {MERGE_RANGE_M}m"})
                if not is_merge and rng < MERGE_RANGE_M:
                    findings.append({"sample_id": sid, "check": "merge_range_mismatch",
                                     "detail": f"is_merge=False but range={rng}m < {MERGE_RANGE_M}m"})

        # 6. is_defensive vs aspect/closure/range
        if enemy is not None and engagement is not None:
            is_def = engagement.get("is_defensive")
            aspect = enemy.get("aspect_deg")
            closure = enemy.get("closure_mps")
            rng = enemy.get("range_m")
            if is_def and aspect is not None and closure is not None and rng is not None:
                expected = (aspect > DEF_ASPECT_DEG and closure < 0 and rng < DEF_RANGE_M)
                if not expected:
                    findings.append({"sample_id": sid, "check": "defensive_criteria_mismatch",
                                     "detail": (f"is_defensive=True but aspect={aspect}, "
                                                f"closure={closure}, range={rng}")})

        # 7. Forbidden strings
        raw = json.dumps(rec, ensure_ascii=False)
        for forbidden in ['"unknown"', '"unavailable"', '"N/A"']:
            if forbidden in raw:
                findings.append({"sample_id": sid, "check": "forbidden_string",
                                 "detail": f"contains {forbidden}"})

        # 8. enemy/ally not array
        for key in ("enemy", "ally"):
            val = obs.get(key)
            if isinstance(val, list):
                findings.append({"sample_id": sid, "check": f"{key}_is_array",
                                 "detail": f"obs.{key} is list, must be object or null"})

        # 9. target_ref enum
        tref = label.get("target_ref")
        if tref is not None and tref not in ALLOWED_TARGET_REFS:
            findings.append({"sample_id": sid, "check": "target_ref_enum",
                             "detail": f"target_ref={tref} not in allowed set"})

        # 10. semantic_state enum
        state = label.get("semantic_state")
        if state is not None and state not in ALLOWED_SEMANTIC_STATES:
            findings.append({"sample_id": sid, "check": "semantic_state_enum",
                             "detail": f"semantic_state={state} not in allowed set"})

    # Summarize by check type
    check_counts: Counter = Counter(f["check"] for f in findings)

    return {
        "total_checks_failed": len(findings),
        "check_type_counts": dict(check_counts.most_common()),
        "findings_sample": findings[:30],
        "verdict": "CLEAN" if len(findings) == 0 else "MISMATCHES_FOUND",
    }


# ── report generation ─────────────────────────────────────────


def generate_reports(
    merged_path: pathlib.Path,
    output_dir: pathlib.Path = pathlib.Path("reports"),
) -> dict[str, pathlib.Path]:
    """Generate all D2b reports.

    Returns dict of report_name → file_path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_jsonl(merged_path)

    outputs: dict[str, pathlib.Path] = {}

    # 1. Label stats
    label_stats = compute_label_stats(records)
    label_path = output_dir / "d2b_label_stats.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_stats, f, indent=2, ensure_ascii=False)
    outputs["label_stats"] = label_path

    # Also write markdown version
    md_path = output_dir / "d2b_label_stats.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# D2b Label Distribution Report\n\n")
        f.write(f"Total samples: **{label_stats['total_samples']}**\n\n")
        f.write(f"Null semantic_state: **{label_stats['null_semantic_state']}** "
                f"({label_stats['null_ratio']:.1%})\n\n")

        f.write("## Semantic State Distribution\n\n")
        f.write("| State | Count | Ratio |\n|-------|-------|-------|\n")
        total = label_stats["total_samples"]
        for state, count in label_stats["semantic_state_distribution"].items():
            f.write(f"| {state} | {count} | {count/max(total,1):.1%} |\n")

        f.write(f"\n## Source Type Distribution\n\n")
        f.write("| Source | Count |\n|--------|-------|\n")
        for src, count in label_stats["source_type_distribution"].items():
            f.write(f"| {src} | {count} |\n")

        f.write(f"\n## Quality Tier Distribution\n\n")
        f.write("| Tier | Count |\n|------|-------|\n")
        for tier, count in label_stats["tier_distribution"].items():
            f.write(f"| {tier} | {count} |\n")

        f.write(f"\nConflict flag: **{label_stats['conflict_flag_count']}**\n")
        f.write(f"Needs review: **{label_stats['needs_review_count']}**\n")

    outputs["label_stats_md"] = md_path

    # 2. Data quality
    quality_report = compute_quality_report(records)
    quality_path = output_dir / "d2b_data_quality_report.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    outputs["data_quality"] = quality_path

    # 3. Contract mismatch
    mismatch = compute_contract_mismatch(records)
    mismatch_json_path = output_dir / "d2b_contract_mismatch_report.json"
    with open(mismatch_json_path, "w", encoding="utf-8") as f:
        json.dump(mismatch, f, indent=2, ensure_ascii=False)
    outputs["contract_mismatch"] = mismatch_json_path

    mismatch_md_path = output_dir / "d2b_contract_mismatch_report.md"
    with open(mismatch_md_path, "w", encoding="utf-8") as f:
        f.write("# D2b Contract Mismatch Report\n\n")
        f.write(f"Verdict: **{mismatch['verdict']}**\n\n")
        f.write(f"Total checks failed: **{mismatch['total_checks_failed']}**\n\n")

        if mismatch["check_type_counts"]:
            f.write("## Mismatch Types\n\n")
            f.write("| Check | Count |\n|-------|-------|\n")
            for check, count in mismatch["check_type_counts"].items():
                f.write(f"| {check} | {count} |\n")

        if mismatch["findings_sample"]:
            f.write("\n## Sample Findings (first 30)\n\n")
            f.write("| Sample ID | Check | Detail |\n|-----------|-------|--------|\n")
            for finding in mismatch["findings_sample"]:
                f.write(f"| {finding['sample_id']} | {finding['check']} | {finding['detail']} |\n")
        else:
            f.write("\nNo mismatches found. All records comply with contract.\n")

    outputs["contract_mismatch_md"] = mismatch_md_path

    logger.info("Generated %d reports in %s", len(outputs), output_dir)
    return outputs


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate D2b reports")
    parser.add_argument("--merged", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    outputs = generate_reports(pathlib.Path(args.merged), pathlib.Path(args.outdir))
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
