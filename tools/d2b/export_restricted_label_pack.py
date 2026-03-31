#!/usr/bin/env python
"""Export a restricted-label training pack from merged canonical JSONL.

Produces two JSONL files in SFT format and a manifest JSON:
  - restricted_nonnull_core.jsonl   : records with semantic_state in RESTRICTED_LABELS (not None),
                                      quality.tier != "weak", conflict_flag != True
  - restricted_with_null_review.jsonl : same core records PLUS records where semantic_state is None
  - restricted_manifest.json        : build metadata and statistics

Usage:
    python -m tools.d2b.export_restricted_label_pack \
        --input datasets/d2b/acmi_v2_merged.jsonl \
        --outdir datasets/trainpacks/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESTRICTED_LABELS = {
    "commit_intercept",
    "press_attack",
    "hold_geometry",
    "extend",
    "merge_entry",
    "defensive_break",
}

EXCLUDED_LABELS = {
    "support",
    "offensive_turn",
    "energy_manage",
}

SFT_INSTRUCTION = "根据以下空战态势观测，判断当前最合适的战术意图标签。"

# Fields forwarded into the SFT "input" dict (observation context).
# We keep every key that is NOT part of label/quality/meta bookkeeping.
_META_KEYS = {
    "schema_version",
    "label",
    "semantic_state",
    "quality",
    "conflict_flag",
    "source_type",
    "annotation_source",
    "annotator",
    "review_status",
}

# Fields forwarded into the SFT "output" dict.
_LABEL_KEYS = {
    "label",
    "semantic_state",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_obs_fields(record: dict) -> dict:
    """Return observation fields (everything that is not meta/label)."""
    return {k: v for k, v in record.items() if k not in _META_KEYS}


def _extract_label_fields(record: dict) -> dict:
    """Return label/output fields."""
    return {k: v for k, v in record.items() if k in _LABEL_KEYS}


def _to_sft_line(record: dict) -> str:
    """Convert a canonical record to an SFT JSON line."""
    sft = {
        "instruction": SFT_INSTRUCTION,
        "input": _extract_obs_fields(record),
        "output": _extract_label_fields(record),
    }
    return json.dumps(sft, ensure_ascii=False)


def _get_semantic_state(record: dict) -> str | None:
    """Retrieve semantic_state from top-level or nested label dict."""
    # Top-level semantic_state takes precedence.
    if "semantic_state" in record:
        return record["semantic_state"]
    label = record.get("label")
    if isinstance(label, dict):
        return label.get("semantic_state")
    return None


def _get_quality_tier(record: dict) -> str | None:
    quality = record.get("quality")
    if isinstance(quality, dict):
        return quality.get("tier")
    return None


def _get_conflict_flag(record: dict) -> bool:
    return bool(record.get("conflict_flag", False))


def _get_source_type(record: dict) -> str | None:
    return record.get("source_type")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_and_filter(input_path: Path):
    """Load canonical JSONL, apply base filters, and split into buckets.

    Returns
    -------
    core_records : list[dict]
        Records with restricted labels, non-weak tier, no conflict.
    null_records : list[dict]
        Records where semantic_state is None (for review set).
    stats : dict
        Counters for the manifest.
    """
    core_records: list[dict] = []
    null_records: list[dict] = []

    label_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    source_type_counts: Counter = Counter()

    total_read = 0
    skipped_no_schema = 0
    skipped_no_label_field = 0
    skipped_excluded_label = 0
    skipped_weak = 0
    skipped_conflict = 0
    skipped_other_label = 0

    with open(input_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSON at line {line_no}: {exc}",
                      file=sys.stderr)
                continue

            total_read += 1

            # --- Base filters (apply to both sets) ---

            # Must have schema_version
            if not record.get("schema_version"):
                skipped_no_schema += 1
                continue

            # Must have a label field (dict or value)
            if "label" not in record and "semantic_state" not in record:
                skipped_no_label_field += 1
                continue

            semantic_state = _get_semantic_state(record)
            tier = _get_quality_tier(record)
            source_type = _get_source_type(record)

            # --- Null bucket ---
            if semantic_state is None:
                null_records.append(record)
                label_counts[None] += 1
                tier_counts[tier] += 1
                source_type_counts[source_type] += 1
                continue

            # --- Excluded labels ---
            if semantic_state in EXCLUDED_LABELS:
                skipped_excluded_label += 1
                continue

            # --- Must be in RESTRICTED_LABELS ---
            if semantic_state not in RESTRICTED_LABELS:
                skipped_other_label += 1
                continue

            # At this point the label is in RESTRICTED_LABELS.
            # Track stats for both sets.
            label_counts[semantic_state] += 1
            tier_counts[tier] += 1
            source_type_counts[source_type] += 1

            # --- Quality gate (nonnull_core only) ---
            if tier == "weak":
                skipped_weak += 1
                # Still goes into null_review set (it has a valid label).
                # Actually no -- the requirement says nonnull_core excludes
                # weak, but null_review is "same as above PLUS null records".
                # So weak-tier restricted records are excluded from BOTH.
                continue

            if _get_conflict_flag(record):
                skipped_conflict += 1
                continue

            core_records.append(record)

    stats = {
        "total_read": total_read,
        "skipped_no_schema": skipped_no_schema,
        "skipped_no_label_field": skipped_no_label_field,
        "skipped_excluded_label": skipped_excluded_label,
        "skipped_weak_tier": skipped_weak,
        "skipped_conflict_flag": skipped_conflict,
        "skipped_other_label": skipped_other_label,
        "label_counts": dict(label_counts),
        "tier_counts": dict(tier_counts),
        "source_type_counts": dict(source_type_counts),
    }
    return core_records, null_records, stats


def write_jsonl(records: list[dict], path: Path) -> int:
    """Write SFT-format JSONL. Returns number of lines written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(_to_sft_line(rec) + "\n")
    return len(records)


def build_manifest(
    input_file: str,
    core_count: int,
    null_count: int,
    includes_null: bool,
    stats: dict,
) -> dict:
    """Build the restricted_manifest.json payload."""
    # Normalise None keys to string "null" for JSON compatibility.
    label_counts = {
        (k if k is not None else "null"): v
        for k, v in stats["label_counts"].items()
    }
    tier_counts = {
        (k if k is not None else "null"): v
        for k, v in stats["tier_counts"].items()
    }
    source_type_counts = {
        (k if k is not None else "null"): v
        for k, v in stats["source_type_counts"].items()
    }

    # Simple schema audit: every restricted label should appear at least once.
    schema_audit_passed = all(
        label_counts.get(lbl, 0) > 0 for lbl in RESTRICTED_LABELS
    )

    return {
        "build_time": datetime.now(timezone.utc).isoformat(),
        "input_file": input_file,
        "restricted_label_set": sorted(RESTRICTED_LABELS),
        "excluded_label_set": sorted(EXCLUDED_LABELS),
        "includes_null": includes_null,
        "label_counts": label_counts,
        "tier_counts": tier_counts,
        "source_type_counts": source_type_counts,
        "total_records": {
            "nonnull_core": core_count,
            "with_null_review": core_count + null_count,
        },
        "schema_audit_passed": schema_audit_passed,
        "filter_summary": {
            "total_read": stats["total_read"],
            "skipped_no_schema": stats["skipped_no_schema"],
            "skipped_no_label_field": stats["skipped_no_label_field"],
            "skipped_excluded_label": stats["skipped_excluded_label"],
            "skipped_weak_tier": stats["skipped_weak_tier"],
            "skipped_conflict_flag": stats["skipped_conflict_flag"],
            "skipped_other_label": stats["skipped_other_label"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export restricted-label SFT training pack from merged canonical JSONL.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to merged canonical JSONL (e.g. datasets/d2b/acmi_v2_merged.jsonl)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="datasets/trainpacks/",
        help="Output directory for the training pack (default: datasets/trainpacks/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path} ...")
    core_records, null_records, stats = load_and_filter(input_path)

    # --- Write nonnull_core ---
    core_path = outdir / "restricted_nonnull_core.jsonl"
    core_count = write_jsonl(core_records, core_path)
    print(f"  Wrote {core_count} records -> {core_path}")

    # --- Write with_null_review (core + null) ---
    review_path = outdir / "restricted_with_null_review.jsonl"
    combined = core_records + null_records
    review_count = write_jsonl(combined, review_path)
    print(f"  Wrote {review_count} records -> {review_path}")

    # --- Write manifest ---
    manifest = build_manifest(
        input_file=str(input_path),
        core_count=core_count,
        null_count=len(null_records),
        includes_null=len(null_records) > 0,
        stats=stats,
    )
    manifest_path = outdir / "restricted_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"  Wrote manifest -> {manifest_path}")

    # --- Summary ---
    print()
    print("=== Build Summary ===")
    print(f"  Total read:           {stats['total_read']}")
    print(f"  Nonnull core:         {core_count}")
    print(f"  Null (for review):    {len(null_records)}")
    print(f"  Combined (review):    {review_count}")
    print(f"  Schema audit passed:  {manifest['schema_audit_passed']}")
    if not manifest["schema_audit_passed"]:
        missing = [
            lbl for lbl in sorted(RESTRICTED_LABELS)
            if manifest["label_counts"].get(lbl, 0) == 0
        ]
        print(f"  Missing labels:       {missing}")
    print()
    print("Label distribution (restricted):")
    for lbl in sorted(RESTRICTED_LABELS):
        cnt = manifest["label_counts"].get(lbl, 0)
        print(f"  {lbl:25s} {cnt:>6d}")
    null_cnt = manifest["label_counts"].get("null", 0)
    if null_cnt:
        print(f"  {'(null)':25s} {null_cnt:>6d}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
