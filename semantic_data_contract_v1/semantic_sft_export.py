from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from semantic_record_common import (
    REVIEW_INSTRUCTION,
    STANDARD_INSTRUCTION,
    canonicalize_evidence,
    read_json_or_jsonl,
    strip_nulls,
    validate_record,
    write_jsonl,
)


def _standard_input(obs: dict[str, Any]) -> dict[str, Any]:
    output = {"ego": strip_nulls(obs["ego"]) or {}, "history": strip_nulls(obs["history"]) or {}}
    for key in ("enemy", "ally", "engagement"):
        value = obs.get(key)
        if value is None:
            continue
        cleaned = strip_nulls(value)
        if cleaned:
            output[key] = cleaned
    return output


def _standard_output(label: dict[str, Any], *, include_rationale: bool) -> dict[str, Any]:
    payload = {
        "semantic_state": label.get("semantic_state"),
        "target_ref": label.get("target_ref"),
        "role": label.get("role"),
        "constraints": strip_nulls(label.get("constraints") or {}),
        "profile_hint": label.get("profile_hint"),
        "event_flags": strip_nulls(label.get("event_flags") or {}),
    }
    if include_rationale:
        payload["rationale_short"] = label.get("rationale_short")
    return strip_nulls(payload) or {}


def _review_input(record: dict[str, Any]) -> dict[str, Any]:
    evidence = canonicalize_evidence(record.get("evidence"))
    if record.get("quality", {}).get("conflict_flag") is True:
        evidence["notes"] = ((evidence.get("notes") or "").strip() + " [CONFLICT]").strip()
    return {
        "obs": _standard_input(record["obs"]),
        "evidence": strip_nulls(evidence, keep_empty_arrays=True) or {"rule_ids": [], "acmi_refs": [], "frame_refs": []},
    }


def export_sft_samples(
    canonical_records: list[dict[str, Any]],
    *,
    fmt: str,
    tiers: set[str],
    bronze_meta: bool,
    validate_input: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if fmt not in {"standard", "review"}:
        raise ValueError(f"Unsupported format: {fmt}")

    samples: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats = {
        "input_records": len(canonical_records),
        "skipped_semantic_state_null": 0,
        "skipped_weak": 0,
        "skipped_conflict_standard": 0,
        "schema_invalid": 0,
        "exported": 0,
        "tiers": {"gold": 0, "silver": 0, "bronze": 0},
    }

    for record in canonical_records:
        if validate_input:
            errors = validate_record(record)
            if errors:
                rejected.append({"sample_id": record.get("sample_id"), "errors": errors, "record": record})
                stats["schema_invalid"] += 1
                continue

        label = record["label"]
        quality = record["quality"]
        if label.get("semantic_state") is None:
            stats["skipped_semantic_state_null"] += 1
            continue
        if quality.get("tier") == "weak" or quality.get("tier") not in tiers:
            stats["skipped_weak"] += 1
            continue
        if fmt == "standard" and quality.get("conflict_flag") is True:
            stats["skipped_conflict_standard"] += 1
            continue

        sample = {
            "instruction": STANDARD_INSTRUCTION if fmt == "standard" else REVIEW_INSTRUCTION,
            "input": _standard_input(record["obs"]) if fmt == "standard" else _review_input(record),
            "output": _standard_output(label, include_rationale=(fmt == "review")),
        }
        if bronze_meta and quality.get("tier") == "bronze":
            sample["_meta"] = {
                "tier": "bronze",
                "confidence": quality.get("confidence"),
            }
        samples.append(sample)
        mapping_rows.append(
            {
                "line_number": len(samples),
                "sample_id": record["sample_id"],
                "quality_tier": quality.get("tier"),
                "conflict_flag": quality.get("conflict_flag", False),
            }
        )
        stats["exported"] += 1
        stats["tiers"][quality.get("tier")] += 1

    return samples, mapping_rows, {"stats": stats, "rejected": rejected}


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export semantic_record_v1 records into SFT JSONL samples.")
    parser.add_argument("--input", required=True, help="Input semantic_record_v1 JSON or JSONL")
    parser.add_argument("--output", required=True, help="Output SFT JSONL path")
    parser.add_argument("--format", required=True, choices=["standard", "review"], help="SFT export format")
    parser.add_argument("--tiers", default="gold,silver", help="Comma-separated quality tiers to include")
    parser.add_argument("--exclude-conflict", action="store_true", help="Ignored for review; standard skips conflicts anyway")
    parser.add_argument("--bronze-meta", action="store_true", help="Attach _meta to bronze samples")
    parser.add_argument("--validate", action="store_true", help="Validate canonical records before export")
    parser.add_argument("--output-mapping", help="Optional JSONL mapping from output line_number to sample_id")
    return parser


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()

    records = read_json_or_jsonl(args.input)
    tiers = {tier.strip() for tier in args.tiers.split(",") if tier.strip()}

    samples, mapping_rows, meta = export_sft_samples(
        records,
        fmt=args.format,
        tiers=tiers,
        bronze_meta=args.bronze_meta,
        validate_input=args.validate,
    )

    write_jsonl(args.output, samples)
    if args.output_mapping:
        write_jsonl(args.output_mapping, mapping_rows)
    if meta["rejected"]:
        rejected_path = f"{args.output}.rejected.jsonl"
        write_jsonl(rejected_path, meta["rejected"])

    stats = meta["stats"]
    print(f"[export_sft] input records: {stats['input_records']}", file=sys.stderr)
    print(f"[export_sft] skipped (semantic_state=null): {stats['skipped_semantic_state_null']}", file=sys.stderr)
    print(f"[export_sft] skipped (tier filter or weak): {stats['skipped_weak']}", file=sys.stderr)
    print(f"[export_sft] skipped (conflict, standard): {stats['skipped_conflict_standard']}", file=sys.stderr)
    print(f"[export_sft] schema invalid: {stats['schema_invalid']}", file=sys.stderr)
    print(f"[export_sft] exported: {stats['exported']}", file=sys.stderr)
    print(
        f"[export_sft]   gold: {stats['tiers']['gold']}, silver: {stats['tiers']['silver']}, bronze: {stats['tiers']['bronze']}",
        file=sys.stderr,
    )
    print(f"[export_sft] output file: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
