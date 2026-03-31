"""Audit event uniqueness across the ACMI D2b pipeline.

Traces row counts and dedup-key overlap from candidate_events_balanced
through to acmi_merged_canonical, finding where row expansion occurs.

Usage:
    cd D:\\AF2.9\\afsim-2.9.0-win64\\wow
    python -m tools.acmi_mining.audit_event_uniqueness
    python -m tools.acmi_mining.audit_event_uniqueness --report reports/event_uniqueness_audit.md
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────

BASE = pathlib.Path("datasets")
ACMI_MINING = BASE / "acmi_mining"
D2B = BASE / "d2b"

PIPELINE_FILES = [
    ("candidate_events_balanced",  ACMI_MINING / "candidate_events_balanced.jsonl"),
    ("acmi_obs_samples",           D2B / "acmi_obs_samples.jsonl"),
    ("acmi_formatA_requests",      D2B / "acmi_formatA_requests.jsonl"),
    ("acmi_rule_labeled",          D2B / "acmi_rule_labeled.jsonl"),
    ("acmi_hardcase_requests",     D2B / "acmi_hardcase_requests.jsonl"),
    ("acmi_gpt_responses",         D2B / "acmi_gpt_responses.jsonl"),
    ("acmi_gpt_validated",         D2B / "acmi_gpt_validated.jsonl"),
    ("acmi_canonical_from_gpt",    D2B / "acmi_canonical_from_gpt.jsonl"),
    ("acmi_canonical_guarded",     D2B / "acmi_canonical_guarded.jsonl"),
    ("acmi_merged_canonical",      D2B / "acmi_merged_canonical.jsonl"),
    ("acmi_sft_standard",          D2B / "acmi_sft_standard.jsonl"),
    ("acmi_sft_review",            D2B / "acmi_sft_review.jsonl"),
]


# ── helpers ────────────────────────────────────────────────────

def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _detect_id_field(records: list[dict]) -> str | None:
    """Detect the primary ID field in a list of records."""
    candidates = ["event_id", "sample_id", "request_id"]
    if not records:
        return None
    keys = set(records[0].keys())
    for c in candidates:
        if c in keys:
            return c
    # Check nested meta.request_id
    meta = records[0].get("meta", {})
    if isinstance(meta, dict) and "request_id" in meta:
        return "meta.request_id"
    return None


def _get_id(record: dict, field: str) -> str | None:
    """Extract ID from record given field path."""
    if "." in field:
        parts = field.split(".")
        cur = record
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        return str(cur) if cur is not None else None
    return str(record.get(field)) if record.get(field) is not None else None


def _extract_merge_dedup_key(record: dict) -> str:
    """Replicate merge_and_export._extract_dedup_key for analysis."""
    source = record.get("source", {})
    ego = record.get("obs", {}).get("ego", {})
    parts = [
        source.get("scenario_id", ""),
        str(source.get("timestamp_sec", "")),
        ego.get("runtime_id", "") if ego else "",
    ]
    key = "|".join(parts)
    return key if key != "||" else record.get("sample_id", "")


# ── main audit ─────────────────────────────────────────────────

def audit() -> dict[str, Any]:
    """Run the full audit and return structured results."""
    results: dict[str, Any] = {"stages": {}, "diagnosis": {}}

    # Stage 1: Count rows and IDs per file
    for name, path in PIPELINE_FILES:
        info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            info["rows"] = 0
            results["stages"][name] = info
            continue

        records = _load_jsonl(path)
        info["rows"] = len(records)

        id_field = _detect_id_field(records)
        if id_field:
            ids = [_get_id(r, id_field) for r in records]
            unique_ids = set(ids)
            info["id_field"] = id_field
            info["unique_ids"] = len(unique_ids)
            info["duplicate_ids"] = len(ids) - len(unique_ids)
            if info["duplicate_ids"] > 0:
                counter = Counter(ids)
                info["dup_examples"] = {
                    k: v for k, v in counter.most_common(5) if v > 1
                }

        # Special analysis per file type
        if name == "acmi_rule_labeled":
            canonical = [r for r in records if "schema_version" in r]
            stubs = [r for r in records if r.get("_hard_case")]
            info["canonical_count"] = len(canonical)
            info["stub_count"] = len(stubs)

        if name == "acmi_canonical_from_gpt" or name == "acmi_canonical_guarded":
            null_state = sum(
                1 for r in records
                if r.get("label", {}).get("semantic_state") is None
            )
            info["null_state_count"] = null_state
            info["non_null_state_count"] = len(records) - null_state

        if name == "acmi_merged_canonical":
            canonical = [r for r in records if "schema_version" in r]
            stubs = [r for r in records if r.get("_hard_case")]
            rule_sourced = sum(
                1 for r in records
                if r.get("source", {}).get("type") == "afsim_rule"
            )
            gpt_sourced = sum(
                1 for r in records
                if r.get("source", {}).get("type") == "acmi_ai"
            )
            null_state = sum(
                1 for r in records
                if r.get("label", {}).get("semantic_state") is None
            )
            info["canonical_count"] = len(canonical)
            info["stub_count"] = len(stubs)
            info["rule_sourced"] = rule_sourced
            info["gpt_sourced"] = gpt_sourced
            info["null_state_merged"] = null_state

        results["stages"][name] = info

    # Stage 2: Dedup key analysis for merge
    rule_records = _load_jsonl(D2B / "acmi_rule_labeled.jsonl")
    gpt_records = _load_jsonl(D2B / "acmi_canonical_guarded.jsonl")
    if not gpt_records:
        gpt_records = _load_jsonl(D2B / "acmi_canonical_from_gpt.jsonl")

    rule_keys: dict[str, str] = {}  # dedup_key -> record_type
    for rec in rule_records:
        key = _extract_merge_dedup_key(rec)
        is_stub = bool(rec.get("_hard_case"))
        rule_keys[key] = "stub" if is_stub else "canonical"

    stub_keys = {k for k, v in rule_keys.items() if v == "stub"}
    canonical_rule_keys = {k for k, v in rule_keys.items() if v == "canonical"}

    gpt_keys = set()
    gpt_matching_rule = 0
    gpt_matching_stub = 0
    gpt_new = 0
    for rec in gpt_records:
        key = _extract_merge_dedup_key(rec)
        gpt_keys.add(key)
        if key in canonical_rule_keys:
            gpt_matching_rule += 1
        elif key in stub_keys:
            gpt_matching_stub += 1
        else:
            gpt_new += 1

    results["diagnosis"]["dedup_keys"] = {
        "rule_canonical_keys": len(canonical_rule_keys),
        "rule_stub_keys": len(stub_keys),
        "rule_total_keys": len(rule_keys),
        "gpt_total_keys": len(gpt_keys),
        "gpt_matching_rule_canonical": gpt_matching_rule,
        "gpt_matching_stub": gpt_matching_stub,
        "gpt_new_keys": gpt_new,
        "stub_keys_never_overwritten": len(stub_keys - gpt_keys),
    }

    # Stage 3: Show the orphan stub keys
    orphan_stubs = stub_keys - gpt_keys
    results["diagnosis"]["orphan_stub_keys"] = sorted(orphan_stubs)

    # Stage 4: Expected vs actual merged count
    # After merge: rule_canonical_keys that survive (not overwritten or kept)
    #   + stub_keys that survive (never matched by GPT)
    #   + gpt_new_keys
    #   + gpt_matching_rule_canonical (overwrites, net same count)
    # = canonical_rule_keys + orphan_stubs + gpt_new
    expected_merged = len(canonical_rule_keys) + len(orphan_stubs) + gpt_new
    actual_merged = results["stages"].get("acmi_merged_canonical", {}).get("rows", 0)

    results["diagnosis"]["expected_merged_count"] = expected_merged
    results["diagnosis"]["actual_merged_count"] = actual_merged
    results["diagnosis"]["count_match"] = expected_merged == actual_merged

    # Stage 5: Root cause
    if len(orphan_stubs) > 0:
        results["diagnosis"]["root_cause"] = (
            f"Rule labeler emits hard-case stubs with dedup key '||<runtime_id>' "
            f"(missing source.scenario_id and source.timestamp_sec). "
            f"These keys NEVER match GPT canonical records which have proper keys. "
            f"Result: {len(orphan_stubs)} orphan stubs survive in merged output, "
            f"inflating row count from {len(canonical_rule_keys) + gpt_new} "
            f"to {actual_merged}."
        )
        results["diagnosis"]["is_bug"] = True
        results["diagnosis"]["fix_recommendation"] = (
            "In merge_and_export.merge_sources(), filter out records with "
            "'_hard_case': True before inserting into merged dict. "
            "Alternatively, fix rule_labeler_v1.label_batch() to write stubs "
            "with proper source.scenario_id and source.timestamp_sec from meta, "
            "so their dedup keys match GPT canonical records."
        )
    else:
        results["diagnosis"]["root_cause"] = "No orphan stubs found."
        results["diagnosis"]["is_bug"] = False

    return results


def print_report(results: dict[str, Any]) -> None:
    """Print a human-readable report to stdout."""
    print("=" * 70)
    print("  ACMI Event Uniqueness Audit")
    print("=" * 70)

    print("\n--- Row Counts by Pipeline Stage ---\n")
    print(f"{'Stage':<30} {'Rows':>6}  {'Details'}")
    print("-" * 70)
    for name, info in results["stages"].items():
        rows = info.get("rows", 0)
        details = []
        if "canonical_count" in info:
            details.append(f"canonical={info['canonical_count']}")
        if "stub_count" in info:
            details.append(f"stubs={info['stub_count']}")
        if "rule_sourced" in info:
            details.append(f"rule={info['rule_sourced']}")
        if "gpt_sourced" in info:
            details.append(f"gpt={info['gpt_sourced']}")
        if "null_state_count" in info:
            details.append(f"null_state={info['null_state_count']}")
        if "null_state_merged" in info:
            details.append(f"null_state={info['null_state_merged']}")
        if "duplicate_ids" in info and info["duplicate_ids"] > 0:
            details.append(f"dup_ids={info['duplicate_ids']}")
        if not info.get("exists"):
            details.append("NOT FOUND")
        detail_str = ", ".join(details)
        marker = " <--" if name == "acmi_merged_canonical" else ""
        print(f"  {name:<28} {rows:>6}  {detail_str}{marker}")

    print("\n--- Dedup Key Analysis ---\n")
    dk = results["diagnosis"].get("dedup_keys", {})
    for k, v in dk.items():
        print(f"  {k}: {v}")

    print(f"\n  Expected merged count: {results['diagnosis'].get('expected_merged_count')}")
    print(f"  Actual merged count:   {results['diagnosis'].get('actual_merged_count')}")
    print(f"  Match: {results['diagnosis'].get('count_match')}")

    if results["diagnosis"].get("orphan_stub_keys"):
        print(f"\n  Orphan stub keys ({len(results['diagnosis']['orphan_stub_keys'])}):")
        for k in results["diagnosis"]["orphan_stub_keys"]:
            print(f"    - {k!r}")

    print("\n--- Root Cause ---\n")
    print(f"  {results['diagnosis'].get('root_cause', 'Unknown')}")
    print(f"  Is bug: {results['diagnosis'].get('is_bug', 'Unknown')}")

    if results["diagnosis"].get("fix_recommendation"):
        print(f"\n--- Fix Recommendation ---\n")
        print(f"  {results['diagnosis']['fix_recommendation']}")

    print("\n" + "=" * 70)


def write_markdown_report(
    results: dict[str, Any],
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Write audit results as a Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ACMI Event Uniqueness Audit",
        "",
        "## Row Counts by Pipeline Stage",
        "",
        "| Stage | Rows | Details |",
        "|-------|------|---------|",
    ]

    for name, info in results["stages"].items():
        rows = info.get("rows", 0)
        details = []
        if "canonical_count" in info:
            details.append(f"canonical={info['canonical_count']}")
        if "stub_count" in info:
            details.append(f"stubs={info['stub_count']}")
        if "rule_sourced" in info:
            details.append(f"rule={info['rule_sourced']}")
        if "gpt_sourced" in info:
            details.append(f"gpt={info['gpt_sourced']}")
        if "null_state_count" in info:
            details.append(f"null_state={info['null_state_count']}")
        if "null_state_merged" in info:
            details.append(f"null_state={info['null_state_merged']}")
        if "duplicate_ids" in info and info["duplicate_ids"] > 0:
            details.append(f"dup_ids={info['duplicate_ids']}")
        if not info.get("exists"):
            details.append("NOT FOUND")
        detail_str = ", ".join(details)
        lines.append(f"| {name} | {rows} | {detail_str} |")

    lines += [
        "",
        "## Dedup Key Analysis",
        "",
    ]

    dk = results["diagnosis"].get("dedup_keys", {})
    for k, v in dk.items():
        lines.append(f"- **{k}**: {v}")

    lines += [
        "",
        f"- Expected merged count: **{results['diagnosis'].get('expected_merged_count')}**",
        f"- Actual merged count: **{results['diagnosis'].get('actual_merged_count')}**",
        f"- Match: {results['diagnosis'].get('count_match')}",
    ]

    orphan_keys = results["diagnosis"].get("orphan_stub_keys", [])
    if orphan_keys:
        lines += [
            "",
            f"## Orphan Stub Keys ({len(orphan_keys)})",
            "",
            "These hard-case stubs from `rule_labeler_v1` survive in the merged output",
            "because their dedup keys never match any GPT canonical record:",
            "",
        ]
        for k in orphan_keys:
            lines.append(f"- `{k}`")

    lines += [
        "",
        "## Root Cause",
        "",
        results["diagnosis"].get("root_cause", "Unknown"),
        "",
        f"**Is bug:** {results['diagnosis'].get('is_bug', 'Unknown')}",
    ]

    fix = results["diagnosis"].get("fix_recommendation")
    if fix:
        lines += [
            "",
            "## Fix Recommendation",
            "",
            fix,
        ]

    lines += [
        "",
        "## Data Flow Diagram",
        "",
        "```",
        "candidate_events_balanced (140)",
        "  |",
        "  v  [1:1 adapter: event -> obs+meta]",
        "acmi_obs_samples (140)",
        "  |",
        "  +---> acmi_rule_labeled (140 = 80 canonical + 60 stubs)",
        "  |       |",
        "  |       +---> [merge_sources: rule input]",
        "  |                 |",
        "  +---> acmi_hardcase_requests (121)",
        "          |",
        "          v  [GPT caller]",
        "        acmi_gpt_responses (121)",
        "          |",
        "          v  [validator]",
        "        acmi_gpt_validated (121)",
        "          |",
        "          v  [adapter to canonical]",
        "        acmi_canonical_from_gpt (121, 34 null-state)",
        "          |",
        "          v  [guardrails]",
        "        acmi_canonical_guarded (121)",
        "              |",
        "              +---> [merge_sources: gpt input]",
        "                         |",
        "                         v",
        "                  acmi_merged_canonical (153 = 31 rule + 109 gpt + 13 stubs)",
        "                         |",
        "                         +---> acmi_sft_standard (70)",
        "                         +---> acmi_sft_review (118)",
        "```",
        "",
        "### Why 153 instead of 140?",
        "",
        "The rule labeler writes 60 hard-case stubs. These stubs lack",
        "`source.scenario_id` and `source.timestamp_sec`, so the merge",
        "dedup key becomes `||<runtime_id>`. Multiple stubs with the",
        "same `runtime_id` collapse, leaving 13 distinct stub entries.",
        "",
        "GPT canonical records have proper keys like",
        "`<scenario_id>|<timestamp_sec>|<runtime_id>`,",
        "which never match the stub keys `||<runtime_id>`.",
        "So 13 stubs are added to the merged output alongside",
        "the 140 intended records, inflating the total to 153.",
        "",
        "### Impact",
        "",
        "The 13 stub records in `acmi_merged_canonical.jsonl`:",
        "- Have no `schema_version`, `label`, or `quality` fields",
        "- Carry `_hard_case: true` and `_line_no` fields",
        "- Are mostly harmless: SFT export skips them (no `label.semantic_state`)",
        "- But they pollute the merged file and inflate reported counts",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit event uniqueness across the ACMI D2b pipeline."
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        default=pathlib.Path("reports/event_uniqueness_audit.md"),
        help="Path for Markdown report output.",
    )
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        default=None,
        help="Path for JSON results output.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    results = audit()
    print_report(results)

    md_path = write_markdown_report(results, args.report)
    print(f"\nMarkdown report: {md_path}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON results: {args.json}")


if __name__ == "__main__":
    main()
