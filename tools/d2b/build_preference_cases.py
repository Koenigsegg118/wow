"""Build preference judgment cases from rule vs GPT label conflicts.

Only targets high-value conflicts:
1. Rule label vs GPT label disagree on semantic_state
2. Same sample has two plausible labels with similar confidence
3. target_ref selection conflicts

Usage:
    python -m tools.d2b.build_preference_cases \
        --rule    datasets/d2b/rule_labeled.jsonl \
        --gpt     datasets/d2b/canonical_from_gpt.jsonl \
        --output  datasets/d2b/preference_cases.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PreferenceCase:
    """A conflict requiring human or AI preference judgment."""
    case_id: str
    obs: dict
    option_a: dict  # {label, source_type, confidence, rule_ids}
    option_b: dict
    conflict_type: str  # "state_conflict", "target_conflict", "ambiguous"
    meta: dict = field(default_factory=dict)


def _extract_key(record: dict) -> str:
    """Extract a pairing key from canonical record."""
    meta_fields = []
    source = record.get("source", {})
    if source.get("scenario_id"):
        meta_fields.append(source["scenario_id"])
    if source.get("timestamp_sec") is not None:
        meta_fields.append(f"t{source['timestamp_sec']}")
    obs = record.get("obs", {})
    ego = obs.get("ego", {})
    if ego and ego.get("runtime_id"):
        meta_fields.append(ego["runtime_id"])
    return "|".join(meta_fields) if meta_fields else record.get("sample_id", "")


def find_conflicts(
    rule_records: list[dict],
    gpt_records: list[dict],
) -> list[PreferenceCase]:
    """Find conflicts between rule-labeled and GPT-labeled records.

    Matches records by obs similarity (same scenario + timestamp + ego_id).
    """
    # Build index from rule records
    rule_index: dict[str, dict] = {}
    for rec in rule_records:
        key = _extract_key(rec)
        if key:
            rule_index[key] = rec

    cases: list[PreferenceCase] = []

    for gpt_rec in gpt_records:
        key = _extract_key(gpt_rec)
        rule_rec = rule_index.get(key)
        if rule_rec is None:
            continue

        rule_state = rule_rec.get("label", {}).get("semantic_state")
        gpt_state = gpt_rec.get("label", {}).get("semantic_state")
        rule_target = rule_rec.get("label", {}).get("target_ref")
        gpt_target = gpt_rec.get("label", {}).get("target_ref")

        # Skip if both are null
        if rule_state is None and gpt_state is None:
            continue

        conflict_type = None

        # Check state conflict
        if rule_state != gpt_state and rule_state is not None and gpt_state is not None:
            conflict_type = "state_conflict"

        # Check target conflict
        elif rule_target != gpt_target and rule_target is not None and gpt_target is not None:
            conflict_type = "target_conflict"

        if conflict_type is None:
            continue

        case = PreferenceCase(
            case_id=f"pref_{uuid.uuid4().hex[:10]}",
            obs=gpt_rec.get("obs", {}),
            option_a={
                "label": rule_rec.get("label", {}),
                "source_type": "afsim_rule",
                "confidence": rule_rec.get("quality", {}).get("confidence"),
                "rule_ids": rule_rec.get("evidence", {}).get("rule_ids", []),
            },
            option_b={
                "label": gpt_rec.get("label", {}),
                "source_type": "acmi_ai",
                "confidence": gpt_rec.get("quality", {}).get("confidence"),
                "rule_ids": [],
            },
            conflict_type=conflict_type,
            meta={
                "rule_sample_id": rule_rec.get("sample_id"),
                "gpt_sample_id": gpt_rec.get("sample_id"),
            },
        )
        cases.append(case)

    return cases


def find_ambiguous_gpt(
    gpt_records: list[dict],
    confidence_threshold: float = 0.60,
) -> list[PreferenceCase]:
    """Find GPT-labeled records with low confidence (ambiguous)."""
    cases: list[PreferenceCase] = []

    for rec in gpt_records:
        conf = rec.get("quality", {}).get("confidence")
        state = rec.get("label", {}).get("semantic_state")

        if conf is None or state is None:
            continue
        if conf >= confidence_threshold:
            continue

        case = PreferenceCase(
            case_id=f"pref_amb_{uuid.uuid4().hex[:10]}",
            obs=rec.get("obs", {}),
            option_a={
                "label": rec.get("label", {}),
                "source_type": "acmi_ai",
                "confidence": conf,
                "rule_ids": [],
            },
            option_b={
                "label": {"semantic_state": None},  # placeholder for reviewer
                "source_type": "manual",
                "confidence": None,
                "rule_ids": [],
            },
            conflict_type="ambiguous",
            meta={"gpt_sample_id": rec.get("sample_id")},
        )
        cases.append(case)

    return cases


def build_preference_batch(
    rule_path: pathlib.Path,
    gpt_path: pathlib.Path,
    output_path: pathlib.Path = pathlib.Path("datasets/d2b/preference_cases.jsonl"),
    ambiguity_threshold: float = 0.60,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Build all preference cases and write to JSONL.

    Returns:
        (output_path, stats)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load records
    def _load_jsonl(path: pathlib.Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    rule_records = _load_jsonl(rule_path)
    gpt_records = _load_jsonl(gpt_path)

    # Find conflicts
    conflict_cases = find_conflicts(rule_records, gpt_records)
    ambiguous_cases = find_ambiguous_gpt(gpt_records, ambiguity_threshold)

    all_cases = conflict_cases + ambiguous_cases

    stats = {
        "state_conflicts": len([c for c in conflict_cases if c.conflict_type == "state_conflict"]),
        "target_conflicts": len([c for c in conflict_cases if c.conflict_type == "target_conflict"]),
        "ambiguous": len(ambiguous_cases),
        "total": len(all_cases),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        for case in all_cases:
            record = {
                "case_id": case.case_id,
                "obs": case.obs,
                "option_a": case.option_a,
                "option_b": case.option_b,
                "conflict_type": case.conflict_type,
                "meta": case.meta,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Built %d preference cases → %s", len(all_cases), output_path)
    return output_path, stats


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build preference judgment cases")
    parser.add_argument("--rule", type=str, required=True)
    parser.add_argument("--gpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="datasets/d2b/preference_cases.jsonl")
    parser.add_argument("--ambiguity-threshold", type=float, default=0.60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, stats = build_preference_batch(
        pathlib.Path(args.rule),
        pathlib.Path(args.gpt),
        pathlib.Path(args.output),
        args.ambiguity_threshold,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
