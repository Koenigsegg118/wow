"""Process preference judgment results into canonical records.

Input:  preference_cases.jsonl + human/AI preference choices
Output: canonical records with updated quality tiers

Usage:
    python -m tools.d2b.adapter_preference_results \
        --cases   datasets/d2b/preference_cases.jsonl \
        --results datasets/d2b/preference_results.jsonl \
        --output  datasets/d2b/canonical_from_preference.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def process_single(
    case: dict,
    chosen: str,  # "a" or "b"
    reviewer: str = "human",
    reviewer_notes: str | None = None,
) -> dict:
    """Process one preference judgment into a canonical record.

    The chosen option becomes the canonical label with elevated tier.
    """
    option = case[f"option_{chosen}"]
    obs = case.get("obs", {})
    label = option.get("label", {})
    conf = option.get("confidence")

    # Elevate tier for preference-resolved records
    # If human reviewed → silver minimum
    tier = "silver" if reviewer == "human" else "bronze"
    if conf is not None and conf >= 0.90 and reviewer == "human":
        tier = "gold"

    sample_id = f"pref_{case.get('case_id', uuid.uuid4().hex[:10])}"

    evidence_notes = f"Preference resolved: chose option_{chosen} ({option.get('source_type', '?')})"
    if reviewer_notes:
        evidence_notes += f"; reviewer: {reviewer_notes}"

    # Build rule_ids from chosen option
    rule_ids = option.get("rule_ids", [])

    return {
        "schema_version": "semantic_record_v1",
        "sample_id": sample_id,
        "source": {
            "platform": "afsim_acmi",
            "type": "mixed",
            "scenario_id": case.get("meta", {}).get("scenario_id"),
            "episode_id": None,
            "timestamp_sec": case.get("meta", {}).get("timestamp_sec"),
            "annotator": reviewer,
            "tool": "adapter_preference_results.py",
        },
        "obs": obs,
        "label": {
            "semantic_state": label.get("semantic_state"),
            "target_ref": label.get("target_ref"),
            "role": label.get("role"),
            "constraints": label.get("constraints", {
                "prefer_energy_preserve": None,
                "allow_aggressive_turn": None,
                "must_support_teammate": None,
                "should_abort_if_threatened": None,
            }),
            "profile_hint": label.get("profile_hint"),
            "event_flags": label.get("event_flags", {"target_switch": None}),
            "rationale_short": label.get("rationale_short"),
        },
        "evidence": {
            "rule_ids": rule_ids,
            "acmi_refs": [],
            "frame_refs": [],
            "notes": evidence_notes,
        },
        "quality": {
            "tier": tier,
            "confidence": conf,
            "needs_review": False,
            "conflict_flag": False,  # Resolved by preference
        },
        "extras": {},
    }


def process_batch(
    cases_path: pathlib.Path,
    results_path: pathlib.Path,
    output_path: pathlib.Path = pathlib.Path("datasets/d2b/canonical_from_preference.jsonl"),
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Batch process preference results.

    results_path is a JSONL where each line has:
        {"case_id": "...", "chosen": "a"|"b", "reviewer": "human", "notes": "..."}

    Returns:
        (output_path, stats)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cases indexed by case_id
    cases: dict[str, dict] = {}
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            cases[case["case_id"]] = case

    stats: dict[str, Any] = {
        "total_results": 0,
        "matched": 0,
        "unmatched": 0,
        "chosen_a": 0,
        "chosen_b": 0,
    }

    with open(results_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total_results"] += 1

            result = json.loads(line)
            case_id = result.get("case_id")
            chosen = result.get("chosen")
            reviewer = result.get("reviewer", "human")
            notes = result.get("notes")

            case = cases.get(case_id)
            if case is None:
                logger.warning("No case found for case_id=%s", case_id)
                stats["unmatched"] += 1
                continue

            if chosen not in ("a", "b"):
                logger.warning("Invalid chosen=%s for case_id=%s", chosen, case_id)
                stats["unmatched"] += 1
                continue

            canonical = process_single(case, chosen, reviewer, notes)
            fout.write(json.dumps(canonical, ensure_ascii=False) + "\n")
            stats["matched"] += 1
            stats[f"chosen_{chosen}"] += 1

    logger.info("Processed %d preference results → %s", stats["matched"], output_path)
    return output_path, stats


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Process preference judgment results")
    parser.add_argument("--cases", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str,
                        default="datasets/d2b/canonical_from_preference.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, stats = process_batch(
        pathlib.Path(args.cases),
        pathlib.Path(args.results),
        pathlib.Path(args.output),
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
