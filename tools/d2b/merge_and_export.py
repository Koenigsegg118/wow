"""Merge three data sources and export SFT datasets.

Merge priority: preference > gpt > rule (later source overrides earlier).

Usage:
    python -m tools.d2b.merge_and_export \
        --rule       datasets/d2b/rule_labeled.jsonl \
        --gpt        datasets/d2b/canonical_from_gpt.jsonl \
        --preference datasets/d2b/canonical_from_preference.jsonl \
        --output     datasets/d2b/merged_canonical.jsonl \
        --sft-standard datasets/d2b/sft_standard.jsonl \
        --sft-review   datasets/d2b/sft_review.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


# ── merge ─────────────────────────────────────────────────────


def _extract_dedup_key(record: dict) -> str:
    """Extract dedup key from canonical record.

    Uses source.scenario_id + timestamp_sec + ego.runtime_id.
    Falls back to sample_id.
    """
    source = record.get("source", {})
    ego = record.get("obs", {}).get("ego", {})
    parts = [
        source.get("scenario_id", ""),
        str(source.get("timestamp_sec", "")),
        ego.get("runtime_id", "") if ego else "",
    ]
    key = "|".join(parts)
    return key if key != "||" else record.get("sample_id", "")


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    """Load JSONL file, skip empty lines."""
    records = []
    if not path.exists():
        logger.warning("File not found: %s, skipping", path)
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merge_sources(
    rule_path: pathlib.Path,
    gpt_path: pathlib.Path,
    preference_path: pathlib.Path | None = None,
    output_path: pathlib.Path = pathlib.Path("datasets/d2b/merged_canonical.jsonl"),
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Merge rule + gpt + preference canonical records.

    Dedup strategy: preference > gpt > rule (later overwrites earlier).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rule_records = _load_jsonl(rule_path)
    # Filter out hard-case stubs: they lack source/label/quality and produce
    # orphan dedup keys (||runtime_id) that persist alongside real canonical
    # records.  See reports/event_uniqueness_audit.md for details.
    rule_records = [r for r in rule_records if not r.get("_hard_case")]
    gpt_records = _load_jsonl(gpt_path)
    pref_records = _load_jsonl(preference_path) if preference_path else []

    # Build merged dict: key → record (later sources overwrite)
    merged: dict[str, dict] = {}
    source_counts = {"afsim_rule": 0, "acmi_ai": 0, "mixed": 0}

    for rec in rule_records:
        key = _extract_dedup_key(rec)
        merged[key] = rec

    for rec in gpt_records:
        key = _extract_dedup_key(rec)
        gpt_state = rec.get("label", {}).get("semantic_state")
        existing = merged.get(key)
        if existing is not None and gpt_state is None:
            # Don't overwrite a good rule label with a null GPT label
            continue
        merged[key] = rec  # overwrites rule only if GPT has real label

    for rec in pref_records:
        key = _extract_dedup_key(rec)
        merged[key] = rec  # overwrites gpt/rule

    # Write merged output
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in merged.values():
            stype = rec.get("source", {}).get("type", "unknown")
            if stype in source_counts:
                source_counts[stype] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "rule_input": len(rule_records),
        "gpt_input": len(gpt_records),
        "preference_input": len(pref_records),
        "merged_total": len(merged),
        "source_distribution": source_counts,
    }

    logger.info("Merged %d records → %s", len(merged), output_path)
    return output_path, stats


# ── SFT export ────────────────────────────────────────────────


_STANDARD_INSTRUCTION = (
    "根据以下空战态势信息，判断己方飞机当前的战术意图。"
    "请输出 JSON 格式的标签，包含 semantic_state、target_ref、role、"
    "constraints、profile_hint 和 event_flags。仅输出你有把握判断的字段。"
)

_REVIEW_INSTRUCTION = (
    "请审阅以下空战态势观测和标注证据，综合判断己方飞机当前的战术意图。"
    "请输出 JSON 格式的标签，包含 semantic_state、target_ref、role、"
    "constraints、profile_hint、event_flags 和 rationale_short。"
    "仅输出你有把握判断的字段。"
)


def _strip_nulls(d: dict) -> dict:
    """Remove keys with None values from dict (shallow)."""
    return {k: v for k, v in d.items() if v is not None}


def _build_sft_input(obs: dict, include_evidence: bool = False,
                     evidence: dict | None = None) -> dict:
    """Build SFT input from obs, stripping null fields.

    Per sft_export_spec_v1.md:
    - null fields omitted
    - enemy/ally/engagement omitted if entire object is None
    - history always present (empty {} if all null)
    """
    result: dict[str, Any] = {}

    if include_evidence and evidence:
        # Review format: obs + evidence
        obs_clean: dict[str, Any] = {}
    else:
        obs_clean = result

    # ego
    ego = obs.get("ego")
    if ego:
        obs_clean["ego"] = _strip_nulls(ego)

    # enemy
    enemy = obs.get("enemy")
    if enemy is not None:
        obs_clean["enemy"] = _strip_nulls(enemy)

    # ally
    ally = obs.get("ally")
    if ally is not None:
        obs_clean["ally"] = _strip_nulls(ally)

    # engagement
    engagement = obs.get("engagement")
    if engagement is not None:
        obs_clean["engagement"] = _strip_nulls(engagement)

    # history - always present
    history = obs.get("history", {})
    obs_clean["history"] = _strip_nulls(history) if history else {}

    if include_evidence and evidence:
        result["obs"] = obs_clean
        ev_clean = _strip_nulls(evidence)
        if ev_clean:
            result["evidence"] = ev_clean
    else:
        result = obs_clean

    return result


def _build_sft_output(label: dict, include_rationale: bool = False) -> dict | None:
    """Build SFT output from label, stripping null fields.

    Returns None if semantic_state is null (exclude from SFT).
    """
    if label.get("semantic_state") is None:
        return None

    out: dict[str, Any] = {"semantic_state": label["semantic_state"]}

    if label.get("target_ref") is not None:
        out["target_ref"] = label["target_ref"]
    if label.get("role") is not None:
        out["role"] = label["role"]

    constraints = label.get("constraints", {})
    constraints_clean = _strip_nulls(constraints) if constraints else {}
    if constraints_clean:
        out["constraints"] = constraints_clean

    if label.get("profile_hint") is not None:
        out["profile_hint"] = label["profile_hint"]

    event_flags = label.get("event_flags", {})
    ef_clean = _strip_nulls(event_flags) if event_flags else {}
    if ef_clean:
        out["event_flags"] = ef_clean

    if include_rationale and label.get("rationale_short") is not None:
        out["rationale_short"] = label["rationale_short"]

    return out


def export_sft_tiered(
    merged_path: pathlib.Path,
    gold_output: pathlib.Path = pathlib.Path("datasets/d2b/sft_standard_gold.jsonl"),
    guarded_silver_output: pathlib.Path = pathlib.Path("datasets/d2b/sft_guarded_silver.jsonl"),
) -> dict[str, Any]:
    """Export tiered SFT datasets: gold-only and guarded-silver.

    guarded_silver criteria:
    - source.type == "mixed" (rule + GPT adjudication)
    - evidence contains GUARD marker
    - label.semantic_state is not null
    - quality.tier in {silver, bronze} (guard-protected records)
    """
    gold_output.parent.mkdir(parents=True, exist_ok=True)
    guarded_silver_output.parent.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(merged_path)

    stats: dict[str, Any] = {
        "total_input": len(records),
        "gold_exported": 0,
        "guarded_silver_exported": 0,
        "excluded_weak": 0,
        "excluded_null_state": 0,
        "gold_states": {},
        "guarded_silver_states": {},
    }

    with open(gold_output, "w", encoding="utf-8") as f_gold, \
         open(guarded_silver_output, "w", encoding="utf-8") as f_gs:

        for rec in records:
            quality = rec.get("quality", {})
            tier = quality.get("tier", "weak")
            label = rec.get("label", {})
            obs = rec.get("obs", {})
            source_type = rec.get("source", {}).get("type", "")
            evidence = rec.get("evidence", {})
            notes = evidence.get("notes", "") or ""
            state = label.get("semantic_state")

            if tier == "weak":
                stats["excluded_weak"] += 1
                continue
            if state is None:
                stats["excluded_null_state"] += 1
                continue

            sft_output = _build_sft_output(label, include_rationale=False)
            if sft_output is None:
                continue
            sft_input = _build_sft_input(obs)

            conflict = quality.get("conflict_flag", False)

            # Gold: non-conflict, silver or above
            if not conflict and tier in ("gold", "silver"):
                sample: dict[str, Any] = {
                    "instruction": _STANDARD_INSTRUCTION,
                    "input": sft_input,
                    "output": sft_output,
                }
                f_gold.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["gold_exported"] += 1
                stats["gold_states"][state] = stats["gold_states"].get(state, 0) + 1

            # Guarded silver: guard-protected rule labels
            elif (source_type == "mixed"
                  and "GUARD" in notes
                  and tier in ("silver", "bronze")):
                conf = quality.get("confidence")
                sample = {
                    "instruction": _STANDARD_INSTRUCTION,
                    "input": sft_input,
                    "output": sft_output,
                    "_meta": {
                        "tier": "guarded_silver",
                        "original_tier": tier,
                        "confidence": conf,
                        "guard_route": "rule_protected",
                    },
                }
                f_gs.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["guarded_silver_exported"] += 1
                stats["guarded_silver_states"][state] = (
                    stats["guarded_silver_states"].get(state, 0) + 1
                )

    logger.info(
        "Tiered SFT export: gold=%d, guarded_silver=%d",
        stats["gold_exported"], stats["guarded_silver_exported"],
    )
    return stats


def export_sft(
    merged_path: pathlib.Path,
    standard_output: pathlib.Path = pathlib.Path("datasets/d2b/sft_standard.jsonl"),
    review_output: pathlib.Path = pathlib.Path("datasets/d2b/sft_review.jsonl"),
    allowed_tiers: set[str] = frozenset({"gold", "silver", "bronze"}),
    bronze_meta: bool = True,
) -> dict[str, Any]:
    """Export SFT datasets from merged canonical records.

    Rules (per sft_export_spec_v1.md):
    - Standard SFT: no evidence/quality, exclude conflict_flag=True
    - Review SFT: with evidence, include conflict_flag=True with [CONFLICT] note
    - weak tier: always excluded
    - semantic_state=null: always excluded
    """
    standard_output.parent.mkdir(parents=True, exist_ok=True)
    review_output.parent.mkdir(parents=True, exist_ok=True)

    records = _load_jsonl(merged_path)

    stats: dict[str, Any] = {
        "total_input": len(records),
        "standard_exported": 0,
        "review_exported": 0,
        "excluded_weak": 0,
        "excluded_null_state": 0,
        "excluded_conflict_standard": 0,
    }

    with open(standard_output, "w", encoding="utf-8") as f_std, \
         open(review_output, "w", encoding="utf-8") as f_rev:

        for rec in records:
            quality = rec.get("quality", {})
            tier = quality.get("tier", "weak")
            conflict = quality.get("conflict_flag", False)
            label = rec.get("label", {})
            obs = rec.get("obs", {})
            evidence = rec.get("evidence", {})

            # Skip weak tier
            if tier not in allowed_tiers:
                stats["excluded_weak"] += 1
                continue

            # Skip null semantic_state
            sft_output = _build_sft_output(label, include_rationale=False)
            if sft_output is None:
                stats["excluded_null_state"] += 1
                continue

            # Standard SFT (no conflict)
            if not conflict:
                sft_input = _build_sft_input(obs)
                sample: dict[str, Any] = {
                    "instruction": _STANDARD_INSTRUCTION,
                    "input": sft_input,
                    "output": sft_output,
                }
                if bronze_meta and tier == "bronze":
                    sample["_meta"] = {
                        "tier": tier,
                        "confidence": quality.get("confidence"),
                    }
                f_std.write(json.dumps(sample, ensure_ascii=False) + "\n")
                stats["standard_exported"] += 1
            else:
                stats["excluded_conflict_standard"] += 1

            # Review SFT (always, including conflict)
            review_evidence = dict(evidence)
            if conflict:
                notes = review_evidence.get("notes") or ""
                if "[CONFLICT]" not in notes:
                    review_evidence["notes"] = (notes + " [CONFLICT]").strip()

            rev_output = _build_sft_output(label, include_rationale=True)
            if rev_output is not None:
                rev_input = _build_sft_input(obs, include_evidence=True,
                                             evidence=review_evidence)
                rev_sample: dict[str, Any] = {
                    "instruction": _REVIEW_INSTRUCTION,
                    "input": rev_input,
                    "output": rev_output,
                }
                if bronze_meta and tier == "bronze":
                    rev_sample["_meta"] = {
                        "tier": tier,
                        "confidence": quality.get("confidence"),
                    }
                f_rev.write(json.dumps(rev_sample, ensure_ascii=False) + "\n")
                stats["review_exported"] += 1

    logger.info(
        "SFT export: standard=%d, review=%d",
        stats["standard_exported"], stats["review_exported"],
    )
    return stats


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and export SFT datasets")
    parser.add_argument("--rule", type=str, required=True)
    parser.add_argument("--gpt", type=str, required=True)
    parser.add_argument("--preference", type=str, default=None)
    parser.add_argument("--output", type=str, default="datasets/d2b/merged_canonical.jsonl")
    parser.add_argument("--sft-standard", type=str, default="datasets/d2b/sft_standard.jsonl")
    parser.add_argument("--sft-review", type=str, default="datasets/d2b/sft_review.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    merged_path = pathlib.Path(args.output)
    _, merge_stats = merge_sources(
        pathlib.Path(args.rule),
        pathlib.Path(args.gpt),
        pathlib.Path(args.preference) if args.preference else None,
        merged_path,
    )
    print("Merge stats:", json.dumps(merge_stats, indent=2))

    sft_stats = export_sft(
        merged_path,
        pathlib.Path(args.sft_standard),
        pathlib.Path(args.sft_review),
    )
    print("SFT stats:", json.dumps(sft_stats, indent=2))


if __name__ == "__main__":
    main()
