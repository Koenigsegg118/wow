from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from semantic_record_common import (
    append_note,
    basename_without_suffix,
    build_empty_evidence,
    build_empty_extras,
    build_empty_label,
    build_empty_obs,
    build_quality,
    build_source,
    canonicalize_extras,
    flatten_tags,
    infer_profile_hint,
    infer_role,
    infer_should_abort_if_threatened,
    infer_target_ref,
    make_record,
    map_legacy_semantic_state,
    parse_confidence,
    read_jsonl,
    read_yaml,
    sanitize_sample_id,
    sanitize_string_list,
    truncate_text,
    validate_records,
    write_jsonl,
)


def load_audit_rules(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    records = read_jsonl(path)
    return {record["rule_id"]: record for record in records}


def load_mapping_rows(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {row["rule_id"]: row for row in csv.DictReader(handle)}


def _label_from_legacy(
    *,
    rule_id: str,
    legacy_semantic_state: str | None,
    legacy_profile_hint: str | None,
    target_policy: str | None,
    legacy_constraints: dict[str, Any] | None,
    rule_scope: str | None,
    pair_type: str | None = None,
) -> dict[str, Any]:
    semantic_state = map_legacy_semantic_state(legacy_semantic_state, rule_id=rule_id, pair_type=pair_type)
    label = build_empty_label()
    label["semantic_state"] = semantic_state

    must_support_teammate = None
    allow_aggressive_turn = None
    prefer_energy_preserve = None
    if isinstance(legacy_constraints, dict):
        must_support_teammate = legacy_constraints.get("must_support_teammate")
        allow_aggressive_turn = legacy_constraints.get("allow_aggressive_turn")
        prefer_energy_preserve = legacy_constraints.get("prefer_energy_preserve")

    label["target_ref"] = infer_target_ref(
        rule_id=rule_id,
        semantic_state=semantic_state,
        target_policy=target_policy,
        must_support_teammate=must_support_teammate,
    )
    label["role"] = infer_role(semantic_state, must_support_teammate=must_support_teammate, rule_id=rule_id)
    label["constraints"]["prefer_energy_preserve"] = prefer_energy_preserve
    label["constraints"]["allow_aggressive_turn"] = allow_aggressive_turn
    label["constraints"]["must_support_teammate"] = must_support_teammate
    label["constraints"]["should_abort_if_threatened"] = infer_should_abort_if_threatened(semantic_state, rule_scope)
    label["profile_hint"] = infer_profile_hint(legacy_profile_hint, semantic_state)
    label["event_flags"]["target_switch"] = None
    label["rationale_short"] = truncate_text(
        append_note(
            None,
            f"{rule_id} mapped {legacy_semantic_state or 'null'} -> {semantic_state or 'null'}",
            target_policy,
        )
    )
    return label


def _obs_from_afsim(
    *,
    state_summary: dict[str, Any] | None,
    label: dict[str, Any],
    rule_scope: str | None,
) -> dict[str, Any]:
    obs = build_empty_obs()
    target_ref = label.get("target_ref")
    must_support_teammate = label.get("constraints", {}).get("must_support_teammate")

    if target_ref in {"enemy_1", "enemy_2"}:
        obs["enemy"] = {
            "target_ref": target_ref,
            "target_runtime_id": None,
            "callsign": None,
            "range_m": None,
            "bearing_deg": None,
            "aspect_deg": None,
            "closure_mps": None,
            "alt_diff_m": None,
            "is_primary_threat": None,
        }

    if target_ref == "ally_1" or must_support_teammate:
        obs["ally"] = {
            "ally_ref": "ally_1",
            "ally_runtime_id": None,
            "callsign": None,
            "range_m": None,
            "bearing_deg": None,
            "is_supporting": True if must_support_teammate else None,
        }

    obs["engagement"]["is_merge"] = label.get("semantic_state") == "merge_entry"
    obs["engagement"]["is_defensive"] = label.get("semantic_state") == "defensive_break"
    obs["engagement"]["has_shot_opportunity"] = None

    if isinstance(state_summary, dict):
        ego_section = state_summary.get("ego")
        if isinstance(ego_section, dict):
            coalition = ego_section.get("coalition")
            if coalition in {"red", "blue", "neutral"}:
                obs["ego"]["coalition"] = coalition
        geometry_tags = state_summary.get("geometry")
        if isinstance(geometry_tags, dict) and rule_scope == "disengage":
            obs["engagement"]["is_defensive"] = True if "abort" in json.dumps(geometry_tags).lower() else obs["engagement"]["is_defensive"]

    return obs


def _evidence_from_afsim(
    *,
    record_id: str,
    rule_id: str,
    target_policy: str | None,
    legacy_profile_hint: str | None,
    legacy_constraints: dict[str, Any] | None,
    state_summary: dict[str, Any] | None,
    audit_rule: dict[str, Any] | None,
    pair_reason: str | None = None,
    extra_note: str | None = None,
) -> dict[str, Any]:
    evidence = build_empty_evidence()
    evidence["rule_ids"] = [rule_id]
    evidence["frame_refs"] = [record_id]
    if audit_rule and audit_rule.get("source_file"):
        evidence["frame_refs"].append(basename_without_suffix(audit_rule.get("source_file")))
    evidence["notes"] = append_note(
        None,
        target_policy,
        f"legacy_profile_hint={legacy_profile_hint}" if legacy_profile_hint else None,
        f"decision={audit_rule.get('decision_output_raw')}" if audit_rule else None,
        f"cooldown={audit_rule.get('cooldown_or_hysteresis')}" if audit_rule and audit_rule.get("cooldown_or_hysteresis") else None,
        f"p6dof_risk={audit_rule.get('p6dof_risk_reason')}" if audit_rule and audit_rule.get("p6dof_risk_flag") else None,
        pair_reason,
        extra_note,
    )

    tags = []
    if legacy_constraints:
        if legacy_constraints.get("must_not_descend") is not None:
            tags.append(f"must_not_descend={legacy_constraints.get('must_not_descend')}")
        if legacy_constraints.get("fire_gate") is not None:
            tags.append(f"fire_gate={legacy_constraints.get('fire_gate')}")
        if legacy_constraints.get("reject_reason"):
            tags.append(f"reject_reason={legacy_constraints.get('reject_reason')}")
    if state_summary:
        tags.extend(flatten_tags(state_summary))
    if tags:
        evidence["notes"] = append_note(evidence["notes"], "; ".join(tags))
    return evidence


def _extras_from_afsim(
    *,
    dataset_tag: str,
    record: dict[str, Any],
    audit_rule: dict[str, Any] | None,
    pair_type: str | None = None,
    pair_role: str | None = None,
    template_id: str | None = None,
) -> dict[str, Any] | None:
    extras = build_empty_extras()
    raw_tags = [f"dataset:{dataset_tag}"]
    if record.get("semantic_state") is not None:
        raw_tags.append(f"legacy_semantic_state:{record['semantic_state']}")
    if record.get("profile_hint") is not None:
        raw_tags.append(f"legacy_profile_hint:{record['profile_hint']}")
    if record.get("quality_tag") is not None:
        raw_tags.append(f"legacy_quality_tag:{record['quality_tag']}")
    if pair_type:
        raw_tags.append(f"pair_type:{pair_type}")
    if pair_role:
        raw_tags.append(f"pair_role:{pair_role}")
    if template_id:
        raw_tags.append(f"template_id:{template_id}")
    if audit_rule and audit_rule.get("rule_scope"):
        raw_tags.append(f"rule_scope:{audit_rule.get('rule_scope')}")
    extras["raw_tags"] = sanitize_string_list(raw_tags)
    return canonicalize_extras(extras)


def _quality_from_afsim(
    *,
    confidence: float | None,
    conflict_flag: bool,
    needs_review: bool,
) -> dict[str, Any]:
    return build_quality(
        source_type="afsim_rule",
        confidence=confidence,
        conflict_flag=conflict_flag,
        needs_review=needs_review,
    )


def adapt_afsim_sources(
    *,
    sft_records: list[dict[str, Any]],
    preference_records: list[dict[str, Any]],
    hardcase_templates: list[dict[str, Any]],
    audit_rules: dict[str, dict[str, Any]] | None = None,
    mapping_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    audit_rules = audit_rules or {}
    mapping_rows = mapping_rows or {}
    outputs: list[dict[str, Any]] = []

    for record in sft_records:
        rule_id = record["source_rule_id"]
        audit_rule = audit_rules.get(rule_id, {})
        label = _label_from_legacy(
            rule_id=rule_id,
            legacy_semantic_state=record.get("semantic_state"),
            legacy_profile_hint=record.get("profile_hint"),
            target_policy=record.get("target_policy") or audit_rule.get("suggested_target_policy"),
            legacy_constraints=record.get("constraints"),
            rule_scope=audit_rule.get("rule_scope"),
        )
        outputs.append(
            make_record(
                sample_id=sanitize_sample_id(f"afsim_rule_{record['id']}"),
                source=build_source(
                    platform="mixed",
                    source_type="afsim_rule",
                    scenario_id=basename_without_suffix(audit_rule.get("source_file")),
                    episode_id=record.get("split"),
                    timestamp_sec=None,
                    annotator="afsim_rule_adapter",
                    tool="afsim_rule_to_semantic_record.py",
                ),
                obs=_obs_from_afsim(state_summary=record.get("state_summary"), label=label, rule_scope=audit_rule.get("rule_scope")),
                label=label,
                evidence=_evidence_from_afsim(
                    record_id=record["id"],
                    rule_id=rule_id,
                    target_policy=record.get("target_policy") or audit_rule.get("suggested_target_policy"),
                    legacy_profile_hint=record.get("profile_hint"),
                    legacy_constraints=record.get("constraints"),
                    state_summary=record.get("state_summary"),
                    audit_rule=audit_rule,
                    extra_note=f"split={record.get('split')}",
                ),
                quality=_quality_from_afsim(
                    confidence=parse_confidence(mapping_rows.get(rule_id, {}).get("confidence")),
                    conflict_flag=False,
                    needs_review=False,
                ),
                extras=_extras_from_afsim(dataset_tag="safe_core", record=record, audit_rule=audit_rule),
            )
        )

    for record in preference_records:
        pair_type = record.get("pair_type")
        for pair_role in ("chosen", "rejected"):
            pair_record = record[pair_role]
            rule_id = pair_record["rule_id"]
            audit_rule = audit_rules.get(rule_id, {})
            label = _label_from_legacy(
                rule_id=rule_id,
                legacy_semantic_state=pair_record.get("semantic_state"),
                legacy_profile_hint=pair_record.get("profile_hint"),
                target_policy=pair_record.get("target_policy") or audit_rule.get("suggested_target_policy"),
                legacy_constraints=pair_record.get("constraints"),
                rule_scope=audit_rule.get("rule_scope"),
                pair_type=pair_type,
            )
            outputs.append(
                make_record(
                    sample_id=sanitize_sample_id(f"afsim_rule_{record['id']}_{pair_role}"),
                    source=build_source(
                        platform="mixed",
                        source_type="afsim_rule",
                        scenario_id=basename_without_suffix(audit_rule.get("source_file")),
                        episode_id=pair_type,
                        timestamp_sec=None,
                        annotator="afsim_rule_adapter",
                        tool="afsim_rule_to_semantic_record.py",
                    ),
                    obs=_obs_from_afsim(state_summary=record.get("state_summary"), label=label, rule_scope=audit_rule.get("rule_scope")),
                    label=label,
                    evidence=_evidence_from_afsim(
                        record_id=record["id"],
                        rule_id=rule_id,
                        target_policy=pair_record.get("target_policy") or audit_rule.get("suggested_target_policy"),
                        legacy_profile_hint=pair_record.get("profile_hint"),
                        legacy_constraints=pair_record.get("constraints"),
                        state_summary=record.get("state_summary"),
                        audit_rule=audit_rule,
                        pair_reason=record.get("reason"),
                        extra_note=f"pair_type={pair_type}",
                    ),
                    quality=_quality_from_afsim(
                        confidence=parse_confidence(mapping_rows.get(rule_id, {}).get("confidence")) or (0.7 if pair_role == "chosen" else 0.45),
                        conflict_flag=True,
                        needs_review=True,
                    ),
                    extras=_extras_from_afsim(
                        dataset_tag="preference",
                        record=pair_record,
                        audit_rule=audit_rule,
                        pair_type=pair_type,
                        pair_role=pair_role,
                    ),
                )
            )

    for template in hardcase_templates:
        expected_rule_ids = sanitize_string_list(template.get("expected_rule_ids"))
        lead_rule_id = expected_rule_ids[0] if expected_rule_ids else "hard_case"
        label = _label_from_legacy(
            rule_id=lead_rule_id,
            legacy_semantic_state=template.get("semantic_focus"),
            legacy_profile_hint=None,
            target_policy=None,
            legacy_constraints=None,
            rule_scope="fallback",
        )
        evidence = build_empty_evidence()
        evidence["rule_ids"] = expected_rule_ids
        evidence["frame_refs"] = [template["template_id"]]
        evidence["notes"] = append_note(
            None,
            f"required_fields={','.join(sanitize_string_list(template.get('required_fields')))}",
            f"perturb_knobs={','.join(sanitize_string_list(template.get('perturb_knobs')))}",
            f"metric={template.get('suggested_eval_metric')}",
            f"use={template.get('suggested_use')}",
        )
        outputs.append(
            make_record(
                sample_id=sanitize_sample_id(f"afsim_rule_{template['template_id']}"),
                source=build_source(
                    platform="manual",
                    source_type="afsim_rule",
                    scenario_id=template["template_id"],
                    episode_id="hard_case",
                    timestamp_sec=None,
                    annotator="afsim_rule_adapter",
                    tool="afsim_rule_to_semantic_record.py",
                ),
                obs=_obs_from_afsim(state_summary=None, label=label, rule_scope="fallback"),
                label=label,
                evidence=evidence,
                quality=_quality_from_afsim(confidence=None, conflict_flag=True, needs_review=True),
                extras=_extras_from_afsim(
                    dataset_tag="hard_case",
                    record={"semantic_state": template.get("semantic_focus")},
                    audit_rule=None,
                    template_id=template["template_id"],
                ),
            )
        )

    return outputs


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert WOW/AFSIM semantic audit outputs into semantic_record_v1.")
    parser.add_argument("--sft", required=True, help="wow_semantic_sft.jsonl path")
    parser.add_argument("--preference", required=True, help="wow_semantic_preference.jsonl path")
    parser.add_argument("--hardcase", required=True, help="wow_hardcase_adversary_templates.yaml path")
    parser.add_argument("--audit-jsonl", help="Optional afsim_rules.jsonl path")
    parser.add_argument("--mapping-csv", help="Optional semantic_mapping_candidates.csv path")
    parser.add_argument("--output", required=True, help="Output semantic_record_v1 JSONL path")
    parser.add_argument("--validate", action="store_true", help="Validate converted records before writing")
    return parser


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()

    records = adapt_afsim_sources(
        sft_records=read_jsonl(args.sft),
        preference_records=read_jsonl(args.preference),
        hardcase_templates=read_yaml(args.hardcase) or [],
        audit_rules=load_audit_rules(args.audit_jsonl),
        mapping_rows=load_mapping_rows(args.mapping_csv),
    )

    if args.validate:
        errors = validate_records(records)
        if errors:
            for error in errors:
                print(error)
            return 1

    write_jsonl(args.output, records)
    print(f"[afsim_rule_to_semantic_record] wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
