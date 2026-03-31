import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from acmi_ai_to_semantic_record import adapt_acmi_ai_records
from afsim_rule_to_semantic_record import adapt_afsim_sources
from semantic_record_validator import validate_records, validate_standard_sft_samples
from semantic_sft_export import export_sft_samples


def test_afsim_safe_core_record_adapts_into_canonical_schema():
    sft_records = [
        {
            "id": "sft_001",
            "source_rule_id": "AIR001",
            "semantic_state": "OBFM_WEZ_commit",
            "target_policy": "keep current high-priority threat",
            "profile_hint": "turn_hot_from_route_orbit",
            "constraints": {
                "prefer_energy_preserve": False,
                "allow_aggressive_turn": True,
                "must_not_descend": None,
                "must_support_teammate": False,
                "fire_gate": None,
            },
            "state_summary": {
                "ego": {"rule_type": "route", "mission_type": "DCA"},
                "enemy": {"nearest_air_range_m": "1.18*COMMIT_RNG"},
                "ally": {"protected_entity_threatened": False},
                "weapons": {"flight_guiding": False},
                "geometry": {"rng_cls_trk_relation": "below_1.2_commit"},
            },
            "split": "train",
            "quality_tag": "safe_core",
        }
    ]

    records = adapt_afsim_sources(
        sft_records=sft_records,
        preference_records=[],
        hardcase_templates=[],
        audit_rules={
            "AIR001": {
                "rule_scope": "maneuver_selection",
                "source_file": "demo/rule.txt",
                "suggested_target_policy": "keep current high-priority threat",
                "decision_output_raw": "change_rultype(..., 'ftr')",
                "cooldown_or_hysteresis": "recovery back to route needs >1.75*COMMIT_RNG",
                "p6dof_risk_flag": False,
                "p6dof_risk_reason": "",
            }
        },
        mapping_rows={"AIR001": {"confidence": "0.82"}},
    )

    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == "semantic_record_v1"
    assert record["source"]["type"] == "afsim_rule"
    assert record["label"]["semantic_state"] == "commit_intercept"
    assert record["label"]["target_ref"] == "enemy_1"
    assert record["quality"]["tier"] == "bronze"
    assert "source_rule_id" not in record

    errors = validate_records(records)
    assert errors == []


def test_acmi_ai_adapter_normalizes_unknown_to_null_and_derives_quality():
    raw_records = [
        {
            "sample_id": "acmi_ai_demo_001",
            "source": {
                "scenario_id": "demo.acmi",
                "episode_id": "ep01",
                "timestamp_sec": 80.0,
                "annotator": "qwen3-4b",
                "tool": "acmi_auto_label_v1.py",
            },
            "obs": {
                "ego": {"speed_mps": 290.0, "altitude_m": 9000.0},
                "enemy": None,
                "ally": None,
                "engagement": {"is_merge": False, "is_defensive": False},
                "history": {"prev_semantic_state": None, "prev_token_family": None, "prev_target_ref": None},
            },
            "label": {
                "semantic_state": None,
                "target_ref": "unknown",
                "role": "neutral",
                "constraints": {
                    "prefer_energy_preserve": True,
                    "allow_aggressive_turn": None,
                    "must_support_teammate": None,
                    "should_abort_if_threatened": None,
                },
                "profile_hint": "auto",
                "event_flags": {"target_switch": None},
                "rationale_short": "obs insufficient",
            },
            "quality_hint": {"confidence": 0.2, "conflict_flag": False, "needs_review": False},
        }
    ]

    records = adapt_acmi_ai_records(raw_records)
    assert len(records) == 1
    record = records[0]
    assert record["label"]["target_ref"] is None
    assert record["quality"]["tier"] == "weak"

    errors = validate_records(records)
    assert errors == []


def test_sft_export_routes_conflicts_only_to_review():
    canonical_records = [
        {
            "schema_version": "semantic_record_v1",
            "sample_id": "rec_safe",
            "source": {
                "platform": "mixed",
                "type": "afsim_rule",
                "scenario_id": "demo",
                "episode_id": None,
                "timestamp_sec": None,
                "annotator": "adapter",
                "tool": "adapter.py",
            },
            "obs": {
                "ego": {"runtime_id": None, "callsign": None, "coalition": None, "speed_mps": None, "altitude_m": None,
                        "heading_deg": None, "pitch_deg": None, "roll_deg": None, "vertical_speed_mps": None,
                        "pos_east_m": None, "pos_north_m": None, "energy_state": None},
                "enemy": {"target_ref": "enemy_1", "target_runtime_id": None, "callsign": None, "range_m": None,
                          "bearing_deg": None, "aspect_deg": None, "closure_mps": None, "alt_diff_m": None,
                          "is_primary_threat": None},
                "ally": None,
                "engagement": {"is_merge": None, "is_defensive": None, "has_shot_opportunity": None},
                "history": {"prev_semantic_state": None, "prev_token_family": None, "prev_target_ref": None},
            },
            "label": {
                "semantic_state": "commit_intercept",
                "target_ref": "enemy_1",
                "role": "engaged",
                "constraints": {
                    "prefer_energy_preserve": False,
                    "allow_aggressive_turn": True,
                    "must_support_teammate": False,
                    "should_abort_if_threatened": None,
                },
                "profile_hint": "p6dof_semantic",
                "event_flags": {"target_switch": False},
                "rationale_short": "safe",
            },
            "evidence": {"rule_ids": ["AIR001"], "acmi_refs": [], "frame_refs": [], "notes": "safe sample"},
            "quality": {"tier": "bronze", "confidence": 0.82, "needs_review": False, "conflict_flag": False},
            "extras": {"raw_object_ids": None, "raw_tags": ["dataset:safe_core"], "weapon_event_refs": None},
        },
        {
            "schema_version": "semantic_record_v1",
            "sample_id": "rec_conflict",
            "source": {
                "platform": "mixed",
                "type": "afsim_rule",
                "scenario_id": "demo",
                "episode_id": None,
                "timestamp_sec": None,
                "annotator": "adapter",
                "tool": "adapter.py",
            },
            "obs": {
                "ego": {"runtime_id": None, "callsign": None, "coalition": None, "speed_mps": None, "altitude_m": None,
                        "heading_deg": None, "pitch_deg": None, "roll_deg": None, "vertical_speed_mps": None,
                        "pos_east_m": None, "pos_north_m": None, "energy_state": None},
                "enemy": {"target_ref": "enemy_1", "target_runtime_id": None, "callsign": None, "range_m": None,
                          "bearing_deg": None, "aspect_deg": None, "closure_mps": None, "alt_diff_m": None,
                          "is_primary_threat": None},
                "ally": None,
                "engagement": {"is_merge": None, "is_defensive": None, "has_shot_opportunity": None},
                "history": {"prev_semantic_state": None, "prev_token_family": None, "prev_target_ref": None},
            },
            "label": {
                "semantic_state": "commit_intercept",
                "target_ref": "enemy_1",
                "role": "engaged",
                "constraints": {
                    "prefer_energy_preserve": False,
                    "allow_aggressive_turn": True,
                    "must_support_teammate": False,
                    "should_abort_if_threatened": None,
                },
                "profile_hint": "p6dof_semantic",
                "event_flags": {"target_switch": False},
                "rationale_short": "conflict",
            },
            "evidence": {"rule_ids": ["AIR025"], "acmi_refs": [], "frame_refs": [], "notes": "conflict sample"},
            "quality": {"tier": "bronze", "confidence": 0.4, "needs_review": True, "conflict_flag": True},
            "extras": {"raw_object_ids": None, "raw_tags": ["dataset:preference"], "weapon_event_refs": None},
        },
    ]

    standard, standard_map, _ = export_sft_samples(
        canonical_records,
        fmt="standard",
        tiers={"bronze", "silver", "gold"},
        bronze_meta=True,
        validate_input=True,
    )
    review, review_map, _ = export_sft_samples(
        canonical_records,
        fmt="review",
        tiers={"bronze", "silver", "gold"},
        bronze_meta=True,
        validate_input=True,
    )

    assert len(standard) == 1
    assert len(standard_map) == 1
    assert len(review) == 2
    assert any("[CONFLICT]" in (item["input"]["evidence"].get("notes") or "") for item in review)
    assert validate_standard_sft_samples(canonical_records, standard_map) == []


def test_validator_flags_legacy_fields_and_unknown_sentinels():
    bad_record = {
        "schema_version": "semantic_record_v1",
        "sample_id": "bad_001",
        "source": {"platform": "mixed", "type": "afsim_rule"},
        "obs": {"ego": {}},
        "label": {"semantic_state": "commit_intercept"},
        "evidence": {},
        "quality": {"tier": "bronze", "needs_review": False},
        "semantic_state": "OBFM_WEZ_commit",
    }

    errors = validate_records([bad_record])
    joined = "\n".join(errors)
    assert "unknown" not in joined.lower()
    assert "legacy" in joined.lower() or "additional properties" in joined.lower()
