"""Tests for tools.d2b.merge_and_export -- merge dedup and SFT export.

Uses pytest tmp_path fixture for all file I/O.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tools.d2b.merge_and_export import (
    export_sft,
    merge_sources,
)


# ===================================================================
# Helpers
# ===================================================================

def _write_jsonl(path: pathlib.Path, records: list[dict]) -> pathlib.Path:
    """Write a list of dicts as JSONL to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_canonical(
    scenario_id: str = "scn_a",
    timestamp_sec: float = 10.0,
    runtime_id: str = "red_1",
    source_type: str = "afsim_rule",
    semantic_state: str = "commit_intercept",
    tier: str = "bronze",
    confidence: float = 0.85,
    conflict_flag: bool = False,
    sample_id: str | None = None,
    target_ref: str = "enemy_1",
    role: str = "engaged",
) -> dict:
    """Build a canonical record for testing."""
    return {
        "schema_version": "semantic_record_v1",
        "sample_id": sample_id or f"{source_type}_{scenario_id}_{timestamp_sec}",
        "source": {
            "platform": "afsim_acmi",
            "type": source_type,
            "scenario_id": scenario_id,
            "episode_id": "ep01",
            "timestamp_sec": timestamp_sec,
            "annotator": None,
            "tool": "test",
        },
        "obs": {
            "ego": {
                "alt_m": 8000,
                "speed_mps": 300,
                "runtime_id": runtime_id,
            },
            "enemy": {
                "range_m": 20000,
                "bearing_deg": 10,
                "closure_mps": -200,
                "target_ref": target_ref,
            },
            "engagement": {"is_merge": False, "is_defensive": False},
            "history": {},
        },
        "label": {
            "semantic_state": semantic_state,
            "target_ref": target_ref,
            "role": role,
            "constraints": None,
            "profile_hint": "p6dof_semantic",
            "event_flags": {"target_switch": None},
            "rationale_short": "test rationale",
        },
        "evidence": {
            "rule_ids": ["D2B_R01_commit"],
            "notes": "test note",
            "acmi_refs": [],
            "frame_refs": [],
        },
        "quality": {
            "tier": tier,
            "confidence": confidence,
            "needs_review": False,
            "conflict_flag": conflict_flag,
        },
        "extras": {},
    }


# ===================================================================
# Merge dedup
# ===================================================================

class TestMergeDedup:
    def test_gpt_overwrites_rule_on_same_key(self, tmp_path):
        """When rule and gpt have the same dedup key, gpt wins."""
        rule_rec = _make_canonical(
            scenario_id="scn_a", timestamp_sec=10.0, runtime_id="red_1",
            source_type="afsim_rule", semantic_state="hold_geometry",
        )
        gpt_rec = _make_canonical(
            scenario_id="scn_a", timestamp_sec=10.0, runtime_id="red_1",
            source_type="acmi_ai", semantic_state="commit_intercept",
        )

        rule_path = _write_jsonl(tmp_path / "rule.jsonl", [rule_rec])
        gpt_path = _write_jsonl(tmp_path / "gpt.jsonl", [gpt_rec])
        output_path = tmp_path / "merged.jsonl"

        _, stats = merge_sources(rule_path, gpt_path, output_path=output_path)

        merged = _read_jsonl(output_path)
        assert stats["merged_total"] == 1, "Should deduplicate to 1 record"
        assert merged[0]["source"]["type"] == "acmi_ai", "gpt should overwrite rule"
        assert merged[0]["label"]["semantic_state"] == "commit_intercept"

    def test_different_keys_both_kept(self, tmp_path):
        """Records with different dedup keys are both kept."""
        rule_rec = _make_canonical(
            scenario_id="scn_a", timestamp_sec=10.0, runtime_id="red_1",
        )
        gpt_rec = _make_canonical(
            scenario_id="scn_b", timestamp_sec=20.0, runtime_id="red_2",
        )

        rule_path = _write_jsonl(tmp_path / "rule.jsonl", [rule_rec])
        gpt_path = _write_jsonl(tmp_path / "gpt.jsonl", [gpt_rec])
        output_path = tmp_path / "merged.jsonl"

        _, stats = merge_sources(rule_path, gpt_path, output_path=output_path)

        assert stats["merged_total"] == 2

    def test_preference_overwrites_gpt(self, tmp_path):
        """Preference source has highest priority, overwrites gpt."""
        gpt_rec = _make_canonical(
            scenario_id="scn_a", timestamp_sec=10.0, runtime_id="red_1",
            source_type="acmi_ai", semantic_state="commit_intercept",
        )
        pref_rec = _make_canonical(
            scenario_id="scn_a", timestamp_sec=10.0, runtime_id="red_1",
            source_type="mixed", semantic_state="press_attack",
        )

        rule_path = _write_jsonl(tmp_path / "rule.jsonl", [])
        gpt_path = _write_jsonl(tmp_path / "gpt.jsonl", [gpt_rec])
        pref_path = _write_jsonl(tmp_path / "pref.jsonl", [pref_rec])
        output_path = tmp_path / "merged.jsonl"

        _, stats = merge_sources(
            rule_path, gpt_path,
            preference_path=pref_path,
            output_path=output_path,
        )

        merged = _read_jsonl(output_path)
        assert stats["merged_total"] == 1
        assert merged[0]["label"]["semantic_state"] == "press_attack"


# ===================================================================
# SFT export
# ===================================================================

class TestSFTExport:
    def test_weak_tier_excluded(self, tmp_path):
        """Records with tier='weak' are excluded from both SFT outputs."""
        rec = _make_canonical(tier="weak", semantic_state="extend")
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        stats = export_sft(merged_path, std_out, rev_out)

        assert stats["excluded_weak"] == 1
        assert stats["standard_exported"] == 0
        assert stats["review_exported"] == 0
        assert _read_jsonl(std_out) == []
        assert _read_jsonl(rev_out) == []

    def test_null_state_excluded(self, tmp_path):
        """Records with semantic_state=None are excluded from SFT."""
        rec = _make_canonical(semantic_state=None, tier="bronze")
        # semantic_state=None in the label
        rec["label"]["semantic_state"] = None
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        stats = export_sft(merged_path, std_out, rev_out)

        assert stats["excluded_null_state"] == 1
        assert stats["standard_exported"] == 0
        # review also excluded because _build_sft_output returns None
        assert stats["review_exported"] == 0

    def test_conflict_flag_excluded_from_standard(self, tmp_path):
        """conflict_flag=True records are excluded from standard SFT."""
        rec = _make_canonical(
            conflict_flag=True,
            tier="bronze",
            semantic_state="commit_intercept",
        )
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        stats = export_sft(merged_path, std_out, rev_out)

        assert stats["excluded_conflict_standard"] == 1
        assert stats["standard_exported"] == 0
        # But review SFT should include it
        assert stats["review_exported"] == 1
        review_records = _read_jsonl(rev_out)
        assert len(review_records) == 1

    def test_conflict_included_in_review_with_tag(self, tmp_path):
        """conflict_flag=True records get [CONFLICT] in review evidence notes."""
        rec = _make_canonical(
            conflict_flag=True,
            tier="silver",
            semantic_state="extend",
        )
        rec["evidence"]["notes"] = "original note"
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        stats = export_sft(merged_path, std_out, rev_out)

        review_records = _read_jsonl(rev_out)
        assert len(review_records) == 1
        rev_input = review_records[0]["input"]
        evidence = rev_input.get("evidence", {})
        notes = evidence.get("notes", "")
        assert "[CONFLICT]" in notes

    def test_bronze_records_get_meta_tag(self, tmp_path):
        """Bronze-tier records include _meta in SFT output."""
        rec = _make_canonical(tier="bronze", confidence=0.80)
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        stats = export_sft(merged_path, std_out, rev_out)

        std_records = _read_jsonl(std_out)
        assert len(std_records) == 1
        assert "_meta" in std_records[0]
        assert std_records[0]["_meta"]["tier"] == "bronze"
        assert std_records[0]["_meta"]["confidence"] == 0.80

        rev_records = _read_jsonl(rev_out)
        assert len(rev_records) == 1
        assert "_meta" in rev_records[0]
        assert rev_records[0]["_meta"]["tier"] == "bronze"

    def test_silver_records_no_meta_tag(self, tmp_path):
        """Silver-tier records do NOT get _meta (only bronze does)."""
        rec = _make_canonical(tier="silver", confidence=0.90)
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        stats = export_sft(merged_path, std_out, rev_out)

        std_records = _read_jsonl(std_out)
        assert len(std_records) == 1
        assert "_meta" not in std_records[0]

    def test_standard_sft_has_instruction(self, tmp_path):
        """Standard SFT records include the instruction field."""
        rec = _make_canonical(tier="gold")
        merged_path = _write_jsonl(tmp_path / "merged.jsonl", [rec])
        std_out = tmp_path / "sft_std.jsonl"
        rev_out = tmp_path / "sft_rev.jsonl"

        export_sft(merged_path, std_out, rev_out)

        std_records = _read_jsonl(std_out)
        assert len(std_records) == 1
        assert "instruction" in std_records[0]
        assert "input" in std_records[0]
        assert "output" in std_records[0]
        # Output should contain semantic_state
        assert std_records[0]["output"]["semantic_state"] == "commit_intercept"
