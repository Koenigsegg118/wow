"""Tests for guarded silver export and commit/press boundary A/B."""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest

from tools.d2b.merge_and_export import export_sft_tiered
from tools.d2b.run_boundary_ab_test import (
    _classify_doctrine_override_v1,
    _classify_guideline_strict,
    run_ab_test,
)


# ── Helpers ───────────────────────────────────────────────────

def _make_canonical(state, source_type="acmi_ai", tier="silver",
                    conflict=False, notes="", conf=0.85,
                    enemy_range=5000, enemy_bearing=0, enemy_closure=-300,
                    enemy_aspect=5):
    return {
        "schema_version": "semantic_record_v1",
        "sample_id": f"test_{state}_{enemy_range}",
        "source": {"type": source_type, "scenario_id": "test"},
        "obs": {
            "ego": {"runtime_id": "red_1", "speed_mps": 300, "altitude_m": 8000,
                    "heading_deg": 0},
            "enemy": {"target_ref": "enemy_1", "range_m": enemy_range,
                      "bearing_deg": enemy_bearing, "closure_mps": enemy_closure,
                      "aspect_deg": enemy_aspect},
            "engagement": {"is_merge": enemy_range < 3000,
                           "is_defensive": False},
            "history": {},
        },
        "label": {"semantic_state": state, "target_ref": "enemy_1",
                  "role": "engaged", "profile_hint": "p6dof_semantic",
                  "constraints": {}, "event_flags": {},
                  "rationale_short": "test"},
        "evidence": {"rule_ids": [], "notes": notes},
        "quality": {"tier": tier, "confidence": conf,
                    "conflict_flag": conflict, "needs_review": conflict},
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ═══════════════════════════════════════════════════════════════
# Guarded Silver Export Tests
# ═══════════════════════════════════════════════════════════════

class TestGuardedSilverExport:

    def test_gold_non_conflict_silver(self, tmp_path):
        """Non-conflict silver records go to gold."""
        recs = [_make_canonical("merge_entry", tier="silver", conflict=False)]
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, recs)

        stats = export_sft_tiered(merged,
                                  tmp_path / "gold.jsonl",
                                  tmp_path / "gs.jsonl")
        assert stats["gold_exported"] == 1
        assert stats["guarded_silver_exported"] == 0

    def test_guarded_silver_mixed_with_guard(self, tmp_path):
        """Guard-protected mixed records go to guarded_silver."""
        recs = [_make_canonical(
            "hold_geometry", source_type="mixed", tier="bronze",
            conflict=True, notes="[GUARD:standoff_guard] GPT=merge_entry",
        )]
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, recs)

        stats = export_sft_tiered(merged,
                                  tmp_path / "gold.jsonl",
                                  tmp_path / "gs.jsonl")
        assert stats["gold_exported"] == 0
        assert stats["guarded_silver_exported"] == 1

        gs = _read_jsonl(tmp_path / "gs.jsonl")
        assert gs[0]["output"]["semantic_state"] == "hold_geometry"
        assert gs[0]["_meta"]["tier"] == "guarded_silver"
        assert gs[0]["_meta"]["guard_route"] == "rule_protected"

    def test_weak_excluded(self, tmp_path):
        """Weak tier records are excluded from both exports."""
        recs = [_make_canonical("press_attack", tier="weak")]
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, recs)

        stats = export_sft_tiered(merged,
                                  tmp_path / "gold.jsonl",
                                  tmp_path / "gs.jsonl")
        assert stats["gold_exported"] == 0
        assert stats["guarded_silver_exported"] == 0
        assert stats["excluded_weak"] == 1

    def test_null_state_excluded(self, tmp_path):
        """Null semantic_state records are excluded."""
        rec = _make_canonical("hold_geometry")
        rec["label"]["semantic_state"] = None
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, [rec])

        stats = export_sft_tiered(merged,
                                  tmp_path / "gold.jsonl",
                                  tmp_path / "gs.jsonl")
        assert stats["gold_exported"] == 0
        assert stats["guarded_silver_exported"] == 0

    def test_conflict_non_guard_excluded_from_both(self, tmp_path):
        """Conflict records without GUARD marker go to neither gold nor guarded_silver."""
        recs = [_make_canonical("press_attack", tier="bronze",
                                conflict=True, notes="GPT conflict")]
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, recs)

        stats = export_sft_tiered(merged,
                                  tmp_path / "gold.jsonl",
                                  tmp_path / "gs.jsonl")
        assert stats["gold_exported"] == 0
        assert stats["guarded_silver_exported"] == 0

    def test_combined_count(self, tmp_path):
        """Gold + guarded_silver combined correctly."""
        recs = [
            _make_canonical("merge_entry", tier="silver", conflict=False),
            _make_canonical("hold_geometry", source_type="mixed", tier="bronze",
                            conflict=True, notes="[GUARD:standoff_guard]"),
            _make_canonical("press_attack", tier="weak"),
        ]
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, recs)

        stats = export_sft_tiered(merged,
                                  tmp_path / "gold.jsonl",
                                  tmp_path / "gs.jsonl")
        assert stats["gold_exported"] == 1
        assert stats["guarded_silver_exported"] == 1
        assert stats["excluded_weak"] == 1


# ═══════════════════════════════════════════════════════════════
# Boundary A/B Tests
# ═══════════════════════════════════════════════════════════════

class TestBoundaryModes:

    def test_guideline_strict_no_change(self):
        """guideline_strict never overrides."""
        obs = {"enemy": {"range_m": 10000, "bearing_deg": 0, "closure_mps": -600},
               "engagement": {"is_merge": False, "is_defensive": False}}
        result = _classify_guideline_strict(obs, "press_attack")
        assert result is None

    def test_doctrine_override_far_range(self):
        """doctrine_override flips press->commit at >8km, small bearing, strong closure."""
        obs = {"enemy": {"range_m": 10000, "bearing_deg": 5, "closure_mps": -600},
               "engagement": {"is_merge": False, "is_defensive": False}}
        result = _classify_doctrine_override_v1(obs, "press_attack")
        assert result == "commit_intercept"

    def test_doctrine_override_close_range_no_flip(self):
        """doctrine_override does NOT flip at <8km."""
        obs = {"enemy": {"range_m": 6000, "bearing_deg": 5, "closure_mps": -400},
               "engagement": {"is_merge": False, "is_defensive": False}}
        result = _classify_doctrine_override_v1(obs, "press_attack")
        assert result is None

    def test_doctrine_override_wide_bearing_no_flip(self):
        """doctrine_override does NOT flip if bearing > 30."""
        obs = {"enemy": {"range_m": 10000, "bearing_deg": 45, "closure_mps": -600},
               "engagement": {"is_merge": False, "is_defensive": False}}
        result = _classify_doctrine_override_v1(obs, "press_attack")
        assert result is None

    def test_doctrine_override_weak_closure_no_flip(self):
        """doctrine_override does NOT flip if closure > -100."""
        obs = {"enemy": {"range_m": 10000, "bearing_deg": 5, "closure_mps": -50},
               "engagement": {"is_merge": False, "is_defensive": False}}
        result = _classify_doctrine_override_v1(obs, "press_attack")
        assert result is None

    def test_doctrine_override_defensive_no_flip(self):
        """doctrine_override does NOT flip if is_defensive."""
        obs = {"enemy": {"range_m": 10000, "bearing_deg": 5, "closure_mps": -600},
               "engagement": {"is_merge": False, "is_defensive": True}}
        result = _classify_doctrine_override_v1(obs, "press_attack")
        assert result is None

    def test_doctrine_override_not_press_no_flip(self):
        """doctrine_override only applies to press_attack."""
        obs = {"enemy": {"range_m": 10000, "bearing_deg": 5, "closure_mps": -600},
               "engagement": {"is_merge": False, "is_defensive": False}}
        result = _classify_doctrine_override_v1(obs, "merge_entry")
        assert result is None

    def test_run_ab_test_structure(self, tmp_path):
        """run_ab_test returns correct structure."""
        recs = [
            _make_canonical("press_attack", enemy_range=10000, enemy_bearing=2,
                            enemy_closure=-580, enemy_aspect=5),
            _make_canonical("press_attack", enemy_range=6000, enemy_bearing=5,
                            enemy_closure=-400, enemy_aspect=8),
        ]
        merged = tmp_path / "merged.jsonl"
        _write_jsonl(merged, recs)

        results = run_ab_test(merged)
        assert results["dispute_candidates"] == 2
        assert "guideline_strict" in results["modes"]
        assert "doctrine_override_v1" in results["modes"]
        assert results["modes"]["guideline_strict"]["flipped"] == 0
        # Only the >8km one should flip under doctrine
        assert results["modes"]["doctrine_override_v1"]["flipped"] == 1
