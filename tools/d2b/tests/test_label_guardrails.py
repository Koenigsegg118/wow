"""Tests for label_guardrails.py."""

import pytest
from tools.d2b.label_guardrails import (
    reject_merge_entry_for_standoff,
    apply_guardrails,
)


def _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=False):
    return {
        "ego": {"runtime_id": "red_1", "speed_mps": 250},
        "enemy": {
            "target_ref": "enemy_1",
            "range_m": 4500.0,
            "bearing_deg": bearing,
            "aspect_deg": aspect,
            "closure_mps": closure,
            "alt_diff_m": 0.0,
            "is_primary_threat": True,
        },
        "engagement": {"is_merge": is_merge, "is_defensive": False},
        "history": {},
    }


def _gpt_label(state="merge_entry"):
    return {"semantic_state": state, "target_ref": "enemy_1"}


# ── Test 1: standoff geometry -> reject merge_entry ──

class TestStandoffGuard:
    def test_standoff_rejects_merge_entry(self):
        """4.5km / bearing90 / aspect90 / closure0 / is_merge=false -> reject."""
        obs = _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=False)
        result = reject_merge_entry_for_standoff(obs, _gpt_label("merge_entry"))
        assert result.triggered is True

    def test_real_merge_not_rejected(self):
        """Genuine merge geometry should not be rejected."""
        obs = _obs(closure=-400.0, bearing=5.0, aspect=10.0, is_merge=True)
        result = reject_merge_entry_for_standoff(obs, _gpt_label("merge_entry"))
        assert result.triggered is False

    def test_non_merge_label_not_checked(self):
        """press_attack label should not trigger standoff guard."""
        obs = _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=False)
        result = reject_merge_entry_for_standoff(obs, _gpt_label("press_attack"))
        assert result.triggered is False

    def test_is_merge_true_bypasses(self):
        """If is_merge=True, don't reject even with standoff-like geometry."""
        obs = _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=True)
        result = reject_merge_entry_for_standoff(obs, _gpt_label("merge_entry"))
        assert result.triggered is False

    def test_high_closure_not_standoff(self):
        """closure=-200 means actively approaching, not standoff."""
        obs = _obs(closure=-200.0, bearing=90.0, aspect=90.0, is_merge=False)
        result = reject_merge_entry_for_standoff(obs, _gpt_label("merge_entry"))
        assert result.triggered is False


# ── Test 2: apply_guardrails with protected rule ──

class TestApplyGuardrails:
    def test_protected_rule_preserved(self):
        """Rule=hold_geometry(0.85) should be preserved when guard triggers."""
        obs = _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=False)
        gpt_rec = {
            "label": {"semantic_state": "merge_entry", "target_ref": "enemy_1"},
            "quality": {"tier": "bronze", "confidence": 0.72,
                        "conflict_flag": False, "needs_review": False},
            "source": {"type": "acmi_ai"},
            "evidence": {"rule_ids": [], "notes": ""},
            "obs": obs,
        }
        rule_rec = {
            "label": {"semantic_state": "hold_geometry", "target_ref": "enemy_1"},
            "quality": {"tier": "bronze", "confidence": 0.85},
        }
        result = apply_guardrails(obs, gpt_rec, rule_rec)
        assert result["label"]["semantic_state"] == "hold_geometry"
        assert result["quality"]["conflict_flag"] is True
        assert result["quality"]["needs_review"] is True
        assert result["source"]["type"] == "mixed"
        assert "GUARD" in result["evidence"]["notes"]

    def test_no_rule_label_goes_null(self):
        """If no protected rule, guard should nullify."""
        obs = _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=False)
        gpt_rec = {
            "label": {"semantic_state": "merge_entry", "target_ref": "enemy_1"},
            "quality": {"tier": "bronze", "confidence": 0.72,
                        "conflict_flag": False, "needs_review": False},
            "source": {"type": "acmi_ai"},
            "evidence": {"notes": ""},
            "obs": obs,
        }
        result = apply_guardrails(obs, gpt_rec, None)
        assert result["label"]["semantic_state"] is None
        assert result["quality"]["conflict_flag"] is True
        assert result["quality"]["tier"] == "weak"

    def test_low_conf_rule_not_protected(self):
        """Rule with confidence < threshold should not be protected."""
        obs = _obs(closure=0.0, bearing=90.0, aspect=90.0, is_merge=False)
        gpt_rec = {
            "label": {"semantic_state": "merge_entry", "target_ref": "enemy_1"},
            "quality": {"tier": "bronze", "confidence": 0.72,
                        "conflict_flag": False, "needs_review": False},
            "source": {"type": "acmi_ai"},
            "evidence": {"notes": ""},
            "obs": obs,
        }
        rule_rec = {
            "label": {"semantic_state": "hold_geometry", "target_ref": "enemy_1"},
            "quality": {"tier": "bronze", "confidence": 0.50},  # below threshold
        }
        result = apply_guardrails(obs, gpt_rec, rule_rec)
        assert result["label"]["semantic_state"] is None  # nullified, not protected


# ── Test 3: commit/press boundary mode ──

class TestBoundaryMode:
    def test_9km_headon_not_auto_changed(self):
        """9km/bearing0/closure-580: guardrails don't auto-change press to commit."""
        obs = _obs(closure=-580.0, bearing=0.0, aspect=5.0, is_merge=False)
        obs["enemy"]["range_m"] = 9000.0
        gpt_rec = {
            "label": {"semantic_state": "press_attack", "target_ref": "enemy_1"},
            "quality": {"tier": "bronze", "confidence": 0.74,
                        "conflict_flag": False, "needs_review": False},
            "source": {"type": "acmi_ai"},
            "evidence": {"notes": ""},
            "obs": obs,
        }
        # Guardrails should NOT change press_attack to commit_intercept
        # That's a prompt/boundary_mode concern, not a guardrail
        result = apply_guardrails(obs, gpt_rec, None)
        assert result["label"]["semantic_state"] == "press_attack"
