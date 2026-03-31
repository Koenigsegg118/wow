"""Tests for tools.d2b.validate_gpt_output -- V01-V11 validation rules.

Each test verifies a single rule's on-fail behaviour (reject, truncate,
clamp, strip, auto-fix) plus a happy-path valid-output test.
"""

from __future__ import annotations

import copy

import pytest

from tools.d2b.validate_gpt_output import ValidationResult, validate_format_c


# ===================================================================
# Helpers
# ===================================================================

def _valid_gpt_output() -> dict:
    """Return a minimal valid format_C GPT output."""
    return {
        "label": {
            "semantic_state": "commit_intercept",
            "target_ref": "enemy_1",
            "role": "engaged",
            "constraints": {
                "prefer_energy_preserve": None,
                "allow_aggressive_turn": True,
                "must_support_teammate": None,
                "should_abort_if_threatened": True,
            },
            "profile_hint": "p6dof_semantic",
            "rationale_short": "stable intercept geometry",
            "event_flags": {
                "target_switch": False,
            },
        },
        "quality_hint": {
            "confidence": 0.85,
            "needs_review": False,
            "conflict_flag": False,
        },
    }


def _assert_rejected(result: ValidationResult):
    """Assert that the result was rejected."""
    assert result.is_valid is False
    assert result.fixed is None
    assert len(result.errors) >= 1


def _assert_valid(result: ValidationResult):
    """Assert that the result passed validation."""
    assert result.is_valid is True
    assert result.fixed is not None
    assert len(result.errors) == 0


# ===================================================================
# V01 -- semantic_state in enum or None
# ===================================================================

class TestV01:
    def test_invalid_semantic_state_rejected(self):
        doc = _valid_gpt_output()
        doc["label"]["semantic_state"] = "bogus_state"
        result = validate_format_c(doc)
        _assert_rejected(result)
        assert any("V01" in e for e in result.errors)

    def test_none_semantic_state_allowed(self):
        doc = _valid_gpt_output()
        doc["label"]["semantic_state"] = None
        result = validate_format_c(doc)
        _assert_valid(result)


# ===================================================================
# V02 -- target_ref in enum or None
# ===================================================================

class TestV02:
    def test_invalid_target_ref_rejected(self):
        doc = _valid_gpt_output()
        doc["label"]["target_ref"] = "enemy_99"
        result = validate_format_c(doc)
        _assert_rejected(result)
        assert any("V02" in e for e in result.errors)

    def test_none_target_ref_allowed(self):
        doc = _valid_gpt_output()
        doc["label"]["target_ref"] = None
        result = validate_format_c(doc)
        _assert_valid(result)


# ===================================================================
# V03 -- role in enum or None
# ===================================================================

class TestV03:
    def test_invalid_role_rejected(self):
        doc = _valid_gpt_output()
        doc["label"]["role"] = "commander"
        result = validate_format_c(doc)
        _assert_rejected(result)
        assert any("V03" in e for e in result.errors)

    def test_none_role_allowed(self):
        doc = _valid_gpt_output()
        doc["label"]["role"] = None
        result = validate_format_c(doc)
        _assert_valid(result)


# ===================================================================
# V06 -- rationale_short > 120 chars => truncated
# ===================================================================

class TestV06:
    def test_long_rationale_truncated(self):
        doc = _valid_gpt_output()
        long_text = "A" * 200
        doc["label"]["rationale_short"] = long_text
        result = validate_format_c(doc)
        _assert_valid(result)  # truncate, not reject
        assert any("V06" in w for w in result.warnings)
        assert "label.rationale_short" in result.auto_fixes
        assert len(result.fixed["label"]["rationale_short"]) == 120

    def test_exactly_120_chars_ok(self):
        doc = _valid_gpt_output()
        doc["label"]["rationale_short"] = "B" * 120
        result = validate_format_c(doc)
        _assert_valid(result)
        assert "label.rationale_short" not in result.auto_fixes


# ===================================================================
# V07 -- confidence > 1.0 => clamped to 1.0
# ===================================================================

class TestV07:
    def test_confidence_above_1_clamped(self):
        doc = _valid_gpt_output()
        doc["quality_hint"]["confidence"] = 1.5
        result = validate_format_c(doc)
        _assert_valid(result)  # clamp, not reject
        assert any("V07" in w for w in result.warnings)
        assert result.fixed["quality_hint"]["confidence"] == 1.0

    def test_confidence_below_0_clamped(self):
        doc = _valid_gpt_output()
        doc["quality_hint"]["confidence"] = -0.5
        result = validate_format_c(doc)
        _assert_valid(result)
        assert result.fixed["quality_hint"]["confidence"] == 0.0

    def test_confidence_within_range_ok(self):
        doc = _valid_gpt_output()
        doc["quality_hint"]["confidence"] = 0.75
        result = validate_format_c(doc)
        _assert_valid(result)
        assert "quality_hint.confidence" not in result.auto_fixes


# ===================================================================
# V08 -- extra top-level key => stripped
# ===================================================================

class TestV08:
    def test_extra_key_stripped(self):
        doc = _valid_gpt_output()
        doc["debug_info"] = {"internal": True}
        doc["reasoning"] = "some chain of thought"
        result = validate_format_c(doc)
        _assert_valid(result)
        assert any("V08" in w for w in result.warnings)
        assert "debug_info" not in result.fixed
        assert "reasoning" not in result.fixed

    def test_no_extra_keys_no_warning(self):
        doc = _valid_gpt_output()
        result = validate_format_c(doc)
        _assert_valid(result)
        assert not any("V08" in w for w in result.warnings)


# ===================================================================
# V11 -- conflict_flag=True + needs_review=False => auto-fixed
# ===================================================================

class TestV11:
    def test_conflict_flag_auto_fix(self):
        doc = _valid_gpt_output()
        doc["quality_hint"]["conflict_flag"] = True
        doc["quality_hint"]["needs_review"] = False
        result = validate_format_c(doc)
        _assert_valid(result)
        assert any("V11" in w for w in result.warnings)
        assert result.fixed["quality_hint"]["needs_review"] is True

    def test_conflict_flag_with_needs_review_true_no_fix(self):
        doc = _valid_gpt_output()
        doc["quality_hint"]["conflict_flag"] = True
        doc["quality_hint"]["needs_review"] = True
        result = validate_format_c(doc)
        _assert_valid(result)
        assert "quality_hint.needs_review" not in result.auto_fixes


# ===================================================================
# Valid output -- full happy path
# ===================================================================

class TestValidOutput:
    def test_valid_output_passes(self):
        doc = _valid_gpt_output()
        result = validate_format_c(doc)
        _assert_valid(result)
        assert len(result.warnings) == 0
        assert len(result.auto_fixes) == 0
        # fixed output should preserve the label contents
        assert result.fixed["label"]["semantic_state"] == "commit_intercept"
        assert result.fixed["quality_hint"]["confidence"] == 0.85

    def test_valid_with_none_fields_passes(self):
        """All-None optional fields should still pass."""
        doc = {
            "label": {
                "semantic_state": "extend",
                "target_ref": None,
                "role": None,
                "constraints": None,
                "profile_hint": None,
                "rationale_short": None,
                "event_flags": None,
            },
            "quality_hint": {
                "confidence": 0.7,
                "needs_review": False,
                "conflict_flag": False,
            },
        }
        result = validate_format_c(doc)
        _assert_valid(result)


# ===================================================================
# Multiple errors at once
# ===================================================================

class TestMultipleErrors:
    def test_two_reject_errors(self):
        """Both V01 and V02 can fire on the same record."""
        doc = _valid_gpt_output()
        doc["label"]["semantic_state"] = "invalid_state"
        doc["label"]["target_ref"] = "invalid_ref"
        result = validate_format_c(doc)
        _assert_rejected(result)
        assert any("V01" in e for e in result.errors)
        assert any("V02" in e for e in result.errors)
