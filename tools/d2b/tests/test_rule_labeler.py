"""Tests for tools.d2b.rule_labeler_v1 -- conservative rule labeler.

Covers all 5 easy-case labels plus boundary conditions and forbidden states.
"""

from __future__ import annotations

import pytest

from tools.d2b.rule_labeler_v1 import (
    RuleResult,
    _HARD_CASE_STATES,
    classify_easy_case,
    build_canonical_from_rule,
)


# ===================================================================
# Helper
# ===================================================================

def _obs(
    enemy_range=None,
    enemy_bearing=None,
    enemy_aspect=None,
    enemy_closure=None,
    is_merge=False,
    is_defensive=False,
    prev_target_ref=None,
    ego_alt=8000,
    ego_speed=300,
):
    """Build a minimal obs dict for testing."""
    obs: dict = {
        "ego": {
            "alt_m": ego_alt,
            "speed_mps": ego_speed,
        },
        "engagement": {
            "is_merge": is_merge,
            "is_defensive": is_defensive,
        },
        "history": {},
    }

    if prev_target_ref is not None:
        obs["history"]["prev_target_ref"] = prev_target_ref

    # Build enemy sub-dict only when at least one enemy field is provided
    if any(v is not None for v in (enemy_range, enemy_bearing, enemy_aspect, enemy_closure)):
        obs["enemy"] = {
            "range_m": enemy_range,
            "bearing_deg": enemy_bearing,
            "aspect_deg": enemy_aspect,
            "closure_mps": enemy_closure,
            "target_ref": "enemy_1",
        }
    # else: no "enemy" key at all

    return obs


def _assert_rule_result_structure(result: RuleResult):
    """Check that a RuleResult has the expected structural shape."""
    assert isinstance(result, RuleResult)
    assert isinstance(result.rule_ids, list)
    assert len(result.rule_ids) >= 1, "rule_ids must be non-empty"
    assert isinstance(result.evidence_notes, str)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


# ===================================================================
# merge_entry
# ===================================================================

class TestMergeEntry:
    def test_is_merge_flag(self):
        """merge_entry via is_merge=True."""
        obs = _obs(is_merge=True, enemy_range=3000, enemy_closure=-150)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "merge_entry"
        _assert_rule_result_structure(result)

    def test_range_and_closure(self):
        """merge_entry via range<5000 and closure<-100 (no is_merge flag)."""
        obs = _obs(enemy_range=4500, enemy_closure=-120)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "merge_entry"
        _assert_rule_result_structure(result)

    def test_is_merge_flag_no_enemy_details(self):
        """merge_entry via is_merge=True even with minimal enemy info."""
        obs = _obs(is_merge=True, enemy_range=2000)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "merge_entry"
        _assert_rule_result_structure(result)


# ===================================================================
# commit_intercept
# ===================================================================

class TestCommitIntercept:
    def test_standard_commit(self):
        """commit_intercept: bearing=10, closure=-300, range=20000."""
        obs = _obs(enemy_range=20000, enemy_bearing=10, enemy_closure=-300)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "commit_intercept"
        _assert_rule_result_structure(result)

    def test_negative_bearing_commit(self):
        """commit_intercept: bearing=-15 (still |bearing|<30)."""
        obs = _obs(enemy_range=18000, enemy_bearing=-15, enemy_closure=-200)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "commit_intercept"
        _assert_rule_result_structure(result)

    def test_not_commit_if_defensive(self):
        """Defensive flag blocks commit_intercept."""
        obs = _obs(
            enemy_range=20000, enemy_bearing=10, enemy_closure=-300,
            is_defensive=True,
        )
        result = classify_easy_case(obs)
        # Should NOT be commit_intercept -- defensive blocks it
        if result is not None:
            assert result.semantic_state != "commit_intercept"


# ===================================================================
# extend
# ===================================================================

class TestExtend:
    def test_standard_extend(self):
        """extend: closure=50 (separating), bearing=120."""
        obs = _obs(enemy_range=12000, enemy_bearing=120, enemy_closure=50)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "extend"
        _assert_rule_result_structure(result)

    def test_extend_negative_bearing(self):
        """extend: bearing=-100 (|bearing|>90), closure positive."""
        obs = _obs(enemy_range=15000, enemy_bearing=-100, enemy_closure=80)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "extend"
        _assert_rule_result_structure(result)

    def test_extend_blocked_by_merge(self):
        """is_merge=True takes priority over extend conditions."""
        obs = _obs(
            enemy_range=12000, enemy_bearing=120, enemy_closure=50,
            is_merge=True,
        )
        result = classify_easy_case(obs)
        assert result is not None
        # merge_entry has higher priority
        assert result.semantic_state == "merge_entry"


# ===================================================================
# hold_geometry
# ===================================================================

class TestHoldGeometry:
    def test_standard_hold(self):
        """hold_geometry: |closure|<20, not merge, not defensive."""
        obs = _obs(enemy_range=10000, enemy_bearing=45, enemy_closure=5)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "hold_geometry"
        _assert_rule_result_structure(result)

    def test_hold_negative_closure(self):
        """hold_geometry: closure=-10 (|closure|=10 < 20)."""
        obs = _obs(enemy_range=10000, enemy_bearing=45, enemy_closure=-10)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "hold_geometry"
        _assert_rule_result_structure(result)

    def test_hold_blocked_by_defensive(self):
        """is_defensive=True blocks hold_geometry."""
        obs = _obs(enemy_range=10000, enemy_bearing=45, enemy_closure=5,
                    is_defensive=True)
        result = classify_easy_case(obs)
        # Should NOT be hold_geometry
        if result is not None:
            assert result.semantic_state != "hold_geometry"


# ===================================================================
# null
# ===================================================================

class TestNull:
    def test_no_enemy(self):
        """null: no enemy key at all -> semantic_state is None."""
        obs = _obs()  # no enemy fields
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state is None
        _assert_rule_result_structure(result)

    def test_insufficient_info(self):
        """null: enemy exists but only range_m (no bearing/closure)."""
        obs = _obs(enemy_range=10000)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state is None
        _assert_rule_result_structure(result)
        assert "insufficient" in result.rule_ids[0].lower() or \
               "R07" in result.rule_ids[0]


# ===================================================================
# Boundary tests
# ===================================================================

class TestBoundary:
    def test_bearing_exactly_30(self):
        """bearing=30 exactly -- commit requires abs(bearing) < 30, so NOT commit."""
        obs = _obs(enemy_range=20000, enemy_bearing=30, enemy_closure=-200)
        result = classify_easy_case(obs)
        # abs(bearing) = 30 is NOT < 30, so commit_intercept should not match
        if result is not None:
            assert result.semantic_state != "commit_intercept"

    def test_closure_exactly_0(self):
        """closure=0 exactly -- hold_geometry requires |closure|<20, 0 qualifies."""
        obs = _obs(enemy_range=10000, enemy_bearing=45, enemy_closure=0)
        result = classify_easy_case(obs)
        assert result is not None
        assert result.semantic_state == "hold_geometry"
        _assert_rule_result_structure(result)

    def test_range_exactly_at_merge_boundary(self):
        """range=5000 exactly -- merge requires range < 5000, so NOT merge."""
        obs = _obs(enemy_range=5000, enemy_closure=-120)
        result = classify_easy_case(obs)
        if result is not None:
            assert result.semantic_state != "merge_entry"

    def test_bearing_exactly_90(self):
        """bearing=90 exactly -- extend requires |bearing| > 90, so NOT extend."""
        obs = _obs(enemy_range=12000, enemy_bearing=90, enemy_closure=50)
        result = classify_easy_case(obs)
        if result is not None:
            assert result.semantic_state != "extend"


# ===================================================================
# Forbidden states
# ===================================================================

class TestForbiddenStates:
    """Rule labeler must NEVER emit hard-case labels."""

    _FORBIDDEN = frozenset({
        "press_attack",
        "support",
        "offensive_turn",
        "defensive_break",
        "energy_manage",
    })

    @pytest.mark.parametrize("state", sorted(_FORBIDDEN))
    def test_forbidden_state_never_emitted(self, state):
        """No combination of easy-case inputs should produce a hard-case label."""
        # Try a broad range of obs combinations
        test_cases = [
            _obs(),
            _obs(is_merge=True),
            _obs(enemy_range=20000, enemy_bearing=10, enemy_closure=-300),
            _obs(enemy_range=4500, enemy_closure=-120),
            _obs(enemy_range=12000, enemy_bearing=120, enemy_closure=50),
            _obs(enemy_range=10000, enemy_bearing=45, enemy_closure=5),
            _obs(enemy_range=10000),
            _obs(is_defensive=True, enemy_range=8000, enemy_bearing=150,
                 enemy_closure=-50, enemy_aspect=30),
        ]
        for obs in test_cases:
            result = classify_easy_case(obs)
            if result is not None:
                assert result.semantic_state != state, (
                    f"Forbidden state {state!r} was emitted for obs: {obs}"
                )

    def test_hard_case_states_constant(self):
        """_HARD_CASE_STATES matches our expected set."""
        assert _HARD_CASE_STATES == self._FORBIDDEN


# ===================================================================
# build_canonical_from_rule
# ===================================================================

class TestBuildCanonical:
    def test_canonical_shape(self):
        """Canonical record has all required top-level keys."""
        obs = _obs(enemy_range=20000, enemy_bearing=10, enemy_closure=-300)
        result = classify_easy_case(obs)
        assert result is not None

        meta = {
            "scenario_id": "test_scenario",
            "episode_id": "ep01",
            "timestamp_sec": 42.0,
        }
        record = build_canonical_from_rule(obs, result, meta)

        required_keys = {
            "schema_version", "sample_id", "source", "obs",
            "label", "evidence", "quality", "extras",
        }
        assert required_keys.issubset(record.keys())
        assert record["schema_version"] == "semantic_record_v1"
        assert record["source"]["tool"] == "rule_labeler_v1.py"
        assert record["quality"]["tier"] == "bronze"
        assert record["evidence"]["rule_ids"] == result.rule_ids

    def test_target_switch_detected(self):
        """target_switch is True when prev_target_ref differs from current."""
        obs = _obs(
            enemy_range=20000, enemy_bearing=10, enemy_closure=-300,
            prev_target_ref="enemy_2",
        )
        result = classify_easy_case(obs)
        assert result is not None
        meta = {"scenario_id": "sw", "timestamp_sec": 10.0}
        record = build_canonical_from_rule(obs, result, meta)
        # result.target_ref should be "enemy_1", prev is "enemy_2"
        assert record["label"]["event_flags"]["target_switch"] is True

    def test_null_state_needs_review(self):
        """Records with semantic_state=None have needs_review=True."""
        obs = _obs()
        result = classify_easy_case(obs)
        assert result is not None
        meta = {"scenario_id": "null", "timestamp_sec": 0.0}
        record = build_canonical_from_rule(obs, result, meta)
        assert record["quality"]["needs_review"] is True
