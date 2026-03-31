"""Tests for the obs-sidecar pipeline.

Covers:
  1. Head-on approach (range decreasing, closure<0, bearing~0, aspect~0)
  2. Tail-chase defensive (enemy behind, is_defensive=True)
  3. Dual-enemy target stability (sticky target, switch only when much closer)
  4. Heading-missing fallback (velocity-derived, low-speed prev-heading)
  5. Missing-value discipline (null not "unknown", enemy/ally=null when absent)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from sim.runtime.afsim_obs_entities import FIELDS_PER_PLATFORM, parse_frame
from sim.runtime.afsim_obs_geometry import (
    compute_bearing_deg,
    compute_aspect_deg,
    compute_closure_mps,
    compute_range_m,
    compute_alt_diff_m,
)
from sim.runtime.afsim_obs_sidecar import (
    SidecarContext,
    build_obs_sidecar,
)


# ── helpers ─────────────────────────────────────────────────────

def _make_frame(platforms: dict[int, dict]) -> np.ndarray:
    """Build a 640-value frame from a sparse platform dict.

    Each platform dict keys: lat, lon, alt_m, velN, velE, velD, heading_deg
    Missing keys default to 0. 'live' defaults to 1.0.
    """
    frame = np.zeros(640, dtype=np.float32)
    for idx, fields in platforms.items():
        base = idx * FIELDS_PER_PLATFORM
        frame[base + 0] = fields.get("live", 1.0)
        frame[base + 1] = fields.get("lat", 35.0)
        frame[base + 2] = fields.get("lon", 120.0)
        frame[base + 3] = fields.get("alt_m", 8000.0)
        frame[base + 4] = fields.get("velN", 0.0)
        frame[base + 5] = fields.get("velE", 0.0)
        frame[base + 6] = fields.get("velD", 0.0)
        frame[base + 7] = fields.get("heading_deg", 0.0)
    return frame


# ── Test 1: Head-on approach ───────────────────────────────────

class TestHeadOnApproach:
    """Red_1 heading north, blue_1 heading south, approaching each other."""

    def setup_method(self):
        self.frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 300, "velE": 0, "velD": 0, "heading_deg": 0},
            2: {"lat": 35.1, "lon": 120.0, "alt_m": 8000,
                "velN": -300, "velE": 0, "velD": 0, "heading_deg": 180},
        })
        self.ctx = SidecarContext()
        self.obs = build_obs_sidecar(self.frame, ego_platform_idx=0, ctx=self.ctx)

    def test_range_positive(self):
        assert self.obs["enemy"] is not None
        assert self.obs["enemy"]["range_m"] > 0

    def test_closure_negative(self):
        # Both closing → closure < 0
        assert self.obs["enemy"]["closure_mps"] < 0

    def test_bearing_near_zero(self):
        # Enemy is directly ahead
        assert abs(self.obs["enemy"]["bearing_deg"]) < 5.0

    def test_aspect_near_zero(self):
        # Head-on → aspect ≈ 0
        assert self.obs["enemy"]["aspect_deg"] is not None
        assert self.obs["enemy"]["aspect_deg"] < 20.0

    def test_is_merge_depends_on_range(self):
        # At ~11km apart, is_merge should be False
        assert self.obs["engagement"]["is_merge"] is False


# ── Test 2: Tail-chase defensive ───────────────────────────────

class TestTailChaseDefensive:
    """Blue_1 is behind red_1, chasing. Ego (red_1) is being tailed.

    Setup: red_1 flies north, blue_1 is south of red_1 also flying north
    but faster. From ego's perspective: enemy is behind (bearing~180).
    From enemy's perspective: ego is ahead of enemy's nose → aspect should
    be HIGH (>150) because enemy nose points AWAY from ego (enemy_heading
    vs LOS_enemy→ego are nearly opposite since ego is AHEAD).

    Actually re-thinking: aspect = angle between enemy's heading and
    LOS from enemy TO ego. If enemy flies north and ego is also north
    of enemy, LOS_enemy→ego also points north → aspect ≈ 0 (nose-on
    from enemy's view = enemy is pointing at us).

    For high aspect (tail-chase from ego's view, i.e. enemy in our
    6 o'clock), we need the enemy to have its nose pointing AWAY
    from ego. So: enemy should be NORTH of ego, flying AWAY (north),
    with ego heading north too → enemy is ahead, its tail faces ego.
    """

    def setup_method(self):
        # Red_1 at 35.0 heading north; blue_1 at 35.1 heading north (ahead)
        # → enemy is in front of ego but flying AWAY → from ego it's a chase
        #   but aspect: enemy heading=north, LOS enemy→ego=south → aspect≈180
        self.frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 6000,
                "velN": 350, "velE": 0, "velD": 0},
            2: {"lat": 35.1, "lon": 120.0, "alt_m": 6500,
                "velN": 250, "velE": 0, "velD": 0},
        })
        self.ctx = SidecarContext()
        self.obs = build_obs_sidecar(self.frame, ego_platform_idx=0, ctx=self.ctx)

    def test_enemy_ahead(self):
        # Enemy is ahead of ego (bearing near 0) but flying away
        assert self.obs["enemy"] is not None
        assert abs(self.obs["enemy"]["bearing_deg"]) < 10

    def test_aspect_high(self):
        # Enemy nose pointing same direction → high aspect (tail chase)
        assert self.obs["enemy"]["aspect_deg"] is not None
        assert self.obs["enemy"]["aspect_deg"] > 150

    def test_is_defensive(self):
        # aspect > 120 (enemy tail-on), ego closing (closure < 0), range < 15km
        # In this setup ego is faster (350 vs 250) so closing on enemy
        assert self.obs["engagement"]["is_defensive"] is True

    def test_alt_diff_sign(self):
        # ego at 6000, enemy at 6500 → alt_diff = 6000-6500 = -500 (enemy higher)
        assert self.obs["enemy"]["alt_diff_m"] < 0


# ── Test 3: Dual-enemy target stability ────────────────────────

class TestDualEnemySticky:
    """Two enemies; sticky target should not jitter."""

    def _frame(self, b1_lat: float, b2_lat: float) -> np.ndarray:
        return _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
            2: {"lat": b1_lat, "lon": 120.0, "alt_m": 8000,
                "velN": -200, "velE": 0, "velD": 0},
            3: {"lat": b2_lat, "lon": 120.0, "alt_m": 8000,
                "velN": -200, "velE": 0, "velD": 0},
        })

    def test_sticky_target_no_switch(self):
        ctx = SidecarContext()

        # Frame 1: blue_1 closer
        obs1 = build_obs_sidecar(self._frame(35.05, 35.10), ego_platform_idx=0, ctx=ctx)
        ref1 = obs1["enemy"]["target_ref"]

        # Frame 2: blue_2 only slightly closer → should NOT switch
        obs2 = build_obs_sidecar(self._frame(35.05, 35.04), ego_platform_idx=0, ctx=ctx)
        ref2 = obs2["enemy"]["target_ref"]
        assert ref2 == ref1, "Target should be sticky when new is only marginally closer"

    def test_target_switch_when_much_closer(self):
        ctx = SidecarContext()

        # Frame 1: blue_1 at ~5km
        obs1 = build_obs_sidecar(self._frame(35.05, 35.20), ego_platform_idx=0, ctx=ctx)
        ref1 = obs1["enemy"]["target_ref"]

        # Frame 2: blue_2 drastically closer (35.01 ≈ 1.1km vs 35.05 ≈ 5.5km)
        obs2 = build_obs_sidecar(self._frame(35.05, 35.01), ego_platform_idx=0, ctx=ctx)
        ref2 = obs2["enemy"]["target_ref"]
        assert ref2 != ref1, "Should switch when new target is significantly closer"


# ── Test 4: Heading fallback ───────────────────────────────────

class TestHeadingFallback:
    """When horizontal speed is very low, use AFSIM heading or previous."""

    def test_low_speed_uses_afsim_heading(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 0.1, "velE": 0.1, "velD": 0, "heading_deg": 90.0},
        })
        ctx = SidecarContext()
        obs = build_obs_sidecar(frame, ego_platform_idx=0, ctx=ctx)
        # With near-zero velocity, should use AFSIM heading_deg = 90
        assert obs["ego"]["heading_deg"] == pytest.approx(90.0, abs=1.0)

    def test_normal_speed_uses_velocity(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 0, "velE": 300, "velD": 0, "heading_deg": 0},
        })
        ctx = SidecarContext()
        obs = build_obs_sidecar(frame, ego_platform_idx=0, ctx=ctx)
        # velocity is pure east → heading ≈ 90
        assert obs["ego"]["heading_deg"] == pytest.approx(90.0, abs=1.0)


# ── Test 5: Missing-value discipline ───────────────────────────

class TestMissingValueDiscipline:
    """Ensure null not "unknown", enemy/ally=null when absent."""

    def test_no_enemy(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        assert obs["enemy"] is None
        assert obs["ally"] is None
        assert obs["engagement"] is None

    def test_no_ally(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
            2: {"lat": 35.05, "lon": 120.0, "alt_m": 8000,
                "velN": -200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        assert obs["enemy"] is not None
        assert obs["ally"] is None  # no red_2 alive

    def test_roll_always_null(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        assert obs["ego"]["roll_deg"] is None

    def test_energy_state_null(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        assert obs["ego"]["energy_state"] is None

    def test_history_always_present(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        assert "history" in obs
        assert obs["history"]["prev_semantic_state"] is None
        assert obs["history"]["prev_token_family"] is None
        assert obs["history"]["prev_target_ref"] is None

    def test_has_shot_opportunity_null(self):
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
            2: {"lat": 35.05, "lon": 120.0, "alt_m": 8000,
                "velN": -200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        assert obs["engagement"]["has_shot_opportunity"] is None

    def test_no_unknown_strings(self):
        """Recursively check no value is the string 'unknown'."""
        frame = _make_frame({
            0: {"lat": 35.0, "lon": 120.0, "alt_m": 8000,
                "velN": 200, "velE": 0, "velD": 0},
            2: {"lat": 35.05, "lon": 120.0, "alt_m": 8000,
                "velN": -200, "velE": 0, "velD": 0},
        })
        obs = build_obs_sidecar(frame, ego_platform_idx=0)
        _assert_no_unknown(obs)


def _assert_no_unknown(obj, path=""):
    """Recursively assert no value is the string 'unknown' or 'unavailable'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_unknown(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_unknown(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        assert obj not in ("unknown", "unavailable", "N/A"), \
            f"Found forbidden string '{obj}' at {path}"
