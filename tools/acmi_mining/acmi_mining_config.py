"""Centralized configuration for ACMI mining pipeline.

Shared thresholds are imported from runtime; bucket-specific thresholds
are defined here. All angles in degrees, distances in metres, speeds in m/s.
"""

from __future__ import annotations

from sim.runtime.semantic_thresholds import (  # noqa: F401
    DEF_ASPECT_DEG,
    DEF_RANGE_M,
    MERGE_RANGE_M,
    TARGET_SWITCH_RATIO,
)

# ── Bucket A: far_intercept ───────────────────────────────────
FAR_INTERCEPT_MIN_RANGE_M: float = 12_000.0
FAR_INTERCEPT_PREFERRED_RANGE_M: float = 15_000.0
FAR_INTERCEPT_MAX_BEARING_DEG: float = 30.0
FAR_INTERCEPT_MAX_CLOSURE_MPS: float = -50.0   # must be negative (closing)
FAR_INTERCEPT_STABILITY_FRAMES: int = 2         # relaxed for DCS data

# ── Bucket B: standoff_hold ───────────────────────────────────
STANDOFF_MIN_RANGE_M: float = 2_000.0
STANDOFF_MAX_RANGE_M: float = 30_000.0       # widened for DCS multi-ship data
STANDOFF_MAX_CLOSURE_ABS_MPS: float = 100.0  # widened significantly for DCS data
STANDOFF_MIN_BEARING_ABS_DEG: float = 20.0   # very relaxed for DCS data
STANDOFF_MAX_BEARING_ABS_DEG: float = 160.0
STANDOFF_MIN_ASPECT_DEG: float = 20.0
STANDOFF_MAX_ASPECT_DEG: float = 160.0
STANDOFF_STABILITY_FRAMES: int = 1           # single-frame detection for sparse data

# ── Bucket C: extend_disengage ────────────────────────────────
EXTEND_MIN_CLOSURE_MPS: float = 0.0             # positive = opening
EXTEND_MAX_RANGE_M: float = 50_000.0            # cap: >50km is setup, not tactical extend
EXTEND_NOT_MERGE: bool = True
EXTEND_NOT_DEFENSIVE: bool = True
EXTEND_STABILITY_FRAMES: int = 2

# ── Bucket D: turn_defense ────────────────────────────────────
TURN_HDG_RATE_THRESHOLD_DPS: float = 3.0        # very low for initial scan
TURN_BEARING_CHANGE_THRESHOLD_DPS: float = 3.0
TURN_DEFENSIVE_RANGE_M: float = 50_000.0        # widened significantly

# ── Bucket E: handoff_coordination ────────────────────────────
HANDOFF_REQUIRE_TWO_ENEMIES: bool = True
HANDOFF_REQUIRE_ALLY: bool = True

# ── Sampling discipline ──────────────────────────────────────
MAX_EVENTS_PER_FILE: int = 8
MAX_EVENTS_PER_CLUSTER_FRAMES: int = 2
CLUSTER_WINDOW_SEC: float = 4.0                  # events within 4s are a cluster

# ── Target quotas ─────────────────────────────────────────────
BUCKET_QUOTAS = {
    "far_intercept": 35,
    "standoff_hold": 25,
    "extend_disengage": 25,
    "turn_defense": 35,
    "handoff_coordination": 20,
}
TOTAL_TARGET: int = sum(BUCKET_QUOTAS.values())  # 140

# ── v2 enhanced thresholds ───────────────────────────────────
# Bucket D v2: offensive vs defensive turn distinction
TURN_V2_HDG_RATE_ONSET_DPS: float = 10.0         # primary trigger: turn onset
TURN_V2_OFFENSIVE_MAX_RANGE_M: float = 8_000.0   # offensive_turn: range < 8km
TURN_V2_DEFENSIVE_ASPECT_DEG: float = 120.0      # defensive_break: aspect > 120

# Bucket E v2: support / target_switch detection
HANDOFF_V2_SUPPORT_RANGE_KM: float = 15.0        # local_friend_count radius
HANDOFF_V2_CLUSTER_KEEP: int = 2                  # frames to keep per event cluster

# ── v2 Target quotas ─────────────────────────────────────────
BUCKET_QUOTAS_V2 = {
    "far_intercept": 25,
    "standoff_hold": 20,
    "extend_disengage": 20,
    "turn_defense": 35,
    "handoff_coordination": 30,
}
TOTAL_TARGET_V2: int = sum(BUCKET_QUOTAS_V2.values())  # 130

# ── ACMI parsing ──────────────────────────────────────────────
ACMI_RESAMPLE_DT: float = 0.5                    # 2Hz
ACMI_MIN_SPEED_MPS: float = 60.0
ACMI_MAX_SPEED_MPS: float = 800.0
ACMI_FIXED_WING_FILTER: bool = True
