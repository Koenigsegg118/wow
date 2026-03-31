"""Centralised thresholds for engagement flags and target switching.

Unique source for all numeric thresholds used by the obs-sidecar builder.
No other module may hard-code these values — they must import from here.

Thresholds align with ``annotation_guideline_v1.md`` (the canonical unique
source for engagement semantics).
"""

from __future__ import annotations

# ── engagement flags ────────────────────────────────────────────
# Source: annotation_guideline_v1.md §2 engagement 判定

# is_merge: range < MERGE_RANGE_M
MERGE_RANGE_M: float = 3_000.0

# is_defensive: aspect > DEF_ASPECT_DEG AND closure < 0 AND range < DEF_RANGE_M
DEF_ASPECT_DEG: float = 120.0
DEF_RANGE_M: float = 15_000.0

# has_shot_opportunity: |bearing| < SHOT_BEARING_DEG AND range < SHOT_RANGE_M
#                       AND aspect < SHOT_ASPECT_DEG
# Currently set to null at runtime (no reliable weapon envelope).
# Kept here for future activation.
SHOT_BEARING_DEG: float = 30.0
SHOT_RANGE_M: float = 20_000.0
SHOT_ASPECT_DEG: float = 90.0

# ── target switching ────────────────────────────────────────────
# Sticky-target: only switch when new target range < old * ratio.
TARGET_SWITCH_RATIO: float = 0.80

# ── physics constants ───────────────────────────────────────────
EARTH_RADIUS_M: float = 6_378_137.0
GRAVITY_MPS2: float = 9.80665

# ── platform validity ──────────────────────────────────────────
# Minimum speed to consider a platform "alive and moving"
MIN_VALID_SPEED_MPS: float = 10.0
# Minimum range to avoid division-by-zero in closure
MIN_RANGE_M: float = 1.0
# Minimum horizontal speed for heading-from-velocity fallback
MIN_HSPEED_FOR_HEADING_MPS: float = 1.0
