"""Centralised configuration for the D2b (rule + GPT review) pipeline.

Shared thresholds (engagement flags, target switching, physics) are imported
from ``sim.runtime.semantic_thresholds`` -- the single source of truth.
This module adds D2b-specific rule thresholds, tier derivation parameters,
sampling defaults, GPT call settings, rule-ID registry, and allowed-enum
validation sets.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Re-export shared thresholds (do NOT redefine them here)
# ---------------------------------------------------------------------------
from sim.runtime.semantic_thresholds import (  # noqa: F401
    DEF_ASPECT_DEG,
    DEF_RANGE_M,
    GRAVITY_MPS2,
    MERGE_RANGE_M,
    TARGET_SWITCH_RATIO,
)

# ---------------------------------------------------------------------------
# D2b-specific rule thresholds
# Source: annotation_guideline_v1.md
# ---------------------------------------------------------------------------

# Commit-intercept: bearing to target must be within this cone (deg).
COMMIT_MAX_BEARING_DEG: float = 30.0

# Commit-intercept: minimum slant range to qualify (m).
COMMIT_MIN_RANGE_M: float = 15_000.0

# Hold-geometry: absolute closure rate must stay below this value (m/s).
HOLD_MAX_CLOSURE_ABS_MPS: float = 20.0

# Extend: minimum absolute bearing-off angle to qualify (deg).
EXTEND_MIN_BEARING_ABS_DEG: float = 90.0

# Merge-entry: maximum range to flag as entering the merge (m).
MERGE_ENTRY_MAX_RANGE_M: float = 5_000.0

# Merge-entry: closure rate threshold for fast-approach check (m/s, negative = closing).
MERGE_ENTRY_FAST_CLOSURE_MPS: float = -100.0

# ---------------------------------------------------------------------------
# Tier derivation thresholds
# ---------------------------------------------------------------------------

# AI confidence >= this value qualifies for *silver* tier.
CONFIDENCE_SILVER_THRESHOLD: float = 0.80

# AI confidence >= this value is the floor for *bronze* tier.
CONFIDENCE_BRONZE_MIN: float = 0.50

# ---------------------------------------------------------------------------
# Hard-case detection
# ---------------------------------------------------------------------------

# Records with rule confidence below this value are flagged for GPT review.
LOW_CONFIDENCE_THRESHOLD: float = 0.50

# Support candidate: ally must be within this range (m) to consider support.
SUPPORT_ALLY_RANGE_M: float = 10_000.0

# Offensive candidate: enemy must be within this range (m).
OFFENSIVE_RANGE_M: float = 10_000.0

# Energy candidate: enemy must be farther than this range (m).
ENERGY_FAR_RANGE_M: float = 30_000.0

# Energy candidate: |closure| must be below this value (m/s).
ENERGY_LOW_CLOSURE_MPS: float = 30.0

# Press candidate: enemy range lower bound (m).
PRESS_RANGE_MIN_M: float = 5_000.0

# Press candidate: enemy range upper bound (m).
PRESS_RANGE_MAX_M: float = 15_000.0

# Press candidate: closure must be below (more negative than) this value (m/s).
PRESS_CLOSURE_THRESHOLD_MPS: float = -50.0

# ---------------------------------------------------------------------------
# Sampling defaults (for GPT review batch construction)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_SIZE: int = 500
DEFAULT_RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# GPT review configuration
# ---------------------------------------------------------------------------

GPT_MODEL_NAME: str = "gpt-5.4"
GPT_MAX_TOKENS: int = 800
GPT_TEMPERATURE: float = 0.3

# ---------------------------------------------------------------------------
# Rule-ID registry
# ---------------------------------------------------------------------------
# Maps human-readable rule names to canonical rule IDs used in audit logs
# and provenance records.

RULE_IDS: Dict[str, str] = {
    "commit":       "D2B_R01_commit",
    "merge_flag":   "D2B_R02_merge_flag",
    "merge_range":  "D2B_R03_merge_range",
    "extend":       "D2B_R04_extend",
    "hold":         "D2B_R05_hold",
    "no_enemy":     "D2B_R06_no_enemy",
    "insufficient": "D2B_R07_insufficient",
}

# ---------------------------------------------------------------------------
# Allowed enum sets (for validation against enum_registry_v1.json)
# ---------------------------------------------------------------------------

ALLOWED_SEMANTIC_STATES: FrozenSet[str] = frozenset({
    "commit_intercept",
    "hold_geometry",
    "press_attack",
    "extend",
    "support",
    "merge_entry",
    "offensive_turn",
    "defensive_break",
    "energy_manage",
})

ALLOWED_ROLES: FrozenSet[str] = frozenset({
    "engaged",
    "support",
    "egressing",
    "neutral",
})

ALLOWED_PROFILE_HINTS: FrozenSet[str] = frozenset({
    "p6dof_semantic",
    "p6dof_aggressive_turn",
    "auto",
})

ALLOWED_TARGET_REFS: FrozenSet[str] = frozenset({
    "enemy_1",
    "enemy_2",
    "ally_1",
})

ALLOWED_TIERS: FrozenSet[str] = frozenset({
    "gold",
    "silver",
    "bronze",
    "weak",
})

# ---------------------------------------------------------------------------
# Label boundary mode & merge protection (D2b dispute fix)
# ---------------------------------------------------------------------------

# "guideline_strict": commit/press boundary follows annotation_guideline_v1.md
# "doctrine_override_v1": allows commit_intercept at 8-15km (softer boundary)
LABEL_BOUNDARY_MODE: str = "guideline_strict"

# Rule labels in this set are protected from GPT override when rule confidence
# meets the minimum threshold below.
PROTECTED_RULE_LABELS: FrozenSet[str] = frozenset({"hold_geometry"})

# Minimum rule confidence required to protect a rule label from GPT override.
RULE_PROTECT_MIN_CONF: float = 0.80

# Guardrail: reject merge_entry when standoff geometry is detected.
MERGE_ENTRY_STANDOFF_GUARD: bool = True

# Standoff guard thresholds (used in label_guardrails.py)
STANDOFF_MAX_CLOSURE_ABS_MPS: float = 20.0
STANDOFF_MIN_BEARING_ABS_DEG: float = 60.0
STANDOFF_ASPECT_MIN_DEG: float = 60.0
STANDOFF_ASPECT_MAX_DEG: float = 120.0
