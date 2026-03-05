#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_default_config.py — Frozen, validated training-pipeline configuration
===============================================================================
This file is the **single source of truth** for all dataset hyperparameters,
quality-detection thresholds, and entity-filter settings.

Validated against AFSIM TurnToRelativeHeading (dpsi_sign = +1,
heading = atan2(vx_east, vy_north), 0 = North, CW = positive).

Any script in the pipeline (acmi_to_dt_dataset_smooth.py,
build_training_dataset.py, analyze_acmi_coverage.py) imports from here
and uses these values as argparse defaults. CLI arguments always win.

DO NOT change values here without re-running the full coverage analysis and
verifying the heading-turn alignment report.
"""

# ── Temporal resolution ───────────────────────────────────────────────────────
DT      = 0.5   # seconds  — decision rate = 2 Hz
H_SEC   = 2.0   # seconds  — setpoint horizon
K       = 4     # horizon steps = int(round(H_SEC / DT))   [derived, do not change independently]

# ── Sequence / window parameters ──────────────────────────────────────────────
SEQ_LEN = 20    # steps per training window
STRIDE  = 5     # stride between windows

# ── Physical filters ──────────────────────────────────────────────────────────
MIN_SPEED   = 60.0   # m/s  — discard taxi / low-energy samples
MAX_SPEED   = 800.0  # m/s  — hard ceiling; above → corrupted track (≈ Mach 2.4)
TYPE_FILTER = "Air+FixedWing"  # Tacview entity Type substring

# ── Outlier detection ─────────────────────────────────────────────────────────
ACCEL_THRESH = 150.0  # m/s²  — flag acceleration spike between adjacent steps
SPEED_JUMP   = 200.0  # m/s   — flag sudden speed jump between adjacent steps

# ── Sign conventions (verified against AFSIM) ────────────────────────────────
# CW turn → positive dpsi → positive action[0]
# AFSIM TurnToRelativeHeading takes DEGREES; convert rad→deg before sending.
DPSI_SIGN = 1.0   # +1: keep sign  |  −1: flip sign

# ── Heading convention ────────────────────────────────────────────────────────
HEADING_FRAME        = "ENU"
HEADING_ZERO_REF     = "North"
HEADING_POSITIVE_DIR = "clockwise"
HEADING_ATAN2_ORDER  = "atan2(vx_east, vy_north)"

# ── Observation / action field names ─────────────────────────────────────────
OBS_FIELDS = [
    "x_e_m", "y_n_m", "z_u_m",
    "vx_e_mps", "vy_n_mps", "vz_u_mps",
    "heading_rad_unwrapped", "ground_speed_mps",
]
ACT_FIELDS = ["dpsi_rad", "alt_sp_m", "spd_sp_mps"]

# ── Entity filter — name/type regex ──────────────────────────────────────────
# These are imported from entity_filter.py to maintain a single source of truth.
# Override via CLI --exclude_regex / --include_regex.
INCLUDE_REGEX = None   # None = no positive filter (accept all passing exclude check)

# Lazy import to avoid circular dependency issues at module level.
# Use get_exclude_regex() in code that needs the string.
def get_exclude_regex() -> str:
    from entity_filter import DEFAULT_EXCLUDE_REGEX
    return DEFAULT_EXCLUDE_REGEX

# ── Entity filter — orbiter behaviour detection ───────────────────────────────
# Entities that fly for ≥ORB_MIN_STEPS steps at near-constant alt/speed with
# very low heading change are classified as non-combat orbiters (AWACS,
# tankers, patrol circuits) and excluded from training.
EXCLUDE_ORBITERS  = True   # master switch
ORB_ALT_STD_M     = 80.0   # m      — max std(altitude)  to call orbiter
ORB_SPD_STD_MPS   = 7.0    # m/s    — max std(speed)     to call orbiter
ORB_DPSI_P95_RAD  = 0.05   # rad    — max p95(|Δψ|)      to call orbiter
ORB_MIN_STEPS     = 600    # steps  — min track length before heuristic fires
                            #          (= 300 s at 2 Hz; shorter tracks not judged)

# ── Default output ────────────────────────────────────────────────────────────
DEFAULT_DATASET_STEM = "dt2hz_H2s_fighteronly"   # base name; extensions appended
