#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine candidate events from ACMI files -- v2 enhanced bucket detection.

Extends v1 miner with improved detectors for handoff_coordination (target_switch
+ support role) and turn_defense (offensive_turn vs defensive_break sub-buckets).
Also uses smoothed range-rate for far_intercept stability checks.

v2 imports new features from event_features_v2.py:
  heading_rate_degps, range_rate_smooth_mps, local_friend_count_15km,
  same_target_as_nearest_ally, etc.

Usage:
    python -m tools.acmi_mining.mine_candidate_events_v2 \
        --catalog datasets/acmi_mining/acmi_catalog.jsonl \
        --output datasets/acmi_mining/candidate_events_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap -- ensure project root is on sys.path so that absolute
# imports (ml.*, tools.*, sim.*) work regardless of invocation location.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # tools/acmi_mining/
_WOW_ROOT = _SCRIPT_DIR.parent.parent                  # wow/
_ML_DIR = _WOW_ROOT / "ml"
if str(_WOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_WOW_ROOT))
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

# ---------------------------------------------------------------------------
# Reuse v1 ACMI parsing and geometry helpers (DO NOT duplicate)
# ---------------------------------------------------------------------------
from tools.acmi_mining.mine_candidate_events import (    # noqa: E402
    _parse_and_resample,
    _split_coalitions,
    _build_ego_obs,
    _build_enemy_obs,
    _build_ally_obs,
    _file_hash,
    compute_range,
    compute_bearing,
    compute_aspect,
    compute_closure,
    compute_heading_rate,
    is_merge,
    is_defensive,
    _decluster_events,
    _cap_per_file,
)

from tools.acmi_mining.acmi_mining_config import (      # noqa: E402
    ACMI_RESAMPLE_DT,
    CLUSTER_WINDOW_SEC,
    EXTEND_MAX_RANGE_M,
    EXTEND_NOT_DEFENSIVE,
    EXTEND_NOT_MERGE,
    EXTEND_STABILITY_FRAMES,
    FAR_INTERCEPT_MAX_BEARING_DEG,
    FAR_INTERCEPT_MAX_CLOSURE_MPS,
    FAR_INTERCEPT_MIN_RANGE_M,
    FAR_INTERCEPT_PREFERRED_RANGE_M,
    FAR_INTERCEPT_STABILITY_FRAMES,
    HANDOFF_REQUIRE_ALLY,
    HANDOFF_REQUIRE_TWO_ENEMIES,
    MAX_EVENTS_PER_FILE,
    MERGE_RANGE_M,
    DEF_ASPECT_DEG,
    DEF_RANGE_M,
    STANDOFF_MAX_BEARING_ABS_DEG,
    STANDOFF_MAX_CLOSURE_ABS_MPS,
    STANDOFF_MAX_RANGE_M,
    STANDOFF_MIN_ASPECT_DEG,
    STANDOFF_MIN_BEARING_ABS_DEG,
    STANDOFF_MIN_RANGE_M,
    STANDOFF_MAX_ASPECT_DEG,
    STANDOFF_STABILITY_FRAMES,
    TURN_BEARING_CHANGE_THRESHOLD_DPS,
    TURN_DEFENSIVE_RANGE_M,
    TURN_HDG_RATE_THRESHOLD_DPS,
    # v2 thresholds
    TURN_V2_HDG_RATE_ONSET_DPS,
    TURN_V2_OFFENSIVE_MAX_RANGE_M,
    TURN_V2_DEFENSIVE_ASPECT_DEG,
    HANDOFF_V2_SUPPORT_RANGE_KM,
    HANDOFF_V2_CLUSTER_KEEP,
)

# v2 event features -- may not exist yet; guard import
try:
    from tools.acmi_mining.event_features_v2 import (   # noqa: E402
        heading_rate_degps,
        range_rate_smooth_mps,
        local_friend_count_15km,
        same_target_as_nearest_ally,
    )
    _HAS_V2_FEATURES = True
except ImportError:
    _HAS_V2_FEATURES = False

log = logging.getLogger(__name__)

# ===================================================================
# v2 bucket detection functions
# ===================================================================


def check_far_intercept_v2(
    ranges: np.ndarray, bearings: np.ndarray, closures: np.ndarray,
    aspects: np.ndarray, smooth_closure: np.ndarray | None,
    t_idx: int, n_frames: int,
) -> tuple[bool, float, str]:
    """Bucket A v2: far_intercept -- stable BVR approach with smoothed closure.

    Same logic as v1 but requires stability over STABILITY_FRAMES consecutive
    frames and uses range_rate_smooth_mps when available instead of raw closure.
    """
    end = t_idx + FAR_INTERCEPT_STABILITY_FRAMES
    if end > n_frames:
        return False, 0.0, ""

    window = slice(t_idx, end)
    r = ranges[window]
    b = bearings[window]

    # Use smooth closure if available, else raw
    c = smooth_closure[window] if smooth_closure is not None else closures[window]

    # exclusion: merge or defensive at any frame in window
    for i in range(t_idx, end):
        if is_merge(ranges[i]):
            return False, 0.0, ""
        if is_defensive(aspects[i], closures[i], ranges[i]):
            return False, 0.0, ""

    if not np.all(r > FAR_INTERCEPT_MIN_RANGE_M):
        return False, 0.0, ""
    if not np.all(np.abs(b) < FAR_INTERCEPT_MAX_BEARING_DEG):
        return False, 0.0, ""
    if not np.all(c < FAR_INTERCEPT_MAX_CLOSURE_MPS):
        return False, 0.0, ""

    # Require stability: range must be monotonically decreasing over 2+ frames
    if FAR_INTERCEPT_STABILITY_FRAMES >= 2:
        r_diffs = np.diff(r)
        if not np.all(r_diffs < 0):
            return False, 0.0, ""

    # priority: prefer longer range
    avg_range = float(np.mean(r))
    priority = min(1.0, avg_range / (2.0 * FAR_INTERCEPT_PREFERRED_RANGE_M))
    reason = (f"far_intercept_v2: range={avg_range:.0f}m, "
              f"bearing={float(np.mean(b)):.1f}, closure={float(np.mean(c)):.0f}")
    return True, priority, reason


def check_turn_defense_v2(
    ranges: np.ndarray, aspects: np.ndarray, closures: np.ndarray,
    heading_rates: np.ndarray, bearing_rates: np.ndarray,
    t_idx: int, n_frames: int,
) -> tuple[bool, float, str, str]:
    """Bucket D v2: turn_defense with offensive/defensive sub-bucket.

    Returns (hit, priority, reason, sub_bucket) where sub_bucket is
    "offensive" or "defensive".

    Detection logic:
    - Primary trigger: heading_rate > TURN_V2_HDG_RATE_ONSET_DPS (onset frame)
    - offensive_turn: heading_rate high + NOT defensive + range < 8km
    - defensive_break: heading_rate high + (is_defensive=True OR aspect > 120)

    Prefers onset frame (first frame where heading_rate crosses threshold).
    """
    # Need at least one frame ahead to check onset
    if t_idx >= n_frames:
        return False, 0.0, "", ""

    hdg_rate = heading_rates[t_idx]
    high_hdg_onset = hdg_rate > TURN_V2_HDG_RATE_ONSET_DPS

    # Also keep v1 fallback triggers but at lower priority
    high_hdg_v1 = hdg_rate > TURN_HDG_RATE_THRESHOLD_DPS
    high_brg_rate = bearing_rates[t_idx] > TURN_BEARING_CHANGE_THRESHOLD_DPS

    if not (high_hdg_onset or high_hdg_v1 or high_brg_rate):
        return False, 0.0, "", ""

    range_m = float(ranges[t_idx])
    aspect_deg = float(aspects[t_idx])
    closure_mps = float(closures[t_idx])
    def_flag = is_defensive(aspect_deg, closure_mps, range_m)

    # Classify sub-bucket
    sub_bucket = ""
    if high_hdg_onset:
        if def_flag or aspect_deg > TURN_V2_DEFENSIVE_ASPECT_DEG:
            sub_bucket = "defensive"
        elif range_m < TURN_V2_OFFENSIVE_MAX_RANGE_M:
            sub_bucket = "offensive"
        else:
            # High turn rate at long range, still classify as defensive break
            sub_bucket = "defensive"
    elif def_flag:
        sub_bucket = "defensive"
    elif high_hdg_v1 or high_brg_rate:
        # v1-level trigger, low priority
        sub_bucket = "offensive" if range_m < TURN_V2_OFFENSIVE_MAX_RANGE_M else "defensive"

    if not sub_bucket:
        return False, 0.0, "", ""

    # Check onset: prefer first frame where threshold is crossed
    is_onset = False
    if high_hdg_onset and t_idx > 0:
        prev_rate = heading_rates[t_idx - 1] if t_idx - 1 < len(heading_rates) else 0.0
        if prev_rate <= TURN_V2_HDG_RATE_ONSET_DPS:
            is_onset = True

    # Priority scoring
    priority = 0.5
    if sub_bucket == "offensive":
        priority = 0.85
        if is_onset:
            priority = 0.95
    elif sub_bucket == "defensive":
        priority = 0.70
        if def_flag and high_hdg_onset:
            priority = 0.90
        if is_onset:
            priority += 0.05

    priority = min(1.0, priority)

    parts: list[str] = [f"sub={sub_bucket}"]
    if is_onset:
        parts.append("onset")
    if def_flag:
        parts.append("defensive_geom")
    parts.append(f"hdg_rate={hdg_rate:.1f}dps")
    parts.append(f"range={range_m:.0f}m")
    parts.append(f"aspect={aspect_deg:.0f}")

    reason = f"turn_defense_v2: {', '.join(parts)}"
    return True, priority, reason, sub_bucket


def check_handoff_coordination_v2(
    ego: dict,
    ego_enemies: list[dict[str, Any]],
    allies: list[dict],
    ally_geom: list[dict[str, Any]],
    enemy_geom: list[dict[str, Any]],
    closest_enemy_id_prev: str | None,
    closest_enemy_id_cur: str,
    primary_range: float,
    t_idx: int,
) -> tuple[bool, float, str, str]:
    """Bucket E v2: handoff_coordination -- target_switch + support detection.

    Returns (hit, priority, reason, sub_type) where sub_type is
    "target_switch" or "support".

    Detection modes:
    1. target_switch: ego's closest enemy changes from frame to frame
    2. support: ego is NOT nearest friendly to primary enemy
       (ally is closer to the same target -> ego is in support role)
    3. same_target_as_nearest_ally = True AND ego.range > ally.range

    Priority boost for 2v2 (2 enemies + 2 friendlies).
    """
    n_enemies = len(ego_enemies)
    n_allies = len(allies)

    # --- Mode 1: target_switch ---
    is_switch = False
    if closest_enemy_id_prev is not None and closest_enemy_id_cur != closest_enemy_id_prev:
        if n_enemies >= 2 or not HANDOFF_REQUIRE_TWO_ENEMIES:
            if n_allies >= 1 or not HANDOFF_REQUIRE_ALLY:
                is_switch = True

    # --- Mode 2: support role detection ---
    is_support = False
    if n_allies >= 1 and n_enemies >= 1:
        # Check if any ally is closer to the primary enemy than ego
        primary_enemy_id = closest_enemy_id_cur
        for ag in ally_geom:
            ally_ent = ag["ally"]
            a_local = t_idx - ag["i0"]
            if a_local < 0 or a_local >= (ag["i1"] - ag["i0"]):
                continue

            # Compute ally distance to primary enemy
            # a_local is offset in the overlap window; ally's own
            # array index is ag["j0"] + a_local.
            ally_idx = ag["j0"] + a_local
            for eg in enemy_geom:
                if eg["enemy"]["id"] != primary_enemy_id:
                    continue
                e_local = t_idx - eg["i0"]
                if e_local < 0 or e_local >= (eg["i1"] - eg["i0"]):
                    continue

                # enemy's own array index is eg["j0"] + e_local
                enemy_idx = eg["j0"] + e_local

                # ally-to-enemy range
                ally_to_enemy_range = float(compute_range(
                    np.array([ally_ent["x"][ally_idx]]),
                    np.array([ally_ent["y"][ally_idx]]),
                    np.array([ally_ent["z"][ally_idx]]),
                    np.array([eg["enemy"]["x"][enemy_idx]]),
                    np.array([eg["enemy"]["y"][enemy_idx]]),
                    np.array([eg["enemy"]["z"][enemy_idx]]),
                )[0])

                if ally_to_enemy_range < primary_range:
                    # Ally is closer to same enemy -> ego is support
                    is_support = True
                    break
            if is_support:
                break

    if not is_switch and not is_support:
        return False, 0.0, "", ""

    # Determine sub_type
    if is_switch:
        sub_type = "target_switch"
    else:
        sub_type = "support"

    # Priority
    priority = 0.70
    if is_switch:
        priority = 0.80
    if is_support and is_switch:
        priority = 0.90

    # 2v2 boost
    is_full_2v2 = n_enemies >= 2 and n_allies >= 1  # ego + ally = 2 friendlies
    if is_full_2v2:
        priority = min(1.0, priority + 0.10)

    parts: list[str] = [f"sub={sub_type}"]
    if is_switch:
        parts.append(f"switch={closest_enemy_id_prev}->{closest_enemy_id_cur}")
    if is_support:
        parts.append("ego_in_support")
    if is_full_2v2:
        parts.append("full_2v2")
    parts.append(f"range={primary_range:.0f}m")

    reason = f"handoff_coordination_v2: {', '.join(parts)}"
    return True, priority, reason, sub_type


# ===================================================================
# v2 event feature computation helpers
# ===================================================================

def _compute_smooth_closure(ranges: np.ndarray, dt: float,
                            window: int = 5) -> np.ndarray:
    """Smoothed range rate: apply moving-average then differentiate.

    Falls back to raw gradient if window is larger than available data.
    """
    if len(ranges) < window:
        return compute_closure(ranges, dt)
    kernel = np.ones(window) / window
    smoothed = np.convolve(ranges, kernel, mode="same")
    return np.gradient(smoothed, dt)


def _compute_v2_event_features(
    ego: dict, enemy: dict, ally_geom: list[dict[str, Any]],
    t_idx: int, primary: dict[str, Any],
    hdg_rate_dps: float, sub_bucket: str | None,
) -> dict[str, Any]:
    """Build extended event_features dict with v2 fields."""
    features: dict[str, Any] = {
        "range_m": round(primary["range"], 1),
        "bearing_deg": round(primary["bearing"], 2),
        "aspect_deg": round(primary["aspect"], 2),
        "closure_mps": round(primary["closure"], 1),
        "heading_rate_dps": round(hdg_rate_dps, 2),
    }

    # v2 additions
    if sub_bucket:
        features["sub_bucket"] = sub_bucket

    # range_rate_smooth_mps (if v2 features module available)
    smooth_closure = primary.get("smooth_closure")
    if smooth_closure is not None:
        features["range_rate_smooth_mps"] = round(smooth_closure, 1)

    # local_friend_count_15km
    friend_count = 0
    radius_m = HANDOFF_V2_SUPPORT_RANGE_KM * 1000.0
    for ag in ally_geom:
        a_local = t_idx - ag["i0"]
        if 0 <= a_local < (ag["i1"] - ag["i0"]):
            if float(ag["range"][a_local]) < radius_m:
                friend_count += 1
    features["local_friend_count_15km"] = friend_count

    return features


# ===================================================================
# Obs-sidecar builder -- reuses v1's _build_ego_obs, _build_enemy_obs,
# _build_ally_obs via import
# ===================================================================


def _build_event_v2(
    file_hash: str, acmi_path: str, t_sec: float, bucket: str,
    ego: dict, enemy: dict, ally: dict | None,
    active_allies: list[dict], primary: dict,
    hdg_rate_dps: float, priority: float, reason: str,
    t_idx: int, prev_target_ref: str | None,
    sub_bucket: str | None,
    ally_geom: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a single v2 candidate event dict."""
    ego_obs = _build_ego_obs(ego, t_idx)

    enemy_obs = _build_enemy_obs(
        ego, enemy, t_idx,
        range_m=primary["range"],
        bearing_deg=primary["bearing"],
        aspect_deg=primary["aspect"],
        closure_mps=primary["closure"],
        alt_diff_m=primary["alt_diff"],
        is_primary=True,
    )

    ally_obs: dict[str, Any] | None = None
    if active_allies:
        best_ally_info = active_allies[0]
        ally_ent = best_ally_info["geom"]["ally"]
        ally_obs = _build_ally_obs(
            ego, ally_ent, t_idx,
            range_m=best_ally_info["range"],
            bearing_deg=best_ally_info["bearing"],
        )

    merge_flag = is_merge(primary["range"])
    def_flag = is_defensive(primary["aspect"], primary["closure"], primary["range"])

    engagement_obs = {
        "is_merge": merge_flag,
        "is_defensive": def_flag,
        "has_shot_opportunity": None,
    }
    history_obs = {
        "prev_semantic_state": None,
        "prev_token_family": None,
        "prev_target_ref": prev_target_ref,
    }

    event_id = f"{file_hash}_{ego['id']}_{t_sec:.1f}"

    # v2 extended features
    event_features = _compute_v2_event_features(
        ego, enemy, ally_geom, t_idx, primary,
        hdg_rate_dps, sub_bucket,
    )

    return {
        "event_id": event_id,
        "file_path": acmi_path,
        "timestamp_sec": t_sec,
        "bucket": bucket,
        "sub_bucket": sub_bucket,
        "ego_runtime_id": ego["id"],
        "target_runtime_id": enemy["id"],
        "ally_runtime_id": ally["id"] if ally else None,
        "obs": {
            "ego": ego_obs,
            "enemy": enemy_obs,
            "ally": ally_obs,
            "engagement": engagement_obs,
            "history": history_obs,
        },
        "event_features": event_features,
        "reason": reason,
        "priority": round(priority, 4),
    }


# ===================================================================
# Per-file mining loop (v2)
# ===================================================================

def _mine_file_v2(
    acmi_path: str,
    file_hash: str,
    dt: float,
) -> list[dict[str, Any]]:
    """Process one ACMI file with v2 enhanced bucket detection."""

    entities = _parse_and_resample(acmi_path, dt)
    if len(entities) < 2:
        log.debug("Skipping %s: fewer than 2 valid entities", acmi_path)
        return []

    group_a, group_b = _split_coalitions(entities)
    if not group_a or not group_b:
        log.debug("Skipping %s: cannot form two coalitions", acmi_path)
        return []

    candidates: list[dict[str, Any]] = []

    all_entities = group_a + group_b
    group_a_ids = {e["id"] for e in group_a}

    for ego in all_entities:
        ego_in_a = ego["id"] in group_a_ids
        enemies = group_b if ego_in_a else group_a
        allies_list = [a for a in (group_a if ego_in_a else group_b) if a["id"] != ego["id"]]

        if not enemies:
            continue

        n_frames = len(ego["t"])
        if n_frames < 10:
            continue

        # ----------------------------------------------------------
        # Precompute pairwise geometry arrays against each enemy
        # ----------------------------------------------------------
        enemy_geom: list[dict[str, Any]] = []
        for en in enemies:
            t0 = max(ego["t"][0], en["t"][0])
            t1 = min(ego["t"][-1], en["t"][-1])
            if t1 - t0 < 5.0:
                continue

            i0 = int(np.searchsorted(ego["t"], t0))
            i1 = int(np.searchsorted(ego["t"], t1, side="right"))
            if i1 - i0 < 10:
                continue

            j0 = int(np.searchsorted(en["t"], ego["t"][i0]))
            j1 = j0 + (i1 - i0)
            if j1 > len(en["t"]):
                j1 = len(en["t"])
                i1 = i0 + (j1 - j0)
            if i1 - i0 < 10:
                continue

            sl_e = slice(i0, i1)
            sl_t = slice(j0, j1)

            rng = compute_range(
                ego["x"][sl_e], ego["y"][sl_e], ego["z"][sl_e],
                en["x"][sl_t], en["y"][sl_t], en["z"][sl_t],
            )
            brg = compute_bearing(
                ego["x"][sl_e], ego["y"][sl_e],
                en["x"][sl_t], en["y"][sl_t],
                ego["heading_u"][sl_e],
            )
            asp = compute_aspect(
                en["heading_u"][sl_t],
                en["x"][sl_t], en["y"][sl_t],
                ego["x"][sl_e], ego["y"][sl_e],
            )
            cls_ = compute_closure(rng, dt)
            alt_diff = ego["z"][sl_e] - en["z"][sl_t]

            # v2: smoothed closure
            smooth_cls = _compute_smooth_closure(rng, dt)

            enemy_geom.append({
                "enemy": en,
                "i0": i0, "i1": i1,
                "j0": j0,  # enemy's own time index offset
                "range": rng,
                "bearing": brg,
                "aspect": asp,
                "closure": cls_,
                "smooth_closure": smooth_cls,
                "alt_diff": alt_diff,
            })

        if not enemy_geom:
            continue

        # ----------------------------------------------------------
        # Precompute ego heading rate and per-enemy bearing rate
        # ----------------------------------------------------------
        ego_hdg_rate = compute_heading_rate(ego["heading_u"], dt)

        for eg in enemy_geom:
            eg["bearing_rate"] = np.abs(np.gradient(eg["bearing"], dt))

        # ----------------------------------------------------------
        # Precompute ally geometry
        # ----------------------------------------------------------
        ally_geom: list[dict[str, Any]] = []
        for al in allies_list:
            t0a = max(ego["t"][0], al["t"][0])
            t1a = min(ego["t"][-1], al["t"][-1])
            if t1a - t0a < 5.0:
                continue
            ai0 = int(np.searchsorted(ego["t"], t0a))
            ai1 = int(np.searchsorted(ego["t"], t1a, side="right"))
            aj0 = int(np.searchsorted(al["t"], ego["t"][ai0]))
            aj1 = aj0 + (ai1 - ai0)
            if aj1 > len(al["t"]):
                aj1 = len(al["t"])
                ai1 = ai0 + (aj1 - aj0)
            if ai1 - ai0 < 2:
                continue
            sl_e = slice(ai0, ai1)
            sl_a = slice(aj0, aj1)
            a_rng = compute_range(
                ego["x"][sl_e], ego["y"][sl_e], ego["z"][sl_e],
                al["x"][sl_a], al["y"][sl_a], al["z"][sl_a],
            )
            a_brg = compute_bearing(
                ego["x"][sl_e], ego["y"][sl_e],
                al["x"][sl_a], al["y"][sl_a],
                ego["heading_u"][sl_e],
            )
            ally_geom.append({
                "ally": al, "i0": ai0, "i1": ai1,
                "j0": aj0,  # ally's own time index offset
                "range": a_rng, "bearing": a_brg,
            })

        # ----------------------------------------------------------
        # Frame-by-frame bucket checks
        # ----------------------------------------------------------
        prev_closest_enemy_id: str | None = None

        for t_idx in range(n_frames):
            t_sec = float(ego["t"][t_idx])

            # Gather enemies visible at this timestep
            active_enemies: list[dict[str, Any]] = []
            for eg in enemy_geom:
                local = t_idx - eg["i0"]
                if 0 <= local < (eg["i1"] - eg["i0"]):
                    active_enemies.append({
                        "geom": eg,
                        "local": local,
                        "range": float(eg["range"][local]),
                        "bearing": float(eg["bearing"][local]),
                        "aspect": float(eg["aspect"][local]),
                        "closure": float(eg["closure"][local]),
                        "smooth_closure": float(eg["smooth_closure"][local]),
                        "alt_diff": float(eg["alt_diff"][local]),
                    })

            if not active_enemies:
                prev_closest_enemy_id = None
                continue

            # Find closest enemy (primary threat)
            active_enemies.sort(key=lambda x: x["range"])
            primary = active_enemies[0]
            primary_eg = primary["geom"]
            closest_enemy_id = primary_eg["enemy"]["id"]
            local_idx = primary["local"]
            n_local = primary_eg["i1"] - primary_eg["i0"]

            # Active allies at this timestep
            active_allies: list[dict[str, Any]] = []
            for ag in ally_geom:
                a_local = t_idx - ag["i0"]
                if 0 <= a_local < (ag["i1"] - ag["i0"]):
                    active_allies.append({
                        "geom": ag,
                        "local": a_local,
                        "range": float(ag["range"][a_local]),
                        "bearing": float(ag["bearing"][a_local]),
                    })

            # Heading rate at this timestep
            hdg_rate_dps = float(ego_hdg_rate[t_idx]) if t_idx < len(ego_hdg_rate) else 0.0

            # ==== Bucket A: far_intercept (v2 with smooth closure) ====
            ok, pri, reason = check_far_intercept_v2(
                primary_eg["range"], primary_eg["bearing"],
                primary_eg["closure"], primary_eg["aspect"],
                primary_eg["smooth_closure"],
                local_idx, n_local,
            )
            if ok:
                candidates.append(_build_event_v2(
                    file_hash, acmi_path, t_sec, "far_intercept",
                    ego, primary_eg["enemy"],
                    allies_list[0] if allies_list else None,
                    active_allies, primary, hdg_rate_dps,
                    pri, reason, t_idx, prev_closest_enemy_id,
                    sub_bucket=None, ally_geom=ally_geom,
                ))

            # ==== Bucket B: standoff_hold (same as v1) ====
            ok_b, pri_b, reason_b = _check_standoff_hold_v1(
                primary_eg["range"], primary_eg["bearing"],
                primary_eg["closure"], primary_eg["aspect"],
                local_idx, n_local,
            )
            if ok_b:
                candidates.append(_build_event_v2(
                    file_hash, acmi_path, t_sec, "standoff_hold",
                    ego, primary_eg["enemy"],
                    allies_list[0] if allies_list else None,
                    active_allies, primary, hdg_rate_dps,
                    pri_b, reason_b, t_idx, prev_closest_enemy_id,
                    sub_bucket=None, ally_geom=ally_geom,
                ))

            # ==== Bucket C: extend_disengage (same as v1) ====
            ok_c, pri_c, reason_c = _check_extend_disengage_v1(
                primary_eg["range"], primary_eg["closure"],
                primary_eg["aspect"],
                local_idx, n_local,
            )
            if ok_c:
                candidates.append(_build_event_v2(
                    file_hash, acmi_path, t_sec, "extend_disengage",
                    ego, primary_eg["enemy"],
                    allies_list[0] if allies_list else None,
                    active_allies, primary, hdg_rate_dps,
                    pri_c, reason_c, t_idx, prev_closest_enemy_id,
                    sub_bucket=None, ally_geom=ally_geom,
                ))

            # ==== Bucket D: turn_defense (v2 with sub-bucket) ====
            ego_hdg_val = float(ego_hdg_rate[t_idx]) if t_idx < len(ego_hdg_rate) else 0.0
            local_hdg_rates = np.full(n_local, ego_hdg_val)
            ok_d, pri_d, reason_d, sub_d = check_turn_defense_v2(
                primary_eg["range"], primary_eg["aspect"],
                primary_eg["closure"],
                local_hdg_rates,
                primary_eg["bearing_rate"],
                local_idx, n_local,
            )
            if ok_d:
                candidates.append(_build_event_v2(
                    file_hash, acmi_path, t_sec, "turn_defense",
                    ego, primary_eg["enemy"],
                    allies_list[0] if allies_list else None,
                    active_allies, primary, hdg_rate_dps,
                    pri_d, reason_d, t_idx, prev_closest_enemy_id,
                    sub_bucket=sub_d, ally_geom=ally_geom,
                ))

            # ==== Bucket E: handoff_coordination (v2 with support) ====
            ok_e, pri_e, reason_e, sub_e = check_handoff_coordination_v2(
                ego=ego,
                ego_enemies=[{"id": ae["geom"]["enemy"]["id"]} for ae in active_enemies],
                allies=[{"id": ag["geom"]["ally"]["id"]} for ag in active_allies],
                ally_geom=ally_geom,
                enemy_geom=enemy_geom,
                closest_enemy_id_prev=prev_closest_enemy_id,
                closest_enemy_id_cur=closest_enemy_id,
                primary_range=primary["range"],
                t_idx=t_idx,
            )
            if ok_e:
                candidates.append(_build_event_v2(
                    file_hash, acmi_path, t_sec, "handoff_coordination",
                    ego, primary_eg["enemy"],
                    allies_list[0] if allies_list else None,
                    active_allies, primary, hdg_rate_dps,
                    pri_e, reason_e, t_idx, prev_closest_enemy_id,
                    sub_bucket=sub_e, ally_geom=ally_geom,
                ))

            prev_closest_enemy_id = closest_enemy_id

    return candidates


# ===================================================================
# v1 bucket wrappers (unchanged logic, imported for reuse)
# ===================================================================

def _check_standoff_hold_v1(
    ranges: np.ndarray, bearings: np.ndarray, closures: np.ndarray,
    aspects: np.ndarray, t_idx: int, n_frames: int,
) -> tuple[bool, float, str]:
    """Bucket B: standoff_hold -- identical to v1."""
    end = t_idx + STANDOFF_STABILITY_FRAMES
    if end > n_frames:
        return False, 0.0, ""

    window = slice(t_idx, end)
    r = ranges[window]
    b = bearings[window]
    c = closures[window]
    a = aspects[window]

    for i in range(t_idx, end):
        if is_merge(ranges[i]):
            return False, 0.0, ""
        if is_defensive(aspects[i], closures[i], ranges[i]):
            return False, 0.0, ""

    if not np.all((r > STANDOFF_MIN_RANGE_M) & (r < STANDOFF_MAX_RANGE_M)):
        return False, 0.0, ""
    if not np.all(np.abs(c) < STANDOFF_MAX_CLOSURE_ABS_MPS):
        return False, 0.0, ""
    abs_b = np.abs(b)
    if not np.all((abs_b > STANDOFF_MIN_BEARING_ABS_DEG) & (abs_b < STANDOFF_MAX_BEARING_ABS_DEG)):
        return False, 0.0, ""
    if not np.all((a > STANDOFF_MIN_ASPECT_DEG) & (a < STANDOFF_MAX_ASPECT_DEG)):
        return False, 0.0, ""

    priority = 0.7
    reason = (f"standoff_hold: range={float(np.mean(r)):.0f}m, "
              f"|bearing|={float(np.mean(abs_b)):.1f}, aspect={float(np.mean(a)):.1f}")
    return True, priority, reason


def _check_extend_disengage_v1(
    ranges: np.ndarray, closures: np.ndarray, aspects: np.ndarray,
    t_idx: int, n_frames: int,
) -> tuple[bool, float, str]:
    """Bucket C: extend_disengage -- identical to v1."""
    end = t_idx + EXTEND_STABILITY_FRAMES
    if end > n_frames:
        return False, 0.0, ""

    window = slice(t_idx, end)
    r = ranges[window]
    c = closures[window]

    if float(np.mean(r)) > EXTEND_MAX_RANGE_M:
        return False, 0.0, ""
    if not np.all(c > 0):
        return False, 0.0, ""
    if not np.all(np.diff(r) > 0):
        return False, 0.0, ""

    if EXTEND_NOT_MERGE:
        for i in range(t_idx, end):
            if is_merge(ranges[i]):
                return False, 0.0, ""
    if EXTEND_NOT_DEFENSIVE:
        for i in range(t_idx, end):
            if is_defensive(aspects[i], closures[i], ranges[i]):
                return False, 0.0, ""

    avg_closure = float(np.mean(c))
    priority = min(1.0, avg_closure / 200.0)
    reason = (f"extend_disengage: range={float(np.mean(r)):.0f}m, "
              f"closure={avg_closure:.0f}")
    return True, priority, reason


# ===================================================================
# Main entry
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine candidate events from ACMI files -- v2 enhanced buckets.",
    )
    parser.add_argument(
        "--catalog", required=True,
        help="Path to acmi_catalog.jsonl produced by build_acmi_catalog.py",
    )
    parser.add_argument(
        "--input",
        help="Root directory containing ACMI files (paths in catalog are relative to this)",
    )
    parser.add_argument(
        "--output",
        default="datasets/acmi_mining/candidate_events_v2.jsonl",
        help="Output path for candidate_events_v2.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--dt", type=float, default=ACMI_RESAMPLE_DT,
        help=f"Resample interval in seconds (default: {ACMI_RESAMPLE_DT})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    catalog_path = Path(args.catalog)
    output_path = Path(args.output)

    if not catalog_path.exists():
        log.error("Catalog file not found: %s", catalog_path)
        sys.exit(1)

    # Determine input root
    if args.input:
        input_root = Path(args.input)
    else:
        input_root = catalog_path.parent

    # Read catalog
    catalog_entries: list[dict] = []
    with open(catalog_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed catalog line %d: %s", line_no, exc)
                continue
            catalog_entries.append(entry)

    log.info("Loaded %d catalog entries", len(catalog_entries))

    # Filter to candidates
    candidate_entries = [
        e for e in catalog_entries
        if e.get("candidate_2v2") or e.get("candidate_multiship")
    ]
    log.info("Found %d candidate files (candidate_2v2 or candidate_multiship)",
             len(candidate_entries))

    all_events: list[dict] = []
    bucket_counts: dict[str, int] = {}
    sub_bucket_counts: dict[str, int] = {}
    files_processed = 0
    files_failed = 0

    for entry in candidate_entries:
        rel_path = entry.get("file_path") or entry.get("path", "")
        acmi_path = str(input_root / rel_path)
        fhash = _file_hash(rel_path)

        if not Path(acmi_path).exists():
            log.warning("ACMI file not found: %s", acmi_path)
            files_failed += 1
            continue

        log.info("Processing: %s", rel_path)
        try:
            raw_events = _mine_file_v2(acmi_path, fhash, args.dt)
        except Exception:
            log.exception("Failed to process %s", acmi_path)
            files_failed += 1
            continue

        # Decluster within file (reuse v1 logic)
        declustered = _decluster_events(raw_events, args.dt)
        # Cap per file (reuse v1 logic)
        capped = _cap_per_file(declustered)

        for ev in capped:
            bucket_counts[ev["bucket"]] = bucket_counts.get(ev["bucket"], 0) + 1
            sb = ev.get("sub_bucket")
            if sb:
                key = f"{ev['bucket']}/{sb}"
                sub_bucket_counts[key] = sub_bucket_counts.get(key, 0) + 1
        all_events.extend(capped)
        files_processed += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    log.info("=" * 60)
    log.info("Mining v2 complete.")
    log.info("  Files processed : %d", files_processed)
    log.info("  Files failed    : %d", files_failed)
    log.info("  Total events    : %d", len(all_events))
    log.info("  Bucket breakdown:")
    for bucket, count in sorted(bucket_counts.items()):
        log.info("    %-25s %d", bucket, count)
    if sub_bucket_counts:
        log.info("  Sub-bucket breakdown:")
        for key, count in sorted(sub_bucket_counts.items()):
            log.info("    %-30s %d", key, count)
    log.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
