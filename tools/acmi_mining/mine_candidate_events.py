#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine candidate events from ACMI files for 5 semantic buckets.

Reads an ACMI catalog (JSONL), parses each candidate file, computes
per-timestep engagement geometry, and emits candidate events to JSONL.

Usage:
    python -m tools.acmi_mining.mine_candidate_events \
        --catalog datasets/acmi_mining/acmi_catalog.jsonl \
        --input tra_data \
        --output datasets/acmi_mining/candidate_events.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
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
# Project imports
# ---------------------------------------------------------------------------
from ml.acmi_to_dt_dataset_smooth import (              # noqa: E402
    build_entity_tracks,
    parse_header,
    parse_object_updates,
    read_acmi_text,
    resample_entity,
)
from tools.acmi_mining.acmi_cartesian_parser import (   # noqa: E402
    parse_acmi_with_cartesian,
)
from tools.acmi_mining.acmi_mining_config import (      # noqa: E402
    ACMI_FIXED_WING_FILTER,
    ACMI_MAX_SPEED_MPS,
    ACMI_MIN_SPEED_MPS,
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
)

log = logging.getLogger(__name__)

# ===================================================================
# Geometry helpers
# ===================================================================

def compute_range(ego_x: np.ndarray, ego_y: np.ndarray, ego_z: np.ndarray,
                  tgt_x: np.ndarray, tgt_y: np.ndarray, tgt_z: np.ndarray) -> np.ndarray:
    """3-D slant range in metres."""
    dx = tgt_x - ego_x
    dy = tgt_y - ego_y
    dz = tgt_z - ego_z
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def compute_bearing(ego_x: np.ndarray, ego_y: np.ndarray,
                    tgt_x: np.ndarray, tgt_y: np.ndarray,
                    ego_heading_rad: np.ndarray) -> np.ndarray:
    """Bearing from ego heading to target, +right, [-180, 180] degrees."""
    dx = tgt_x - ego_x
    dy = tgt_y - ego_y
    los_angle = np.arctan2(dx, dy)          # 0=north, CW+
    bearing = los_angle - ego_heading_rad
    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(bearing)


def compute_aspect(tgt_heading_rad: np.ndarray,
                   tgt_x: np.ndarray, tgt_y: np.ndarray,
                   ego_x: np.ndarray, ego_y: np.ndarray) -> np.ndarray:
    """Aspect angle: 0=head-on, 180=tail (unsigned, degrees)."""
    dx = ego_x - tgt_x                     # target-to-ego
    dy = ego_y - tgt_y
    los_angle = np.arctan2(dx, dy)
    aspect = los_angle - tgt_heading_rad
    aspect = (aspect + np.pi) % (2 * np.pi) - np.pi
    return np.abs(np.degrees(aspect))


def compute_closure(ranges: np.ndarray, dt: float) -> np.ndarray:
    """Closure rate: d(range)/dt.  Negative = closing."""
    return np.gradient(ranges, dt)


def compute_heading_rate(heading_u: np.ndarray, dt: float) -> np.ndarray:
    """Heading change rate in deg/s from unwrapped heading (rad)."""
    dh = np.gradient(heading_u, dt)         # rad/s
    return np.abs(np.degrees(dh))

# ===================================================================
# Engagement flag helpers
# ===================================================================

def is_merge(range_m: float) -> bool:
    return range_m < MERGE_RANGE_M


def is_defensive(aspect_deg: float, closure_mps: float, range_m: float) -> bool:
    """Ego is defensive when enemy has high aspect, is closing, and close."""
    return aspect_deg > DEF_ASPECT_DEG and closure_mps < 0 and range_m < DEF_RANGE_M

# ===================================================================
# Bucket detection functions
# ===================================================================

def check_far_intercept(
    ranges: np.ndarray, bearings: np.ndarray, closures: np.ndarray,
    aspects: np.ndarray, t_idx: int, n_frames: int,
) -> tuple[bool, float, str]:
    """Bucket A: far_intercept -- stable BVR approach geometry."""
    end = t_idx + FAR_INTERCEPT_STABILITY_FRAMES
    if end > n_frames:
        return False, 0.0, ""

    window = slice(t_idx, end)
    r = ranges[window]
    b = bearings[window]
    c = closures[window]

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

    # priority: prefer longer range
    avg_range = float(np.mean(r))
    priority = min(1.0, avg_range / (2.0 * FAR_INTERCEPT_PREFERRED_RANGE_M))
    reason = (f"far_intercept: range={avg_range:.0f}m, "
              f"bearing={float(np.mean(b)):.1f}, closure={float(np.mean(c)):.0f}")
    return True, priority, reason


def check_standoff_hold(
    ranges: np.ndarray, bearings: np.ndarray, closures: np.ndarray,
    aspects: np.ndarray, t_idx: int, n_frames: int,
) -> tuple[bool, float, str]:
    """Bucket B: standoff_hold -- orbiting at medium range."""
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


def check_extend_disengage(
    ranges: np.ndarray, closures: np.ndarray, aspects: np.ndarray,
    t_idx: int, n_frames: int,
) -> tuple[bool, float, str]:
    """Bucket C: extend_disengage -- ego opening range."""
    end = t_idx + EXTEND_STABILITY_FRAMES
    if end > n_frames:
        return False, 0.0, ""

    window = slice(t_idx, end)
    r = ranges[window]
    c = closures[window]

    # range cap: >50km is initial setup, not tactical extend
    if float(np.mean(r)) > EXTEND_MAX_RANGE_M:
        return False, 0.0, ""

    # closure must be positive (opening) over the entire window
    if not np.all(c > 0):
        return False, 0.0, ""

    # range must be monotonically increasing
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


def check_turn_defense(
    ranges: np.ndarray, aspects: np.ndarray, closures: np.ndarray,
    heading_rates: np.ndarray, bearing_rates: np.ndarray,
    t_idx: int,
) -> tuple[bool, float, str]:
    """Bucket D: turn_defense -- high heading rate or defensive geometry."""
    def_flag = is_defensive(aspects[t_idx], closures[t_idx], ranges[t_idx])
    high_hdg_rate = heading_rates[t_idx] > TURN_HDG_RATE_THRESHOLD_DPS
    high_bearing_rate = bearing_rates[t_idx] > TURN_BEARING_CHANGE_THRESHOLD_DPS

    if not (def_flag or high_hdg_rate or high_bearing_rate):
        return False, 0.0, ""

    parts: list[str] = []
    if def_flag:
        parts.append("defensive")
    if high_hdg_rate:
        parts.append(f"hdg_rate={heading_rates[t_idx]:.1f}dps")
    if high_bearing_rate:
        parts.append(f"brg_rate={bearing_rates[t_idx]:.1f}dps")

    priority = 0.6
    if def_flag and high_hdg_rate:
        priority = 0.9
    elif high_hdg_rate:
        priority = 0.75

    reason = f"turn_defense: {', '.join(parts)}, range={ranges[t_idx]:.0f}m"
    return True, priority, reason


def check_handoff_coordination(
    ego_enemies: list[dict],
    allies: list[dict],
    closest_enemy_id_prev: str | None,
    closest_enemy_id_cur: str,
    t_idx: int,
) -> tuple[bool, float, str]:
    """Bucket E: handoff_coordination -- target switch with ally present."""
    if HANDOFF_REQUIRE_TWO_ENEMIES and len(ego_enemies) < 2:
        return False, 0.0, ""
    if HANDOFF_REQUIRE_ALLY and len(allies) < 1:
        return False, 0.0, ""
    if closest_enemy_id_prev is None:
        return False, 0.0, ""
    if closest_enemy_id_cur == closest_enemy_id_prev:
        return False, 0.0, ""

    reason = f"handoff_coordination: target switched {closest_enemy_id_prev}->{closest_enemy_id_cur}"
    return True, 0.80, reason

# ===================================================================
# ACMI parsing helpers
# ===================================================================

_FIXED_WING_RE_LOWER = "fixedwing"

# Exclude non-combat entities by name (AWACS, tankers, etc.)
# Reuse the same regex from entity_filter.py
import re as _re
_EXCLUDE_NAME_RE = _re.compile(
    r"(?i)(A-50|E-3|E-2|E-767|AWACS|Sentry|Hawkeye|"
    r"KC-135|KC-10|KC-46|IL-78|Il-78|Tanker|"
    r"C-130|C-17|C-5|An-26|An-12|Il-76|"
    r"Tu-95|B-52|B-1|B-2|Bomber|Transport|JSTAR)"
)


def _is_combat_fixed_wing(ename: str, etype: str) -> bool:
    """Return True only for combat fixed-wing aircraft (not AWACS/tanker/bomber)."""
    if _FIXED_WING_RE_LOWER not in etype.lower().replace(" ", "").replace("+", ""):
        return False
    # Exclude non-combat types by name or type string
    combined = f"{ename}|{etype}"
    if _EXCLUDE_NAME_RE.search(combined):
        return False
    return True


def _file_hash(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:8]


def _parse_and_resample(acmi_path: str, dt: float):
    """Parse an ACMI file and return resampled entity tracks (dict[id, entity]).

    Uses the Cartesian parser (vals[6:8]) for accurate horizontal positions,
    falling back to lon/lat ENU conversion when Cartesian data is unavailable.
    """
    all_entities = parse_acmi_with_cartesian(acmi_path, dt=dt)

    result: dict[str, dict] = {}
    for oid, ent in all_entities.items():
        if ACMI_FIXED_WING_FILTER and not _is_combat_fixed_wing(
                ent.get("name", ""), ent.get("type", "")):
            continue
        # Speed filter (already resampled by parse_acmi_with_cartesian)
        valid_mask = (ent["spd"] >= ACMI_MIN_SPEED_MPS) & (ent["spd"] <= ACMI_MAX_SPEED_MPS)
        if np.sum(valid_mask) < 20:
            continue
        result[oid] = ent
    return result


def _split_coalitions(entities: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Split entities into two coalition groups.

    Returns (group_a, group_b) where labels are normalised.  If coalition
    metadata is missing we fall back to a 50/50 split by insertion order.
    """
    by_coalition: dict[str, list[dict]] = {}
    for ent in entities.values():
        coal = (ent.get("coalition") or "unknown").strip().lower()
        by_coalition.setdefault(coal, []).append(ent)

    # Remove unknowns from keyed groups if we have at least two known ones
    known_keys = [k for k in by_coalition if k != "unknown"]
    if len(known_keys) >= 2:
        groups = [by_coalition[k] for k in known_keys[:2]]
    elif len(known_keys) == 1:
        # One known + unknown as the other
        groups = [by_coalition[known_keys[0]],
                  by_coalition.get("unknown", [])]
    else:
        # All unknown -- split by insertion order
        all_ents = list(entities.values())
        mid = max(1, len(all_ents) // 2)
        groups = [all_ents[:mid], all_ents[mid:]]

    return groups[0], groups[1]

# ===================================================================
# Obs-sidecar builder
# ===================================================================

def _build_ego_obs(ent: dict, t_idx: int) -> dict[str, Any]:
    """Build the ``ego`` block of the obs sidecar."""
    heading_deg = float(np.degrees(ent["heading_u"][t_idx])) % 360.0
    return {
        "runtime_id": ent["id"],
        "callsign": ent["name"],
        "coalition": ent.get("coalition", ""),
        "speed_mps": float(ent["spd"][t_idx]),
        "altitude_m": float(ent["z"][t_idx]),
        "heading_deg": round(heading_deg, 2),
        "pitch_deg": None,
        "roll_deg": None,
        "vertical_speed_mps": float(ent["vz"][t_idx]),
        "pos_east_m": float(ent["x"][t_idx]),
        "pos_north_m": float(ent["y"][t_idx]),
        "energy_state": None,
    }


def _build_enemy_obs(
    ego: dict, enemy: dict, t_idx: int,
    range_m: float, bearing_deg: float, aspect_deg: float,
    closure_mps: float, alt_diff_m: float, is_primary: bool,
) -> dict[str, Any]:
    return {
        "target_ref": enemy["id"],
        "target_runtime_id": enemy["id"],
        "callsign": enemy["name"],
        "range_m": round(range_m, 1),
        "bearing_deg": round(bearing_deg, 2),
        "aspect_deg": round(aspect_deg, 2),
        "closure_mps": round(closure_mps, 1),
        "alt_diff_m": round(alt_diff_m, 1),
        "is_primary_threat": is_primary,
    }


def _build_ally_obs(
    ego: dict, ally: dict, t_idx: int,
    range_m: float, bearing_deg: float,
) -> dict[str, Any]:
    return {
        "ally_ref": ally["id"],
        "ally_runtime_id": ally["id"],
        "callsign": ally["name"],
        "range_m": round(range_m, 1),
        "bearing_deg": round(bearing_deg, 2),
        "is_supporting": None,
    }

# ===================================================================
# Per-file mining loop
# ===================================================================

def _mine_file(
    acmi_path: str,
    file_hash: str,
    dt: float,
) -> list[dict[str, Any]]:
    """Process one ACMI file and return raw candidate events (pre-decluster)."""

    entities = _parse_and_resample(acmi_path, dt)
    if len(entities) < 2:
        log.debug("Skipping %s: fewer than 2 valid entities", acmi_path)
        return []

    group_a, group_b = _split_coalitions(entities)
    if not group_a or not group_b:
        log.debug("Skipping %s: cannot form two coalitions", acmi_path)
        return []

    candidates: list[dict[str, Any]] = []

    # For each entity as ego, the *other* group are enemies, same group are allies.
    all_entities = group_a + group_b
    group_a_ids = {e["id"] for e in group_a}

    for ego in all_entities:
        ego_in_a = ego["id"] in group_a_ids
        enemies = group_b if ego_in_a else group_a
        allies = [a for a in (group_a if ego_in_a else group_b) if a["id"] != ego["id"]]

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
            # Find overlapping time window
            t0 = max(ego["t"][0], en["t"][0])
            t1 = min(ego["t"][-1], en["t"][-1])
            if t1 - t0 < 5.0:
                continue

            # Indices into ego's time grid
            i0 = int(np.searchsorted(ego["t"], t0))
            i1 = int(np.searchsorted(ego["t"], t1, side="right"))
            if i1 - i0 < 10:
                continue

            # Indices into enemy's time grid (aligned)
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

            enemy_geom.append({
                "enemy": en,
                "i0": i0, "i1": i1,
                "range": rng,
                "bearing": brg,
                "aspect": asp,
                "closure": cls_,
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
        # Precompute ally geometry (for obs sidecar)
        # ----------------------------------------------------------
        ally_geom: list[dict[str, Any]] = []
        for al in allies:
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

            # --- Bucket A: far_intercept ---
            ok, pri, reason = check_far_intercept(
                primary_eg["range"], primary_eg["bearing"],
                primary_eg["closure"], primary_eg["aspect"],
                local_idx, n_local,
            )
            if ok:
                candidates.append(_build_event(
                    file_hash, acmi_path, t_sec, "far_intercept",
                    ego, primary_eg["enemy"],
                    allies[0] if allies else None,
                    active_allies, primary, hdg_rate_dps,
                    pri, reason, t_idx, prev_closest_enemy_id,
                ))

            # --- Bucket B: standoff_hold ---
            ok, pri, reason = check_standoff_hold(
                primary_eg["range"], primary_eg["bearing"],
                primary_eg["closure"], primary_eg["aspect"],
                local_idx, n_local,
            )
            if ok:
                candidates.append(_build_event(
                    file_hash, acmi_path, t_sec, "standoff_hold",
                    ego, primary_eg["enemy"],
                    allies[0] if allies else None,
                    active_allies, primary, hdg_rate_dps,
                    pri, reason, t_idx, prev_closest_enemy_id,
                ))

            # --- Bucket C: extend_disengage ---
            ok, pri, reason = check_extend_disengage(
                primary_eg["range"], primary_eg["closure"],
                primary_eg["aspect"],
                local_idx, n_local,
            )
            if ok:
                candidates.append(_build_event(
                    file_hash, acmi_path, t_sec, "extend_disengage",
                    ego, primary_eg["enemy"],
                    allies[0] if allies else None,
                    active_allies, primary, hdg_rate_dps,
                    pri, reason, t_idx, prev_closest_enemy_id,
                ))

            # --- Bucket D: turn_defense ---
            # check_turn_defense indexes all arrays by local_idx, so we
            # broadcast the ego heading-rate scalar into a local-sized array.
            ego_hdg_val = float(ego_hdg_rate[t_idx]) if t_idx < len(ego_hdg_rate) else 0.0
            local_hdg_rates = np.full(n_local, ego_hdg_val)
            ok, pri, reason = check_turn_defense(
                primary_eg["range"], primary_eg["aspect"],
                primary_eg["closure"],
                local_hdg_rates,
                primary_eg["bearing_rate"],
                local_idx,
            )
            if ok:
                candidates.append(_build_event(
                    file_hash, acmi_path, t_sec, "turn_defense",
                    ego, primary_eg["enemy"],
                    allies[0] if allies else None,
                    active_allies, primary, hdg_rate_dps,
                    pri, reason, t_idx, prev_closest_enemy_id,
                ))

            # --- Bucket E: handoff_coordination ---
            ok, pri, reason = check_handoff_coordination(
                ego_enemies=[{"id": ae["geom"]["enemy"]["id"]} for ae in active_enemies],
                allies=[{"id": ag["geom"]["ally"]["id"]} for ag in active_allies],
                closest_enemy_id_prev=prev_closest_enemy_id,
                closest_enemy_id_cur=closest_enemy_id,
                t_idx=t_idx,
            )
            if ok:
                # Use the *previous* primary for the "from" target
                candidates.append(_build_event(
                    file_hash, acmi_path, t_sec, "handoff_coordination",
                    ego, primary_eg["enemy"],
                    allies[0] if allies else None,
                    active_allies, primary, hdg_rate_dps,
                    pri, reason, t_idx, prev_closest_enemy_id,
                ))

            prev_closest_enemy_id = closest_enemy_id

    return candidates


def _build_event(
    file_hash: str, acmi_path: str, t_sec: float, bucket: str,
    ego: dict, enemy: dict, ally: dict | None,
    active_allies: list[dict], primary: dict,
    hdg_rate_dps: float, priority: float, reason: str,
    t_idx: int, prev_target_ref: str | None,
) -> dict[str, Any]:
    """Assemble a single candidate event dict."""
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

    return {
        "event_id": event_id,
        "file_path": acmi_path,
        "timestamp_sec": t_sec,
        "bucket": bucket,
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
        "event_features": {
            "range_m": round(primary["range"], 1),
            "bearing_deg": round(primary["bearing"], 2),
            "aspect_deg": round(primary["aspect"], 2),
            "closure_mps": round(primary["closure"], 1),
            "heading_rate_dps": round(hdg_rate_dps, 2),
        },
        "reason": reason,
        "priority": round(priority, 4),
    }

# ===================================================================
# De-clustering & sampling
# ===================================================================

def _decluster_events(events: list[dict], dt: float) -> list[dict]:
    """Within each (ego, bucket) group, keep only the highest-priority
    event in each CLUSTER_WINDOW_SEC window."""

    cluster_frames = int(CLUSTER_WINDOW_SEC / dt)

    # Group by (ego_runtime_id, bucket)
    groups: dict[tuple[str, str], list[dict]] = {}
    for ev in events:
        key = (ev["ego_runtime_id"], ev["bucket"])
        groups.setdefault(key, []).append(ev)

    kept: list[dict] = []
    for _key, group in groups.items():
        group.sort(key=lambda e: e["timestamp_sec"])
        i = 0
        while i < len(group):
            t_start = group[i]["timestamp_sec"]
            cluster = [group[i]]
            j = i + 1
            while j < len(group) and (group[j]["timestamp_sec"] - t_start) < CLUSTER_WINDOW_SEC:
                cluster.append(group[j])
                j += 1
            # Keep highest priority in cluster
            best = max(cluster, key=lambda e: e["priority"])
            kept.append(best)
            i = j

    return kept


def _cap_per_file(events: list[dict]) -> list[dict]:
    """Keep at most MAX_EVENTS_PER_FILE events, balanced across buckets.

    Strategy: round-robin through buckets, taking the highest-priority
    event from each bucket in turn, until the cap is reached.
    This ensures rare buckets (turn_defense, standoff) are not
    crowded out by dominant buckets (far_intercept).
    """
    if len(events) <= MAX_EVENTS_PER_FILE:
        return events

    # Group by bucket, sorted by priority within each bucket
    by_bucket: dict[str, list[dict]] = {}
    for ev in events:
        by_bucket.setdefault(ev["bucket"], []).append(ev)
    for bucket_events in by_bucket.values():
        bucket_events.sort(key=lambda e: e["priority"], reverse=True)

    # Round-robin: take one from each bucket in turn
    kept: list[dict] = []
    bucket_names = list(by_bucket.keys())
    indices = {b: 0 for b in bucket_names}

    while len(kept) < MAX_EVENTS_PER_FILE:
        added_this_round = False
        for b in bucket_names:
            if len(kept) >= MAX_EVENTS_PER_FILE:
                break
            idx = indices[b]
            if idx < len(by_bucket[b]):
                kept.append(by_bucket[b][idx])
                indices[b] += 1
                added_this_round = True
        if not added_this_round:
            break  # all buckets exhausted

    return kept

# ===================================================================
# Main entry
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine candidate events from ACMI files for 5 semantic buckets.",
    )
    parser.add_argument(
        "--catalog", required=True,
        help="Path to acmi_catalog.jsonl produced by build_acmi_catalog.py",
    )
    parser.add_argument(
        "--input", required=True,
        help="Root directory containing ACMI files (paths in catalog are relative to this)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for candidate_events.jsonl",
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
    input_root = Path(args.input)
    output_path = Path(args.output)

    if not catalog_path.exists():
        log.error("Catalog file not found: %s", catalog_path)
        sys.exit(1)

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
    log.info("Found %d candidate files (candidate_2v2 or candidate_multiship)", len(candidate_entries))

    all_events: list[dict] = []
    bucket_counts: dict[str, int] = {}
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
            raw_events = _mine_file(acmi_path, fhash, args.dt)
        except Exception:
            log.exception("Failed to process %s", acmi_path)
            files_failed += 1
            continue

        # Decluster within file
        declustered = _decluster_events(raw_events, args.dt)
        # Cap per file
        capped = _cap_per_file(declustered)

        for ev in capped:
            bucket_counts[ev["bucket"]] = bucket_counts.get(ev["bucket"], 0) + 1
        all_events.extend(capped)
        files_processed += 1

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ev in all_events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    log.info("=" * 60)
    log.info("Mining complete.")
    log.info("  Files processed : %d", files_processed)
    log.info("  Files failed    : %d", files_failed)
    log.info("  Total events    : %d", len(all_events))
    for bucket, count in sorted(bucket_counts.items()):
        log.info("    %-25s %d", bucket, count)
    log.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
