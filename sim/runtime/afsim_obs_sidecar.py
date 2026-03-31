"""Build an obs-compatible runtime sidecar from a 640-value AFSIM frame.

Output structure aligns with ``semantic_record_schema_v1.json`` ``obs`` node:
  { ego, enemy, ally, engagement, history }

This is NOT a full canonical record — no source/label/evidence/quality/extras.
It is the runtime obs that can be directly used as AI annotation input
(``acmi_auto_label_io_examples.json`` format_A).

Usage:
    ctx = SidecarContext()
    sidecar = build_obs_sidecar(frame_values, ego_platform_idx=0, ctx=ctx)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import afsim_obs_entities as ent
from . import afsim_obs_geometry as geo
from .semantic_thresholds import (
    DEF_ASPECT_DEG,
    DEF_RANGE_M,
    MERGE_RANGE_M,
    TARGET_SWITCH_RATIO,
)


# ── persistent context across frames ───────────────────────────

@dataclass
class SidecarContext:
    """Carries per-platform state across frames for history and sticky target."""
    prev_headings: dict[int, float] = field(default_factory=dict)
    prev_target_ref: Optional[str] = None
    prev_target_idx: Optional[int] = None
    prev_semantic_state: Optional[str] = None
    prev_token_family: Optional[str] = None
    ref_lat: float = 0.0
    ref_lon: float = 0.0


# ── target / ally ref naming ────────────────────────────────────

def _enemy_ref(ego_idx: int, enemy_idx: int) -> str:
    """Assign stable enemy_1 / enemy_2 ref within an episode.

    Within each coalition, enemies are numbered by their platform index order.
    """
    ego_coal = ent.COALITION_MAP.get(ego_idx)
    # Collect all enemy indices (opposite coalition), sorted
    enemy_indices = sorted(
        idx for idx, coal in ent.COALITION_MAP.items()
        if coal != ego_coal and idx < 4
    )
    if enemy_idx in enemy_indices:
        pos = enemy_indices.index(enemy_idx)
        return f"enemy_{pos + 1}"
    return "enemy_1"


def _ally_ref(ego_idx: int, ally_idx: int) -> str:
    """Always ally_1 for 2v2."""
    return "ally_1"


# ── target selection (sticky) ──────────────────────────────────

def _select_primary_enemy(
    ego: ent.EntityState,
    enemies: list[ent.EntityState],
    ctx: SidecarContext,
) -> Optional[ent.EntityState]:
    """Select primary enemy using sticky-target logic.

    1. If prev_target is still alive, prefer it.
    2. Switch only if a new target is significantly closer (ratio < 0.8).
    3. Otherwise pick nearest.
    """
    if not enemies:
        return None

    # Sort by range
    enemies_with_range = [
        (e, geo.compute_range_m(ego, e)) for e in enemies
    ]
    enemies_with_range.sort(key=lambda x: x[1])

    nearest, nearest_range = enemies_with_range[0]

    # Try sticky target
    if ctx.prev_target_idx is not None:
        prev = next((e for e in enemies if e.platform_idx == ctx.prev_target_idx), None)
        if prev is not None:
            prev_range = geo.compute_range_m(ego, prev)
            # Only switch if nearest is significantly closer
            if nearest_range < prev_range * TARGET_SWITCH_RATIO:
                return nearest
            return prev

    return nearest


def _select_ally(
    ego: ent.EntityState,
    allies: list[ent.EntityState],
) -> Optional[ent.EntityState]:
    """Select nearest same-coalition ally (excluding self)."""
    if not allies:
        return None
    by_range = [(a, geo.compute_range_m(ego, a)) for a in allies]
    by_range.sort(key=lambda x: x[1])
    return by_range[0][0]


# ── engagement flags ────────────────────────────────────────────

def _compute_engagement(
    ego: ent.EntityState,
    enemy: Optional[ent.EntityState],
) -> Optional[dict]:
    """Compute engagement flags per annotation_guideline_v1.md thresholds."""
    if enemy is None:
        return None

    geom = geo.compute_all(ego, enemy)
    range_m = geom["range_m"]
    aspect_deg = geom["aspect_deg"]
    closure_mps = geom["closure_mps"]

    is_merge = range_m < MERGE_RANGE_M

    is_defensive = False
    if aspect_deg is not None:
        is_defensive = (
            aspect_deg > DEF_ASPECT_DEG
            and closure_mps < 0
            and range_m < DEF_RANGE_M
        )

    return {
        "is_merge": is_merge,
        "is_defensive": is_defensive,
        "has_shot_opportunity": None,  # no reliable weapon envelope
    }


# ── ego dict builder ───────────────────────────────────────────

def _build_ego(e: ent.EntityState) -> dict:
    """Build canonical obs.ego dict."""
    return {
        "runtime_id": e.runtime_id,
        "callsign": e.callsign,
        "coalition": e.coalition,
        "speed_mps": round(e.speed_mps, 1),
        "altitude_m": round(e.altitude_m, 1),
        "heading_deg": round(e.heading_deg, 1),
        "pitch_deg": round(e.pitch_deg, 1) if e.pitch_deg is not None else None,
        "roll_deg": None,  # AFSIM runtime does not report roll
        "vertical_speed_mps": round(e.vertical_speed_mps, 1),
        "pos_east_m": round(e.pos_east_m, 1),
        "pos_north_m": round(e.pos_north_m, 1),
        "energy_state": None,  # q30/q70 thresholds not yet configured
    }


def _build_enemy(
    ego: ent.EntityState,
    enemy: ent.EntityState,
    ego_idx: int,
) -> dict:
    """Build canonical obs.enemy dict."""
    geom = geo.compute_all(ego, enemy)
    return {
        "target_ref": _enemy_ref(ego_idx, enemy.platform_idx),
        "target_runtime_id": enemy.runtime_id,
        "callsign": enemy.callsign,
        "range_m": geom["range_m"],
        "bearing_deg": geom["bearing_deg"],
        "aspect_deg": geom["aspect_deg"],
        "closure_mps": geom["closure_mps"],
        "alt_diff_m": geom["alt_diff_m"],
        "is_primary_threat": True,  # selected as primary
    }


def _build_ally(
    ego: ent.EntityState,
    ally: ent.EntityState,
    ego_idx: int,
) -> dict:
    """Build canonical obs.ally dict."""
    geom = geo.compute_all(ego, ally)
    return {
        "ally_ref": _ally_ref(ego_idx, ally.platform_idx),
        "ally_runtime_id": ally.runtime_id,
        "callsign": ally.callsign,
        "range_m": geom["range_m"],
        "bearing_deg": geom["bearing_deg"],
        "is_supporting": None,  # cannot determine from 640 frame alone
    }


# ── main entry point ───────────────────────────────────────────

def build_obs_sidecar(
    frame_values: np.ndarray,
    ego_platform_idx: int = 0,
    ctx: Optional[SidecarContext] = None,
) -> dict:
    """Build obs-compatible runtime sidecar from a 640-value AFSIM frame.

    Args:
        frame_values: 640-element float array.
        ego_platform_idx: Which platform is "ego" (0=red_1, 1=red_2, etc).
        ctx: Persistent context for history and sticky target.
             Mutated in-place. Create one per ego platform.

    Returns:
        Dict with keys: ego, enemy, ally, engagement, history.
        Structure matches semantic_record_schema_v1.json obs node.
    """
    if ctx is None:
        ctx = SidecarContext()

    # Parse all platforms
    entities = ent.parse_frame(
        frame_values,
        ref_lat=ctx.ref_lat,
        ref_lon=ctx.ref_lon,
        prev_headings=ctx.prev_headings,
    )

    # Lock reference origin on first frame
    if ctx.ref_lat == 0.0 and ctx.ref_lon == 0.0 and entities:
        first = next(iter(entities.values()))
        ctx.ref_lat = first.lat
        ctx.ref_lon = first.lon
        # Re-parse with locked origin
        entities = ent.parse_frame(
            frame_values,
            ref_lat=ctx.ref_lat,
            ref_lon=ctx.ref_lon,
            prev_headings=ctx.prev_headings,
        )

    # Update heading cache
    for idx, e in entities.items():
        ctx.prev_headings[idx] = e.heading_deg

    ego = entities.get(ego_platform_idx)
    if ego is None:
        # Ego not alive — return minimal sidecar
        return {
            "ego": None,
            "enemy": None,
            "ally": None,
            "engagement": None,
            "history": {
                "prev_semantic_state": ctx.prev_semantic_state,
                "prev_token_family": ctx.prev_token_family,
                "prev_target_ref": ctx.prev_target_ref,
            },
        }

    ego_coal = ent.COALITION_MAP.get(ego_platform_idx)

    # Separate enemies and allies
    enemies = [
        e for idx, e in entities.items()
        if idx != ego_platform_idx and ent.COALITION_MAP.get(idx) != ego_coal
    ]
    allies = [
        e for idx, e in entities.items()
        if idx != ego_platform_idx and ent.COALITION_MAP.get(idx) == ego_coal
    ]

    # Select primary enemy (sticky)
    primary_enemy = _select_primary_enemy(ego, enemies, ctx)
    # Select ally
    primary_ally = _select_ally(ego, allies)

    # Build sub-dicts
    ego_dict = _build_ego(ego)

    enemy_dict = None
    enemy_ref = None
    if primary_enemy is not None:
        enemy_dict = _build_enemy(ego, primary_enemy, ego_platform_idx)
        enemy_ref = enemy_dict["target_ref"]
        ctx.prev_target_idx = primary_enemy.platform_idx

    ally_dict = None
    if primary_ally is not None:
        ally_dict = _build_ally(ego, primary_ally, ego_platform_idx)

    engagement_dict = _compute_engagement(ego, primary_enemy)

    # Detect target switch
    target_switched = (
        ctx.prev_target_ref is not None
        and enemy_ref is not None
        and ctx.prev_target_ref != enemy_ref
    )

    history_dict = {
        "prev_semantic_state": ctx.prev_semantic_state,
        "prev_token_family": ctx.prev_token_family,
        "prev_target_ref": ctx.prev_target_ref,
    }

    # Update context for next frame
    ctx.prev_target_ref = enemy_ref

    return {
        "ego": ego_dict,
        "enemy": enemy_dict,
        "ally": ally_dict,
        "engagement": engagement_dict,
        "history": history_dict,
    }
