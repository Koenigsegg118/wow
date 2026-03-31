"""Relative geometry calculations between two EntityState objects.

All formulas align with ``acmi_field_mapping_v1.md`` and
``annotation_guideline_v1.md``.

Sign conventions (frozen at v1.1):
  bearing_deg : positive = starboard (right), range [-180, 180]
  aspect_deg  : 0 = head-on, 180 = tail-chase, range [0, 180]
  closure_mps : negative = closing, positive = opening
  alt_diff_m  : ego_alt - other_alt, positive = ego higher
"""

from __future__ import annotations

import math

from .afsim_obs_entities import EntityState
from .semantic_thresholds import MIN_RANGE_M


def _wrap_pm180(angle_deg: float) -> float:
    """Wrap angle to [-180, 180] degrees."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def compute_range_m(ego: EntityState, other: EntityState) -> float:
    """Slant range in metres."""
    dx = other.pos_east_m - ego.pos_east_m
    dy = other.pos_north_m - ego.pos_north_m
    dz = other.altitude_m - ego.altitude_m
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_bearing_deg(ego: EntityState, other: EntityState) -> float:
    """Bearing from ego to other in ego's body frame.

    0 = nose, positive = starboard (right), range [-180, 180].
    Formula: atan2(dx, dy) - ego_heading, wrapped to [-180, 180].
    """
    dx = other.pos_east_m - ego.pos_east_m
    dy = other.pos_north_m - ego.pos_north_m
    # Absolute azimuth from ego to other (0=North, CW+)
    abs_az_deg = math.degrees(math.atan2(dx, dy))
    bearing = abs_az_deg - ego.heading_deg
    return _wrap_pm180(bearing)


def compute_aspect_deg(
    ego: EntityState,
    other: EntityState,
    other_heading_deg: float | None = None,
) -> float | None:
    """Aspect angle: angle between other's nose and LOS from other to ego.

    0 = head-on (other's nose points at ego), 180 = tail-on.
    Returns absolute value in [0, 180].

    If other's heading is unavailable, returns None.
    """
    hdg = other_heading_deg if other_heading_deg is not None else other.heading_deg
    if hdg is None:
        return None

    # LOS from other to ego
    dx = ego.pos_east_m - other.pos_east_m
    dy = ego.pos_north_m - other.pos_north_m
    los_deg = math.degrees(math.atan2(dx, dy))  # 0=North, CW+

    # Angle between other's heading and LOS to ego
    raw = los_deg - hdg
    wrapped = _wrap_pm180(raw)
    return abs(wrapped)


def compute_closure_mps(ego: EntityState, other: EntityState) -> float:
    """Range rate (closure): d(range)/dt projected along LOS.

    Negative = closing, positive = opening.
    """
    dx = other.pos_east_m - ego.pos_east_m
    dy = other.pos_north_m - ego.pos_north_m
    dz = other.altitude_m - ego.altitude_m
    r = math.sqrt(dx * dx + dy * dy + dz * dz)

    if r < MIN_RANGE_M:
        return 0.0

    # Relative velocity (other - ego)
    dvx = other.vx_e_mps - ego.vx_e_mps
    dvy = other.vy_n_mps - ego.vy_n_mps
    dvz = other.vz_u_mps - ego.vz_u_mps

    # Project onto unit LOS vector (ego -> other)
    return (dx * dvx + dy * dvy + dz * dvz) / r


def compute_alt_diff_m(ego: EntityState, other: EntityState) -> float:
    """Altitude difference: ego_alt - other_alt.

    Positive = ego is higher.
    """
    return ego.altitude_m - other.altitude_m


def compute_all(
    ego: EntityState,
    other: EntityState,
) -> dict:
    """Compute all relative geometry fields for ego vs other.

    Returns dict with keys matching canonical schema obs.enemy / obs.ally fields.
    """
    range_m = compute_range_m(ego, other)
    bearing_deg = compute_bearing_deg(ego, other)
    aspect_deg = compute_aspect_deg(ego, other)
    closure_mps = compute_closure_mps(ego, other)
    alt_diff_m = compute_alt_diff_m(ego, other)

    return {
        "range_m": round(range_m, 1),
        "bearing_deg": round(bearing_deg, 1),
        "aspect_deg": round(aspect_deg, 1) if aspect_deg is not None else None,
        "closure_mps": round(closure_mps, 1),
        "alt_diff_m": round(alt_diff_m, 1),
    }
