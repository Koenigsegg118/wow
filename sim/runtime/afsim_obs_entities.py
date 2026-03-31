"""Parse a 640-value AFSIM frame into per-platform entity dicts.

640 values = 80 platforms × 8 fields each.
Per-platform layout (base = 8 * idx):
  [base+0] live          float  1.0=alive, <=0=dead/invalid
  [base+1] lat           float  degrees
  [base+2] lon           float  degrees
  [base+3] alt_m         float  metres MSL
  [base+4] velN_mps      float  north velocity m/s
  [base+5] velE_mps      float  east  velocity m/s
  [base+6] velD_mps      float  down  velocity m/s (positive DOWN)
  [base+7] heading_deg   float  degrees (AFSIM diagnostic)

Platform index mapping (fixed for 2v2):
  0 = red_1    1 = red_2    2 = blue_1    3 = blue_2
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .semantic_thresholds import (
    EARTH_RADIUS_M,
    MIN_HSPEED_FOR_HEADING_MPS,
    MIN_VALID_SPEED_MPS,
)

# ── platform index assignments ──────────────────────────────────

PLATFORM_NAMES = {0: "red_1", 1: "red_2", 2: "blue_1", 3: "blue_2"}
COALITION_MAP = {0: "red", 1: "red", 2: "blue", 3: "blue"}
N_PLATFORMS = 80
FIELDS_PER_PLATFORM = 8


# ── data classes ────────────────────────────────────────────────

class EntityState:
    """Parsed state of a single platform in ENU frame."""

    __slots__ = (
        "platform_idx", "runtime_id", "callsign", "coalition",
        "alive", "lat", "lon",
        "pos_east_m", "pos_north_m", "altitude_m",
        "vx_e_mps", "vy_n_mps", "vz_u_mps",
        "heading_deg", "speed_mps", "vertical_speed_mps",
        "pitch_deg", "roll_deg",
    )

    def __init__(self) -> None:
        self.platform_idx: int = -1
        self.runtime_id: Optional[str] = None
        self.callsign: Optional[str] = None
        self.coalition: Optional[str] = None
        self.alive: bool = False
        self.lat: float = 0.0
        self.lon: float = 0.0
        self.pos_east_m: float = 0.0
        self.pos_north_m: float = 0.0
        self.altitude_m: float = 0.0
        self.vx_e_mps: float = 0.0
        self.vy_n_mps: float = 0.0
        self.vz_u_mps: float = 0.0
        self.heading_deg: float = 0.0
        self.speed_mps: float = 0.0
        self.vertical_speed_mps: float = 0.0
        self.pitch_deg: Optional[float] = None
        self.roll_deg: Optional[float] = None  # AFSIM runtime does not report roll


def _latlon_to_enu(
    lat: float, lon: float,
    lat0: float, lon0: float,
) -> tuple[float, float]:
    """Convert lat/lon (degrees) to East-North (metres) relative to origin."""
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    avg_lat = math.radians(0.5 * (lat + lat0))
    east_m = dlon * math.cos(avg_lat) * EARTH_RADIUS_M
    north_m = dlat * EARTH_RADIUS_M
    return east_m, north_m


def parse_frame(
    values: np.ndarray,
    ref_lat: float = 0.0,
    ref_lon: float = 0.0,
    prev_headings: Optional[dict[int, float]] = None,
) -> dict[int, EntityState]:
    """Parse 640-value frame into EntityState dict keyed by platform index.

    Only parses platforms 0-3 (the 2v2 combatants).

    Args:
        values: 640-element float array from AFSIM.
        ref_lat, ref_lon: Reference origin for ENU conversion (degrees).
            If 0.0, uses platform 0's lat/lon as origin.
        prev_headings: dict of platform_idx -> previous valid heading_deg,
            used as fallback when horizontal speed is too low.

    Returns:
        Dict mapping platform_idx to EntityState for alive platforms.
    """
    if prev_headings is None:
        prev_headings = {}

    # Auto-detect reference from platform 0 if not provided
    if ref_lat == 0.0 and ref_lon == 0.0:
        ref_lat = float(values[1])
        ref_lon = float(values[2])

    entities: dict[int, EntityState] = {}

    for idx in range(4):  # only 2v2 platforms
        base = idx * FIELDS_PER_PLATFORM
        live = float(values[base + 0])
        if live <= 0:
            continue

        e = EntityState()
        e.platform_idx = idx
        e.runtime_id = PLATFORM_NAMES.get(idx, f"platform_{idx}")
        e.callsign = e.runtime_id  # no separate callsign in 640 frame
        e.coalition = COALITION_MAP.get(idx)
        e.alive = True

        e.lat = float(values[base + 1])
        e.lon = float(values[base + 2])
        e.altitude_m = float(values[base + 3])

        # Velocities: AFSIM gives velN, velE, velD (positive DOWN)
        e.vy_n_mps = float(values[base + 4])
        e.vx_e_mps = float(values[base + 5])
        e.vz_u_mps = -float(values[base + 6])  # convert to UP-positive

        # ENU position
        e.pos_east_m, e.pos_north_m = _latlon_to_enu(
            e.lat, e.lon, ref_lat, ref_lon)

        # Ground speed
        hspeed = math.sqrt(e.vx_e_mps ** 2 + e.vy_n_mps ** 2)
        e.speed_mps = hspeed
        e.vertical_speed_mps = e.vz_u_mps

        # Heading: prefer velocity-derived, fallback to AFSIM field or prev
        if hspeed >= MIN_HSPEED_FOR_HEADING_MPS:
            e.heading_deg = math.degrees(math.atan2(e.vx_e_mps, e.vy_n_mps))
            if e.heading_deg < 0:
                e.heading_deg += 360.0
        else:
            raw_hdg = float(values[base + 7])
            if raw_hdg != 0.0 or idx not in prev_headings:
                e.heading_deg = raw_hdg
            else:
                e.heading_deg = prev_headings.get(idx, 0.0)

        # Pitch: approximate from velocity
        speed_3d = math.sqrt(hspeed ** 2 + e.vz_u_mps ** 2)
        if speed_3d > MIN_VALID_SPEED_MPS:
            e.pitch_deg = math.degrees(math.asin(
                max(-1.0, min(1.0, e.vz_u_mps / speed_3d))))
        else:
            e.pitch_deg = None

        # Roll: not available from 640 frame
        e.roll_deg = None

        entities[idx] = e

    return entities
