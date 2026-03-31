"""ACMI parser that extracts native Cartesian coordinates (x/y/z) when available.

Tacview ACMI T= field format:
    T=lon|lat|alt|roll|pitch|yaw|u|v|heading  (short, 3 values)
    T=lon|lat|alt|roll|pitch|yaw|x|y|z        (extended, 9 values)

When 9+ values are present, vals[6:9] give native Cartesian coordinates in
metres, which are far more accurate than lon/lat with an unknown reference
point. This parser prefers Cartesian when available, falling back to lon/lat
ENU conversion when only 3 values exist.

This module does NOT modify the training pipeline's existing parser.
It is used only by the ACMI mining tools.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure ml/ is importable
_WOW_ROOT = Path(__file__).resolve().parent.parent.parent
_ML_DIR = _WOW_ROOT / "ml"
if str(_WOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_WOW_ROOT))
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from ml.acmi_to_dt_dataset_smooth import (  # noqa: E402
    read_acmi_text,
    resample_entity,
)

logger = logging.getLogger(__name__)

_HEADER_KV = re.compile(r"^(\w+)=(.*)")


def parse_acmi_with_cartesian(acmi_path: str, dt: float = 0.5) -> dict[str, dict]:
    """Parse ACMI file, preferring native Cartesian x/y/z coordinates.

    Returns:
        dict[entity_id -> entity_dict] with fields:
        - name, type, coalition, id
        - t, x, y, z, vx, vy, vz, spd, heading_u  (after resampling)
    """
    text = read_acmi_text(acmi_path)
    lines = text.splitlines()

    # ── Parse header ──
    meta: dict[str, str] = {}
    start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") and len(line) > 1:
            start_idx = i
            break
        if line.startswith("T=") or re.match(r"^\w+,", line):
            start_idx = i
            break
        m = _HEADER_KV.match(line)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
        start_idx = i + 1

    ref_lat = float(meta.get("ReferenceLatitude", "0") or "0")
    ref_lon = float(meta.get("ReferenceLongitude", "0") or "0")

    # ── Parse object updates (no state accumulation) ──
    #
    # ACMI/Tacview sends full 9-value T= updates (~88% of frames) and
    # short delta updates (~12%). We ONLY record positions from full updates
    # with Cartesian x/y to avoid stale coordinate bugs. Short updates are
    # skipped — resample_entity will interpolate the gaps.
    t = 0.0
    objects: dict[str, dict] = {}
    tracks_cart: dict[str, list] = {}
    tracks_lla: dict[str, list] = {}
    has_cartesian: dict[str, bool] = {}

    for idx in range(start_idx, len(lines)):
        raw = lines[idx].strip()
        if not raw:
            continue
        if raw.startswith("#"):
            try:
                t = float(raw[1:])
            except ValueError:
                pass
            continue
        if raw.startswith("T="):
            try:
                t = float(raw[2:])
            except ValueError:
                pass
            continue

        parts = raw.split(",")
        if len(parts) < 2:
            continue
        oid = parts[0].strip()
        if oid == "-1":  # removal marker
            continue

        kvs = parts[1:]
        props: dict[str, str] = {}
        for kv in kvs:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            props[k.strip()] = v.strip()

        objects.setdefault(oid, {})
        for k in ("Name", "Type", "Coalition", "Country", "Group"):
            if k in props:
                objects[oid][k] = props[k]

        if "T" in props:
            vals = props["T"].split("|")

            try:
                # Full update with Cartesian x,y (9 values, slots 6+7 non-empty)
                if len(vals) >= 9 and vals[6] and vals[7]:
                    x_m = float(vals[6])
                    y_m = float(vals[7])
                    # Altitude from vals[2] (MSL), NOT vals[8] (native Z)
                    alt = float(vals[2]) if vals[2] else 0.0
                    tracks_cart.setdefault(oid, []).append((t, x_m, y_m, alt))
                    has_cartesian[oid] = True
                elif len(vals) >= 3 and vals[0] and vals[1]:
                    # Lon/lat only (no Cartesian) — fallback
                    lon = float(vals[0])
                    lat = float(vals[1])
                    alt = float(vals[2]) if vals[2] else 0.0
                    tracks_lla.setdefault(oid, []).append((t, lon, lat, alt))
                    has_cartesian.setdefault(oid, False)
                # Short delta updates (empty x/y) are SKIPPED —
                # resample_entity will interpolate
            except (ValueError, IndexError):
                pass

    # ── Build entity tracks ──
    result: dict[str, dict] = {}

    for oid, info in objects.items():
        if has_cartesian.get(oid, False) and oid in tracks_cart:
            pts = tracks_cart[oid]
        elif oid in tracks_lla:
            pts = tracks_lla[oid]
        else:
            continue

        if len(pts) < 5:
            continue

        t_arr = np.array([p[0] for p in pts], dtype=np.float64)
        x_arr = np.array([p[1] for p in pts], dtype=np.float64)
        y_arr = np.array([p[2] for p in pts], dtype=np.float64)
        z_arr = np.array([p[3] for p in pts], dtype=np.float64)

        if not has_cartesian.get(oid, False):
            # Fallback: convert lon/lat to ENU (same as original parser)
            # x_arr = lon, y_arr = lat here
            lon_arr = x_arr
            lat_arr = y_arr

            if np.nanmax(np.abs(lon_arr)) < 5.0 and np.nanmax(np.abs(lat_arr)) < 5.0:
                lon_abs = ref_lon + lon_arr
                lat_abs = ref_lat + lat_arr
            else:
                lon_abs = lon_arr
                lat_abs = lat_arr

            cos_lat = np.cos(np.radians(ref_lat)) if ref_lat != 0 else np.cos(np.radians(np.mean(lat_abs)))
            x_arr = (lon_abs - ref_lon) * 111320.0 * cos_lat
            y_arr = (lat_abs - ref_lat) * 110540.0
            # z_arr stays as altitude

        # Build entity dict compatible with resample_entity
        ent = {
            "id": oid,
            "name": info.get("Name", oid),
            "type": info.get("Type", ""),
            "coalition": info.get("Coalition", ""),
            "t": t_arr,
            "x": x_arr,
            "y": y_arr,
            "z": z_arr,
        }

        resampled = resample_entity(ent, dt=dt)
        if resampled is None:
            continue

        result[oid] = resampled

    n_cart = sum(1 for v in has_cartesian.values() if v)
    n_lla = sum(1 for v in has_cartesian.values() if not v)
    logger.info(
        "Parsed %s: %d entities (%d Cartesian, %d lon/lat fallback), %d resampled",
        Path(acmi_path).name, len(objects), n_cart, n_lla, len(result),
    )

    return result
