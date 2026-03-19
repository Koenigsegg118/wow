#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Tacview ACMI into shared/schema episode contract JSON."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from shared.schema import SCHEMA_VERSION, dump_episode
from sim.tools.acmi2tspi import (
    extract_reference_latlon,
    latlon_to_enu,
    maybe_apply_reference_offset,
    parse_acmi_objects_and_tracks,
    parse_header,
    read_acmi_text,
    side_from_props,
)


def _wrap_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def _resolve_reference(
    header: Dict[str, str], lines: List[str], tracks: Dict[str, List[Tuple[float, float, float, float, float, float, float | None]]]
) -> Tuple[float, float]:
    ref_lat: float | None = None
    ref_lon: float | None = None
    try:
        ref_lat = float(header.get("ReferenceLatitude", "nan"))
        ref_lon = float(header.get("ReferenceLongitude", "nan"))
        if math.isnan(ref_lat) or math.isnan(ref_lon):
            ref_lat, ref_lon = None, None
    except Exception:
        ref_lat, ref_lon = None, None

    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = extract_reference_latlon(lines)

    if ref_lat is None or ref_lon is None:
        for pts in tracks.values():
            if pts:
                ref_lon = float(pts[0][1])
                ref_lat = float(pts[0][2])
                break

    if ref_lat is None or ref_lon is None:
        raise ValueError("Unable to resolve reference latitude/longitude from ACMI")
    return ref_lat, ref_lon


def _resample_track(
    pts: List[Tuple[float, float, float, float, float, float, float | None]],
    dt: float,
    ref_lat: float,
    ref_lon: float,
    trim_speed_mps: float,
) -> Dict[str, np.ndarray] | None:
    pts_sorted = sorted(pts, key=lambda x: x[0])
    if len(pts_sorted) < 3:
        return None

    t = np.asarray([p[0] for p in pts_sorted], dtype=np.float64)
    lon = np.asarray([p[1] for p in pts_sorted], dtype=np.float64)
    lat = np.asarray([p[2] for p in pts_sorted], dtype=np.float64)
    alt = np.asarray([p[3] for p in pts_sorted], dtype=np.float64)

    if float(t[-1] - t[0]) < dt * 2.0:
        return None

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for i in range(len(t)):
        x[i], y[i] = latlon_to_enu(float(lat[i]), float(lon[i]), ref_lat, ref_lon)

    if trim_speed_mps > 0.0 and len(t) >= 3:
        dt_raw = np.diff(t)
        dt_raw[dt_raw < 1e-6] = 1e-6
        speed_raw = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(alt) ** 2) / dt_raw
        moving_idx = np.flatnonzero(speed_raw >= trim_speed_mps)
        if moving_idx.size > 0 and int(moving_idx[0]) > 0:
            keep = slice(int(moving_idx[0]), None)
            t = t[keep]
            x = x[keep]
            y = y[keep]
            alt = alt[keep]

    if len(t) < 3 or float(t[-1] - t[0]) < dt * 2.0:
        return None

    grid = np.arange(float(t[0]), float(t[-1]) + 1e-9, dt, dtype=np.float64)
    if grid.size < 3:
        return None

    x_i = np.interp(grid, t, x)
    y_i = np.interp(grid, t, y)
    z_i = np.interp(grid, t, alt)
    vx = np.gradient(x_i, dt)
    vy = np.gradient(y_i, dt)
    vz = np.gradient(z_i, dt)
    heading = np.arctan2(vx, vy)
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)

    return {
        "t": grid,
        "x": x_i,
        "y": y_i,
        "z": z_i,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "heading": heading,
        "speed": speed,
    }


def _collect_entities(
    objects: Dict[str, Dict[str, str]],
    tracks: Dict[str, List[Tuple[float, float, float, float, float, float, float | None]]],
    dt: float,
    ref_lat: float,
    ref_lon: float,
    type_filter: str,
    trim_speed_mps: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for oid in sorted(tracks.keys()):
        obj = objects.get(oid, {})
        obj_type = obj.get("Type", "") or ""
        if type_filter and type_filter not in obj_type:
            continue

        rs = _resample_track(tracks[oid], dt=dt, ref_lat=ref_lat, ref_lon=ref_lon, trim_speed_mps=trim_speed_mps)
        if rs is None:
            continue

        side = side_from_props(obj)
        out.append(
            {
                "entity_id": oid,
                "name": obj.get("Name", oid) or oid,
                "type": obj_type,
                "side": side,
                "rs": rs,
            }
        )
    return out


def build_episode(
    acmi_path: Path,
    dt: float = 0.5,
    type_filter: str = "Air+FixedWing",
    max_entities: int | None = None,
    trim_speed_mps: float = 0.0,
) -> Dict[str, Any]:
    text = read_acmi_text(acmi_path)
    lines = text.splitlines()
    header, start_idx = parse_header(lines)
    objects, tracks = parse_acmi_objects_and_tracks(lines, start_idx)
    ref_lat, ref_lon = _resolve_reference(header, lines, tracks)
    maybe_apply_reference_offset(tracks, ref_lat, ref_lon)

    entities = _collect_entities(
        objects=objects,
        tracks=tracks,
        dt=dt,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        type_filter=type_filter,
        trim_speed_mps=trim_speed_mps,
    )
    if not entities and type_filter:
        entities = _collect_entities(
            objects=objects,
            tracks=tracks,
            dt=dt,
            ref_lat=ref_lat,
            ref_lon=ref_lon,
            type_filter="",
            trim_speed_mps=trim_speed_mps,
        )

    if not entities:
        raise ValueError("No entities with valid tracks were extracted from ACMI")

    entities.sort(key=lambda item: (-len(item["rs"]["t"]), item["entity_id"]))
    if max_entities is not None:
        entities = entities[: max(1, max_entities)]

    global_start = min(float(item["rs"]["t"][0]) for item in entities)
    global_end = max(float(item["rs"]["t"][-1]) for item in entities)
    n_ticks = int(round((global_end - global_start) / dt)) + 1
    if n_ticks < 2:
        raise ValueError("Episode too short after resampling")

    state_by_entity: Dict[str, Dict[int, Dict[str, float | str | bool]]] = {}
    entity_meta: List[Dict[str, str]] = []
    for item in entities:
        oid = item["entity_id"]
        rs = item["rs"]
        ticks = np.rint((rs["t"] - global_start) / dt).astype(np.int64)
        state_map: Dict[int, Dict[str, float | str | bool]] = {}
        for i, tick in enumerate(ticks):
            state_map[int(tick)] = {
                "entity_id": oid,
                "alive": True,
                "x_e_m": float(rs["x"][i]),
                "y_n_m": float(rs["y"][i]),
                "z_u_m": float(rs["z"][i]),
                "vx_e_mps": float(rs["vx"][i]),
                "vy_n_mps": float(rs["vy"][i]),
                "vz_u_mps": float(rs["vz"][i]),
                "heading_rad": float(rs["heading"][i]),
            }
        state_by_entity[oid] = state_map
        entity_meta.append(
            {
                "entity_id": oid,
                "name": item["name"],
                "type": item["type"],
                "side": item["side"],
            }
        )

    frames: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    entity_ids = [item["entity_id"] for item in entity_meta]
    for tick in range(n_ticks):
        states = []
        for oid in entity_ids:
            st = state_by_entity[oid].get(tick)
            if st is not None:
                states.append(st)
        if states:
            frames.append({"t_s": float(round(tick * dt, 6)), "states": states})

    for tick in range(n_ticks - 1):
        commands = []
        for oid in entity_ids:
            current_state = state_by_entity[oid].get(tick)
            next_state = state_by_entity[oid].get(tick + 1)
            if current_state is None or next_state is None:
                continue
            dpsi = _wrap_pi(float(next_state["heading_rad"]) - float(current_state["heading_rad"]))
            spd_sp = math.sqrt(
                float(next_state["vx_e_mps"]) ** 2
                + float(next_state["vy_n_mps"]) ** 2
                + float(next_state["vz_u_mps"]) ** 2
            )
            commands.append(
                {
                    "entity_id": oid,
                    "dpsi_rad": float(dpsi),
                    "alt_sp_m": float(next_state["z_u_m"]),
                    "spd_sp_mps": float(spd_sp),
                }
            )
        if commands:
            actions.append({"t_s": float(round(tick * dt, 6)), "commands": commands})

    if not frames:
        raise ValueError("No frames produced from extracted entities")

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": acmi_path.stem,
        "source": {
            "kind": "acmi",
            "file": str(acmi_path),
            "reference_lat": ref_lat,
            "reference_lon": ref_lon,
        },
        "time_step_s": float(dt),
        "coordinate_frame": "ENU",
        "entities": entity_meta,
        "frames": frames,
        "actions": actions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ACMI to shared episode schema JSON")
    ap.add_argument("--acmi", required=True, help="Input ACMI file (.acmi or zipped)")
    ap.add_argument("--out", required=True, help="Output episode JSON file")
    ap.add_argument("--dt", type=float, default=0.5, help="Episode frame interval in seconds (default: 0.5)")
    ap.add_argument(
        "--type_filter",
        type=str,
        default="Air+FixedWing",
        help="Only keep entities with this Type substring (default: Air+FixedWing)",
    )
    ap.add_argument(
        "--max_entities",
        type=int,
        default=None,
        help="Optional max number of entities to include (longest tracks first)",
    )
    ap.add_argument(
        "--trim_speed",
        type=float,
        default=0.0,
        help="Trim leading samples below this 3D speed (m/s); 0 disables (default: 0)",
    )
    args = ap.parse_args()

    acmi_path = Path(args.acmi)
    out_path = Path(args.out)
    episode = build_episode(
        acmi_path=acmi_path,
        dt=args.dt,
        type_filter=args.type_filter,
        max_entities=args.max_entities,
        trim_speed_mps=args.trim_speed,
    )
    dump_episode(out_path, episode, indent=2)

    n_frames = len(episode["frames"])
    n_actions = len(episode["actions"])
    n_entities = len(episode["entities"])
    duration_s = float(episode["frames"][-1]["t_s"]) if n_frames else 0.0
    print(
        f"[OK] exported episode: {out_path} "
        f"(entities={n_entities}, frames={n_frames}, actions={n_actions}, duration_s={duration_s:.1f})"
    )


if __name__ == "__main__":
    main()
