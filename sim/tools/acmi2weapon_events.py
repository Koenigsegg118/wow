#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract missile launch events from Tacview ACMI for later AFSIM weapon rebuild."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

from sim.tools import acmi2tspi


RELATION_KEYS = ("Parent", "ParentId", "ParentID", "Source", "Launcher", "Target", "LockedTarget")


def parse_acmi_objects_and_tracks(lines, start_idx):
    t = 0.0
    objects = {}
    tracks = {}
    state_by_raw_oid = {}
    identity_keys = getattr(acmi2tspi, "_IDENTITY_KEYS")
    identity_tuple = getattr(acmi2tspi, "_identity_tuple")

    for idx in range(start_idx, len(lines)):
        raw = lines[idx].strip()
        if not raw:
            continue
        if raw.startswith("#"):
            if len(raw) > 1:
                try:
                    t = float(raw[1:])
                except Exception:
                    pass
            continue
        if raw.startswith("T="):
            try:
                t = float(raw[2:])
            except Exception:
                pass
            continue

        parts = raw.split(",")
        if len(parts) < 2:
            continue

        raw_oid = parts[0].strip()
        props = {}
        for kv in parts[1:]:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            props[k.strip()] = v.strip()

        state = state_by_raw_oid.get(raw_oid)
        if state is None:
            oid = raw_oid
            obj = {"source_id": raw_oid, "segment_index": 1}
            objects[oid] = obj
            tracks.setdefault(oid, [])
            state = {"oid": oid, "segment_index": 1, "identity": None}
            state_by_raw_oid[raw_oid] = state
        else:
            oid = state["oid"]
            obj = objects[oid]

        has_identity_update = any(k in props for k in identity_keys)
        if has_identity_update:
            candidate = dict(obj)
            for k, v in props.items():
                if k != "T":
                    candidate[k] = v
            candidate["side"] = acmi2tspi.side_from_props(candidate)
            candidate_identity = identity_tuple(candidate)

            if state["identity"] is not None and candidate_identity != state["identity"] and len(tracks.get(oid, [])) > 0:
                state["segment_index"] += 1
                oid = f"{raw_oid}__seg{state['segment_index']}"
                obj = {"source_id": raw_oid, "segment_index": state["segment_index"]}
                objects[oid] = obj
                tracks.setdefault(oid, [])
                state["oid"] = oid

            for k, v in props.items():
                if k != "T":
                    obj[k] = v
            obj["side"] = acmi2tspi.side_from_props(obj)
            state["identity"] = identity_tuple(obj)
        else:
            for k, v in props.items():
                if k != "T":
                    obj[k] = v
            obj.setdefault("source_id", raw_oid)
            obj.setdefault("segment_index", state["segment_index"])
            obj["side"] = acmi2tspi.side_from_props(obj)
            if state["identity"] is None:
                state["identity"] = identity_tuple(obj)

        if "T" not in props:
            continue

        vals = props["T"].split("|")
        try:
            if len(vals) >= 3:
                lon = float(vals[0])
                lat = float(vals[1])
                alt = float(vals[2])
                roll_deg = float(vals[3]) if len(vals) > 3 and vals[3] != "" else 0.0
                pitch_deg = float(vals[4]) if len(vals) > 4 and vals[4] != "" else 0.0
                yaw_deg = float(vals[5]) if len(vals) > 5 and vals[5] != "" else None
                tracks.setdefault(oid, []).append((t, lon, lat, alt, roll_deg, pitch_deg, yaw_deg))
        except Exception:
            continue

    return objects, tracks


def _is_weapon(obj):
    return "weapon+" in (obj.get("Type", "") or "").lower()


def _is_aircraft(obj):
    typ = (obj.get("Type", "") or "").lower()
    return typ.startswith("air+") and not _is_weapon(obj)


def _build_track_cache(track, ref_lat, ref_lon):
    times = []
    xs = []
    ys = []
    zs = []
    for t, lon, lat, alt, _roll_deg, _pitch_deg, _yaw_deg in track:
        x, y = acmi2tspi.latlon_to_enu(lat, lon, ref_lat, ref_lon)
        times.append(float(t))
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(alt))
    return {"times": times, "xs": xs, "ys": ys, "zs": zs}


def _interp_track(cache, t):
    times = cache["times"]
    if not times or t < times[0] or t > times[-1]:
        return None
    idx = bisect.bisect_left(times, t)
    if idx < len(times) and times[idx] == t:
        return (cache["xs"][idx], cache["ys"][idx], cache["zs"][idx])
    if idx == 0 or idx >= len(times):
        return None
    t0 = times[idx - 1]
    t1 = times[idx]
    if t1 <= t0:
        return (cache["xs"][idx - 1], cache["ys"][idx - 1], cache["zs"][idx - 1])
    alpha = (t - t0) / (t1 - t0)
    x = cache["xs"][idx - 1] + alpha * (cache["xs"][idx] - cache["xs"][idx - 1])
    y = cache["ys"][idx - 1] + alpha * (cache["ys"][idx] - cache["ys"][idx - 1])
    z = cache["zs"][idx - 1] + alpha * (cache["zs"][idx] - cache["zs"][idx - 1])
    return (x, y, z)


def _distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)


def _resolve_relation_id(value, objects, desired_side=None):
    if not value:
        return None
    if value in objects and (desired_side is None or objects[value].get("side") == desired_side):
        return value

    matches = []
    for oid, obj in objects.items():
        if obj.get("source_id") == value and (desired_side is None or obj.get("side") == desired_side):
            matches.append((obj.get("segment_index", 1), oid))
    if not matches:
        return None
    matches.sort()
    return matches[0][1]


def _best_shooter_candidate(weapon_obj, weapon_cache, objects, track_cache):
    launch_time = weapon_cache["times"][0]
    launch_pos = (weapon_cache["xs"][0], weapon_cache["ys"][0], weapon_cache["zs"][0])
    best = None

    for oid, obj in objects.items():
        if oid == weapon_obj.get("id") or not _is_aircraft(obj):
            continue
        if obj.get("side") != weapon_obj.get("side"):
            continue
        cache = track_cache.get(oid)
        if cache is None:
            continue
        pos = _interp_track(cache, launch_time)
        if pos is None:
            continue
        dist = _distance(launch_pos, pos)
        if best is None or dist < best["distance_m"]:
            best = {"id": oid, "distance_m": dist}
    return best


def _best_target_candidate(weapon_obj, weapon_cache, objects, track_cache):
    best = None
    weapon_positions = list(zip(weapon_cache["times"], weapon_cache["xs"], weapon_cache["ys"], weapon_cache["zs"]))

    for oid, obj in objects.items():
        if not _is_aircraft(obj):
            continue
        if obj.get("side") == weapon_obj.get("side") or obj.get("side") == "unknown":
            continue
        cache = track_cache.get(oid)
        if cache is None:
            continue

        min_dist = None
        for t, x, y, z in weapon_positions:
            pos = _interp_track(cache, t)
            if pos is None:
                continue
            dist = _distance((x, y, z), pos)
            if min_dist is None or dist < min_dist:
                min_dist = dist

        if min_dist is None:
            continue
        if best is None or min_dist < best["distance_m"]:
            best = {"id": oid, "distance_m": min_dist}
    return best


def _slot_lookup(slots_meta):
    by_id = {}
    if not slots_meta:
        return by_id
    payload = json.loads(Path(slots_meta).read_text(encoding="utf-8"))
    slots = payload.get("slots", {})
    for slot, info in slots.items():
        oid = info.get("id")
        if oid:
            by_id[oid] = slot
        source_id = info.get("source_id")
        if source_id and source_id not in by_id:
            by_id[source_id] = slot
    return by_id


def _entity_payload(oid, obj, slot_by_id, method, distance_m=None):
    if oid is None or obj is None:
        return {
            "id": None,
            "slot": None,
            "name": None,
            "group": None,
            "pilot": None,
            "method": method,
            "distance_m": distance_m,
        }
    return {
        "id": oid,
        "slot": slot_by_id.get(oid) or slot_by_id.get(obj.get("source_id")),
        "name": obj.get("Name"),
        "group": obj.get("Group"),
        "pilot": obj.get("Pilot"),
        "method": method,
        "distance_m": distance_m,
    }


def extract_weapon_events(acmi_path: Path, slots_meta: Path | None = None):
    lines = acmi2tspi.read_acmi_text(acmi_path).splitlines()
    header, start_idx = acmi2tspi.parse_header(lines)
    objects, tracks = parse_acmi_objects_and_tracks(lines, start_idx)
    ref_lat, ref_lon = acmi2tspi.extract_reference_latlon(lines)
    acmi2tspi.maybe_apply_reference_offset(tracks, ref_lat, ref_lon)

    if ref_lat is None or ref_lon is None:
        for pts in tracks.values():
            if pts:
                ref_lon = pts[0][1]
                ref_lat = pts[0][2]
                break
    if ref_lat is None or ref_lon is None:
        raise SystemExit("Could not determine a reference latitude/longitude from ACMI.")

    slot_by_id = _slot_lookup(slots_meta) if slots_meta else {}

    track_cache = {}
    for oid, pts in tracks.items():
        if pts:
            track_cache[oid] = _build_track_cache(pts, ref_lat, ref_lon)

    events = []
    for oid, obj in objects.items():
        obj["id"] = oid
        if not _is_weapon(obj):
            continue
        weapon_cache = track_cache.get(oid)
        if weapon_cache is None or not weapon_cache["times"]:
            continue

        shooter_oid = None
        shooter_method = "unresolved"
        shooter_distance = None
        for key in ("Parent", "ParentId", "ParentID", "Source", "Launcher"):
            shooter_oid = _resolve_relation_id(obj.get(key), objects, desired_side=obj.get("side"))
            if shooter_oid is not None:
                shooter_method = key.lower()
                break
        if shooter_oid is None:
            shooter_guess = _best_shooter_candidate(obj, weapon_cache, objects, track_cache)
            if shooter_guess is not None:
                shooter_oid = shooter_guess["id"]
                shooter_distance = shooter_guess["distance_m"]
                shooter_method = "nearest_same_side_launch"

        target_oid = None
        target_method = "unresolved"
        target_distance = None
        for key in ("Target", "LockedTarget"):
            target_oid = _resolve_relation_id(obj.get(key), objects)
            if target_oid is not None:
                target_method = key.lower()
                break
        if target_oid is None:
            target_guess = _best_target_candidate(obj, weapon_cache, objects, track_cache)
            if target_guess is not None:
                target_oid = target_guess["id"]
                target_distance = target_guess["distance_m"]
                target_method = "closest_approach_opposite_side"

        launch_idx = 0
        end_idx = len(weapon_cache["times"]) - 1
        events.append(
            {
                "missile_id": oid,
                "source_id": obj.get("source_id"),
                "segment_index": obj.get("segment_index"),
                "missile_name": obj.get("Name"),
                "missile_type": obj.get("Type"),
                "side": obj.get("side"),
                "launch_time": weapon_cache["times"][launch_idx],
                "end_time": weapon_cache["times"][end_idx],
                "duration_s": weapon_cache["times"][end_idx] - weapon_cache["times"][launch_idx],
                "n_points": len(weapon_cache["times"]),
                "shooter": _entity_payload(shooter_oid, objects.get(shooter_oid), slot_by_id, shooter_method, shooter_distance),
                "target": _entity_payload(target_oid, objects.get(target_oid), slot_by_id, target_method, target_distance),
            }
        )

    events.sort(key=lambda item: (item["launch_time"], item["missile_id"]))
    return {
        "acmi": str(acmi_path),
        "reference_latitude": ref_lat,
        "reference_longitude": ref_lon,
        "n_events": len(events),
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract missile launch events from ACMI for AFSIM weapon rebuild")
    ap.add_argument("--acmi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots-meta", default="")
    args = ap.parse_args()

    acmi_path = Path(args.acmi)
    out_path = Path(args.out)
    slots_meta = Path(args.slots_meta) if args.slots_meta else None
    payload = extract_weapon_events(acmi_path, slots_meta=slots_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] extracted {payload['n_events']} weapon events -> {out_path}")


if __name__ == "__main__":
    main()
