#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Tacview ACMI to AFSIM TSPI trajectories with side/group manifests.

Output TSPI format (WSF_TSPI_MOVER compatible):
  <time_s> <lat_deg> <lon_deg> <alt_m> <speed_mps> <heading_rad> <pitch_rad> <roll_rad>
"""

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np

EARTH_R = 6378137.0


def read_acmi_text(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            cand = None
            best = -1
            for n in names:
                if n.lower().endswith(".txt") or "acmi" in n.lower():
                    info = z.getinfo(n)
                    if info.file_size > best:
                        best = info.file_size
                        cand = n
            if cand is None:
                cand = names[0]
            data = z.read(cand)
        return data.decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


_header_kv = re.compile(r"^([^=]+)=(.*)$")


def parse_header(lines):
    meta = {}
    i = 0
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("T=") or line.startswith("0,") or re.match(r"^\w+,", line):
            break
        m = _header_kv.match(line)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
    return meta, i


def extract_reference_latlon(lines):
    ref_lat = None
    ref_lon = None
    for raw in lines[:200]:
        line = raw.strip()
        if "ReferenceLatitude=" in line:
            try:
                ref_lat = float(line.split("ReferenceLatitude=", 1)[1].split(",")[0])
            except Exception:
                pass
        if "ReferenceLongitude=" in line:
            try:
                ref_lon = float(line.split("ReferenceLongitude=", 1)[1].split(",")[0])
            except Exception:
                pass
        if ref_lat is not None and ref_lon is not None:
            break
    return ref_lat, ref_lon


def side_from_props(props):
    color = (props.get("Color", "") or "").strip().lower()
    coalition = (props.get("Coalition", "") or "").strip().lower()
    group = (props.get("Group", "") or "").strip().lower()
    name = (props.get("Name", "") or "").strip().lower()

    if "blue" in color:
        return "blue"
    if "red" in color:
        return "red"
    if coalition in ("blue", "red"):
        return coalition
    if "blue" in group or "blue" in name:
        return "blue"
    if "red" in group or "red" in name:
        return "red"
    if coalition == "enemies":
        return "blue"
    if coalition == "allies":
        return "red"
    return "unknown"


def parse_acmi_objects_and_tracks(lines, start_idx):
    t = 0.0
    objects = {}
    tracks = {}
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
        oid = parts[0].strip()
        props = {}
        for kv in parts[1:]:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            props[k.strip()] = v.strip()

        obj = objects.setdefault(oid, {})
        for k in ("Name", "Type", "Coalition", "Country", "Group", "Color", "Pilot"):
            if k in props:
                obj[k] = props[k]
        obj["side"] = side_from_props(obj)

        if "T" in props:
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
                pass
    return objects, tracks


def maybe_apply_reference_offset(tracks, ref_lat, ref_lon):
    if ref_lat is None or ref_lon is None:
        return
    for oid, pts in list(tracks.items()):
        if not pts:
            continue
        lon_arr = np.array([p[1] for p in pts], dtype=np.float64)
        lat_arr = np.array([p[2] for p in pts], dtype=np.float64)
        # Tacview exports may encode lon/lat as deltas around ReferenceLon/Lat.
        looks_relative = (
            np.nanmax(np.abs(lon_arr)) < 20.0
            and np.nanmax(np.abs(lat_arr)) < 20.0
            and (abs(ref_lat) >= 20.0 or abs(ref_lon) >= 20.0)
        )
        if not looks_relative:
            continue
        new_pts = []
        for p in pts:
            t, lon, lat, alt, roll_deg, pitch_deg, yaw_deg = p
            new_pts.append((t, lon + ref_lon, lat + ref_lat, alt, roll_deg, pitch_deg, yaw_deg))
        tracks[oid] = new_pts


def latlon_to_enu(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x_e = (lon - lon0) * math.cos(0.5 * (lat + lat0)) * EARTH_R
    y_n = (lat - lat0) * EARTH_R
    return x_e, y_n


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_series(a):
    out = np.asarray(a, dtype=np.float64).copy()
    if out.size == 0:
        return out
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > math.pi:
            out[i:] -= 2.0 * math.pi
        elif d < -math.pi:
            out[i:] += 2.0 * math.pi
    return out


def _smooth(arr, window):
    """Simple moving average with edge padding."""
    if window < 3 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def compute_attitude_from_trajectory(grid, x, y, z, dt, heading, smooth_window=21):
    """Derive pitch and roll from position trajectory.

    pitch = flight-path angle = atan2(vz, v_horizontal)
    roll  = coordinated-turn bank angle = atan(v * heading_rate / g)
    """
    g = 9.80665
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    v_horiz = np.sqrt(vx ** 2 + vy ** 2)
    v_horiz = np.maximum(v_horiz, 1.0)

    pitch = np.arctan2(vz, v_horiz)
    pitch = _smooth(pitch, smooth_window)
    pitch = np.clip(pitch, -math.radians(80), math.radians(80))

    hdg_unwrap = unwrap_series(heading)
    hdg_rate = np.gradient(hdg_unwrap, dt)
    hdg_rate = _smooth(hdg_rate, smooth_window)
    speed_3d = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    speed_3d = np.maximum(speed_3d, 1.0)
    roll = np.arctan2(speed_3d * hdg_rate, g)
    roll = _smooth(roll, smooth_window)
    roll = np.clip(roll, -math.radians(75), math.radians(75))

    return pitch, roll


def build_manifest(objects, tracks):
    out = {"blue": {}, "red": {}, "unknown": {}}
    for oid, obj in objects.items():
        side = obj.get("side", "unknown")
        group = obj.get("Group", "") or "__ungrouped__"
        item = {
            "id": oid,
            "name": obj.get("Name", oid),
            "group": group,
            "type": obj.get("Type", ""),
            "coalition": obj.get("Coalition", ""),
            "color": obj.get("Color", ""),
            "pilot": obj.get("Pilot", ""),
            "n_points": len(tracks.get(oid, [])),
        }
        out.setdefault(side, {}).setdefault(group, []).append(item)
    for side in out:
        for group in out[side]:
            out[side][group] = sorted(out[side][group], key=lambda x: (-x["n_points"], x["name"], x["id"]))
    return out


def parse_slot_map(items):
    slot_map = {}
    for kv in items:
        if "=" not in kv:
            raise ValueError(f"Bad --map item: {kv}")
        k, v = kv.split("=", 1)
        slot_map[k.strip()] = v.strip()
    return slot_map


def resolve_slot_target(spec, default_group):
    if "/" in spec:
        group, name = spec.split("/", 1)
        return group.strip(), name.strip()
    return default_group, spec.strip()


def select_object(manifest, side, group, name):
    side_data = manifest.get(side, {})
    cand = side_data.get(group, [])
    if not cand:
        return None
    if name == "*":
        return cand[0]["id"]
    name_low = name.lower()
    for item in cand:
        if item["name"].lower() == name_low:
            return item["id"]
    return None


def resample_track(pts, dt, trim_speed_mps=50.0):
    pts = sorted(pts, key=lambda x: x[0])
    t = np.array([p[0] for p in pts], dtype=np.float64)
    lon = np.array([p[1] for p in pts], dtype=np.float64)
    lat = np.array([p[2] for p in pts], dtype=np.float64)
    alt = np.array([p[3] for p in pts], dtype=np.float64)
    # pitch/roll from ACMI are often accumulated rotation values, not physical attitude.
    # Physical pitch/roll will be computed from the resampled trajectory later.
    roll = np.zeros(len(pts), dtype=np.float64)
    pitch = np.zeros(len(pts), dtype=np.float64)
    yaw_raw = np.array([np.nan if p[6] is None else math.radians(p[6]) for p in pts], dtype=np.float64)

    if len(t) < 3:
        return None
    t0 = float(t[0])
    t1 = float(t[-1])
    if t1 - t0 < dt * 4:
        return None

    # --- trim leading stationary phase ---
    if trim_speed_mps > 0:
        # compute rough 3D speed at each raw sample via finite difference
        x_raw = np.zeros(len(t), dtype=np.float64)
        y_raw = np.zeros(len(t), dtype=np.float64)
        lat0_r, lon0_r = float(lat[0]), float(lon[0])
        for i in range(len(t)):
            ex, ny = latlon_to_enu(lat[i], lon[i], lat0_r, lon0_r)
            x_raw[i] = ex
            y_raw[i] = ny
        z_raw = alt
        dt_raw = np.diff(t)
        dt_raw[dt_raw < 1e-6] = 1e-6
        vx_raw = np.diff(x_raw) / dt_raw
        vy_raw = np.diff(y_raw) / dt_raw
        vz_raw = np.diff(z_raw) / dt_raw
        spd_raw = np.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)  # len = N-1
        # find first sample index where speed (between i and i+1) >= threshold
        trim_idx = 0
        for i in range(len(spd_raw)):
            if spd_raw[i] >= trim_speed_mps:
                trim_idx = i
                break
        if trim_idx > 0:
            pts = pts[trim_idx:]
            t = t[trim_idx:]
            lon = lon[trim_idx:]
            lat = lat[trim_idx:]
            alt = alt[trim_idx:]
            roll = roll[trim_idx:]
            pitch = pitch[trim_idx:]
            yaw_raw = yaw_raw[trim_idx:]

    t_offset = float(t[0])
    t = t - t_offset  # re-zero time
    t0 = 0.0
    t1 = float(t[-1])
    if t1 < dt * 4:
        return None

    grid = np.arange(t0, t1 + 1e-9, dt, dtype=np.float64)
    lon_i = np.interp(grid, t, lon)
    lat_i = np.interp(grid, t, lat)
    alt_i = np.interp(grid, t, alt)
    # heading source priority: yaw in ACMI -> fallback by ENU velocity
    heading = None
    valid_yaw = ~np.isnan(yaw_raw)
    if valid_yaw.sum() >= 3:
        yaw_u = unwrap_series(np.interp(grid, t[valid_yaw], unwrap_series(yaw_raw[valid_yaw])))
        heading = yaw_u
    else:
        x = np.zeros_like(grid)
        y = np.zeros_like(grid)
        lat0 = float(lat_i[0])
        lon0 = float(lon_i[0])
        for i in range(len(grid)):
            x[i], y[i] = latlon_to_enu(lat_i[i], lon_i[i], lat0, lon0)
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        heading = unwrap_series(np.arctan2(vx, vy))

    # ENU coordinates for speed and attitude derivation
    x = np.zeros_like(grid)
    y = np.zeros_like(grid)
    lat0 = float(lat_i[0])
    lon0 = float(lon_i[0])
    for i in range(len(grid)):
        x[i], y[i] = latlon_to_enu(lat_i[i], lon_i[i], lat0, lon0)
    z = alt_i
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    speed_from_pos = np.sqrt(vx * vx + vy * vy + vz * vz)
    speed = speed_from_pos.copy()
    speed = np.maximum(speed, 1.0)

    pitch_i, roll_i = compute_attitude_from_trajectory(grid, x, y, z, dt, heading)

    return {
        "t": grid - grid[0],
        "lat": lat_i,
        "lon": lon_i,
        "alt": alt_i,
        "speed": speed,
        "heading": heading,
        "pitch": pitch_i,
        "roll": roll_i,
    }


def write_tspi(path, rs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(len(rs["t"])):
            f.write(
                f"{rs['t'][i]:.2f} {rs['lat'][i]:.7f} {rs['lon'][i]:.7f} "
                f"{rs['alt'][i]:.3f} {rs['speed'][i]:.3f} {rs['heading'][i]:.7f} "
                f"{rs['pitch'][i]:.7f} {rs['roll'][i]:.7f}\n"
            )


def write_csv(path, rs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "lat_deg", "lon_deg", "alt_m", "speed_mps", "heading_rad", "pitch_rad", "roll_rad"])
        for i in range(len(rs["t"])):
            w.writerow(
                [
                    f"{rs['t'][i]:.3f}",
                    f"{rs['lat'][i]:.7f}",
                    f"{rs['lon'][i]:.7f}",
                    f"{rs['alt'][i]:.3f}",
                    f"{rs['speed'][i]:.3f}",
                    f"{rs['heading'][i]:.7f}",
                    f"{rs['pitch'][i]:.7f}",
                    f"{rs['roll'][i]:.7f}",
                ]
            )


def main():
    ap = argparse.ArgumentParser(description="ACMI -> side/group manifest + AFSIM TSPI replay files")
    ap.add_argument("--acmi", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--map", nargs="*", default=[], help="slot mapping, e.g. blue_1=BLUE-GROUP/F-16C_50")
    ap.add_argument("--blue-group", default="", help="default group for map values without group/")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--trim-speed", type=float, default=50.0,
                    help="trim leading frames where 3D speed < this (m/s). 0=disable (default: 50.0)")
    args = ap.parse_args()

    acmi = Path(args.acmi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    txt = read_acmi_text(acmi)
    lines = txt.splitlines()
    header, start_idx = parse_header(lines)
    objects, tracks = parse_acmi_objects_and_tracks(lines, start_idx)
    ref_lat = None
    ref_lon = None
    try:
        ref_lat = float(header.get("ReferenceLatitude", "nan"))
        ref_lon = float(header.get("ReferenceLongitude", "nan"))
        if math.isnan(ref_lat) or math.isnan(ref_lon):
            ref_lat, ref_lon = None, None
    except Exception:
        ref_lat, ref_lon = None, None
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = extract_reference_latlon(lines)
    maybe_apply_reference_offset(tracks, ref_lat, ref_lon)
    manifest = build_manifest(objects, tracks)

    (outdir / "groups_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "blue_groups.json").write_text(json.dumps(manifest.get("blue", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "red_groups.json").write_text(json.dumps(manifest.get("red", {}), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] wrote manifests into {outdir}")

    if args.list_only:
        for side in ("blue", "red", "unknown"):
            print(f"\n[{side}]")
            for group, items in sorted(manifest.get(side, {}).items()):
                names = ", ".join([x["name"] for x in items[:6]])
                print(f"  {group}: {len(items)} ({names})")
        return

    slot_map = parse_slot_map(args.map)
    if not slot_map:
        raise SystemExit("No --map provided. Use --list-only first or pass --map blue_1=GROUP/NAME blue_2=GROUP/NAME")

    selected = {}
    for slot, spec in slot_map.items():
        if not slot.startswith("blue_"):
            raise SystemExit(f"Only blue slots are supported now: {slot}")
        group, name = resolve_slot_target(spec, args.blue_group)
        if not group:
            raise SystemExit(f"Missing group in mapping '{slot}={spec}'. Use GROUP/NAME or --blue-group.")
        oid = select_object(manifest, "blue", group, name)
        if oid is None:
            raise SystemExit(f"Cannot find blue object for slot={slot}, group={group}, name={name}")
        selected[slot] = oid

    export_meta = {"acmi": str(acmi), "dt": args.dt, "slots": {}}
    for slot, oid in selected.items():
        pts = tracks.get(oid, [])
        rs = resample_track(pts, args.dt, trim_speed_mps=args.trim_speed)
        if rs is None:
            raise SystemExit(f"Track too short or invalid for {slot} (oid={oid})")
        tspi_path = outdir / f"tspi_{slot}.tspi"
        csv_path = outdir / f"tspi_{slot}.csv"
        write_tspi(tspi_path, rs)
        write_csv(csv_path, rs)
        obj = objects.get(oid, {})
        export_meta["slots"][slot] = {
            "id": oid,
            "name": obj.get("Name", oid),
            "group": obj.get("Group", ""),
            "type": obj.get("Type", ""),
            "coalition": obj.get("Coalition", ""),
            "color": obj.get("Color", ""),
            "n_points": len(rs["t"]),
            "tspi_file": str(tspi_path),
            "csv_file": str(csv_path),
        }
        print(f"[OK] {slot}: {tspi_path}")

    (outdir / "export_meta.json").write_text(json.dumps(export_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] exported {len(selected)} blue TSPI tracks.")


if __name__ == "__main__":
    main()
