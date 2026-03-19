#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tacview .acmi (zip) -> smooth setpoint dataset (.npz)

Default: 2 Hz decision rate (dt=0.5s), 2.0s horizon (H=2s, horizon_steps=4)
Action (metric): [dpsi_rad, alt_sp_m, spd_sp_mps]
  dpsi_rad = wrap_pi(track_angle_unwrapped[t+horizon_steps] - track_angle_unwrapped[t])
Obs (metric ENU): [x_e,y_n,z_u,vx_e,vy_n,vz_u,track_angle_rad_unwrapped,ground_speed_mps]
  track_angle_rad_unwrapped = atan2(vx_east, vy_north), 0=N CW+, continuously unwrapped
  ground_speed_mps = sqrt(vx_east^2 + vy_north^2)  (2-D horizontal only)

Usage:
  python acmi_to_dt_dataset_smooth.py --in a.acmi b.acmi --out out.npz
"""

import argparse, zipfile, re, math, json, sys
from pathlib import Path
import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from shared.conventions.heading_convention import CONVENTION
from entity_filter import EntityFilter, DEFAULT_EXCLUDE_REGEX
import dataset_default_config as CFG

EARTH_R = 6378137.0  # meters

def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def unwrap_angle(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    out = a.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i-1]
        if d > math.pi:
            out[i:] -= 2.0*math.pi
        elif d < -math.pi:
            out[i:] += 2.0*math.pi
    return out

def compute_dpsi_series(heading_u, horizon_steps, dpsi_sign=1.0):
    if dpsi_sign not in (-1.0, 1.0):
        raise ValueError("dpsi_sign must be +1 or -1")
    return np.array(
        [dpsi_sign * wrap_pi(heading_u[i + horizon_steps] - heading_u[i]) for i in range(len(heading_u) - horizon_steps)],
        dtype=np.float32,
    )

def latlon_to_enu_m(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    x_e = (lon - lon0) * math.cos(0.5*(lat+lat0)) * EARTH_R
    y_n = (lat - lat0) * EARTH_R
    return x_e, y_n

def read_acmi_text(path):
    p = Path(path)
    with open(p, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":
        with zipfile.ZipFile(p, "r") as z:
            names = z.namelist()
            cand = None; best = -1
            for n in names:
                if n.lower().endswith(".txt") or "acmi" in n.lower():
                    info = z.getinfo(n)
                    if info.file_size > best:
                        best = info.file_size; cand = n
            if cand is None:
                cand = names[0]
            data = z.read(cand)
        return data.decode("utf-8", errors="replace")
    return p.read_text(encoding="utf-8", errors="replace")

_header_kv = re.compile(r"^([^=]+)=(.*)$")

def parse_header(lines):
    meta = {}; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1; continue
        if line.startswith("T=") or re.match(r"^\w+,", line):
            break
        m = _header_kv.match(line)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
        i += 1
    return meta, i

def parse_object_updates(lines, start_idx):
    t = 0.0
    objects = {}
    tracks = {}
    for idx in range(start_idx, len(lines)):
        raw = lines[idx].strip()
        if not raw or raw.startswith("#"):
            # Tacview ACMI often uses '#<time_seconds>' markers.
            if raw.startswith("#") and len(raw) > 1:
                try:
                    t = float(raw[1:])
                except:
                    pass
            continue
        if raw.startswith("T="):
            try: t = float(raw[2:])
            except: pass
            continue
        parts = raw.split(",")
        if len(parts) < 2:
            continue
        oid = parts[0].strip()
        kvs = parts[1:]
        props = {}
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
            try:
                vals = props["T"].split("|")
                if len(vals) >= 3:
                    lon = float(vals[0]); lat = float(vals[1]); alt = float(vals[2])
                    tracks.setdefault(oid, []).append((t, lon, lat, alt))
            except:
                pass
    return objects, tracks

def build_entity_tracks(objects, tracks, ref_lat, ref_lon):
    entity = {}
    for oid, pts in tracks.items():
        if len(pts) < 5:
            continue
        pts = sorted(pts, key=lambda x: x[0])
        t_arr = np.array([p[0] for p in pts], dtype=np.float64)
        lon_arr = np.array([p[1] for p in pts], dtype=np.float64)
        lat_arr = np.array([p[2] for p in pts], dtype=np.float64)
        alt_arr = np.array([p[3] for p in pts], dtype=np.float64)

        # Heuristic: small magnitude => delta degrees relative to reference
        if np.nanmax(np.abs(lon_arr)) < 5.0 and np.nanmax(np.abs(lat_arr)) < 5.0:
            lon_abs = ref_lon + lon_arr
            lat_abs = ref_lat + lat_arr
        else:
            lon_abs = lon_arr
            lat_abs = lat_arr

        x_e = np.zeros_like(lon_abs)
        y_n = np.zeros_like(lat_abs)
        for i in range(len(lon_abs)):
            x_e[i], y_n[i] = latlon_to_enu_m(lat_abs[i], lon_abs[i], ref_lat, ref_lon)
        z_u = alt_arr

        info = objects.get(oid, {})
        entity[oid] = dict(
            id=oid,
            name=info.get("Name", oid),
            type=info.get("Type", ""),
            coalition=info.get("Coalition", ""),
            t=t_arr, x=x_e, y=y_n, z=z_u
        )
    return entity

def resample_entity(ent, dt=0.5):
    t = ent["t"]
    if len(t) < 5:
        return None
    t0, t1 = float(t[0]), float(t[-1])
    if t1 - t0 < 5.0:
        return None
    grid = np.arange(t0, t1 + 1e-9, dt, dtype=np.float64)
    x = np.interp(grid, t, ent["x"])
    y = np.interp(grid, t, ent["y"])
    z = np.interp(grid, t, ent["z"])
    vx = np.gradient(x, dt)
    vy = np.gradient(y, dt)
    vz = np.gradient(z, dt)
    spd = np.sqrt(vx*vx + vy*vy)
    heading = np.arctan2(vx, vy)       # 0=N, +E
    heading_u = unwrap_angle(heading)  # for differencing
    return dict(id=ent["id"], name=ent["name"], type=ent["type"], coalition=ent["coalition"],
                t=grid, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, spd=spd, heading_u=heading_u)

def make_windows(obs, act, seq_len=20, stride=5):
    Xo, Xa = [], []
    for s in range(0, len(obs) - seq_len + 1, stride):
        e = s + seq_len
        Xo.append(obs[s:e]); Xa.append(act[s:e])
    if not Xo:
        return None, None
    return np.stack(Xo, 0), np.stack(Xa, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt",          type=float, default=CFG.DT)
    ap.add_argument("--H_sec",       type=float, default=CFG.H_SEC)
    ap.add_argument("--seq_len",     type=int,   default=CFG.SEQ_LEN)
    ap.add_argument("--stride",      type=int,   default=CFG.STRIDE)
    ap.add_argument("--min_speed",   type=float, default=CFG.MIN_SPEED)
    ap.add_argument("--type_filter", type=str,   default=CFG.TYPE_FILTER)
    ap.add_argument("--dpsi_sign",   type=float, default=CFG.DPSI_SIGN, choices=[-1.0, 1.0],
                    help="Sign convention for dpsi label: +1 keep, -1 flip")
    # Entity filter args (defaults from dataset_default_config)
    ap.add_argument("--include_regex", type=str, default=None)
    ap.add_argument("--exclude_regex", type=str, default=None,
                    help="Exclude by Name|Type regex. Default (when exclude_orbiters=1): CFG.EXCLUDE_REGEX")
    ap.add_argument("--exclude_orbiters", type=int, default=int(CFG.EXCLUDE_ORBITERS), choices=[0, 1])
    ap.add_argument("--orb_alt_std_m",    type=float, default=CFG.ORB_ALT_STD_M)
    ap.add_argument("--orb_spd_std_mps",  type=float, default=CFG.ORB_SPD_STD_MPS)
    ap.add_argument("--orb_dpsi_p95_rad", type=float, default=CFG.ORB_DPSI_P95_RAD)
    ap.add_argument("--orb_min_steps",    type=int,   default=CFG.ORB_MIN_STEPS)
    args = ap.parse_args()

    dt = args.dt
    k = int(round(args.H_sec / dt))
    if k < 1:
        raise SystemExit("H_sec too small")

    ent_filter = EntityFilter(
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        exclude_orbiters=bool(args.exclude_orbiters),
        orb_alt_std_m=args.orb_alt_std_m,
        orb_spd_std_mps=args.orb_spd_std_mps,
        orb_dpsi_p95_rad=args.orb_dpsi_p95_rad,
        orb_min_steps=args.orb_min_steps,
    )

    excluded_counts = {"regex_exclude": 0, "regex_include_miss": 0, "orbiter": 0}
    all_obs, all_act, meta_files = [], [], []

    for path in args.inputs:
        txt = read_acmi_text(path)
        lines = txt.splitlines()
        header, start_idx = parse_header(lines)
        ref_lat = float(header.get("ReferenceLatitude", "0") or "0")
        ref_lon = float(header.get("ReferenceLongitude", "0") or "0")

        objects, tracks = parse_object_updates(lines, start_idx)
        entities = build_entity_tracks(objects, tracks, ref_lat, ref_lon)

        for oid, ent in entities.items():
            typ = ent.get("type", "")
            name = ent.get("name", oid)
            if args.type_filter and (args.type_filter not in typ):
                continue

            # Name/type regex filter (fast, before resampling)
            keep, reason = ent_filter.check_name_type(name, typ)
            if not keep:
                excluded_counts[reason] = excluded_counts.get(reason, 0) + 1
                continue

            rs = resample_entity(ent, dt=dt)
            if rs is None:
                continue
            if len(rs["spd"]) <= k + 2:
                continue

            spd = rs["spd"]
            heading = rs["heading_u"]
            obs = np.stack([rs["x"], rs["y"], rs["z"], rs["vx"], rs["vy"], rs["vz"], heading, spd], axis=1).astype(np.float32)

            dpsi = compute_dpsi_series(heading, k, dpsi_sign=args.dpsi_sign)
            alt_sp = rs["z"][k:].astype(np.float32)
            spd_sp = spd[k:].astype(np.float32)
            act = np.stack([dpsi, alt_sp, spd_sp], axis=1).astype(np.float32)
            obs = obs[:-k]

            mask = obs[:, 7] >= args.min_speed
            if mask.sum() < args.seq_len:
                continue
            obs = obs[mask]; act = act[mask]

            # Orbiter behaviour filter (applied after speed filter on valid samples)
            if ent_filter.exclude_orbiters:
                valid_alt = obs[:, 2]
                valid_spd = obs[:, 7]
                valid_adpsi = np.abs(act[:, 0])
                n_valid = len(valid_spd)
                entity_stats = {
                    "n_after_filter": n_valid,
                    "alt":      {"std": float(valid_alt.std())  if n_valid > 1 else 9999.0},
                    "speed":    {"std": float(valid_spd.std())  if n_valid > 1 else 9999.0},
                    "abs_dpsi": {"p95": float(np.percentile(valid_adpsi, 95)) if n_valid > 0 else 9999.0},
                }
                is_orb, orb_reason = ent_filter.is_orbiter_from_stats(entity_stats)
                if is_orb:
                    excluded_counts["orbiter"] = excluded_counts.get("orbiter", 0) + 1
                    continue

            Xo, Xa = make_windows(obs, act, seq_len=args.seq_len, stride=args.stride)
            if Xo is None:
                continue
            all_obs.append(Xo); all_act.append(Xa)
            meta_files.append(dict(file=str(path), entity=rs["name"], type=rs["type"], coalition=rs["coalition"],
                                   n_steps=int(len(obs)), n_windows=int(Xo.shape[0])))

    if not all_obs:
        raise SystemExit("No data extracted. Check type_filter/min_speed.")

    obs_arr = np.concatenate(all_obs, axis=0)
    act_arr = np.concatenate(all_act, axis=0)

    obs_mean = obs_arr.reshape(-1, obs_arr.shape[-1]).mean(axis=0)
    obs_std  = obs_arr.reshape(-1, obs_arr.shape[-1]).std(axis=0) + 1e-6
    act_mean = act_arr.reshape(-1, act_arr.shape[-1]).mean(axis=0)
    act_std  = act_arr.reshape(-1, act_arr.shape[-1]).std(axis=0) + 1e-6

    meta = dict(
        schema="obs->action (smooth setpoint)",
        dt=dt, horizon_sec=args.H_sec, horizon_steps=k,
        seq_len=args.seq_len, stride=args.stride,
        obs_fields=["x_e_m","y_n_m","z_u_m","vx_e_mps","vy_n_mps","vz_u_mps","heading_rad_unwrapped","ground_speed_mps"],
        act_fields=["dpsi_rad","alt_sp_m","spd_sp_mps"],
        units=dict(x="m", y="m", z="m", v="m/s", heading="rad", dpsi="rad"),
        heading_convention=dict(
            frame=CONVENTION.heading_frame,
            zero_reference=CONVENTION.heading_zero_reference,
            positive_direction=CONVENTION.heading_positive_direction,
            atan2_order=CONVENTION.heading_atan2_order,
        ),
        dpsi_sign=args.dpsi_sign,
        min_speed_mps=args.min_speed,
        type_filter=args.type_filter,
        filter_config=ent_filter.to_dict(),
        excluded_counts=excluded_counts,
        files=meta_files,
    )

    np.savez_compressed(args.out,
        obs=obs_arr.astype(np.float32),
        action=act_arr.astype(np.float32),
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
        act_mean=act_mean.astype(np.float32),
        act_std=act_std.astype(np.float32),
        meta=json.dumps(meta).encode("utf-8"),
    )
    if any(excluded_counts.values()):
        print(f"Excluded entities: {excluded_counts}")
    print(f"Saved {args.out}: obs {obs_arr.shape}, action {act_arr.shape}")

if __name__ == "__main__":
    main()
