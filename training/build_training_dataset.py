#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_training_dataset.py — One-click fighter-only training set builder
=======================================================================
Scans wow/tra_data/**/*.acmi, applies the validated fighter-only filter,
and writes a ready-to-train NPZ + companion metadata files.

Default outputs (wow/datasets/):
  dt2hz_H2s_fighteronly.npz          Training tensors (obs, action, stats)
  dt2hz_H2s_fighteronly.meta.json    Full metadata (human-readable)
  dt2hz_H2s_fighteronly.filelist.txt Files that contributed windows
  dt2hz_H2s_fighteronly.rejected.txt Files/entities that were skipped
  dt2hz_H2s_fighteronly.stats.json   Distribution snapshot (alt/speed/|Δψ|)

NPZ schema (compatible with train_bc_transformer_smooth.py)
-----------------------------------------------------------
  obs        : float32 [N, T, 8]   — ENU state sequences
  action     : float32 [N, T, 3]   — [dpsi_rad, alt_sp_m, spd_sp_mps]
  obs_mean   : float32 [8]
  obs_std    : float32 [8]
  act_mean   : float32 [3]
  act_std    : float32 [3]
  meta       : bytes  — JSON-encoded full metadata

Usage
-----
  # Default (all tra_data, default filter)
  python wow/build_training_dataset.py

  # Custom output name
  python wow/build_training_dataset.py --out wow/datasets/myset.npz

  # Override filter rules
  python wow/build_training_dataset.py \\
      --exclude_regex "(?i)(A-50|E-3|AWACS|KC-135|Tanker)" \\
      --exclude_orbiters 1

  # Disable orbiter filter (keep all fixed-wing)
  python wow/build_training_dataset.py --exclude_orbiters 0

All CLI flags override dataset_default_config.py values.
"""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from acmi_to_dt_dataset_smooth import (  # noqa: E402
    read_acmi_text,
    parse_header,
    parse_object_updates,
    build_entity_tracks,
    resample_entity,
    make_windows,
    compute_dpsi_series,
)
from entity_filter import EntityFilter   # noqa: E402
import dataset_default_config as CFG     # noqa: E402


# ── JSON encoder ──────────────────────────────────────────────────────────────
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── Per-file entity extraction ────────────────────────────────────────────────
def process_file(
    path: Path,
    k: int,
    seq_len: int,
    stride: int,
    min_speed: float,
    dpsi_sign: float,
    entity_filter: EntityFilter,
    type_filter: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[dict], List[dict], Optional[str]]:
    """Parse one ACMI file and extract fighter-only training windows.

    Returns
    -------
    (obs_windows, act_windows, entity_metas, filter_log, error_str)
      error_str is None on success, a one-line message on failure.
    """
    try:
        txt    = read_acmi_text(str(path))
        lines  = txt.splitlines()
        header, start = parse_header(lines)
        ref_lat = float(header.get("ReferenceLatitude",  "0") or "0")
        ref_lon = float(header.get("ReferenceLongitude", "0") or "0")

        objects, tracks = parse_object_updates(lines, start)
        entities = build_entity_tracks(objects, tracks, ref_lat, ref_lon)

        obs_list:     List[np.ndarray] = []
        act_list:     List[np.ndarray] = []
        entity_metas: List[dict]       = []
        filter_log:   List[dict]       = []

        for oid, ent in entities.items():
            typ  = ent.get("type",      "")
            name = ent.get("name",      oid)
            coal = ent.get("coalition", "")

            if type_filter and type_filter not in typ:
                continue   # not a fixed-wing aircraft

            # ── Fast regex filter (before expensive resampling) ──────────────
            keep, reason = entity_filter.check_name_type(name, typ)
            if not keep:
                filter_log.append({"name": name, "type": typ, "reason": reason})
                continue

            rs = resample_entity(ent, dt=CFG.DT)
            if rs is None:
                continue
            if len(rs["spd"]) <= k + 2:
                continue

            spd     = rs["spd"]
            heading = rs["heading_u"]
            z       = rs["z"]

            obs = np.stack(
                [rs["x"], rs["y"], z, rs["vx"], rs["vy"], rs["vz"], heading, spd],
                axis=1,
            ).astype(np.float32)

            dpsi   = compute_dpsi_series(heading, k, dpsi_sign=dpsi_sign)
            alt_sp = z[k:].astype(np.float32)
            spd_sp = spd[k:].astype(np.float32)
            act    = np.stack([dpsi, alt_sp, spd_sp], axis=1).astype(np.float32)
            obs    = obs[:-k]

            # Apply min-speed + MAX_SPEED hard ceiling (mirrors analyze_acmi_coverage.py)
            mask = (obs[:, 7] >= min_speed) & (obs[:, 7] <= CFG.MAX_SPEED)
            if mask.sum() < seq_len:
                continue
            obs = obs[mask]
            act = act[mask]

            # ── Orbiter behaviour filter (post speed-filter) ─────────────────
            if entity_filter.exclude_orbiters:
                n_valid = len(obs)
                entity_stats = {
                    "n_after_filter": n_valid,
                    "alt":      {"std": float(obs[:, 2].std())
                                 if n_valid > 1 else 9999.0},
                    "speed":    {"std": float(obs[:, 7].std())
                                 if n_valid > 1 else 9999.0},
                    "abs_dpsi": {"p95": float(np.percentile(np.abs(act[:, 0]), 95))
                                 if n_valid > 0 else 9999.0},
                }
                is_orb, _ = entity_filter.is_orbiter_from_stats(entity_stats)
                if is_orb:
                    filter_log.append({"name": name, "type": typ, "reason": "orbiter"})
                    continue

            Xo, Xa = make_windows(obs, act, seq_len=seq_len, stride=stride)
            if Xo is None:
                continue

            obs_list.append(Xo)
            act_list.append(Xa)
            entity_metas.append(dict(
                file=str(path), entity=name, type=typ, coalition=coal,
                n_steps=int(len(obs)), n_windows=int(Xo.shape[0]),
            ))

        return obs_list, act_list, entity_metas, filter_log, None

    except Exception as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else repr(exc)
        return [], [], [], [], f"{first_line}\n{traceback.format_exc()}"


# ── Distribution statistics ───────────────────────────────────────────────────
def _pct(a: np.ndarray) -> dict:
    if len(a) == 0:
        return {}
    return dict(
        p5=float(np.percentile(a, 5)),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        min=float(a.min()), max=float(a.max()),
        mean=float(a.mean()), std=float(a.std()),
    )


def _entropy(a: np.ndarray, bins: int = 20) -> float:
    if len(a) == 0:
        return 0.0
    counts, _ = np.histogram(a, bins=bins)
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def compute_dataset_stats(obs_arr: np.ndarray, act_arr: np.ndarray) -> dict:
    """Compute distribution snapshot for the training set."""
    flat_obs = obs_arr.reshape(-1, obs_arr.shape[-1])
    flat_act = act_arr.reshape(-1, act_arr.shape[-1])
    alt   = flat_obs[:, 2]   # z_u_m
    spd   = flat_obs[:, 7]   # ground_speed_mps
    adpsi = np.abs(flat_act[:, 0])  # |dpsi_rad|
    return dict(
        alt=_pct(alt),
        speed=_pct(spd),
        abs_dpsi=_pct(adpsi),
        entropy=dict(
            alt=_entropy(alt),
            speed=_entropy(spd),
            abs_dpsi=_entropy(adpsi),
        ),
        n_windows=int(obs_arr.shape[0]),
        n_samples=int(len(alt)),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    default_tra  = _THIS_DIR.parent / "tra_data"
    default_out  = _THIS_DIR.parent / "datasets" / f"{CFG.DEFAULT_DATASET_STEM}.npz"

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in",  dest="inputs", nargs="*",
                    help="Input .acmi files (default: tra_data/**/*.acmi)")
    ap.add_argument("--out", default=str(default_out),
                    help=f"Output .npz path (default: {default_out})")
    # Hyperparams — default from CFG, overridable at CLI
    ap.add_argument("--dt",          type=float, default=CFG.DT)
    ap.add_argument("--H_sec",       type=float, default=CFG.H_SEC)
    ap.add_argument("--seq_len",     type=int,   default=CFG.SEQ_LEN)
    ap.add_argument("--stride",      type=int,   default=CFG.STRIDE)
    ap.add_argument("--min_speed",   type=float, default=CFG.MIN_SPEED)
    ap.add_argument("--dpsi_sign",   type=float, default=CFG.DPSI_SIGN,
                    choices=[-1.0, 1.0])
    ap.add_argument("--type_filter", type=str,   default=CFG.TYPE_FILTER)
    # Entity filter — default from CFG
    ap.add_argument("--include_regex",    type=str,   default=CFG.INCLUDE_REGEX)
    ap.add_argument("--exclude_regex",    type=str,   default=None,
                    help="Override exclude regex (default: CFG / entity_filter default)")
    ap.add_argument("--exclude_orbiters", type=int,   default=int(CFG.EXCLUDE_ORBITERS),
                    choices=[0, 1])
    ap.add_argument("--orb_alt_std_m",    type=float, default=CFG.ORB_ALT_STD_M)
    ap.add_argument("--orb_spd_std_mps",  type=float, default=CFG.ORB_SPD_STD_MPS)
    ap.add_argument("--orb_dpsi_p95_rad", type=float, default=CFG.ORB_DPSI_P95_RAD)
    ap.add_argument("--orb_min_steps",    type=int,   default=CFG.ORB_MIN_STEPS)
    args = ap.parse_args()

    dt    = args.dt
    H_sec = args.H_sec
    k     = int(round(H_sec / dt))
    if k < 1:
        sys.exit("H_sec too small relative to dt")

    # exclude_regex: CLI None → fall through to EntityFilter which uses DEFAULT_EXCLUDE_REGEX
    exclude_regex = args.exclude_regex  # may be None (EntityFilter handles it)

    # ── Output paths (all derived from --out stem) ────────────────────────────
    out_npz      = Path(args.out)
    stem         = str(out_npz.with_suffix(""))
    out_meta     = Path(stem + ".meta.json")
    out_filelist = Path(stem + ".filelist.txt")
    out_rejected = Path(stem + ".rejected.txt")
    out_stats    = Path(stem + ".stats.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # ── Input files ───────────────────────────────────────────────────────────
    if args.inputs:
        acmi_files = [Path(p) for p in args.inputs if Path(p).is_file()]
    else:
        acmi_files = sorted(default_tra.rglob("*.acmi"))
    acmi_files = [p for p in acmi_files if p.is_file()]

    if not acmi_files:
        sys.exit(f"No .acmi files found. Searched: {default_tra}")

    # ── Entity filter ─────────────────────────────────────────────────────────
    ent_filter = EntityFilter(
        include_regex=args.include_regex,
        exclude_regex=exclude_regex,
        exclude_orbiters=bool(args.exclude_orbiters),
        orb_alt_std_m=args.orb_alt_std_m,
        orb_spd_std_mps=args.orb_spd_std_mps,
        orb_dpsi_p95_rad=args.orb_dpsi_p95_rad,
        orb_min_steps=args.orb_min_steps,
    )

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  build_training_dataset.py")
    print(f"  Config  : dataset_default_config.py  (dt={dt}s, H={H_sec}s, k={k})")
    print(f"  Filter  : exclude_orbiters={bool(args.exclude_orbiters)}"
          f"  regex={'yes' if ent_filter.exclude_regex else 'none'}")
    print(f"  Input   : {len(acmi_files)} ACMI files")
    print(f"  Output  : {out_npz}")
    print("=" * 65)
    print()

    # ── Process each file ─────────────────────────────────────────────────────
    all_obs:        List[np.ndarray] = []
    all_act:        List[np.ndarray] = []
    all_ent_metas:  List[dict]       = []
    all_filter_log: List[dict]       = []
    used_files:     List[str]        = []
    rejected_lines: List[str]        = []

    for i, p in enumerate(acmi_files, 1):
        label = p.name[:52]
        print(f"[{i:3d}/{len(acmi_files)}] {label:<52s} ...", end=" ", flush=True)

        obs_list, act_list, ent_metas, flog, err = process_file(
            p, k=k, seq_len=args.seq_len, stride=args.stride,
            min_speed=args.min_speed, dpsi_sign=args.dpsi_sign,
            entity_filter=ent_filter, type_filter=args.type_filter,
        )
        all_filter_log.extend(flog)

        if err:
            short = err.splitlines()[0][:60]
            print(f"FAIL  {short}")
            rejected_lines.append(f"PARSE_ERROR\t{p}\t{short}")
            continue

        if not obs_list:
            n_ex = len(flog)
            print(f"SKIP  (0 windows, {n_ex} excluded)")
            reasons = {}
            for e in flog:
                r = e.get("reason", "?")
                reasons[r] = reasons.get(r, 0) + 1
            rejected_lines.append(f"NO_WINDOWS\t{p}\t{n_ex} excluded {reasons}")
            continue

        nw = sum(x.shape[0] for x in obs_list)
        ne = len(obs_list)
        n_ex = len(flog)
        suffix = f", {n_ex} excl" if n_ex else ""
        print(f"OK    ({nw:5d} win, {ne} entities{suffix})")

        all_obs.extend(obs_list)
        all_act.extend(act_list)
        all_ent_metas.extend(ent_metas)
        used_files.append(str(p))

    print()
    if not all_obs:
        sys.exit("No data extracted. Check --type_filter / --min_speed / filter settings.")

    # ── Concatenate ───────────────────────────────────────────────────────────
    print("Concatenating and normalising ...")
    obs_arr = np.concatenate(all_obs, axis=0).astype(np.float32)
    act_arr = np.concatenate(all_act, axis=0).astype(np.float32)

    flat_obs = obs_arr.reshape(-1, obs_arr.shape[-1])
    flat_act = act_arr.reshape(-1, act_arr.shape[-1])
    obs_mean = flat_obs.mean(axis=0).astype(np.float32)
    obs_std  = (flat_obs.std(axis=0) + 1e-6).astype(np.float32)
    act_mean = flat_act.mean(axis=0).astype(np.float32)
    act_std  = (flat_act.std(axis=0) + 1e-6).astype(np.float32)

    n_windows       = int(obs_arr.shape[0])
    n_samples       = int(flat_obs.shape[0])
    n_entities_used = len(all_ent_metas)
    n_entities_excl = len(all_filter_log)

    excl_counts: Dict[str, int] = {}
    for e in all_filter_log:
        r = e.get("reason", "unknown")
        excl_counts[r] = excl_counts.get(r, 0) + 1

    # ── Distribution stats ────────────────────────────────────────────────────
    print("Computing distribution stats ...")
    dist = compute_dataset_stats(obs_arr, act_arr)

    # ── Build meta dict ───────────────────────────────────────────────────────
    meta = dict(
        schema="obs->action (smooth setpoint)",
        created=datetime.now(timezone.utc).isoformat(),
        config_source="dataset_default_config.py",
        hyperparams=dict(
            dt=dt, H_sec=H_sec, k=k,
            seq_len=args.seq_len, stride=args.stride,
            min_speed=args.min_speed, dpsi_sign=args.dpsi_sign,
            type_filter=args.type_filter,
        ),
        filter_config=ent_filter.to_dict(),
        heading_convention=dict(
            frame=CFG.HEADING_FRAME,
            zero_reference=CFG.HEADING_ZERO_REF,
            positive_direction=CFG.HEADING_POSITIVE_DIR,
            atan2_order=CFG.HEADING_ATAN2_ORDER,
        ),
        action_convention=dict(
            dpsi_rad="CW positive; to send to AFSIM: math.degrees(dpsi)",
            alt_sp_m="metres AGL setpoint",
            spd_sp_mps="m/s ground speed setpoint",
        ),
        obs_fields=CFG.OBS_FIELDS,
        act_fields=CFG.ACT_FIELDS,
        normalisation=dict(
            obs_mean=obs_mean.tolist(),
            obs_std=obs_std.tolist(),
            act_mean=act_mean.tolist(),
            act_std=act_std.tolist(),
        ),
        stats=dict(
            n_files_total=len(acmi_files),
            n_files_used=len(used_files),
            n_files_rejected=len(acmi_files) - len(used_files),
            n_entities_used=n_entities_used,
            n_entities_excluded=n_entities_excl,
            excluded_by_reason=excl_counts,
            n_windows=n_windows,
            n_samples=n_samples,
        ),
        distribution=dist,
        files=all_ent_metas,
    )

    # ── Save NPZ ──────────────────────────────────────────────────────────────
    print(f"Saving {out_npz} ...")
    np.savez_compressed(
        out_npz,
        obs=obs_arr,
        action=act_arr,
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
        meta=json.dumps(meta, cls=_NpEncoder).encode("utf-8"),
    )

    # ── Save meta.json ────────────────────────────────────────────────────────
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, cls=_NpEncoder)

    # ── Save filelist.txt ─────────────────────────────────────────────────────
    with open(out_filelist, "w", encoding="utf-8") as f:
        f.write(f"# {len(used_files)} files contributed to {out_npz.name}\n")
        for fp in sorted(used_files):
            f.write(fp + "\n")

    # ── Save rejected.txt ─────────────────────────────────────────────────────
    with open(out_rejected, "w", encoding="utf-8") as f:
        f.write(f"# {len(rejected_lines)} rejected files\n")
        f.write("# FORMAT: REASON<TAB>FILE<TAB>DETAIL\n")
        for line in rejected_lines:
            f.write(line + "\n")

    # ── Save stats.json ───────────────────────────────────────────────────────
    stats_payload = dict(
        source=str(out_npz),
        generated=datetime.now(timezone.utc).isoformat(),
        n_windows=n_windows,
        n_samples=n_samples,
        alt=dist["alt"],
        speed=dist["speed"],
        abs_dpsi=dist["abs_dpsi"],
        entropy=dist["entropy"],
    )
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2, ensure_ascii=False, cls=_NpEncoder)

    # ── Top-10 files by window contribution ───────────────────────────────────
    file_wins: Dict[str, int] = {}
    for em in all_ent_metas:
        file_wins[em["file"]] = file_wins.get(em["file"], 0) + em["n_windows"]
    top10 = sorted(file_wins.items(), key=lambda x: -x[1])[:10]

    # ── Final summary ──────────────────────────────────────────────────────────
    d = dist
    print()
    print("=" * 65)
    print(f"  {out_npz.name}")
    print(f"  obs    {obs_arr.shape}  (N, T={args.seq_len}, D=8)")
    print(f"  action {act_arr.shape}  (N, T={args.seq_len}, A=3)")
    print()
    print(f"  Files   total / used / rejected : "
          f"{len(acmi_files)} / {len(used_files)} / {len(acmi_files)-len(used_files)}")
    print(f"  Entities  used / excluded       : "
          f"{n_entities_used} / {n_entities_excl}  {excl_counts}")
    print(f"  Windows / samples               : {n_windows:,} / {n_samples:,}")
    print()
    print(f"  Altitude  (m)   "
          f"p50={d['alt']['p50']:.0f}  p95={d['alt']['p95']:.0f}  p99={d['alt']['p99']:.0f}"
          f"  entropy={d['entropy']['alt']:.4f} nats")
    print(f"  Speed (m/s)     "
          f"p50={d['speed']['p50']:.1f}  p95={d['speed']['p95']:.1f}  p99={d['speed']['p99']:.1f}"
          f"  entropy={d['entropy']['speed']:.4f} nats")
    adp = d["abs_dpsi"]
    print(f"  |Δψ| (rad)      "
          f"p50={adp['p50']:.5f}  p95={adp['p95']:.4f}  p99={adp['p99']:.4f}"
          f"  entropy={d['entropy']['abs_dpsi']:.4f} nats")
    print()
    print("  Top-10 files by window count:")
    for fp, nw in top10:
        print(f"    {nw:6d}  {Path(fp).name}")
    print()
    print("  Outputs:")
    for f_out in [out_npz, out_meta, out_filelist, out_rejected, out_stats]:
        print(f"    {f_out}")
    print("=" * 65)


if __name__ == "__main__":
    main()
