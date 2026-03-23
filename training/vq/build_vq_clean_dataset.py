#!/usr/bin/env python
"""Build a VQ-clean action-chunk dataset from raw ACMI files.

Differences from the BC main-line builder:
  - Action semantics: [dpsi_rad, dalt_sp_m, dspd_sp_mps] (relative, not absolute)
  - Outlier steps (accel spike / speed jump) are truly removed, not just flagged
  - Time-gap aware: entities are split at gaps > threshold; no window crosses a gap
  - Per-entity and per-file window caps prevent long sorties from dominating

Usage (from repo root):
  python -m training.vq.build_vq_clean_dataset --out datasets/dt2hz_H2s_vqclean.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

# ── reuse existing ml/ parsing pipeline ──────────────────────────────
# ml/ uses bare imports internally (e.g. "from entity_filter import ..."),
# so we must add the ml/ directory to sys.path before importing.
_ML_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent / "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from acmi_to_dt_dataset_smooth import (  # noqa: E402
    read_acmi_text,
    parse_header,
    parse_object_updates,
    build_entity_tracks,
    resample_entity,
    compute_dpsi_series,
    make_windows,
    wrap_pi,
)
from entity_filter import EntityFilter, DEFAULT_EXCLUDE_REGEX  # noqa: E402
import dataset_default_config as _CFG  # noqa: E402

DT = _CFG.DT
H_SEC = _CFG.H_SEC
SEQ_LEN = _CFG.SEQ_LEN
STRIDE = _CFG.STRIDE
MIN_SPEED = _CFG.MIN_SPEED
MAX_SPEED = _CFG.MAX_SPEED
TYPE_FILTER = _CFG.TYPE_FILTER
ACCEL_THRESH = _CFG.ACCEL_THRESH
SPEED_JUMP = _CFG.SPEED_JUMP
DPSI_SIGN = _CFG.DPSI_SIGN
ORB_ALT_STD_M = _CFG.ORB_ALT_STD_M
ORB_SPD_STD_MPS = _CFG.ORB_SPD_STD_MPS
ORB_DPSI_P95_RAD = _CFG.ORB_DPSI_P95_RAD
ORB_MIN_STEPS = _CFG.ORB_MIN_STEPS

OBS_FIELDS = [
    "x_e_m", "y_n_m", "z_u_m",
    "vx_e_mps", "vy_n_mps", "vz_u_mps",
    "track_angle_rad_unwrapped", "ground_speed_mps",
]
ACT_FIELDS = ["dpsi_rad", "dalt_sp_m", "dspd_sp_mps"]


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got '{v}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build VQ-clean dataset from ACMI files")

    p.add_argument("--in", dest="input_dir", type=str, default="tra_data",
                   help="Input directory containing *.acmi (recursive)")
    p.add_argument("--out", type=str, default="datasets/dt2hz_H2s_vqclean.npz")

    # resampling & windowing
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--H_sec", type=float, default=H_SEC)
    p.add_argument("--seq_len", type=int, default=SEQ_LEN)
    p.add_argument("--stride", type=int, default=STRIDE)

    # speed filter
    p.add_argument("--min_speed", type=float, default=MIN_SPEED)
    p.add_argument("--max_speed", type=float, default=MAX_SPEED)

    # entity filter
    p.add_argument("--type_filter", type=str, default=TYPE_FILTER)
    p.add_argument("--include_regex", type=str, default=None)
    p.add_argument("--exclude_regex", type=str, default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--exclude_orbiters", type=_str2bool, default=True)
    p.add_argument("--orb_alt_std_m", type=float, default=ORB_ALT_STD_M)
    p.add_argument("--orb_spd_std_mps", type=float, default=ORB_SPD_STD_MPS)
    p.add_argument("--orb_dpsi_p95_rad", type=float, default=ORB_DPSI_P95_RAD)
    p.add_argument("--orb_min_steps", type=int, default=ORB_MIN_STEPS)

    # outlier thresholds
    p.add_argument("--accel_thresh", type=float, default=ACCEL_THRESH)
    p.add_argument("--speed_jump", type=float, default=SPEED_JUMP)

    # gap splitting
    p.add_argument("--gap_thresh", type=float, default=2.0,
                   help="Max allowed time gap (s) before splitting entity")

    # window caps
    p.add_argument("--max_windows_per_entity", type=int, default=None)
    p.add_argument("--max_windows_per_file", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════

def split_by_gaps(t: np.ndarray, gap_thresh: float) -> list[tuple[int, int]]:
    """Return list of (start, end) index pairs for contiguous segments."""
    if len(t) < 2:
        return [(0, len(t))]
    dt = np.diff(t)
    gap_idx = np.where(dt > gap_thresh)[0]  # indices where gap occurs
    segments = []
    prev = 0
    for gi in gap_idx:
        segments.append((prev, gi + 1))
        prev = gi + 1
    segments.append((prev, len(t)))
    return [(s, e) for s, e in segments if e - s >= 2]


def build_outlier_mask(rs: dict, accel_thresh: float,
                       speed_jump_thresh: float) -> np.ndarray:
    """Return boolean mask (True = valid) per timestep."""
    n = len(rs["t"])
    valid = np.ones(n, dtype=bool)

    # acceleration magnitude
    ax = np.gradient(rs["vx"], rs["t"])
    ay = np.gradient(rs["vy"], rs["t"])
    az = np.gradient(rs["vz"], rs["t"])
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    valid &= accel_mag <= accel_thresh

    # speed jump between consecutive steps — mark BOTH sides of a jump
    spd_diff = np.abs(np.diff(rs["spd"]))
    bad_jump = spd_diff > speed_jump_thresh  # [n-1]
    jump_mask = np.ones(n, dtype=bool)
    jump_mask[:-1] &= ~bad_jump   # step before the jump
    jump_mask[1:] &= ~bad_jump    # step after the jump
    valid &= jump_mask

    return valid


def build_obs_and_relative_action(
    rs: dict, k: int, dpsi_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build obs [L, 8] and relative action [L, 3] from resampled entity.

    action = [dpsi_rad, dalt_sp_m, dspd_sp_mps] where:
      - dpsi_rad  = wrap_pi(heading[t+k] - heading[t])
      - dalt_sp_m = z[t+k] - z[t]
      - dspd_sp_mps = spd[t+k] - spd[t]
    """
    heading = rs["heading_u"]
    spd = rs["spd"]
    z = rs["z"]

    dpsi = compute_dpsi_series(heading, k, dpsi_sign)  # [L]
    dalt = (z[k:] - z[:-k]).astype(np.float32)         # [L]
    dspd = (spd[k:] - spd[:-k]).astype(np.float32)     # [L]

    L = len(dpsi)
    obs = np.stack([
        rs["x"][:L], rs["y"][:L], rs["z"][:L],
        rs["vx"][:L], rs["vy"][:L], rs["vz"][:L],
        heading[:L], spd[:L],
    ], axis=1).astype(np.float32)

    act = np.stack([dpsi, dalt, dspd], axis=1).astype(np.float32)
    return obs, act


# ═════════════════════════════════════════════════════════════════════
#  Per-file processing
# ═════════════════════════════════════════════════════════════════════

def process_file(
    path: pathlib.Path,
    args: argparse.Namespace,
    entity_filter: EntityFilter,
) -> dict:
    """Process one ACMI file. Returns a result dict."""
    result = {
        "path": str(path),
        "obs_list": [],
        "act_list": [],
        "entity_metas": [],
        "filter_log": [],
        "error": None,
        "gap_splits": 0,
        "outlier_steps_removed": 0,
    }

    k = int(round(args.H_sec / args.dt))

    try:
        txt = read_acmi_text(str(path))
    except Exception as e:
        result["error"] = f"read error: {e}"
        return result

    lines = txt.splitlines()
    try:
        header, start_idx = parse_header(lines)
        ref_lat = float(header.get("ReferenceLatitude", "0") or "0")
        ref_lon = float(header.get("ReferenceLongitude", "0") or "0")
        objects, tracks = parse_object_updates(lines, start_idx)
        entities = build_entity_tracks(objects, tracks, ref_lat, ref_lon)
    except Exception as e:
        result["error"] = f"parse error: {e}"
        return result

    for oid, ent in entities.items():
        etype = ent.get("type", "")
        ename = ent.get("name", "")

        # type filter
        if args.type_filter and args.type_filter not in etype:
            continue

        # name/type regex filter
        keep, reason = entity_filter.check_name_type(ename, etype)
        if not keep:
            result["filter_log"].append({
                "name": ename, "type": etype, "reason": reason})
            continue

        # resample
        rs = resample_entity(ent, dt=args.dt)
        if rs is None:
            continue
        n_steps = len(rs["t"])
        min_steps_needed = k + args.seq_len + 1
        if n_steps < min_steps_needed:
            continue

        # outlier mask (on full resampled entity)
        valid_mask = build_outlier_mask(rs, args.accel_thresh, args.speed_jump)

        # speed filter
        valid_mask &= rs["spd"] >= args.min_speed
        valid_mask &= rs["spd"] <= args.max_speed

        # gap splitting on the original time array
        segments = split_by_gaps(rs["t"], args.gap_thresh)
        if len(segments) > 1:
            result["gap_splits"] += len(segments) - 1

        entity_obs_list = []
        entity_act_list = []

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < min_steps_needed:
                continue

            # slice resampled arrays for this segment
            seg_rs = {
                key: rs[key][seg_start:seg_end]
                for key in ("t", "x", "y", "z", "vx", "vy", "vz",
                            "spd", "heading_u")
            }
            seg_valid = valid_mask[seg_start:seg_end]

            # build obs & relative action
            obs, act = build_obs_and_relative_action(seg_rs, k, DPSI_SIGN)
            # align valid mask with obs/act (which are truncated by k)
            seg_valid_aligned = seg_valid[:len(obs)]

            outlier_count = int((~seg_valid_aligned).sum())
            result["outlier_steps_removed"] += outlier_count

            # apply valid mask: only keep contiguous runs ≥ seq_len
            # find contiguous valid runs
            runs = _contiguous_runs(seg_valid_aligned)
            for run_s, run_e in runs:
                run_obs = obs[run_s:run_e]
                run_act = act[run_s:run_e]
                if len(run_obs) < args.seq_len:
                    continue
                wo, wa = make_windows(run_obs, run_act,
                                      seq_len=args.seq_len,
                                      stride=args.stride)
                if wo is not None:
                    entity_obs_list.append(wo)
                    entity_act_list.append(wa)

        if not entity_obs_list:
            continue

        # orbiter check (post speed-filter stats)
        all_entity_obs = np.concatenate(entity_obs_list, axis=0)
        all_entity_act = np.concatenate(entity_act_list, axis=0)
        flat_act = all_entity_act.reshape(-1, 3)
        flat_obs = all_entity_obs.reshape(-1, 8)

        if entity_filter.is_active() and args.exclude_orbiters:
            entity_stats = {
                "n_after_filter": len(flat_obs),
                "alt": {"std": float(flat_obs[:, 2].std())},
                "speed": {"std": float(flat_obs[:, 7].std())},
                "abs_dpsi": {"p95": float(np.percentile(
                    np.abs(flat_act[:, 0]), 95))},
            }
            is_orb, orb_reason = entity_filter.is_orbiter_from_stats(
                entity_stats)
            if is_orb:
                result["filter_log"].append({
                    "name": ename, "type": etype, "reason": orb_reason})
                continue

        # per-entity cap
        n_windows = all_entity_obs.shape[0]
        if (args.max_windows_per_entity is not None
                and n_windows > args.max_windows_per_entity):
            rng = np.random.RandomState(args.seed + hash(ename) % 2**31)
            sel = rng.choice(n_windows, args.max_windows_per_entity,
                             replace=False)
            sel.sort()
            all_entity_obs = all_entity_obs[sel]
            all_entity_act = all_entity_act[sel]

        result["obs_list"].append(all_entity_obs)
        result["act_list"].append(all_entity_act)
        result["entity_metas"].append({
            "name": ename, "type": etype,
            "n_windows": all_entity_obs.shape[0],
            "file": str(path.name),
        })

    # per-file cap
    if result["obs_list"]:
        file_obs = np.concatenate(result["obs_list"], axis=0)
        file_act = np.concatenate(result["act_list"], axis=0)
        n = file_obs.shape[0]
        if (args.max_windows_per_file is not None
                and n > args.max_windows_per_file):
            rng = np.random.RandomState(args.seed + hash(str(path)) % 2**31)
            sel = rng.choice(n, args.max_windows_per_file, replace=False)
            sel.sort()
            file_obs = file_obs[sel]
            file_act = file_act[sel]
        result["obs_list"] = [file_obs]
        result["act_list"] = [file_act]

    return result


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True-runs in a boolean array. Returns (start, end) pairs."""
    if len(mask) == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return list(zip(starts.tolist(), ends.tolist()))


# ═════════════════════════════════════════════════════════════════════
#  Statistics
# ═════════════════════════════════════════════════════════════════════

def compute_vq_stats(obs: np.ndarray, act: np.ndarray) -> dict:
    """Compute distribution stats for the VQ-clean dataset."""
    def _dim_stats(arr, name):
        return {
            "name": name,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    flat_obs = obs.reshape(-1, 8)
    flat_act = act.reshape(-1, 3)

    stats = {
        "obs": {OBS_FIELDS[i]: _dim_stats(flat_obs[:, i], OBS_FIELDS[i])
                for i in range(8)},
        "action": {ACT_FIELDS[i]: _dim_stats(flat_act[:, i], ACT_FIELDS[i])
                   for i in range(3)},
        "abs_dpsi_rad": _dim_stats(np.abs(flat_act[:, 0]), "|dpsi_rad|"),
        "dalt_sp_m": _dim_stats(flat_act[:, 1], "dalt_sp_m"),
        "dspd_sp_mps": _dim_stats(flat_act[:, 2], "dspd_sp_mps"),
    }
    return stats


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    t0 = time.time()

    input_dir = pathlib.Path(args.input_dir)
    acmi_files = sorted(input_dir.rglob("*.acmi"))
    if not acmi_files:
        print(f"No .acmi files found in {input_dir}")
        return
    print(f"Found {len(acmi_files)} ACMI files in {input_dir}/")

    entity_filter = EntityFilter(
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        exclude_orbiters=args.exclude_orbiters,
        orb_alt_std_m=args.orb_alt_std_m,
        orb_spd_std_mps=args.orb_spd_std_mps,
        orb_dpsi_p95_rad=args.orb_dpsi_p95_rad,
        orb_min_steps=args.orb_min_steps,
    )

    # ── process all files ────────────────────────────────────────
    all_obs = []
    all_act = []
    all_entity_metas = []
    all_filter_log = []
    file_list = []
    rejected_list = []
    total_gap_splits = 0
    total_outlier_steps = 0
    per_file_windows = {}
    per_entity_windows = defaultdict(int)

    for i, fpath in enumerate(acmi_files):
        result = process_file(fpath, args, entity_filter)

        if result["error"]:
            rejected_list.append(f"{fpath.name}\t{result['error']}")
            continue

        total_gap_splits += result["gap_splits"]
        total_outlier_steps += result["outlier_steps_removed"]
        all_filter_log.extend(result["filter_log"])

        if not result["obs_list"]:
            rejected_list.append(f"{fpath.name}\tno valid windows")
            continue

        file_obs = np.concatenate(result["obs_list"], axis=0)
        file_act = np.concatenate(result["act_list"], axis=0)
        n_win = file_obs.shape[0]

        all_obs.append(file_obs)
        all_act.append(file_act)
        file_list.append(str(fpath))
        per_file_windows[fpath.name] = n_win
        for em in result["entity_metas"]:
            all_entity_metas.append(em)
            per_entity_windows[em["name"]] += em["n_windows"]

        if (i + 1) % 10 == 0 or i == len(acmi_files) - 1:
            print(f"  [{i+1}/{len(acmi_files)}] {fpath.name} → "
                  f"{n_win} windows")

    if not all_obs:
        print("No data produced. Check filters and input files.")
        return

    # ── concatenate ──────────────────────────────────────────────
    obs_arr = np.concatenate(all_obs, axis=0)
    act_arr = np.concatenate(all_act, axis=0)

    # stats
    obs_mean = obs_arr.reshape(-1, 8).mean(axis=0).astype(np.float32)
    obs_std = obs_arr.reshape(-1, 8).std(axis=0).astype(np.float32)
    act_mean = act_arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    act_std = act_arr.reshape(-1, 3).std(axis=0).astype(np.float32)

    vq_stats = compute_vq_stats(obs_arr, act_arr)

    # per-entity / per-file distribution
    ent_counts = sorted(per_entity_windows.values(), reverse=True)
    file_counts = sorted(per_file_windows.values(), reverse=True)

    def _count_summary(counts):
        if not counts:
            return {}
        a = np.array(counts)
        return {
            "n": len(a), "total": int(a.sum()),
            "min": int(a.min()), "max": int(a.max()),
            "p50": int(np.percentile(a, 50)),
            "p95": int(np.percentile(a, 95)),
        }

    vq_stats["per_entity_windows"] = _count_summary(ent_counts)
    vq_stats["per_file_windows"] = _count_summary(file_counts)

    # ── save NPZ ─────────────────────────────────────────────────
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")

    meta_dict = {
        "action_semantics": ACT_FIELDS,
        "obs_fields": OBS_FIELDS,
        "args": vars(args),
        "filter_config": entity_filter.to_dict(),
        "n_files_input": len(acmi_files),
        "n_files_used": len(file_list),
        "n_entities_used": len(all_entity_metas),
        "n_entities_filtered": len(all_filter_log),
        "filter_reasons": _count_reasons(all_filter_log),
        "total_windows": int(obs_arr.shape[0]),
        "total_gap_splits": total_gap_splits,
        "total_outlier_steps_removed": total_outlier_steps,
    }

    np.savez_compressed(
        str(out_path),
        obs=obs_arr,
        action=act_arr,
        obs_mean=obs_mean,
        obs_std=obs_std,
        act_mean=act_mean,
        act_std=act_std,
        meta=np.array(json.dumps(meta_dict).encode("utf-8")),
    )

    # companion files
    with open(f"{stem}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    with open(f"{stem}.stats.json", "w", encoding="utf-8") as f:
        json.dump(vq_stats, f, indent=2, ensure_ascii=False)

    with open(f"{stem}.filelist.txt", "w") as f:
        f.write("\n".join(file_list) + "\n")

    with open(f"{stem}.rejected.txt", "w") as f:
        f.write("\n".join(rejected_list) + "\n")

    elapsed = time.time() - t0

    # ── terminal summary ─────────────────────────────────────────
    N = obs_arr.shape[0]
    total_all = sum(per_entity_windows.values())
    top10_ent = sorted(per_entity_windows.items(),
                       key=lambda x: x[1], reverse=True)[:10]
    top10_ent_pct = sum(v for _, v in top10_ent) / total_all * 100

    top10_file = sorted(per_file_windows.items(),
                        key=lambda x: x[1], reverse=True)[:10]
    top10_file_pct = sum(v for _, v in top10_file) / total_all * 100

    print(f"\n{'='*60}")
    print(f"  VQ Clean Dataset Built")
    print(f"{'='*60}")
    print(f"  Input ACMI files     : {len(acmi_files)}")
    print(f"  Used files           : {len(file_list)}")
    print(f"  Used entities        : {len(all_entity_metas)}")
    print(f"  Filtered entities    : {len(all_filter_log)}")
    print(f"  Gap splits           : {total_gap_splits}")
    print(f"  Outlier steps removed: {total_outlier_steps}")
    print(f"  Total windows        : {N:,}")
    print(f"  obs shape            : {obs_arr.shape}")
    print(f"  action shape         : {act_arr.shape}")
    print(f"  action semantics     : {ACT_FIELDS}")
    print()
    print(f"  act_mean = [{act_mean[0]:.6f}, {act_mean[1]:.2f}, {act_mean[2]:.2f}]")
    print(f"  act_std  = [{act_std[0]:.6f}, {act_std[1]:.2f}, {act_std[2]:.2f}]")
    print()
    print(f"  Top-10 entity window share : {top10_ent_pct:.1f}%")
    for name, cnt in top10_ent:
        print(f"    {name:30s} {cnt:6d}  ({cnt/N*100:.1f}%)")
    print()
    print(f"  Top-10 file window share   : {top10_file_pct:.1f}%")
    for name, cnt in top10_file:
        print(f"    {name:30s} {cnt:6d}  ({cnt/N*100:.1f}%)")
    print()
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Saved: {out_path}")
    print(f"         {stem}.meta.json")
    print(f"         {stem}.stats.json")
    print(f"         {stem}.filelist.txt")
    print(f"         {stem}.rejected.txt")


def _count_reasons(filter_log: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in filter_log:
        counts[entry.get("reason", "unknown")] += 1
    return dict(counts)


if __name__ == "__main__":
    main()
