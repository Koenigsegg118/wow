#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_acmi_coverage.py — ACMI data coverage & quality analyser
=================================================================
Scans wow/tra_data/**/*.acmi by default (or explicit --in list).

Outputs under <out>/ (default: wow/reports/coverage/):
  stats.json    machine-readable metrics
  report.md     human-readable summary
  plots/        PNG figures

Usage
-----
  # default: scan all tra_data/*.acmi
  python wow/analyze_acmi_coverage.py

  # explicit files + custom output
  python wow/analyze_acmi_coverage.py --in wow/tra_data/*.acmi --out wow/reports/coverage

  # with baseline/expanded comparison
  python wow/analyze_acmi_coverage.py \\
      --baseline_list wow/baseline.txt --expanded_list wow/expanded.txt
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Reuse existing parsing utilities ─────────────────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent
for _p in [str(_ROOT_DIR), str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from acmi_to_dt_dataset_smooth import (   # noqa: E402
    wrap_pi,
    unwrap_angle,
    compute_dpsi_series,
    latlon_to_enu_m,
    read_acmi_text,
    parse_header,
    parse_object_updates,
    build_entity_tracks,
    resample_entity,
    make_windows,
)
from shared.conventions.heading_convention import CONVENTION  # noqa: E402
from entity_filter import EntityFilter, DEFAULT_EXCLUDE_REGEX  # noqa: E402

# ── Constants (aligned with existing dataset pipeline) ────────────────────────
DT           = 0.5          # seconds (2 Hz decision rate)
H_SEC        = 2.0          # setpoint horizon (seconds)
K            = int(round(H_SEC / DT))   # = 4 horizon steps
SEQ_LEN      = 20           # sequence length for window counting
STRIDE       = 5            # window stride
MIN_SPEED    = 60.0         # m/s  — low-speed samples discarded
TYPE_FILTER  = "Air+FixedWing"
ACCEL_THRESH = 150.0        # m/s² — flag acceleration outlier above this
SPEED_JUMP   = 200.0        # m/s  — flag sudden speed jump between steps
MAX_SPEED    = 800.0        # m/s  — hard ceiling (≈Mach 2.4); above = corrupted track


# ── JSON encoder that handles numpy scalars / arrays ─────────────────────────
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Statistics helpers ────────────────────────────────────────────────────────
def _pct_dict(a: np.ndarray) -> dict:
    """Compute a standard percentile summary dict for array a."""
    if len(a) == 0:
        return {}
    return dict(
        p5=float(np.percentile(a, 5)),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        p99=float(np.percentile(a, 99)),
        min=float(a.min()),
        max=float(a.max()),
        mean=float(a.mean()),
        std=float(a.std()),
    )


def histogram_entropy(data: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of histogram (nats). Higher = more uniform coverage."""
    if len(data) == 0:
        return 0.0
    counts, _ = np.histogram(data, bins=bins)
    probs = counts / max(counts.sum(), 1)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def quality_grade(n_valid: int, outlier_frac: float) -> str:
    """A/B/C/F quality grade for one entity track."""
    if n_valid >= 500 and outlier_frac < 0.05:
        return "A"
    if n_valid >= 100 and outlier_frac < 0.15:
        return "B"
    if n_valid >= SEQ_LEN:
        return "C"
    return "F"


# ── Per-entity analysis ───────────────────────────────────────────────────────
def analyse_entity(rs: dict) -> Optional[dict]:
    """
    Compute coverage statistics for one resampled entity.

    Returns None if track is too short even before speed filtering.
    Returns a dict with n_windows=0 and skipped_reason if not enough valid
    samples remain after filtering.
    """
    n = len(rs["t"])
    if n <= K + 2:
        return None

    spd       = rs["spd"]
    heading_u = rs["heading_u"]
    z         = rs["z"]
    vx, vy, vz = rs["vx"], rs["vy"], rs["vz"]

    # ── Outlier detection: large acceleration or speed jump ──────────────────
    ax = np.gradient(vx, DT)
    ay = np.gradient(vy, DT)
    az = np.gradient(vz, DT)
    accel_mag   = np.sqrt(ax**2 + ay**2 + az**2)
    dspd        = np.abs(np.diff(spd, prepend=spd[0]))
    outlier_mask = (accel_mag > ACCEL_THRESH) | (dspd > SPEED_JUMP)
    n_outlier   = int(outlier_mask.sum())
    outlier_frac = n_outlier / max(n, 1)

    # ── Compute dpsi over horizon k ──────────────────────────────────────────
    n_valid_for_dpsi = n - K
    if n_valid_for_dpsi < 2:
        return None

    dpsi     = compute_dpsi_series(heading_u, K, dpsi_sign=CONVENTION.dpsi_sign)
    obs_spd  = spd[:n - K]          # aligned with dpsi
    obs_alt  = z[:n - K]
    obs_dpsi = dpsi                  # same length as obs_spd

    # Align outlier_mask with the truncated obs arrays (drop last K elements).
    outlier_obs = outlier_mask[:n - K]

    # ── Min-speed + outlier + hard ceiling filter ─────────────────────────────
    n_before = len(obs_spd)
    speed_ok = (obs_spd >= MIN_SPEED) & (obs_spd <= MAX_SPEED)
    valid    = speed_ok & (~outlier_obs)
    n_after  = int(valid.sum())
    low_speed_frac = 1.0 - int(speed_ok.sum()) / max(n_before, 1)

    base = dict(
        id=rs["id"], name=rs["name"],
        type=rs["type"], coalition=rs["coalition"],
        n_raw=n,
        n_before_filter=n_before,
        n_after_filter=n_after,
        low_speed_frac=float(low_speed_frac),
        outlier_frac=float(outlier_frac),
    )

    if n_after < SEQ_LEN:
        base.update(n_windows=0, quality_grade="F",
                    skipped_reason="too_few_after_speed_filter")
        return base

    alt_f   = obs_alt[valid]
    spd_f   = obs_spd[valid]
    dpsi_f  = obs_dpsi[valid]
    adpsi_f = np.abs(dpsi_f)

    n_windows = max(0, (n_after - SEQ_LEN) // STRIDE + 1)
    grade     = quality_grade(n_after, outlier_frac)

    base.update(
        n_windows=n_windows,
        quality_grade=grade,
        alt=_pct_dict(alt_f),
        speed=_pct_dict(spd_f),
        abs_dpsi=_pct_dict(adpsi_f),
        turn_rate_proxy=_pct_dict(adpsi_f / H_SEC),   # |Δψ|/H  rad/s
        # Raw arrays stored as Python lists for per-file aggregation.
        # Keys prefixed with _ are stripped before JSON output.
        _alt=alt_f.tolist(),
        _spd=spd_f.tolist(),
        _dpsi=dpsi_f.tolist(),
        _abs_dpsi=adpsi_f.tolist(),
    )
    return base


# ── Per-file analysis ─────────────────────────────────────────────────────────
def analyse_file(path: Path, entity_filter: Optional["EntityFilter"] = None) -> dict:
    """
    Parse one ACMI file and return per-entity and aggregate stats.
    Never raises: errors are caught and returned as ok=False dicts.

    When *entity_filter* is supplied the result contains both before-filter
    (_alt_before / _spd_before / _abs_dpsi_before) and after-filter
    (_alt / _spd / _abs_dpsi) raw arrays, as well as filter_log.
    """
    try:
        txt            = read_acmi_text(str(path))
        lines          = txt.splitlines()
        header, start  = parse_header(lines)
        ref_lat = float(header.get("ReferenceLatitude",  "0") or "0")
        ref_lon = float(header.get("ReferenceLongitude", "0") or "0")

        objects, tracks = parse_object_updates(lines, start)
        entities        = build_entity_tracks(objects, tracks, ref_lat, ref_lon)

        # ── Original-track sampling interval check ───────────────────────────
        sampling_gaps = []
        for oid, ent in entities.items():
            t_arr = ent.get("t")
            if t_arr is not None and len(t_arr) > 1:
                gaps = np.diff(t_arr)
                sampling_gaps.extend(gaps.tolist())
        sampling_gap_stats = {}
        if sampling_gaps:
            sg = np.array(sampling_gaps)
            sampling_gap_stats = dict(
                median=float(np.median(sg)),
                p95=float(np.percentile(sg, 95)),
                max=float(sg.max()),
                n_large_gaps=int((sg > 2.0).sum()),   # gaps > 2 s
            )

        use_filter = entity_filter is not None and entity_filter.is_active()

        # Per-entity accumulator lists split into before/after filter
        ent_before: List[dict] = []
        ent_after:  List[dict] = []
        arr_before = {"alt": [], "spd": [], "dpsi": [], "adpsi": []}
        arr_after  = {"alt": [], "spd": [], "dpsi": [], "adpsi": []}
        filter_log: List[dict] = []

        for oid, ent in entities.items():
            if TYPE_FILTER and TYPE_FILTER not in ent.get("type", ""):
                continue
            rs = resample_entity(ent, dt=DT)
            if rs is None:
                continue
            res = analyse_entity(rs)
            if res is None:
                continue

            # Extract raw arrays immediately (before they might be consumed).
            e_alt   = np.array(res.pop("_alt",      []), dtype=np.float32)
            e_spd   = np.array(res.pop("_spd",      []), dtype=np.float32)
            e_dpsi  = np.array(res.pop("_dpsi",     []), dtype=np.float32)
            e_adpsi = np.array(res.pop("_abs_dpsi", []), dtype=np.float32)

            # Always add to "before" group.
            ent_before.append(res)
            arr_before["alt"].append(e_alt)
            arr_before["spd"].append(e_spd)
            arr_before["dpsi"].append(e_dpsi)
            arr_before["adpsi"].append(e_adpsi)

            # Apply entity filter to decide "after" group membership.
            if use_filter:
                keep, reason = entity_filter.check_name_type(res["name"], res["type"])
                if not keep:
                    filter_log.append({"name": res["name"], "type": res["type"],
                                       "reason": reason})
                    continue
                is_orb, orb_reason = entity_filter.is_orbiter_from_stats(res)
                if is_orb:
                    filter_log.append({"name": res["name"], "type": res["type"],
                                       "reason": orb_reason})
                    continue

            ent_after.append(res)
            arr_after["alt"].append(e_alt)
            arr_after["spd"].append(e_spd)
            arr_after["dpsi"].append(e_dpsi)
            arr_after["adpsi"].append(e_adpsi)

        def _cat(lst):
            return np.concatenate(lst) if lst else np.array([], dtype=np.float32)

        # "After" is the primary result; "before" is stored for comparison.
        all_alt   = _cat(arr_after["alt"])
        all_spd   = _cat(arr_after["spd"])
        all_dpsi  = _cat(arr_after["dpsi"])
        all_adpsi = _cat(arr_after["adpsi"])

        all_alt_b   = _cat(arr_before["alt"])
        all_spd_b   = _cat(arr_before["spd"])
        all_dpsi_b  = _cat(arr_before["dpsi"])
        all_adpsi_b = _cat(arr_before["adpsi"])

        n_entities        = len(ent_after)
        n_windows         = sum(e["n_windows"] for e in ent_after)
        n_entities_before = len(ent_before)
        n_windows_before  = sum(e["n_windows"] for e in ent_before)

        if n_entities_before == 0:
            return dict(file=str(path), ok=True, n_entities=0, n_windows=0,
                        n_entities_before=0, n_windows_before=0,
                        entities=[], sampling_gap_stats=sampling_gap_stats,
                        filter_log=filter_log,
                        note="no_aircraft_found")

        grades        = [e["quality_grade"] for e in ent_after]
        grades_before = [e["quality_grade"] for e in ent_before]

        return dict(
            file=str(path), ok=True,
            ref_lat=ref_lat, ref_lon=ref_lon,
            # After-filter (primary)
            n_entities=n_entities,
            n_windows=n_windows,
            quality_grades=grades,
            quality_summary=_grade_summary(grades),
            alt=_pct_dict(all_alt)   if len(all_alt)   else {},
            speed=_pct_dict(all_spd) if len(all_spd)   else {},
            abs_dpsi=_pct_dict(all_adpsi) if len(all_adpsi) else {},
            # Before-filter (for comparison)
            n_entities_before=n_entities_before,
            n_windows_before=n_windows_before,
            quality_grades_before=grades_before,
            quality_summary_before=_grade_summary(grades_before),
            # Filter provenance
            filter_log=filter_log,
            sampling_gap_stats=sampling_gap_stats,
            entities=ent_after,
            # Raw arrays for global aggregation — stripped before JSON.
            _alt=all_alt,
            _spd=all_spd,
            _dpsi=all_dpsi,
            _abs_dpsi=all_adpsi,
            _alt_before=all_alt_b,
            _spd_before=all_spd_b,
            _dpsi_before=all_dpsi_b,
            _abs_dpsi_before=all_adpsi_b,
        )

    except Exception as exc:
        return dict(
            file=str(path),
            ok=False,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def _grade_summary(grades: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {g: 0 for g in "ABCF"}
    for g in grades:
        counts[g] = counts.get(g, 0) + 1
    return counts


def _filter_summary(filter_log: List[dict]) -> Dict[str, int]:
    """Aggregate filter_log entries by reason."""
    counts: Dict[str, int] = {}
    for entry in filter_log:
        reason = entry.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _make_before_proxy(file_results: List[dict]) -> List[dict]:
    """Return proxy dicts where _alt/_spd/_abs_dpsi are the pre-filter arrays.

    Allows calling aggregate_global() unchanged on the before-filter data.
    """
    proxies = []
    for r in file_results:
        if not r.get("ok"):
            proxies.append(r)
            continue
        p = dict(r)
        p["_alt"]       = r.get("_alt_before",      r.get("_alt",      np.array([])))
        p["_spd"]       = r.get("_spd_before",      r.get("_spd",      np.array([])))
        p["_dpsi"]      = r.get("_dpsi_before",     r.get("_dpsi",     np.array([])))
        p["_abs_dpsi"]  = r.get("_abs_dpsi_before", r.get("_abs_dpsi", np.array([])))
        p["n_windows"]  = r.get("n_windows_before", r.get("n_windows",  0))
        p["n_entities"] = r.get("n_entities_before",r.get("n_entities", 0))
        proxies.append(p)
    return proxies


def write_recommended_list(
    file_results: List[dict],
    out_path: Path,
    min_windows: int = 200,
    min_entities: int = 2,
) -> int:
    """Write a text file listing ACMI files that meet a training quality bar.

    Returns the number of files written.
    """
    good = [
        r for r in file_results
        if r.get("ok") and r.get("n_windows", 0) >= min_windows
        and r.get("n_entities", 0) >= min_entities
    ]
    good.sort(key=lambda x: -x.get("n_windows", 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Recommended training files — {len(good)} files\n")
        f.write(f"# Criteria: n_windows>={min_windows}, n_entities>={min_entities}\n")
        for r in good:
            f.write(r["file"] + "\n")
    return len(good)


# ── Global aggregation ────────────────────────────────────────────────────────
def aggregate_global(file_results: List[dict]) -> dict:
    ok = [r for r in file_results if r.get("ok") and r.get("n_windows", 0) > 0]
    if not ok:
        return {}

    all_alt   = np.concatenate([r["_alt"]      for r in ok])
    all_spd   = np.concatenate([r["_spd"]      for r in ok])
    all_dpsi  = np.concatenate([r["_dpsi"]     for r in ok])
    all_adpsi = np.concatenate([r["_abs_dpsi"] for r in ok])

    total_windows  = sum(r["n_windows"]  for r in ok)
    total_entities = sum(r["n_entities"] for r in ok)

    return dict(
        n_files=len(ok),
        n_windows=total_windows,
        n_samples=len(all_alt),
        n_entities=total_entities,
        alt=_pct_dict(all_alt),
        speed=_pct_dict(all_spd),
        abs_dpsi=_pct_dict(all_adpsi),
        turn_rate_proxy=_pct_dict(all_adpsi / H_SEC),
        entropy=dict(
            alt=histogram_entropy(all_alt),
            speed=histogram_entropy(all_spd),
            abs_dpsi=histogram_entropy(all_adpsi),
        ),
        # Raw arrays for plotting — stripped before JSON.
        _alt=all_alt,
        _spd=all_spd,
        _dpsi=all_dpsi,
        _abs_dpsi=all_adpsi,
    )


# ── Resampling weight suggestion ──────────────────────────────────────────────
def compute_resampling_weights(
    alt: np.ndarray,
    spd: np.ndarray,
    abs_dpsi: np.ndarray,
    n_bins: int = 5,
) -> dict:
    """
    Suggest inverse-frequency weights per (alt, speed, |Δψ|) bucket.
    Under-represented buckets get higher weight → balanced sampling.

    Uses percentile-based edges for alt and speed, but semantically
    fixed edges for |Δψ| (which is near-zero-dominated).
    """
    def _pct_edges(arr):
        edges = np.unique(np.percentile(arr, np.linspace(0, 100, n_bins + 1)))
        # If too few unique edges (degenerate distribution), fall back to uniform
        if len(edges) < 3:
            edges = np.linspace(float(arr.min()), float(arr.max()) + 1e-6, n_bins + 1)
        return edges

    alt_edges  = _pct_edges(alt)
    spd_edges  = _pct_edges(spd)
    # |Δψ|: fixed manoeuvre-intensity breakpoints (rad)
    # near-straight / gentle / moderate / sharp / extreme
    dpsi_edges = np.array([0.0, 0.05, 0.15, 0.30, 0.50, max(float(abs_dpsi.max()) + 1e-6, math.pi)])

    alt_idx   = np.digitize(alt,      alt_edges[1:-1])
    spd_idx   = np.digitize(spd,      spd_edges[1:-1])
    dpsi_idx  = np.digitize(abs_dpsi, dpsi_edges[1:-1])

    bucket_counts: Dict[Tuple, int] = {}
    for a, s, d in zip(alt_idx, spd_idx, dpsi_idx):
        key = (int(a), int(s), int(d))
        bucket_counts[key] = bucket_counts.get(key, 0) + 1

    total = len(alt)
    bucket_weights = {
        str(k): round(total / max(v, 1), 2)
        for k, v in bucket_counts.items()
    }

    def bin_labels(edges, unit=""):
        u = f" {unit}" if unit else ""
        return [f"[{edges[i]:.2f}{u}, {edges[i+1]:.2f}{u})" for i in range(len(edges) - 1)]

    return dict(
        alt_bins=bin_labels(alt_edges, "m"),
        spd_bins=bin_labels(spd_edges, "m/s"),
        dpsi_bins=bin_labels(dpsi_edges, "rad"),
        bucket_weights=bucket_weights,
    )


# ── Baseline / expanded comparison ───────────────────────────────────────────
def compare_groups(
    baseline_results: List[dict],
    expanded_results: List[dict],
) -> dict:
    """Compute coverage delta metrics between two groups of file results."""

    def extract(results):
        ok = [r for r in results if r.get("ok") and r.get("n_windows", 0) > 0]
        if not ok:
            return np.array([]), np.array([]), np.array([])
        alt   = np.concatenate([r["_alt"]      for r in ok])
        spd   = np.concatenate([r["_spd"]      for r in ok])
        adpsi = np.concatenate([r["_abs_dpsi"] for r in ok])
        return alt, spd, adpsi

    b_alt, b_spd, b_adpsi = extract(baseline_results)
    e_alt, e_spd, e_adpsi = extract(expanded_results)

    def _delta(b, e, p):
        if len(b) == 0 or len(e) == 0:
            return None
        return float(np.percentile(e, p)) - float(np.percentile(b, p))

    def _max_delta(b, e):
        if len(b) == 0 or len(e) == 0:
            return None
        return float(e.max()) - float(b.max())

    def _entropy_delta(b, e):
        if len(b) == 0 or len(e) == 0:
            return None
        return histogram_entropy(e) - histogram_entropy(b)

    def _axis(b, e):
        return dict(
            p5_delta=_delta(b, e, 5),
            p50_delta=_delta(b, e, 50),
            p95_delta=_delta(b, e, 95),
            max_delta=_max_delta(b, e),
            entropy_delta=_entropy_delta(b, e),
        )

    return dict(
        baseline_windows=sum(r.get("n_windows", 0) for r in baseline_results if r.get("ok")),
        expanded_windows=sum(r.get("n_windows", 0) for r in expanded_results if r.get("ok")),
        alt=_axis(b_alt, e_alt),
        speed=_axis(b_spd, e_spd),
        abs_dpsi=_axis(b_adpsi, e_adpsi),
        # Raw arrays for comparison plots — stripped before JSON.
        _b_alt=b_alt,   _e_alt=e_alt,
        _b_spd=b_spd,   _e_spd=e_spd,
        _b_adpsi=b_adpsi, _e_adpsi=e_adpsi,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────
def _get_plt():
    """Lazy-import matplotlib so the module is importable without it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_hist(
    data: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    bins: int = 40,
    color: str = "steelblue",
) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    p5, p50, p95 = np.percentile(data, [5, 50, 95])
    ax.axvline(p5,  color="orange", ls="--", lw=1.2, label=f"p5={p5:.1f}")
    ax.axvline(p50, color="red",    ls="-",  lw=1.5, label=f"p50={p50:.1f}")
    ax.axvline(p95, color="orange", ls="--", lw=1.2, label=f"p95={p95:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    alpha: float = 0.15,
    s: float = 4,
) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=s, alpha=alpha, color="steelblue", rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_bar_windows(file_results: List[dict], path: Path, max_show: int = 60) -> None:
    plt = _get_plt()
    import matplotlib.patches as mpatches
    ok = [
        (Path(r["file"]).name, r["n_windows"])
        for r in file_results
        if r.get("ok")
    ]
    ok.sort(key=lambda x: -x[1])
    ok = ok[:max_show]
    if not ok:
        return
    names, vals = zip(*ok)
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.38), 5))
    bars = ax.bar(range(len(names)), vals, color="steelblue")
    palette = {"A": "mediumseagreen", "B": "gold", "C": "darkorange", "F": "tomato"}
    for bar, name in zip(bars, names):
        for r in file_results:
            if Path(r["file"]).name == name:
                gs = r.get("quality_grades", [])
                if gs:
                    dominant = max(set(gs), key=gs.count)
                    bar.set_color(palette.get(dominant, "steelblue"))
                break
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Valid windows")
    ax.set_title("Valid windows per ACMI file  (colour = dominant quality grade)")
    handles = [mpatches.Patch(color=c, label=g) for g, c in palette.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_comparison_hist(
    baseline: np.ndarray,
    expanded: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    bins: int = 40,
) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    kw = dict(bins=bins, density=True, edgecolor="white", linewidth=0.4, alpha=0.65)
    if len(baseline):
        ax.hist(baseline, **kw, color="steelblue", label=f"baseline (n={len(baseline):,})")
    if len(expanded):
        ax.hist(expanded, **kw, color="tomato",    label=f"expanded (n={len(expanded):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_before_after_hist(
    before: np.ndarray,
    after: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    bins: int = 40,
) -> None:
    """Side-by-side density histogram: before filter vs after filter."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    kw = dict(bins=bins, density=True, edgecolor="white", linewidth=0.4, alpha=0.65)
    if len(before):
        ax.hist(before, **kw, color="steelblue", label=f"before filter (n={len(before):,})")
    if len(after):
        ax.hist(after,  **kw, color="tomato",    label=f"after filter  (n={len(after):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_per_file_scatter(file_results: List[dict], plots_dir: Path) -> None:
    """Scatter: each file as a point — median altitude vs median speed."""
    rows = [
        (Path(r["file"]).name, r["alt"].get("p50", 0), r["speed"].get("p50", 0),
         r["abs_dpsi"].get("p95", 0), r["n_windows"])
        for r in file_results
        if r.get("ok") and r.get("n_windows", 0) > 0
    ]
    if not rows:
        return
    plt = _get_plt()
    names, alts, spds, dps, nw = zip(*rows)
    nw_arr = np.array(nw, dtype=float)
    sizes  = 20 + 200 * (nw_arr / max(nw_arr.max(), 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(spds, alts, s=sizes, c=dps, cmap="plasma", alpha=0.8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("|Δψ| p95 (rad)")
    ax.set_xlabel("Median ground speed (m/s)")
    ax.set_ylabel("Median altitude (m)")
    ax.set_title("Per-file: median alt vs speed  (size=windows, colour=|Δψ|p95)")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_file_scatter.png", dpi=120)
    plt.close(fig)


# ── Markdown report ───────────────────────────────────────────────────────────
def _fmt_pct(d: dict, unit: str = "") -> str:
    if not d:
        return "—"
    u = f" {unit}" if unit else ""
    return (f"p5={d.get('p5',0):.1f}{u} | p50={d.get('p50',0):.1f}{u} | "
            f"p95={d.get('p95',0):.1f}{u} | p99={d.get('p99',0):.1f}{u} | "
            f"max={d.get('max',0):.1f}{u}")


def _add_skew_warnings(gs: dict, lines: List[str]) -> None:
    alt  = gs.get("alt",      {})
    spd  = gs.get("speed",    {})
    dpsi = gs.get("abs_dpsi", {})
    if not alt:
        return
    if alt.get("p50", 0) > 8000:
        lines.append("- ⚠️  **High altitude bias**: median > 8000 m — "
                     "low-altitude combat scenarios under-represented.")
    if (alt.get("p95", 0) - alt.get("p5", 0)) < 2000:
        lines.append("- ⚠️  **Narrow altitude spread**: p95−p5 < 2000 m — "
                     "dataset lacks vertical diversity.")
    if spd.get("p50", 0) > 250:
        lines.append("- ⚠️  **High cruise speed**: median > 250 m/s — "
                     "low-speed / depleted-energy manoeuvre data may be scarce.")
    if dpsi.get("p95", 1) < 0.3:
        lines.append("- ⚠️  **Low manoeuvre intensity**: |Δψ| p95 < 0.3 rad — "
                     "high-g turns under-represented.")
    if dpsi.get("p50", 1) < 0.05:
        lines.append("- ⚠️  **Cruise-dominated**: |Δψ| p50 < 0.05 rad — "
                     "most samples are near straight-line flight.")


def generate_report(
    file_results: List[dict],
    global_stats: dict,
    comparison: Optional[dict],
    resampling: Optional[dict],
    failed: List[dict],
    global_stats_before: Optional[dict] = None,
    filter_config: Optional[dict] = None,
    total_filter_log: Optional[List[dict]] = None,
) -> str:
    lines: List[str] = []
    A = lines.append

    A("# ACMI Coverage Analysis Report\n")
    n_ok = sum(1 for r in file_results if r.get("ok"))
    n_windows_after  = global_stats.get("n_windows", 0)
    n_windows_before = (global_stats_before or {}).get("n_windows", n_windows_after)
    A(f"| | |")
    A(f"|---|---|")
    A(f"| Files scanned | **{n_ok + len(failed)}** |")
    A(f"| Parsed OK     | **{n_ok}** |")
    A(f"| Failed        | **{len(failed)}** |")
    A(f"| Windows (before filter) | **{n_windows_before:,}** |")
    A(f"| Windows (after filter)  | **{n_windows_after:,}** |")
    A(f"| Total samples (after)   | **{global_stats.get('n_samples', 0):,}** |")
    A("")

    # ── Entity filter summary ─────────────────────────────────────────────────
    if filter_config:
        A("## Entity Filter Configuration\n")
        A("```")
        for k, v in filter_config.items():
            A(f"  {k}: {v}")
        A("```\n")

    if total_filter_log:
        fsum = _filter_summary(total_filter_log)
        total_excluded = sum(fsum.values())
        A(f"**Excluded entities:** {total_excluded} total\n")
        A("| Reason | Count |")
        A("|--------|-------|")
        for reason, cnt in sorted(fsum.items(), key=lambda x: -x[1]):
            A(f"| `{reason}` | {cnt} |")
        A("")

    # ── Global stats ──────────────────────────────────────────────────────────
    A("## Global Statistics\n")
    A("### Altitude (m)")
    A(f"{_fmt_pct(global_stats.get('alt', {}), 'm')}")
    A(f"\nShannon entropy: **{global_stats.get('entropy', {}).get('alt', 0):.4f}** nats\n")

    A("### Ground Speed (m/s)")
    A(f"{_fmt_pct(global_stats.get('speed', {}), 'm/s')}")
    A(f"\nShannon entropy: **{global_stats.get('entropy', {}).get('speed', 0):.4f}** nats\n")

    A("### |Δψ| — heading change over H=2 s (rad)")
    A(f"{_fmt_pct(global_stats.get('abs_dpsi', {}), 'rad')}")
    A(f"\nShannon entropy: **{global_stats.get('entropy', {}).get('abs_dpsi', 0):.4f}** nats\n")

    tr = global_stats.get("turn_rate_proxy", {})
    if tr:
        A("### Turn-rate proxy |Δψ|/H (rad/s)")
        A(f"{_fmt_pct(tr, 'rad/s')}\n")

    # ── Per-file summary table ────────────────────────────────────────────────
    A("## Per-File Summary\n")
    A("| File | Windows | Grades | Alt p50 (m) | Spd p50 (m/s) | |Δψ| p95 (rad) |")
    A("|------|---------|--------|-------------|---------------|---------------|")
    for r in sorted(file_results, key=lambda x: -x.get("n_windows", 0)):
        if not r.get("ok"):
            continue
        fname = Path(r["file"]).name
        nw    = r.get("n_windows", 0)
        gs_   = ",".join(r.get("quality_grades", []))
        a50   = r.get("alt",      {}).get("p50", float("nan"))
        s50   = r.get("speed",    {}).get("p50", float("nan"))
        d95   = r.get("abs_dpsi", {}).get("p95", float("nan"))
        A(f"| {fname} | {nw} | {gs_} | {a50:.0f} | {s50:.1f} | {d95:.4f} |")

    # ── Failed files ──────────────────────────────────────────────────────────
    if failed:
        A("\n## Failed Files\n")
        A("| File | Error |")
        A("|------|-------|")
        for f in failed:
            err = str(f.get("error", "?"))[:80].replace("|", "\\|")
            A(f"| {Path(f['file']).name} | {err} |")

    # ── Baseline / expanded comparison ────────────────────────────────────────
    if comparison:
        A("\n## Baseline vs Expanded Comparison\n")
        A(f"- Baseline windows: **{comparison['baseline_windows']:,}**")
        A(f"- Expanded windows: **{comparison['expanded_windows']:,}**\n")

        A("| Metric | Δp5 | Δp50 | Δp95 | Δmax | ΔEntropy (nats) |")
        A("|--------|-----|------|------|------|-----------------|")

        def _row(label, d):
            def _f(v):
                return f"{v:+.2f}" if v is not None else "—"
            def _fe(v):
                return f"{v:+.4f}" if v is not None else "—"
            return (f"| {label} "
                    f"| {_f(d.get('p5_delta'))} "
                    f"| {_f(d.get('p50_delta'))} "
                    f"| {_f(d.get('p95_delta'))} "
                    f"| {_f(d.get('max_delta'))} "
                    f"| {_fe(d.get('entropy_delta'))} |")

        A(_row("Altitude (m)",   comparison["alt"]))
        A(_row("Speed (m/s)",    comparison["speed"]))
        A(_row("|Δψ| (rad)",     comparison["abs_dpsi"]))

        # Auto verdict
        A("\n### Coverage Verdict\n")
        alt_d  = comparison["alt"].get("p95_delta")   or 0
        spd_d  = comparison["speed"].get("p95_delta") or 0
        dps_d  = comparison["abs_dpsi"].get("p95_delta") or 0
        wins, reasons = 0, []
        if alt_d > 200:
            wins += 1; reasons.append(f"altitude p95 +{alt_d:.0f} m")
        if spd_d > 10:
            wins += 1; reasons.append(f"speed p95 +{spd_d:.1f} m/s")
        if dps_d > 0.05:
            wins += 1; reasons.append(f"|Δψ| p95 +{dps_d:.3f} rad (more manoeuvres)")
        if wins >= 2:
            A(f"✅ **Expanded data EXTENDS coverage** across {wins}/3 axes: {'; '.join(reasons)}.")
        elif wins == 1:
            A(f"⚠️  **Marginal expansion** on 1/3 axis: {'; '.join(reasons)}.  "
              "Verify that added data is not just noise.")
        else:
            A("❌ **Expanded data does NOT significantly extend coverage.**  "
              "Risk of adding noise without diversity — consider curating or weighting.")

    # ── Before vs After filter comparison ────────────────────────────────────
    if global_stats_before and total_filter_log:
        A("\n## Before vs After Filter Comparison\n")
        gs_b = global_stats_before
        gs_a = global_stats
        A("| Metric | Before filter | After filter | Δ |")
        A("|--------|--------------|--------------|---|")

        def _cmp_row(label, key, subkey, unit=""):
            vb = gs_b.get(key, {}).get(subkey, 0)
            va = gs_a.get(key, {}).get(subkey, 0)
            delta = va - vb
            u = f" {unit}" if unit else ""
            return (f"| {label} | {vb:.2f}{u} | {va:.2f}{u} | "
                    f"{'↑' if delta > 0 else '↓'}{abs(delta):.2f}{u} |")

        A(_cmp_row("Alt p50", "alt", "p50", "m"))
        A(_cmp_row("Alt std", "alt", "std", "m"))
        A(_cmp_row("Speed p50", "speed", "p50", "m/s"))
        A(_cmp_row("Speed std", "speed", "std", "m/s"))
        A(_cmp_row("|Δψ| p95", "abs_dpsi", "p95", "rad"))
        A(_cmp_row("|Δψ| p50", "abs_dpsi", "p50", "rad"))

        eb_a = gs_a.get("entropy", {})
        eb_b = gs_b.get("entropy", {})
        for axis in ("alt", "speed", "abs_dpsi"):
            vb = eb_b.get(axis, 0)
            va = eb_a.get(axis, 0)
            delta = va - vb
            A(f"| Entropy ({axis}) | {vb:.4f} nats | {va:.4f} nats | "
              f"{'↑' if delta > 0 else '↓'}{abs(delta):.4f} nats |")
        A("")

        # Auto verdict on filter effect
        A("### Filter Effect Verdict\n")
        dpsi_b = gs_b.get("abs_dpsi", {})
        dpsi_a = gs_a.get("abs_dpsi", {})
        ent_b  = gs_b.get("entropy", {})
        ent_a  = gs_a.get("entropy", {})
        improvements = []
        if dpsi_a.get("p95", 0) > dpsi_b.get("p95", 0) + 0.01:
            improvements.append(f"|Δψ| p95 +{dpsi_a['p95']-dpsi_b['p95']:.3f} rad")
        if dpsi_a.get("p50", 0) > dpsi_b.get("p50", 0) + 0.002:
            improvements.append(f"|Δψ| p50 +{dpsi_a['p50']-dpsi_b['p50']:.4f} rad")
        if ent_a.get("abs_dpsi", 0) > ent_b.get("abs_dpsi", 0) + 0.01:
            improvements.append(f"entropy(|Δψ|) +{ent_a['abs_dpsi']-ent_b['abs_dpsi']:.4f} nats")
        if improvements:
            A(f"✅ **Filter improves manoeuvre diversity**: {'; '.join(improvements)}.")
        else:
            A("ℹ️  Filter removed entities without significantly changing manoeuvre distribution.")

    # ── Distribution skew warnings ────────────────────────────────────────────
    A("\n## Distribution Skew Warnings\n")
    skew_lines: List[str] = []
    _add_skew_warnings(global_stats, skew_lines)
    if skew_lines:
        lines.extend(skew_lines)
    else:
        A("No significant skew detected.")

    # ── Resampling hint ───────────────────────────────────────────────────────
    if resampling:
        A("\n## Suggested Resampling Weights\n")
        A("Inverse-frequency weights per (altitude, speed, |Δψ|) bucket.\n")
        A(f"- Altitude bins (m):   {resampling['alt_bins']}")
        A(f"- Speed bins (m/s):    {resampling['spd_bins']}")
        A(f"- |Δψ| bins (rad):     {resampling['dpsi_bins']}\n")
        A("Top 5 most under-represented buckets (highest weight → most in need of data):\n")
        top5 = sorted(resampling["bucket_weights"].items(), key=lambda x: -x[1])[:5]
        for bk, w in top5:
            A(f"- bucket `{bk}`:  weight = {w:.1f}")

    # ── Plot index ────────────────────────────────────────────────────────────
    A("\n## Output Plots\n")
    A("| File | Description |")
    A("|------|-------------|")
    A("| `plots/alt_hist.png`        | Global altitude distribution (m) |")
    A("| `plots/speed_hist.png`      | Global ground speed distribution (m/s) |")
    A("| `plots/dpsi_hist.png`       | Global \\|Δψ\\| distribution (deg) |")
    A("| `plots/alt_vs_speed.png`    | Altitude vs speed scatter |")
    A("| `plots/speed_vs_dpsi.png`   | Speed vs \\|Δψ\\| scatter |")
    A("| `plots/windows_per_file.png`| Valid windows per file (quality-coloured) |")
    A("| `plots/per_file_scatter.png`| Per-file median alt vs speed |")
    if total_filter_log:
        A("| `plots/filter_alt.png`      | Altitude: before vs after filter |")
        A("| `plots/filter_speed.png`    | Speed: before vs after filter |")
        A("| `plots/filter_dpsi.png`     | \\|Δψ\\|: before vs after filter |")
    if comparison:
        A("| `plots/cmp_alt.png`         | Altitude: baseline vs expanded |")
        A("| `plots/cmp_speed.png`       | Speed: baseline vs expanded |")
        A("| `plots/cmp_dpsi.png`        | \\|Δψ\\|: baseline vs expanded |")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def _load_file_list(list_path: str, repo_root: Path) -> set:
    p = Path(list_path)
    if not p.exists():
        sys.exit(f"List file not found: {p}")
    paths = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fp = Path(line)
        if not fp.is_absolute():
            fp = repo_root / fp
        paths.add(str(fp.resolve()))
    return paths


def _strip_private(d: dict) -> dict:
    """Remove keys starting with '_' or values that are numpy arrays."""
    return {
        k: v for k, v in d.items()
        if not k.startswith("_") and not isinstance(v, np.ndarray)
    }


def main() -> None:
    default_tra = Path(__file__).parent.parent / "tra_data"
    ap = argparse.ArgumentParser(description="ACMI coverage & quality analyser")
    ap.add_argument("--in", dest="inputs", nargs="*",
                    help="Input .acmi files. Default: tra_data/**/*.acmi")
    ap.add_argument("--out", default=str(Path(__file__).parent.parent / "reports" / "coverage"),
                    help="Output directory  (default: wow/reports/coverage)")
    ap.add_argument("--baseline_list", default=None,
                    help="Text file: one baseline .acmi path per line")
    ap.add_argument("--expanded_list", default=None,
                    help="Text file: one expanded .acmi path per line")
    ap.add_argument("--min_speed", type=float, default=MIN_SPEED,
                    help=f"Min ground speed filter (default: {MIN_SPEED} m/s)")
    # Entity filter args (same defaults as acmi_to_dt_dataset_smooth.py)
    ap.add_argument("--include_regex", type=str, default=None,
                    help="Keep only entities whose Name|Type matches this regex")
    ap.add_argument("--exclude_regex", type=str, default=None,
                    help=(f"Exclude entities by Name|Type regex. "
                          f"Default when --exclude_orbiters=1: {DEFAULT_EXCLUDE_REGEX}"))
    ap.add_argument("--exclude_orbiters", type=int, default=1, choices=[0, 1],
                    help="Exclude stable-orbit platforms via behaviour filter (default 1)")
    ap.add_argument("--orb_alt_std_m",    type=float, default=80.0)
    ap.add_argument("--orb_spd_std_mps",  type=float, default=7.0)
    ap.add_argument("--orb_dpsi_p95_rad", type=float, default=0.05)
    ap.add_argument("--orb_min_steps",    type=int,   default=600)
    # Recommended list args
    ap.add_argument("--write_recommended_list", type=int, default=0, choices=[0, 1],
                    help="Write wow/recommended_train_list.txt with high-quality files (default 0)")
    ap.add_argument("--recommended_min_windows",  type=int, default=200)
    ap.add_argument("--recommended_min_fighters", type=int, default=2,
                    help="Min fighter entities per file for recommended list")
    args = ap.parse_args()

    # ── Resolve input files ───────────────────────────────────────────────────
    if args.inputs:
        acmi_files = [Path(p) for p in args.inputs if Path(p).is_file()]
    else:
        acmi_files = sorted(default_tra.rglob("*.acmi"))
    acmi_files = [p for p in acmi_files if p.is_file()]

    if not acmi_files:
        sys.exit(f"No .acmi files found. Searched: {default_tra}")
    print(f"Found {len(acmi_files)} ACMI files.\n")

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_dir   = Path(args.out)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    # ── Build entity filter ───────────────────────────────────────────────────
    ent_filter = EntityFilter(
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        exclude_orbiters=bool(args.exclude_orbiters),
        orb_alt_std_m=args.orb_alt_std_m,
        orb_spd_std_mps=args.orb_spd_std_mps,
        orb_dpsi_p95_rad=args.orb_dpsi_p95_rad,
        orb_min_steps=args.orb_min_steps,
    )
    filter_active = ent_filter.is_active()
    if filter_active:
        print(f"Entity filter: active  (exclude_orbiters={bool(args.exclude_orbiters)}, "
              f"exclude_regex={bool(ent_filter.exclude_regex)})\n")

    # ── Process each file (errors do not abort the batch) ────────────────────
    file_results: List[dict] = []
    failed:       List[dict] = []
    for i, p in enumerate(acmi_files, 1):
        print(f"[{i:3d}/{len(acmi_files)}] {p.name[:60]} ...", end="  ", flush=True)
        result = analyse_file(p, entity_filter=ent_filter if filter_active else None)
        if result.get("ok"):
            file_results.append(result)
            nw   = result["n_windows"]
            ne   = result["n_entities"]
            nwb  = result.get("n_windows_before", nw)
            excl = len(result.get("filter_log", []))
            suffix = f"OK  ({nw}/{nwb} windows, {ne} entities" + (f", {excl} excluded)" if excl else ")")
            print(suffix)
        else:
            failed.append(result)
            print(f"FAIL: {result.get('error','?')[:60]}")

    print(f"\nProcessed: {len(file_results)} OK, {len(failed)} failed.\n")

    # ── Collect global filter log ─────────────────────────────────────────────
    total_filter_log: List[dict] = []
    for r in file_results:
        total_filter_log.extend(r.get("filter_log", []))
    if total_filter_log:
        fsum = _filter_summary(total_filter_log)
        print(f"Total excluded entities: {sum(fsum.values())}  {fsum}")

    # ── Global aggregation (after-filter primary, before-filter for comparison)
    global_stats = aggregate_global(file_results)
    global_stats_before: Optional[dict] = None
    if filter_active and total_filter_log:
        before_proxy = _make_before_proxy(file_results)
        global_stats_before = aggregate_global(before_proxy)

    # ── Baseline / expanded comparison ────────────────────────────────────────
    comparison: Optional[dict] = None
    repo_root = Path(__file__).parent.parent
    if args.baseline_list or args.expanded_list:
        b_paths = _load_file_list(args.baseline_list, repo_root) if args.baseline_list else set()
        e_paths = _load_file_list(args.expanded_list, repo_root) if args.expanded_list else set()
        b_res = [r for r in file_results if str(Path(r["file"]).resolve()) in b_paths]
        e_res = [r for r in file_results if str(Path(r["file"]).resolve()) in e_paths]
        if b_res or e_res:
            comparison = compare_groups(b_res, e_res)
            print(f"Comparison: {len(b_res)} baseline files, {len(e_res)} expanded files.")
        else:
            print("Warning: --baseline_list / --expanded_list produced no matching files.")

    # ── Resampling weights ────────────────────────────────────────────────────
    resampling: Optional[dict] = None
    if global_stats and len(global_stats.get("_alt", [])) > 50:
        resampling = compute_resampling_weights(
            global_stats["_alt"], global_stats["_spd"], global_stats["_abs_dpsi"]
        )

    # ── Generate plots ────────────────────────────────────────────────────────
    if global_stats and len(global_stats.get("_alt", [])) > 0:
        g_alt   = global_stats["_alt"]
        g_spd   = global_stats["_spd"]
        g_adpsi = global_stats["_abs_dpsi"]
        print("Generating plots ...")

        _save_hist(g_alt,
                   "Altitude (m)", "Global Altitude Distribution",
                   plots_dir / "alt_hist.png")

        _save_hist(g_spd,
                   "Ground Speed (m/s)", "Global Ground Speed Distribution",
                   plots_dir / "speed_hist.png", color="coral")

        _save_hist(np.degrees(g_adpsi),
                   "|Δψ| (deg)", "Global |Δψ| Distribution  (heading change over H=2 s)",
                   plots_dir / "dpsi_hist.png", color="mediumseagreen")

        _save_scatter(g_spd, g_alt,
                      "Ground Speed (m/s)", "Altitude (m)",
                      "Altitude vs Ground Speed",
                      plots_dir / "alt_vs_speed.png")

        _save_scatter(g_spd, np.degrees(g_adpsi),
                      "Ground Speed (m/s)", "|Δψ| (deg)",
                      "Ground Speed vs |Δψ|",
                      plots_dir / "speed_vs_dpsi.png")

        _save_bar_windows(file_results, plots_dir / "windows_per_file.png")
        _save_per_file_scatter(file_results, plots_dir)

        # Before/after filter comparison plots
        if global_stats_before and total_filter_log:
            _save_before_after_hist(
                global_stats_before["_alt"], g_alt,
                "Altitude (m)", "Altitude: Before vs After Filter",
                plots_dir / "filter_alt.png")
            _save_before_after_hist(
                global_stats_before["_spd"], g_spd,
                "Ground Speed (m/s)", "Speed: Before vs After Filter",
                plots_dir / "filter_speed.png")
            _save_before_after_hist(
                np.degrees(global_stats_before["_abs_dpsi"]), np.degrees(g_adpsi),
                "|Δψ| (deg)", "|Δψ|: Before vs After Filter",
                plots_dir / "filter_dpsi.png")

        if comparison:
            _save_comparison_hist(comparison["_b_alt"],   comparison["_e_alt"],
                                  "Altitude (m)",
                                  "Altitude: Baseline vs Expanded",
                                  plots_dir / "cmp_alt.png")
            _save_comparison_hist(comparison["_b_spd"],   comparison["_e_spd"],
                                  "Ground Speed (m/s)",
                                  "Speed: Baseline vs Expanded",
                                  plots_dir / "cmp_speed.png")
            _save_comparison_hist(np.degrees(comparison["_b_adpsi"]),
                                  np.degrees(comparison["_e_adpsi"]),
                                  "|Δψ| (deg)",
                                  "|Δψ|: Baseline vs Expanded",
                                  plots_dir / "cmp_dpsi.png")

    # ── Serialise stats.json / stats_before.json / stats_after.json ──────────
    base_params = dict(
        dt=DT, H_sec=H_SEC, k=K,
        seq_len=SEQ_LEN, stride=STRIDE,
        min_speed=MIN_SPEED, type_filter=TYPE_FILTER,
        accel_thresh=ACCEL_THRESH, speed_jump=SPEED_JUMP,
        filter_config=ent_filter.to_dict() if filter_active else None,
    )
    clean_global = _strip_private(global_stats)
    clean_files  = [_strip_private(r) for r in file_results]
    clean_comp   = (
        {k: v for k, v in comparison.items() if not k.startswith("_")}
        if comparison else None
    )

    stats_payload = dict(
        params=base_params,
        global_stats=clean_global,
        file_results=clean_files,
        failed_files=[{"file": f["file"], "error": f.get("error", "")} for f in failed],
        comparison=clean_comp,
        resampling=resampling,
        filter_summary=_filter_summary(total_filter_log) if total_filter_log else {},
    )

    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2, ensure_ascii=False, cls=_NpEncoder)
    print(f"Saved stats  → {stats_path}")

    # When filter is active, also write stats_after.json (same as stats.json)
    # and stats_before.json for comparison.
    if filter_active and global_stats_before:
        stats_after_path = out_dir / "stats_after.json"
        stats_path.replace(stats_after_path)  # rename stats.json → stats_after.json
        # Re-write stats.json as the primary (after-filter) copy
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2, ensure_ascii=False, cls=_NpEncoder)
        print(f"Saved stats_after  → {stats_after_path}")

        clean_global_before = _strip_private(global_stats_before)
        stats_before_payload = dict(
            params=base_params,
            global_stats=clean_global_before,
            file_results=[{**_strip_private(r),
                           "n_windows":  r.get("n_windows_before", r.get("n_windows", 0)),
                           "n_entities": r.get("n_entities_before", r.get("n_entities", 0)),
                           "quality_grades": r.get("quality_grades_before", r.get("quality_grades", [])),
                          } for r in file_results],
            failed_files=[{"file": f["file"], "error": f.get("error", "")} for f in failed],
        )
        stats_before_path = out_dir / "stats_before.json"
        with open(stats_before_path, "w", encoding="utf-8") as f:
            json.dump(stats_before_payload, f, indent=2, ensure_ascii=False, cls=_NpEncoder)
        print(f"Saved stats_before → {stats_before_path}")

    # ── Write Markdown report ─────────────────────────────────────────────────
    report_md = generate_report(
        file_results, global_stats, comparison, resampling, failed,
        global_stats_before=global_stats_before,
        filter_config=ent_filter.to_dict() if filter_active else None,
        total_filter_log=total_filter_log if total_filter_log else None,
    )
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved report → {report_path}")

    # ── Recommended training file list ────────────────────────────────────────
    if args.write_recommended_list:
        rec_path = Path(__file__).parent.parent / "recommended_train_list.txt"
        n_rec = write_recommended_list(
            file_results, rec_path,
            min_windows=args.recommended_min_windows,
            min_entities=args.recommended_min_fighters,
        )
        print(f"Saved recommended list ({n_rec} files) → {rec_path}")

    print(f"\nDone.  Output directory: {out_dir}/")
    if failed:
        print(f"  ⚠  {len(failed)} file(s) failed — see stats.json::failed_files.")


if __name__ == "__main__":
    main()
