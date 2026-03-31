"""Filter and balance raw candidate events across 5 tactical buckets.

Reads candidate_events.jsonl (produced by mine_candidate_events.py),
applies per-bucket quotas and per-file caps, then writes a balanced
JSONL and a Markdown report.

Usage:
    python -m tools.acmi_mining.filter_and_balance_events \
        --input  datasets/acmi_mining/candidate_events.jsonl \
        --output datasets/acmi_mining/candidate_events_balanced.jsonl \
        --report reports/acmi_candidate_balance_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.acmi_mining.acmi_mining_config import BUCKET_QUOTAS, MAX_EVENTS_PER_FILE

logger = logging.getLogger(__name__)


# ── I/O helpers ──────────────────────────────────────────────────


def _load_events(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file, one JSON object per line."""
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", lineno, exc)
    logger.info("Loaded %d candidate events from %s", len(events), path)
    return events


def _write_events(events: list[dict[str, Any]], path: Path) -> None:
    """Write events as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    logger.info("Wrote %d balanced events to %s", len(events), path)


# ── Core algorithm ───────────────────────────────────────────────


def _group_by_bucket(
    events: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        bucket = ev.get("bucket", "unknown")
        groups[bucket].append(ev)
    return groups


def _select_per_bucket(
    groups: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Select top-priority candidates up to quota for each bucket.

    Returns (selected_events, bucket_stats) where bucket_stats maps
    bucket -> {"raw": int, "quota": int, "selected": int, "shortfall": int}.
    """
    selected: list[dict[str, Any]] = []
    bucket_stats: dict[str, dict[str, int]] = {}

    for bucket, quota in BUCKET_QUOTAS.items():
        candidates = groups.get(bucket, [])
        raw_count = len(candidates)

        # Sort descending by priority (higher = better)
        candidates_sorted = sorted(
            candidates, key=lambda e: e.get("priority", 0), reverse=True
        )
        chosen = candidates_sorted[:quota]
        shortfall = max(0, quota - len(chosen))

        bucket_stats[bucket] = {
            "raw": raw_count,
            "quota": quota,
            "selected": len(chosen),
            "shortfall": shortfall,
        }
        selected.extend(chosen)

        if shortfall > 0:
            logger.warning(
                "Bucket '%s': shortfall of %d (have %d, need %d)",
                bucket,
                shortfall,
                raw_count,
                quota,
            )

    return selected, bucket_stats


def _enforce_per_file_cap(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop lowest-priority events from files exceeding MAX_EVENTS_PER_FILE."""
    # Group by file_path
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        fp = ev.get("file_path", "<unknown>")
        by_file[fp].append(ev)

    kept: list[dict[str, Any]] = []
    dropped_total = 0

    for fp, file_events in by_file.items():
        if len(file_events) <= MAX_EVENTS_PER_FILE:
            kept.extend(file_events)
            continue

        # Sort descending by priority, keep only top MAX_EVENTS_PER_FILE
        file_events_sorted = sorted(
            file_events, key=lambda e: e.get("priority", 0), reverse=True
        )
        n_drop = len(file_events_sorted) - MAX_EVENTS_PER_FILE
        dropped_total += n_drop
        kept.extend(file_events_sorted[:MAX_EVENTS_PER_FILE])
        logger.info(
            "File '%s': dropped %d lowest-priority events (had %d, cap %d)",
            fp,
            n_drop,
            len(file_events_sorted),
            MAX_EVENTS_PER_FILE,
        )

    if dropped_total:
        logger.info(
            "Per-file cap enforcement dropped %d events total", dropped_total
        )
    return kept


def _sort_for_output(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by file_path then timestamp_sec for readability."""
    return sorted(
        events,
        key=lambda e: (e.get("file_path", ""), e.get("timestamp_sec", 0.0)),
    )


# ── Report generation ───────────────────────────────────────────


def _priority_stats(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute min/max/mean priority per bucket for selected events."""
    by_bucket: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        by_bucket[ev.get("bucket", "unknown")].append(
            ev.get("priority", 0.0)
        )

    stats: dict[str, dict[str, float]] = {}
    for bucket in BUCKET_QUOTAS:
        vals = by_bucket.get(bucket, [])
        if vals:
            stats[bucket] = {
                "min": min(vals),
                "max": max(vals),
                "mean": round(statistics.mean(vals), 3),
            }
        else:
            stats[bucket] = {"min": 0.0, "max": 0.0, "mean": 0.0}
    return stats


def _per_file_distribution(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarise selected events per file."""
    by_file: dict[str, dict[str, Any]] = {}
    for ev in events:
        fp = ev.get("file_path", "<unknown>")
        if fp not in by_file:
            by_file[fp] = {"file": fp, "count": 0, "buckets": set()}
        by_file[fp]["count"] += 1
        by_file[fp]["buckets"].add(ev.get("bucket", "unknown"))

    rows = []
    for fp in sorted(by_file):
        entry = by_file[fp]
        rows.append(
            {
                "file": entry["file"],
                "count": entry["count"],
                "buckets": ", ".join(sorted(entry["buckets"])),
            }
        )
    return rows


def _generate_report(
    bucket_stats: dict[str, dict[str, int]],
    selected: list[dict[str, Any]],
) -> str:
    """Build the Markdown balance report."""
    lines: list[str] = []
    lines.append("# ACMI Candidate Balance Report\n")

    # ── Raw Candidate Counts ──
    lines.append("## Raw Candidate Counts")
    lines.append(
        "| Bucket | Raw Count | Quota | Selected | Shortfall |"
    )
    lines.append(
        "|--------|-----------|-------|----------|-----------|"
    )
    for bucket in BUCKET_QUOTAS:
        s = bucket_stats.get(bucket, {})
        lines.append(
            f"| {bucket} | {s.get('raw', 0)} | {s.get('quota', 0)} "
            f"| {s.get('selected', 0)} | {s.get('shortfall', 0)} |"
        )
    lines.append("")

    # ── Per-File Distribution ──
    lines.append("## Per-File Distribution")
    lines.append("| File | Events Selected | Buckets |")
    lines.append("|------|-----------------|---------|")
    for row in _per_file_distribution(selected):
        lines.append(f"| {row['file']} | {row['count']} | {row['buckets']} |")
    lines.append("")

    # ── Coverage Assessment ──
    shortfall_buckets = [
        b
        for b, s in bucket_stats.items()
        if s.get("shortfall", 0) > 0
    ]
    lines.append("## Coverage Assessment")
    if shortfall_buckets:
        lines.append(
            f"- Buckets with shortfall: {', '.join(shortfall_buckets)}"
        )
        lines.append(
            "- Recommendation: Mine additional ACMI files to fill "
            "under-represented buckets, or relax detection thresholds "
            "for shortfall categories."
        )
    else:
        lines.append("- All buckets met their quotas. No shortfall detected.")
        lines.append("- Recommendation: Dataset is well-balanced; proceed to annotation.")
    lines.append("")

    # ── Per-Bucket Priority Stats ──
    pstats = _priority_stats(selected)
    lines.append("## Per-Bucket Priority Stats")
    lines.append(
        "| Bucket | Min Priority | Max Priority | Mean Priority |"
    )
    lines.append(
        "|--------|-------------|-------------|---------------|"
    )
    for bucket in BUCKET_QUOTAS:
        ps = pstats.get(bucket, {})
        lines.append(
            f"| {bucket} | {ps.get('min', 0):.3f} "
            f"| {ps.get('max', 0):.3f} | {ps.get('mean', 0):.3f} |"
        )
    lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────


def balance_events(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    """End-to-end: load, balance, write output + report."""

    # 1. Load
    raw_events = _load_events(input_path)
    if not raw_events:
        logger.error("No candidate events found in %s", input_path)
        return []

    # 2. Group by bucket
    groups = _group_by_bucket(raw_events)
    unknown = groups.pop("unknown", [])
    if unknown:
        logger.warning(
            "%d events have no recognised bucket and will be skipped",
            len(unknown),
        )

    # 3. Select per-bucket (top priority, up to quota)
    selected, bucket_stats = _select_per_bucket(groups)
    logger.info(
        "After per-bucket selection: %d events (from %d raw)",
        len(selected),
        len(raw_events),
    )

    # 4. Enforce per-file cap
    selected = _enforce_per_file_cap(selected)
    logger.info("After per-file cap (%d max): %d events", MAX_EVENTS_PER_FILE, len(selected))

    # Re-compute bucket stats after per-file cap enforcement so the
    # report reflects final selected counts.
    final_groups = _group_by_bucket(selected)
    for bucket in BUCKET_QUOTAS:
        quota = BUCKET_QUOTAS[bucket]
        final_count = len(final_groups.get(bucket, []))
        raw_count = bucket_stats[bucket]["raw"]
        bucket_stats[bucket]["selected"] = final_count
        bucket_stats[bucket]["shortfall"] = max(0, quota - final_count)
        # Keep raw count from original grouping
        bucket_stats[bucket]["raw"] = raw_count

    # 5. Sort for readability
    selected = _sort_for_output(selected)

    # 6. Write balanced JSONL
    _write_events(selected, output_path)

    # 7. Write report
    report_text = _generate_report(bucket_stats, selected)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Wrote balance report to %s", report_path)

    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and balance ACMI candidate events across buckets.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/acmi_mining/candidate_events.jsonl"),
        help="Path to raw candidate events JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/acmi_mining/candidate_events_balanced.jsonl"),
        help="Path for balanced output JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/acmi_candidate_balance_report.md"),
        help="Path for balance report Markdown (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = balance_events(args.input, args.output, args.report)
    if not result:
        logger.error("No events after balancing — check input file and thresholds.")
        sys.exit(1)
    logger.info("Done. %d balanced events ready for annotation.", len(result))


if __name__ == "__main__":
    main()
