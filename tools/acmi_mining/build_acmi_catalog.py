#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan all ACMI files in a directory and produce a lightweight catalog index.

Output:
  - JSONL catalog: one JSON object per ACMI file
  - Markdown report: summary statistics

Usage:
  python -m tools.acmi_mining.build_acmi_catalog \
      --input tra_data \
      --output datasets/acmi_mining/acmi_catalog.jsonl \
      --report reports/acmi_catalog_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure wow/ root and ml/ are importable.
# ml/acmi_to_dt_dataset_smooth.py uses bare imports (entity_filter,
# dataset_default_config) that expect the ml/ directory on sys.path.
# ---------------------------------------------------------------------------
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # tools/acmi_mining -> tools -> wow
_ML_DIR = _ROOT_DIR / "ml"
for _d in (_ROOT_DIR, _ML_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from ml.acmi_to_dt_dataset_smooth import (
    read_acmi_text,
    parse_header,
    parse_object_updates,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_WING_KEYWORDS = ("fixedwing", "fighter")

import re as _re
_EXCLUDE_NAME_RE = _re.compile(
    r"(?i)(A-50|E-3|E-2|E-767|AWACS|Sentry|Hawkeye|"
    r"KC-135|KC-10|KC-46|IL-78|Il-78|Tanker|"
    r"C-130|C-17|C-5|An-26|An-12|Il-76|"
    r"Tu-95|B-52|B-1|B-2|Bomber|Transport|JSTAR)"
)


def _is_fixed_wing(type_str: str) -> bool:
    """Return True if entity Type contains 'FixedWing' or 'Fighter' (case-insensitive)."""
    t = type_str.lower()
    return any(kw in t for kw in _FIXED_WING_KEYWORDS)


def _is_combat_fixed_wing(name: str, type_str: str) -> bool:
    """Return True only for combat fixed-wing (excludes AWACS, tankers, etc.)."""
    if not _is_fixed_wing(type_str):
        return False
    combined = f"{name}|{type_str}"
    if _EXCLUDE_NAME_RE.search(combined):
        return False
    return True


def _coalition_bucket(coalition: str) -> str | None:
    """Normalise coalition string to 'red' / 'blue' / None."""
    c = coalition.strip().lower()
    if c in ("red", "enemies"):
        return "red"
    if c in ("blue", "allies"):
        return "blue"
    return None


# ---------------------------------------------------------------------------
# Per-file cataloguing
# ---------------------------------------------------------------------------

def catalog_one_file(acmi_path: Path) -> dict:
    """Parse a single ACMI file and return a catalog record (dict).

    On failure the record will have ``notes`` set to the error description
    and most other fields set to None.
    """
    record: dict = {
        "file_path": str(acmi_path),
        "file_name": acmi_path.name,
        "duration_sec": None,
        "entity_names": None,
        "fixed_wing_count": None,
        "red_count": None,
        "blue_count": None,
        "candidate_2v2": None,
        "candidate_multiship": None,
        "time_range": None,
        "notes": None,
    }

    try:
        text = read_acmi_text(acmi_path)
    except Exception as exc:
        record["notes"] = f"parse_error: read_acmi_text failed: {exc}"
        logger.warning("Failed to read %s: %s", acmi_path, exc)
        return record

    try:
        lines = text.splitlines()
        header, start_idx = parse_header(lines)
        objects, tracks = parse_object_updates(lines, start_idx)
    except Exception as exc:
        record["notes"] = f"parse_error: parsing failed: {exc}"
        logger.warning("Failed to parse %s: %s", acmi_path, exc)
        return record

    # ---- Collect per-entity info ----
    entity_names: list[str] = []
    combat_fw_names: list[str] = []
    fixed_wing_count = 0
    combat_fw_count = 0
    red_fw = 0
    blue_fw = 0
    red_total = 0
    blue_total = 0

    all_times: list[float] = []

    for oid, info in objects.items():
        name = info.get("Name", oid)
        entity_names.append(name)

        etype = info.get("Type", "")
        coalition = info.get("Coalition", "")

        is_fw = _is_fixed_wing(etype)
        is_combat_fw = _is_combat_fixed_wing(name, etype)
        if is_fw:
            fixed_wing_count += 1
        if is_combat_fw:
            combat_fw_count += 1
            combat_fw_names.append(name)

        bucket = _coalition_bucket(coalition)
        if bucket == "red":
            red_total += 1
            if is_combat_fw:
                red_fw += 1
        elif bucket == "blue":
            blue_total += 1
            if is_combat_fw:
                blue_fw += 1

        # Gather time extents from tracks (if entity has position data)
        pts = tracks.get(oid)
        if pts:
            for pt in pts:
                all_times.append(pt[0])

    # ---- Time range ----
    if all_times:
        t_min = min(all_times)
        t_max = max(all_times)
        record["time_range"] = {"start": round(t_min, 3), "end": round(t_max, 3)}
        record["duration_sec"] = round(t_max - t_min, 3)
    else:
        record["time_range"] = {"start": None, "end": None}
        record["duration_sec"] = 0.0

    record["entity_names"] = entity_names
    record["combat_fw_names"] = combat_fw_names
    record["fixed_wing_count"] = fixed_wing_count
    record["combat_fw_count"] = combat_fw_count
    record["red_count"] = red_fw    # now counts combat FW only
    record["blue_count"] = blue_fw  # now counts combat FW only

    # ---- Scenario shape heuristics (based on combat FW, not all entities) ----
    record["candidate_2v2"] = (red_fw == 2 and blue_fw == 2)
    record["candidate_multiship"] = (combat_fw_count > 4)

    # ---- Notes ----
    warnings: list[str] = []
    if not entity_names:
        warnings.append("no_entities")
    if not all_times:
        warnings.append("no_position_data")
    record["notes"] = "; ".join(warnings) if warnings else None

    return record


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------

def find_acmi_files(input_dir: Path) -> list[Path]:
    """Return sorted list of .acmi and .zip.acmi files under *input_dir*."""
    files: list[Path] = []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if name_lower.endswith(".acmi") or name_lower.endswith(".zip.acmi"):
            files.append(p)
    return files


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(records: list[dict], report_path: Path) -> None:
    """Write a Markdown summary report."""
    total = len(records)
    ok = [r for r in records if r["duration_sec"] is not None and r["duration_sec"] > 0]
    failed = [r for r in records if r["notes"] and r["notes"].startswith("parse_error")]

    n_2v2 = sum(1 for r in ok if r.get("candidate_2v2"))
    n_multi = sum(1 for r in ok if r.get("candidate_multiship"))

    durations = [r["duration_sec"] for r in ok if r["duration_sec"]]
    fw_counts = [r["fixed_wing_count"] for r in ok if r["fixed_wing_count"] is not None]
    entity_counts = [len(r["entity_names"]) for r in ok if r["entity_names"] is not None]

    lines = [
        "# ACMI Catalog Report",
        "",
        "## Overview",
        "",
        f"- **Total files scanned**: {total}",
        f"- **Successfully parsed**: {len(ok)}",
        f"- **Parse failures**: {len(failed)}",
        "",
        "## Scenario Shape",
        "",
        f"- **Candidate 2v2**: {n_2v2}",
        f"- **Candidate multiship (>4 FW)**: {n_multi}",
        "",
    ]

    if durations:
        lines += [
            "## Duration Statistics (seconds)",
            "",
            f"- min: {min(durations):.1f}",
            f"- max: {max(durations):.1f}",
            f"- mean: {sum(durations) / len(durations):.1f}",
            f"- median: {sorted(durations)[len(durations) // 2]:.1f}",
            "",
        ]

    if fw_counts:
        lines += [
            "## Fixed-Wing Count Distribution",
            "",
        ]
        from collections import Counter
        dist = Counter(fw_counts)
        for k in sorted(dist):
            lines.append(f"- {k} fixed-wing: {dist[k]} files")
        lines.append("")

    if entity_counts:
        lines += [
            "## Entity Count Statistics",
            "",
            f"- min: {min(entity_counts)}",
            f"- max: {max(entity_counts)}",
            f"- mean: {sum(entity_counts) / len(entity_counts):.1f}",
            "",
        ]

    if failed:
        lines += [
            "## Parse Failures",
            "",
        ]
        for r in failed:
            lines.append(f"- `{r['file_name']}`: {r['notes']}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight ACMI catalog index.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="tra_data",
        help="Root directory to scan for ACMI files (default: tra_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/acmi_mining/acmi_catalog.jsonl",
        help="Output JSONL catalog path (default: datasets/acmi_mining/acmi_catalog.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/acmi_catalog_report.md",
        help="Output Markdown report path (default: reports/acmi_catalog_report.md)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve paths relative to wow/ root
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = _ROOT_DIR / input_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _ROOT_DIR / output_path

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = _ROOT_DIR / report_path

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    acmi_files = find_acmi_files(input_dir)
    logger.info("Found %d ACMI file(s) in %s", len(acmi_files), input_dir)

    if not acmi_files:
        logger.warning("No ACMI files found. Exiting.")
        sys.exit(0)

    # ---- Build catalog ----
    records: list[dict] = []
    for i, fpath in enumerate(acmi_files, 1):
        logger.info("[%d/%d] Processing %s", i, len(acmi_files), fpath.name)
        try:
            rec = catalog_one_file(fpath)
        except Exception:
            logger.error("Unexpected error on %s:\n%s", fpath, traceback.format_exc())
            rec = {
                "file_path": str(fpath),
                "file_name": fpath.name,
                "duration_sec": None,
                "entity_names": None,
                "fixed_wing_count": None,
                "red_count": None,
                "blue_count": None,
                "candidate_2v2": None,
                "candidate_multiship": None,
                "time_range": None,
                "notes": f"parse_error: unexpected: {traceback.format_exc(limit=1)}",
            }
        records.append(rec)

    # ---- Write JSONL ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Catalog written to %s (%d records)", output_path, len(records))

    # ---- Write report ----
    generate_report(records, report_path)

    # ---- Summary ----
    ok_count = sum(1 for r in records if r["duration_sec"] is not None and r["duration_sec"] > 0)
    fail_count = sum(1 for r in records if r["notes"] and r["notes"].startswith("parse_error"))
    logger.info("Done. %d OK, %d failed, %d total.", ok_count, fail_count, len(records))


if __name__ == "__main__":
    main()
