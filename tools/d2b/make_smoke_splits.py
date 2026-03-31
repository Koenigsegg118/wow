#!/usr/bin/env python
"""Split a restricted JSONL training pack into train / dev sets.

Stratified by ``semantic_state`` (extracted from the SFT "output" field).
For labels with fewer than 5 samples, at least 1 is guaranteed in dev.

Usage
-----
python -m tools.d2b.make_smoke_splits \
    --input  datasets/trainpacks/restricted_nonnull_core.jsonl \
    --outdir datasets/trainpacks/ \
    --report reports/restricted_split_report.md
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _extract_semantic_state(record: dict) -> str:
    """Return the semantic_state string from an SFT record.

    The SFT format stores the model output as a JSON string in the
    ``output`` field.  We parse that and pull ``semantic_state``.
    Falls back to ``"UNKNOWN"`` if the field is missing or unparseable.
    """
    raw = record.get("output", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    # output may be {"label": {"semantic_state": ...}} or {"semantic_state": ...}
    label = raw.get("label", raw)
    return label.get("semantic_state", "UNKNOWN")


def _extract_source_type(record: dict) -> str:
    """Best-effort extraction of source_type from a record."""
    # Try inside input.source.type (canonical SFT format)
    inp = record.get("input", {})
    if isinstance(inp, str):
        try:
            inp = json.loads(inp)
        except (json.JSONDecodeError, TypeError):
            inp = {}
    if isinstance(inp, dict):
        src = inp.get("source", {})
        if isinstance(src, dict):
            t = src.get("type")
            if t:
                return str(t)
    return "unknown"


# ---------------------------------------------------------------------------
# core split logic
# ---------------------------------------------------------------------------

def stratified_split(
    records: list[dict],
    dev_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified 80/20 split on semantic_state.

    For labels with < 5 samples, at least 1 goes to dev.
    """
    rng = random.Random(seed)

    # group by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        label = _extract_semantic_state(r)
        by_label[label].append(r)

    train_out: list[dict] = []
    dev_out: list[dict] = []

    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        n = len(items)
        n_dev = max(1, round(n * dev_ratio)) if n < 5 else round(n * dev_ratio)
        # ensure at least 1 dev for any non-empty bucket
        n_dev = max(n_dev, 1) if n >= 1 else 0
        dev_out.extend(items[:n_dev])
        train_out.extend(items[n_dev:])

    # final shuffle within each split for good measure
    rng.shuffle(train_out)
    rng.shuffle(dev_out)
    return train_out, dev_out


# ---------------------------------------------------------------------------
# report generation
# ---------------------------------------------------------------------------

def generate_report(
    records: list[dict],
    train: list[dict],
    dev: list[dict],
) -> str:
    """Return Markdown report string."""

    total = len(records)
    n_train = len(train)
    n_dev = len(dev)

    label_all = Counter(_extract_semantic_state(r) for r in records)
    label_train = Counter(_extract_semantic_state(r) for r in train)
    label_dev = Counter(_extract_semantic_state(r) for r in dev)

    src_train = Counter(_extract_source_type(r) for r in train)
    src_dev = Counter(_extract_source_type(r) for r in dev)

    all_labels = sorted(label_all.keys())

    # check dev covers all non-zero labels
    nonzero_labels = {l for l, c in label_all.items() if l not in ("None", "null", "UNKNOWN", "")}
    dev_labels = set(label_dev.keys())
    dev_covers_all = nonzero_labels.issubset(dev_labels)

    lines: list[str] = []
    lines.append("# Restricted Split Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total records | {total} |")
    lines.append(f"| Train | {n_train} ({n_train/total*100:.1f}%) |")
    lines.append(f"| Dev | {n_dev} ({n_dev/total*100:.1f}%) |")
    lines.append(f"| Dev covers all non-zero labels | {'YES' if dev_covers_all else 'NO'} |")
    lines.append("")

    # per-label distribution
    lines.append("## Per-Label Distribution")
    lines.append("")
    lines.append("| Label | Total | Train | Dev |")
    lines.append("|-------|-------|-------|-----|")
    for label in all_labels:
        lines.append(
            f"| {label} | {label_all[label]} "
            f"| {label_train.get(label, 0)} "
            f"| {label_dev.get(label, 0)} |"
        )
    lines.append("")

    # source type distribution
    lines.append("## Source Type Distribution")
    lines.append("")
    all_sources = sorted(set(list(src_train.keys()) + list(src_dev.keys())))
    lines.append("| Source | Train | Dev |")
    lines.append("|--------|-------|-----|")
    for src in all_sources:
        lines.append(f"| {src} | {src_train.get(src, 0)} | {src_dev.get(src, 0)} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split restricted JSONL training pack into train / dev."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to restricted_nonnull_core.jsonl",
    )
    parser.add_argument(
        "--outdir", required=True, type=Path,
        help="Directory for train/dev JSONL outputs",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Path for the Markdown split report (default: <outdir>/restricted_split_report.md)",
    )
    parser.add_argument(
        "--dev-ratio", type=float, default=0.20,
        help="Fraction of data for dev set (default: 0.20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # read input
    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {input_path}")

    # split
    train, dev = stratified_split(records, dev_ratio=args.dev_ratio, seed=args.seed)
    print(f"Split: train={len(train)}, dev={len(dev)}")

    # write outputs
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "restricted_nonnull_train.jsonl"
    dev_path = outdir / "restricted_nonnull_dev.jsonl"

    for path, data in [(train_path, train), (dev_path, dev)]:
        with open(path, "w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} records to {path}")

    # report
    report_path: Path = args.report or (outdir / "restricted_split_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = generate_report(records, train, dev)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
