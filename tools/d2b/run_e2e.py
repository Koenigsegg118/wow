"""End-to-end D2b pipeline runner.

Generates synthetic 2v2 data, runs the full pipeline, and produces
all output files + reports.

Usage:
    cd D:\\AF2.9\\afsim-2.9.0-win64\\wow
    python -m tools.d2b.run_e2e
    python -m tools.d2b.run_e2e --skip-synthetic  # if obs already exists
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────

BASE = pathlib.Path("datasets/d2b")
REPORTS = pathlib.Path("reports")

P_SYNTHETIC = BASE / "synthetic_2v2_obs.jsonl"
P_OBS = BASE / "obs_samples.jsonl"
P_FORMAT_A = BASE / "obs_formatA_requests.jsonl"
P_RULE = BASE / "rule_labeled.jsonl"
P_HARD = BASE / "hardcase_requests.jsonl"
P_GPT_RESP = BASE / "gpt_responses.jsonl"
P_GPT_VALID = BASE / "gpt_validated.jsonl"
P_GPT_CANON = BASE / "canonical_from_gpt.jsonl"
P_GPT_GUARDED = BASE / "canonical_from_gpt_guarded.jsonl"
P_MERGED = BASE / "merged_canonical.jsonl"
P_SFT_STD = BASE / "sft_standard.jsonl"
P_SFT_REV = BASE / "sft_review.jsonl"


def _count_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run(skip_synthetic: bool = False) -> None:
    """Run the full D2b pipeline end-to-end."""
    BASE.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate synthetic data ───────────────────────
    if not skip_synthetic:
        print("\n=== Step 1: Generate synthetic 2v2 obs ===")
        from tools.d2b.generate_synthetic_2v2 import generate_all
        n = generate_all(output_path=P_SYNTHETIC)
        print(f"  Generated {n} samples → {P_SYNTHETIC}")
    else:
        print("\n=== Step 1: Skipped (--skip-synthetic) ===")

    if not P_SYNTHETIC.exists():
        print(f"ERROR: {P_SYNTHETIC} not found. Run without --skip-synthetic.")
        sys.exit(1)

    # ── Step 2: Copy as obs_samples ───────────────────────────
    print("\n=== Step 2: Copy to obs_samples.jsonl ===")
    shutil.copy2(P_SYNTHETIC, P_OBS)
    print(f"  {_count_lines(P_OBS)} samples → {P_OBS}")

    # ── Step 3: Convert to format_A ───────────────────────────
    print("\n=== Step 3: Convert to format_A ===")
    from tools.d2b.adapter_obs_to_formatA import build_format_a_batch
    build_format_a_batch(P_OBS, P_FORMAT_A)
    print(f"  {_count_lines(P_FORMAT_A)} requests → {P_FORMAT_A}")

    # ── Step 4: Rule labeling ─────────────────────────────────
    print("\n=== Step 4: Rule labeling (easy cases) ===")
    from tools.d2b.rule_labeler_v1 import label_batch
    _, rule_stats = label_batch(P_OBS, P_RULE)
    print(f"  Labeled {rule_stats.get('labeled', '?')} easy cases → {P_RULE}")
    print(f"  Stats: {json.dumps(rule_stats, indent=2, ensure_ascii=False)}")

    # ── Step 5: Build hard-case requests ──────────────────────
    print("\n=== Step 5: Build hard-case requests ===")
    from tools.d2b.build_hardcase_requests import build_hardcase_batch
    _, hard_stats = build_hardcase_batch(P_OBS, P_RULE, P_HARD)
    print(f"  {hard_stats.get('total_hard_cases', '?')} hard cases → {P_HARD}")

    # ── Step 6: GPT stub (returns null labels) ────────────────
    print("\n=== Step 6: GPT stub labeling ===")
    from tools.d2b.gpt_caller_stub import GPTCallerStub
    stub = GPTCallerStub()
    stub.call_batch(P_HARD, P_GPT_RESP)
    print(f"  {_count_lines(P_GPT_RESP)} responses → {P_GPT_RESP}")

    # ── Step 7: Validate GPT output ───────────────────────────
    print("\n=== Step 7: Validate GPT output ===")
    from tools.d2b.validate_gpt_output import validate_batch
    _, val_stats = validate_batch(P_GPT_RESP, P_OBS, P_GPT_VALID)
    print(f"  {val_stats.get('valid', '?')} valid → {P_GPT_VALID}")

    # ── Step 8: Adapt to canonical ────────────────────────────
    print("\n=== Step 8: Adapt GPT → canonical ===")
    from tools.d2b.adapter_ai_to_canonical import adapt_batch
    _, adapt_stats = adapt_batch(P_GPT_VALID, P_OBS, P_GPT_CANON)
    print(f"  {adapt_stats.get('converted', '?')} canonical records → {P_GPT_CANON}")

    # ── Step 8b: Label guardrails ───────────────────────────────
    print("\n=== Step 8b: Apply label guardrails ===")
    from tools.d2b.label_guardrails import apply_guardrails_batch
    _, guard_stats = apply_guardrails_batch(
        P_GPT_CANON, P_OBS, P_RULE, P_GPT_GUARDED)
    print(f"  Guarded: {guard_stats.get('guarded', 0)} "
          f"(protected={guard_stats.get('protected', 0)}, "
          f"nullified={guard_stats.get('nullified', 0)}) → {P_GPT_GUARDED}")

    # ── Step 9: Merge + SFT export ────────────────────────────
    print("\n=== Step 9: Merge + SFT export ===")
    from tools.d2b.merge_and_export import merge_sources, export_sft
    _, merge_stats = merge_sources(P_RULE, P_GPT_GUARDED, output_path=P_MERGED)
    sft_stats = export_sft(P_MERGED, P_SFT_STD, P_SFT_REV)
    print(f"  Merged: {merge_stats.get('merged_total', '?')} → {P_MERGED}")
    print(f"  SFT standard: {sft_stats.get('standard_exported', '?')} → {P_SFT_STD}")
    print(f"  SFT review:   {sft_stats.get('review_exported', '?')} → {P_SFT_REV}")

    # ── Step 10: Reports ──────────────────────────────────────
    print("\n=== Step 10: Generate reports ===")
    from tools.d2b.report import generate_reports
    report_files = generate_reports(P_MERGED, REPORTS)
    for name, path in report_files.items():
        print(f"  {name}: {path}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  D2b End-to-End Pipeline Complete")
    print("=" * 60)
    files = [P_SYNTHETIC, P_OBS, P_FORMAT_A, P_RULE, P_HARD,
             P_GPT_RESP, P_GPT_VALID, P_GPT_CANON, P_MERGED,
             P_SFT_STD, P_SFT_REV]
    for f in files:
        lines = _count_lines(f)
        status = "OK" if lines > 0 else "EMPTY"
        print(f"  [{status:5s}] {f} ({lines} lines)")
    print("=" * 60)

    # ── Sanity checks ─────────────────────────────────────────
    errors = []
    if _count_lines(P_MERGED) == 0:
        errors.append("merged_canonical.jsonl is empty")
    if _count_lines(P_SFT_STD) == 0:
        errors.append("sft_standard.jsonl is empty")
    if _count_lines(P_RULE) == 0:
        errors.append("rule_labeled.jsonl is empty")

    # Check for forbidden strings
    for f in [P_MERGED, P_SFT_STD, P_SFT_REV]:
        if f.exists():
            content = f.read_text(encoding="utf-8")
            for forbidden in ['"unknown"', '"unavailable"', '"N/A"']:
                if forbidden in content:
                    errors.append(f"{f.name} contains {forbidden}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("\nAll sanity checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D2b pipeline end-to-end")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip synthetic data generation (use existing obs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    run(skip_synthetic=args.skip_synthetic)


if __name__ == "__main__":
    main()
