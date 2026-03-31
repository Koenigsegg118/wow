from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from semantic_record_common import (
    read_json_or_jsonl,
    validate_records,
    validate_standard_sft_samples,
    write_json,
)


def _read_mapping(path: str | Path) -> list[dict[str, Any]]:
    payload = read_json_or_jsonl(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected mapping list in {path}")
    return payload


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate semantic_record_v1 canonical records and SFT mappings.")
    parser.add_argument("--input", required=True, help="Canonical semantic_record_v1 JSON or JSONL file.")
    parser.add_argument(
        "--standard-sft-mapping",
        help="Optional JSON or JSONL mapping produced by semantic_sft_export.py --output-mapping.",
    )
    parser.add_argument("--report-json", help="Optional JSON file to write the validation report.")
    return parser


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()

    records = read_json_or_jsonl(args.input)
    errors = validate_records(records)

    mapping_errors: list[str] = []
    if args.standard_sft_mapping:
        mapping_errors = validate_standard_sft_samples(records, _read_mapping(args.standard_sft_mapping))
        errors.extend(mapping_errors)

    report = {
        "input": str(Path(args.input)),
        "record_count": len(records),
        "error_count": len(errors),
        "errors": errors,
    }

    if args.report_json:
        write_json(args.report_json, report)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
