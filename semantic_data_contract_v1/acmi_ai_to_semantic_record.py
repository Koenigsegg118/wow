from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from semantic_record_common import (
    append_note,
    build_quality,
    build_source,
    canonicalize_evidence,
    canonicalize_extras,
    canonicalize_label,
    canonicalize_obs,
    compute_target_switch,
    parse_confidence,
    read_json_or_jsonl,
    sanitize_sample_id,
    sanitize_string_list,
    validate_records,
    write_jsonl,
)


def _extract_ai_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "ai_output" in raw and isinstance(raw["ai_output"], dict):
        label = raw["ai_output"].get("label") or {}
        quality_hint = raw["ai_output"].get("quality_hint") or {}
        return label, quality_hint
    return raw.get("label") or {}, raw.get("quality_hint") or {}


def adapt_acmi_ai_records(
    records: list[dict[str, Any]],
    *,
    prompt_file: str | None = None,
    examples_spec: str | None = None,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    prompt_name = Path(prompt_file).name if prompt_file else None
    examples_name = Path(examples_spec).name if examples_spec else None

    for index, raw in enumerate(records, start=1):
        label_raw, quality_hint = _extract_ai_payload(raw)
        obs = canonicalize_obs(raw.get("obs") if "obs" in raw else raw)
        label = canonicalize_label(label_raw)

        expected_switch = compute_target_switch(obs["history"].get("prev_target_ref"), label.get("target_ref"))
        conflict_flag = bool(quality_hint.get("conflict_flag", False))
        needs_review = bool(quality_hint.get("needs_review", False))
        evidence_note = None
        if label["event_flags"]["target_switch"] is None and expected_switch is not None:
            label["event_flags"]["target_switch"] = expected_switch
        elif expected_switch is not None and label["event_flags"]["target_switch"] != expected_switch:
            conflict_flag = True
            needs_review = True
            evidence_note = "target_switch mismatch between history.prev_target_ref and label.target_ref"

        if conflict_flag and not needs_review:
            needs_review = True

        confidence = parse_confidence(quality_hint.get("confidence"))
        quality = build_quality(
            source_type="acmi_ai",
            confidence=confidence,
            conflict_flag=conflict_flag,
            needs_review=needs_review,
        )

        source_raw = raw.get("source") or {}
        source = build_source(
            platform="afsim_acmi",
            source_type="acmi_ai",
            scenario_id=source_raw.get("scenario_id"),
            episode_id=source_raw.get("episode_id"),
            timestamp_sec=source_raw.get("timestamp_sec"),
            annotator=source_raw.get("annotator"),
            tool=source_raw.get("tool") or "acmi_ai_to_semantic_record.py",
        )

        evidence = canonicalize_evidence(
            {
                "rule_ids": raw.get("rule_ids") or [],
                "acmi_refs": raw.get("acmi_refs") or [],
                "frame_refs": raw.get("frame_refs") or [],
                "notes": append_note(
                    None,
                    "; ".join(sanitize_string_list(raw.get("evidence_hints"))),
                    evidence_note,
                    f"prompt={prompt_name}" if prompt_name else None,
                    f"examples={examples_name}" if examples_name else None,
                ),
            }
        )

        extras = canonicalize_extras(
            raw.get("extras")
            or {
                "raw_object_ids": raw.get("raw_object_ids"),
                "raw_tags": sanitize_string_list(raw.get("raw_tags")),
                "weapon_event_refs": sanitize_string_list(raw.get("weapon_event_refs")),
            }
        )

        sample_id = sanitize_sample_id(raw.get("sample_id") or f"acmi_ai_{index:04d}")
        outputs.append(
            {
                "schema_version": "semantic_record_v1",
                "sample_id": sample_id,
                "source": source,
                "obs": obs,
                "label": label,
                "evidence": evidence,
                "quality": quality,
                **({"extras": extras} if extras is not None else {}),
            }
        )

    return outputs


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ACMI AI label outputs into semantic_record_v1.")
    parser.add_argument("--input", required=True, help="JSON or JSONL ACMI AI adapter input")
    parser.add_argument("--output", required=True, help="Output semantic_record_v1 JSONL path")
    parser.add_argument("--prompt-file", help="Optional acmi_annotation_prompt_v1.txt path")
    parser.add_argument("--examples-spec", help="Optional acmi_auto_label_io_examples.json path")
    parser.add_argument("--validate", action="store_true", help="Validate converted records before writing")
    return parser


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()
    records = adapt_acmi_ai_records(
        read_json_or_jsonl(args.input),
        prompt_file=args.prompt_file,
        examples_spec=args.examples_spec,
    )

    if args.validate:
        errors = validate_records(records)
        if errors:
            for error in errors:
                print(error)
            return 1

    write_jsonl(args.output, records)
    print(f"[acmi_ai_to_semantic_record] wrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
