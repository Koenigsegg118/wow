# Adapter README

## Purpose

This directory now treats `semantic_record_v1` as the only canonical semantic record. Three upstream routes must enter through adapters:

- ACMI automatic / AI annotation
- AFSIM rule-audit and WOW semantic pack data
- downstream semantic-supervisor / SFT export

`CLAUDE.md` sits above the adapters as a governance rule source only. It can clarify precedence, conflict handling, and semantic boundaries, but it cannot change schema structure.

## Route Into Canonical

### 1. AFSIM rule data

Use `afsim_rule_to_semantic_record.py`.

Inputs:

- `wow_semantic_sft.jsonl`
- `wow_semantic_preference.jsonl`
- `wow_hardcase_adversary_templates.yaml`
- optional AFSIM audit files (`afsim_rules.jsonl`, `semantic_mapping_candidates.csv`)

Output:

- canonical `semantic_record_v1` JSONL

Rules:

- old `semantic_state` is mapped into `label.semantic_state`
- old `target_policy` never becomes a canonical field; it informs `label.target_ref` only when the target can be grounded without fabrication
- old `profile_hint` is mapped into canonical `label.profile_hint`; the raw string is preserved in `evidence` / `extras`
- fire-gate and other non-canonical tactical leftovers stay in `evidence.notes` or `extras.raw_tags`

### 2. ACMI AI data

Use `acmi_ai_to_semantic_record.py`.

Inputs:

- canonical or near-canonical `obs`
- AI `label`
- AI `quality_hint`
- optional prompt/spec references and evidence hints

Output:

- canonical `semantic_record_v1` JSONL

Rules:

- fill only schema-defined fields
- missing values become `null`
- `quality.tier` is derived, not copied from model output
- `target_switch` is checked against `history.prev_target_ref`

### 3. SFT export

Use `semantic_sft_export.py`.

Input:

- canonical `semantic_record_v1` records only

Outputs:

- standard SFT JSONL
- review SFT JSONL

Rules:

- standard SFT uses `obs -> input` and `label -> output`
- review SFT additionally includes `evidence` and `label.rationale_short`
- `conflict_flag=true` records are review-only

## What Belongs Where

### `label`

Put only canonical tactical outputs here:

- `semantic_state`
- `target_ref`
- `role`
- `constraints`
- `profile_hint`
- `event_flags`
- `rationale_short`

### `evidence`

Put traceable support here:

- `rule_ids`
- `acmi_refs`
- `frame_refs`
- threshold notes
- mapping rationale
- conflict notes

### `extras`

Put source-private leftovers here only if the schema allows them:

- `raw_object_ids`
- `raw_tags`
- `weapon_event_refs`

Do not use `extras` to smuggle new structured canonical fields.

## Common Errors

### Error: legacy fields leaked into canonical top level

Symptoms:

- `semantic_state`, `target_policy`, `source_rule_id`, `state_summary` still appear at top level

Fix:

- map them into `label`, `evidence`, or `extras`
- rerun `semantic_record_validator.py`

### Error: `"unknown"` or `"unavailable"` appears in output

Fix:

- replace with `null`
- never use string sentinels for missing values

### Error: conflict samples exported to standard SFT

Fix:

- export with `semantic_sft_export.py --format standard`
- validate standard mapping with `semantic_record_validator.py --standard-sft-mapping`

### Error: AFSIM private semantics overwrote canonical enums

Fix:

- keep canonical enums from `enum_registry_v1.json`
- preserve raw AFSIM semantic names only in `evidence.notes` / `extras.raw_tags`

### Error: `CLAUDE.md` appears to require a new field

Fix:

- it does not
- land the requirement inside existing `label`, `evidence`, `quality`, or `extras`
- if that is impossible, schema wins and the requirement remains documentation-only
