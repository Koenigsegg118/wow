# Semantic Alignment Report

## Canonical Contract

`semantic_record_v1` is the single canonical record. By precedence, the contract is fixed by:

1. `semantic_record_schema_v1.json`
2. `enum_registry_v1.json`
3. `annotation_guideline_v1.md`
4. `sft_export_spec_v1.md`
5. `acmi_field_mapping_v1.md`
6. `contract_diff_checklist_v1.md`
7. `CLAUDE.md`
8. AFSIM audit outputs and `wow_semantic_pack`

The canonical top-level shape is fixed to:

- Required: `schema_version`, `sample_id`, `source`, `obs`, `label`, `evidence`, `quality`
- Optional: `extras`
- Forbidden: any other top-level field (`additionalProperties=false`)

The canonical field facts that adapters must preserve:

- `schema_version` must be exactly `semantic_record_v1`
- sample granularity is one decision instant
- missing values are `null`
- string sentinels like `"unknown"` / `"unavailable"` are invalid
- `enemy` and `ally` remain single objects or `null`
- old source-specific fields never become canonical top-level fields

## Canonical Enums

The enum source is only `enum_registry_v1.json`.

Key enums:

- `source.platform`: `afsim_acmi`, `manual`, `mixed`
- `source.type`: `manual`, `acmi_ai`, `afsim_rule`, `mixed`
- `obs.ego.coalition`: `red`, `blue`, `neutral`, `null`
- `obs.ego.energy_state`: `low`, `medium`, `high`, `null`
- `obs.enemy.target_ref`: `enemy_1`, `enemy_2`, `null`
- `obs.ally.ally_ref`: `ally_1`, `null`
- `label.semantic_state`: `commit_intercept`, `hold_geometry`, `press_attack`, `extend`, `support`, `merge_entry`, `offensive_turn`, `defensive_break`, `energy_manage`, `null`
- `label.role`: `engaged`, `support`, `egressing`, `neutral`, `null`
- `label.target_ref`: `enemy_1`, `enemy_2`, `ally_1`, `null`
- `label.profile_hint`: `p6dof_semantic`, `p6dof_aggressive_turn`, `auto`, `null`
- `quality.tier`: `gold`, `silver`, `bronze`, `weak`

## CLAUDE.md Additions

`CLAUDE.md` does not define new canonical fields. It adds repository-level semantic governance:

- `semantic_data_contract_v1/` is the single semantic data contract
- schema changes must follow the order `schema -> enum registry -> docs/prompts/export spec -> patch summary`
- different sources must not define divergent field names or enum sets
- downstream code must not bypass the canonical contract

These are admissible as process rules and adapter-side conflict-resolution rules. They are not allowed to mutate schema structure.

## CLAUDE.md Semantic Landing

Because `CLAUDE.md` only adds governance and boundary rules, the adapters land its requirements in existing schema slots only:

- mapping rationale goes to `label.rationale_short` or `evidence.notes`
- provenance goes to `source.tool`, `source.annotator`, `evidence.rule_ids`, `evidence.frame_refs`
- source-specific leftovers go to `extras.raw_tags`

No `CLAUDE.md` rule required a new field.

## Conflict Check

No structural conflict was found between `CLAUDE.md` and higher-precedence specs.

Resolved precedence decisions:

| Topic | Higher-precedence source | `CLAUDE.md` status | Final rule |
| --- | --- | --- | --- |
| Canonical top-level fields | `semantic_record_schema_v1.json` | compatible | use schema exactly |
| Enum authority | `enum_registry_v1.json` | compatible | use enum registry exactly |
| `null` vs `"unknown"` | schema + guideline | compatible | always use `null` |
| `enemy` / `ally` cardinality | schema + ACMI field mapping | compatible | keep single object only |
| SFT export routing | `sft_export_spec_v1.md` | compatible | conflict records go review-only |
| Source-specific semantic leftovers | schema + diff checklist | compatible | put in `evidence` or `extras`, never new fields |

## Adapter-Side Resolution Rules

These rules are adapter logic, not schema changes:

- AFSIM legacy semantic labels are mapped into canonical `label.semantic_state`; the original label names stay in `extras.raw_tags`
- AFSIM legacy profile hints are mapped into canonical `label.profile_hint`; the raw hint string stays in `evidence.notes` or `extras.raw_tags`
- AFSIM fire-gate and `must_not_descend` style fields remain auxiliary only and are preserved in `evidence.notes` / `extras.raw_tags`
- ACMI AI `quality_hint` is mapped into canonical `quality`; `quality.tier` is derived, not copied from model output
- `target_switch` is checked against `history.prev_target_ref`; mismatch raises `conflict_flag` instead of changing schema

## Practical Implication

All three routes now converge on one contract:

- ACMI auto/AI labels -> `semantic_record_v1`
- AFSIM audit / semantic pack -> `semantic_record_v1`
- LLM semantic supervisor training exports -> derived only from `semantic_record_v1`

Auxiliary sources may suggest mappings, but they no longer define the contract.
