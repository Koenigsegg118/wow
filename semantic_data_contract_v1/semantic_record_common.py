from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "semantic_record_schema_v1.json"
ENUM_REGISTRY_PATH = ROOT / "enum_registry_v1.json"

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
ENUM_REGISTRY = json.loads(ENUM_REGISTRY_PATH.read_text(encoding="utf-8"))
SCHEMA_VALIDATOR = Draft202012Validator(SCHEMA)

MISSING_SENTINELS = {"unknown", "unavailable"}
LEGACY_TOP_LEVEL_FIELDS = {
    "id",
    "semantic_state",
    "target_policy",
    "profile_hint",
    "constraints",
    "source_rule_id",
    "state_summary",
    "split",
    "quality_tag",
    "chosen",
    "rejected",
    "pair_type",
}

EGO_FIELDS = [
    "runtime_id",
    "callsign",
    "coalition",
    "speed_mps",
    "altitude_m",
    "heading_deg",
    "pitch_deg",
    "roll_deg",
    "vertical_speed_mps",
    "pos_east_m",
    "pos_north_m",
    "energy_state",
]
ENEMY_FIELDS = [
    "target_ref",
    "target_runtime_id",
    "callsign",
    "range_m",
    "bearing_deg",
    "aspect_deg",
    "closure_mps",
    "alt_diff_m",
    "is_primary_threat",
]
ALLY_FIELDS = ["ally_ref", "ally_runtime_id", "callsign", "range_m", "bearing_deg", "is_supporting"]
ENGAGEMENT_FIELDS = ["is_merge", "is_defensive", "has_shot_opportunity"]
HISTORY_FIELDS = ["prev_semantic_state", "prev_token_family", "prev_target_ref"]
CONSTRAINT_FIELDS = [
    "prefer_energy_preserve",
    "allow_aggressive_turn",
    "must_support_teammate",
    "should_abort_if_threatened",
]
EVENT_FLAG_FIELDS = ["target_switch"]

STANDARD_INSTRUCTION = (
    "根据以下空战态势信息，判断己方飞机当前的战术意图。请输出 JSON 格式的标签，"
    "包含 semantic_state（战术状态）、target_ref（目标引用）、role（角色）、"
    "constraints（行为约束）、profile_hint（飞行剖面建议）和 event_flags（事件标志）。"
    "仅输出你有把握判断的字段。"
)
REVIEW_INSTRUCTION = (
    "请审阅以下空战态势观测和标注证据，综合判断己方飞机当前的战术意图。请输出 JSON "
    "格式的标签，包含 semantic_state（战术状态）、target_ref（目标引用）、"
    "role（角色）、constraints（行为约束）、profile_hint（飞行剖面建议）、"
    "event_flags（事件标志）和 rationale_short（一句话判断理由）。仅输出你有把握判断的字段。"
)

LEGACY_STATE_TO_CANONICAL = {
    "OBFM_WEZ_commit": "commit_intercept",
    "OBFM_AW_assess": "hold_geometry",
    "OBFM_CZ_hold": "support",
    "OBFM_repo": "extend",
    "OBFM_TC_entry": "support",
    "DBFM_jink_assess": "defensive_break",
    "HABFM_TCX": "merge_entry",
}

RULE_ID_TO_CANONICAL = {
    "AIR001": "commit_intercept",
    "AIR002": "commit_intercept",
    "AIR005": "extend",
    "AIR006": "extend",
    "AIR009": "commit_intercept",
    "AIR020": "support",
    "AIR022": "hold_geometry",
    "AIR023": "hold_geometry",
    "AIR025": "commit_intercept",
    "AIR026": "commit_intercept",
    "AIR017": "commit_intercept",
    "BRW001": "merge_entry",
    "BRW002": "support",
    "BRW003": "defensive_break",
    "BRW008": "commit_intercept",
    "BRW009": "commit_intercept",
    "OAB002": "support",
    "OAB003": "hold_geometry",
    "OAB004": "commit_intercept",
}

RULE_ID_TO_TARGET_REF = {
    "AIR001": "enemy_1",
    "AIR002": "enemy_1",
    "AIR006": "enemy_1",
    "AIR009": "enemy_1",
    "AIR020": "enemy_1",
    "AIR022": "enemy_1",
    "AIR023": "enemy_1",
    "AIR025": "enemy_1",
    "AIR026": "enemy_1",
    "AIR017": "enemy_1",
    "BRW001": "enemy_1",
    "BRW002": "ally_1",
    "BRW003": None,
    "BRW008": "enemy_1",
    "BRW009": "enemy_1",
    "OAB002": "enemy_1",
    "OAB003": "enemy_1",
    "OAB004": "enemy_1",
}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return payload["records"]
        return [payload]
    raise ValueError(f"Unsupported input payload type in {path}")


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(path: str | Path, payload: Any) -> None:
    Path(path).write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _null_object(fields: Iterable[str]) -> dict[str, Any]:
    return {field: None for field in fields}


def build_empty_obs() -> dict[str, Any]:
    return {
        "ego": _null_object(EGO_FIELDS),
        "enemy": None,
        "ally": None,
        "engagement": _null_object(ENGAGEMENT_FIELDS),
        "history": _null_object(HISTORY_FIELDS),
    }


def build_empty_label() -> dict[str, Any]:
    return {
        "semantic_state": None,
        "target_ref": None,
        "role": None,
        "constraints": _null_object(CONSTRAINT_FIELDS),
        "profile_hint": None,
        "event_flags": _null_object(EVENT_FLAG_FIELDS),
        "rationale_short": None,
    }


def build_empty_evidence() -> dict[str, Any]:
    return {"rule_ids": [], "acmi_refs": [], "frame_refs": [], "notes": None}


def build_empty_quality() -> dict[str, Any]:
    return {"tier": "bronze", "confidence": None, "needs_review": False, "conflict_flag": False}


def build_empty_extras() -> dict[str, Any]:
    return {"raw_object_ids": None, "raw_tags": None, "weapon_event_refs": None}


def normalize_missing(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in MISSING_SENTINELS:
            return None
        return stripped
    if isinstance(value, dict):
        return {key: normalize_missing(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_missing(item) for item in value]
    return value


def sanitize_string_list(values: Iterable[Any] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for item in values:
        item = normalize_missing(item)
        if item is None:
            continue
        result.append(str(item))
    return result


def _coerce_object(raw: Any, fields: list[str]) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    output = _null_object(fields)
    for field in fields:
        output[field] = normalize_missing(raw.get(field))
    return output


def canonicalize_obs(raw_obs: Any) -> dict[str, Any]:
    payload = raw_obs or {}
    if isinstance(payload, dict) and "obs" in payload and isinstance(payload["obs"], dict):
        payload = payload["obs"]
    obs = build_empty_obs()
    obs["ego"] = _coerce_object(payload.get("ego") if isinstance(payload, dict) else {}, EGO_FIELDS) or _null_object(EGO_FIELDS)
    obs["enemy"] = _coerce_object(payload.get("enemy") if isinstance(payload, dict) else None, ENEMY_FIELDS)
    obs["ally"] = _coerce_object(payload.get("ally") if isinstance(payload, dict) else None, ALLY_FIELDS)
    obs["engagement"] = _coerce_object(payload.get("engagement") if isinstance(payload, dict) else None, ENGAGEMENT_FIELDS) or _null_object(ENGAGEMENT_FIELDS)
    obs["history"] = _coerce_object(payload.get("history") if isinstance(payload, dict) else None, HISTORY_FIELDS) or _null_object(HISTORY_FIELDS)
    return obs


def canonicalize_constraints(raw: Any) -> dict[str, Any]:
    values = _null_object(CONSTRAINT_FIELDS)
    if isinstance(raw, dict):
        for field in CONSTRAINT_FIELDS:
            values[field] = normalize_missing(raw.get(field))
    return values


def canonicalize_event_flags(raw: Any) -> dict[str, Any]:
    values = _null_object(EVENT_FLAG_FIELDS)
    if isinstance(raw, dict):
        for field in EVENT_FLAG_FIELDS:
            values[field] = normalize_missing(raw.get(field))
    return values


def canonicalize_label(raw_label: Any) -> dict[str, Any]:
    label = build_empty_label()
    if not isinstance(raw_label, dict):
        return label
    label["semantic_state"] = normalize_missing(raw_label.get("semantic_state"))
    label["target_ref"] = normalize_missing(raw_label.get("target_ref"))
    label["role"] = normalize_missing(raw_label.get("role"))
    label["constraints"] = canonicalize_constraints(raw_label.get("constraints"))
    label["profile_hint"] = normalize_missing(raw_label.get("profile_hint"))
    label["event_flags"] = canonicalize_event_flags(raw_label.get("event_flags"))
    label["rationale_short"] = truncate_text(raw_label.get("rationale_short"))
    return label


def canonicalize_evidence(raw: Any) -> dict[str, Any]:
    evidence = build_empty_evidence()
    if not isinstance(raw, dict):
        return evidence
    evidence["rule_ids"] = sanitize_string_list(raw.get("rule_ids"))
    evidence["acmi_refs"] = sanitize_string_list(raw.get("acmi_refs"))
    evidence["frame_refs"] = sanitize_string_list(raw.get("frame_refs"))
    evidence["notes"] = normalize_missing(raw.get("notes"))
    return evidence


def canonicalize_extras(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    extras = build_empty_extras()
    if not isinstance(raw, dict):
        return extras
    raw_object_ids = raw.get("raw_object_ids")
    if isinstance(raw_object_ids, dict):
        extras["raw_object_ids"] = {str(key): str(value) for key, value in raw_object_ids.items() if normalize_missing(value) is not None}
    extras["raw_tags"] = sanitize_string_list(raw.get("raw_tags"))
    extras["weapon_event_refs"] = sanitize_string_list(raw.get("weapon_event_refs"))
    if extras["raw_object_ids"] == {}:
        extras["raw_object_ids"] = None
    if not extras["raw_tags"]:
        extras["raw_tags"] = None
    if not extras["weapon_event_refs"]:
        extras["weapon_event_refs"] = None
    if extras == build_empty_extras():
        return None
    return extras


def append_note(existing: str | None, *parts: Any) -> str | None:
    chunks = [existing] if existing else []
    for part in parts:
        part = normalize_missing(part)
        if part in (None, "", [], {}):
            continue
        chunks.append(str(part))
    if not chunks:
        return None
    return " | ".join(chunks)


def flatten_tags(payload: Any, prefix: str = "") -> list[str]:
    tags: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}{key}"
            if isinstance(value, (dict, list)):
                tags.extend(flatten_tags(value, f"{next_prefix}."))
            elif value is not None:
                tags.append(f"{next_prefix}={value}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            tags.extend(flatten_tags(value, f"{prefix}{index}."))
    elif payload is not None:
        tags.append(f"{prefix.rstrip('.')}={payload}")
    return tags


def parse_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def truncate_text(value: Any, max_length: int = 120) -> str | None:
    value = normalize_missing(value)
    if value is None:
        return None
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def derive_quality_tier(source_type: str, confidence: float | None, conflict_flag: bool, needs_review: bool) -> str:
    if source_type == "manual":
        tier = "gold" if not conflict_flag else "silver"
    elif source_type == "acmi_ai":
        if confidence is None:
            tier = "bronze"
        elif confidence < 0.5:
            tier = "weak"
        elif confidence <= 0.8:
            tier = "bronze"
        else:
            tier = "silver"
    else:
        tier = "bronze"

    if conflict_flag and tier == "gold":
        tier = "silver"
    if conflict_flag and needs_review and tier == "weak":
        tier = "bronze"
    return tier


def compute_target_switch(prev_target_ref: Any, target_ref: Any) -> bool | None:
    prev_target_ref = normalize_missing(prev_target_ref)
    target_ref = normalize_missing(target_ref)
    if prev_target_ref is None or target_ref is None:
        return None
    return prev_target_ref != target_ref


def map_legacy_semantic_state(
    legacy_state: str | None,
    *,
    rule_id: str | None = None,
    pair_type: str | None = None,
) -> str | None:
    if rule_id and rule_id in RULE_ID_TO_CANONICAL:
        return RULE_ID_TO_CANONICAL[rule_id]
    if legacy_state in LEGACY_STATE_TO_CANONICAL:
        return LEGACY_STATE_TO_CANONICAL[legacy_state]
    if pair_type == "fire_gate":
        return "commit_intercept"
    if pair_type == "wvr_router":
        return "merge_entry"
    return None


def infer_profile_hint(legacy_profile_hint: str | None, semantic_state: str | None) -> str | None:
    hint = (legacy_profile_hint or "").lower()
    if semantic_state in {"merge_entry", "offensive_turn", "defensive_break"}:
        return "p6dof_aggressive_turn"
    aggressive_tokens = {"jink", "evade", "wvr", "router", "missile_defense"}
    if any(token in hint for token in aggressive_tokens):
        return "p6dof_aggressive_turn"
    if normalize_missing(legacy_profile_hint) is None:
        return None
    return "p6dof_semantic"


def infer_role(semantic_state: str | None, must_support_teammate: Any = None, rule_id: str | None = None) -> str | None:
    if rule_id in {"AIR020", "BRW002", "OAB002"}:
        return "support"
    if semantic_state == "support":
        return "support"
    if semantic_state == "extend":
        return "egressing"
    if semantic_state in {"commit_intercept", "press_attack", "merge_entry", "offensive_turn", "defensive_break"}:
        if must_support_teammate:
            return "support"
        return "engaged"
    if semantic_state in {"hold_geometry", "energy_manage"}:
        if must_support_teammate:
            return "support"
        return "neutral"
    return None


def infer_target_ref(
    *,
    rule_id: str | None = None,
    semantic_state: str | None = None,
    target_policy: str | None = None,
    must_support_teammate: Any = None,
) -> str | None:
    if rule_id and rule_id in RULE_ID_TO_TARGET_REF:
        return RULE_ID_TO_TARGET_REF[rule_id]
    policy = (target_policy or "").lower()
    if semantic_state == "support" and ("ally" in policy or "escort" in policy or must_support_teammate):
        return "ally_1"
    if semantic_state in {"commit_intercept", "press_attack", "merge_entry", "defensive_break", "hold_geometry", "support"}:
        return "enemy_1"
    return None


def infer_should_abort_if_threatened(semantic_state: str | None, rule_scope: str | None = None) -> bool | None:
    if semantic_state in {"extend", "defensive_break"}:
        return True
    if semantic_state in {"commit_intercept", "press_attack", "merge_entry", "offensive_turn"}:
        return False
    if semantic_state == "support":
        return True
    if rule_scope == "disengage":
        return True
    return None


def build_quality(
    *,
    source_type: str,
    confidence: float | None,
    conflict_flag: bool,
    needs_review: bool,
) -> dict[str, Any]:
    quality = build_empty_quality()
    quality["confidence"] = confidence
    quality["conflict_flag"] = conflict_flag
    quality["needs_review"] = bool(needs_review or conflict_flag)
    quality["tier"] = derive_quality_tier(source_type, confidence, quality["conflict_flag"], quality["needs_review"])
    return quality


def make_record(
    *,
    sample_id: str,
    source: dict[str, Any],
    obs: dict[str, Any],
    label: dict[str, Any],
    evidence: dict[str, Any],
    quality: dict[str, Any],
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": "semantic_record_v1",
        "sample_id": sample_id,
        "source": source,
        "obs": obs,
        "label": label,
        "evidence": evidence,
        "quality": quality,
    }
    if extras is not None:
        record["extras"] = extras
    return record


def build_source(
    *,
    platform: str,
    source_type: str,
    scenario_id: str | None = None,
    episode_id: str | None = None,
    timestamp_sec: float | None = None,
    annotator: str | None = None,
    tool: str | None = None,
) -> dict[str, Any]:
    return {
        "platform": platform,
        "type": source_type,
        "scenario_id": scenario_id,
        "episode_id": episode_id,
        "timestamp_sec": timestamp_sec,
        "annotator": annotator,
        "tool": tool,
    }


def strip_nulls(value: Any, *, keep_empty_arrays: bool = True, keep_empty_history: bool = False) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = strip_nulls(item, keep_empty_arrays=keep_empty_arrays, keep_empty_history=keep_empty_history)
            if cleaned is None:
                continue
            if cleaned == {} and not (keep_empty_history and key == "history"):
                continue
            if cleaned == [] and not keep_empty_arrays:
                continue
            result[key] = cleaned
        return result
    if isinstance(value, list):
        result = [strip_nulls(item, keep_empty_arrays=keep_empty_arrays, keep_empty_history=keep_empty_history) for item in value]
        result = [item for item in result if item is not None]
        return result
    if value is None:
        return None
    return value


def _collect_string_sentinels(payload: Any, path: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}" if path else key
            issues.extend(_collect_string_sentinels(value, child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            issues.extend(_collect_string_sentinels(value, f"{path}[{index}]"))
    elif isinstance(payload, str) and payload.strip().lower() in MISSING_SENTINELS:
        issues.append(f"{path}: uses sentinel string {payload!r} instead of null")
    return issues


def validate_record(record: dict[str, Any], index: int | None = None) -> list[str]:
    prefix = f"record[{index}]" if index is not None else "record"
    errors: list[str] = []
    for error in SCHEMA_VALIDATOR.iter_errors(record):
        path = ".".join(str(part) for part in error.path)
        location = f"{prefix}.{path}" if path else prefix
        errors.append(f"{location}: {error.message}")

    for key in LEGACY_TOP_LEVEL_FIELDS:
        if key in record:
            errors.append(f"{prefix}: legacy top-level field retained: {key}")

    errors.extend(f"{prefix}.{issue}" for issue in _collect_string_sentinels(record))

    if record.get("schema_version") != "semantic_record_v1":
        errors.append(f"{prefix}: schema_version must equal semantic_record_v1")

    source = record.get("source") or {}
    evidence = record.get("evidence") or {}
    label = record.get("label") or {}
    if source.get("type") in {"afsim_rule", "acmi_ai", "mixed"}:
        has_trace = bool(normalize_missing(evidence.get("notes"))) or bool(normalize_missing(label.get("rationale_short")))
        if not has_trace:
            errors.append(f"{prefix}: claude_alignment_missing; adapted record needs evidence.notes or label.rationale_short")

    return errors


def validate_records(records: Iterable[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, record in enumerate(records):
        errors.extend(validate_record(record, index))
    return errors


def validate_standard_sft_samples(
    canonical_records: Iterable[dict[str, Any]],
    mapping_rows: Iterable[dict[str, Any]],
) -> list[str]:
    by_sample_id = {record["sample_id"]: record for record in canonical_records}
    errors: list[str] = []
    for row in mapping_rows:
        sample_id = row.get("sample_id")
        record = by_sample_id.get(sample_id)
        if record is None:
            errors.append(f"standard_sft_mapping: sample_id {sample_id!r} missing from canonical records")
            continue
        if record.get("quality", {}).get("conflict_flag") is True:
            errors.append(f"standard_sft_mapping: conflict sample leaked into standard export: {sample_id}")
    return errors


def basename_without_suffix(path_str: str | None) -> str | None:
    if not path_str:
        return None
    return Path(path_str).name


def sanitize_sample_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
