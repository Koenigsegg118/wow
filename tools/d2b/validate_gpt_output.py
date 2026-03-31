"""Validate GPT format_C output against schema rules V01--V11.

Applies the validation rules defined in ``acmi_auto_label_io_examples.json``
(``format_D_validation_rules``).  Each GPT response (format_C) is checked for
enum membership, type correctness, range limits, and cross-field consistency.

Rule summary
------------
=====  ===================================  ==================
Rule   Check                                On-fail action
=====  ===================================  ==================
V01    semantic_state in enum or None        reject
V02    target_ref in enum or None            reject
V03    role in enum or None                  reject
V04    profile_hint in enum or None          reject
V05    constraint fields are bool or None    reject
V06    rationale_short <= 120 chars          truncate_and_warn
V07    confidence in [0, 1]                  clamp_and_warn
V08    top-level keys are {label, qh}        strip_extra
V09    target_switch vs prev_target_ref      flag_conflict
V10    state=None => confidence < 0.3        warn
V11    conflict_flag => needs_review         auto_fix
=====  ===================================  ==================

Usage::

    from tools.d2b.validate_gpt_output import validate_format_c, validate_batch

    result = validate_format_c(gpt_output, obs=obs_dict)
    if result.is_valid:
        clean = result.fixed
    else:
        print(result.errors)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.d2b.d2b_config import (
    ALLOWED_PROFILE_HINTS,
    ALLOWED_ROLES,
    ALLOWED_SEMANTIC_STATES,
    ALLOWED_TARGET_REFS,
)

logger = logging.getLogger(__name__)

# Constraint fields expected inside label.constraints
_CONSTRAINT_FIELDS = (
    "prefer_energy_preserve",
    "allow_aggressive_turn",
    "must_support_teammate",
    "should_abort_if_threatened",
)

# Only these two keys are allowed at the top level of format_C output
_ALLOWED_TOP_KEYS = {"label", "quality_hint"}

# Maximum length for rationale_short (V06)
_MAX_RATIONALE_LEN = 120


# ═══════════════════════════════════════════════════════════════════════
# Data class
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    """Result of validating a single GPT format_C output.

    Attributes:
        is_valid:  True if no reject-level errors were found.
        errors:    List of reject-level error messages (rule ID + detail).
        warnings:  List of warning-level messages.
        auto_fixes: Dict of fields that were auto-fixed.
                    Each entry: ``{field: {"old": ..., "new": ..., "rule": ...}}``.
        original:  The original GPT output dict (unmodified).
        fixed:     The fixed output dict with all auto-fixes applied,
                   or ``None`` if the record was rejected.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    auto_fixes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    original: Dict[str, Any] = field(default_factory=dict)
    fixed: Optional[Dict[str, Any]] = None


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_get(d: Optional[dict], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ═══════════════════════════════════════════════════════════════════════
# Core validation
# ═══════════════════════════════════════════════════════════════════════

def validate_format_c(
    gpt_output: dict,
    obs: dict | None = None,
) -> ValidationResult:
    """Validate a single GPT format_C output against rules V01--V11.

    Parameters
    ----------
    gpt_output:
        The raw GPT response dict.  Expected shape::

            {"label": {...}, "quality_hint": {...}}

    obs:
        Optional observation dict.  When provided, V09 (target-switch
        consistency) can cross-reference ``obs.history.prev_target_ref``.

    Returns
    -------
    ValidationResult
        Contains validation outcome, errors, warnings, auto-fixes,
        the original output, and the fixed output (or None if rejected).
    """
    result = ValidationResult(original=gpt_output)
    doc = copy.deepcopy(gpt_output)

    label = doc.get("label") or {}
    qh = doc.get("quality_hint") or {}

    # ------------------------------------------------------------------
    # V08 -- top-level keys must only be "label" and "quality_hint"
    # ------------------------------------------------------------------
    extra_keys = set(doc.keys()) - _ALLOWED_TOP_KEYS
    if extra_keys:
        for k in extra_keys:
            result.auto_fixes[f"top_level.{k}"] = {
                "old": doc[k],
                "new": "<removed>",
                "rule": "V08",
            }
            del doc[k]
        result.warnings.append(
            f"V08: stripped extra top-level keys: {sorted(extra_keys)}"
        )

    # ------------------------------------------------------------------
    # V01 -- semantic_state must be in allowed set or None
    # ------------------------------------------------------------------
    ss = label.get("semantic_state")
    if ss is not None and ss not in ALLOWED_SEMANTIC_STATES:
        result.errors.append(
            f"V01: invalid semantic_state={ss!r}; "
            f"allowed={sorted(ALLOWED_SEMANTIC_STATES)}"
        )

    # ------------------------------------------------------------------
    # V02 -- target_ref must be in allowed set or None
    # ------------------------------------------------------------------
    tr = label.get("target_ref")
    if tr is not None and tr not in ALLOWED_TARGET_REFS:
        result.errors.append(
            f"V02: invalid target_ref={tr!r}; "
            f"allowed={sorted(ALLOWED_TARGET_REFS)}"
        )

    # ------------------------------------------------------------------
    # V03 -- role must be in allowed set or None
    # ------------------------------------------------------------------
    role = label.get("role")
    if role is not None and role not in ALLOWED_ROLES:
        result.errors.append(
            f"V03: invalid role={role!r}; "
            f"allowed={sorted(ALLOWED_ROLES)}"
        )

    # ------------------------------------------------------------------
    # V04 -- profile_hint must be in allowed set or None
    # ------------------------------------------------------------------
    ph = label.get("profile_hint")
    if ph is not None and ph not in ALLOWED_PROFILE_HINTS:
        result.errors.append(
            f"V04: invalid profile_hint={ph!r}; "
            f"allowed={sorted(ALLOWED_PROFILE_HINTS)}"
        )

    # ------------------------------------------------------------------
    # V05 -- constraint fields must be bool or None
    # ------------------------------------------------------------------
    constraints = label.get("constraints") or {}
    for cf in _CONSTRAINT_FIELDS:
        val = constraints.get(cf)
        if val is not None and not isinstance(val, bool):
            result.errors.append(
                f"V05: constraints.{cf} must be bool or None, "
                f"got {type(val).__name__}={val!r}"
            )

    # ------------------------------------------------------------------
    # V06 -- rationale_short <= 120 chars (truncate if over)
    # ------------------------------------------------------------------
    rationale = label.get("rationale_short")
    if rationale is not None and len(rationale) > _MAX_RATIONALE_LEN:
        old_rationale = rationale
        truncated = rationale[:_MAX_RATIONALE_LEN]
        label["rationale_short"] = truncated
        result.auto_fixes["label.rationale_short"] = {
            "old": old_rationale,
            "new": truncated,
            "rule": "V06",
        }
        result.warnings.append(
            f"V06: rationale_short truncated from {len(old_rationale)} "
            f"to {_MAX_RATIONALE_LEN} chars"
        )

    # ------------------------------------------------------------------
    # V07 -- confidence in [0.0, 1.0] (clamp if out of range)
    # ------------------------------------------------------------------
    conf = qh.get("confidence")
    if conf is not None:
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            result.errors.append(
                f"V07: confidence is not numeric: {conf!r}"
            )
            conf_f = None

        if conf_f is not None and (conf_f < 0.0 or conf_f > 1.0):
            old_conf = conf_f
            clamped = max(0.0, min(1.0, conf_f))
            qh["confidence"] = clamped
            result.auto_fixes["quality_hint.confidence"] = {
                "old": old_conf,
                "new": clamped,
                "rule": "V07",
            }
            result.warnings.append(
                f"V07: confidence clamped from {old_conf} to {clamped}"
            )

    # ------------------------------------------------------------------
    # V09 -- target_switch consistency with prev_target_ref
    # ------------------------------------------------------------------
    target_switch = _safe_get(label, "event_flags", "target_switch")
    if target_switch is True and obs is not None:
        prev_ref = _safe_get(obs, "history", "prev_target_ref")
        cur_ref = label.get("target_ref")
        # If both refs are known and they are the same, target_switch=True
        # is inconsistent (no actual switch happened).
        if prev_ref is not None and cur_ref is not None and prev_ref == cur_ref:
            conflict_note = (
                f"V09: target_switch=True but target_ref={cur_ref!r} == "
                f"prev_target_ref={prev_ref!r} (no actual switch)"
            )
            result.warnings.append(conflict_note)
            # Flag conflict in quality_hint
            qh["conflict_flag"] = True
            result.auto_fixes["quality_hint.conflict_flag"] = {
                "old": qh.get("conflict_flag", False),
                "new": True,
                "rule": "V09",
            }

    # ------------------------------------------------------------------
    # V10 -- if semantic_state is None, confidence should be < 0.3
    # ------------------------------------------------------------------
    if ss is None:
        conf_val = qh.get("confidence")
        if conf_val is not None:
            try:
                conf_num = float(conf_val)
            except (TypeError, ValueError):
                conf_num = None
            if conf_num is not None and conf_num >= 0.3:
                result.warnings.append(
                    f"V10: semantic_state=None but confidence={conf_num:.2f} "
                    f"(expected < 0.3)"
                )

    # ------------------------------------------------------------------
    # V11 -- conflict_flag=True => needs_review must be True
    # ------------------------------------------------------------------
    if qh.get("conflict_flag") is True and qh.get("needs_review") is not True:
        old_nr = qh.get("needs_review")
        qh["needs_review"] = True
        result.auto_fixes["quality_hint.needs_review"] = {
            "old": old_nr,
            "new": True,
            "rule": "V11",
        }
        result.warnings.append(
            "V11: auto-fixed needs_review to True (conflict_flag=True)"
        )

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------
    # Re-assemble doc with possibly mutated label / quality_hint
    doc["label"] = label
    doc["quality_hint"] = qh

    if result.errors:
        result.is_valid = False
        result.fixed = None
    else:
        result.is_valid = True
        result.fixed = doc

    return result


# ═══════════════════════════════════════════════════════════════════════
# Batch validation
# ═══════════════════════════════════════════════════════════════════════

def _load_jsonl(path: Path) -> List[dict]:
    """Load a JSON-lines file, skipping blank lines."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def validate_batch(
    gpt_responses_path: Path,
    obs_samples_path: Path | None = None,
    output_valid_path: Path | None = None,
    output_rejected_path: Path | None = None,
) -> Tuple[Path, dict]:
    """Batch-validate GPT format_C responses.

    Each line in *gpt_responses_path* is a JSON object (format_C) with an
    optional ``request_id`` field for traceability.

    If *obs_samples_path* is provided, each obs sample is paired with the
    GPT response by ``request_id`` match first, then by line index as
    fallback.  This enables V09 (target-switch consistency) checking.

    Parameters
    ----------
    gpt_responses_path:
        Path to JSONL file of GPT format_C outputs.
    obs_samples_path:
        Optional path to JSONL file of obs samples.  Each line should
        have at least ``"obs"`` and optionally ``"request_id"``.
    output_valid_path:
        Path to write validated (fixed) records.  Defaults to
        ``<input_stem>_valid.jsonl`` in the same directory.
    output_rejected_path:
        Path to write rejected records.  Defaults to
        ``<input_stem>_rejected.jsonl`` in the same directory.

    Returns
    -------
    (valid_path, stats_dict)
        *stats_dict* has keys: total, valid, rejected, warnings,
        auto_fixes, errors_by_rule.
    """
    gpt_responses_path = Path(gpt_responses_path)
    if output_valid_path is None:
        output_valid_path = gpt_responses_path.with_name(
            f"{gpt_responses_path.stem}_valid.jsonl"
        )
    if output_rejected_path is None:
        output_rejected_path = gpt_responses_path.with_name(
            f"{gpt_responses_path.stem}_rejected.jsonl"
        )

    output_valid_path = Path(output_valid_path)
    output_rejected_path = Path(output_rejected_path)
    output_valid_path.parent.mkdir(parents=True, exist_ok=True)
    output_rejected_path.parent.mkdir(parents=True, exist_ok=True)

    # Load GPT responses
    gpt_records = _load_jsonl(gpt_responses_path)

    # Load obs samples (if provided) and build lookup
    obs_by_id: Dict[str, dict] = {}
    obs_by_index: List[dict] = []
    if obs_samples_path is not None:
        obs_samples_path = Path(obs_samples_path)
        obs_records = _load_jsonl(obs_samples_path)
        obs_by_index = obs_records
        for rec in obs_records:
            rid = rec.get("request_id")
            if rid is not None:
                obs_by_id[str(rid)] = rec

    # Stats
    stats: Dict[str, Any] = {
        "total": len(gpt_records),
        "valid": 0,
        "rejected": 0,
        "warnings": 0,
        "auto_fixes": 0,
        "errors_by_rule": {},
    }

    with open(output_valid_path, "w", encoding="utf-8") as f_valid, \
         open(output_rejected_path, "w", encoding="utf-8") as f_reject:

        for idx, gpt_rec in enumerate(gpt_records):
            # Resolve corresponding obs sample
            obs_dict: dict | None = None
            rid = gpt_rec.get("request_id")
            if rid is not None and str(rid) in obs_by_id:
                obs_sample = obs_by_id[str(rid)]
                obs_dict = obs_sample.get("obs", obs_sample)
            elif idx < len(obs_by_index):
                obs_sample = obs_by_index[idx]
                obs_dict = obs_sample.get("obs", obs_sample)

            # Strip request_id (and any other meta) before validation;
            # we validate only the format_C payload.
            gpt_payload = {
                k: v for k, v in gpt_rec.items()
                if k not in ("request_id",)
            }

            vr = validate_format_c(gpt_payload, obs=obs_dict)

            # Count warnings and auto-fixes
            stats["warnings"] += len(vr.warnings)
            stats["auto_fixes"] += len(vr.auto_fixes)

            # Count errors by rule
            for err_msg in vr.errors:
                rule_id = err_msg.split(":")[0].strip()
                stats["errors_by_rule"][rule_id] = (
                    stats["errors_by_rule"].get(rule_id, 0) + 1
                )

            if vr.is_valid:
                stats["valid"] += 1
                out_rec = vr.fixed
                # Re-attach request_id for traceability
                if rid is not None:
                    out_rec["request_id"] = rid
                f_valid.write(
                    json.dumps(out_rec, ensure_ascii=False) + "\n"
                )
            else:
                stats["rejected"] += 1
                reject_rec = {
                    "original": vr.original,
                    "errors": vr.errors,
                    "warnings": vr.warnings,
                }
                if rid is not None:
                    reject_rec["request_id"] = rid
                f_reject.write(
                    json.dumps(reject_rec, ensure_ascii=False) + "\n"
                )

            # Log individual issues at DEBUG level
            if vr.errors:
                logger.debug(
                    "REJECTED [%s]: %s",
                    rid or f"idx={idx}",
                    "; ".join(vr.errors),
                )
            if vr.warnings:
                logger.debug(
                    "WARNINGS [%s]: %s",
                    rid or f"idx={idx}",
                    "; ".join(vr.warnings),
                )

    # Summary log
    logger.info(
        "Validation complete: %d total, %d valid, %d rejected, "
        "%d warnings, %d auto-fixes",
        stats["total"],
        stats["valid"],
        stats["rejected"],
        stats["warnings"],
        stats["auto_fixes"],
    )
    if stats["errors_by_rule"]:
        logger.info("Errors by rule: %s", stats["errors_by_rule"])

    return output_valid_path, stats


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate GPT format_C output against schema rules V01--V11.  "
            "Valid records (with auto-fixes applied) are written to the "
            "valid output file; rejected records to the rejected output file."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to GPT responses JSONL file (format_C, one per line).",
    )
    parser.add_argument(
        "--obs",
        type=Path,
        default=None,
        help=(
            "Path to obs samples JSONL file.  When provided, enables V09 "
            "(target-switch consistency) checking by pairing each GPT "
            "response with its obs sample via request_id or line index."
        ),
    )
    parser.add_argument(
        "-o", "--output-valid",
        type=Path,
        default=None,
        help=(
            "Path for valid output JSONL.  "
            "Defaults to <input_stem>_valid.jsonl."
        ),
    )
    parser.add_argument(
        "--output-rejected",
        type=Path,
        default=None,
        help=(
            "Path for rejected output JSONL.  "
            "Defaults to <input_stem>_rejected.jsonl."
        ),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Path to write stats JSON.  Defaults to <valid_stem>_stats.json.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise SystemExit(1)

    valid_path, stats = validate_batch(
        gpt_responses_path=input_path,
        obs_samples_path=args.obs,
        output_valid_path=args.output_valid,
        output_rejected_path=args.output_rejected,
    )

    # Write stats
    stats_path: Path = args.stats or valid_path.with_name(
        f"{valid_path.stem}_stats.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Valid output:    %s", valid_path)
    logger.info("Stats:           %s", stats_path)
    logger.info("Summary:         %s", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
