"""Detect hard cases and package them as GPT-5.4 annotation requests.

Hard cases are obs samples that cannot be reliably auto-labeled by the
conservative rule labeler.  This module identifies 10 categories of
ambiguity, builds per-sample evidence hints in Chinese, and outputs a
JSONL batch ready for the GPT caller.

Usage:
    # Programmatic
    reasons = detect_hard_cases(obs, rule_result)
    if reasons:
        req = build_hardcase_request(obs, reasons, meta={"scenario_id": "s1"})

    # Batch CLI
    python -m tools.d2b.build_hardcase_requests \\
        --obs obs_samples.jsonl \\
        --rule-labeled rule_labelled.jsonl \\
        -o hardcase_requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.d2b.d2b_config import (
    ALLOWED_SEMANTIC_STATES,
    ENERGY_FAR_RANGE_M,
    ENERGY_LOW_CLOSURE_MPS,
    LOW_CONFIDENCE_THRESHOLD,
    MERGE_RANGE_M,
    OFFENSIVE_RANGE_M,
    PRESS_CLOSURE_THRESHOLD_MPS,
    PRESS_RANGE_MAX_M,
    PRESS_RANGE_MIN_M,
    SUPPORT_ALLY_RANGE_M,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Data classes
# =========================================================================

@dataclass
class HardcaseReason:
    """Single reason why a sample qualifies as a hard case.

    Attributes:
        reason_code: Machine-readable tag (e.g. ``"target_switch"``).
        details: Human-readable explanation (Chinese, for evidence_hints).
    """

    reason_code: str
    details: str


# =========================================================================
# Internal helpers
# =========================================================================

def _safe_get(d: Optional[dict], *keys: str, default=None):
    """Safely traverse nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _enemy_present(obs: dict) -> bool:
    """Return True if obs contains a valid enemy with range_m."""
    enemy = obs.get("enemy")
    if enemy is None:
        return False
    return enemy.get("range_m") is not None


def _ally_present(obs: dict) -> bool:
    """Return True if obs contains a valid ally with range_m."""
    ally = obs.get("ally")
    if ally is None:
        return False
    return ally.get("range_m") is not None


# =========================================================================
# Active engagement states (for missing_critical detection)
# =========================================================================

_ACTIVE_ENGAGEMENT_STATES = frozenset({
    "commit_intercept",
    "press_attack",
    "merge_entry",
    "offensive_turn",
    "defensive_break",
    "support",
})


# =========================================================================
# Core detection logic
# =========================================================================

def detect_hard_cases(
    obs: dict,
    rule_result=None,
) -> List[HardcaseReason]:
    """Detect hard-case features from a single obs sample.

    Parameters
    ----------
    obs:
        Observation dict in format_A style (ego/enemy/ally/engagement/history).
    rule_result:
        Optional result from ``rule_labeler_v1.classify_easy_case``.
        If provided, its ``semantic_state`` and ``confidence`` are inspected.
        Can be a ``RuleResult`` dataclass or a plain dict with those keys.

    Returns
    -------
    list[HardcaseReason]
        Empty list means the sample is NOT a hard case.
    """
    reasons: List[HardcaseReason] = []

    # -- extract rule_result fields --
    rule_state: Optional[str] = None
    rule_confidence: float = 1.0
    if rule_result is not None:
        if hasattr(rule_result, "semantic_state"):
            rule_state = rule_result.semantic_state
            rule_confidence = getattr(rule_result, "confidence", 1.0)
        elif isinstance(rule_result, dict):
            rule_state = rule_result.get("semantic_state")
            rule_confidence = rule_result.get("confidence", 1.0)

    # -- extract obs fields --
    has_enemy = _enemy_present(obs)
    has_ally = _ally_present(obs)

    enemy = obs.get("enemy") or {}
    ally = obs.get("ally") or {}
    engagement = obs.get("engagement") or {}
    history = obs.get("history") or {}

    range_m: Optional[float] = enemy.get("range_m")
    bearing_deg: Optional[float] = enemy.get("bearing_deg")
    aspect_deg: Optional[float] = enemy.get("aspect_deg")
    closure_mps: Optional[float] = enemy.get("closure_mps")
    target_ref: Optional[str] = enemy.get("target_ref")

    ally_range_m: Optional[float] = ally.get("range_m")

    is_merge: bool = bool(engagement.get("is_merge", False))
    is_defensive: bool = bool(engagement.get("is_defensive", False))

    prev_target_ref: Optional[str] = history.get("prev_target_ref")
    prev_semantic_state: Optional[str] = history.get("prev_semantic_state")

    # ----------------------------------------------------------------
    # 1. unclassified: rule labeler returned None (no easy-case match)
    # ----------------------------------------------------------------
    if rule_result is not None and rule_state is None:
        reasons.append(HardcaseReason(
            reason_code="unclassified",
            details="规则标注器未匹配任何 easy-case 规则, 返回 None",
        ))

    # ----------------------------------------------------------------
    # 2. low_confidence: rule confidence < threshold
    # ----------------------------------------------------------------
    if rule_result is not None and rule_confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append(HardcaseReason(
            reason_code="low_confidence",
            details=(
                f"规则置信度={rule_confidence:.2f} "
                f"低于阈值 {LOW_CONFIDENCE_THRESHOLD}"
            ),
        ))

    # ----------------------------------------------------------------
    # 3. target_switch: prev target differs from current
    # ----------------------------------------------------------------
    if (
        prev_target_ref is not None
        and target_ref is not None
        and prev_target_ref != target_ref
    ):
        reasons.append(HardcaseReason(
            reason_code="target_switch",
            details=(
                f"prev_target_ref={prev_target_ref} "
                f"但当前最近敌机为 {target_ref}"
            ),
        ))

    # ----------------------------------------------------------------
    # 4. support_candidate: ally nearby AND enemy exists
    # ----------------------------------------------------------------
    if (
        has_ally
        and ally_range_m is not None
        and ally_range_m < SUPPORT_ALLY_RANGE_M
        and has_enemy
    ):
        reasons.append(HardcaseReason(
            reason_code="support_candidate",
            details=(
                f"友机距离={ally_range_m:.0f}m "
                f"< {SUPPORT_ALLY_RANGE_M:.0f}m, "
                f"且敌机存在, 可能为 support"
            ),
        ))

    # ----------------------------------------------------------------
    # 5. defensive_candidate: is_defensive flag
    # ----------------------------------------------------------------
    if is_defensive:
        aspect_str = f"{aspect_deg:.0f}" if aspect_deg is not None else "N/A"
        reasons.append(HardcaseReason(
            reason_code="defensive_candidate",
            details=(
                f"is_defensive=True, aspect={aspect_str}\u00b0, "
                f"可能为 defensive_break"
            ),
        ))

    # ----------------------------------------------------------------
    # 6. offensive_candidate: close enemy, not merge
    # ----------------------------------------------------------------
    if (
        has_enemy
        and range_m is not None
        and range_m < OFFENSIVE_RANGE_M
        and not is_merge
    ):
        reasons.append(HardcaseReason(
            reason_code="offensive_candidate",
            details=(
                f"range={range_m:.0f}m < {OFFENSIVE_RANGE_M:.0f}m, "
                f"非 merge, 可能为 offensive_turn"
            ),
        ))

    # ----------------------------------------------------------------
    # 7. energy_candidate: no enemy OR far enemy with low closure
    # ----------------------------------------------------------------
    if not has_enemy:
        reasons.append(HardcaseReason(
            reason_code="energy_candidate",
            details="无敌方目标, 可能为 energy_manage",
        ))
    elif (
        range_m is not None
        and closure_mps is not None
        and range_m > ENERGY_FAR_RANGE_M
        and abs(closure_mps) < ENERGY_LOW_CLOSURE_MPS
    ):
        reasons.append(HardcaseReason(
            reason_code="energy_candidate",
            details=(
                f"range={range_m:.0f}m > {ENERGY_FAR_RANGE_M:.0f}m, "
                f"|closure|={abs(closure_mps):.1f}m/s "
                f"< {ENERGY_LOW_CLOSURE_MPS:.0f}m/s, "
                f"可能为 energy_manage"
            ),
        ))

    # ----------------------------------------------------------------
    # 8. missing_critical: enemy gone but prev state was active
    # ----------------------------------------------------------------
    if (
        not has_enemy
        and prev_semantic_state is not None
        and prev_semantic_state in _ACTIVE_ENGAGEMENT_STATES
    ):
        reasons.append(HardcaseReason(
            reason_code="missing_critical",
            details=(
                f"敌机丢失, 但前一状态为 {prev_semantic_state} "
                f"(活跃交战), 状态转换不明确"
            ),
        ))

    # ----------------------------------------------------------------
    # 9. engagement_mismatch: contradictory engagement flags vs rule
    # ----------------------------------------------------------------
    if is_merge and rule_state == "commit_intercept":
        reasons.append(HardcaseReason(
            reason_code="engagement_mismatch",
            details=(
                f"is_merge=True 但规则判定为 {rule_state}, "
                f"已进入 merge 范围({MERGE_RANGE_M:.0f}m 内)"
            ),
        ))
    if is_defensive and rule_state == "hold_geometry":
        reasons.append(HardcaseReason(
            reason_code="engagement_mismatch",
            details=(
                f"is_defensive=True 但规则判定为 {rule_state}, "
                f"可能应为 defensive_break"
            ),
        ))

    # ----------------------------------------------------------------
    # 10. press_candidate: between commit and merge boundary
    # ----------------------------------------------------------------
    if (
        has_enemy
        and range_m is not None
        and closure_mps is not None
        and PRESS_RANGE_MIN_M < range_m < PRESS_RANGE_MAX_M
        and closure_mps < PRESS_CLOSURE_THRESHOLD_MPS
    ):
        reasons.append(HardcaseReason(
            reason_code="press_candidate",
            details=(
                f"range={range_m:.0f}m, "
                f"closure={closure_mps:.0f}m/s, "
                f"介于 commit 和 merge 边界"
                f"({PRESS_RANGE_MIN_M:.0f}~{PRESS_RANGE_MAX_M:.0f}m, "
                f"closure<{PRESS_CLOSURE_THRESHOLD_MPS:.0f}m/s), "
                f"可能为 press_attack"
            ),
        ))

    return reasons


# =========================================================================
# Candidate label inference
# =========================================================================

# Maps reason_code -> most likely candidate semantic_states.
_CANDIDATE_MAP: Dict[str, List[str]] = {
    "unclassified": sorted(ALLOWED_SEMANTIC_STATES),
    "low_confidence": sorted(ALLOWED_SEMANTIC_STATES),
    "target_switch": [
        "commit_intercept", "press_attack", "support", "extend",
    ],
    "support_candidate": [
        "support", "commit_intercept", "hold_geometry",
    ],
    "defensive_candidate": [
        "defensive_break", "extend", "energy_manage",
    ],
    "offensive_candidate": [
        "offensive_turn", "press_attack", "merge_entry",
    ],
    "energy_candidate": [
        "energy_manage", "hold_geometry", "extend",
    ],
    "missing_critical": [
        "energy_manage", "extend", "hold_geometry",
    ],
    "engagement_mismatch": [
        "merge_entry", "defensive_break", "press_attack",
    ],
    "press_candidate": [
        "press_attack", "commit_intercept", "merge_entry",
    ],
}


def _infer_candidate_labels(reasons: List[HardcaseReason]) -> List[str]:
    """Merge candidate labels from all detected reasons, deduplicated."""
    seen: set[str] = set()
    result: List[str] = []
    for r in reasons:
        for label in _CANDIDATE_MAP.get(r.reason_code, []):
            if label not in seen:
                seen.add(label)
                result.append(label)
    return result


# =========================================================================
# Request builder
# =========================================================================

def build_hardcase_request(
    obs: dict,
    reasons: List[HardcaseReason],
    meta: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build a single hard-case request dict for GPT annotation.

    Parameters
    ----------
    obs:
        Observation dict in format_A style
        (ego/enemy/ally/engagement/history).
    reasons:
        Non-empty list of :class:`HardcaseReason` instances.
    meta:
        Optional metadata (scenario_id, episode_id, timestamp_sec, etc.).

    Returns
    -------
    dict
        Request with keys:
        ``request_id``, ``obs``, ``evidence_hints``,
        ``hard_case_reasons``, ``candidate_labels``, ``meta``.
    """
    if meta is None:
        meta = {}

    request_id = _make_request_id(meta)

    evidence_hints = [r.details for r in reasons]
    reason_codes = [r.reason_code for r in reasons]
    candidate_labels = _infer_candidate_labels(reasons)

    return {
        "request_id": request_id,
        "obs": obs,
        "evidence_hints": evidence_hints,
        "hard_case_reasons": reason_codes,
        "candidate_labels": candidate_labels,
        "meta": meta,
    }


def _make_request_id(meta: dict) -> str:
    """Generate a unique request ID from metadata."""
    scenario = meta.get("scenario_id", "unk")
    ts = meta.get("timestamp_sec")
    ts_str = f"{ts:.1f}" if ts is not None else "na"
    short = uuid.uuid4().hex[:8]
    return f"hc_{scenario}_{ts_str}_{short}"


# =========================================================================
# Batch processing
# =========================================================================

def build_hardcase_batch(
    obs_samples_path: Path,
    rule_labeled_path: Path,
    output_path: Path,
) -> Tuple[Path, dict]:
    """Batch-process obs samples and emit hard-case requests.

    Reads all obs samples, cross-references with rule-labeled output to
    find samples that are either absent from the rule-labeled set or have
    issues (hard_case flag, low confidence, None state).

    Parameters
    ----------
    obs_samples_path:
        JSONL file of obs samples.  Each line: ``{"obs": {...}, "meta": {...}}``.
    rule_labeled_path:
        JSONL file produced by ``rule_labeler_v1.label_batch``.  Lines are
        either canonical records (with ``label.semantic_state``) or stubs
        (with ``_hard_case: true``).
    output_path:
        Output JSONL path for hard-case requests (one per line).

    Returns
    -------
    (output_path, stats)
        *stats* contains counts: ``total_obs``, ``rule_labeled``,
        ``hard_cases_detected``, ``reason_breakdown``.
    """
    # -- Step 1: Index rule-labeled results by line number or sample_id --
    rule_index: Dict[int, dict] = {}       # line_no -> record
    rule_by_id: Dict[str, dict] = {}       # sample_id -> record

    if rule_labeled_path.exists():
        with open(rule_labeled_path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                rule_index[line_no] = rec
                sid = rec.get("sample_id") or rec.get("meta", {}).get("sample_id")
                if sid:
                    rule_by_id[sid] = rec

    # -- Step 2: Read obs samples and detect hard cases --
    stats: Dict[str, Any] = {
        "total_obs": 0,
        "rule_labeled": len(rule_index),
        "hard_cases_detected": 0,
        "reason_breakdown": {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(obs_samples_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_no, raw_line in enumerate(fin, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("obs 行 %d: JSON 解析失败: %s", line_no, exc)
                continue

            obs = sample.get("obs", sample)
            meta = sample.get("meta", {})
            meta["_source_line"] = line_no
            stats["total_obs"] += 1

            # Try to find matching rule result
            rule_rec = rule_index.get(line_no)
            rule_result = _extract_rule_result(rule_rec)

            # Detect hard cases
            reasons = detect_hard_cases(obs, rule_result)

            # Also treat explicitly flagged hard-case stubs
            if rule_rec is not None and rule_rec.get("_hard_case"):
                if not any(r.reason_code == "unclassified" for r in reasons):
                    reasons.append(HardcaseReason(
                        reason_code="unclassified",
                        details="规则标注器标记为 _hard_case",
                    ))

            if not reasons:
                continue

            # Build request
            request = build_hardcase_request(obs, reasons, meta)
            fout.write(json.dumps(request, ensure_ascii=False) + "\n")

            stats["hard_cases_detected"] += 1
            for r in reasons:
                stats["reason_breakdown"][r.reason_code] = (
                    stats["reason_breakdown"].get(r.reason_code, 0) + 1
                )

    logger.info(
        "Hard-case 检测完成: %d 总样本, %d hard-case, 原因分布: %s",
        stats["total_obs"],
        stats["hard_cases_detected"],
        json.dumps(stats["reason_breakdown"], ensure_ascii=False),
    )

    return output_path, stats


def _extract_rule_result(rule_rec: Optional[dict]) -> Optional[dict]:
    """Extract a minimal rule_result dict from a rule-labeled record.

    Handles both canonical records and hard-case stubs.
    Returns None if no rule record exists.
    """
    if rule_rec is None:
        return None

    # Hard-case stub from rule_labeler
    if rule_rec.get("_hard_case"):
        return {"semantic_state": None, "confidence": 0.0}

    # Canonical record
    label = rule_rec.get("label", {})
    quality = rule_rec.get("quality", {})
    return {
        "semantic_state": label.get("semantic_state"),
        "confidence": quality.get("confidence", 1.0),
    }


# =========================================================================
# CLI entry point
# =========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect hard-case obs samples and package them as GPT-5.4 "
            "annotation requests."
        ),
    )
    parser.add_argument(
        "--obs",
        type=Path,
        required=True,
        help="Path to obs samples JSONL (one {obs, meta} per line).",
    )
    parser.add_argument(
        "--rule-labeled",
        type=Path,
        required=True,
        help=(
            "Path to rule-labeled JSONL from rule_labeler_v1. "
            "Used to cross-reference easy-case results."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Output path for hard-case requests JSONL. "
            "Defaults to <obs_stem>_hardcase_requests.jsonl."
        ),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Path to write stats JSON. Defaults to <output_stem>_stats.json.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for hard-case request building."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    obs_path: Path = args.obs
    rule_path: Path = args.rule_labeled

    if not obs_path.exists():
        logger.error("obs 文件不存在: %s", obs_path)
        raise SystemExit(1)
    if not rule_path.exists():
        logger.error("rule-labeled 文件不存在: %s", rule_path)
        raise SystemExit(1)

    output_path: Path = args.output or obs_path.with_name(
        f"{obs_path.stem}_hardcase_requests.jsonl"
    )
    stats_path: Path = args.stats or output_path.with_name(
        f"{output_path.stem}_stats.json"
    )

    out, stats = build_hardcase_batch(obs_path, rule_path, output_path)

    # Write stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("输出: %s", out)
    logger.info("统计: %s", stats_path)
    logger.info("统计摘要: %s", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
