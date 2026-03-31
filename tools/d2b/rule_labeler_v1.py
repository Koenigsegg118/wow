"""Conservative rule labeler for the D2b pipeline.

Only auto-labels 5 easy-case outcomes with high precision:

    merge_entry, commit_intercept, extend, hold_geometry, null

All other semantic states (press_attack, support, offensive_turn,
defensive_break, energy_manage) are deliberately left to the hard-case
GPT review path.

Priority order: merge_entry > commit_intercept > extend > hold_geometry > null

Every numeric threshold is imported from ``tools.d2b.d2b_config`` (which
itself re-exports from ``sim.runtime.semantic_thresholds``).  Zero
hardcoded magic numbers appear in the business-logic functions.

Output conforms to ``semantic_record_schema_v1.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# All thresholds come from d2b_config (single source of truth).
# ---------------------------------------------------------------------------
from tools.d2b.d2b_config import (
    COMMIT_MAX_BEARING_DEG,
    COMMIT_MIN_RANGE_M,
    DEF_ASPECT_DEG,
    DEF_RANGE_M,
    EXTEND_MIN_BEARING_ABS_DEG,
    HOLD_MAX_CLOSURE_ABS_MPS,
    MERGE_ENTRY_FAST_CLOSURE_MPS,
    MERGE_ENTRY_MAX_RANGE_M,
    MERGE_RANGE_M,
    RULE_IDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version constant
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "semantic_record_v1"

# ── labels that this module will NEVER emit ─────────────────────────
# These require nuanced judgment and belong in the GPT-review hard-case path.
_HARD_CASE_STATES = frozenset({
    "press_attack",
    "support",
    "offensive_turn",
    "defensive_break",
    "energy_manage",
})


# ═══════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RuleResult:
    """Result of a rule-based classification attempt.

    If *semantic_state* is ``None`` the sample is a hard case and should be
    forwarded to the GPT review stage.
    """

    semantic_state: Optional[str]
    rule_ids: List[str] = field(default_factory=list)
    evidence_notes: str = ""
    confidence: float = 0.0
    target_ref: Optional[str] = None
    role: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    profile_hint: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_get(d: Optional[dict], *keys: str, default=None):
    """Safely traverse nested dicts, returning *default* on any miss."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _is_merge_flag(obs: dict) -> bool:
    """Check engagement.is_merge flag from the obs dict."""
    return bool(_safe_get(obs, "engagement", "is_merge", default=False))


def _is_defensive(obs: dict) -> bool:
    """Check engagement.is_defensive flag from the obs dict."""
    return bool(_safe_get(obs, "engagement", "is_defensive", default=False))


def _enemy_present(obs: dict) -> bool:
    """Return True if an enemy sub-dict with at least range_m exists."""
    enemy = obs.get("enemy")
    if enemy is None:
        return False
    return enemy.get("range_m") is not None


def _get_enemy_fields(obs: dict) -> Tuple[
    Optional[float],  # range_m
    Optional[float],  # bearing_deg
    Optional[float],  # closure_mps
    Optional[str],    # target_ref
]:
    """Extract frequently used enemy fields."""
    enemy = obs.get("enemy") or {}
    return (
        enemy.get("range_m"),
        enemy.get("bearing_deg"),
        enemy.get("closure_mps"),
        enemy.get("target_ref"),
    )


# ═══════════════════════════════════════════════════════════════════════
# Core classification logic
# ═══════════════════════════════════════════════════════════════════════

def classify_easy_case(obs: dict) -> Optional[RuleResult]:
    """Try to classify *obs* into one of the 5 easy-case labels.

    Returns a :class:`RuleResult` when the sample matches an easy rule,
    or ``None`` if it falls into the hard-case bucket (forwarded to GPT).

    Priority order:

    1. **merge_entry** -- is_merge flag OR (range < 5000 AND closure < -100)
    2. **commit_intercept** -- enemy exists, NOT merge, NOT defensive,
       |bearing| < 30, closure < 0, range > 15000
    3. **extend** -- closure > 0, |bearing| > 90, NOT merge
    4. **hold_geometry** -- enemy exists, |closure| < 20,
       NOT merge, NOT defensive (very conservative)
    5. **null** -- no enemy OR insufficient info

    If none of these match the sample is a hard case and ``None`` is returned.
    """

    merge_flag = _is_merge_flag(obs)
    defensive = _is_defensive(obs)
    has_enemy = _enemy_present(obs)

    range_m, bearing_deg, closure_mps, target_ref = _get_enemy_fields(obs)

    # ------------------------------------------------------------------
    # Rule 1: merge_entry
    # ------------------------------------------------------------------
    if merge_flag:
        return RuleResult(
            semantic_state="merge_entry",
            rule_ids=[RULE_IDS["merge_flag"]],
            evidence_notes=(
                f"{RULE_IDS['merge_flag']}: is_merge=True, "
                f"range={range_m}m"
            ),
            confidence=0.95,
            target_ref=target_ref,
            role="engaged",
            constraints={
                "prefer_energy_preserve": False,
                "allow_aggressive_turn": True,
                "must_support_teammate": None,
                "should_abort_if_threatened": None,
            },
            profile_hint="auto",
        )

    # merge via range + fast closure (not flagged but geometrically clear)
    if (
        has_enemy
        and range_m is not None
        and closure_mps is not None
        and range_m < MERGE_ENTRY_MAX_RANGE_M
        and closure_mps < MERGE_ENTRY_FAST_CLOSURE_MPS
    ):
        return RuleResult(
            semantic_state="merge_entry",
            rule_ids=[RULE_IDS["merge_range"]],
            evidence_notes=(
                f"{RULE_IDS['merge_range']}: range={range_m:.1f}m "
                f"< {MERGE_ENTRY_MAX_RANGE_M}m, "
                f"closure={closure_mps:.1f}m/s "
                f"< {MERGE_ENTRY_FAST_CLOSURE_MPS}m/s"
            ),
            confidence=0.90,
            target_ref=target_ref,
            role="engaged",
            constraints={
                "prefer_energy_preserve": False,
                "allow_aggressive_turn": True,
                "must_support_teammate": None,
                "should_abort_if_threatened": None,
            },
            profile_hint="auto",
        )

    # ------------------------------------------------------------------
    # Rule 2: commit_intercept
    # ------------------------------------------------------------------
    if (
        has_enemy
        and not merge_flag
        and not defensive
        and bearing_deg is not None
        and closure_mps is not None
        and range_m is not None
        and abs(bearing_deg) < COMMIT_MAX_BEARING_DEG
        and closure_mps < 0
        and range_m > COMMIT_MIN_RANGE_M
    ):
        return RuleResult(
            semantic_state="commit_intercept",
            rule_ids=[RULE_IDS["commit"]],
            evidence_notes=(
                f"{RULE_IDS['commit']}: "
                f"enemy在正前方(bearing={bearing_deg:.1f}°), "
                f"正在接近(closure={closure_mps:.1f}), "
                f"range={range_m:.1f}m, 非merge/defensive"
            ),
            confidence=0.90,
            target_ref=target_ref,
            role="engaged",
            constraints={
                "prefer_energy_preserve": None,
                "allow_aggressive_turn": None,
                "must_support_teammate": None,
                "should_abort_if_threatened": True,
            },
            profile_hint="p6dof_semantic",
        )

    # ------------------------------------------------------------------
    # Rule 3: extend
    # ------------------------------------------------------------------
    if (
        has_enemy
        and not merge_flag
        and closure_mps is not None
        and bearing_deg is not None
        and closure_mps > 0
        and abs(bearing_deg) > EXTEND_MIN_BEARING_ABS_DEG
    ):
        return RuleResult(
            semantic_state="extend",
            rule_ids=[RULE_IDS["extend"]],
            evidence_notes=(
                f"{RULE_IDS['extend']}: "
                f"closure={closure_mps:.1f}m/s (正在远离), "
                f"|bearing|={abs(bearing_deg):.1f}° "
                f"> {EXTEND_MIN_BEARING_ABS_DEG}°, 非merge"
            ),
            confidence=0.85,
            target_ref=target_ref,
            role="egressing",
            constraints={
                "prefer_energy_preserve": True,
                "allow_aggressive_turn": False,
                "must_support_teammate": None,
                "should_abort_if_threatened": None,
            },
            profile_hint="p6dof_semantic",
        )

    # ------------------------------------------------------------------
    # Rule 4: hold_geometry  (very conservative)
    # ------------------------------------------------------------------
    if (
        has_enemy
        and not merge_flag
        and not defensive
        and closure_mps is not None
        and abs(closure_mps) < HOLD_MAX_CLOSURE_ABS_MPS
    ):
        return RuleResult(
            semantic_state="hold_geometry",
            rule_ids=[RULE_IDS["hold"]],
            evidence_notes=(
                f"{RULE_IDS['hold']}: "
                f"|closure|={abs(closure_mps):.1f}m/s "
                f"< {HOLD_MAX_CLOSURE_ABS_MPS}m/s, "
                f"非merge/defensive, 几何稳定"
            ),
            confidence=0.85,
            target_ref=target_ref,
            role="neutral",
            constraints={
                "prefer_energy_preserve": None,
                "allow_aggressive_turn": False,
                "must_support_teammate": None,
                "should_abort_if_threatened": None,
            },
            profile_hint="p6dof_semantic",
        )

    # ------------------------------------------------------------------
    # Rule 5: null (no enemy or insufficient info)
    # ------------------------------------------------------------------
    if not has_enemy:
        return RuleResult(
            semantic_state=None,
            rule_ids=[RULE_IDS["no_enemy"]],
            evidence_notes=(
                f"{RULE_IDS['no_enemy']}: "
                "无敌方目标, 无法判断战术意图"
            ),
            confidence=0.90,
            target_ref=None,
            role=None,
            constraints={},
            profile_hint=None,
        )

    # Insufficient info (enemy exists but key fields are missing)
    missing: List[str] = []
    if bearing_deg is None:
        missing.append("bearing_deg")
    if closure_mps is None:
        missing.append("closure_mps")
    if range_m is None:
        missing.append("range_m")
    if missing:
        return RuleResult(
            semantic_state=None,
            rule_ids=[RULE_IDS["insufficient"]],
            evidence_notes=(
                f"{RULE_IDS['insufficient']}: "
                f"敌方存在但缺少关键字段 ({', '.join(missing)}), "
                "无法判定"
            ),
            confidence=0.50,
            target_ref=target_ref,
            role=None,
            constraints={},
            profile_hint=None,
        )

    # ------------------------------------------------------------------
    # Hard case -- none of the conservative rules matched.
    # ------------------------------------------------------------------
    return None


# ═══════════════════════════════════════════════════════════════════════
# Canonical record builder
# ═══════════════════════════════════════════════════════════════════════

def _detect_target_switch(obs: dict, rule_result: RuleResult) -> Optional[bool]:
    """Detect target switch by comparing with history.prev_target_ref."""
    prev_ref = _safe_get(obs, "history", "prev_target_ref")
    cur_ref = rule_result.target_ref
    if prev_ref is None or cur_ref is None:
        return None
    return prev_ref != cur_ref


def build_canonical_from_rule(
    obs: dict,
    rule_result: RuleResult,
    meta: dict,
) -> dict:
    """Build a full canonical record conforming to *semantic_record_schema_v1.json*.

    Parameters
    ----------
    obs:
        The observation dict (ego / enemy / ally / engagement / history).
    rule_result:
        Output from :func:`classify_easy_case`.
    meta:
        Additional metadata dict.  Expected keys:

        - ``scenario_id`` (str | None): ACMI filename or scenario identifier.
        - ``episode_id`` (str | None): Episode / sub-round identifier.
        - ``timestamp_sec`` (float | None): Simulation time in seconds.
        - ``sample_id`` (str | None): If provided, used as-is; otherwise
          auto-generated.

    Returns
    -------
    dict
        A canonical record ready for serialisation to JSON.
    """

    sample_id = meta.get("sample_id") or _make_sample_id(meta)

    # Determine quality tier
    tier = "bronze"  # rule-only, single source = bronze
    needs_review = rule_result.semantic_state is None or rule_result.confidence < 0.80

    # Build constraints dict (ensure all canonical keys are present)
    constraints = {
        "prefer_energy_preserve": rule_result.constraints.get("prefer_energy_preserve"),
        "allow_aggressive_turn": rule_result.constraints.get("allow_aggressive_turn"),
        "must_support_teammate": rule_result.constraints.get("must_support_teammate"),
        "should_abort_if_threatened": rule_result.constraints.get("should_abort_if_threatened"),
    }

    # Short rationale for human review
    rationale = _build_rationale(rule_result)

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "source": {
            "platform": "afsim_acmi",
            "type": "afsim_rule",
            "scenario_id": meta.get("scenario_id"),
            "episode_id": meta.get("episode_id"),
            "timestamp_sec": meta.get("timestamp_sec"),
            "annotator": None,
            "tool": "rule_labeler_v1.py",
        },
        "obs": obs,
        "label": {
            "semantic_state": rule_result.semantic_state,
            "target_ref": rule_result.target_ref,
            "role": rule_result.role,
            "constraints": constraints if any(v is not None for v in constraints.values()) else None,
            "profile_hint": rule_result.profile_hint,
            "event_flags": {
                "target_switch": _detect_target_switch(obs, rule_result),
            },
            "rationale_short": rationale[:120] if rationale else None,
        },
        "evidence": {
            "rule_ids": rule_result.rule_ids,
            "notes": rule_result.evidence_notes or None,
            "acmi_refs": [],
            "frame_refs": [],
        },
        "quality": {
            "tier": tier,
            "confidence": rule_result.confidence,
            "needs_review": needs_review,
            "conflict_flag": False,
        },
        "extras": {},
    }


def _make_sample_id(meta: dict) -> str:
    """Generate a deterministic sample ID from metadata."""
    scenario = meta.get("scenario_id") or "unknown"
    ts = meta.get("timestamp_sec")
    ts_str = f"{ts:.1f}" if ts is not None else "na"
    short_uuid = uuid.uuid4().hex[:6]
    return f"afsim_rule_{scenario}_{ts_str}_{short_uuid}"


def _build_rationale(rule_result: RuleResult) -> Optional[str]:
    """Build a short human-readable rationale from the rule result."""
    state = rule_result.semantic_state
    if state is None:
        return "规则无法判定, 需GPT审阅"
    mapping = {
        "merge_entry": "进入merge阶段, 近距交战",
        "commit_intercept": "主动截击, 稳定接敌",
        "extend": "脱离交战, 拉距恢复能量",
        "hold_geometry": "几何稳定, 保持当前态势",
    }
    return mapping.get(state, state)


# ═══════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════

def label_batch(
    obs_samples_path: Path,
    output_path: Path,
) -> Tuple[Path, dict]:
    """Batch-label a JSON-lines file of obs samples.

    Parameters
    ----------
    obs_samples_path:
        Path to a ``.jsonl`` file where each line is a JSON object with
        at least ``"obs"`` and optionally ``"meta"`` keys.
    output_path:
        Path where the labelled ``.jsonl`` output will be written.

    Returns
    -------
    (output_path, stats_dict)
        *stats_dict* contains counts of each label assigned plus the
        number of hard-case samples that were not labelled.
    """

    stats: Dict[str, int] = {
        "total": 0,
        "merge_entry": 0,
        "commit_intercept": 0,
        "extend": 0,
        "hold_geometry": 0,
        "null": 0,
        "hard_case": 0,
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
                logger.warning("行 %d: JSON 解析失败: %s", line_no, exc)
                continue

            obs = sample.get("obs", {})
            meta = sample.get("meta", {})
            stats["total"] += 1

            rule_result = classify_easy_case(obs)

            if rule_result is None:
                # Hard case -- write a stub record for downstream GPT review
                stub = {
                    "obs": obs,
                    "meta": meta,
                    "_hard_case": True,
                    "_line_no": line_no,
                }
                fout.write(json.dumps(stub, ensure_ascii=False) + "\n")
                stats["hard_case"] += 1
                continue

            # Build canonical record
            record = build_canonical_from_rule(obs, rule_result, meta)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update stats
            label_key = rule_result.semantic_state or "null"
            if label_key in stats:
                stats[label_key] += 1
            else:
                stats[label_key] = 1

    # Summary log
    easy = stats["total"] - stats["hard_case"]
    logger.info(
        "标注完成: %d 总样本, %d easy-case (%d merge_entry, %d commit, "
        "%d extend, %d hold, %d null), %d hard-case",
        stats["total"],
        easy,
        stats["merge_entry"],
        stats["commit_intercept"],
        stats["extend"],
        stats["hold_geometry"],
        stats["null"],
        stats["hard_case"],
    )

    return output_path, stats


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "D2b conservative rule labeler -- auto-label easy-case samples "
            "and emit hard-case stubs for GPT review."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input .jsonl file (one obs sample per line).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Path to output .jsonl file.  Defaults to "
            "<input_stem>_labelled.jsonl in the same directory."
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
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("输入文件不存在: %s", input_path)
        raise SystemExit(1)

    output_path: Path = args.output or input_path.with_name(
        f"{input_path.stem}_labelled.jsonl"
    )
    stats_path: Path = args.stats or output_path.with_name(
        f"{output_path.stem}_stats.json"
    )

    out, stats = label_batch(input_path, output_path)

    # Write stats
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("输出: %s", out)
    logger.info("统计: %s", stats_path)
    logger.info("统计摘要: %s", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
