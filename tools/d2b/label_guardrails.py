"""Post-GPT label guardrails for D2b pipeline.

Applies guideline-derived guards AFTER GPT labeling but BEFORE merge.
Guards can reject or downgrade GPT labels that violate known constraints.

Current guards:
- reject_merge_entry_for_standoff: prevents merge_entry when geometry
  indicates a standoff (closure≈0, bearing≈90°, aspect≈90°).
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

from tools.d2b.d2b_config import (
    MERGE_ENTRY_STANDOFF_GUARD,
    PROTECTED_RULE_LABELS,
    RULE_PROTECT_MIN_CONF,
    STANDOFF_ASPECT_MAX_DEG,
    STANDOFF_ASPECT_MIN_DEG,
    STANDOFF_MAX_CLOSURE_ABS_MPS,
    STANDOFF_MIN_BEARING_ABS_DEG,
)

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result of a guardrail check."""
    triggered: bool
    guard_name: str
    reason: str
    recommended_state: str | None = None  # None = use null


def reject_merge_entry_for_standoff(obs: dict, gpt_label: dict) -> GuardResult:
    """Reject merge_entry when geometry indicates standoff.

    Conditions (ALL must hold):
    - gpt_label.semantic_state == "merge_entry"
    - enemy is not None
    - engagement.is_merge == False
    - |enemy.closure_mps| <= STANDOFF_MAX_CLOSURE_ABS_MPS
    - |enemy.bearing_deg| >= STANDOFF_MIN_BEARING_ABS_DEG
    - STANDOFF_ASPECT_MIN_DEG <= enemy.aspect_deg <= STANDOFF_ASPECT_MAX_DEG
    """
    if not MERGE_ENTRY_STANDOFF_GUARD:
        return GuardResult(False, "standoff_guard", "guard disabled")

    state = gpt_label.get("semantic_state")
    if state != "merge_entry":
        return GuardResult(False, "standoff_guard", "not merge_entry")

    enemy = obs.get("enemy")
    if enemy is None:
        return GuardResult(False, "standoff_guard", "no enemy")

    engagement = obs.get("engagement") or {}
    is_merge = engagement.get("is_merge")
    if is_merge is True:
        return GuardResult(False, "standoff_guard", "is_merge=True")

    closure = enemy.get("closure_mps")
    bearing = enemy.get("bearing_deg")
    aspect = enemy.get("aspect_deg")

    if closure is None or bearing is None or aspect is None:
        return GuardResult(False, "standoff_guard", "missing geometry fields")

    if (abs(closure) <= STANDOFF_MAX_CLOSURE_ABS_MPS
            and abs(bearing) >= STANDOFF_MIN_BEARING_ABS_DEG
            and STANDOFF_ASPECT_MIN_DEG <= aspect <= STANDOFF_ASPECT_MAX_DEG):
        return GuardResult(
            triggered=True,
            guard_name="standoff_guard",
            reason=(
                f"merge_entry rejected: standoff geometry "
                f"(closure={closure:.1f}, bearing={bearing:.1f}, "
                f"aspect={aspect:.1f}, is_merge={is_merge})"
            ),
        )

    return GuardResult(False, "standoff_guard", "geometry does not match standoff")


def apply_guardrails(
    obs: dict,
    gpt_canonical: dict,
    rule_canonical: dict | None = None,
) -> dict:
    """Apply all guardrails to a GPT canonical record.

    Args:
        obs: The obs-sidecar dict.
        gpt_canonical: The canonical record from GPT adapter.
        rule_canonical: Optional rule-labeled canonical record for same sample.

    Returns:
        Modified canonical record (may have label/quality/source changed).
    """
    gpt_label = gpt_canonical.get("label", {})
    result = dict(gpt_canonical)

    # --- Guard 1: standoff merge_entry rejection ---
    guard = reject_merge_entry_for_standoff(obs, gpt_label)
    if guard.triggered:
        # Check if rule has a protected label
        rule_label = rule_canonical.get("label", {}) if rule_canonical else {}
        rule_state = rule_label.get("semantic_state")
        rule_conf = (rule_canonical or {}).get("quality", {}).get("confidence", 0)

        if (rule_state in PROTECTED_RULE_LABELS
                and rule_conf >= RULE_PROTECT_MIN_CONF):
            # Use protected rule label
            result["label"] = dict(rule_label)
            result["source"] = dict(result.get("source", {}))
            result["source"]["type"] = "mixed"
            result["quality"] = dict(result.get("quality", {}))
            result["quality"]["conflict_flag"] = True
            result["quality"]["needs_review"] = True
            # Cap tier at silver
            tier = result["quality"].get("tier")
            if tier == "gold":
                result["quality"]["tier"] = "silver"

            # Add guard evidence
            evidence = dict(result.get("evidence", {}))
            notes = evidence.get("notes") or ""
            evidence["notes"] = (
                f"{notes} [GUARD:{guard.guard_name}] "
                f"GPT={gpt_label.get('semantic_state')}, "
                f"Rule={rule_state}(conf={rule_conf}), "
                f"Final={rule_state}"
            ).strip()
            rule_ids = list(evidence.get("rule_ids", []))
            rule_ids.append("GUARD_standoff_merge")
            evidence["rule_ids"] = rule_ids
            result["evidence"] = evidence

            logger.info(
                "Guard protected rule label: %s -> %s (guard=%s)",
                gpt_label.get("semantic_state"), rule_state, guard.guard_name,
            )
        else:
            # No protected rule label -> set to null
            result["label"] = dict(gpt_label)
            result["label"]["semantic_state"] = None
            result["quality"] = dict(result.get("quality", {}))
            result["quality"]["conflict_flag"] = True
            result["quality"]["needs_review"] = True
            result["quality"]["tier"] = "weak"

            evidence = dict(result.get("evidence", {}))
            notes = evidence.get("notes") or ""
            evidence["notes"] = (
                f"{notes} [GUARD:{guard.guard_name}] "
                f"GPT={gpt_label.get('semantic_state')} rejected, "
                f"no protected rule -> null"
            ).strip()
            result["evidence"] = evidence

            logger.info(
                "Guard nullified GPT label: %s -> null (guard=%s)",
                gpt_label.get("semantic_state"), guard.guard_name,
            )

    return result


def apply_guardrails_batch(
    gpt_canonical_path: pathlib.Path,
    obs_path: pathlib.Path,
    rule_labeled_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Apply guardrails to a batch of GPT canonical records.

    Returns:
        (output_path, stats_dict)
    """
    if output_path is None:
        output_path = gpt_canonical_path.with_name(
            gpt_canonical_path.stem + "_guarded.jsonl"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load obs index
    obs_index: dict[str, dict] = {}
    with open(obs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            key = (
                f"{meta.get('scenario_id', '')}|"
                f"{meta.get('timestamp_sec', '')}|"
                f"{meta.get('ego_platform_idx', '')}"
            )
            obs_index[key] = rec.get("obs", rec)

    # Load rule index
    rule_index: dict[str, dict] = {}
    if rule_labeled_path and rule_labeled_path.exists():
        with open(rule_labeled_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                src = rec.get("source", {})
                ego = rec.get("obs", {}).get("ego", {})
                key = (
                    f"{src.get('scenario_id', '')}|"
                    f"{src.get('timestamp_sec', '')}|"
                    f"{ego.get('runtime_id', '')}"
                )
                rule_index[key] = rec

    stats = {"total": 0, "guarded": 0, "protected": 0, "nullified": 0}

    with open(gpt_canonical_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            gpt_rec = json.loads(line)
            stats["total"] += 1

            # Find matching obs and rule
            src = gpt_rec.get("source", {})
            ego = gpt_rec.get("obs", {}).get("ego", {})
            key = (
                f"{src.get('scenario_id', '')}|"
                f"{src.get('timestamp_sec', '')}|"
                f"{ego.get('runtime_id', '')}"
            )
            obs = obs_index.get(key, gpt_rec.get("obs", {}))
            rule_rec = rule_index.get(key)

            # Apply guards
            result = apply_guardrails(obs, gpt_rec, rule_rec)

            if result is not gpt_rec:
                old_state = gpt_rec.get("label", {}).get("semantic_state")
                new_state = result.get("label", {}).get("semantic_state")
                if old_state != new_state:
                    stats["guarded"] += 1
                    if new_state is not None:
                        stats["protected"] += 1
                    else:
                        stats["nullified"] += 1

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(
        "Guardrails: %d total, %d guarded (%d protected, %d nullified) -> %s",
        stats["total"], stats["guarded"],
        stats["protected"], stats["nullified"], output_path,
    )
    return output_path, stats
