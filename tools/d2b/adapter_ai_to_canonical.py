"""Adapt validated GPT format_C output to full canonical records.

Maps:
    AI output.label.*         → canonical.label.*
    AI output.quality_hint.*  → canonical.quality.{confidence, conflict_flag, needs_review}
    canonical.quality.tier    ← derived by rules (not set by LLM)
    canonical.source.type     = "acmi_ai"

Usage:
    python -m tools.d2b.adapter_ai_to_canonical \
        --validated datasets/d2b/gpt_validated.jsonl \
        --obs       datasets/d2b/obs_samples.jsonl \
        --output    datasets/d2b/canonical_from_gpt.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import uuid
from typing import Any

from tools.d2b.d2b_config import (
    CONFIDENCE_BRONZE_MIN,
    CONFIDENCE_SILVER_THRESHOLD,
    GPT_MODEL_NAME,
)

logger = logging.getLogger(__name__)


# ── tier derivation ───────────────────────────────────────────


def derive_tier(
    confidence: float | None,
    conflict_flag: bool,
    semantic_state: str | None,
) -> str:
    """Derive quality.tier from confidence + conflict_flag.

    Rules (from V1_1_PATCH_SUMMARY.md):
    - conflict_flag=True → tier upper bound = silver
    - confidence >= 0.80 → silver (or bronze if conflict)
    - 0.50 <= confidence < 0.80 → bronze
    - confidence < 0.50 or None → weak
    - semantic_state is None → weak (cannot train on null labels)
    """
    if semantic_state is None:
        return "weak"

    if confidence is None:
        return "weak"

    if conflict_flag:
        # Upper bound is silver, but only if confidence is high
        if confidence >= CONFIDENCE_SILVER_THRESHOLD:
            return "silver"
        if confidence >= CONFIDENCE_BRONZE_MIN:
            return "bronze"
        return "weak"

    # No conflict
    if confidence >= CONFIDENCE_SILVER_THRESHOLD:
        return "silver"
    if confidence >= CONFIDENCE_BRONZE_MIN:
        return "bronze"
    return "weak"


# ── canonical record builder ──────────────────────────────────


def _make_sample_id(meta: dict | None) -> str:
    """Generate a unique sample_id."""
    if meta and meta.get("request_id"):
        return f"gpt_{meta['request_id']}"
    return f"gpt_{uuid.uuid4().hex[:12]}"


def adapt_single(
    validated_output: dict,
    obs: dict,
    meta: dict | None = None,
) -> dict:
    """Convert validated GPT format_C + obs → full canonical record.

    Args:
        validated_output: Validated format_C dict with 'label' and 'quality_hint'.
        obs: The obs-sidecar dict used as input to GPT.
        meta: Optional metadata (scenario_id, timestamp_sec, etc.)

    Returns:
        Full canonical record per semantic_record_schema_v1.json.
    """
    label = validated_output.get("label", {})
    qhint = validated_output.get("quality_hint", {})

    confidence = qhint.get("confidence")
    conflict_flag = qhint.get("conflict_flag", False)
    needs_review = qhint.get("needs_review", False)
    semantic_state = label.get("semantic_state")

    tier = derive_tier(confidence, conflict_flag, semantic_state)

    # Ensure conflict_flag=True → needs_review=True
    if conflict_flag and not needs_review:
        needs_review = True

    sample_id = _make_sample_id(meta)
    meta = meta or {}

    return {
        "schema_version": "semantic_record_v1",
        "sample_id": sample_id,
        "source": {
            "platform": "afsim_acmi",
            "type": "acmi_ai",
            "scenario_id": meta.get("scenario_id"),
            "episode_id": meta.get("episode_id"),
            "timestamp_sec": meta.get("timestamp_sec"),
            "annotator": GPT_MODEL_NAME,
            "tool": "adapter_ai_to_canonical.py",
        },
        "obs": obs,
        "label": {
            "semantic_state": label.get("semantic_state"),
            "target_ref": label.get("target_ref"),
            "role": label.get("role"),
            "constraints": label.get("constraints", {
                "prefer_energy_preserve": None,
                "allow_aggressive_turn": None,
                "must_support_teammate": None,
                "should_abort_if_threatened": None,
            }),
            "profile_hint": label.get("profile_hint"),
            "event_flags": label.get("event_flags", {"target_switch": None}),
            "rationale_short": label.get("rationale_short"),
        },
        "evidence": {
            "rule_ids": [],
            "acmi_refs": [],
            "frame_refs": [],
            "notes": f"GPT auto-label by {GPT_MODEL_NAME}",
        },
        "quality": {
            "tier": tier,
            "confidence": confidence,
            "needs_review": needs_review,
            "conflict_flag": conflict_flag,
        },
        "extras": {},
    }


# ── batch processing ──────────────────────────────────────────


def _load_obs_index(obs_path: pathlib.Path) -> dict[str, dict]:
    """Load obs samples indexed by meta identifiers for pairing."""
    index: dict[str, dict] = {}
    with open(obs_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            obs = record.get("obs", record)
            meta = record.get("meta", {})
            # Index by multiple possible keys
            key = meta.get("request_id") or str(i)
            index[key] = {"obs": obs, "meta": meta}
    return index


def adapt_batch(
    validated_path: pathlib.Path,
    obs_path: pathlib.Path,
    output_path: pathlib.Path = pathlib.Path("datasets/d2b/canonical_from_gpt.jsonl"),
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Batch convert validated GPT outputs → canonical records.

    Pairs validated responses with their obs by request_id or line index.

    Returns:
        (output_path, stats_dict)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obs_index = _load_obs_index(obs_path)

    stats: dict[str, Any] = {
        "total": 0,
        "converted": 0,
        "skipped_null_state": 0,
        "tier_distribution": {"gold": 0, "silver": 0, "bronze": 0, "weak": 0},
        "state_distribution": {},
    }

    with open(validated_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            validated = json.loads(line)
            request_id = validated.get("request_id", str(i))

            # Find matching obs
            obs_record = obs_index.get(request_id, obs_index.get(str(i)))
            if obs_record is None:
                logger.warning("No obs found for request_id=%s, skipping", request_id)
                continue

            obs = obs_record["obs"]
            meta = obs_record.get("meta", {})
            meta["request_id"] = request_id

            canonical = adapt_single(validated, obs, meta)

            # Track stats
            state = canonical["label"]["semantic_state"]
            tier = canonical["quality"]["tier"]
            stats["tier_distribution"][tier] += 1
            if state:
                stats["state_distribution"][state] = (
                    stats["state_distribution"].get(state, 0) + 1
                )
            else:
                stats["skipped_null_state"] += 1

            stats["converted"] += 1
            fout.write(json.dumps(canonical, ensure_ascii=False) + "\n")

    logger.info(
        "Adapted %d/%d GPT outputs → %s",
        stats["converted"], stats["total"], output_path,
    )
    return output_path, stats


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt validated GPT outputs to canonical records")
    parser.add_argument("--validated", type=str, required=True,
                        help="Path to validated GPT responses JSONL")
    parser.add_argument("--obs", type=str, required=True,
                        help="Path to obs samples JSONL")
    parser.add_argument("--output", type=str,
                        default="datasets/d2b/canonical_from_gpt.jsonl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, stats = adapt_batch(
        pathlib.Path(args.validated),
        pathlib.Path(args.obs),
        pathlib.Path(args.output),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
