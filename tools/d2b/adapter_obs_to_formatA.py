"""Module A part 2: Convert obs-sidecar output to format_A input for AI annotation.

The format_A input is the obs dict structure as defined in
``acmi_auto_label_io_examples.json`` ``format_A_minimal_input``:
top-level keys are ego / enemy / ally / engagement / history.

Rules:
- Strip any debug fields not in the canonical schema.
- Keep null values as null (never convert to "unknown").
- If enemy / ally / engagement is None, omit the key entirely.
- Always include history (even if all fields are null, output ``{}``).

Usage (CLI):
    python -m tools.d2b.adapter_obs_to_formatA \\
        --input obs_samples.jsonl \\
        --output obs_formatA_requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# ── allowed keys per sub-dict (from semantic_record_schema_v1.json obs node) ──

_EGO_KEYS = frozenset({
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
})

_ENEMY_KEYS = frozenset({
    "target_ref",
    "target_runtime_id",
    "callsign",
    "range_m",
    "bearing_deg",
    "aspect_deg",
    "closure_mps",
    "alt_diff_m",
    "is_primary_threat",
})

_ALLY_KEYS = frozenset({
    "ally_ref",
    "ally_runtime_id",
    "callsign",
    "range_m",
    "bearing_deg",
    "is_supporting",
})

_ENGAGEMENT_KEYS = frozenset({
    "is_merge",
    "is_defensive",
    "has_shot_opportunity",
})

_HISTORY_KEYS = frozenset({
    "prev_semantic_state",
    "prev_token_family",
    "prev_target_ref",
})


def _filter_keys(d: dict | None, allowed: frozenset[str]) -> dict | None:
    """Return a new dict with only allowed keys, or None if input is None."""
    if d is None:
        return None
    return {k: v for k, v in d.items() if k in allowed}


# ── public API ───────────────────────────────────────────────────


def obs_to_format_a(obs: dict) -> dict:
    """Convert obs-sidecar output to format_A input for AI annotation.

    Args:
        obs: Dict produced by ``build_obs_sidecar`` with keys
             ego, enemy, ally, engagement, history.

    Returns:
        Dict with top-level keys ego/enemy/ally/engagement/history.
        Keys whose value would be None (enemy, ally, engagement)
        are omitted entirely.  History is always present (at least ``{}``).
    """
    result: dict = {}

    # ego: always include (even if None, to signal dead platform)
    ego = _filter_keys(obs.get("ego"), _EGO_KEYS)
    result["ego"] = ego

    # enemy: omit key if None
    enemy = _filter_keys(obs.get("enemy"), _ENEMY_KEYS)
    if enemy is not None:
        result["enemy"] = enemy

    # ally: omit key if None
    ally = _filter_keys(obs.get("ally"), _ALLY_KEYS)
    if ally is not None:
        result["ally"] = ally

    # engagement: omit key if None
    engagement = _filter_keys(obs.get("engagement"), _ENGAGEMENT_KEYS)
    if engagement is not None:
        result["engagement"] = engagement

    # history: always include, at least empty dict
    history = _filter_keys(obs.get("history"), _HISTORY_KEYS)
    result["history"] = history if history is not None else {}

    return result


def build_format_a_batch(
    obs_samples_path: Path,
    output_path: Path,
) -> Path:
    """Batch convert obs_samples.jsonl to obs_formatA_requests.jsonl.

    Each input line is a JSON object with at least an ``obs`` key.
    Each output line is::

        {"request_id": "<uuid>", "obs_input": <format_A dict>}

    If the input record contains ``meta``, a deterministic request_id is
    derived from scenario_id, ego_platform_idx, and frame_idx.  Otherwise
    a random UUID is used.

    Args:
        obs_samples_path: Path to input JSONL (one obs sample per line).
        output_path: Path to write output JSONL.

    Returns:
        The *output_path* that was written.
    """
    obs_samples_path = Path(obs_samples_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(obs_samples_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            obs = record.get("obs", record)

            format_a = obs_to_format_a(obs)

            # Build request_id
            meta = record.get("meta")
            if meta is not None:
                sid = meta.get("scenario_id", "unknown")
                ego = meta.get("ego_platform_idx", 0)
                fidx = meta.get("frame_idx", count)
                request_id = f"{sid}_ego{ego}_f{fidx}"
            else:
                request_id = str(uuid.uuid4())

            out_record = {
                "request_id": request_id,
                "obs_input": format_a,
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(
        "Converted %d obs samples → format_A requests: %s",
        count, output_path,
    )
    return output_path


# ── CLI ──────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert obs-sidecar samples to format_A annotation requests."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        dest="input_path",
        help="Input obs_samples.jsonl path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        dest="output_path",
        help="Output obs_formatA_requests.jsonl path.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    build_format_a_batch(
        obs_samples_path=args.input_path,
        output_path=args.output_path,
    )
