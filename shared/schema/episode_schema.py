#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal versioned episode contract and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .action_schema import ACTION_KEYS, validate_action_command


SCHEMA_VERSION = (Path(__file__).with_name("VERSION")).read_text(encoding="utf-8").strip()

_TOP_KEYS = {
    "schema_version",
    "episode_id",
    "source",
    "time_step_s",
    "coordinate_frame",
    "entities",
    "frames",
    "actions",
}
_ENTITY_KEYS = {"entity_id", "name", "type", "side"}
_STATE_KEYS = {
    "entity_id",
    "alive",
    "x_e_m",
    "y_n_m",
    "z_u_m",
    "vx_e_mps",
    "vy_n_mps",
    "vz_u_mps",
    "heading_rad",
}
def _expect_keys(obj: Dict[str, Any], required: set[str], ctx: str, errors: List[str]) -> None:
    missing = required.difference(obj.keys())
    if missing:
        errors.append(f"{ctx}: missing keys {sorted(missing)}")


def validate_episode(episode: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(episode, dict):
        return ["episode must be a dict"]

    _expect_keys(episode, _TOP_KEYS, "episode", errors)

    if episode.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"episode.schema_version={episode.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )

    if not isinstance(episode.get("time_step_s"), (float, int)) or float(episode["time_step_s"]) <= 0:
        errors.append("episode.time_step_s must be a positive number")

    entities = episode.get("entities", [])
    if not isinstance(entities, list) or not entities:
        errors.append("episode.entities must be a non-empty list")
    else:
        for i, ent in enumerate(entities):
            if not isinstance(ent, dict):
                errors.append(f"entities[{i}] must be a dict")
                continue
            _expect_keys(ent, _ENTITY_KEYS, f"entities[{i}]", errors)

    frames = episode.get("frames", [])
    if not isinstance(frames, list) or not frames:
        errors.append("episode.frames must be a non-empty list")
    else:
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict):
                errors.append(f"frames[{i}] must be a dict")
                continue
            _expect_keys(frame, {"t_s", "states"}, f"frames[{i}]", errors)
            states = frame.get("states", [])
            if not isinstance(states, list):
                errors.append(f"frames[{i}].states must be a list")
                continue
            for j, st in enumerate(states):
                if not isinstance(st, dict):
                    errors.append(f"frames[{i}].states[{j}] must be a dict")
                    continue
                _expect_keys(st, _STATE_KEYS, f"frames[{i}].states[{j}]", errors)

    actions = episode.get("actions", [])
    if not isinstance(actions, list):
        errors.append("episode.actions must be a list")
    else:
        for i, frame_action in enumerate(actions):
            if not isinstance(frame_action, dict):
                errors.append(f"actions[{i}] must be a dict")
                continue
            _expect_keys(frame_action, {"t_s", "commands"}, f"actions[{i}]", errors)
            commands = frame_action.get("commands", [])
            if not isinstance(commands, list):
                errors.append(f"actions[{i}].commands must be a list")
                continue
            for j, cmd in enumerate(commands):
                if not isinstance(cmd, dict):
                    errors.append(f"actions[{i}].commands[{j}] must be a dict")
                    continue
                _expect_keys(cmd, ACTION_KEYS, f"actions[{i}].commands[{j}]", errors)
                for e in validate_action_command(cmd):
                    errors.append(f"actions[{i}].commands[{j}]: {e}")

    return errors


def assert_valid_episode(episode: Dict[str, Any]) -> None:
    errors = validate_episode(episode)
    if errors:
        raise ValueError("Invalid episode schema:\n- " + "\n- ".join(errors))


def load_episode(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_valid_episode(data)
    return data


def dump_episode(path: str | Path, episode: Dict[str, Any], *, indent: int = 2) -> None:
    assert_valid_episode(episode)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(episode, ensure_ascii=False, indent=indent), encoding="utf-8")
