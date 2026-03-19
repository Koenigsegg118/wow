#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Action command contract shared by sim<->inference<->ml."""

from __future__ import annotations

from typing import Any, Dict, List


ACTION_KEYS = {"entity_id", "dpsi_rad", "alt_sp_m", "spd_sp_mps"}


def validate_action_command(command: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(command, dict):
        return ["action command must be a dict"]
    missing = ACTION_KEYS.difference(command.keys())
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
        return errors
    if not isinstance(command.get("entity_id"), str) or not command["entity_id"]:
        errors.append("entity_id must be a non-empty string")
    for key in ("dpsi_rad", "alt_sp_m", "spd_sp_mps"):
        if not isinstance(command.get(key), (float, int)):
            errors.append(f"{key} must be numeric")
    return errors
