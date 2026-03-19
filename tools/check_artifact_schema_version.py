#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that artifact policy.yaml schema_version matches shared/schema/VERSION."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid policy payload in {path}")
    return data


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    shared_version = (repo_root / "shared" / "schema" / "VERSION").read_text(encoding="utf-8").strip()
    artifact_root = repo_root / "artifacts"
    policies = sorted(artifact_root.rglob("policy.yaml")) if artifact_root.exists() else []

    if not policies:
        print("Artifact schema check PASSED (no policy.yaml found under artifacts/).")
        return

    mismatches: List[str] = []
    for policy_path in policies:
        payload = _load_yaml_or_json(policy_path)
        schema_version = str(payload.get("schema_version", "")).strip()
        if schema_version != shared_version:
            rel = policy_path.relative_to(repo_root).as_posix()
            mismatches.append(
                f"{rel}: policy.schema_version={schema_version!r} != shared/schema/VERSION={shared_version!r}"
            )

    if mismatches:
        print("Artifact schema check FAILED:")
        for line in mismatches:
            print(f"  - {line}")
        raise SystemExit(1)

    print(f"Artifact schema check PASSED ({len(policies)} policies checked).")


if __name__ == "__main__":
    main()
