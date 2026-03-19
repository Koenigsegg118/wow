#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guard schema changes: contract code updates require version + docs updates."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Set


VERSION_FILE = "shared/schema/VERSION"
DOC_FILES = {
    "shared/schema/units.md",
    "shared/schema/coordinate_frames.md",
    "data_contract.md",
}
DOC_OR_VERSION = {VERSION_FILE, *DOC_FILES}


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _git_changed(repo_root: Path, base: str | None) -> List[str]:
    if base:
        range_expr = f"{base}...HEAD"
    else:
        env_base = os.getenv("SCHEMA_GUARD_BASE")
        if env_base:
            range_expr = f"{env_base}...HEAD"
        elif os.getenv("GITHUB_BASE_REF"):
            range_expr = f"origin/{os.getenv('GITHUB_BASE_REF')}...HEAD"
        else:
            range_expr = "HEAD~1...HEAD"
    cmd = ["git", "-C", str(repo_root), "diff", "--name-only", range_expr]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [_norm(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def _guard(changed: Iterable[str]) -> List[str]:
    changed_set: Set[str] = {_norm(x) for x in changed}
    schema_changed = [p for p in changed_set if p.startswith("shared/schema/")]
    if not schema_changed:
        return []

    meaningful_schema_changes = [p for p in schema_changed if p not in DOC_OR_VERSION]
    if not meaningful_schema_changes:
        return []

    errors: List[str] = []
    if VERSION_FILE not in changed_set:
        errors.append(
            "shared/schema contract code changed but VERSION was not updated."
        )
    if not any(p in changed_set for p in DOC_FILES):
        errors.append(
            "shared/schema contract code changed but docs were not updated "
            f"(expected one of: {sorted(DOC_FILES)})."
        )
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="Diff base ref for git diff")
    ap.add_argument(
        "--changed",
        nargs="*",
        default=None,
        help="Optional explicit changed files (bypass git diff)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    changed = args.changed if args.changed is not None else _git_changed(repo_root, args.base)
    errors = _guard(changed)

    if errors:
        print("Schema guard FAILED:")
        for msg in errors:
            print(f"  - {msg}")
        print("  changed files:")
        for path in sorted({_norm(x) for x in changed}):
            print(f"    * {path}")
        raise SystemExit(1)

    print("Schema guard PASSED.")


if __name__ == "__main__":
    main()
