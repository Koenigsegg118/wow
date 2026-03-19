#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enforce architecture dependency direction constraints."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _collect_imports(tree: ast.AST) -> List[Tuple[int, str]]:
    imports: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((int(node.lineno), alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                imports.append((int(node.lineno), node.module))
    return imports


def _scan_package(repo_root: Path, package: str) -> List[Tuple[str, int, str]]:
    package_dir = repo_root / package
    violations: List[Tuple[str, int, str]] = []
    if not package_dir.exists():
        return violations

    forbidden: Dict[str, List[str]] = {
        "sim": ["ml"],
        "ml": ["sim"],
        "inference": ["sim", "ml"],
    }
    deny = forbidden.get(package, [])

    for py_path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in py_path.parts:
            continue
        rel = py_path.relative_to(repo_root).as_posix()
        text = py_path.read_text(encoding="utf-8-sig", errors="replace")
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            violations.append((rel, int(exc.lineno or 0), f"syntax_error: {exc.msg}"))
            continue

        for lineno, module_name in _collect_imports(tree):
            top_mod = module_name.split(".")[0]
            if top_mod in deny:
                violations.append((rel, lineno, f"{package} -> {top_mod} is forbidden"))
            if package == "sim" and module_name.startswith("shared.models"):
                violations.append(
                    (
                        rel,
                        lineno,
                        "sim must not import shared.models directly; use inference runtime/artifacts instead",
                    )
                )
    return violations


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    targets = ["sim", "ml", "inference"]
    violations: List[Tuple[str, int, str]] = []
    for package in targets:
        violations.extend(_scan_package(repo_root, package))

    if violations:
        print("Dependency direction check FAILED:")
        for rel, line, reason in violations:
            print(f"  - {rel}:{line}  {reason}")
        raise SystemExit(1)

    print("Dependency direction check PASSED.")
    print("  enforced: sim->inference/shared, ml->shared, inference->shared")


if __name__ == "__main__":
    main()
