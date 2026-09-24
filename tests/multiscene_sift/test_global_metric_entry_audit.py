"""Regression guards for global-consistency metric entry points."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts")


def _global_consistency_calls():
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if isinstance(function, ast.Name):
                    name = function.id
                elif isinstance(function, ast.Attribute):
                    name = function.attr
                else:
                    continue
                if name == "global_consistency_diagnostics":
                    yield path, node


def test_global_consistency_entries_have_no_pixel_size_one_placeholder():
    """Never reintroduce the old pixel_size=1.0 metric-unit bug."""
    offenders = []
    for path, call in _global_consistency_calls():
        for keyword in call.keywords:
            if keyword.arg != "pixel_size":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant) and value.value == 1.0:
                offenders.append(f"{path}:{call.lineno}")

    assert offenders == [], (
        "global consistency must use scene-derived pixel_size_m or explicit "
        f"resolution, not pixel_size=1.0: {offenders}"
    )
