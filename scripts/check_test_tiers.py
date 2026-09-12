"""Validate test naming and cross-tier imports without importing test modules."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re


TIERS = ("unit", "integration", "gpu", "benchmark", "live", "slow")
TEST_NAME = re.compile(r"test_(" + "|".join(TIERS) + r")_.+\.py$")


def inspect_tests(root: Path) -> list[str]:
    problems = []
    for path in sorted(root.rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        match = TEST_NAME.fullmatch(path.name)
        if match is None:
            problems.append(f"{relative}: missing execution-tier prefix")
            continue
        if match[1] != "unit":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.ImportFrom):
                parts = (node.module or "").split(".")
                if node.level or parts[0] == "tests":
                    modules = [part for part in parts if part.startswith("test_")]
                    imported.extend(modules)
                    # Names imported from a test module are functions/classes,
                    # whereas names imported from its package may be modules.
                    if not modules:
                        if node.level:
                            base = path.parent
                            for _ in range(node.level - 1):
                                base = base.parent
                            module_path = base.joinpath(*parts)
                        else:
                            module_path = root.joinpath(*parts[1:])
                        if not module_path.with_suffix(".py").is_file():
                            imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.extend(
                    part for alias in node.names
                    if alias.name.startswith("tests.")
                    for part in alias.name.split(".")[1:]
                    if part.startswith("test_")
                )
            for name in imported:
                if name.startswith("test_") and not name.startswith("test_unit_"):
                    problems.append(
                        f"{relative}:{node.lineno}: unit imports {name}; "
                        "extract a non-test CPU helper or move the test to its dependency tier"
                    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("tests"))
    args = parser.parse_args()
    problems = inspect_tests(args.root)
    for problem in problems:
        print(problem)
    print(f"test tiers: {len(problems)} issue(s)")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
