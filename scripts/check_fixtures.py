#!/usr/bin/env python3
"""Run committed CPU-reference JSON fixtures without touching GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.kernels.cpu_reference import register_cpu_reference_kernels
from hipengine.kernels.cpu_reference.fixtures import load_fixture, run_fixture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("tests/fixtures/cpu_reference")],
        help="Fixture JSON file(s) or directories. Defaults to tests/fixtures/cpu_reference.",
    )
    args = parser.parse_args()

    register_cpu_reference_kernels()
    fixture_paths = tuple(_iter_fixture_paths(args.paths))
    if not fixture_paths:
        print("no fixture JSON files found", file=sys.stderr)
        return 2

    failed = 0
    for path in fixture_paths:
        fixture = load_fixture(path)
        result = run_fixture(fixture)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status} {path}: layer={result.layer} quant={result.quant} "
            f"max_abs={result.max_abs:.6g} max_rel={result.max_rel:.6g}"
        )
        failed += 0 if result.passed else 1
    return 1 if failed else 0


def _iter_fixture_paths(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.json"))
        else:
            yield path


if __name__ == "__main__":
    raise SystemExit(main())
