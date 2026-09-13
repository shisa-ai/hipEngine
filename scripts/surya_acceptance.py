"""Run every Surya test and fail if any test or module is skipped.

Usage: uv run --extra surya --extra dev python scripts/surya_acceptance.py
Requires the Surya checkpoint and supported HIP hardware for the full gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path


class NoSkippedTests:
    def __init__(self):
        self.skipped: list[str] = []

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.skipped and exitstatus == 0:
            session.exitstatus = 1

    def pytest_terminal_summary(self, terminalreporter):
        if self.skipped:
            terminalreporter.write_sep("=", "Surya acceptance failed: skipped coverage")
            for nodeid in self.skipped:
                terminalreporter.write_line(nodeid)


def main(argv=None):
    import pytest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    tests = sorted((root / "tests").glob("test_*_surya*.py"))
    if not tests:
        parser.error("no Surya test files found")
    options = ["-ra", *(str(path) for path in tests)]
    if args.junitxml is not None:
        options.append(f"--junitxml={args.junitxml}")
    return int(pytest.main(options, plugins=[NoSkippedTests()]))


if __name__ == "__main__":
    raise SystemExit(main())
