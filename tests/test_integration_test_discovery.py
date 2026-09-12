"""Exercise discovery against synthetic modules, never the real full suite."""

from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from tests.conftest import pytest_addoption, pytest_configure


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ("unit",)),
        (["--suite", "gpu"], ("gpu",)),
        (["--suite", "all"], ("unit", "gpu", "legacy")),
        (["test_legacy_probe.py"], ("legacy",)),
        (["test_gpu_probe.py"], ("gpu",)),
    ],
)
def test_suite_discovery(tmp_path, args, expected):
    # Reuse the real hooks in an isolated root to avoid collecting this repo.
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n"
        + inspect.getsource(pytest_addoption)
        + "\n"
        + inspect.getsource(pytest_configure),
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    for tier in ("unit", "gpu", "legacy"):
        (tmp_path / f"test_{tier}_probe.py").write_text(
            f"def test_{tier}():\n    pass\n", encoding="utf-8"
        )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = {
        line for line in result.stdout.splitlines() if "::test_" in line
    }
    assert collected == {
        f"test_{tier}_probe.py::test_{tier}" for tier in expected
    }
