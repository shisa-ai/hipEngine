"""The release gate must reject both module-level and individual test skips."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("source, expected", [
    ("def test_ok(): pass", 0),
    ("import pytest\ndef test_skip(): pytest.skip('checkpoint absent')", 1),
    ("import pytest\npytest.skip('dependency absent', allow_module_level=True)", 5),
    ("def test_bad(): assert False", 1),
])
def test_acceptance_rejects_incomplete_coverage(tmp_path, source, expected):
    path = tmp_path / "test_gate.py"
    path.write_text(source)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([
        sys.executable, "-c",
        "import pytest,sys; from scripts.surya_acceptance import NoSkippedTests; "
        "sys.exit(pytest.main(['-q',sys.argv[1]],plugins=[NoSkippedTests()]))",
        str(path),
    ], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stdout + result.stderr
    if "skip" in source:
        assert "skipped coverage" in result.stdout
