"""Process boundary for a GPU regression that may poison later work."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _command(function: str) -> list[str]:
    return [sys.executable, "-c", (
        "import pytest\n"
        "from tests import test_live_qwen35_gguf_mtp_e2e as gate\n"
        "try:\n"
        f"    gate.{function}()\n"
        "except pytest.skip.Exception as exc:\n"
        "    print(str(exc), flush=True)\n"
        "    raise SystemExit(77)\n"
    )]


def run_context_crossing_isolated(*, timeout_seconds: float = 600) -> None:
    """Retire the crossing child before starting an independent health child.

    A failed or timed-out child never triggers more GPU work automatically.
    The caller reports the failure and coordinates recovery instead. The
    original model/backend/resource guards run inside each child as well.
    """
    for function in ("_run_dense_mtp_context_crossing", "_run_dense_mtp_fresh_health"):
        try:
            result = subprocess.run(
                _command(function), cwd=ROOT, capture_output=True, text=True,
                timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"{function} timed out after {timeout_seconds}s; child retired. "
                f"No following health request was started. stdout={exc.stdout!r} stderr={exc.stderr!r}"
            )
        if result.returncode == 77:
            pytest.skip(result.stdout.strip() or result.stderr.strip())
        assert result.returncode == 0, (
            f"{function} exited {result.returncode}; no further GPU work started.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
