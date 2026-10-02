"""The runner lifecycle tier must run on a host that has no ROCm at all.

``Gemma4LayerScratch.free`` used to call ``get_hip_runtime()`` unconditionally,
so ``close()`` on a fake-allocated runner failed wherever ``libamdhip64.so`` is
absent -- a no-ROCm CI or publish runner. Proving that fix by monkeypatching
``get_hip_runtime`` only shows that a mock was reached.

This test re-runs the CPU-only lifecycle and layer-contract modules in a
subprocess where ``ctypes.CDLL`` refuses to load the HIP runtime for real. Any
code path that reaches for it -- directly, through a module-level import, or
through a helper this repository has not been told about -- raises ``OSError``
instead of silently passing.

The HIP-availability guards in the GPU and live modules call ``ctypes.CDLL`` for
the same library, so under this bootstrap they see no runtime and skip. That is
the point: the modules named here must pass, and nothing in them may skip.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CPU_ONLY_MODULES = (
    "tests/test_unit_gemma4_runner_lifecycle.py",
    "tests/test_unit_gemma4_int8_kv_layer_contract.py",
)

_BOOTSTRAP = textwrap.dedent(
    """
    import ctypes
    import sys

    _real_cdll = ctypes.CDLL

    def _guarded_cdll(name, *args, **kwargs):
        if "amdhip64" in str(name):
            raise OSError(f"HIP runtime loading is prohibited in this test: {name}")
        return _real_cdll(name, *args, **kwargs)

    ctypes.CDLL = _guarded_cdll

    import pytest

    sys.exit(
        pytest.main(
            [
                "-p",
                "no:cacheprovider",
                # The repo's ``addopts`` carries ``-q``, which would suppress
                # the final summary line. ``-v`` cancels it back to normal
                # verbosity so the counts below can be read off the output.
                "-v",
                *sys.argv[1:],
            ]
        )
    )
    """
)


@pytest.mark.parametrize("module", _CPU_ONLY_MODULES)
def test_cpu_only_module_passes_with_hip_runtime_loading_prohibited(
    module: str, tmp_path: Path
) -> None:
    bootstrap = tmp_path / "prohibit_hip.py"
    bootstrap.write_text(_BOOTSTRAP)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    environment.pop("HIP_VISIBLE_DEVICES", None)

    completed = subprocess.run(
        [sys.executable, str(bootstrap), module],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"{module} needs the HIP runtime when none is loadable\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    # A guard that turned cases into skips would also exit zero, and would hide
    # exactly the regression this test exists to catch. Require every collected
    # case to have run and passed, with nothing skipped, failed, or errored.
    for bad in ("failed", "error", "skipped"):
        assert bad not in completed.stdout, completed.stdout
    collected = re.search(r"collected (\d+) items", completed.stdout)
    assert collected is not None, completed.stdout
    assert f"{collected.group(1)} passed" in completed.stdout, completed.stdout
