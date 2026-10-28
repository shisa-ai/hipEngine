"""Selected T16 DP4A launches share one host-side scheduler control."""
from __future__ import annotations

import ctypes
from pathlib import Path
import re
import shutil
import subprocess

import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.hip"
_COMMON = "HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS"
_RETIRED = "HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS"


def _block_selector(source: str, launcher: str) -> str:
    body = source.split(f"int {launcher}(", 1)[1].split("\n}", 1)[0]
    return re.search(r"const dim3 block\((\w+)\(\)\);", body).group(1)


@pytest.fixture(scope="module")
def schedulers(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for host-only scheduler test")
    source = _SOURCE.read_text()
    # Compile the actual host functions and the selectors used by the launchers;
    # no HIP SDK or device is needed to check getenv/default precedence.
    helpers = source.split("int parse_selected_t16_dp4a_threads(", 1)[1].split("__device__", 1)[0]
    launchers = (
        "launch_q4_dual_q8_1_dp4a_direct",
        "launch_qk_q8_1_dp4a_direct",
        "launch_q4_dual_q8_1x2_dp4a_silu_direct",
        "launch_q4_dense_dual_q8_1x2_dp4a_silu",
    )
    selectors = [_block_selector(source, launcher) for launcher in launchers]
    root = tmp_path_factory.mktemp("selected-t16-scheduler")
    cpp = root / "scheduler.cpp"
    cpp.write_text(
        "#include <cstdlib>\nint parse_selected_t16_dp4a_threads(" + helpers
        + "".join(
            f'\nextern "C" int threads_{index}() {{ return {selector}(); }}\n'
            for index, selector in enumerate(selectors)
        )
    )
    library = root / "scheduler.so"
    subprocess.run([compiler, "-shared", "-fPIC", str(cpp), "-o", str(library)], check=True, capture_output=True)
    loaded = ctypes.CDLL(str(library))
    return [getattr(loaded, f"threads_{index}") for index in range(len(selectors))]


@pytest.mark.parametrize("value,expected", [(None, 64), ("", 64), ("32", 32), ("64", 64), ("128", 128), ("16", 64), ("invalid", 64)])
@pytest.mark.parametrize("stale_q5", [None, "32", "128", "invalid"])
def test_all_shared_launchers_resolve_valid_widths_and_ignore_retired_override(monkeypatch, schedulers, value, expected, stale_q5):
    for name, setting in ((_COMMON, value), (_RETIRED, stale_q5)):
        if setting is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, setting)
    assert [scheduler() for scheduler in schedulers] == [expected] * len(schedulers)


def test_q5_uses_common_selector_without_duplicate_lookup():
    source = _SOURCE.read_text()
    assert _block_selector(source, "launch_qk_q8_1_dp4a_direct") == "selected_t16_dp4a_threads"
    assert _RETIRED not in source
