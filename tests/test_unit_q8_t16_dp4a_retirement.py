"""Keep the retired Q8T16 integer-dot experiment out of callable routes."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_t16_gemv as kernels
from hipengine.kernels.registry import KernelKey, is_registered

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gguf_q8_0_t16_pair_microbench.py"
VARIANT = "t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out"


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_rejected_pair_is_not_callable_or_registered(backend):
    load_backend_kernel_package(backend)
    assert not hasattr(kernels, f"gguf_q8_0_{VARIANT}")
    assert not is_registered(KernelKey(backend, "linear", "gguf_q8_0_t16_v1", VARIANT))
    for variant in (
        "t16_dual_gemv_decode_bf16_bf16_out",
        "t16_dual_gemv_decode_rowtile2_bf16_bf16_out",
        "t16_dual_gemv_decode_rowtile4_bf16_bf16_out",
        "t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out",
    ):
        assert is_registered(KernelKey(backend, "linear", "gguf_q8_0_t16_v1", variant))


def test_rejected_device_body_and_exclusive_helpers_are_removed():
    source = Path(kernels.__file__).with_suffix(".hip").read_text()
    for name in ("dual_split_q8_1_dp4a", "gguf_q8_1_block", "pack_t16_col4", "q8_0_dp4a"):
        assert name not in source


@pytest.mark.parametrize("mode", ["q8_1_dp4a", "prequant_q8_1_dp4a"])
def test_microbench_rejects_retired_mode_before_loading_hip(mode):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--modes", mode],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert f"unknown modes: {mode}" in result.stderr
    assert "libamdhip64" not in result.stderr


def test_microbench_defaults_and_help_list_only_exact_variants():
    tree = ast.parse(SCRIPT.read_text())
    mode_arg = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument" and node.args
        and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--modes"
    )
    default = next(ast.literal_eval(kw.value) for kw in mode_arg.keywords if kw.arg == "default")
    assert default == "exact,rowtile2,rowtile4,rowtile4_col8"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert "rowtile4_col8" in result.stdout
    assert "q8_1" not in result.stdout
