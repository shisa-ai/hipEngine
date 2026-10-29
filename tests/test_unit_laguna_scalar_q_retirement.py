"""Bind scalar-Q retirement while preserving distinct pair-Q references."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_t16_selected_gemv as t16
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered, resolve


_PREFIX = "selected_dual_t16_natural_tile8_parallel_silu_"
_SUFFIX = "gemv_decode_bf16_bf16_out"
_QUANT = "gguf_q4_k_t16_v1"


def test_scalar_q_rollback_is_already_retired():
    t16.register_gguf_t16_selected_gemv_kernels()
    register_gfx1151_kernels(replace=True)
    name = "gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_scalarq_gemv_bf16_bf16_out"
    assert not hasattr(t16, name)
    assert name not in t16.__all__
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        assert not is_registered(KernelKey(backend, "moe_linear", _QUANT, _PREFIX + "scalarq_" + _SUFFIX))
    hip = Path(t16.__file__).with_suffix(".hip").read_text()
    assert "launch_q4_dual_natural_tile8_parallel_silu(" not in hip
    assert "hipengine_gguf_q4_k_t16_selected_dual_natural_tile8_parallel_silu_gemv_bf16_bf16_out(" not in hip
    for relative in (
        "scripts/laguna_selected_natural_decode_leaf.py",
        "tests/test_gpu_gguf_t16_selected_gemv_decode.py",
    ):
        assert "scalarq" not in (Path(__file__).resolve().parents[1] / relative).read_text()


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize("variant,symbol", [
    ("", t16._Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRCOEFF_BF16),
    ("paircoeff_", t16._Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRCOEFF_BF16),
    ("pairq_", t16._Q4_DUAL_NATURAL_TILE8_PARALLEL_SILU_PAIRQ_BF16),
])
def test_surviving_pair_q_registrations_reach_the_declared_symbol(backend, variant, symbol):
    t16.register_gguf_t16_selected_gemv_kernels()
    register_gfx1151_kernels(replace=True)
    calls = []

    class Function:
        def __call__(self, *args):
            calls.append(tuple(arg.value for arg in args))
            return 0

    def unexpected_runtime(*args):
        pytest.fail(f"unexpected HIP error: {args}")

    fn = resolve(backend=backend, layer="moe_linear", quant=_QUANT, variant=_PREFIX + variant + _SUFFIX)
    # Supplying only this symbol makes selecting the wrong reference fail.
    library = SimpleNamespace(**{symbol: Function()})
    fn(11, 12, 13, 14, 15, 1, 10, 256, 3072, 1024,
       stream=7, library=library, runtime=SimpleNamespace(check=unexpected_runtime))
    assert calls == [(11, 12, 13, 14, 15, 1, 10, 256, 3072, 1024, 7)]
