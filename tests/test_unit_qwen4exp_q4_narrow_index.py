"""EG-5 checked Q4 iu8-risk index-domain and raw ABI contracts (no HIP)."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.kernels.registry import resolve

VARIANT = "selected_dual_wmma_iu8_risk_narrow_prefill_bf16_bf16_out"
FUNCTION = "gguf_q4_k_" + VARIANT
SYMBOL = "hipengine_" + FUNCTION
LIMIT = 2**31 - 1


def admits(rows=40960, k=2560, a=640, b=640, experts=512, padded=48640):
    return q4.q4_iu8_narrow_index_admissible(rows, k, a, b, experts, padded)


@pytest.mark.parametrize("rows,padded", [(1, 16), (5120, 12800), (10240, 17920), (40960, 48640)])
def test_model_shapes_fit(rows, padded):
    assert admits(rows=rows, padded=padded)


@pytest.mark.parametrize("field,value", [
    ("rows", 0), ("k", 0), ("a", 0), ("b", 0), ("experts", 0), ("padded", 0),
    ("rows", -1), ("k", 255), ("a", 127), ("b", 127), ("padded", 17),
    ("rows", LIMIT), ("k", 2**32), ("a", 2**32), ("b", 2**32),
    ("experts", 2**32), ("padded", 2**32),
])
def test_invalid_or_overflow_dimensions_fail_closed(field, value):
    assert not admits(**{field: value})


def test_every_complete_byte_range_is_checked():
    # Independently overflow activations, outputs and each expert weight pool.
    assert not admits(rows=500000, k=2560, a=128, b=128, experts=1, padded=500000)
    assert not admits(rows=500000, k=256, a=2560, b=2560, experts=1, padded=500000)
    assert not admits(rows=16, k=2560, a=640, b=128, experts=3000, padded=16)
    assert not admits(rows=16, k=2560, a=128, b=640, experts=3000, padded=16)
    assert not admits(rows=32, padded=16)
    assert not admits(rows=16, experts=1, padded=48)  # impossible padding
    assert not admits(rows=16, experts=LIMIT, padded=16)  # map byte range


def test_last_safe_activation_byte_product():
    rows = LIMIT // (256 * 2)
    padded = (rows + 15) // 16 * 16
    assert admits(rows=rows, k=256, a=128, b=128, experts=1, padded=padded)
    assert not admits(rows=rows + 1, k=256, a=128, b=128, experts=1, padded=padded)


def test_registry_keeps_parent_and_registers_diagnostic_sibling():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    register_gfx1151_kernels(replace=True)
    assert resolve(backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
                   variant=VARIANT) is getattr(q4, FUNCTION)
    assert resolve(backend="hip_gfx1151", layer="moe_linear", quant="gguf_q4_k",
                   variant="selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out") is q4.gguf_q4_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out


class FakeFunction:
    def __call__(self, *args):
        self.args = args
        return 0


def test_sibling_uses_int64_host_dimensions_and_same_stream():
    fn = FakeFunction()
    library = SimpleNamespace(**{SYMBOL: fn})
    getattr(q4, FUNCTION)(*range(1, 10), 20480, 1.0, 16, 2560, 640, 640, 512, 16,
                          stream=77, library=library, runtime=SimpleNamespace())
    assert fn.argtypes == [ctypes.c_void_p] * 9 + [ctypes.c_int64, ctypes.c_float] + [ctypes.c_int64] * 6 + [ctypes.c_void_p]
    assert [v.value for v in fn.args[9:]] == [20480, 1.0, 16, 2560, 640, 640, 512, 16, 77]


def test_python_refuses_unsafe_domain_before_build(monkeypatch):
    monkeypatch.setattr(q4, "build_gguf_q4_k_selected_prefill",
                        lambda **kw: pytest.fail("unsafe shape reached build"))
    with pytest.raises(ValueError, match="int32|narrow"):
        getattr(q4, FUNCTION)(*range(1, 10), 1, 1.0, 500000, 2560, 128, 128, 1, 500000)


@pytest.mark.parametrize("capacity", [-1, 2**31 // 4, 2**32])
def test_risk_buffer_byte_capacity_refused_before_build(monkeypatch, capacity):
    monkeypatch.setattr(q4, "build_gguf_q4_k_selected_prefill",
                        lambda **kw: pytest.fail("unsafe capacity reached build"))
    with pytest.raises(ValueError):
        getattr(q4, FUNCTION)(*range(1, 10), capacity, 1.0, 16, 256, 128, 128, 1, 16)
