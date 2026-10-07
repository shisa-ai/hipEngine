"""Checked-int32/full-four-row dense Q8_0 coltile DPP sibling: CPU contracts.

The diagnostic sibling ``coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out``
narrows the dense pipelined-DPP leaf's weight/activation/output byte and element
offsets from int64 to int32 and drops the per-row validity control for complete
four-row groups. It is only valid after a complete-range admission proves every
signed-int32 intermediate and byte range fits, and only for rows divisible by
four; every other shape keeps the generic DPP parent. These cases pin the
admission and the refusal-before-build contract. No HIP is required.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as q8
from hipengine.kernels.registry import resolve

VARIANT = "coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out"
FUNCTION = "gguf_q8_0_gemv_" + VARIANT
SYMBOL = "hipengine_gguf_q8_0_gemv_" + VARIANT
PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
LIMIT = 2**31 - 1

# Dense Q8_0 projections of one Qwen3.8-Flash-Next layer, by name.
DENSE_SHAPES = {
    "attn_qkv": (10240, 2560),
    "attn_gate": (6144, 2560),
    "ssm_out": (2560, 6144),
    "ffn_gate_shexp": (640, 2560),
    "ffn_up_shexp": (640, 2560),
    "ffn_down_shexp": (2560, 640),
    "hc_attn_down": (320, 10240),
}


def admits(rows, in_features, out_features, col_tile=8, row_batch=4):
    return q8.q8_0_coltile_checked_int32_admissible(
        rows, in_features, out_features, col_tile, row_batch
    )


@pytest.mark.parametrize("name", sorted(DENSE_SHAPES))
@pytest.mark.parametrize("rows", [4, 512, 1024, 4096])
def test_model_projection_shapes_fit(name, rows):
    out_features, in_features = DENSE_SHAPES[name]
    assert admits(rows, in_features, out_features)


@pytest.mark.parametrize("k", [32, 96, 160, 2560, 6144, 10240])
def test_small_and_production_k_fit(k):
    assert admits(4, k, 8)
    assert admits(4096, k, 10240)


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (0, 2560, 10240),
        (-4, 2560, 10240),
        (4, 0, 10240),
        (4, 31, 10240),  # not a Q8_0 block multiple
        (4, 2560, 0),
        (4, 2560, 7),  # not a COL_TILE multiple
        (6, 2560, 10240),  # not a complete four-row group
        (2, 2560, 10240),
        (4, 2**32, 10240),
        (4, 2560, 2**32),
        (2**32, 2560, 10240),
        (LIMIT, 2560, 10240),
    ],
)
def test_invalid_or_overflow_dimensions_fail_closed(rows, in_features, out_features):
    assert not admits(rows, in_features, out_features)


def test_every_complete_byte_range_is_independently_checked():
    # Activation element range: rows * in_features must fit signed int32.
    safe_rows = (LIMIT // 32) // 4 * 4
    assert admits(safe_rows, 32, 8)
    assert not admits(safe_rows + 4, 32, 8)
    # Output element range: rows * out_features must fit signed int32, with the
    # raw-weight byte range still inside int32 so the output range is binding.
    row_bytes = 34
    out_cols = (LIMIT // row_bytes) // 8 * 8
    assert admits(32, 32, out_cols)
    assert not admits(36, 32, out_cols)
    # Raw weight byte range: out_features * row_bytes must fit signed int32.
    assert admits(4, 32, out_cols)
    assert not admits(4, 32, out_cols + 8)


def test_registry_keeps_parent_and_registers_diagnostic_sibling():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        assert (
            resolve(
                backend=backend,
                layer="linear",
                quant="gguf_q8_0",
                variant=VARIANT,
            )
            is getattr(q8, FUNCTION)
        )
        assert (
            resolve(
                backend=backend,
                layer="linear",
                quant="gguf_q8_0",
                variant="coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out",
            )
            is getattr(q8, PARENT)
        )


class FakeFunction:
    def __call__(self, *args):
        self.args = args
        return 0


def test_sibling_uses_int64_host_dimensions_and_forwards_the_stream():
    fn = FakeFunction()
    library = SimpleNamespace(**{SYMBOL: fn})
    getattr(q8, FUNCTION)(
        1, 2, 3, 4, 2560, 10240,
        stream=77, library=library, runtime=SimpleNamespace(),
    )
    assert fn.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    assert list(fn.args[3:]) == [4, 2560, 10240, 128, 77]


@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (2, 2560, 10240),  # tail rows: the generic DPP parent owns these
        (1000000, 2560, 128),
        (4, 2**32, 10240),
        (4, 2560, 2**32),
        (0, 2560, 10240),
        (4, 31, 10240),
    ],
)
def test_python_refuses_unsafe_domain_before_build(
    monkeypatch, rows, in_features, out_features
):
    monkeypatch.setattr(
        q8, "build_gguf_k_gemv", lambda **kw: pytest.fail("unsafe shape reached build")
    )
    with pytest.raises(ValueError, match="int32|four-row|divisible"):
        getattr(q8, FUNCTION)(
            1, 2, 3, rows, in_features, out_features, library=None, runtime=None
        )
