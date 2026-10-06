"""DP-5 grouped norm publication must have a real registered fallback."""
from __future__ import annotations

import pytest

from hipengine.kernels.registry import resolve
from hipengine.kernels.hip_gfx1100.fused import qwen4_exp_gr as gr


def test_dual_publication_and_unfused_fallback_registered():
    gr.register_qwen4_exp_gr_kernels()
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    register_gfx1151_kernels(replace=True)
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        for variant, name in (
            ("f32_f16_publish", "qwen4_exp_grouped_rmsnorm_bf16_f32_f16"),
            ("f32_f16_publish_unfused", "qwen4_exp_grouped_rmsnorm_bf16_f32_f16_unfused"),
        ):
            assert resolve(backend=backend, layer="gr_grouped_rmsnorm",
                           quant="bf16_f32", variant=variant) is getattr(gr, name)
        assert resolve(backend=backend, layer="gr_grouped_rmsnorm",
                       quant="bf16_f32", variant="strict") is gr.qwen4_exp_grouped_rmsnorm_bf16_f32


@pytest.mark.parametrize("rows,branches,hidden", [(0, 4, 2560), (1, 0, 2560), (1, 4, -1)])
def test_invalid_shape_rejected_before_loading(rows, branches, hidden):
    with pytest.raises(ValueError):
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(1, 2, 3, 4, rows, branches, hidden)


@pytest.mark.parametrize("pointers", [(0, 2, 3, 4), (1, 0, 3, 4), (1, 2, 0, 4),
                                      (1, 2, 3, 0), (1, 2, 3, 3)])
def test_invalid_or_overlapping_publication_rejected_before_loading(pointers):
    with pytest.raises(ValueError):
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(*pointers, 1, 4, 2560)


@pytest.mark.parametrize("rows,branches,hidden", [(1 << 32, 4, 2560), (1, 1, 1 << 63)])
def test_overflow_rejected_before_loading(rows, branches, hidden):
    with pytest.raises(ValueError, match="bounds"):
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
            0x100000, 0x200000, 0x300000, 0x400000, rows, branches, hidden)


@pytest.mark.parametrize("offset", [1, 127, 511])
def test_partial_overlap_rejected_before_loading(offset):
    with pytest.raises(ValueError, match="overlap"):
        gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
            0x100000, 0x200000, 0x300000, 0x300000 + offset, 1, 4, 33)


def test_unfused_fallback_preserves_order_stream_and_cast_library(monkeypatch):
    calls = []
    monkeypatch.setattr(gr, "qwen4_exp_grouped_rmsnorm_bf16_f32",
                        lambda *a, **k: calls.append(("norm", a, k)))
    monkeypatch.setattr(
        "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
        lambda *a, **k: calls.append(("cast", a, k)))
    library, runtime = object(), object()
    gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16_unfused(
        0x100000, 0x200000, 0x300000, 0x400000, 3, 4, 33,
        stream=9, library=library, runtime=runtime)
    assert [c[0] for c in calls] == ["norm", "cast"]
    assert calls[0][2] == {"stream": 9, "library": library, "runtime": runtime}
    assert calls[1][1] == (0x300000, 0x400000, 3 * 4 * 33)
    assert calls[1][2] == {"stream": 9, "runtime": runtime}


def test_publisher_raw_host_abi(monkeypatch):
    calls = []
    def signed(library, symbol, args, restype):
        assert symbol == "hipengine_qwen4_exp_grouped_rmsnorm_bf16_f32_f16"
        assert len(args) == 9
        return lambda *a: calls.append(a) or 0
    monkeypatch.setattr(gr, "signed_kernel_fn", signed)
    gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(
        0x100000, 0x200000, 0x300000, 0x400000, 3, 4, 33,
        stream=9, library=object(), runtime=object())
    assert calls == [(0x100000, 0x200000, 0x300000, 0x400000, 3, 4, 33, 1e-6, 9)]
