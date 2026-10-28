from __future__ import annotations

import ctypes

import pytest

import hipengine.runtime.moe_c1_dispatch as moe_c1_dispatch


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
def test_moe_c1_dispatch_always_binds_promoted_dual_split(monkeypatch, retired) -> None:
    if retired is None:
        monkeypatch.delenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", retired)
    from hipengine.kernels.hip_gfx1100.fused import paro_combine, paro_silu
    from hipengine.kernels.hip_gfx1100.moe import router
    from hipengine.kernels.hip_gfx1100.quant import paro_awq_gemv
    from hipengine.kernels.hip_gfx1100.rotary import paro_rotate

    callback = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0)
    class Library:
        def __getattr__(self, name):
            return callback
    for module, name in (
        (router, "build_qwen35_router"), (paro_rotate, "build_paro_rotate"),
        (paro_awq_gemv, "build_paro_awq_gemv"), (paro_silu, "build_paro_silu"),
        (paro_combine, "build_paro_combine"),
    ):
        monkeypatch.setattr(module, name, lambda **kwargs: Library())
    table = moe_c1_dispatch._build_fns_table()
    assert table.gemv_awq_dual_pack8_output_tiled_split_transposed_fp16 == ctypes.cast(callback, ctypes.c_void_p).value


def test_moe_c1_dispatch_shared_down_combine_fused_default_and_optout(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED", raising=False)

    assert moe_c1_dispatch._linear_shared_down_combine_fused_enabled()

    monkeypatch.setenv("HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED", "0")

    assert not moe_c1_dispatch._linear_shared_down_combine_fused_enabled()


def test_moe_c1_dispatch_full_shared_down_combine_fused_default_and_optout(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED", raising=False)

    assert moe_c1_dispatch._full_shared_down_combine_fused_enabled()

    monkeypatch.setenv("HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED", "0")

    assert not moe_c1_dispatch._full_shared_down_combine_fused_enabled()


def test_moe_c1_dispatch_linear_shared_silu_rotate_fused_default_and_optout(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED", raising=False)

    assert moe_c1_dispatch._linear_shared_silu_rotate_fused_enabled()

    monkeypatch.setenv("HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED", "0")

    assert not moe_c1_dispatch._linear_shared_silu_rotate_fused_enabled()


def test_moe_c1_dispatch_linear_shared_down_mode_reuses_existing_w4_gates(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_W4_OUTPUT_TILED_PREFILL", raising=False)
    monkeypatch.delenv("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH", raising=False)

    assert moe_c1_dispatch._linear_shared_down_mode() == 2

    monkeypatch.setenv("HIPENGINE_W4_OUTPUT_TILED_PREFILL", "0")

    assert moe_c1_dispatch._linear_shared_down_mode() == 1

    monkeypatch.setenv("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH", "prefill")

    assert moe_c1_dispatch._linear_shared_down_mode() == 0

    monkeypatch.setenv("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH", "gemv")

    assert moe_c1_dispatch._linear_shared_down_mode() == 3

    monkeypatch.setenv("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH", "multi_row")

    assert moe_c1_dispatch._linear_shared_down_mode() == 4
