from __future__ import annotations

import hipengine.runtime.moe_c1_dispatch as moe_c1_dispatch


def test_moe_c1_dispatch_split_output_tiled_prefill_default_and_optout(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", raising=False)

    assert moe_c1_dispatch._w4_dual_output_tiled_split_prefill_enabled()

    monkeypatch.setenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", "0")

    assert not moe_c1_dispatch._w4_dual_output_tiled_split_prefill_enabled()


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
