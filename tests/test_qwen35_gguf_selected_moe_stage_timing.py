from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as qgr


class _Runtime:
    def __init__(self) -> None:
        self.syncs = 0

    def device_synchronize(self) -> None:
        self.syncs += 1


def _weight(quant_key: str):
    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=11 if name == "tiles" else 12))

    return Weight()


def test_selected_pair_dp4a_sync_timing_splits_quantize_and_gemv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_T16_SELECTED_DP4A", "1")
    calls: list[str] = []

    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_quantize_bf16_q8_1",
        lambda *args, **kwargs: calls.append("quantize"),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("gemv"),
    )
    timings: dict[str, float] = {}
    runtime = _Runtime()

    assert qgr._launch_selected_raw_gguf_moe_pair(
        _weight("gguf_q4_k_t16_v1"),
        _weight("gguf_q4_k_t16_v1"),
        100,
        200,
        300,
        400,
        x_rows=2,
        rows=4,
        num_experts=256,
        in_features=2048,
        out_features=512,
        q8_1_workspace_ptr=500,
        stream=7,
        runtime=runtime,
        stage_timings=timings,
        sync_stage_timings=True,
        stage_prefix="moe_pair",
    )

    assert calls == ["quantize", "gemv"]
    assert "moe_pair_q8_quantize" in timings
    assert "moe_pair_gemv" in timings
    assert runtime.syncs == 2


def test_selected_pair_exact_sync_timing_reports_gemv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_T16_SELECTED_DP4A", raising=False)
    calls: list[str] = []

    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("gemv"),
    )
    timings: dict[str, float] = {}
    runtime = _Runtime()

    assert qgr._launch_selected_raw_gguf_moe_pair(
        _weight("gguf_q4_k_t16_v1"),
        _weight("gguf_q4_k_t16_v1"),
        100,
        200,
        300,
        400,
        x_rows=2,
        rows=4,
        num_experts=256,
        in_features=2048,
        out_features=512,
        q8_1_workspace_ptr=500,
        stream=7,
        runtime=runtime,
        stage_timings=timings,
        sync_stage_timings=True,
        stage_prefix="moe_pair",
    )

    assert calls == ["gemv"]
    assert "moe_pair_q8_quantize" not in timings
    assert "moe_pair_gemv" in timings
    assert runtime.syncs == 1


def test_selected_linear_dp4a_sync_timing_splits_quantize_and_gemv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_T16_SELECTED_DP4A", "1")
    calls: list[str] = []

    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_quantize_bf16_q8_1",
        lambda *args, **kwargs: calls.append("quantize"),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append("gemv"),
    )
    timings: dict[str, float] = {}
    runtime = _Runtime()

    qgr._launch_selected_raw_gguf_moe_linear(
        _weight("gguf_q5_k_t16_v1"),
        100,
        200,
        300,
        x_rows=4,
        rows=4,
        num_experts=256,
        in_features=512,
        out_features=2048,
        q8_1_workspace_ptr=500,
        stream=7,
        runtime=runtime,
        stage_timings=timings,
        sync_stage_timings=True,
        stage_prefix="moe_down",
    )

    assert calls == ["quantize", "gemv"]
    assert "moe_down_q8_quantize" in timings
    assert "moe_down_gemv" in timings
    assert runtime.syncs == 2
