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


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_raw_q4_single_resolves_registered_backend(monkeypatch, backend):
    calls = []
    monkeypatch.setattr(qgr, "_resolve_exact_selected_moe_kernel", lambda *args: None)
    monkeypatch.setattr(qgr, "_validate_raw_rank3_expert_weight", lambda *args, **kwargs: None)
    monkeypatch.setenv("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "0")

    def resolve(**key):
        calls.append(key)
        return lambda *args, **kwargs: calls.append((args, kwargs))

    monkeypatch.setattr(qgr, "resolve", resolve)
    qgr._launch_selected_raw_gguf_moe_linear(
        _weight("gguf_q4_k"), 100, 200, 300,
        x_rows=2, rows=4, num_experts=8, in_features=256,
        out_features=7, stream=7, runtime=_Runtime(), backend=backend,
    )
    assert calls[0] == dict(backend=backend, layer="linear", quant="gguf_q4_k",
                            variant="selected_gemv_bf16_bf16_out")
    assert calls[1][0][:4] == (100, 200, 12, 300)
    assert calls[1][1]["stream"] == 7


@pytest.mark.parametrize("kind", ["pair_exact", "pair_dp4a", "linear_dp4a"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("prefix", [None, "", "target_moe"])
@pytest.mark.parametrize("stream", [0, 7])
def test_selected_sync_timing_requires_flag_sink_and_prefix(
    monkeypatch, kind, enabled, record, prefix, stream,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_T16_SELECTED_DP4A", "1" if kind != "pair_exact" else "0")
    monkeypatch.setenv("HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_RAW_SELECTED_DP4A", "0")
    calls = []

    def capture(name):
        def launch(*args, **kwargs):
            calls.append((name, kwargs["stream"]))
        return launch

    for name in (
        "gguf_q4_k_quantize_bf16_q8_1",
        "gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out",
        "gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        "gguf_q5_k_t16_selected_q8_1_dp4a_gemv_bf16_bf16_out",
    ):
        monkeypatch.setattr(qgr, name, capture("quantize" if "quantize" in name else "gemv"))
    timings = {} if record else None
    runtime = _Runtime()
    options = dict(
        x_rows=2, rows=4, num_experts=256, in_features=2048,
        out_features=512, q8_1_workspace_ptr=500, stream=stream, runtime=runtime,
        stage_timings=timings, sync_stage_timings=enabled, stage_prefix=prefix,
    )
    if kind == "linear_dp4a":
        qgr._launch_selected_raw_gguf_moe_linear(
            _weight("gguf_q5_k_t16_v1"), 100, 200, 300, **options,
        )
    else:
        assert qgr._launch_selected_raw_gguf_moe_pair(
            _weight("gguf_q4_k_t16_v1"), _weight("gguf_q4_k_t16_v1"),
            100, 200, 300, 400, **options,
        )
    stages = ["gemv"] if kind == "pair_exact" else ["quantize", "gemv"]
    assert calls == [(stage, stream) for stage in stages]
    active = enabled and record and bool(prefix)
    # Target timing intentionally drains the device even for nonzero streams.
    assert runtime.syncs == (len(stages) if active else 0)
    if record:
        expected = {f"{prefix}_gemv"}
        if kind != "pair_exact":
            expected.add(f"{prefix}_q8_quantize")
        assert timings.keys() == (expected if active else set())
        assert all(value >= 0 for value in timings.values())


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


def test_selected_pair_dp4a_uses_f32_quantizer_when_source_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_T16_SELECTED_DP4A", "1")
    calls: list[tuple[str, tuple]] = []

    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_quantize_bf16_q8_1",
        lambda *args, **kwargs: pytest.fail("bf16 quantizer should not run"),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_quantize_f32_q8_1",
        lambda *args, **kwargs: calls.append(("quantize_f32", args)),
    )
    monkeypatch.setattr(
        qgr,
        "gguf_q4_k_t16_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append(("gemv", args)),
    )

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
        x_f32_ptr=900,
        stream=7,
        runtime=_Runtime(),
    )

    assert [name for name, _args in calls] == ["quantize_f32", "gemv"]
    assert calls[0][1][:4] == (900, 500, 2, 2048)
    assert calls[1][1][0] == 500


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
