"""Host contracts for stable Laguna source-row compaction comparisons."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_moe as moe
from scripts import laguna_group_compact_parallel_ab as comparison


@pytest.mark.parametrize("mode", ["serial", "parallel"])
@pytest.mark.parametrize("rows", [32, 55, 128])
def test_gate_up_selects_source_compactor_without_changing_metadata_ownership(mode, rows):
    calls = []

    def record(label):
        return lambda *args, **kwargs: calls.append((label, args, kwargs))

    plan = SimpleNamespace(
        hidden_size=256, expert_ffn_size=256, expert_count=11, top_k=3,
        grouped_smallm_downs={"gguf_q6_k_t16_v1": object()},
        grouped_compact_source_rows=record("serial"),
        grouped_compact_source_rows_parallel=record("parallel"),
        mmq_tile_map=record("map"), activation_quant_f32=record("quant"),
        selected_gate_up_prefill_f32=record("gate"), selected_dual_silu=record("silu"),
    )
    scratch = SimpleNamespace(plan=plan)
    for index, name in enumerate((
        "selected_experts", "scaled_routing_weights", "grouped_expert_start", "grouped_active_experts",
        "grouped_active_count", "grouped_sorted_lanes", "grouped_lane_to_row", "grouped_sorted_weights",
        "grouped_expert_start_mmq32", "grouped_sorted_experts", "grouped_mmq_total", "activation_q8",
        "expert_down", "expert_gate_up", "expert_intermediate",
    )):
        setattr(scratch, name, SimpleNamespace(ptr=0x2000 + index * 0x100, nbytes=65536))
    weights = {
        name: SimpleNamespace(
            allocations={}, spec=SimpleNamespace(quant_key=quant),
            allocation=lambda key: SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000)),
        )
        for name, quant in (("ffn_gate_exps", "gguf_q4_k_t16_v1"), ("ffn_up_exps", "gguf_q4_k_t16_v1"), ("ffn_down_exps", "gguf_q6_k_t16_v1"))
    }
    layer = SimpleNamespace(weight=weights.__getitem__)
    runtime = object()
    assert moe._launch_selected_gate_up_mmq32_d4x3(
        0x9000, layer, scratch, x_rows=rows, lanes=rows * 3, stream=7, runtime=runtime,
        libraries=None, residual_passes=1, f32_wide=True, group_compact_mode=mode,
    )
    assert [label for label, _, _ in calls] == [mode, "map", "quant", "gate", "silu"]
    assert calls[0][1][:2] == (scratch.selected_experts.ptr, scratch.scaled_routing_weights.ptr)
    assert calls[0][1][-3:] == (rows * 3, 11, 3)
    assert all(kwargs["stream"] == 7 and kwargs["runtime"] is runtime for _, _, kwargs in calls)


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_compaction_defaults_to_parallel_with_explicit_serial_comparison(backend):
    assert moe.resolve_laguna_group_compact_mode(backend) == "parallel"
    assert moe.resolve_laguna_group_compact_mode(backend, "serial") == "serial"
    assert moe.resolve_laguna_group_compact_mode(backend, "parallel") == "parallel"
    with pytest.raises(ValueError, match="unsupported Laguna MoE group compact mode"):
        moe.resolve_laguna_group_compact_mode(backend, "atomic")


@pytest.mark.parametrize("mode,expected", list(comparison._MODES.items()))
@pytest.mark.parametrize("failure", [False, True])
def test_compaction_ab_closes_child_after_selected_mode_or_failure(monkeypatch, mode, expected, failure):
    calls = []

    def prefill(tokens):
        calls.append(("prefill", tuple(tokens)))
        if failure:
            raise RuntimeError("synthetic prefill failure")
        return SimpleNamespace(next_token_id=7)

    child = SimpleNamespace(
        set_group_compact_mode=lambda selected: calls.append(("mode", selected)),
        prefill=prefill, close=lambda: calls.append(("close",)),
        runtime=SimpleNamespace(device_synchronize=lambda: calls.append(("sync",))),
    )
    monkeypatch.setattr(comparison, "_child_session", lambda owner: child)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            comparison._run(object(), mode, [1, 2])
    else:
        clock = iter((1.0, 2.0))
        monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(clock))
        assert comparison._run(object(), mode, [1, 2]) == (512.0, 7)
    assert calls[0] == ("mode", expected)
    assert calls[-1] == ("close",)
