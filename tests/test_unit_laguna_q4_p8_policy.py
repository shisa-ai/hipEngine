"""Q4 P8 shape-policy and activation-sum comparison ownership."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q4_k_t16_selected_dual_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out as launch_gate,
    gguf_q4_k_t16_selected_q8_1_ds4_f32_mmq64x32_prefill_compact32_bf16_bf16_out as launch_down,
)
from hipengine.runtime import laguna_moe as moe
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from scripts import laguna_gate_doublebuf_ab as comparison


@pytest.mark.parametrize("rows", [17, 511, 512, 513, 777])
@pytest.mark.parametrize("sums", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_real_moe_caller_binds_p8_threshold_and_sum_producer(monkeypatch, rows, sums, prefetch):
    calls = []
    plan = SimpleNamespace(
        hidden_size=8, expert_count=11, top_k=3, routed_scaling_factor=2.5,
        router_logits_functions={"token_tile_4": lambda *a, **kw: None},
        router_select=lambda *a, **kw: None,
        grouped_weighted_sum_shared_add=lambda *a, **kw: calls.append(("combine", a, kw)),
    )
    scratch = SimpleNamespace(plan=plan, max_rows=1024)
    for index, name in enumerate((
        "router_logits", "routing_scores", "selection_scores", "selected_experts",
        "routing_weights", "scaled_routing_weights", "expert_down", "expert_gate_up",
        "grouped_sorted_weights", "grouped_sorted_lanes", "grouped_lane_to_row", "shared_output", "output",
    )):
        setattr(scratch, name, SimpleNamespace(ptr=0x2000 + index * 0x100, nbytes=1024))
    layer = SimpleNamespace(weight=lambda name: SimpleNamespace(allocation=lambda key: SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))))
    monkeypatch.setattr(moe, "validate_laguna_moe_layer", lambda *a: None)
    for name, label in (
        ("_launch_selected_gate_up_mmq32_d4x3", "gate"),
        ("_launch_selected_down_mmq64x32_d4x3_f32", "down"),
        ("_launch_laguna_shared_rows", "shared"),
    ):
        monkeypatch.setattr(moe, name, lambda *a, _label=label, **kw: calls.append((_label, a, kw)) or True)
    gate = (
        "mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512" if prefetch
        else "mmq128x32_d8_f32_wavecols_direct_doublebuf"
    )
    down = (
        "mmq64x64_d4_f32_q6_wavecols_direct_rawprefetch_q4_ge512" if prefetch
        else "mmq64x64_d4_f32_q6_wavecols_direct_q4"
    )
    runtime = object()
    output = moe.run_laguna_moe_rows(
        0x9000, layer, scratch, rows=rows, selected_gate_up_mode=gate,
        selected_down_mode=down, q4_precomputed_activation_sums=sums,
        stream=7, runtime=runtime,
    )
    assert [label for label, _, _ in calls] == ["gate", "down", "shared", "combine"]
    gate_kw, down_kw = calls[0][2], calls[1][2]
    eligible = prefetch and rows >= 512
    assert gate_kw["raw_weight_prefetch_packs"] == down_kw["q4_raw_weight_prefetch_packs"] == (8 if eligible else 0)
    assert gate_kw["precomputed_activation_sums"] is (sums and eligible)
    assert gate_kw["lanes"] == down_kw["lanes"] == rows * plan.top_k
    assert gate_kw["residual_passes"] == down_kw["residual_passes"] == 1
    assert gate_kw["double_buffer_activation"]
    assert all(kw["stream"] == 7 and kw["runtime"] is runtime for _, _, kw in calls)
    assert output is scratch.output and calls[-1][1][5] == scratch.output.ptr


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("mode,expected", [("q4_precomputed_sums_off", False), ("q4_precomputed_sums_on", True)])
def test_q4_sum_comparison_uses_real_setter_without_changing_q6(initial, mode, expected):
    session = object.__new__(LagunaGGUFResidentSession)
    session.q4_precomputed_activation_sums = initial
    session.q6_precomputed_activation_sums = object()
    before = session.q6_precomputed_activation_sums
    comparison._apply_mode(session, mode)
    assert session.q4_precomputed_activation_sums is expected
    assert session.q6_precomputed_activation_sums is before


class _Library:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        class Function:
            def __call__(_, *args):
                self.calls.append((name, tuple(arg.value for arg in args)))
                return 0

        return Function()


def _scratch(names, **plan):
    scratch = SimpleNamespace(plan=SimpleNamespace(**plan))
    for index, name in enumerate(names):
        setattr(scratch, name, SimpleNamespace(ptr=0x1000 + index * 0x100))
    return scratch


def _weight(pointer):
    return SimpleNamespace(
        spec=SimpleNamespace(quant_key="gguf_q4_k_t16_v1"),
        allocations={"tiles": object()},
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=pointer)),
    )


@pytest.mark.parametrize("prefetch,sums", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("defer", [False, True])
def test_real_gate_caller_and_wrapper_share_sum_sidecar_and_silu_target(prefetch, sums, defer):
    calls = []
    scratch = _scratch(
        (
            "selected_experts", "scaled_routing_weights", "grouped_expert_start",
            "grouped_active_experts", "grouped_active_count", "grouped_sorted_lanes",
            "grouped_lane_to_row", "grouped_sorted_weights", "grouped_expert_start_mmq32",
            "grouped_sorted_experts", "grouped_mmq_total", "activation_q8",
            "activation_q8_half_sums", "expert_down", "expert_gate_up", "expert_intermediate",
        ),
        expert_count=3, top_k=3, hidden_size=256, expert_ffn_size=64,
        grouped_smallm_downs={"gguf_q4_k_t16_v1": object()},
        grouped_compact_source_rows=lambda *a, **kw: None,
        mmq_tile_map=lambda *a, **kw: None,
        activation_quant_f32=lambda *a, **kw: calls.append(("quant", a, kw)),
        selected_gate_up_prefill_f32=launch_gate,
        selected_dual_silu=lambda *a, **kw: calls.append(("silu", a, kw)),
    )
    weights = {"ffn_gate_exps": _weight(0x9000), "ffn_up_exps": _weight(0xA000), "ffn_down_exps": _weight(0xB000)}
    library = _Library()
    runtime = object()
    assert moe._launch_selected_gate_up_mmq32_d4x3(
        0xF000, SimpleNamespace(weight=weights.__getitem__), scratch,
        x_rows=513, lanes=1539, stream=7, runtime=runtime,
        libraries={"selected_gate_up_prefill": library}, residual_passes=1,
        f32_wide=True, split16=True, rowvec=True, wave_cols=True,
        direct_wave_decode=True, double_buffer_activation=True,
        raw_weight_prefetch_packs=8 if prefetch else 0,
        precomputed_activation_sums=sums, defer_silu_pack=defer,
    )
    _, quant_args, quant_kw = calls[0]
    assert quant_args == (0xF000, scratch.activation_q8.ptr, 513, 256)
    assert quant_kw.get("q4_half_sums_ptr", 0) == (scratch.activation_q8_half_sums.ptr if sums else 0)
    assert quant_kw["runtime"] is runtime and quant_kw["stream"] == 7
    name, args = library.calls.pop()
    assert ("raw_prefetch_p8" in name) is prefetch
    assert ("precomputed_sums" in name) is sums
    pointers = (
        scratch.activation_q8.ptr,
        *((scratch.activation_q8_half_sums.ptr,) if sums else ()),
        scratch.grouped_lane_to_row.ptr, scratch.grouped_expert_start.ptr,
        scratch.grouped_expert_start_mmq32.ptr, scratch.grouped_sorted_experts.ptr,
        0x9000, 0xA000,
        scratch.expert_down.ptr if defer else scratch.expert_gate_up.ptr,
    )
    assert args[:len(pointers)] == pointers
    assert args[len(pointers):len(pointers) + 6] == (1539, 513, 256, 64, 64, 3)
    assert args[-1] == 7
    assert [label for label, _, _ in calls] == (["quant"] if defer else ["quant", "silu"])
    if not defer:
        assert calls[1][1] == (scratch.expert_gate_up.ptr, scratch.expert_intermediate.ptr, 1539, 64)


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_real_q4_down_caller_and_wrapper_preserve_pack_and_pointer_ownership(prefetch, fused):
    quant = []
    scratch = _scratch(
        ("expert_down", "expert_intermediate", "expert_gate_up", "grouped_expert_start",
         "grouped_expert_start_mmq32", "grouped_sorted_experts"),
        expert_count=3, expert_ffn_size=256, hidden_size=64, q6_qmicro=True,
        selected_down_prefill_q4=launch_down,
        down_activation_quant=lambda *a, **kw: quant.append(("plain", a, kw)),
        fused_selected_silu_pack=lambda *a, **kw: quant.append(("fused", a, kw)),
    )
    library = _Library()
    runtime = object()
    assert moe._launch_selected_down_mmq64x32_d4x3_f32(
        SimpleNamespace(weight=lambda name: _weight(0x9000)), scratch,
        lanes=1539, stream=7, runtime=runtime,
        libraries={"selected_gate_up_prefill": library}, residual_passes=1,
        rowvec=True, wave_cols_q4=True, direct_wave_decode_q4=True,
        q4_raw_weight_prefetch_packs=8 if prefetch else 0, fused_silu_pack=fused,
        q6_precomputed_activation_sums=True,
    )
    label, quant_args, quant_kw = quant.pop()
    assert label == ("fused" if fused else "plain")
    assert quant_args == (
        scratch.expert_down.ptr if fused else scratch.expert_intermediate.ptr,
        scratch.expert_gate_up.ptr, 1539, 256,
    )
    assert "q6_half_sums" not in quant_kw
    assert quant_kw["runtime"] is runtime and quant_kw["stream"] == 7
    name, args = library.calls.pop()
    assert ("raw_prefetch_p8" in name) is prefetch
    assert args[:6] == (
        scratch.expert_gate_up.ptr, scratch.grouped_expert_start.ptr,
        scratch.grouped_expert_start_mmq32.ptr, scratch.grouped_sorted_experts.ptr,
        0x9000, scratch.expert_down.ptr,
    )
    assert args[6:10] == (1539, 256, 64, 3) and args[-1] == 7
