"""CPU-only launch-symbol contracts for Q6 planar WMMA reference controls."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_moe as moe

from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q6_k_t16_selected_q8_1_ds4x3_f32_mmq64x32_prefill_compact32_bf16_bf16_out as launch,
)


@pytest.mark.parametrize("integer,hoist,symbol", [
    (None, None, "planar_integer_wmma_hoist_activation_skip"),
    (True, None, "planar_integer_wmma_hoist_activation_skip"),
    (True, False, "planar_integer_wmma_skip"),
    (False, None, "planar_skip"),
    (False, False, "planar_skip"),
])
@pytest.mark.parametrize("rows", [63, 64, 65, 193])
def test_planar_integer_and_hoist_controls_choose_distinct_owned_symbols(integer, hoist, symbol, rows):
    selected = []
    calls = []

    class Function:
        def __call__(self, *args):
            calls.append(tuple(value.value for value in args))
            return 0

    class Library:
        def __getattr__(self, name):
            selected.append(name)
            return Function()

    total = ((rows + 63) // 64) * 64
    launch(
        1, 2, 3, 4, 5, 6, rows, 256, 64, 3, total,
        residual_passes=1, rowvec=True, tile_rows=64, qmicro=True,
        compact_activation=True, half_row_activation=True,
        skip_padded_activation=True, qmicro_planar=True,
        integer_wmma=integer, wmma_hoist_activation=hoist,
        stream=7, library=Library(), runtime=object(),
    )
    name, = selected
    assert symbol in name
    args, = calls
    assert args == (1, 2, 3, 4, 5, 6, rows, 256, 64, 3, total, 7)


@pytest.mark.parametrize("overrides,error", [
    ({"qmicro_planar": False, "integer_wmma": True}, "integer_wmma requires qmicro_planar"),
    ({"integer_wmma": False, "wmma_hoist_activation": True}, "wmma_hoist_activation requires integer_wmma"),
    ({"wmma_hoist_activation": False, "wmma_prefetch_weight": True}, "wmma_prefetch_weight requires wmma_hoist_activation"),
    ({"wmma_prefetch_activation": True}, "wmma_prefetch_activation requires wmma_prefetch_weight"),
    ({"precomputed_activation_sums": True}, "precomputed_activation_sums requires production WMMA prefetch"),
])
def test_incompatible_wmma_reference_requests_fail_before_build_or_runtime(overrides, error):
    kwargs = dict(
        residual_passes=1, rowvec=True, tile_rows=64, qmicro=True,
        compact_activation=True, half_row_activation=True,
        skip_padded_activation=True, qmicro_planar=True,
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=error):
        launch(1, 2, 3, 4, 5, 6, 65, 256, 64, 3, 128, **kwargs)


@pytest.mark.parametrize("weight_prefetch,activation_prefetch,sums,symbol", [
    (False, False, False, "hoist_activation_skip"),
    (True, False, False, "prefetch_weight_skip"),
    (True, True, False, "prefetch_weight_activation_skip"),
    (True, True, True, "prefetch_weight_activation_precomputed_sums_skip"),
])
def test_real_prefetch_caller_agrees_on_sum_producer_and_consumer(weight_prefetch, activation_prefetch, sums, symbol):
    quant_calls = []
    launches = []

    class Function:
        def __call__(self, *args):
            launches.append(tuple(arg.value for arg in args))
            return 0

    class Library:
        def __getattr__(self, name):
            assert symbol in name
            return Function()

    scratch = SimpleNamespace(**{
        name: SimpleNamespace(ptr=0x1000 + index * 0x100)
        for index, name in enumerate((
            "grouped_expert_start", "grouped_expert_start_mmq32", "grouped_sorted_experts",
            "grouped_mmq_total", "expert_down", "expert_intermediate", "expert_gate_up",
        ))
    })
    scratch.plan = SimpleNamespace(
        expert_count=3, expert_ffn_size=256, hidden_size=64,
        q6_qmicro=True, q6_compact_activation=True, q6_half_row_activation=True,
        q6_skip_padded_activation=True, q6_qmicro_planar=True, q6_qmicro_permute=False,
        mmq64_tile_map=lambda *a, **kw: None, selected_down_prefill=launch,
        down_activation_quant=lambda *a, **kw: quant_calls.append((a, kw)),
    )
    weight = SimpleNamespace(
        spec=SimpleNamespace(quant_key="gguf_q6_k_t16_v1"),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=0x9000)),
    )
    runtime = object()
    assert moe._launch_selected_down_mmq64x32_d4x3_f32(
        SimpleNamespace(weight=lambda name: weight), scratch, lanes=65,
        stream=7, runtime=runtime, libraries={"selected_gate_up_prefill": Library()},
        residual_passes=1, rowvec=True, q6_tile_rows=64,
        q6_wmma_prefetch_weight=weight_prefetch,
        q6_wmma_prefetch_activation=activation_prefetch,
        q6_precomputed_activation_sums=sums,
    )
    args, kwargs = quant_calls.pop()
    assert args[:2] == (scratch.expert_intermediate.ptr, scratch.expert_gate_up.ptr)
    assert kwargs.get("q6_half_sums", False) is sums
    assert kwargs["runtime"] is runtime and kwargs["stream"] == 7
    args, = launches
    assert args[0] == scratch.expert_gate_up.ptr and args[4] == 0x9000
    assert args[5] == scratch.expert_down.ptr and args[-1] == 7
