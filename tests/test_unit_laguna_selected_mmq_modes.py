from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_moe
from hipengine.runtime.laguna_moe import (
    resolve_laguna_selected_down_mode,
    resolve_laguna_selected_gate_up_mode,
)


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize("mode,resolver", [
    ("mmq128x32_d8_f32_rowvec", resolve_laguna_selected_gate_up_mode),
    ("mmq64x32_d4_f32_rowvec", resolve_laguna_selected_down_mode),
    ("mmq128x32_d8_f32_wavecols", resolve_laguna_selected_gate_up_mode),
    ("mmq64x32_d4_f32_wavecols_q4", resolve_laguna_selected_down_mode),
    ("mmq64x32_d4_f32_wavecols_direct_q4", resolve_laguna_selected_down_mode),
])
def test_legacy_mmq_comparison_modes_remain_explicit_not_defaults(backend, mode, resolver):
    assert resolver(backend, mode) == mode
    assert resolver(backend) != mode


def test_row_vector_primitives_remain_separate_from_session_mode_retirement():
    gate_parameters = inspect.signature(laguna_moe._launch_selected_gate_up_mmq32_d4x3).parameters
    down_parameters = inspect.signature(laguna_moe._launch_selected_down_mmq64x32_d4x3_f32).parameters
    assert gate_parameters["rowvec"].default is False
    assert gate_parameters["wave_cols"].default is False
    assert down_parameters["rowvec"].default is False
    assert down_parameters["wave_cols_q4"].default is False
    # Later wave-column Q4 modes still use the row-vector staging primitive;
    # deleting the two old session names must not delete that implementation.
    assert laguna_moe._should_fuse_selected_silu_pack(
        requested=True,
        selected_gate_up_mode="mmq128x32_d8_f32_wavecols_direct",
        selected_down_mode="mmq64x32_d4_f32_wavecols_direct_q4",
    )


@pytest.mark.parametrize("rows", [3, 6, 17])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("q6_rows", [32, 64])
def test_pair_decode_and_direct_modes_preserve_staging_and_launch_ownership(monkeypatch, rows, direct, q6_rows):
    calls = []
    plan = SimpleNamespace(
        hidden_size=8, expert_count=11, top_k=3, routed_scaling_factor=2.5,
        router_logits_functions={"token_tile_4": lambda *a, **kw: None},
        router_select=lambda *a, **kw: None,
        grouped_weighted_sum_shared_add=lambda *a, **kw: calls.append(("combine", a, kw)),
    )
    scratch = SimpleNamespace(plan=plan, max_rows=32)
    for index, name in enumerate((
        "router_logits", "routing_scores", "selection_scores", "selected_experts",
        "routing_weights", "scaled_routing_weights", "expert_down", "expert_gate_up",
        "grouped_sorted_weights", "grouped_sorted_lanes", "grouped_lane_to_row", "shared_output", "output",
    )):
        setattr(scratch, name, SimpleNamespace(ptr=0x2000 + index * 0x100, nbytes=4096))
    layer = SimpleNamespace(weight=lambda name: SimpleNamespace(allocation=lambda key: SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))))
    monkeypatch.setattr(laguna_moe, "validate_laguna_moe_layer", lambda *a: None)
    for name, label in (
        ("_launch_selected_gate_up_mmq32_d4x3", "gate"),
        ("_launch_selected_down_mmq64x32_d4x3_f32", "down"),
        ("_launch_laguna_shared_rows", "shared"),
    ):
        monkeypatch.setattr(laguna_moe, name, lambda *a, _label=label, **kw: calls.append((_label, a, kw)) or True)
    gate = "mmq128x32_d8_f32_wavecols" + ("_direct" if direct else "")
    down = (
        "mmq64x64_d4_f32_q6_wavecols_direct_q4" if q6_rows == 64
        else "mmq64x32_d4_f32_wavecols" + ("_direct_q4" if direct else "_q4")
    )
    runtime = object()
    result = laguna_moe.run_laguna_moe_rows(
        0x9000, layer, scratch, rows=rows, selected_gate_up_mode=gate, selected_down_mode=down,
        stream=7, runtime=runtime,
    )
    assert result is scratch.output
    assert [kind for kind, _, _ in calls] == ["gate", "down", "shared", "combine"]
    gate_kw, down_kw = calls[0][2], calls[1][2]
    assert gate_kw["rowvec"] is down_kw["rowvec"] is True
    assert gate_kw["wave_cols"] is down_kw["wave_cols_q4"] is True
    assert gate_kw["direct_wave_decode"] is direct
    assert down_kw["direct_wave_decode_q4"] is (direct or q6_rows == 64)
    assert gate_kw["lanes"] == down_kw["lanes"] == rows * plan.top_k
    assert gate_kw["residual_passes"] == down_kw["residual_passes"] == 1
    assert down_kw["q6_tile_rows"] == q6_rows
    assert all(kwargs["stream"] == 7 and kwargs["runtime"] is runtime for _, _, kwargs in calls)
    assert calls[-1][1][0] == scratch.expert_down.ptr
    assert calls[-1][1][5] == scratch.output.ptr


def test_rejected_one_plane_d4_gate_up_selector_is_retired() -> None:
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        with pytest.raises(ValueError, match="unsupported Laguna selected gate/up mode"):
            resolve_laguna_selected_gate_up_mode(backend, "mmq64x32_d4_f32")


@pytest.mark.parametrize(
    "mode",
    ("mmq64x32_d4_f32", "mmq64x32_d4x2_f32", "mmq64x32_d4x3_f32"),
)
def test_d4_down_modes_remain_independent_of_retired_gate_up_selector(mode: str) -> None:
    assert resolve_laguna_selected_down_mode("hip_gfx1151", mode) == mode


@pytest.mark.parametrize(
    "mode", ("mmq128x32_d8_f32", "mmq64x32_d4x2_f32", "mmq64x32_d4x3_f32")
)
def test_d8_and_residual_gate_up_modes_remain_available(mode: str) -> None:
    assert resolve_laguna_selected_gate_up_mode("hip_gfx1151", mode) == mode
