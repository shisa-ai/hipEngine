from __future__ import annotations

import inspect

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
])
def test_row_vector_comparison_modes_remain_explicit_not_defaults(backend, mode, resolver):
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
