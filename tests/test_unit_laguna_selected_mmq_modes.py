from __future__ import annotations

import pytest

from hipengine.runtime.laguna_moe import (
    resolve_laguna_selected_down_mode,
    resolve_laguna_selected_gate_up_mode,
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
