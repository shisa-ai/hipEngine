"""CPU-only launch-symbol contracts for Q6 planar WMMA reference controls."""

import pytest

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
