"""The production eight-input-run expert GEMVs against the strict selected owners.

``selected_chunk8_gemv_bf16_bf16_out`` (Q4T16 gate/up tiles) and
``selected_pack8_chunk8_gemv_bf16_bf16_out`` (Q5_1 down) sum each output in a
different order from the strict selected owners, so they are compared within a
bf16 tolerance. What must hold exactly is lane-count invariance: a row's result
cannot depend on how many lanes share the launch, because a production verify
block and a production decode step launch the same row with different lane
counts and have to agree bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available
from tests.test_gpu_gemma4_assistant_forward import BACKBONE

pytestmark = pytest.mark.skipif(
    not (hip_runtime_available() and BACKBONE.exists()),
    reason="the Gemma 4 backbone and HIP are required",
)

_EXPERTS = 128
_HIDDEN = 2816
_INTERMEDIATE = 704


@pytest.fixture(scope="module")
def layer():
    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.gemma4 import load_gemma4_device_weights

    weights = load_gemma4_device_weights(GGUFReader(str(BACKBONE)))
    try:
        yield weights.layers[7]
    finally:
        weights.free()


def _run(fn, layer, which, x_bits, selected):
    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, free, malloc

    lanes = len(selected)
    if which == "gate_up":
        weight = layer.experts_gate_up_proj.allocation("tiles").buffer.ptr
        in_f, out_f = _HIDDEN, 2 * _INTERMEDIATE
    else:
        weight = layer.experts_down_proj.allocation("raw").buffer.ptr
        in_f, out_f = _INTERMEDIATE, _HIDDEN
    x = malloc(x_bits[:, :in_f].nbytes)
    sel = malloc(selected.nbytes)
    out = malloc(lanes * out_f * 2)
    try:
        copy_host_array_to_device(x, np.ascontiguousarray(x_bits[:, :in_f]))
        copy_host_array_to_device(sel, selected)
        fn(x.ptr, sel.ptr, weight, out.ptr, lanes, lanes, _EXPERTS, in_f, out_f, stream=0)
        host = np.empty(lanes * out_f, dtype=np.uint16)
        copy_device_to_host(host.ctypes.data, out, host.nbytes)
        return host.reshape(lanes, out_f)
    finally:
        for buffer in (x, sel, out):
            free(buffer)


def _kernels(which):
    if which == "gate_up":
        from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
            gguf_q4_k_t16_selected_chunk8_gemv_bf16_bf16_out as production,
            gguf_q4_k_t16_selected_gemv_bf16_bf16_out as strict,
        )
    else:
        from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
            qwen4_exp_q5_1_selected_pack8_chunk8_gemv_bf16_bf16_out as production,
            qwen4_exp_q5_1_selected_pack8_gemv_bf16_bf16_out as strict,
        )
    return production, strict


def _float(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("which", ["gate_up", "down"])
def test_a_row_does_not_depend_on_the_lanes_beside_it(layer, which) -> None:
    production, _ = _kernels(which)
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((40, _HIDDEN)).astype(np.float32) * 0.5)
    x_bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    selected = np.sort(rng.choice(_EXPERTS, 40)).astype(np.int64)

    block = _run(production, layer, which, x_bits, selected)
    for lane in (0, 13, 39):
        alone = _run(production, layer, which, x_bits[lane:lane + 1], selected[lane:lane + 1])
        np.testing.assert_array_equal(alone[0], block[lane])


@pytest.mark.parametrize("which", ["gate_up", "down"])
def test_the_production_sum_stays_within_bf16_rounding_of_the_strict_owner(layer, which) -> None:
    production, strict = _kernels(which)
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((8, _HIDDEN)).astype(np.float32) * 0.5)
    x_bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    selected = rng.choice(_EXPERTS, 8, replace=False).astype(np.int64)

    got = _float(_run(production, layer, which, x_bits, selected))
    want = _float(_run(strict, layer, which, x_bits, selected))
    # Two bf16 ulps of each row's largest magnitude: reassociation moves the
    # f32 sum by far less, then the bf16 store can round it either way.
    scale = np.abs(want).max(axis=1, keepdims=True)
    assert np.all(np.abs(got - want) <= 2 * scale * 2.0 ** -8)
