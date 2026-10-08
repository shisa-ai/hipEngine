"""RED/GREEN gate: the fused router select+cast is bit-identical to the pair.

The fused kernel reuses ``qwen35_router_select_from_logits`` verbatim and the
same round-to-nearest-even bf16 conversion the standalone cast kernel uses, so
selection indices, routing weights and cast bits must match the unfused pair
exactly on synthetic logits and activations - including tie-heavy logits,
denormal and boundary values for the cast, and the top_k<=16 extremes.
"""

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free, malloc
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_select,
    qwen35_router_select_cast_bf16,
)
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _alloc, _download, _upload


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not hip_available(), reason="HIP runtime not available")


def _reference_bf16_bits(x: np.ndarray) -> np.ndarray:
    u32 = x.astype(np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = (u32 + 0x7FFF + lsb).astype(np.uint64).astype(np.uint32)
    return (u32 >> 16).astype(np.uint16)


@pytest.mark.parametrize("tokens,experts,top_k,hidden", [
    (1, 512, 10, 2560),      # decode shape
    (3, 512, 10, 2560),      # multi-row
    (4, 64, 4, 320),         # small
    (2, 512, 16, 2560),      # max top_k
])
def test_fused_router_select_cast_is_bit_identical_to_pair(
    tokens, experts, top_k, hidden
):
    runtime = get_hip_runtime()
    rng = np.random.default_rng(20261008)
    logits = rng.normal(0, 3.0, (tokens, experts)).astype(np.float32)
    # Force exact ties and extremes into the selection.
    logits[0, 0] = logits[0, 1] = 7.5
    logits[0, 2] = -3.4028234663852886e38
    if tokens > 1:
        logits[1, :3] = 0.0
    values = rng.normal(0, 1.0, (tokens, hidden)).astype(np.float32)
    values[0, 0] = 1.0
    values[0, 1] = -1.0
    values[0, 2] = np.float32(1.5).view(np.uint32).view(np.float32) if hidden > 2 else values[0, 2]
    values[0, 3] = 6.1e-39 if hidden > 3 else values[0, 3]  # denormal region
    values[0, 4] = 3.3895e38 if hidden > 4 else values[0, 4]  # near bf16 max

    allocations = []
    try:
        d_logits = _upload(logits, runtime, allocations)
        d_values = _upload(values, runtime, allocations)
        # Parent pair.
        p_selected = _alloc((tokens, top_k), np.int64, runtime, allocations)
        p_routing = _alloc((tokens, top_k), np.float32, runtime, allocations)
        p_hidden_bf16 = _alloc((tokens, hidden), np.uint16, runtime, allocations)
        # Fused sibling.
        f_selected = _alloc((tokens, top_k), np.int64, runtime, allocations)
        f_routing = _alloc((tokens, top_k), np.float32, runtime, allocations)
        f_hidden_bf16 = _alloc((tokens, hidden), np.uint16, runtime, allocations)

        qwen35_router_select(
            d_logits.ptr, p_selected.ptr, p_routing.ptr,
            tokens, experts, experts, top_k, runtime=runtime,
        )
        f32_to_bf16(d_values.ptr, p_hidden_bf16.ptr, tokens * hidden, runtime=runtime)
        qwen35_router_select_cast_bf16(
            d_logits.ptr, f_selected.ptr, f_routing.ptr,
            d_values.ptr, f_hidden_bf16.ptr,
            tokens, experts, experts, top_k, hidden, runtime=runtime,
        )
        runtime.device_synchronize()

        np.testing.assert_array_equal(
            _download(p_selected, (tokens, top_k), np.int64, runtime),
            _download(f_selected, (tokens, top_k), np.int64, runtime),
        )
        np.testing.assert_array_equal(
            _download(p_routing, (tokens, top_k), np.float32, runtime),
            _download(f_routing, (tokens, top_k), np.float32, runtime),
        )
        fused_bits = _download(f_hidden_bf16, (tokens, hidden), np.uint16, runtime)
        parent_bits = _download(p_hidden_bf16, (tokens, hidden), np.uint16, runtime)
        np.testing.assert_array_equal(parent_bits, fused_bits)
        # Cross-check the cast against the CPU RNE reference.
        np.testing.assert_array_equal(_reference_bf16_bits(values), fused_bits)
    finally:
        for buffer in reversed(allocations):
            free(buffer, runtime=runtime)
