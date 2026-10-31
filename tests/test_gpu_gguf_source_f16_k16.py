"""Numerical and bounded-publication regression for source-F16 K16 consumers."""
import ctypes

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (copy_device_to_host, copy_host_to_device, free,
                                   host_array_ptr, malloc, memory_stats)
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_f16_rocblas_prefill as source


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP is not available")
@pytest.mark.parametrize("rows,hidden,cols", [(1, 256, 16), (17, 512, 32), (517, 6144, 16)])
def test_source_f16_k16_cpu_reference_and_strided_publication(rows, hidden, cols):
    rt = get_hip_runtime()
    rng = np.random.default_rng(219)
    # Dyadic operands make the independent FP64 reference exact. WMMA may
    # perturb exact BF16 ties, so this is a numerical CPU gate, not an
    # assertion that WMMA implements a scalar IEEE summation schedule.
    # The deep-K sentinel exposes FP16 accumulation losing small terms.
    x = (rng.integers(-8, 9, (rows, hidden)) / 32).astype(np.float16)
    w = (rng.integers(-8, 9, (cols, hidden)) / 32).astype(np.float16)
    x[0] = 1
    w[0] = 1
    w[0, 0] = 2048
    stride, offset = cols + 32, 16
    initial = np.full((rows, stride), 0xA55A, dtype=np.uint16)
    expected_f32 = (x.astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)
    bits = expected_f32.view(np.uint32)
    expected = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    before = memory_stats()
    buffers = []
    try:
        for array in (x, w, initial):
            buf = malloc(array.nbytes, runtime=rt)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=rt)
        source._launch_source_f16_k16(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr,
            rows, hidden, cols, stride, offset, stream=0, library=None, runtime=rt)
        actual = np.empty_like(initial)
        copy_device_to_host(host_array_ptr(actual), buffers[2], actual.nbytes, runtime=rt)
        published = actual[:, offset:offset + cols].copy()
        actual_f32 = (published.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_allclose(actual_f32, expected_f32, rtol=0.008, atol=1e-5)
        gate = evaluate_logits(expected_f32, actual_f32)
        assert gate.passed, gate
        assert published[0, 0] == expected[0, 0]
        np.testing.assert_array_equal(actual[:, :offset], initial[:, :offset])
        np.testing.assert_array_equal(actual[:, offset + cols:], initial[:, offset + cols:])
    finally:
        rt.device_synchronize()
        for buf in reversed(buffers):
            free(buf, runtime=rt)
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]


@pytest.mark.skipif(not _hip_available(), reason="HIP is not available")
@pytest.mark.parametrize("rows,hidden", [(17, 256), (517, 6144)])
@pytest.mark.parametrize("quant", ["q4", "q4_pair", "q5", "q6_planar"])
def test_bounded_k16_matches_registered_parent(rows, hidden, quant):
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
        gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
        gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
    )
    from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
    from hipengine.quant.gguf_t16 import (
        repack_gguf_q5_k_tile16, repack_gguf_q6_k_tile16_qmicro_planar,
    )
    from tests.test_gpu_gguf_k_gemv import make_q5_k_weight, make_q6_k_weight
    from tests.test_gpu_gguf_q4_k_gemv import make_q4_k_weight

    make_weight, repack, parent_fn, candidate_fn = {
        "q4": (make_q4_k_weight, repack_gguf_q4_k_tile16,
               gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
               source.gguf_q4_k_t16_f16_rocblas_bf16_bf16_out),
        "q4_pair": (make_q4_k_weight, repack_gguf_q4_k_tile16,
                    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
                    source.gguf_q4_k_t16_f16_rocblas_pair_bf16_bf16_out),
        "q5": (make_q5_k_weight, repack_gguf_q5_k_tile16,
               gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
               source.gguf_q5_k_t16_f16_rocblas_octet_bf16_bf16_out),
        "q6_planar": (make_q6_k_weight, repack_gguf_q6_k_tile16_qmicro_planar,
                      gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
                      source.gguf_q6_k_t16_qmicro_planar_f16_rocblas_bf16_bf16_out),
    }[quant]

    rt = get_hip_runtime()
    cols = 48
    rng = np.random.default_rng(219)
    x = rng.normal(0, 0.2, (rows, hidden)).astype(np.float32)
    bits = x.view(np.uint32)
    x_bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    tiles = repack(make_weight(cols, hidden)[None, ...]).tiles[0]
    before = memory_stats()
    buffers = []
    try:
        for array in (x_bf16, tiles):
            buf = malloc(array.nbytes, runtime=rt)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes, runtime=rt)
        # Output, parent output, input cast and a 32-column weight tile only.
        # A null legacy output scratch proves direct BF16 publication.
        for nbytes in (rows * cols * 2, rows * cols * 2,
                       rows * hidden * 2, 32 * hidden * 2):
            buffers.append(malloc(nbytes, runtime=rt))
        parent_fn(
            buffers[0].ptr, buffers[1].ptr, buffers[3].ptr,
            rows, hidden, cols, runtime=rt)
        candidate_fn(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr,
            buffers[4].ptr, buffers[5].ptr, 0, rows, hidden, cols,
            tile_out_features=32, runtime=rt)
        actual = np.empty((rows, cols), dtype=np.uint16)
        parent = np.empty_like(actual)
        copy_device_to_host(host_array_ptr(actual), buffers[2], actual.nbytes, runtime=rt)
        copy_device_to_host(host_array_ptr(parent), buffers[3], parent.nbytes, runtime=rt)
        np.testing.assert_array_equal(actual, parent)
    finally:
        rt.device_synchronize()
        for buf in reversed(buffers):
            free(buf, runtime=rt)
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
