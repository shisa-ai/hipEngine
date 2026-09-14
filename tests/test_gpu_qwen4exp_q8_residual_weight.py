"""Two-plane Q8 weight decode must improve error against raw-weight truth."""

import ctypes

import numpy as np
import pytest


def available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


@pytest.mark.skipif(not available(), reason="HIP runtime unavailable")
@pytest.mark.parametrize("k,n", [(512, 128), (640, 2560), (640, 130)])
def test_residual_weight_plane_improves_cpu_reference_error(k, n):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
        gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out as parent,
        gguf_q8_0_selected_grouped_wmma_residual_prefill_bf16_bf16_out as candidate,
    )
    from hipengine.quant.gguf import dequantize_gguf_data, GGMLQuantizationType
    from tests.test_gpu_qwen4exp_q8_0_grouped_wmma_down import _quantize_q8_0, _tile_map, _f32_to_bf16_bits
    from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download

    rng = np.random.default_rng(31947)
    counts = [17, 33, 1, 48]
    rows, experts = sum(counts), 4
    raw = np.stack([_quantize_q8_0(rng.normal(0, .1, (n, k)).astype(np.float32))[0]
                    for _ in range(experts)])
    xbits = _f32_to_bf16_bits(rng.normal(0, 1, (rows, k)).astype(np.float32))
    x = (xbits.astype(np.uint32) << 16).view(np.float32)
    starts, padded, total = _tile_map(counts)
    tiles = np.repeat(np.arange(experts, dtype=np.int64), [(c + 15) // 16 for c in counts])
    truth = np.concatenate([
        x[starts[e]:starts[e + 1]].astype(np.float64)
        @ dequantize_gguf_data(raw[e], GGMLQuantizationType.Q8_0).reshape(n, k).astype(np.float64).T
        for e in range(experts)
    ]).astype(np.float32)
    truth = (_f32_to_bf16_bits(truth).astype(np.uint32) << 16).view(np.float32)
    runtime = get_hip_runtime()
    allocations = []
    results = []
    try:
        inputs = [_upload(a, runtime, allocations) for a in (xbits, starts, padded, tiles, raw)]
        output = _alloc(rows * n, np.uint16, runtime, allocations)
        for fn in (parent, candidate, candidate, candidate):
            fn(*(b.ptr for b in inputs), output.ptr, rows, experts, k, n, total * 16, runtime=runtime)
            runtime.device_synchronize()
            bits = _download(output, (rows, n), np.uint16, runtime)
            results.append((bits.astype(np.uint32) << 16).view(np.float32))
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    assert results[1].tobytes() == results[2].tobytes() == results[3].tobytes()
    assert np.isfinite(results[1]).all()
    parent_mse = np.mean((results[0].astype(np.float64) - truth) ** 2)
    candidate_mse = np.mean((results[1].astype(np.float64) - truth) ** 2)
    assert candidate_mse < parent_mse * .25, (parent_mse, candidate_mse)
    assert np.mean(np.argmax(results[1], axis=1) == np.argmax(truth, axis=1)) >= .90
    p = np.exp(truth.astype(np.float64) - truth.max(axis=1, keepdims=True))
    q = np.exp(results[1].astype(np.float64) - results[1].max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    assert np.mean(np.sum(p * np.log(p / q), axis=1)) <= .05
