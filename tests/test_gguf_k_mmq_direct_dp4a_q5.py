"""Direct-load dp4a Q5_K MMQ32 candidate: contract and leaf numerics.

C8-P2 residual candidate (iteration 38): the same raw d4s4 weight/activation
formats as the retained raw mmq32 owner, but no LDS staging and no per-k
barriers (direct u32 record loads, dp4a partials, exact-integer min
correction). Arithmetic class is integer dp4a decode with a changed f32
accumulation order (T2 vs the raw owner). Contract mirrors the registered
mmq32 test: CPU q8-oracle agreement, exact-outer KL/top-1 floor, parity
against the raw owner, and run-to-run determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
    build_gguf_k_mmq_prefill,
    gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_bf16_out,
    gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_f32_out,
    gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out,
    gguf_q8_1_d4s4_f32_quantize_bf16,
)
from tests.test_gguf_k_mmq_prefill import (
    _bf16_bits,
    _bf16_to_f32,
    _pack_cpu,
    _quality,
    _unpack_cpu,
)
from tests.test_gguf_q6_k_t16_planar_q8_1_grouped_gemv import HIP_AVAILABLE
from tests.test_gguf_k_gemv import make_q5_k_weight


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (8, 24, 32))
def test_q5_direct_dp4a_oracle_outer_floor_and_parity(rows: int) -> None:
    """Candidate passes the q8-oracle gate, outer floor, and raw parity."""
    hidden, out_features = 512, 48
    rng = np.random.default_rng(20260729 + rows)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.125, size=(rows, hidden)).astype(np.float32)
    )
    qweight = make_q5_k_weight(out_features, hidden)
    packed = np.zeros((rows, hidden // 128, 160), dtype=np.uint8)
    out_f32 = np.zeros((rows, out_features), dtype=np.float32)
    raw_f32 = np.zeros((rows, out_features), dtype=np.float32)

    runtime = get_hip_runtime()
    library = build_gguf_k_mmq_prefill(load=True)
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        packed_dev = malloc(packed.nbytes, runtime=runtime)
        weight_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out_f32.nbytes, runtime=runtime)
        raw_dev = malloc(raw_f32.nbytes, runtime=runtime)
        buffers.extend((x_dev, packed_dev, weight_dev, out_dev, raw_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        gguf_q8_1_d4s4_f32_quantize_bf16(
            x_dev.ptr,
            packed_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_f32_out(
            packed_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q5_k_mmq32_q8_1_d4s4_f32_bf16_f32_out(
            packed_dev.ptr,
            weight_dev.ptr,
            raw_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(packed), packed_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_f32), out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(raw_f32), raw_dev, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    reconstructed = _unpack_cpu(_pack_cpu(x_bf16))
    q8_reference = gguf_q5_k_gemv(reconstructed, qweight)
    np.testing.assert_allclose(out_f32, q8_reference, rtol=2e-2, atol=1e-2)

    # Parity vs the retained raw owner: both accumulate exact-integer dp4a
    # terms in f32; only the summation order differs.
    scale = np.maximum(np.abs(raw_f32).max(), 1e-6)
    np.testing.assert_allclose(out_f32, raw_f32, rtol=1e-4, atol=scale * 1e-5)

    exact_reference = gguf_q5_k_gemv(_bf16_to_f32(x_bf16), qweight)
    max_kl, top1 = _quality(out_f32, exact_reference)
    assert max_kl <= 0.05
    assert top1 >= 0.9


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q5_direct_dp4a_bf16_out_matches_f32_and_deterministic() -> None:
    """bf16 out == bf16(f32 out), and repeats are bit-identical."""
    rows, hidden, out_features = 32, 512, 96
    rng = np.random.default_rng(20260730)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.125, size=(rows, hidden)).astype(np.float32)
    )
    qweight = make_q5_k_weight(out_features, hidden)

    runtime = get_hip_runtime()
    library = build_gguf_k_mmq_prefill(load=True)
    out_bf16 = np.zeros((rows, out_features), dtype=np.uint16)
    out_bf16_second = np.zeros_like(out_bf16)
    out_f32 = np.zeros((rows, out_features), dtype=np.float32)
    buffers = []
    try:
        x_dev = malloc(x_bf16.nbytes, runtime=runtime)
        packed_dev = malloc(rows * (hidden // 128) * 160, runtime=runtime)
        weight_dev = malloc(qweight.nbytes, runtime=runtime)
        bf16_dev = malloc(out_bf16.nbytes, runtime=runtime)
        f32_dev = malloc(out_f32.nbytes, runtime=runtime)
        buffers.extend((x_dev, packed_dev, weight_dev, bf16_dev, f32_dev))
        copy_host_to_device(x_dev, host_array_ptr(x_bf16), runtime=runtime)
        copy_host_to_device(weight_dev, host_array_ptr(qweight), runtime=runtime)
        gguf_q8_1_d4s4_f32_quantize_bf16(
            x_dev.ptr,
            packed_dev.ptr,
            rows,
            hidden,
            library=library,
            runtime=runtime,
        )
        gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_bf16_out(
            packed_dev.ptr,
            weight_dev.ptr,
            bf16_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_f32_out(
            packed_dev.ptr,
            weight_dev.ptr,
            f32_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_bf16), bf16_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_f32), f32_dev, runtime=runtime)
        gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_bf16_out(
            packed_dev.ptr,
            weight_dev.ptr,
            bf16_dev.ptr,
            rows,
            hidden,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(out_bf16_second), bf16_dev, runtime=runtime
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(out_bf16, _bf16_bits(out_f32))
    np.testing.assert_array_equal(out_bf16, out_bf16_second)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q5_direct_dp4a_rejects_bad_shapes() -> None:
    library = build_gguf_k_mmq_prefill(load=True)
    with pytest.raises(RuntimeError):
        gguf_q5_k_mmq32_direct_dp4a_q8_1_d4s4_f32_bf16_bf16_out(
            1, 1, 1, 8, 100, 16, library=library
        )
