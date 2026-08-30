"""Correctness tests for the P1 device-driven grouped Q8_0 down owner.

The new kernel (``gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out``)
replaces the Q8 path's ``group_expert_start`` D2H copy + Python loop over 512
experts with a single device-driven launch that reads ``expert_start`` on
device and iterates experts internally (fixed worker grid, no host roundtrip).
It mirrors the Q5_1 device-driven grouped rowbatch8 dataflow.

RED contract: for compact rows grouped by expert, each output row equals
``gguf_q8_0_gemv(x_row, weight[expert])`` (the strict Q8_0 gemv reference)
within one bf16 ULP.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_q8_0_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    build_gguf_q8_0_prefill,
    gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out,
)
from tests.test_gguf_k_gemv import make_q8_0_weight

Q8_0_BLOCK_BYTES = 34


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = arr.astype(np.float32, copy=False)
    bits = f32.view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _make_grouped_fixture(*, num_experts, in_features, out_features, counts, seed=0):
    rng = np.random.default_rng(seed)
    compact_rows = int(sum(counts))
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(compact_rows, in_features)).astype(np.float32)
    )
    x = _bf16_bits_to_f32(x_bits)
    # Rank-3 raw Q8_0 weights [experts, out_features, row_bytes].
    base = make_q8_0_weight(out_features, in_features)
    raw = np.stack(
        [np.roll(base, shift=e, axis=0) for e in range(num_experts)], axis=0
    )
    # Build expert_start and the grouped reference.
    expert_start = np.zeros(num_experts + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    reference = np.zeros((compact_rows, out_features), dtype=np.float32)
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        start = int(expert_start[expert])
        stop = start + count
        reference[start:stop] = gguf_q8_0_gemv(
            x[start:stop], raw[expert]
        )
    return {
        "x_bits": np.ascontiguousarray(x_bits),
        "expert_start": expert_start,
        "weights": np.ascontiguousarray(raw),
        "reference": reference,
        "compact_rows": compact_rows,
        "num_experts": num_experts,
        "in_features": in_features,
        "out_features": out_features,
    }


def _run_grouped_gpu(fixture) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q8_0_prefill(load=True)
    out_dtype = np.uint16
    host_out = np.zeros(
        (fixture["compact_rows"], fixture["out_features"]), dtype=out_dtype
    )
    bufs = []
    try:
        x_dev = malloc(fixture["x_bits"].nbytes, runtime=runtime)
        start_dev = malloc(fixture["expert_start"].nbytes, runtime=runtime)
        w_dev = malloc(fixture["weights"].nbytes, runtime=runtime)
        out_dev = malloc(host_out.nbytes, runtime=runtime)
        bufs.extend((x_dev, start_dev, w_dev, out_dev))
        copy_host_to_device(
            x_dev, host_array_ptr(fixture["x_bits"]),
            fixture["x_bits"].nbytes, runtime=runtime,
        )
        copy_host_to_device(
            start_dev, host_array_ptr(fixture["expert_start"]),
            fixture["expert_start"].nbytes, runtime=runtime,
        )
        copy_host_to_device(
            w_dev, host_array_ptr(fixture["weights"]),
            fixture["weights"].nbytes, runtime=runtime,
        )
        gguf_q8_0_selected_grouped_prefill_compact_bf16_bf16_out(
            x_dev.ptr,
            start_dev.ptr,
            w_dev.ptr,
            out_dev.ptr,
            fixture["compact_rows"],
            fixture["num_experts"],
            fixture["in_features"],
            fixture["out_features"],
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(host_out), out_dev, host_out.nbytes, runtime=runtime,
        )
    finally:
        for buf in bufs:
            free(buf, runtime=runtime)
    return _bf16_bits_to_f32(host_out)


@pytest.mark.parametrize(
    "num_experts,in_features,out_features,counts",
    [
        # Layer-2 down shape: ffn=640 in -> hidden=2560 out, 4 experts, tails.
        (4, 640, 2560, [64, 0, 32, 16]),
        # Route order with empty experts and uneven tails.
        (8, 512, 2048, [3, 0, 29, 7, 0, 0, 41, 5]),
        # Full 512-expert layer-2 shape (ffn=640 in -> hidden=2560 out).
        # Stresses the fixed-capacity device grid across the whole expert set
        # with scattered empty experts, matching the real frozen MoE map.
        (512, 640, 2560, [3, 0, 5, 0, 2, 0, 0, 4, 1, 0, 2, 0, 6, 0, 0, 1]
         + [0] * 496),
        # Q8_0 block-multiple in_features, small out.
        (4, 32, 64, [2, 6, 0, 4]),
    ],
)
def test_qwen4_exp_q8_0_grouped_down_matches_reference(
    num_experts, in_features, out_features, counts
):
    if not _hip_available():
        pytest.skip("HIP runtime not available")
    fixture = _make_grouped_fixture(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        counts=counts,
        seed=7,
    )
    actual = _run_grouped_gpu(fixture)
    ref_bits = _f32_to_bf16_bits(fixture["reference"])
    ref = _bf16_bits_to_f32(ref_bits)
    # Every grouped output row must equal the strict Q8_0 gemv within one bf16
    # ULP of the reference output (the reference is already bf16-rounded the
    # same way the kernel stores).
    np.testing.assert_allclose(actual, ref, rtol=1e-3, atol=1e-3)
    assert np.all(np.isfinite(actual))
