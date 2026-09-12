"""PN3 RED fixture for candidate P3-LAQ1 (c1 linear-attention QKV/gate GEMV).

Candidate: the exact ``dense_single_local32_bf16_bf16_out`` Q4_K T16 GEMV that
serves the c1 linear-attention ``attn_qkv`` (2048->8192) and ``attn_gate``
(2048->4096) projections on gfx1151. Mechanism under test: software-pipelined
super-block loads / improved memory-level parallelism that keeps the per-lane K
ownership, FMA order, wave tree, and BF16 rounding bit-exact (T0).

RED state (PN3, before implementation): the leaf performance ceiling tests
below FAIL against the current kernel --

* attn_qkv local32 leaf median ~0.0328 ms at burst=50, target <= 0.026 ms
  (~21% leaf reduction; ~0.5 ms/token over 30 layers once gate is included);
* pair (attn_qkv + attn_gate) leaf median ~0.0508 ms/layer, target <= 0.042 ms.

The bit-exact correctness guards (local32 T16 output == pack8 control output)
must stay GREEN for every retained variant.

Fresh-profile basis (PN3 reproduction): the PN2 host-wall stage attribution
over-attributes ``decode_linear_attn_qkv_gate`` (7.74 ms/token host wall vs
2.29 ms/token GPU-timestamped nested-exclusive from
``scripts/pn3_stage_ranking_from_trace.py``). The qkv/gate stage is ~70%
kernel-bound (leaf pair 1.52 ms/token of the 2.29 ms stage) and is the most
kernel-bound clean family in the top-10, so a T0 leaf mechanism has a real
cycle ceiling. See docs/QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md "PN3".

Deterministic synthetic weights (fixed seed); timing is wall-clock burst with
device sync (HIP event elapsed reports 0 on this gfx1151/ROCm combo).
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_gemv_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8, repack_gguf_q4_k_tile16

# RED perf ceilings (burst=50 wall-clock, gfx1151). Current leaf medians from
# scripts/pn3_q4_t16_laq_gemv_leaf.py: attn_qkv ~0.0328 ms, attn_gate ~0.018 ms,
# pair ~0.0508 ms. Targets are ~20% below current => the RED fails today.
ATTN_QKV_MS_RED = 0.026
ATTN_GATE_MS_RED = 0.0145
PAIR_MS_RED = 0.042

QK_K = 256
GGUF_Q4_K_BLOCK_BYTES = 144
SEED = 20260817


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _read_bf16(runtime, buffer, shape):
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(result), buffer, runtime=runtime)
    return result


def _burst_ms(runtime, fn, burst: int) -> float:
    import time

    fn()
    runtime.device_synchronize()
    start = time.perf_counter()
    for _ in range(burst):
        fn()
    runtime.device_synchronize()
    return (time.perf_counter() - start) / burst * 1e3


def _make_raw_q4_k(rng, in_features: int, out_features: int) -> np.ndarray:
    """Deterministic, finite Q4_K bytes: valid fp16 d=1.0/dmin=0.0 metadata so
    both the local32 and pack8 kernels dequantize to finite values and compare
    bit-exact."""
    blocks_per_row = in_features // QK_K
    bytes_per_row = blocks_per_row * GGUF_Q4_K_BLOCK_BYTES
    raw = np.zeros((out_features, bytes_per_row), dtype=np.uint8)
    blocks = raw.reshape(out_features, blocks_per_row, GGUF_Q4_K_BLOCK_BYTES)
    # d = 1.0 (fp16 0x3C00 little-endian), dmin = 0.0
    blocks[:, :, 0:2] = np.array([0x00, 0x3C], dtype=np.uint8)
    blocks[:, :, 2:4] = 0
    blocks[:, :, 4:16] = rng.integers(0, 256, size=blocks[:, :, 4:16].shape, dtype=np.uint8)
    blocks[:, :, 16:144] = rng.integers(0, 256, size=blocks[:, :, 16:144].shape, dtype=np.uint8)
    return raw


def _laq_harness(hip_test_target_arch, in_features: int, out_features: int):
    """Upload deterministic weights; return (local32_launcher, control, read_out).

    The local32 launcher closes over the T16 tiles; the pack8 control closes
    over qweight/scales/mins. Both write to their own out buffers.
    """
    runtime = get_hip_runtime()
    t16_lib = build_gguf_t16_selected_gemv(require_cached=True)
    pack8_lib = build_gguf_q4_k_gemv(require_cached=True)
    rng = np.random.default_rng(SEED)
    x_host = _bf16_bits(rng.standard_normal(in_features))
    raw = _make_raw_q4_k(rng, in_features, out_features)

    tiles = np.ascontiguousarray(repack_gguf_q4_k_tile16(raw[None, ...]).tiles)
    pack8 = repack_gguf_q4_k_pack8(raw)
    qweight = np.ascontiguousarray(pack8.qweight)
    scales = np.ascontiguousarray(pack8.scales)
    mins = np.ascontiguousarray(pack8.mins)

    x = _upload(runtime, x_host)
    tiles_dev = _upload(runtime, tiles)
    qw = _upload(runtime, qweight)
    sc = _upload(runtime, scales)
    mn = _upload(runtime, mins)
    out_local = _upload(runtime, np.zeros(out_features, np.uint16))
    out_pack8 = _upload(runtime, np.zeros(out_features, np.uint16))
    buffers = [x, tiles_dev, qw, sc, mn, out_local, out_pack8]

    def local32():
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x.ptr, tiles_dev.ptr, out_local.ptr, 1, in_features, out_features,
            library=t16_lib, runtime=runtime,
        )

    def control():
        gguf_q4_k_pack8_gemv_bf16_bf16_out(
            x.ptr, qw.ptr, sc.ptr, mn.ptr, out_pack8.ptr,
            1, in_features, out_features, threads=32, library=pack8_lib,
            runtime=runtime,
        )

    def read_out():
        return _read_bf16(runtime, out_local, (out_features,)), _read_bf16(
            runtime, out_pack8, (out_features,)
        )

    return local32, control, read_out, buffers


@pytest.fixture()
def laq_harness(hip_test_target_arch):
    runtime = get_hip_runtime()
    local32_q, control_q, read_q, buffers_q = _laq_harness(
        hip_test_target_arch, 2048, 8192
    )
    local32_g, control_g, read_g, buffers_g = _laq_harness(
        hip_test_target_arch, 2048, 4096
    )
    try:
        yield {
            "qkv": (local32_q, control_q, read_q),
            "gate": (local32_g, control_g, read_g),
        }
    finally:
        for buffer in list(reversed(buffers_q)) + list(reversed(buffers_g)):
            free(buffer, runtime=runtime)


def _warmup(*fns, runtime, repeats: int = 3):
    for _ in range(repeats):
        for fn in fns:
            fn()
    runtime.device_synchronize()


def test_laq_local32_bit_exact_vs_pack8(laq_harness, hip_test_target_arch):
    """GREEN guard: local32 T16 output must be bit-identical to the pack8 control."""
    runtime = get_hip_runtime()
    for key in ("qkv", "gate"):
        local32, control, read_out = laq_harness[key]
        _warmup(local32, control, runtime=runtime)
        local32()
        control()
        runtime.device_synchronize()
        local_out, pack8_out = read_out()
        assert np.array_equal(local_out, pack8_out), (
            f"{key} local32 vs pack8 control bit mismatch "
            f"({int(np.count_nonzero(local_out != pack8_out))} of "
            f"{local_out.size} outputs)"
        )


def test_laq_attn_qkv_leaf_red(laq_harness, hip_test_target_arch):
    """RED: attn_qkv local32 leaf must complete in <= 0.026 ms (currently ~0.033).

    xfail after the P3-LAQ1 GEMV-leaf mechanism was REJECTED: two bit-exact T0
    variants (word-load ``vecq``, 16-column ``tile16``) were implemented in
    PN4 and neither flips this ceiling (vecq ~10% slower; tile16 +5% on the
    larger qkv but -7% on the gate). The leaf runs ~0.029 ms (~350 GB/s
    effective) and is at its practical limit for the Q4_T16 byte-scattered
    layout; the 2.29 ms/token stage is dispatch-bound, not kernel-bound. See
    benchmarks/results/2026-08-17-zbook-qwen36-pn3-laq1-rejected.json and the
    plan's PN3/P3-LAQ1 section.
    """
    pytest.xfail(
        "P3-LAQ1 GEMV-leaf mechanism rejected: leaf at practical limit, "
        "no bit-exact T0 variant flips the attn_qkv ceiling"
    )
    runtime = get_hip_runtime()
    local32, control, _ = laq_harness["qkv"]
    _warmup(local32, control, runtime=runtime)
    medians = [
        _burst_ms(runtime, local32, 50) for _ in range(7)
    ]
    median = statistics.median(medians)
    assert median <= ATTN_QKV_MS_RED, (
        f"P3-LAQ1 RED: attn_qkv local32 leaf median {median:.4f} ms exceeds "
        f"ceiling {ATTN_QKV_MS_RED} ms"
    )


def test_laq_gate_leaf_red(laq_harness, hip_test_target_arch):
    """RED: attn_gate local32 leaf must complete in <= 0.0145 ms (currently ~0.018).

    xfail after the P3-LAQ1 GEMV-leaf mechanism was REJECTED (see
    ``test_laq_attn_qkv_leaf_red`` docstring).
    """
    pytest.xfail(
        "P3-LAQ1 GEMV-leaf mechanism rejected: leaf at practical limit, "
        "no bit-exact T0 variant flips the attn_gate ceiling"
    )
    runtime = get_hip_runtime()
    local32, control, _ = laq_harness["gate"]
    _warmup(local32, control, runtime=runtime)
    medians = [
        _burst_ms(runtime, local32, 50) for _ in range(7)
    ]
    median = statistics.median(medians)
    assert median <= ATTN_GATE_MS_RED, (
        f"P3-LAQ1 RED: attn_gate local32 leaf median {median:.4f} ms exceeds "
        f"ceiling {ATTN_GATE_MS_RED} ms"
    )


def test_laq_pair_leaf_red(laq_harness, hip_test_target_arch):
    """RED: pair (attn_qkv + attn_gate) local32 leaf must be <= 0.042 ms/layer.

    xfail after the P3-LAQ1 GEMV-leaf mechanism was REJECTED (see
    ``test_laq_attn_qkv_leaf_red`` docstring).
    """
    pytest.xfail(
        "P3-LAQ1 GEMV-leaf mechanism rejected: leaf at practical limit, "
        "no bit-exact T0 variant flips the pair ceiling"
    )
    runtime = get_hip_runtime()
    local32_q, _, _ = laq_harness["qkv"]
    local32_g, _, _ = laq_harness["gate"]
    _warmup(local32_q, local32_g, runtime=runtime)

    def pair():
        local32_q()
        local32_g()

    medians = [_burst_ms(runtime, pair, 50) for _ in range(7)]
    median = statistics.median(medians)
    assert median <= PAIR_MS_RED, (
        f"P3-LAQ1 RED: qkv+gate pair leaf median {median:.4f} ms/layer exceeds "
        f"ceiling {PAIR_MS_RED} ms/layer"
    )
