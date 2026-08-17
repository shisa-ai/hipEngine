"""PN4 RED fixture for candidate P3-LAQ1-B (Q8_0 t16 dual GEMV, c1 QKV/gate).

Candidate: the exact ``q8_0_t16_dual_split_gemv_kernel`` that serves the c1
linear-attention ``attn_qkv`` (2048->8192) and ``attn_gate`` (2048->4096)
projections on gfx1151 (both Q8_0 T16, already fused via ``q8_t16_dual_split``,
one launch per layer). Mechanism under test: T0 bit-exact kernel variants
(word-loads / occupancy / ILP) that keep per-lane K ownership, FMA order, wave
tree, and BF16 rounding identical to the owner.

RED state (PN4, after rejection): the leaf performance ceiling test below
FAILS to flip even with the best T0 variant --

* owner (committed) pair leaf ~0.062 ms/layer (burst30+sync), target <= 0.048;
* wordload / wordload+occ(128,8) / wordload+ILP+occ(128,8) T0 variants are all
  bit-exact but land at 0.058-0.062 ms/layer (best case ~9%, mostly within
  run-to-run clock noise) -- the 20-30% RED is unreachable.

The bit-exact correctness guard (t16 dual output == CPU reference, bf16-rounded
atol/rtol) must stay GREEN for every retained variant.

Evidence: ``benchmarks/results/2026-08-17-zbook-qwen36-pn4-laq1b-rejected.json``
and the host-bound A/B diagnostic ``scripts/pn4_host_bound_ab_probe.py``. The
kernel is latency/occupancy-bound (one block per 16-col tile, 768 blocks,
per-block wave reduce + xchg + syncthreads) at ~510-540 GB/s vs a ~650 GB/s
marginal L2 ceiling; T0 variants cannot close the latency gap, matching the
Q4_K T16 precedent (P3-LAQ1: vecq -10%, tile16 +5/-7%). See
docs/QWEN36-35B-ZBOOK-PRODUCTION-NUMERICS.md "PN4 / P3-LAQ1-B rejected".
"""

from __future__ import annotations

import ctypes
import statistics
import time

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
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
    build_gguf_q8_0_t16_gemv,
    gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import repack_gguf_q8_0_tile16
from tests._gguf_synthetic_weights import make_q8_0_weight

# RED perf ceiling (burst30+sync wall-clock, gfx1151). Owner median ~0.062 ms;
# target is ~20% below => the RED fails today and no T0 variant flips it.
PAIR_MS_RED = 0.048

SEED = 20260817

_TOL = {"atol": 5e-4, "rtol": 5e-3}


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _upload(runtime, values: np.ndarray):
    array = np.ascontiguousarray(values)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=runtime)
    return buffer


def _burst_ms(runtime, fn, burst: int) -> float:
    fn()
    runtime.device_synchronize()
    start = time.perf_counter()
    for _ in range(burst):
        fn()
    runtime.device_synchronize()
    return (time.perf_counter() - start) / burst * 1e3


@pytest.fixture(scope="module")
def q8_t16_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return build_gguf_q8_0_t16_gemv(load=True)


def _dual_pair_harness(q8_t16_library):
    """Upload deterministic Q8_0 t16 dual weights at the exact Qwen3.6-35B c1
    dims; return the pair launcher and the CPU-reference oracle outputs."""
    runtime = get_hip_runtime()
    in_features, oa, ob = 2048, 8192, 4096
    rng = np.random.default_rng(SEED)
    x_f32 = rng.normal(0.0, 0.3, size=(1, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x_f32)
    x_ref = _bf16_u16_to_f32(x_bf16)
    qa = make_q8_0_weight(oa, in_features)
    qb = make_q8_0_weight(ob, in_features)
    ta = repack_gguf_q8_0_tile16(qa).tiles
    tb = repack_gguf_q8_0_tile16(qb).tiles

    x = _upload(runtime, x_bf16)
    ta_dev = _upload(runtime, ta)
    tb_dev = _upload(runtime, tb)
    out_a = _upload(runtime, np.zeros(oa, np.uint16))
    out_b = _upload(runtime, np.zeros(ob, np.uint16))
    buffers = [x, ta_dev, tb_dev, out_a, out_b]

    def pair():
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out(
            x.ptr, ta_dev.ptr, tb_dev.ptr, out_a.ptr, out_b.ptr,
            1, in_features, oa, ob, library=q8_t16_library,
        )

    def read_out():
        a = np.empty(oa, np.uint16)
        b = np.empty(ob, np.uint16)
        copy_device_to_host(host_array_ptr(a), out_a, runtime=runtime)
        copy_device_to_host(host_array_ptr(b), out_b, runtime=runtime)
        return a, b

    ea = gguf_quant_gemv(x_ref, qa, GGMLQuantizationType.Q8_0)
    eb = gguf_quant_gemv(x_ref, qb, GGMLQuantizationType.Q8_0)
    ea_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(ea)).reshape(-1)
    eb_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(eb)).reshape(-1)
    return pair, read_out, (ea_bf16, eb_bf16), buffers


def test_laq1b_dual_bit_exact_vs_cpu_oracle(q8_t16_library):
    """GREEN guard: the Q8_0 t16 dual GEMV output must match the CPU reference
    (bf16-rounded) at the exact c1 dims."""
    pair, read_out, expected, buffers = _dual_pair_harness(q8_t16_library)
    try:
        pair()
        a, b = read_out()
        np.testing.assert_allclose(_bf16_u16_to_f32(a), expected[0], **_TOL)
        np.testing.assert_allclose(_bf16_u16_to_f32(b), expected[1], **_TOL)
    finally:
        for buffer in buffers:
            free(buffer, runtime=get_hip_runtime())


def test_laq1b_pair_leaf_red(q8_t16_library):
    """RED: the Q8_0 t16 dual pair leaf must complete in <= 0.048 ms/layer
    (currently ~0.062).

    xfail after the P3-LAQ1-B GEMV-leaf mechanism was REJECTED: three bit-exact
    T0 variants (wordload, wordload+occupancy 128,8, wordload+ILP+occupancy)
    were implemented and measured in PN4 and none flips this ceiling (all land
    at 0.058-0.062 ms/layer; best case ~9%, mostly within run-to-run clock
    noise). The leaf runs at ~510-540 GB/s vs a ~650 GB/s marginal L2 ceiling
    and is latency/occupancy-bound, at its practical limit for the Q8_T16
    one-block-per-tile layout. See
    benchmarks/results/2026-08-17-zbook-qwen36-pn4-laq1b-rejected.json and the
    plan's PN4 / P3-LAQ1-B-rejected section.
    """
    pytest.xfail(
        "P3-LAQ1-B GEMV-leaf mechanism rejected: leaf at practical limit, "
        "no bit-exact T0 variant flips the pair ceiling"
    )
    pair, read_out, expected, buffers = _dual_pair_harness(q8_t16_library)
    runtime = get_hip_runtime()
    try:
        for _ in range(3):
            pair()
        runtime.device_synchronize()
        medians = [_burst_ms(runtime, pair, 30) for _ in range(7)]
        median = statistics.median(medians)
        assert median <= PAIR_MS_RED, (
            f"P3-LAQ1-B RED: Q8_0 t16 dual pair leaf median {median:.4f} ms "
            f"exceeds ceiling {PAIR_MS_RED} ms"
        )
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
