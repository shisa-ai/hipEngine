"""Legacy-vs-decode association parity for the dense Q8_0 pack8 GEMV decode rewrite.

The ``HIPENGINE_GGUF_GEMV_DECODE`` switch substitutes exactly one kernel in
the Gemma 4 26B-A4B dense decode path::

    gguf_q8_0  pack8_gemv_bf16_bf16_out  ->  pack8_gemv_decode_bf16_bf16_out

A resolve-level probe shows 205 launches change and nothing else; forcing the
rewrite with the env off reproduces the env-on logits **bit-identically**, so
this substitution is the whole candidate.

The two kernels are *not* bit-exact to each other, by construction:

* legacy ``gguf_k_pack8_prefill_out_kernel`` walks ``k += blockDim.x`` (unit
  stride, one element per thread per iteration);
* the decode kernel walks ``vec_stride = blockDim.x * 8`` with eight
  *consecutive* k per thread and a per-block scale hoisted out of the inner
  loop (``fmaf(xv, d * qs, acc)``).

Per-thread partial sums therefore cover different k-sets, so f32 rounding
differs and a small fraction of bf16 outputs land one ULP apart (~1e-4 of
elements; measured 1-2 per 8192 at ``blk.5.attn_q.weight``). Each kernel is
individually accurate against the f32 oracle, but the noise is enough to fail
the binding production gate (``docs/EXECUTION-PROFILES.md``: top-1 0.9687 vs
0.99, kl_max 2.45 vs 5e-2 on the frozen teacher-forced chain). The
8-consecutive/hoisted-scale walk *is* the optimization, so bit-exactness and
the speed trick are mutually exclusive at this design point.

This module pins both facts: per-kernel accuracy vs the oracle, and the
bounded (not zero) legacy-vs-decode divergence rate. Real Q8_0 shapes from
``gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`` (ggml order is ``(in, out)``): in
{2816, 4096, 2112, 704}, out {1024, 2048, 2112, 2816, 4096, 8192}.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q8_0_pack8_gemv_bf16_bf16_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv import (
    build_gguf_q8_0_pack8_gemv,
    gguf_q8_0_pack8_gemv_decode_bf16_bf16_out,
    register_gguf_q8_0_pack8_gemv_kernels,
)
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q8_0_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()

# Real dense Q8_0 projections from the Gemma 4 26B-A4B UD-Q4_K_XL artifact.
REAL_SHAPES = [
    (2816, 2048),   # attn_k / attn_v (sliding layers, 2 kv heads)
    (2816, 1024),   # attn_k / attn_v (5 sliding layers, 1 kv head)
    (2816, 4096),   # attn_q
    (2816, 8192),   # attn_q (5 wide layers) -- widest real dense row count
    (4096, 2816),   # attn_output
    (2112, 2816),   # ffn_down (dense)
    (2816, 2112),   # ffn_gate / ffn_up (dense)
]

# The association difference is a rare rounding-boundary event; bound the
# observed rate rather than demanding exactness, which the design cannot give.
MAX_DIVERGENCE_RATE = 1.0e-3


@pytest.fixture(scope="module")
def decode_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q8_0_pack8_gemv(load=True)


@pytest.fixture(scope="module")
def legacy_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_k_gemv(load=True)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _run_single(fn, x, qweight, rows, in_features, out_features, out_dtype, library):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    w_buf = malloc(qweight.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=library)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, w_buf, out_buf):
            free(b)


def _bf16_ulp(value: float) -> float:
    v = abs(float(value))
    if v == 0.0:
        return 2.0 ** -133
    exponent = int(np.floor(np.log2(v)))
    return 2.0 ** (exponent - 7)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features,out_features", REAL_SHAPES)
def test_both_kernels_are_within_one_bf16_ulp_of_the_cpu_oracle(
    in_features, out_features, decode_library, legacy_library
) -> None:
    """Both kernels must stay inside the final bf16 rounding of the f32 oracle."""

    register_gguf_k_gemv_kernels()
    register_gguf_q8_0_pack8_gemv_kernels()

    rng = np.random.default_rng(in_features * 17 + out_features * 3)
    qweight = make_q8_0_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(1, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    reference = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q8_0)

    for name, fn, library in (
        ("legacy", gguf_q8_0_pack8_gemv_bf16_bf16_out, legacy_library),
        ("decode", gguf_q8_0_pack8_gemv_decode_bf16_bf16_out, decode_library),
    ):
        actual = _bf16_u16_to_f32(
            _run_single(fn, x_bf16, qweight, 1, in_features, out_features, np.uint16, library)
        )
        err = np.abs(actual - reference)
        worst = np.unravel_index(int(np.argmax(err)), err.shape)
        ulps = float(err[worst]) / _bf16_ulp(float(reference[worst]))
        assert ulps <= 1.0, (
            f"{name} kernel exceeds one bf16 ulp vs the f32 CPU oracle at "
            f"in={in_features}, out={out_features}: {ulps:.2f} ulp at {worst} "
            f"(ref={float(reference[worst]):.6g}, got={float(actual[worst]):.6g})"
        )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_legacy_vs_decode_divergence_rate_is_bounded_but_nonzero(
    decode_library, legacy_library
) -> None:
    """Characterize the association gap at the widest real dense shape.

    The decode rewrite is *not* a bit-exact drop-in: its per-thread k-sets
    differ from the legacy kernel's, so a small fraction of bf16 outputs land
    one ULP apart. This test bounds that rate; it deliberately does not demand
    zero, which the design cannot deliver (and the production gate rejects the
    consequence -- see the module docstring and the campaign worklog entry).
    """

    register_gguf_k_gemv_kernels()
    register_gguf_q8_0_pack8_gemv_kernels()

    in_features, out_features = 2816, 8192
    qweight = make_q8_0_weight(out_features, in_features)
    total = 0
    differing = 0
    worst_ulps = 0.0
    for seed in range(32):
        rng = np.random.default_rng(seed)
        x_bf16 = _f32_to_bf16_u16(
            rng.normal(0.0, 0.3, size=(1, in_features)).astype(np.float32)
        )
        legacy = _run_single(
            gguf_q8_0_pack8_gemv_bf16_bf16_out,
            x_bf16, qweight, 1, in_features, out_features, np.uint16, legacy_library,
        )
        decode = _run_single(
            gguf_q8_0_pack8_gemv_decode_bf16_bf16_out,
            x_bf16, qweight, 1, in_features, out_features, np.uint16, decode_library,
        )
        diff = np.abs(_bf16_u16_to_f32(decode) - _bf16_u16_to_f32(legacy))
        differing += int(np.count_nonzero(decode != legacy))
        total += decode.size
        if diff.size and diff.max() > 0.0:
            worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
            worst_ulps = max(
                worst_ulps, float(diff[worst]) / _bf16_ulp(float(_bf16_u16_to_f32(legacy)[worst]))
            )

    rate = differing / total
    assert rate <= MAX_DIVERGENCE_RATE, (
        f"legacy-vs-decode divergence rate {rate:.3e} exceeds the documented "
        f"association-noise bound {MAX_DIVERGENCE_RATE:.3e} "
        f"({differing}/{total} elements over 32 seeds)"
    )
    # Characterized on the real artifact at this shape: 1-2 elements per 8192,
    # always within a single bf16 ulp.
    assert worst_ulps <= 1.0, f"divergence reached {worst_ulps:.2f} bf16 ulp"