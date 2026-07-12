"""Correctness for the small-B weight-amortized GGUF raw-K row-tile GEMV.

`gguf_k_prefill_out_rowtile_kernel<...,qtype,ROW_TILE>` is the verifier small-B
(rows 2..8) replacement for the per-row `gguf_k_prefill_out_kernel` for the raw
K-quants Q8_0 (qtype=8), Q5_K (5), Q6_K (6). Q8_0 is the qwen35moe dense
projection quant (attn_qkv/gate, ssm_out), which dominates the target verifier.

Gate: bit-exact vs the per-row kernel + within tolerance of a CPU dequant oracle,
for Q8_0/Q5_K/Q6_K bf16->bf16 and bf16->f32, across rows 2..8 and several shapes.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    register_gguf_k_gemv_kernels,
    gguf_q5_k_gemv_bf16_bf16_out,
    gguf_q5_k_gemv_rowtile_bf16_bf16_out,
    gguf_q6_k_gemv_bf16_bf16_out,
    gguf_q6_k_gemv_rowtile_bf16_bf16_out,
    gguf_q8_0_gemv_bf16_bf16_out,
    gguf_q8_0_gemv_bf16_f32_out,
    gguf_q8_0_gemv_rowtile_bf16_bf16_out,
    gguf_q8_0_gemv_rowtile_bf16_f32_out,
)
from hipengine.kernels.registry import resolve

QK_K = 256


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def make_q8_0_weight(out_f: int, in_f: int) -> np.ndarray:
    """Valid Q8_0 bytes: per 32-wide block = fp16 scale + 32 int8 quants."""
    blocks = in_f // 32
    rng = np.random.default_rng(out_f * 13 + in_f)
    data = np.empty((out_f, blocks * 34), dtype=np.uint8)
    for c in range(out_f):
        for b in range(blocks):
            base = b * 34
            d = np.float16(0.01 * (1 + ((c + b) % 7)))
            data[c, base : base + 2] = np.asarray([d], dtype=np.float16).view(np.uint8)
            q = rng.integers(-127, 128, size=32, dtype=np.int8)
            data[c, base + 2 : base + 34] = q.view(np.uint8)
    return data


def _q8_0_dequant(weight_row: np.ndarray, in_f: int) -> np.ndarray:
    blocks = in_f // 32
    out = np.empty(in_f, dtype=np.float32)
    for b in range(blocks):
        base = b * 34
        d = weight_row[base : base + 2].view(np.float16).astype(np.float32)[0]
        q = weight_row[base + 2 : base + 34].view(np.int8).astype(np.float32)
        out[b * 32 : (b + 1) * 32] = d * q
    return out


def _cpu_ref_q8_0(x: np.ndarray, qw: np.ndarray, in_f: int, out_f: int) -> np.ndarray:
    w = np.stack([_q8_0_dequant(qw[c], in_f) for c in range(out_f)], axis=0)
    return x.astype(np.float32) @ w.T.astype(np.float32)


def _run(wrapper, x_host, qw, out_host):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )

    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    rows, in_f = x_host.shape
    out_f = out_host.shape[1]
    bufs = []
    try:
        xd = malloc(x_host.nbytes, runtime=runtime)
        qd = malloc(qw.nbytes, runtime=runtime)
        od = malloc(out_host.nbytes, runtime=runtime)
        bufs.extend((xd, qd, od))
        copy_host_to_device(xd, host_array_ptr(np.ascontiguousarray(x_host)), runtime=runtime)
        copy_host_to_device(qd, host_array_ptr(np.ascontiguousarray(qw)), runtime=runtime)
        wrapper(xd.ptr, qd.ptr, od.ptr, rows, in_f, out_f, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_host), od, runtime=runtime)
    finally:
        for b in reversed(bufs):
            free(b, runtime=runtime)
    return out_host


def test_gguf_k_rowtile_registry_binds() -> None:
    register_gguf_k_gemv_kernels()
    for quant in ("gguf_q8_0", "gguf_q5_k", "gguf_q6_k"):
        for variant in ("rowtile_bf16_bf16_out", "rowtile_bf16_f32_out", "rowtile_f32_f32_out"):
            assert callable(resolve(backend="hip_gfx1100", layer="linear", quant=quant, variant=variant))


_SHAPES = [(256, 16), (512, 48), (1024, 64)]
_ROWS = [2, 3, 4, 8]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("in_f,out_f", _SHAPES)
def test_q8_0_rowtile_bit_exact_and_oracle(rows, in_f, out_f) -> None:
    qw = make_q8_0_weight(out_f, in_f)
    rng = np.random.default_rng(7 + rows + in_f)
    x = rng.standard_normal((rows, in_f)).astype(np.float32) * 0.1
    xb = _bf16_bits(x).reshape(rows, in_f)

    ref = _run(gguf_q8_0_gemv_bf16_bf16_out, xb, qw, np.zeros((rows, out_f), np.uint16)).copy()
    got = _run(gguf_q8_0_gemv_rowtile_bf16_bf16_out, xb, qw, np.zeros((rows, out_f), np.uint16)).copy()
    np.testing.assert_array_equal(got, ref)  # bit-exact vs per-row

    ref32 = _run(gguf_q8_0_gemv_bf16_f32_out, xb, qw, np.zeros((rows, out_f), np.float32)).copy()
    got32 = _run(gguf_q8_0_gemv_rowtile_bf16_f32_out, xb, qw, np.zeros((rows, out_f), np.float32)).copy()
    np.testing.assert_array_equal(got32, ref32)
    cpu = _cpu_ref_q8_0(_bf16_to_f32(xb), qw, in_f, out_f)
    np.testing.assert_allclose(got32, cpu, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [2, 4, 8])
def test_q5k_q6k_rowtile_bit_exact_vs_per_row(rows) -> None:
    # Random raw bytes are valid Q5_K/Q6_K superblocks; only kernel equivalence
    # is asserted here (the Q8_0 case covers the CPU oracle).
    in_f, out_f = 512, 32
    rng = np.random.default_rng(100 + rows)
    qw5 = rng.integers(0, 256, size=(out_f, (in_f // QK_K) * 176), dtype=np.uint8)  # Q5_K block = 176 B
    qw6 = rng.integers(0, 256, size=(out_f, (in_f // QK_K) * 210), dtype=np.uint8)  # Q6_K block = 210 B
    x = rng.standard_normal((rows, in_f)).astype(np.float32) * 0.1
    xb = _bf16_bits(x).reshape(rows, in_f)

    ref5 = _run(gguf_q5_k_gemv_bf16_bf16_out, xb, qw5, np.zeros((rows, out_f), np.uint16)).copy()
    got5 = _run(gguf_q5_k_gemv_rowtile_bf16_bf16_out, xb, qw5, np.zeros((rows, out_f), np.uint16)).copy()
    np.testing.assert_array_equal(got5, ref5)

    ref6 = _run(gguf_q6_k_gemv_bf16_bf16_out, xb, qw6, np.zeros((rows, out_f), np.uint16)).copy()
    got6 = _run(gguf_q6_k_gemv_rowtile_bf16_bf16_out, xb, qw6, np.zeros((rows, out_f), np.uint16)).copy()
    np.testing.assert_array_equal(got6, ref6)
