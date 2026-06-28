"""RED-first correctness gate for the dense Q8_0 q8_1+dp4a GEMV (task #13).

The dense Q8_0 attention projections are the top GPU cost of the GGUF MTP
verifier. ``gguf_q8_0_dp4a_gemv_bf16_bf16_out`` quantizes the bf16 activation to
q8_1 and does the int8 dot (v_dot4) against raw Q8_0 weights. This test pins it
against (1) a q8_1-aware NumPy oracle (kernel wiring, tight) and (2) the
full-precision Q8_0 GEMV via the KL<=0.05 / top1>=0.90 quality gate.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q8_0_weight

_Q8_1_BLOCK = 32
_Q8_0_RAW_BYTES = 34
_Q8_1_BLOCK_BYTES = 36  # half d + half s + 32 int8


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="requires ROCm/libamdhip64.so")


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return (_to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)


def _quantize_q8_1_cpu(x: np.ndarray):
    rows, in_features = x.shape
    blocks = in_features // _Q8_1_BLOCK
    q = np.empty((rows, blocks, _Q8_1_BLOCK), dtype=np.int8)
    d = np.empty((rows, blocks), dtype=np.float32)
    for r in range(rows):
        for b in range(blocks):
            chunk = x[r, b * _Q8_1_BLOCK : (b + 1) * _Q8_1_BLOCK].astype(np.float32)
            amax = float(np.max(np.abs(chunk)))
            scale = 0.0 if amax == 0.0 else amax / 127.0
            d[r, b] = np.float16(scale).astype(np.float32)
            q[r, b] = 0 if amax == 0.0 else np.clip(np.round(chunk / scale), -128, 127).astype(np.int8)
    return q, d


def _q8_0_q8_1_oracle(x_bf16: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Mirror the device q8_1 + dp4a computation on the CPU."""
    rows, in_features = x_bf16.shape
    out_features = weight.shape[0]
    blocks = in_features // _Q8_1_BLOCK
    q8, d8 = _quantize_q8_1_cpu(x_bf16)
    out = np.zeros((rows, out_features), dtype=np.float32)
    for c in range(out_features):
        row_bytes = weight[c]
        for b in range(blocks):
            blk = row_bytes[b * _Q8_0_RAW_BYTES : (b + 1) * _Q8_0_RAW_BYTES]
            wd = blk[0:2].view(np.float16).astype(np.float32)[0]
            wq = blk[2:34].view(np.int8).astype(np.int32)
            dots = (q8[:, b, :].astype(np.int32) * wq[None, :]).sum(axis=1)  # [rows]
            out[:, c] += d8[:, b] * float(wd) * dots.astype(np.float32)
    return out


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    def logsm(z):
        z = z.astype(np.float64)
        s = z - z.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))
    lr, lc = logsm(ref), logsm(cand)
    return np.sum(np.exp(lr) * (lr - lc), axis=-1)


def test_q8_0_dp4a_gemv_matches_q8_1_oracle_and_quality_gate() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_bf16_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
        build_gguf_q8_0_dp4a_gemv,
        gguf_q8_0_dp4a_gemv_bf16_bf16_out,
    )

    rng = np.random.default_rng(20260628)
    rows, in_features, out_features = 4, 512, 48
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    x_bf16 = _bf16_round(x)
    weight = make_q8_0_weight(out_features, in_features)  # [out, blocks*34] uint8

    x_bits = _to_bf16_bits(x)  # device reads bf16
    q4_lib = build_gguf_q4_k_gemv(load=True)
    dp4a_lib = build_gguf_q8_0_dp4a_gemv(load=True)

    bufs = []

    def _dev(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return b

    try:
        x_buf = _dev(x_bits)
        w_buf = _dev(np.ascontiguousarray(weight, dtype=np.uint8))
        xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES); bufs.append(xq_buf)
        out_buf = malloc(rows * out_features * 2); bufs.append(out_buf)

        gguf_q4_k_quantize_bf16_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_lib)
        gguf_q8_0_dp4a_gemv_bf16_bf16_out(
            xq_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=dp4a_lib
        )
        out_bits = np.empty(rows * out_features, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out_bits), out_buf, out_bits.nbytes)
        out_dev = (out_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_features)
    finally:
        for b in reversed(bufs):
            free(b)

    # (1) kernel wiring: matches the q8_1-aware oracle (bf16-output tolerance).
    ref_q8 = _q8_0_q8_1_oracle(x_bf16, weight)
    rel_l2 = float(np.linalg.norm(out_dev - ref_q8) / (np.linalg.norm(ref_q8) + 1e-8))
    assert rel_l2 <= 2e-2, f"kernel vs q8_1 oracle rel_l2={rel_l2}"

    # (2) quality gate vs full-precision Q8_0 GEMV.
    ref_full = gguf_quant_gemv(x_bf16, weight, GGMLQuantizationType.Q8_0)
    kl = _softmax_kl(ref_full, out_dev)
    assert float(np.mean(kl)) <= 0.05, f"KL={float(np.mean(kl))}"
    top1 = float(np.mean(np.argmax(ref_full, axis=-1) == np.argmax(out_dev, axis=-1)))
    assert top1 >= 0.90, f"top1={top1}"
