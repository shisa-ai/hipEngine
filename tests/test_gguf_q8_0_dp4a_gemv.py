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


def test_q8_0_dp4a_single_rowtile_matches_q8_1_oracle_and_quality_gate() -> None:
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
        gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out,
    )

    rng = np.random.default_rng(20260702)
    rows, in_features, out_features = 5, 512, 48
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    x_bf16 = _bf16_round(x)
    weight = make_q8_0_weight(out_features, in_features)

    x_bits = _to_bf16_bits(x)
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
        gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out(
            xq_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=dp4a_lib
        )
        out_bits = np.empty(rows * out_features, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out_bits), out_buf, out_bits.nbytes)
        out_dev = (out_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_features)
    finally:
        for b in reversed(bufs):
            free(b)

    ref_q8 = _q8_0_q8_1_oracle(x_bf16, weight)
    rel_l2 = float(np.linalg.norm(out_dev - ref_q8) / (np.linalg.norm(ref_q8) + 1e-8))
    assert rel_l2 <= 2e-2, f"rowtile kernel vs q8_1 oracle rel_l2={rel_l2}"

    ref_full = gguf_quant_gemv(x_bf16, weight, GGMLQuantizationType.Q8_0)
    kl = _softmax_kl(ref_full, out_dev)
    assert float(np.mean(kl)) <= 0.05, f"KL={float(np.mean(kl))}"
    top1 = float(np.mean(np.argmax(ref_full, axis=-1) == np.argmax(out_dev, axis=-1)))
    assert top1 >= 0.90, f"top1={top1}"


def test_q8_0_dp4a_f32_quantizer_single_rowtile_matches_q8_1_oracle_and_quality_gate() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_f32_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
        build_gguf_q8_0_dp4a_gemv,
        gguf_q8_0_dp4a_rowtile4_gemv_f32_f32_out,
    )

    rng = np.random.default_rng(20260703)
    rows, in_features, out_features = 3, 512, 48
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    weight = make_q8_0_weight(out_features, in_features)

    q4_lib = build_gguf_q4_k_gemv(load=True)
    dp4a_lib = build_gguf_q8_0_dp4a_gemv(load=True)

    bufs = []

    def _dev(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return b

    try:
        x_buf = _dev(x)
        w_buf = _dev(np.ascontiguousarray(weight, dtype=np.uint8))
        xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES); bufs.append(xq_buf)
        out_buf = malloc(rows * out_features * 4); bufs.append(out_buf)

        gguf_q4_k_quantize_f32_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_lib)
        gguf_q8_0_dp4a_rowtile4_gemv_f32_f32_out(
            xq_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=dp4a_lib
        )
        out_dev = np.empty((rows, out_features), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out_dev), out_buf, out_dev.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    ref_q8 = _q8_0_q8_1_oracle(x, weight)
    rel_l2 = float(np.linalg.norm(out_dev - ref_q8) / (np.linalg.norm(ref_q8) + 1e-8))
    assert rel_l2 <= 5e-4, f"f32 rowtile kernel vs q8_1 oracle rel_l2={rel_l2}"

    ref_full = gguf_quant_gemv(x, weight, GGMLQuantizationType.Q8_0)
    kl = _softmax_kl(ref_full, out_dev)
    assert float(np.mean(kl)) <= 0.05, f"KL={float(np.mean(kl))}"
    top1 = float(np.mean(np.argmax(ref_full, axis=-1) == np.argmax(out_dev, axis=-1)))
    assert top1 >= 0.90, f"top1={top1}"


def test_q8_0_dp4a_f32_quantizer_dual_split_float_output_matches_q8_1_oracle() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_f32_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
        build_gguf_q8_0_dp4a_gemv,
        gguf_q8_0_dp4a_dual_split_rowtile4_gemv_f32_f32_out,
    )

    rng = np.random.default_rng(20260704)
    rows, in_features, out_a, out_b = 3, 512, 40, 24
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    weight_a = make_q8_0_weight(out_a, in_features)
    weight_b = make_q8_0_weight(out_b, in_features)

    q4_lib = build_gguf_q4_k_gemv(load=True)
    dp4a_lib = build_gguf_q8_0_dp4a_gemv(load=True)

    bufs = []

    def _dev(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return b

    try:
        x_buf = _dev(x)
        wa_buf = _dev(np.ascontiguousarray(weight_a, dtype=np.uint8))
        wb_buf = _dev(np.ascontiguousarray(weight_b, dtype=np.uint8))
        xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES); bufs.append(xq_buf)
        out_a_buf = malloc(rows * out_a * 4); bufs.append(out_a_buf)
        out_b_buf = malloc(rows * out_b * 4); bufs.append(out_b_buf)

        gguf_q4_k_quantize_f32_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_lib)
        gguf_q8_0_dp4a_dual_split_rowtile4_gemv_f32_f32_out(
            xq_buf.ptr,
            wa_buf.ptr,
            wb_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=dp4a_lib,
        )
        out_a_dev = np.empty((rows, out_a), dtype=np.float32)
        out_b_dev = np.empty((rows, out_b), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out_a_dev), out_a_buf, out_a_dev.nbytes)
        copy_device_to_host(host_array_ptr(out_b_dev), out_b_buf, out_b_dev.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    ref_a_q8 = _q8_0_q8_1_oracle(x, weight_a)
    ref_b_q8 = _q8_0_q8_1_oracle(x, weight_b)
    rel_l2_a = float(np.linalg.norm(out_a_dev - ref_a_q8) / (np.linalg.norm(ref_a_q8) + 1e-8))
    rel_l2_b = float(np.linalg.norm(out_b_dev - ref_b_q8) / (np.linalg.norm(ref_b_q8) + 1e-8))
    assert rel_l2_a <= 5e-4, f"f32 dual A kernel vs q8_1 oracle rel_l2={rel_l2_a}"
    assert rel_l2_b <= 5e-4, f"f32 dual B kernel vs q8_1 oracle rel_l2={rel_l2_b}"


def test_q8_0_dp4a_dual_split_rowtile_matches_q8_1_oracle_and_quality_gate() -> None:
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
        gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out,
    )

    rng = np.random.default_rng(20260701)
    rows, in_features, out_a, out_b = 5, 512, 40, 24
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    x_bf16 = _bf16_round(x)
    weight_a = make_q8_0_weight(out_a, in_features)
    weight_b = make_q8_0_weight(out_b, in_features)

    x_bits = _to_bf16_bits(x)
    q4_lib = build_gguf_q4_k_gemv(load=True)
    dp4a_lib = build_gguf_q8_0_dp4a_gemv(load=True)

    bufs = []

    def _dev(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return b

    try:
        x_buf = _dev(x_bits)
        wa_buf = _dev(np.ascontiguousarray(weight_a, dtype=np.uint8))
        wb_buf = _dev(np.ascontiguousarray(weight_b, dtype=np.uint8))
        xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES); bufs.append(xq_buf)
        out_a_buf = malloc(rows * out_a * 2); bufs.append(out_a_buf)
        out_b_buf = malloc(rows * out_b * 2); bufs.append(out_b_buf)

        gguf_q4_k_quantize_bf16_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_lib)
        gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out(
            xq_buf.ptr,
            wa_buf.ptr,
            wb_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=dp4a_lib,
        )
        out_a_bits = np.empty(rows * out_a, dtype=np.uint16)
        out_b_bits = np.empty(rows * out_b, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out_a_bits), out_a_buf, out_a_bits.nbytes)
        copy_device_to_host(host_array_ptr(out_b_bits), out_b_buf, out_b_bits.nbytes)
        out_a_dev = (out_a_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_a)
        out_b_dev = (out_b_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_b)
    finally:
        for b in reversed(bufs):
            free(b)

    ref_a_q8 = _q8_0_q8_1_oracle(x_bf16, weight_a)
    ref_b_q8 = _q8_0_q8_1_oracle(x_bf16, weight_b)
    rel_l2_a = float(np.linalg.norm(out_a_dev - ref_a_q8) / (np.linalg.norm(ref_a_q8) + 1e-8))
    rel_l2_b = float(np.linalg.norm(out_b_dev - ref_b_q8) / (np.linalg.norm(ref_b_q8) + 1e-8))
    assert rel_l2_a <= 2e-2, f"pair A kernel vs q8_1 oracle rel_l2={rel_l2_a}"
    assert rel_l2_b <= 2e-2, f"pair B kernel vs q8_1 oracle rel_l2={rel_l2_b}"

    ref_a_full = gguf_quant_gemv(x_bf16, weight_a, GGMLQuantizationType.Q8_0)
    ref_b_full = gguf_quant_gemv(x_bf16, weight_b, GGMLQuantizationType.Q8_0)
    ref_full = np.concatenate([ref_a_full, ref_b_full], axis=1)
    out_dev = np.concatenate([out_a_dev, out_b_dev], axis=1)
    kl = _softmax_kl(ref_full, out_dev)
    assert float(np.mean(kl)) <= 0.05, f"KL={float(np.mean(kl))}"
    top1 = float(np.mean(np.argmax(ref_full, axis=-1) == np.argmax(out_dev, axis=-1)))
    assert top1 >= 0.90, f"top1={top1}"


def test_q8_0_dp4a_triple_split_rowtile_matches_q8_1_oracle_and_quality_gate() -> None:
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
        gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out,
    )

    rng = np.random.default_rng(20260703)
    rows, in_features, out_a, out_b, out_c = 5, 512, 40, 24, 32
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    x_bf16 = _bf16_round(x)
    weight_a = make_q8_0_weight(out_a, in_features)
    weight_b = make_q8_0_weight(out_b, in_features)
    weight_c = make_q8_0_weight(out_c, in_features)

    x_bits = _to_bf16_bits(x)
    q4_lib = build_gguf_q4_k_gemv(load=True)
    dp4a_lib = build_gguf_q8_0_dp4a_gemv(load=True)

    bufs = []

    def _dev(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return b

    try:
        x_buf = _dev(x_bits)
        wa_buf = _dev(np.ascontiguousarray(weight_a, dtype=np.uint8))
        wb_buf = _dev(np.ascontiguousarray(weight_b, dtype=np.uint8))
        wc_buf = _dev(np.ascontiguousarray(weight_c, dtype=np.uint8))
        xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES); bufs.append(xq_buf)
        out_a_buf = malloc(rows * out_a * 2); bufs.append(out_a_buf)
        out_b_buf = malloc(rows * out_b * 2); bufs.append(out_b_buf)
        out_c_buf = malloc(rows * out_c * 2); bufs.append(out_c_buf)

        gguf_q4_k_quantize_bf16_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_lib)
        gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out(
            xq_buf.ptr,
            wa_buf.ptr,
            wb_buf.ptr,
            wc_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            out_c_buf.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            out_c,
            library=dp4a_lib,
        )
        out_a_bits = np.empty(rows * out_a, dtype=np.uint16)
        out_b_bits = np.empty(rows * out_b, dtype=np.uint16)
        out_c_bits = np.empty(rows * out_c, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(out_a_bits), out_a_buf, out_a_bits.nbytes)
        copy_device_to_host(host_array_ptr(out_b_bits), out_b_buf, out_b_bits.nbytes)
        copy_device_to_host(host_array_ptr(out_c_bits), out_c_buf, out_c_bits.nbytes)
        out_a_dev = (out_a_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_a)
        out_b_dev = (out_b_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_b)
        out_c_dev = (out_c_bits.astype(np.uint32) << 16).view(np.float32).reshape(rows, out_c)
    finally:
        for b in reversed(bufs):
            free(b)

    ref_a_q8 = _q8_0_q8_1_oracle(x_bf16, weight_a)
    ref_b_q8 = _q8_0_q8_1_oracle(x_bf16, weight_b)
    ref_c_q8 = _q8_0_q8_1_oracle(x_bf16, weight_c)
    rel_l2_a = float(np.linalg.norm(out_a_dev - ref_a_q8) / (np.linalg.norm(ref_a_q8) + 1e-8))
    rel_l2_b = float(np.linalg.norm(out_b_dev - ref_b_q8) / (np.linalg.norm(ref_b_q8) + 1e-8))
    rel_l2_c = float(np.linalg.norm(out_c_dev - ref_c_q8) / (np.linalg.norm(ref_c_q8) + 1e-8))
    assert rel_l2_a <= 2e-2, f"triple A kernel vs q8_1 oracle rel_l2={rel_l2_a}"
    assert rel_l2_b <= 2e-2, f"triple B kernel vs q8_1 oracle rel_l2={rel_l2_b}"
    assert rel_l2_c <= 2e-2, f"triple C kernel vs q8_1 oracle rel_l2={rel_l2_c}"

    ref_a_full = gguf_quant_gemv(x_bf16, weight_a, GGMLQuantizationType.Q8_0)
    ref_b_full = gguf_quant_gemv(x_bf16, weight_b, GGMLQuantizationType.Q8_0)
    ref_c_full = gguf_quant_gemv(x_bf16, weight_c, GGMLQuantizationType.Q8_0)
    ref_full = np.concatenate([ref_a_full, ref_b_full, ref_c_full], axis=1)
    out_dev = np.concatenate([out_a_dev, out_b_dev, out_c_dev], axis=1)
    kl = _softmax_kl(ref_full, out_dev)
    assert float(np.mean(kl)) <= 0.05, f"KL={float(np.mean(kl))}"
    top1 = float(np.mean(np.argmax(ref_full, axis=-1) == np.argmax(out_dev, axis=-1)))
    assert top1 >= 0.90, f"top1={top1}"


def test_q8_0_dp4a_f32_quantizer_triple_split_float_output_matches_q8_1_oracle() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        build_gguf_q4_k_gemv,
        gguf_q4_k_quantize_f32_q8_1,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dp4a_gemv import (
        build_gguf_q8_0_dp4a_gemv,
        gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out,
    )

    rng = np.random.default_rng(20260705)
    rows, in_features, out_a, out_b, out_c = 3, 512, 40, 24, 32
    blocks = in_features // _Q8_1_BLOCK

    x = rng.standard_normal((rows, in_features)).astype(np.float32)
    weight_a = make_q8_0_weight(out_a, in_features)
    weight_b = make_q8_0_weight(out_b, in_features)
    weight_c = make_q8_0_weight(out_c, in_features)

    q4_lib = build_gguf_q4_k_gemv(load=True)
    dp4a_lib = build_gguf_q8_0_dp4a_gemv(load=True)

    bufs = []

    def _dev(arr):
        b = malloc(arr.nbytes); bufs.append(b)
        copy_host_to_device(b, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return b

    try:
        x_buf = _dev(x)
        wa_buf = _dev(np.ascontiguousarray(weight_a, dtype=np.uint8))
        wb_buf = _dev(np.ascontiguousarray(weight_b, dtype=np.uint8))
        wc_buf = _dev(np.ascontiguousarray(weight_c, dtype=np.uint8))
        xq_buf = malloc(rows * blocks * _Q8_1_BLOCK_BYTES); bufs.append(xq_buf)
        out_a_buf = malloc(rows * out_a * 4); bufs.append(out_a_buf)
        out_b_buf = malloc(rows * out_b * 4); bufs.append(out_b_buf)
        out_c_buf = malloc(rows * out_c * 4); bufs.append(out_c_buf)

        gguf_q4_k_quantize_f32_q8_1(x_buf.ptr, xq_buf.ptr, rows, in_features, library=q4_lib)
        gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out(
            xq_buf.ptr,
            wa_buf.ptr,
            wb_buf.ptr,
            wc_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            out_c_buf.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            out_c,
            library=dp4a_lib,
        )
        out_a_dev = np.empty((rows, out_a), dtype=np.float32)
        out_b_dev = np.empty((rows, out_b), dtype=np.float32)
        out_c_dev = np.empty((rows, out_c), dtype=np.float32)
        copy_device_to_host(host_array_ptr(out_a_dev), out_a_buf, out_a_dev.nbytes)
        copy_device_to_host(host_array_ptr(out_b_dev), out_b_buf, out_b_dev.nbytes)
        copy_device_to_host(host_array_ptr(out_c_dev), out_c_buf, out_c_dev.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    ref_a_q8 = _q8_0_q8_1_oracle(x, weight_a)
    ref_b_q8 = _q8_0_q8_1_oracle(x, weight_b)
    ref_c_q8 = _q8_0_q8_1_oracle(x, weight_c)
    rel_l2_a = float(np.linalg.norm(out_a_dev - ref_a_q8) / (np.linalg.norm(ref_a_q8) + 1e-8))
    rel_l2_b = float(np.linalg.norm(out_b_dev - ref_b_q8) / (np.linalg.norm(ref_b_q8) + 1e-8))
    rel_l2_c = float(np.linalg.norm(out_c_dev - ref_c_q8) / (np.linalg.norm(ref_c_q8) + 1e-8))
    assert rel_l2_a <= 5e-4, f"f32 triple A kernel vs q8_1 oracle rel_l2={rel_l2_a}"
    assert rel_l2_b <= 5e-4, f"f32 triple B kernel vs q8_1 oracle rel_l2={rel_l2_b}"
    assert rel_l2_c <= 5e-4, f"f32 triple C kernel vs q8_1 oracle rel_l2={rel_l2_c}"
