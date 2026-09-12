"""RED-first numerical gate for the device-resident MTP draft MoE-down + combine.

Task #3 (sub-win A): the resident GGUF MTP draft runner used to read the
selected-expert indices + routing weights back to host and run a Python
per-expert down loop (silu_mul + Q5_K down GEMV + scale + add per expert) plus a
separate shared-gate combine.  ``apply_moe_down_combine`` replaces that with the
verifier's proven device-resident sequence:

    silu_mul_separate_out_bf16            (one launch over all top_k)
    gguf_q5_k_selected_gemv_bf16_bf16_out (one selected-down GEMV, raw Q5_K)
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w (one combine)

This test pins the numerics of that sequence against a NumPy/cpu-reference oracle
without needing the 35B GGUF model.  It is bf16-aware: the device path rounds the
intermediates to bf16 (matching the verifier), so the oracle rounds the same way.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q5_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="requires ROCm/libamdhip64.so")


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round f32 -> bf16 (round to nearest even) and return the uint16 bit pattern."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    bias = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + bias) >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _bf16_round(x: np.ndarray) -> np.ndarray:
    return _bf16_bits_to_f32(_to_bf16_bits(x))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def test_apply_moe_down_combine_matches_cpu_reference() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.convert.cast import build_cast
    from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import build_paro_silu
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import build_gguf_k_gemv
    from hipengine.speculative.mtp_resident_draft import apply_moe_down_combine

    rng = np.random.default_rng(20260628)
    top_k = 2
    num_experts = 4
    inter = 256  # Q5_K block multiple
    hidden = 128

    gate = rng.standard_normal((top_k, inter)).astype(np.float32)
    up = rng.standard_normal((top_k, inter)).astype(np.float32)
    selected = np.array([3, 1], dtype=np.int64)
    routing = rng.uniform(0.1, 1.0, size=top_k).astype(np.float32)
    down_exps = np.stack(
        [make_q5_k_weight(hidden, inter) for _ in range(num_experts)], axis=0
    )  # [E, hidden, bytes]
    shared_out = rng.standard_normal(hidden).astype(np.float32)
    shared_gate_logit = rng.standard_normal(1).astype(np.float32)
    residual = rng.standard_normal(hidden).astype(np.float32)

    # ---- CPU reference (bf16-aware, mirrors the device rounding) ----
    gate_bf = _bf16_round(gate)
    up_bf = _bf16_round(up)
    inter_ref = _bf16_round(_silu(gate_bf) * up_bf)  # device down reads bf16 inter
    down_bf = np.empty((top_k, hidden), dtype=np.float32)
    for k in range(top_k):
        row = gguf_quant_gemv(
            inter_ref[k : k + 1], down_exps[int(selected[k])], GGMLQuantizationType.Q5_K
        )[0]
        down_bf[k] = _bf16_round(row)
    acc = _bf16_round((down_bf * routing[:, None]).sum(axis=0))
    gate_scale = 1.0 / (1.0 + np.exp(-float(shared_gate_logit[0])))
    out_ref = _bf16_round(
        _bf16_round(residual) + acc + gate_scale * _bf16_round(shared_out)
    )

    # ---- device path ----
    silu_lib = build_paro_silu(load=True)
    k_lib = build_gguf_k_gemv(load=True)
    combine_lib = build_paro_combine(load=True)
    cast_lib = build_cast(load=True)

    gate_bits = _to_bf16_bits(gate)
    up_bits = _to_bf16_bits(up)
    bufs: list = []

    def _dev(arr: np.ndarray):
        buf = malloc(arr.nbytes)
        bufs.append(buf)
        copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
        return buf

    try:
        gate_buf = _dev(gate_bits)
        up_buf = _dev(up_bits)
        selected_buf = _dev(selected)
        routing_buf = _dev(routing)
        down_buf = _dev(np.ascontiguousarray(down_exps, dtype=np.uint8))
        shared_buf = _dev(shared_out)
        gate_logit_buf = _dev(shared_gate_logit)
        residual_buf = _dev(residual)

        inter_bf16 = malloc(top_k * inter * 2); bufs.append(inter_bf16)
        down_out_bf16 = malloc(top_k * hidden * 2); bufs.append(down_out_bf16)
        attended_bf16 = malloc(hidden * 2); bufs.append(attended_bf16)
        shared_bf16 = malloc(hidden * 2); bufs.append(shared_bf16)
        ffn_out_bf16 = malloc(hidden * 2); bufs.append(ffn_out_bf16)
        ffn_out_f32 = malloc(hidden * 4); bufs.append(ffn_out_f32)

        apply_moe_down_combine(
            gate_bf16_ptr=gate_buf.ptr,
            up_bf16_ptr=up_buf.ptr,
            selected_ptr=selected_buf.ptr,
            routing_ptr=routing_buf.ptr,
            shared_out_ptr=shared_buf.ptr,
            shared_gate_logit_ptr=gate_logit_buf.ptr,
            residual_ptr=residual_buf.ptr,
            down_exps_ptr=down_buf.ptr,
            inter_bf16_ptr=inter_bf16.ptr,
            down_out_bf16_ptr=down_out_bf16.ptr,
            attended_bf16_ptr=attended_bf16.ptr,
            shared_bf16_ptr=shared_bf16.ptr,
            ffn_out_bf16_ptr=ffn_out_bf16.ptr,
            ffn_out_f32_ptr=ffn_out_f32.ptr,
            top_k=top_k,
            inter=inter,
            hidden=hidden,
            num_experts=num_experts,
            silu_lib=silu_lib,
            k_lib=k_lib,
            combine_lib=combine_lib,
            cast_lib=cast_lib,
        )

        out_dev = np.empty(hidden, dtype=np.float32)
        copy_device_to_host(host_array_ptr(out_dev), ffn_out_f32, out_dev.nbytes)
    finally:
        for buf in reversed(bufs):
            free(buf)

    rel_l2 = float(np.linalg.norm(out_dev - out_ref) / (np.linalg.norm(out_ref) + 1e-8))
    assert rel_l2 <= 2e-2, f"rel_l2={rel_l2} too high\nref={out_ref[:8]}\ndev={out_dev[:8]}"
