"""Correctness tests for the qwen35 GGUF GDN prefill chain (P9.A2 / task #18).

These tests build synthetic and Qwen3.6-35B-A3B-shaped inputs and compare the
three GDN prefill kernel paths registered for ``gguf_qwen35`` against a CPU
reference assembled from the in-tree ``gdn_prefill_recurrent_segments``
oracle.

Paths under test:

1. ``qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order``
   (single fused kernel; legacy path)
2. ``qwen35_linear_attn_prefill_prepare_f32_bf16``
   -> ``qwen35_gdn_prefill_recurrent_k2_f32``
   -> ``qwen35_gdn_prefill_rmsnorm_gate_bf16``
   (chained path; rows < segment threshold)
3. ``qwen35_linear_attn_prefill_prepare_f32_bf16``
   -> ``qwen35_gdn_prefill_recurrent_segments_k2_f32`` (segments=1)
   -> ``qwen35_gdn_prefill_rmsnorm_gate_bf16``
   (chained path; rows >= segment threshold)

The CPU oracle replays the prepare math (q/k L2 normalization with rsqrt
epsilon, value passthrough, sigmoid beta, exp/softplus decay), the recurrent
update via :func:`hipengine.kernels.cpu_reference.gdn_prefill_recurrent_segments`,
and the final RMSNorm + sigmoid gate. Tolerances reflect the BF16 boundary on
``a``, ``b``, ``gate``, and the final BF16 output: state is F32, output is
BF16-rounded.

Coverage:

* No-GPU: registry lookups for ``gguf_qwen35`` aliases.
* Synthetic small shape (8 tokens, 2 v_heads, 1 k_head, 128/128 dims): all
  three paths vs CPU oracle, on state and output.
* Qwen3.6-35B-A3B shape (64 tokens, 32 v_heads, 16 k_heads, 128/128 dims):
  all three paths vs CPU oracle.
* Segment-boundary cases: 255, 256, 257 tokens.

The end-to-end qwen35moe 512/128 KL/top-1 gate against the legacy row-GEMV
reference is exercised by ``scripts/qwen35_gguf_gdn_correctness_probe.py``
(see WORKLOG); pytest stays scoped to kernel-level synthetic correctness.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gdn_prefill_recurrent_segments
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out,
    qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16,
    qwen35_gdn_prefill_recurrent_decode_order_exact_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32,
    qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32,
    qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32,
    qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32,
    qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32,
    qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
    qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32,
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_f32,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
    qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16,
    qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16,
    qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16,
    register_qwen35_linear_attn_gdn_kernels,
)
from hipengine.kernels.registry import resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


# ---------------------------------------------------------------------------
# BF16 helpers (RNE, matching the in-kernel scalar_to_float_qwen35 conversion).
# ---------------------------------------------------------------------------


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """Round float32 array to BF16 with RNE; return uint16 bit pattern."""

    f32 = np.asarray(arr, dtype=np.float32, order="C")
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    # round-to-nearest-even: add 0x7FFF and lsb of the kept half
    lsb = (u32 >> 16) & 0x1
    rounded = (u32 + 0x7FFF + lsb) >> 16
    rounded = rounded.astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    """Convert uint16 BF16 bit pattern to float32."""

    u16 = np.asarray(arr, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(u16.shape).copy()


def _bf16_round_inplace(arr: np.ndarray) -> np.ndarray:
    """Round float32 array through BF16 quantization (returns float32)."""

    return _bf16_u16_to_f32(_f32_to_bf16_u16(arr))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float64)


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU(x) = x * sigmoid(x), matching the kernel-side ``silu_f32``."""

    return (x.astype(np.float64) * _sigmoid(x)).astype(np.float64)


def _softplus(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    return np.where(x64 > 20.0, x64, np.log1p(np.exp(x64)))


# ---------------------------------------------------------------------------
# CPU oracle.
# ---------------------------------------------------------------------------


def _cpu_prepare(
    conv_out_f32: np.ndarray,
    a_u16: np.ndarray,
    b_u16: np.ndarray,
    dt_bias_f32: np.ndarray,
    a_log_f32: np.ndarray,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    qk_eps: float = 1.0e-6,
):
    tokens = conv_out_f32.shape[0]
    key_offset = num_k_heads * head_k_dim
    value_offset = 2 * num_k_heads * head_k_dim

    query = np.zeros((tokens, num_v_heads, head_k_dim), dtype=np.float32)
    key = np.zeros((tokens, num_v_heads, head_k_dim), dtype=np.float32)
    value = np.zeros((tokens, num_v_heads, head_v_dim), dtype=np.float32)
    beta = np.zeros((tokens, num_v_heads), dtype=np.float32)
    decay = np.zeros((tokens, num_v_heads), dtype=np.float32)

    a_f32 = _bf16_u16_to_f32(a_u16)
    b_f32 = _bf16_u16_to_f32(b_u16)

    for token in range(tokens):
        conv_row = conv_out_f32[token]
        for v_head in range(num_v_heads):
            # Match llama.cpp/GGML_OP_GATED_DELTA_NET for Qwen3.5: K heads are
            # interleaved across V heads, not grouped by contiguous repeats.
            k_head = v_head % num_k_heads
            q_base = k_head * head_k_dim
            k_base = key_offset + k_head * head_k_dim
            v_base = value_offset + v_head * head_v_dim
            scalar_idx = token * num_v_heads + v_head

            q_slice = conv_row[q_base : q_base + head_k_dim]
            k_slice = conv_row[k_base : k_base + head_k_dim]
            v_slice = conv_row[v_base : v_base + head_v_dim]

            q_sum = float(np.sum(q_slice.astype(np.float32) ** 2))
            k_sum = float(np.sum(k_slice.astype(np.float32) ** 2))
            q_scale = 1.0 / math.sqrt(q_sum + qk_eps) / math.sqrt(head_k_dim)
            k_scale = 1.0 / math.sqrt(k_sum + qk_eps)

            query[token, v_head] = (q_slice * q_scale).astype(np.float32)
            key[token, v_head] = (k_slice * k_scale).astype(np.float32)
            value[token, v_head] = v_slice.astype(np.float32)

            beta[token, v_head] = np.float32(_sigmoid(np.float32(b_f32[scalar_idx])))
            decay[token, v_head] = np.float32(
                np.exp(
                    -np.exp(a_log_f32[v_head])
                    * _softplus(np.float32(a_f32[scalar_idx] + dt_bias_f32[v_head]))
                )
            )
    return query, key, value, beta, decay


def _cpu_rmsnorm_gate(
    recurrent_out: np.ndarray,
    gate_u16: np.ndarray,
    norm_weight_f32: np.ndarray,
    eps: float,
) -> np.ndarray:
    tokens, num_v_heads, head_v_dim = recurrent_out.shape
    gate_f32 = _bf16_u16_to_f32(gate_u16).reshape(tokens, num_v_heads, head_v_dim)
    out = np.zeros_like(recurrent_out, dtype=np.float32)
    for token in range(tokens):
        for v_head in range(num_v_heads):
            r = recurrent_out[token, v_head].astype(np.float32)
            square_sum = float(np.sum(r * r))
            rms_scale = 1.0 / math.sqrt(square_sum / head_v_dim + eps)
            # The kernel applies SiLU(gate) = gate * sigmoid(gate), not a bare
            # sigmoid. See ``qwen35_gdn_prefill_rmsnorm_gate_bf16_kernel`` in
            # ``hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip``.
            gate_v = _silu(gate_f32[token, v_head]).astype(np.float32)
            out[token, v_head] = (r * rms_scale * norm_weight_f32 * gate_v).astype(np.float32)
    return _bf16_round_inplace(out)


def _cpu_full_chain(
    inputs: "_GDNInputs",
    rms_norm_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full prepare -> recurrent -> rmsnorm_gate chain on CPU."""

    query, key, value, beta, decay = _cpu_prepare(
        inputs.conv_out_f32,
        inputs.a_u16,
        inputs.b_u16,
        inputs.dt_bias_f32,
        inputs.a_log_f32,
        num_k_heads=inputs.num_k_heads,
        num_v_heads=inputs.num_v_heads,
        head_k_dim=inputs.head_k_dim,
        head_v_dim=inputs.head_v_dim,
    )
    cu_seqlens = np.array([0, inputs.tokens], dtype=np.int64)
    state_indices = np.array([0], dtype=np.int64)
    recurrent_out, final_state = gdn_prefill_recurrent_segments(
        query,
        key,
        value,
        beta,
        decay,
        inputs.init_state_f32[np.newaxis, ...],
        cu_seqlens,
        state_indices,
    )
    out_bf16 = _cpu_rmsnorm_gate(
        recurrent_out, inputs.gate_u16, inputs.norm_weight_f32, rms_norm_eps
    )
    return out_bf16, final_state[0]


# ---------------------------------------------------------------------------
# Input fixtures.
# ---------------------------------------------------------------------------


class _GDNInputs:
    def __init__(
        self,
        *,
        tokens: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        seed: int = 0,
    ) -> None:
        if num_v_heads % num_k_heads != 0:
            raise ValueError("num_v_heads must divide by num_k_heads")
        if head_k_dim != 128:
            raise ValueError("k2 GDN kernels require head_k_dim == 128")
        rng = np.random.default_rng(seed)
        self.tokens = tokens
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        qkv_width = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        # Realistic magnitudes: conv_out post-SiLU, BF16-rounded to mimic
        # how the real prefill chain feeds GDN.
        conv_out = rng.normal(0.0, 0.5, size=(tokens, qkv_width)).astype(np.float32)
        self.conv_out_f32 = _bf16_round_inplace(conv_out)
        # a, b are BF16-quantized ssm projections.
        a_f32 = rng.normal(0.0, 0.3, size=(tokens * num_v_heads,)).astype(np.float32)
        b_f32 = rng.normal(0.0, 0.3, size=(tokens * num_v_heads,)).astype(np.float32)
        self.a_u16 = _f32_to_bf16_u16(a_f32)
        self.b_u16 = _f32_to_bf16_u16(b_f32)
        # dt_bias and a_log live as F32 scalars per v_head.
        self.dt_bias_f32 = rng.normal(0.0, 0.1, size=(num_v_heads,)).astype(np.float32)
        # a_log values cluster around exp(-1) magnitude so decay stays in (0, 1).
        self.a_log_f32 = rng.normal(0.0, 0.5, size=(num_v_heads,)).astype(np.float32)
        # gate (linear_z) is per-token-per-v_head-per-head_v_dim, BF16.
        gate_f32 = rng.normal(0.0, 0.3, size=(tokens, num_v_heads, head_v_dim)).astype(np.float32)
        self.gate_u16 = _f32_to_bf16_u16(gate_f32)
        # norm_weight is per-head_v_dim F32 (positive small).
        self.norm_weight_f32 = (
            0.8 + 0.2 * rng.normal(0.0, 1.0, size=(head_v_dim,)).astype(np.float32)
        )
        # Initial recurrent state matches what the runner zero-fills on first call.
        # Seed it with small non-zero values so the test exercises the multiply path.
        self.init_state_f32 = rng.normal(
            0.0, 0.05, size=(num_v_heads, head_k_dim, head_v_dim)
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# GPU runners.
# ---------------------------------------------------------------------------


class _Buf:
    """Tiny RAII wrapper around the DeviceBuffer API used by the WMMA tests."""

    def __init__(self, nbytes: int) -> None:
        self.buffer = malloc(nbytes)
        self.nbytes = nbytes

    @property
    def ptr(self) -> int:
        return self.buffer.ptr

    def free(self) -> None:
        if self.buffer is not None:
            free(self.buffer)
            self.buffer = None


def _to_device(arr: np.ndarray) -> _Buf:
    arr = np.ascontiguousarray(arr)
    buf = _Buf(arr.nbytes)
    copy_host_to_device(buf.buffer, host_array_ptr(arr), arr.nbytes)
    return buf


def _from_device(buf: _Buf, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(out), buf.buffer, out.nbytes)
    return out


def _run_decode_order_bf16(
    inputs: _GDNInputs, rms_norm_eps: float
) -> tuple[np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    gate = _to_device(inputs.gate_u16)
    a = _to_device(inputs.a_u16)
    b = _to_device(inputs.b_u16)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    state = _to_device(inputs.init_state_f32)
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * np.dtype(np.uint16).itemsize)
    try:
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order(
            conv_out.ptr,
            gate.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            state.ptr,
            out.ptr,
            rms_norm_eps,
            inputs.tokens,
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        out_u16 = _from_device(out, out_shape, np.uint16)
        state_f32 = _from_device(
            state,
            (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
            np.float32,
        )
        return _bf16_u16_to_f32(out_u16), state_f32
    finally:
        for buf in (conv_out, gate, a, b, dt_bias, a_log, norm_weight, state, out):
            buf.free()


def _run_decode_order_state_rows(
    inputs: _GDNInputs, rms_norm_eps: float, *, no_copy: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    gate = _to_device(inputs.gate_u16)
    a = _to_device(inputs.a_u16)
    b = _to_device(inputs.b_u16)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    state = _to_device(inputs.init_state_f32)
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    state_shape = (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim)
    rows_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * np.dtype(np.uint16).itemsize)
    state_rows = _Buf(int(np.prod(rows_shape)) * np.dtype(np.float32).itemsize)
    try:
        fn = (
            qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows_no_copy
            if no_copy
            else qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows
        )
        fn(
            conv_out.ptr,
            gate.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            state.ptr,
            state_rows.ptr,
            out.ptr,
            rms_norm_eps,
            inputs.tokens,
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        out_u16 = _from_device(out, out_shape, np.uint16)
        state_f32 = _from_device(state, state_shape, np.float32)
        rows_f32 = _from_device(state_rows, rows_shape, np.float32)
        return _bf16_u16_to_f32(out_u16), state_f32, rows_f32
    finally:
        for buf in (
            conv_out,
            gate,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            state,
            out,
            state_rows,
        ):
            buf.free()


def _run_decode_order_segments_state_rows(
    inputs: _GDNInputs,
    rms_norm_eps: float,
    *,
    cu_seqlens_arr: np.ndarray,
    state_indices_arr: np.ndarray,
    init_state_slots: np.ndarray,
    kernel=qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    gate = _to_device(inputs.gate_u16)
    a = _to_device(inputs.a_u16)
    b = _to_device(inputs.b_u16)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    state = _to_device(init_state_slots)
    cu = _to_device(np.ascontiguousarray(cu_seqlens_arr, dtype=np.int32))
    state_indices = _to_device(np.ascontiguousarray(state_indices_arr, dtype=np.int64))
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    state_shape = (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim)
    rows_shape = (inputs.tokens, *state_shape)
    out = _Buf(int(np.prod(out_shape)) * np.dtype(np.uint16).itemsize)
    state_rows = _Buf(int(np.prod(rows_shape)) * np.dtype(np.float32).itemsize)
    try:
        kernel(
            conv_out.ptr,
            gate.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            state.ptr,
            state_rows.ptr,
            out.ptr,
            cu.ptr,
            state_indices.ptr,
            rms_norm_eps,
            inputs.tokens,
            int(len(cu_seqlens_arr) - 1),
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        out_u16 = _from_device(out, out_shape, np.uint16)
        state_after = _from_device(state, init_state_slots.shape, np.float32)
        rows_f32 = _from_device(state_rows, rows_shape, np.float32)
        return _bf16_u16_to_f32(out_u16), state_after, rows_f32
    finally:
        for buf in (
            conv_out,
            gate,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            state,
            cu,
            state_indices,
            out,
            state_rows,
        ):
            buf.free()


def _run_decode_order_segments_mutating(
    inputs: _GDNInputs,
    rms_norm_eps: float,
    *,
    cu_seqlens_arr: np.ndarray,
    state_indices_arr: np.ndarray,
    init_state_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    gate = _to_device(inputs.gate_u16)
    a = _to_device(inputs.a_u16)
    b = _to_device(inputs.b_u16)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    state = _to_device(init_state_slots)
    cu = _to_device(np.ascontiguousarray(cu_seqlens_arr, dtype=np.int32))
    state_indices = _to_device(np.ascontiguousarray(state_indices_arr, dtype=np.int64))
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * np.dtype(np.uint16).itemsize)
    try:
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments(
            conv_out.ptr,
            gate.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            state.ptr,
            out.ptr,
            cu.ptr,
            state_indices.ptr,
            rms_norm_eps,
            inputs.tokens,
            int(len(cu_seqlens_arr) - 1),
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        out_u16 = _from_device(out, out_shape, np.uint16)
        state_after = _from_device(state, init_state_slots.shape, np.float32)
        return _bf16_u16_to_f32(out_u16), state_after
    finally:
        for buf in (
            conv_out,
            gate,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            state,
            cu,
            state_indices,
            out,
        ):
            buf.free()


def _run_lowp_bf16_single(
    inputs: _GDNInputs, eps: float, state_arr: np.ndarray, *, fused_bf16: bool = False
):
    owners = [_to_device(value) for value in (
        inputs.conv_out_f32, inputs.gate_u16, inputs.a_u16, inputs.b_u16,
        inputs.dt_bias_f32, inputs.a_log_f32, inputs.norm_weight_f32, state_arr,
    )]
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * 4)
    out_bf16 = _Buf(int(np.prod(out_shape)) * 2) if fused_bf16 else None
    try:
        if out_bf16 is None:
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(
                *(owner.ptr for owner in owners), out.ptr, eps, inputs.num_k_heads,
                inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim,
            )
        else:
            qwen35_gdn_recurrent_rmsnorm_gate_lowp_f32_bf16_out(
                *(owner.ptr for owner in owners), out.ptr, out_bf16.ptr, eps,
                inputs.num_k_heads, inputs.num_v_heads, inputs.head_k_dim,
                inputs.head_v_dim,
            )
        return (
            _from_device(
                out if out_bf16 is None else out_bf16,
                out_shape,
                np.float32 if out_bf16 is None else np.uint16,
            ),
            _from_device(owners[-1], state_arr.shape, np.float32),
        )
    finally:
        for owner in (*owners, out, out_bf16):
            if owner is not None:
                owner.free()


def _run_lowp_bf16_segments_rows(inputs, eps, cu_arr, indices_arr, states):
    owners = [_to_device(value) for value in (
        inputs.conv_out_f32, inputs.gate_u16, inputs.a_u16, inputs.b_u16,
        inputs.dt_bias_f32, inputs.a_log_f32, inputs.norm_weight_f32, states,
        np.asarray(cu_arr, dtype=np.int32), np.asarray(indices_arr, dtype=np.int64),
    )]
    state_shape = (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim)
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * 4)
    out_bf16 = _Buf(int(np.prod(out_shape)) * 2)
    rows = _Buf(inputs.tokens * int(np.prod(state_shape)) * 4)
    try:
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_state_rows_bf16(
            *(owner.ptr for owner in owners[:8]), rows.ptr, out.ptr, out_bf16.ptr,
            owners[8].ptr, owners[9].ptr, inputs.tokens, len(cu_arr) - 1, eps,
            inputs.num_k_heads, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim,
        )
        return (
            _from_device(out_bf16, out_shape, np.uint16),
            _from_device(out, out_shape, np.float32),
            _from_device(owners[7], states.shape, np.float32),
            _from_device(rows, (inputs.tokens, *state_shape), np.float32),
        )
    finally:
        for owner in (*owners, out, out_bf16, rows):
            owner.free()


def _run_no_copy_f32_bf16_segments_rows(inputs, eps, cu_arr, indices_arr, states):
    owners = [_to_device(value) for value in (
        inputs.conv_out_f32, inputs.gate_u16, inputs.a_u16, inputs.b_u16,
        inputs.dt_bias_f32, inputs.a_log_f32, inputs.norm_weight_f32, states,
        np.asarray(cu_arr, dtype=np.int32), np.asarray(indices_arr, dtype=np.int64),
    )]
    state_shape = (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim)
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    rows = _Buf(inputs.tokens * int(np.prod(state_shape)) * 4)
    out = _Buf(int(np.prod(out_shape)) * 2)
    out_f32 = _Buf(int(np.prod(out_shape)) * 4)
    try:
        qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_f32(
            *(owner.ptr for owner in owners[:8]), rows.ptr, out.ptr, out_f32.ptr,
            owners[8].ptr, owners[9].ptr, eps, inputs.tokens, len(cu_arr) - 1,
            inputs.num_k_heads, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim,
        )
        return (
            _from_device(out, out_shape, np.uint16),
            _from_device(out_f32, out_shape, np.float32),
            _from_device(rows, (inputs.tokens, *state_shape), np.float32),
        )
    finally:
        for owner in (*owners, rows, out, out_f32):
            owner.free()


def _lowp_fp16_arrays(inputs: _GDNInputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = _bf16_u16_to_f32(inputs.gate_u16).astype(np.float16)
    a = _bf16_u16_to_f32(inputs.a_u16).astype(np.float16)
    b = _bf16_u16_to_f32(inputs.b_u16).astype(np.float16)
    return gate, a, b


def _run_lowp_fp16_single(
    inputs: _GDNInputs,
    rms_norm_eps: float,
    *,
    init_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    gate_arr, a_arr, b_arr = _lowp_fp16_arrays(inputs)
    gate = _to_device(gate_arr)
    a = _to_device(a_arr)
    b = _to_device(b_arr)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    state = _to_device(init_state)
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * np.dtype(np.float32).itemsize)
    try:
        qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(
            conv_out.ptr,
            gate.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            state.ptr,
            out.ptr,
            rms_norm_eps,
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        out_f32 = _from_device(out, out_shape, np.float32)
        state_after = _from_device(
            state,
            (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
            np.float32,
        )
        return out_f32, state_after
    finally:
        for buf in (
            conv_out,
            gate,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            state,
            out,
        ):
            buf.free()


def _run_lowp_fp16_segments(
    inputs: _GDNInputs,
    rms_norm_eps: float,
    *,
    cu_seqlens_arr: np.ndarray,
    state_indices_arr: np.ndarray,
    init_state_slots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    gate_arr, a_arr, b_arr = _lowp_fp16_arrays(inputs)
    gate = _to_device(gate_arr)
    a = _to_device(a_arr)
    b = _to_device(b_arr)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    state = _to_device(init_state_slots)
    cu = _to_device(np.ascontiguousarray(cu_seqlens_arr, dtype=np.int32))
    state_indices = _to_device(np.ascontiguousarray(state_indices_arr, dtype=np.int64))
    out_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    out = _Buf(int(np.prod(out_shape)) * np.dtype(np.float32).itemsize)
    try:
        qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_fp16(
            conv_out.ptr,
            gate.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            norm_weight.ptr,
            state.ptr,
            out.ptr,
            cu.ptr,
            state_indices.ptr,
            inputs.tokens,
            int(len(cu_seqlens_arr) - 1),
            rms_norm_eps,
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        out_f32 = _from_device(out, out_shape, np.float32)
        state_after = _from_device(state, init_state_slots.shape, np.float32)
        return out_f32, state_after
    finally:
        for buf in (
            conv_out,
            gate,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            state,
            cu,
            state_indices,
            out,
        ):
            buf.free()


def _run_chain(
    inputs: _GDNInputs,
    rms_norm_eps: float,
    *,
    use_segments: bool,
    recurrent_variant: str = "k2",
) -> tuple[np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    a = _to_device(inputs.a_u16)
    b = _to_device(inputs.b_u16)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    gate = _to_device(inputs.gate_u16)
    state = _to_device(inputs.init_state_f32)

    qk_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_k_dim)
    v_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    scalar_shape = (inputs.tokens, inputs.num_v_heads)
    query = _Buf(int(np.prod(qk_shape)) * np.dtype(np.float32).itemsize)
    key = _Buf(int(np.prod(qk_shape)) * np.dtype(np.float32).itemsize)
    value = _Buf(int(np.prod(v_shape)) * np.dtype(np.float32).itemsize)
    beta = _Buf(int(np.prod(scalar_shape)) * np.dtype(np.float32).itemsize)
    decay = _Buf(int(np.prod(scalar_shape)) * np.dtype(np.float32).itemsize)
    recurrent_out = _Buf(int(np.prod(v_shape)) * np.dtype(np.float32).itemsize)
    out = _Buf(int(np.prod(v_shape)) * np.dtype(np.uint16).itemsize)

    cu_arr = np.array([0, inputs.tokens], dtype=np.int32)
    state_indices_arr = np.array([0], dtype=np.int64)
    cu = _to_device(cu_arr)
    state_indices = _to_device(state_indices_arr)
    try:
        prepare = (
            qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16
            if recurrent_variant in {"normalized_wave32_xor", "normalized_cluster8"}
            else qwen35_linear_attn_prefill_prepare_f32_bf16
        )
        prepare(
            conv_out.ptr,
            a.ptr,
            b.ptr,
            dt_bias.ptr,
            a_log.ptr,
            query.ptr,
            key.ptr,
            value.ptr,
            beta.ptr,
            decay.ptr,
            inputs.tokens,
            inputs.num_k_heads,
            inputs.num_v_heads,
            inputs.head_k_dim,
            inputs.head_v_dim,
        )
        recurrent = {
            "k2": qwen35_gdn_prefill_recurrent_k2_f32,
            "normalized_wave32_xor": qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32,
            "normalized_cluster8": qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
        }[recurrent_variant]
        recurrent_segments = {
            "k2": qwen35_gdn_prefill_recurrent_segments_k2_f32,
            "normalized_wave32_xor": qwen35_gdn_prefill_recurrent_normalized_segments_wave32_xor_f32,
            "normalized_cluster8": qwen35_gdn_prefill_recurrent_normalized_segments_cluster8_f32,
        }[recurrent_variant]
        if use_segments:
            recurrent_segments(
                query.ptr,
                key.ptr,
                value.ptr,
                beta.ptr,
                decay.ptr,
                state.ptr,
                recurrent_out.ptr,
                cu.ptr,
                state_indices.ptr,
                inputs.tokens,
                1,
                inputs.num_v_heads,
                inputs.head_k_dim,
                inputs.head_v_dim,
            )
        else:
            recurrent(
                query.ptr,
                key.ptr,
                value.ptr,
                beta.ptr,
                decay.ptr,
                state.ptr,
                recurrent_out.ptr,
                inputs.tokens,
                inputs.num_v_heads,
                inputs.head_k_dim,
                inputs.head_v_dim,
            )
        qwen35_gdn_prefill_rmsnorm_gate_bf16(
            recurrent_out.ptr,
            gate.ptr,
            norm_weight.ptr,
            out.ptr,
            rms_norm_eps,
            inputs.tokens,
            inputs.num_v_heads,
            inputs.head_v_dim,
        )
        out_u16 = _from_device(out, v_shape, np.uint16)
        state_f32 = _from_device(
            state,
            (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
            np.float32,
        )
        return _bf16_u16_to_f32(out_u16), state_f32
    finally:
        for buf in (
            conv_out,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            gate,
            state,
            query,
            key,
            value,
            beta,
            decay,
            recurrent_out,
            out,
            cu,
            state_indices,
        ):
            buf.free()


def _run_exact_split_chain(
    inputs: _GDNInputs,
    rms_norm_eps: float,
    *,
    use_segments: bool,
    value_tile: int = 128,
    wave32: bool = False,
    wave32_tree: bool = False,
    lds_tile: int | None = None,
    direct_conv_lds32: bool = False,
    direct_conv_lds32_nonvolatile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    conv_out = _to_device(inputs.conv_out_f32)
    a = _to_device(inputs.a_u16)
    b = _to_device(inputs.b_u16)
    dt_bias = _to_device(inputs.dt_bias_f32)
    a_log = _to_device(inputs.a_log_f32)
    norm_weight = _to_device(inputs.norm_weight_f32)
    gate = _to_device(inputs.gate_u16)
    state = _to_device(inputs.init_state_f32)

    qk_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_k_dim)
    v_shape = (inputs.tokens, inputs.num_v_heads, inputs.head_v_dim)
    scalar_shape = (inputs.tokens, inputs.num_v_heads)
    query_raw = _Buf(int(np.prod(qk_shape)) * np.dtype(np.float32).itemsize)
    key_raw = _Buf(int(np.prod(qk_shape)) * np.dtype(np.float32).itemsize)
    value = _Buf(int(np.prod(v_shape)) * np.dtype(np.float32).itemsize)
    beta = _Buf(int(np.prod(scalar_shape)) * np.dtype(np.float32).itemsize)
    decay = _Buf(int(np.prod(scalar_shape)) * np.dtype(np.float32).itemsize)
    query_scale = _Buf(int(np.prod(scalar_shape)) * np.dtype(np.float32).itemsize)
    key_scale = _Buf(int(np.prod(scalar_shape)) * np.dtype(np.float32).itemsize)
    recurrent_out = _Buf(int(np.prod(v_shape)) * np.dtype(np.float32).itemsize)
    out = _Buf(int(np.prod(v_shape)) * np.dtype(np.uint16).itemsize)
    cu = _to_device(np.asarray([0, inputs.tokens], dtype=np.int32))
    state_indices = _to_device(np.asarray([0], dtype=np.int64))
    try:
        if direct_conv_lds32 or direct_conv_lds32_nonvolatile:
            qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16(
                conv_out.ptr,
                a.ptr,
                b.ptr,
                dt_bias.ptr,
                a_log.ptr,
                beta.ptr,
                decay.ptr,
                query_scale.ptr,
                key_scale.ptr,
                inputs.tokens,
                inputs.num_k_heads,
                inputs.num_v_heads,
                inputs.head_k_dim,
                inputs.head_v_dim,
            )
        else:
            qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16(
                conv_out.ptr,
                a.ptr,
                b.ptr,
                dt_bias.ptr,
                a_log.ptr,
                query_raw.ptr,
                key_raw.ptr,
                value.ptr,
                beta.ptr,
                decay.ptr,
                query_scale.ptr,
                key_scale.ptr,
                inputs.tokens,
                inputs.num_k_heads,
                inputs.num_v_heads,
                inputs.head_k_dim,
                inputs.head_v_dim,
            )
        recurrent = {
            128: qwen35_gdn_prefill_recurrent_decode_order_exact_f32,
            64: qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32,
            32: qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32,
        }[value_tile]
        recurrent_segments = {
            128: qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32,
            64: qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32,
            32: qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32,
        }[value_tile]
        if wave32:
            recurrent = qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32
            recurrent_segments = (
                qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32
            )
        if wave32_tree:
            recurrent = qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32
            recurrent_segments = (
                qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32
            )
        if lds_tile is not None:
            recurrent = {
                64: qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32,
                32: qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32,
            }[lds_tile]
            recurrent_segments = {
                64: qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32,
                32: qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32,
            }[lds_tile]
        if direct_conv_lds32:
            recurrent = qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32
            recurrent_segments = (
                qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32
            )
        if direct_conv_lds32_nonvolatile:
            recurrent = qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32
            recurrent_segments = (
                qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32
            )
        if use_segments:
            if direct_conv_lds32 or direct_conv_lds32_nonvolatile:
                recurrent_segments(
                    conv_out.ptr,
                    beta.ptr,
                    decay.ptr,
                    query_scale.ptr,
                    key_scale.ptr,
                    state.ptr,
                    recurrent_out.ptr,
                    cu.ptr,
                    state_indices.ptr,
                    inputs.tokens,
                    1,
                    inputs.num_k_heads,
                    inputs.num_v_heads,
                    inputs.head_k_dim,
                    inputs.head_v_dim,
                )
            else:
                recurrent_segments(
                    query_raw.ptr,
                    key_raw.ptr,
                    value.ptr,
                    beta.ptr,
                    decay.ptr,
                    query_scale.ptr,
                    key_scale.ptr,
                    state.ptr,
                    recurrent_out.ptr,
                    cu.ptr,
                    state_indices.ptr,
                    inputs.tokens,
                    1,
                    inputs.num_v_heads,
                    inputs.head_k_dim,
                    inputs.head_v_dim,
                )
        else:
            if direct_conv_lds32 or direct_conv_lds32_nonvolatile:
                recurrent(
                    conv_out.ptr,
                    beta.ptr,
                    decay.ptr,
                    query_scale.ptr,
                    key_scale.ptr,
                    state.ptr,
                    recurrent_out.ptr,
                    inputs.tokens,
                    inputs.num_k_heads,
                    inputs.num_v_heads,
                    inputs.head_k_dim,
                    inputs.head_v_dim,
                )
            else:
                recurrent(
                    query_raw.ptr,
                    key_raw.ptr,
                    value.ptr,
                    beta.ptr,
                    decay.ptr,
                    query_scale.ptr,
                    key_scale.ptr,
                    state.ptr,
                    recurrent_out.ptr,
                    inputs.tokens,
                    inputs.num_v_heads,
                    inputs.head_k_dim,
                    inputs.head_v_dim,
                )
        qwen35_gdn_prefill_rmsnorm_gate_bf16(
            recurrent_out.ptr,
            gate.ptr,
            norm_weight.ptr,
            out.ptr,
            rms_norm_eps,
            inputs.tokens,
            inputs.num_v_heads,
            inputs.head_v_dim,
        )
        out_u16 = _from_device(out, v_shape, np.uint16)
        state_f32 = _from_device(
            state,
            (inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
            np.float32,
        )
        return _bf16_u16_to_f32(out_u16), state_f32
    finally:
        for buf in (
            conv_out,
            a,
            b,
            dt_bias,
            a_log,
            norm_weight,
            gate,
            state,
            query_raw,
            key_raw,
            value,
            beta,
            decay,
            query_scale,
            key_scale,
            recurrent_out,
            out,
            cu,
            state_indices,
        ):
            buf.free()


# ---------------------------------------------------------------------------
# No-GPU registry surface.
# ---------------------------------------------------------------------------


def test_gguf_qwen35_gdn_registry_resolves_all_chain_aliases() -> None:
    register_qwen35_linear_attn_gdn_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_k2",
        )
        is qwen35_gdn_prefill_recurrent_k2_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_k2_segments",
        )
        is qwen35_gdn_prefill_recurrent_segments_k2_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="decode_order_bf16",
        )
        is qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="decode_order_bf16_segments",
        )
        is qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="decode_order_bf16_segments_state_rows_no_copy",
        )
        is qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="decode_order_bf16_segments_state_rows_no_copy_wave_reduce",
        )
        is qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_prefill_prepare",
            quant="gguf_qwen35",
            variant="f32_bf16",
        )
        is qwen35_linear_attn_prefill_prepare_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_prefill_prepare",
            quant="gguf_qwen35",
            variant="f32_bf16_raw_scales",
        )
        is qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_prefill_prepare",
            quant="gguf_qwen35",
            variant="f32_bf16_compact_scales",
        )
        is qwen35_linear_attn_prefill_prepare_compact_scales_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_tile64",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_tile64_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_tile32",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_tile32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_tile64",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile64_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_tile32",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_tile32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_lds64",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds64_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_lds32",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_lds32_direct",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_lds32_direct_nonvolatile",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_direct_nonvolatile_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_lds64",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds64_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_lds32",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_lds32_direct",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_lds32_direct_nonvolatile",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_lds32_direct_nonvolatile_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_wave32",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_wave32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_exact_segments_wave32",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_segments_wave32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_wave32_tree",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_qwen35",
            variant="f32_decode_order_segments_wave32_tree",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_segments_wave32_tree_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_rmsnorm_gate",
            quant="gguf_qwen35",
            variant="bf16",
        )
        is qwen35_gdn_prefill_rmsnorm_gate_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_attn_prefill_prepare",
            quant="gguf_ud_q3_k_m",
            variant="f32_bf16",
        )
        is qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_ud_q3_k_m",
            variant="f32_k2",
        )
        is qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gdn_prefill_recurrent",
            quant="gguf_ud_q3_k_m",
            variant="f32_k2_segments",
            missing="none",
        )
        is None
    )


# ---------------------------------------------------------------------------
# GPU correctness.
# ---------------------------------------------------------------------------


_RMS_EPS = 1.0e-6


def _assert_state_close(
    actual: np.ndarray, expected: np.ndarray, *, label: str
) -> None:
    diff = np.abs(actual - expected)
    max_diff = float(diff.max())
    denom = np.maximum(np.abs(expected), 1.0e-3)
    rel = float((diff / denom).max())
    # State is F32 with single-precision accumulation. After per-token decay/key
    # updates, error accumulates linearly in tokens. Allow generous absolute
    # margin scaled by the per-step magnitude.
    assert max_diff < 5.0e-3, f"{label}: state max|delta|={max_diff:g}"
    assert rel < 5.0e-2, f"{label}: state max_rel={rel:g}"


def _assert_output_close(
    actual: np.ndarray, expected: np.ndarray, *, label: str
) -> None:
    diff = np.abs(actual - expected)
    max_diff = float(diff.max())
    denom = np.maximum(np.abs(expected), 1.0e-2)
    rel = float((diff / denom).max())
    # Output is BF16-rounded. Allow ~1% absolute (a few BF16 ULPs at the
    # post-RMS magnitude) and ~10% relative for the tiniest values.
    assert max_diff < 5.0e-2, f"{label}: output max|delta|={max_diff:g}"
    assert rel < 1.5e-1, f"{label}: output max_rel={rel:g}"


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_state_rows_no_copy_matches_mutating_capture() -> None:
    inputs = _GDNInputs(
        tokens=8,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        seed=11,
    )
    mut_out, mut_final_state, mut_rows = _run_decode_order_state_rows(
        inputs, _RMS_EPS, no_copy=False
    )
    no_copy_out, no_copy_state, no_copy_rows = _run_decode_order_state_rows(
        inputs, _RMS_EPS, no_copy=True
    )
    _assert_output_close(no_copy_out, mut_out, label="no-copy vs mutating output")
    _assert_state_close(
        no_copy_rows.reshape(mut_rows.shape),
        mut_rows,
        label="no-copy vs mutating state rows",
    )
    _assert_state_close(
        mut_final_state,
        mut_rows[-1],
        label="mutating final state vs captured final row",
    )
    np.testing.assert_array_equal(no_copy_state, inputs.init_state_f32)


def _slice_gdn_inputs(source: _GDNInputs, start: int, end: int, init_state: np.ndarray) -> _GDNInputs:
    sliced = _GDNInputs(
        tokens=end - start,
        num_k_heads=source.num_k_heads,
        num_v_heads=source.num_v_heads,
        head_k_dim=source.head_k_dim,
        head_v_dim=source.head_v_dim,
        seed=0,
    )
    scalar = source.num_v_heads
    sliced.conv_out_f32 = np.ascontiguousarray(source.conv_out_f32[start:end])
    sliced.gate_u16 = np.ascontiguousarray(source.gate_u16[start:end])
    sliced.a_u16 = np.ascontiguousarray(source.a_u16.reshape(source.tokens, scalar)[start:end].reshape(-1))
    sliced.b_u16 = np.ascontiguousarray(source.b_u16.reshape(source.tokens, scalar)[start:end].reshape(-1))
    sliced.dt_bias_f32 = np.ascontiguousarray(source.dt_bias_f32)
    sliced.a_log_f32 = np.ascontiguousarray(source.a_log_f32)
    sliced.norm_weight_f32 = np.ascontiguousarray(source.norm_weight_f32)
    sliced.init_state_f32 = np.ascontiguousarray(init_state)
    return sliced


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_segments_state_rows_no_copy_matches_per_segment_capture() -> None:
    inputs = _GDNInputs(
        tokens=7,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        seed=19,
    )
    rng = np.random.default_rng(1919)
    init_slots = rng.normal(
        0.0,
        0.05,
        size=(2, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
    ).astype(np.float32)
    cu_seqlens = np.asarray([0, 4, 7], dtype=np.int32)
    state_indices = np.asarray([0, 1], dtype=np.int64)

    packed_out, packed_state_after, packed_rows = _run_decode_order_segments_state_rows(
        inputs,
        _RMS_EPS,
        cu_seqlens_arr=cu_seqlens,
        state_indices_arr=state_indices,
        init_state_slots=init_slots,
    )

    expected_out = np.empty_like(packed_out)
    expected_rows = np.empty_like(packed_rows)
    for segment, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        sliced = _slice_gdn_inputs(inputs, int(start), int(end), init_slots[segment])
        seg_out, seg_state_after, seg_rows = _run_decode_order_state_rows(
            sliced,
            _RMS_EPS,
            no_copy=True,
        )
        expected_out[int(start) : int(end)] = seg_out
        expected_rows[int(start) : int(end)] = seg_rows
        np.testing.assert_array_equal(seg_state_after, init_slots[segment])

    _assert_output_close(packed_out, expected_out, label="packed segments output")
    _assert_state_close(packed_rows, expected_rows, label="packed segments state rows")
    np.testing.assert_array_equal(packed_state_after, init_slots)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_segments_state_rows_wave_reduce_is_parent_bit_exact() -> None:
    inputs = _GDNInputs(
        tokens=18,
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        seed=0x38A6,
    )
    rng = np.random.default_rng(0x38A6)
    init_slots = rng.normal(
        0.0,
        0.05,
        size=(6, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
    ).astype(np.float32)
    cu_seqlens = np.arange(0, 19, 3, dtype=np.int32)
    state_indices = np.arange(6, dtype=np.int64)

    parent = _run_decode_order_segments_state_rows(
        inputs,
        _RMS_EPS,
        cu_seqlens_arr=cu_seqlens,
        state_indices_arr=state_indices,
        init_state_slots=init_slots,
    )
    candidate = _run_decode_order_segments_state_rows(
        inputs,
        _RMS_EPS,
        cu_seqlens_arr=cu_seqlens,
        state_indices_arr=state_indices,
        init_state_slots=init_slots,
        kernel=qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_wave_reduce,
    )
    for actual, expected in zip(candidate, parent, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_segments_mutating_matches_per_segment_decode_order() -> None:
    inputs = _GDNInputs(
        tokens=7,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        seed=23,
    )
    rng = np.random.default_rng(2323)
    init_slots = rng.normal(
        0.0,
        0.05,
        size=(2, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
    ).astype(np.float32)
    cu_seqlens = np.asarray([0, 4, 7], dtype=np.int32)
    state_indices = np.asarray([0, 1], dtype=np.int64)

    packed_out, packed_state_after = _run_decode_order_segments_mutating(
        inputs,
        _RMS_EPS,
        cu_seqlens_arr=cu_seqlens,
        state_indices_arr=state_indices,
        init_state_slots=init_slots,
    )

    expected_out = np.empty_like(packed_out)
    expected_state_after = np.empty_like(init_slots)
    for segment, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)):
        sliced = _slice_gdn_inputs(inputs, int(start), int(end), init_slots[segment])
        seg_out, seg_state_after = _run_decode_order_bf16(sliced, _RMS_EPS)
        expected_out[int(start) : int(end)] = seg_out
        expected_state_after[segment] = seg_state_after

    _assert_output_close(packed_out, expected_out, label="packed mutating segments output")
    _assert_state_close(
        packed_state_after,
        expected_state_after,
        label="packed mutating segments final state",
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("num_k_heads,num_v_heads", [(1, 2), (16, 32)])
def test_gdn_segments_lowp_state_rows_match_scalar_c1(
    num_k_heads: int, num_v_heads: int
) -> None:
    inputs = _GDNInputs(tokens=6, num_k_heads=num_k_heads, num_v_heads=num_v_heads,
                        head_k_dim=128, head_v_dim=128, seed=20260828)
    cu = np.asarray([0, 3, 6], dtype=np.int32)
    indices = np.asarray([0, 1], dtype=np.int64)
    states = np.stack((inputs.init_state_f32, inputs.init_state_f32 * 0.75)).astype(np.float32)
    actual_out, actual_f32, actual_final, actual_rows = _run_lowp_bf16_segments_rows(
        inputs, 1e-6, cu, indices, states.copy())
    expected_out = np.empty_like(actual_out)
    expected_rows = np.empty_like(actual_rows)
    expected_final = states.copy()
    for segment, (start, end) in enumerate(zip(cu[:-1], cu[1:], strict=True)):
        state = states[segment].copy()
        for token in range(int(start), int(end)):
            row_inputs = _slice_gdn_inputs(inputs, token, token + 1, state)
            row_out, state = _run_lowp_bf16_single(
                row_inputs, 1e-6, state, fused_bf16=True
            )
            expected_out[token] = row_out[0]
            expected_rows[token] = state
        expected_final[segment] = state
    np.testing.assert_array_equal(actual_out, expected_out)
    np.testing.assert_array_equal(actual_rows, expected_rows)
    np.testing.assert_array_equal(actual_final, expected_final)
    no_copy_out, no_copy_f32, no_copy_rows = _run_no_copy_f32_bf16_segments_rows(
        inputs, 1e-6, cu, indices, states.copy()
    )
    np.testing.assert_array_equal(no_copy_out, expected_out)
    np.testing.assert_array_equal(no_copy_rows, expected_rows)
    np.testing.assert_array_equal(no_copy_f32.view(np.uint32), actual_f32.view(np.uint32))


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_segments_lowp_fp16_c6_one_token_segments_match_independent_c1() -> None:
    inputs = _GDNInputs(
        tokens=6,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        seed=31,
    )
    rng = np.random.default_rng(3131)
    init_slots = rng.normal(
        0.0,
        0.05,
        size=(6, inputs.num_v_heads, inputs.head_k_dim, inputs.head_v_dim),
    ).astype(np.float32)
    cu_seqlens = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    state_indices = np.asarray([0, 2, 4, 1, 3, 5], dtype=np.int64)

    packed_out, packed_state_after = _run_lowp_fp16_segments(
        inputs,
        _RMS_EPS,
        cu_seqlens_arr=cu_seqlens,
        state_indices_arr=state_indices,
        init_state_slots=init_slots,
    )

    expected_out = np.empty_like(packed_out)
    expected_state_after = init_slots.copy()
    for segment, slot in enumerate(state_indices.tolist()):
        sliced = _slice_gdn_inputs(inputs, segment, segment + 1, init_slots[int(slot)])
        seg_out, seg_state_after = _run_lowp_fp16_single(
            sliced,
            _RMS_EPS,
            init_state=init_slots[int(slot)],
        )
        expected_out[segment : segment + 1] = seg_out
        expected_state_after[int(slot)] = seg_state_after

    np.testing.assert_allclose(packed_out, expected_out, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(packed_state_after, expected_state_after, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_paths_match_cpu_oracle_small_shape() -> None:
    inputs = _GDNInputs(
        tokens=8,
        num_k_heads=1,
        num_v_heads=2,
        head_k_dim=128,
        head_v_dim=128,
        seed=1,
    )
    expected_out, expected_state = _cpu_full_chain(inputs, _RMS_EPS)
    for label, fn in (
        ("decode_order_bf16", lambda: _run_decode_order_bf16(inputs, _RMS_EPS)),
        ("chain_k2", lambda: _run_chain(inputs, _RMS_EPS, use_segments=False)),
        (
            "chain_segments_k2",
            lambda: _run_chain(inputs, _RMS_EPS, use_segments=True),
        ),
    ):
        actual_out, actual_state = fn()
        _assert_state_close(actual_state, expected_state, label=label)
        _assert_output_close(actual_out, expected_out, label=label)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_paths_match_cpu_oracle_qwen36_shape() -> None:
    inputs = _GDNInputs(
        tokens=64,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=2,
    )
    expected_out, expected_state = _cpu_full_chain(inputs, _RMS_EPS)
    for label, fn in (
        ("decode_order_bf16", lambda: _run_decode_order_bf16(inputs, _RMS_EPS)),
        ("chain_k2", lambda: _run_chain(inputs, _RMS_EPS, use_segments=False)),
        (
            "chain_segments_k2",
            lambda: _run_chain(inputs, _RMS_EPS, use_segments=True),
        ),
    ):
        actual_out, actual_state = fn()
        _assert_state_close(actual_state, expected_state, label=label)
        _assert_output_close(actual_out, expected_out, label=label)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("tokens", [1024, 1025, 1026])
def test_gdn_prefill_segment_boundary_paths_agree(tokens: int) -> None:
    """The segments_k2 and k2 paths must agree at the segment-threshold boundary.

    The runtime opts into segments_k2 at rows >=
    ``HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD`` (default 1025). This test
    pins both paths at 1024/1025/1026 against the CPU oracle so neither one
    silently drifts as we tune the default.
    """

    inputs = _GDNInputs(
        tokens=tokens,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=128,
        head_v_dim=128,
        seed=tokens,
    )
    expected_out, expected_state = _cpu_full_chain(inputs, _RMS_EPS)
    out_k2, state_k2 = _run_chain(inputs, _RMS_EPS, use_segments=False)
    out_seg, state_seg = _run_chain(inputs, _RMS_EPS, use_segments=True)
    _assert_state_close(state_k2, expected_state, label=f"chain_k2 tokens={tokens}")
    _assert_state_close(
        state_seg, expected_state, label=f"chain_segments_k2 tokens={tokens}"
    )
    _assert_output_close(out_k2, expected_out, label=f"chain_k2 tokens={tokens}")
    _assert_output_close(
        out_seg, expected_out, label=f"chain_segments_k2 tokens={tokens}"
    )
    # k2 and segments_k2 must also agree with each other within F32 tolerance
    # (same math, different scheduling).
    state_diff = float(np.abs(state_k2 - state_seg).max())
    out_diff = float(np.abs(out_k2 - out_seg).max())
    assert state_diff < 5.0e-3, f"k2 vs segments_k2 state diff = {state_diff:g} @ tokens={tokens}"
    assert out_diff < 5.0e-2, f"k2 vs segments_k2 output diff = {out_diff:g} @ tokens={tokens}"


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_gdn_prefill_chain_matches_decode_order_within_drift_budget() -> None:
    """Pin the cross-implementation drift documented in P9.A1.

    The fused ``decode_order_bf16`` and the ``prepare + k2 + rmsnorm_gate``
    chain perform mathematically equivalent GDN updates but in slightly
    different reduction orders, and the chain materializes BF16->F32 tensors
    between stages. This test fails if the drift grows beyond the budget that
    P9.A2 set after task #17 landed (state F32 within 5e-3 absolute / 5%
    relative; output BF16 within 5e-2 / 15%).
    """

    inputs = _GDNInputs(
        tokens=64,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=3,
    )
    out_fused, state_fused = _run_decode_order_bf16(inputs, _RMS_EPS)
    out_chain, state_chain = _run_chain(inputs, _RMS_EPS, use_segments=False)
    _assert_state_close(state_chain, state_fused, label="chain_k2 vs fused")
    _assert_output_close(out_chain, out_fused, label="chain_k2 vs fused")


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
@pytest.mark.parametrize("value_tile", [128, 64, 32])
def test_gdn_prefill_chain_is_bit_exact_to_decode_order(
    use_segments: bool,
    value_tile: int,
) -> None:
    """Resident prefill state must not depend on the selected GDN scheduler."""

    inputs = _GDNInputs(
        tokens=17,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=17,
    )
    out_fused, state_fused = _run_decode_order_bf16(inputs, _RMS_EPS)
    out_chain, state_chain = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        value_tile=value_tile,
    )
    np.testing.assert_array_equal(state_chain, state_fused)
    np.testing.assert_array_equal(out_chain, out_fused)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
def test_gdn_prefill_wave32_chain_is_bit_exact_to_decode_order(
    use_segments: bool,
) -> None:
    """The wave-sharded schedule must retain the production state contract."""

    inputs = _GDNInputs(
        tokens=17,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=23,
    )
    out_fused, state_fused = _run_decode_order_bf16(inputs, _RMS_EPS)
    out_wave, state_wave = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        wave32=True,
    )
    np.testing.assert_array_equal(out_wave, out_fused)
    np.testing.assert_array_equal(state_wave, state_fused)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
@pytest.mark.parametrize("lds_tile", [64, 32])
def test_gdn_prefill_lds_resident_chain_is_bit_exact_to_decode_order(
    use_segments: bool,
    lds_tile: int,
) -> None:
    """LDS state residency must retain the scalar fused recurrence contract."""

    inputs = _GDNInputs(
        tokens=17,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=31,
    )
    out_fused, state_fused = _run_decode_order_bf16(inputs, _RMS_EPS)
    out_lds, state_lds = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        lds_tile=lds_tile,
    )
    np.testing.assert_array_equal(out_lds, out_fused)
    np.testing.assert_array_equal(state_lds, state_fused)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
def test_gdn_prefill_direct_conv_lds32_is_bit_exact_to_materialized_chain(
    use_segments: bool,
) -> None:
    """Compact scales plus direct conv reads must retain every exact output bit."""

    inputs = _GDNInputs(
        tokens=17,
        num_k_heads=4,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=37,
    )
    out_materialized, state_materialized = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        lds_tile=32,
    )
    out_direct, state_direct = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        direct_conv_lds32=True,
    )
    out_cpu, state_cpu = _cpu_full_chain(inputs, _RMS_EPS)
    np.testing.assert_array_equal(out_direct, out_materialized)
    np.testing.assert_array_equal(state_direct, state_materialized)
    _assert_output_close(out_direct, out_cpu, label="direct LDS32 vs CPU")
    _assert_state_close(state_direct, state_cpu, label="direct LDS32 vs CPU")


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
def test_gdn_prefill_direct_nonvolatile_is_bit_exact_to_direct(
    use_segments: bool,
) -> None:
    """Compiler-cacheable LDS accesses must preserve direct recurrence bits."""

    inputs = _GDNInputs(
        tokens=17,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=128,
        head_v_dim=32,
        seed=113,
    )
    out_baseline, state_baseline = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        direct_conv_lds32=True,
    )
    out_candidate, state_candidate = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        direct_conv_lds32_nonvolatile=True,
    )
    np.testing.assert_array_equal(out_candidate, out_baseline)
    np.testing.assert_array_equal(state_candidate, state_baseline)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
def test_gdn_prefill_normalized_wave32_xor_stays_within_correctness_budget(
    use_segments: bool,
) -> None:
    """The llama.cpp HIP schedule must satisfy the peer numerical contract."""

    inputs = _GDNInputs(
        tokens=64,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=31,
    )
    expected_out, expected_state = _cpu_full_chain(inputs, _RMS_EPS)
    out_peer, state_peer = _run_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        recurrent_variant="normalized_wave32_xor",
    )
    _assert_output_close(out_peer, expected_out, label="normalized wave32 XOR vs CPU")
    _assert_state_close(state_peer, expected_state, label="normalized wave32 XOR vs CPU")


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
def test_gdn_prefill_normalized_cluster8_stays_within_correctness_budget(
    use_segments: bool,
) -> None:
    """The Vulkan clustered schedule must satisfy the peer numerical contract."""

    inputs = _GDNInputs(
        tokens=64,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=37,
    )
    expected_out, expected_state = _cpu_full_chain(inputs, _RMS_EPS)
    out_peer, state_peer = _run_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        recurrent_variant="normalized_cluster8",
    )
    _assert_output_close(out_peer, expected_out, label="normalized cluster8 vs CPU")
    _assert_state_close(state_peer, expected_state, label="normalized cluster8 vs CPU")


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("use_segments", [False, True])
def test_gdn_prefill_wave32_tree_stays_within_correctness_budget(
    use_segments: bool,
) -> None:
    """Tree reduction may reorder sums but must stay inside the GDN gate."""

    inputs = _GDNInputs(
        tokens=64,
        num_k_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        seed=29,
    )
    out_fused, state_fused = _run_decode_order_bf16(inputs, _RMS_EPS)
    out_tree, state_tree = _run_exact_split_chain(
        inputs,
        _RMS_EPS,
        use_segments=use_segments,
        wave32_tree=True,
    )
    _assert_output_close(out_tree, out_fused, label="wave32 tree vs fused")
    _assert_state_close(state_tree, state_fused, label="wave32 tree vs fused")
