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
    qwen35_gdn_prefill_recurrent_k2_f32,
    qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order,
    qwen35_gdn_prefill_recurrent_segments_k2_f32,
    qwen35_gdn_prefill_rmsnorm_gate_bf16,
    qwen35_linear_attn_prefill_prepare_f32_bf16,
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
    repeat = num_v_heads // num_k_heads
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
            k_head = v_head // repeat
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


def _run_chain(
    inputs: _GDNInputs, rms_norm_eps: float, *, use_segments: bool
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
        qwen35_linear_attn_prefill_prepare_f32_bf16(
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
        if use_segments:
            qwen35_gdn_prefill_recurrent_segments_k2_f32(
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
            qwen35_gdn_prefill_recurrent_k2_f32(
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
            layer="linear_attn_prefill_prepare",
            quant="gguf_qwen35",
            variant="f32_bf16",
        )
        is qwen35_linear_attn_prefill_prepare_f32_bf16
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
