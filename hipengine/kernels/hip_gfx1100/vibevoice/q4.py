"""Q4_K_M backbone linear primitives for the VibeVoice Qwen2 runner.

Same signatures as the bf16 dense primitives the runtime already calls,
so the runner is unchanged apart from weight content (raw Q4_K block
bytes from the merged GGUF) and which kernels namespace is resolved.
The fused gate+up dual GEMV is emulated as two pack8 GEMVs writing the
bf16 dual layout directly at rows=1, and as two raw-block GEMVs plus
f32->bf16 casts above it; the f32 scratch for that upper path is
caller-owned (``scratch=(gate, up)``) so concurrent or interleaved
requests on separate runners cannot overwrite each other.

Decode (rows=1) routes through the raw-block **pack8 decode** families,
which read the unmodified GGUF Q4_K/Q6_K block bytes on device (no
repack, no sidecar, no extra resident memory). Measured on the real
VibeVoice tensors, the raw-block block GEMVs reach 18-60 GB/s while the
pack8 decode families reach 107-132 GB/s, which is the bandwidth wall for
these shapes.
"""

from __future__ import annotations

import ctypes

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free
from hipengine.kernels.registry import KernelKey, is_registered, register
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16, bf16_to_f32
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q4_k_wmma_prefill_bf16_f32_out as _q4_prefill_bf16_f32,
    gguf_q4_k_wmma_prefill_f32_f32_out as _q4_prefill_f32_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q6_k_wmma_prefill_bf16_bf16_out as _q6_prefill_bf16_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q5_k_gemv_bf16_bf16_out as _q5_gemv_bf16_bf16,
    gguf_q5_k_gemv_bf16_f32_out as _q5_gemv_bf16_f32,
    gguf_q5_k_gemv_f32_f32_out as _q5_gemv_f32_f32,
    gguf_q6_k_gemv_bf16_bf16_out as _q6_gemv_bf16_bf16,
    gguf_q6_k_gemv_bf16_f32_out as _q6_gemv_bf16_f32,
    gguf_q6_k_gemv_f32_f32_out as _q6_gemv_f32_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_gemv_bf16_bf16_out,
    gguf_q4_k_gemv_bf16_f32_out,
    gguf_q4_k_gemv_f32_f32_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_pack8_gemv import (
    gguf_q4_k_pack8_gemv_decode_bf16_bf16_out as _q4_p8_bf16_bf16,
    gguf_q4_k_pack8_gemv_decode_bf16_f32_out as _q4_p8_bf16_f32,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    gguf_q6_k_pack8_gemv_decode_bf16_bf16_out as _q6_p8_bf16_bf16,
    gguf_q6_k_pack8_gemv_decode_bf16_f32_out as _q6_p8_bf16_f32,
)

# GGML K-quant type ids used by llama-quantize's Q4_K_M ruleset; the
# sensitive tensors (attn_v, ffn_down) are adaptively upgraded per layer.
GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 12, 13, 14

# rows=1 decode: raw GGUF blocks in, no repack. Q5_K has no dense decode
# entry point in this family and falls through to the raw-block GEMV.
_PACK8_DECODE = {
    (GGML_Q4_K, False): _q4_p8_bf16_f32,
    (GGML_Q4_K, True): _q4_p8_bf16_bf16,
    (GGML_Q6_K, False): _q6_p8_bf16_f32,
    (GGML_Q6_K, True): _q6_p8_bf16_bf16,
}


def _pick_gemv(w_ptr, bf16_in, bf16_out, rows=1):
    """Select the raw-block GEMV family for the weight's GGML type."""
    weight_type = GGUF_WEIGHT_TYPES.get(int(w_ptr), GGML_Q4_K)
    if rows == 1 and bf16_in:
        pack8 = _PACK8_DECODE.get((weight_type, bool(bf16_out)))
        if pack8 is not None:
            return pack8
    if bf16_in:
        return {
            GGML_Q4_K: gguf_q4_k_gemv_bf16_f32_out if not bf16_out else gguf_q4_k_gemv_bf16_bf16_out,
            GGML_Q5_K: _q5_gemv_bf16_f32 if not bf16_out else _q5_gemv_bf16_bf16,
            GGML_Q6_K: _q6_gemv_bf16_f32 if not bf16_out else _q6_gemv_bf16_bf16,
        }[weight_type]
    return {
        GGML_Q4_K: gguf_q4_k_gemv_f32_f32_out,
        GGML_Q5_K: _q5_gemv_f32_f32,
        GGML_Q6_K: _q6_gemv_f32_f32,
    }[weight_type]
from hipengine.loading.vibevoice_asr_gguf import GGUF_WEIGHT_TYPES


def q4_dense_gemv_bf16_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    _pick_gemv(w_ptr, bf16_in=True, bf16_out=False, rows=rows)(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_gemv_f32_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    _pick_gemv(w_ptr, bf16_in=False, bf16_out=False, rows=rows)(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_gemv_out_bf16(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    _pick_gemv(w_ptr, bf16_in=True, bf16_out=True, rows=rows)(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_dual_gemv_out_bf16(x_ptr, gate_w_ptr, up_w_ptr, out_ptr, rows,
                                in_features, out_features, out_width, **kwargs):
    """Two GEMVs into the fused [gate | up] bf16 dual layout.

    At rows=1 both halves use the raw-block pack8 decode family, which
    writes bf16 directly into the dual layout - no f32 scratch and no
    cast pass. Above rows=1 it falls back to f32 GEMVs plus casts.

    ``scratch`` is an optional ``(gate, up)`` pair of f32 device buffers
    owned by the calling runner, used only by the fallback path. Pass it:
    a process-global cache keyed by width would let interleaved requests
    corrupt each other's intermediates and would never be released by
    ``runner.close()``. Without it, per-call buffers are allocated and
    freed (correct, but two allocations per layer per token).
    """
    runtime = kwargs.get("runtime") or get_hip_runtime()
    stream = kwargs.get("stream", 0)
    gate_p8 = _PACK8_DECODE.get((GGUF_WEIGHT_TYPES.get(int(gate_w_ptr), GGML_Q4_K), True))
    up_p8 = _PACK8_DECODE.get((GGUF_WEIGHT_TYPES.get(int(up_w_ptr), GGML_Q4_K), True))
    if rows == 1 and gate_p8 is not None and up_p8 is not None:
        gate_p8(x_ptr, gate_w_ptr, out_ptr, rows, in_features, out_features,
                stream=stream, runtime=runtime)
        up_p8(x_ptr, up_w_ptr, out_ptr + out_features * 2, rows, in_features, out_features,
              stream=stream, runtime=runtime)
        return
    if rows != 1:
        raise ValueError("q4 dual gemv emulation is decode-only (rows=1)")
    scratch = kwargs.get("scratch")
    if scratch is None:
        gate_f32 = malloc(out_features * 4)
        up_f32 = malloc(out_features * 4)
        owned = True
    else:
        gate_f32, up_f32 = scratch
        if gate_f32.nbytes < out_features * 4 or up_f32.nbytes < out_features * 4:
            raise ValueError("q4 dual gemv scratch is too small for this width")
        owned = False
    try:
        _pick_gemv(gate_w_ptr, bf16_in=True, bf16_out=False, rows=rows)(
            x_ptr, gate_w_ptr, gate_f32.ptr, rows, in_features, out_features,
            stream=stream, runtime=runtime,
        )
        _pick_gemv(up_w_ptr, bf16_in=True, bf16_out=False, rows=rows)(
            x_ptr, up_w_ptr, up_f32.ptr, rows, in_features, out_features,
            stream=stream, runtime=runtime,
        )
        f32_to_bf16(gate_f32.ptr, out_ptr, out_features, stream=stream, runtime=runtime)
        f32_to_bf16(up_f32.ptr, out_ptr + out_features * 2, out_features, stream=stream, runtime=runtime)
    finally:
        if owned:
            free(gate_f32)
            free(up_f32)


def q4_dense_gemv_f32_bf16w_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    """o_proj: f32 activations against Q4 blocks, f32 out.

    The pack8 decode families take bf16/fp16 activations, so at rows=1 the
    f32 activation is narrowed once into a caller-owned bf16 scratch and
    the fast decode GEMV runs on that. The extra pass is over
    ``in_features`` elements and buys the whole GEMV its bandwidth wall.
    """
    runtime = kwargs.get("runtime") or get_hip_runtime()
    stream = kwargs.get("stream", 0)
    pack8 = _PACK8_DECODE.get((GGUF_WEIGHT_TYPES.get(int(w_ptr), GGML_Q4_K), False))
    if rows == 1 and pack8 is not None:
        x_bf16 = kwargs.get("x_bf16_scratch")
        if x_bf16 is None:
            x_bf16 = malloc(in_features * 2)
            owned = True
        else:
            if x_bf16.nbytes < in_features * 2:
                raise ValueError("q4 o_proj bf16 scratch is too small")
            owned = False
        try:
            f32_to_bf16(x_ptr, x_bf16.ptr, in_features, stream=stream, runtime=runtime)
            pack8(x_bf16.ptr, w_ptr, out_ptr, rows, in_features, out_features,
                  stream=stream, runtime=runtime)
        finally:
            if owned:
                free(x_bf16)
        return
    q4_dense_gemv_f32_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs)


def q4_prefill(runner, hidden_rows, rows, start):
    """Batched causal Q4_K_M prefill over raw GGUF K-quant blocks.

    Mirrors the dense bf16 ``_prefill_batched`` orchestration but replaces
    the hipBLASLt fp16 GEMMs with the raw-block Q4_K/Q6_K WMMA prefill
    kernels, which read the unmodified GGUF block bytes (no repack, no fp16
    weight copies) and run at parity with the fp16 route: measured on
    gfx1151 at 146 rows, ffn_gate 1.92 ms and ffn_down 1.51 ms against
    ~1.5 ms per GEMM for the bf16 lane, versus 170 ms and 162 ms for the
    naive raw-block prefill kernels.

    Q6_K has no f32-output prefill entry point, so those tensors
    (attn_v/ffn_down on half the layers) land in a bf16 scratch and are
    widened once. Scratch is allocated per call and released in ``finally``.
    """
    from numbers import Integral

    from hipengine.runtime.vibevoice_qwen2 import _upload
    from hipengine.core.memory import copy_host_to_device, host_array_ptr

    spec = runner.spec
    hidden = spec.hidden_size
    heads = spec.num_attention_heads
    kv_heads = spec.num_key_value_heads
    head_dim = spec.head_dim
    ffn = spec.intermediate_size
    kv_dim = kv_heads * head_dim
    runtime = runner.runtime
    stream = 0
    runner._validate_position(start)
    if isinstance(rows, bool) or not isinstance(rows, Integral) or rows <= 0:
        raise ValueError("prefill rows must be a positive integer")
    if start + rows > runner.max_context:
        raise ValueError("prefill exceeds max_context")

    def prefill_weight_type(w_ptr, role):
        """GGML type of a weight, from the loader's pointer-keyed registry.

        The registry is the same side channel the decode GEMV dispatch reads.
        A missing entry means this pointer did not come from the Q4 GGUF
        loader, and guessing a type there is exactly how a different quant
        (Q5_K, Q8_0, IQ4_XS) would get silently decoded by the Q4_K kernel.
        """
        weight_type = GGUF_WEIGHT_TYPES.get(int(w_ptr))
        if weight_type is None:
            raise ValueError(
                f"batched prefill: no GGUF type registered for the {role} weight "
                f"at 0x{int(w_ptr):x}; load the model through the Q4 GGUF loader"
            )
        return weight_type

    def gemm_bf16_f32(x_ptr, w_ptr, out_ptr, in_f, out_f, bf16_scratch, role):
        """bf16 activations x raw K-quant blocks -> f32.

        Only the types with a WMMA prefill kernel are routed; anything else
        fails loudly instead of falling through to the Q4_K decoder.
        """
        weight_type = prefill_weight_type(w_ptr, role)
        if weight_type == GGML_Q4_K:
            _q4_prefill_bf16_f32(x_ptr, w_ptr, out_ptr, rows, in_f, out_f,
                                 stream=stream, runtime=runtime)
        elif weight_type == GGML_Q6_K:
            # Q6_K has no f32-output prefill entry point: widen once.
            _q6_prefill_bf16_bf16(x_ptr, w_ptr, bf16_scratch.ptr, rows, in_f, out_f,
                                  stream=stream, runtime=runtime)
            bf16_to_f32(bf16_scratch.ptr, out_ptr, rows * out_f, stream=stream, runtime=runtime)
        else:
            raise ValueError(
                f"batched prefill: no WMMA prefill kernel for GGML type "
                f"{weight_type} ({role}); this route wires Q4_K and Q6_K. Q5_K has "
                f"only the naive raw-block prefill (~100x slower) and the other "
                f"types have none, so routing them here would mis-decode."
            )

    def gemm_f32_f32(x_ptr, w_ptr, out_ptr, in_f, out_f, role):
        """f32 activations x raw Q4_K blocks -> f32 (o_proj keeps f32)."""
        weight_type = prefill_weight_type(w_ptr, role)
        if weight_type != GGML_Q4_K:
            raise ValueError(
                f"batched prefill: {role} needs the f32/f32 Q4_K WMMA prefill but "
                f"its weight is GGML type {weight_type}"
            )
        _q4_prefill_f32_f32(x_ptr, w_ptr, out_ptr, rows, in_f, out_f,
                            stream=stream, runtime=runtime)

    pos_host = np.arange(start, start + rows, dtype=np.int64)
    positions = _upload(pos_host)
    counts = _upload(pos_host + 1)
    spans = runner._spans(positions, counts, rows)
    qkv_w = rows * hidden * 4
    kv_w = rows * kv_dim * 4
    scratch = [
        malloc(rows * hidden * 2),   # normed bf16
        malloc(qkv_w),               # q f32
        malloc(qkv_w),               # q_out f32 (rope)
        malloc(kv_w),                # k f32
        malloc(kv_w),                # v f32
        malloc(kv_w),                # k_out f32 (rope)
        malloc(rows * kv_dim * 2),   # k bf16
        malloc(rows * kv_dim * 2),   # v bf16
        malloc(qkv_w),               # attn f32
        malloc(rows * hidden * 4),   # o f32
        malloc(rows * hidden * 2),   # normed2 bf16
        malloc(rows * ffn * 4),      # gate f32
        malloc(rows * ffn * 4),      # up f32
        malloc(rows * ffn * 2),      # gate bf16
        malloc(rows * ffn * 2),      # up bf16
        malloc(rows * ffn * 2),      # act bf16
        malloc(rows * ffn * 2),      # bf16 scratch for Q6_K prefill output
        malloc(rows * hidden * 4),   # down f32
        malloc(rows * hidden * 2),   # down bf16
    ]
    (normed, q_f32, q_out, k_f32, v_f32, k_out, k_bf16, v_bf16, attn, o_f32,
     normed2, gate_f32, up_f32, gate, up, act, q6_scratch, down_f32, down_bf16) = scratch
    try:
        for layer in runner.layers:
            runner.kernels.vv_rmsnorm_bf16(
                hidden_rows.ptr, layer.input_ln.ptr, normed.ptr, rows, hidden,
                spec.rms_norm_eps, library=runner.library, runtime=runtime)
            gemm_bf16_f32(normed.ptr, layer.q_w.ptr, q_f32.ptr, hidden, hidden, q6_scratch, "q_proj")
            gemm_bf16_f32(normed.ptr, layer.k_w.ptr, k_f32.ptr, hidden, kv_dim, q6_scratch, "k_proj")
            gemm_bf16_f32(normed.ptr, layer.v_w.ptr, v_f32.ptr, hidden, kv_dim, q6_scratch, "v_proj")
            runner.kernels.vv_add_bias_f32(q_f32.ptr, layer.q_b.ptr, q_f32.ptr, rows * hidden, hidden,
                                           library=runner.library, runtime=runtime)
            runner.kernels.vv_add_bias_f32(k_f32.ptr, layer.k_b.ptr, k_f32.ptr, rows * kv_dim, kv_dim,
                                           library=runner.library, runtime=runtime)
            runner.kernels.vv_add_bias_f32(v_f32.ptr, layer.v_b.ptr, v_f32.ptr, rows * kv_dim, kv_dim,
                                           library=runner.library, runtime=runtime)
            runner.kernels.vv_rope_positions_f32(
                q_f32.ptr, k_f32.ptr, runner._cos.ptr, runner._sin.ptr, positions.ptr,
                q_out.ptr, k_out.ptr, rows, heads, kv_heads, head_dim,
                stream=stream, runtime=runtime)
            runner.kernels.f32_to_bf16(k_out.ptr, k_bf16.ptr, rows * kv_dim, stream=stream, runtime=runtime)
            runner.kernels.f32_to_bf16(v_f32.ptr, v_bf16.ptr, rows * kv_dim, stream=stream, runtime=runtime)
            runner.kernels.vv_kv_write_spans(
                k_bf16.ptr, v_bf16.ptr, layer.k_cache.ptr, layer.v_cache.ptr,
                spans, rows, kv_heads, head_dim, library=runner.library, runtime=runtime)
            runner.kernels.vv_attention_spans(
                q_out.ptr, layer.k_cache.ptr, layer.v_cache.ptr, attn.ptr,
                spans, rows, heads, kv_heads, head_dim, runner._scale,
                library=runner.library, runtime=runtime)
            # o projection keeps f32 activations, so it uses the f32/f32 variant.
            gemm_f32_f32(attn.ptr, layer.o_w.ptr, o_f32.ptr, heads * head_dim, hidden, "o_proj")
            runner.kernels.f32_to_bf16(o_f32.ptr, down_bf16.ptr, rows * hidden,
                                       stream=stream, runtime=runtime)
            runner.kernels.vv_scale_residual_bf16(
                hidden_rows.ptr, down_bf16.ptr, runner._ones_hidden.ptr, hidden_rows.ptr,
                rows * hidden, hidden, library=runner.library, runtime=runtime)
            runner.kernels.vv_rmsnorm_bf16(
                hidden_rows.ptr, layer.post_ln.ptr, normed2.ptr, rows, hidden,
                spec.rms_norm_eps, library=runner.library, runtime=runtime)
            gemm_bf16_f32(normed2.ptr, layer.gate_w.ptr, gate_f32.ptr, hidden, ffn, q6_scratch, "gate_proj")
            gemm_bf16_f32(normed2.ptr, layer.up_w.ptr, up_f32.ptr, hidden, ffn, q6_scratch, "up_proj")
            runner.kernels.f32_to_bf16(gate_f32.ptr, gate.ptr, rows * ffn, stream=stream, runtime=runtime)
            runner.kernels.f32_to_bf16(up_f32.ptr, up.ptr, rows * ffn, stream=stream, runtime=runtime)
            runner.kernels.silu_mul_separate_out_bf16(gate.ptr, up.ptr, act.ptr, rows, ffn,
                                                     stream=stream, runtime=runtime)
            gemm_bf16_f32(act.ptr, layer.down_w.ptr, down_f32.ptr, ffn, hidden, q6_scratch, "down_proj")
            runner.kernels.f32_to_bf16(down_f32.ptr, down_bf16.ptr, rows * hidden,
                                       stream=stream, runtime=runtime)
            runner.kernels.vv_scale_residual_bf16(
                hidden_rows.ptr, down_bf16.ptr, runner._ones_hidden.ptr, hidden_rows.ptr,
                rows * hidden, hidden, library=runner.library, runtime=runtime)
        runner._ctx_len_host[0] = start + rows
        copy_host_to_device(runner._ctx_len, host_array_ptr(runner._ctx_len_host))
    finally:
        for buf in scratch:
            free(buf)
        free(positions)
        free(counts)


LINEAR_VARIANTS = (
    ("dense_gemv_bf16_f32_out", q4_dense_gemv_bf16_f32_out),
    ("dense_gemv_f32_bf16w_f32_out", q4_dense_gemv_f32_bf16w_f32_out),
    ("dense_gemv_out_bf16", q4_dense_gemv_out_bf16),
    ("dense_dual_gemv_out_bf16", q4_dense_dual_gemv_out_bf16),
)


def register_vibevoice_q4_kernels(backend):
    for name, fn in LINEAR_VARIANTS:
        key = KernelKey(backend, name, "q4_k_m", "strict")
        if not is_registered(key):
            register(key, fn)
    key = KernelKey(backend, "vibevoice_prefill", "q4_k_m", "q4")
    if not is_registered(key):
        register(key, q4_prefill)
