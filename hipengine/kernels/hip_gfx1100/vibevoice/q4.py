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

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free
from hipengine.kernels.registry import KernelKey, is_registered, register
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
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
    """Correctness-first q4 prefill: row-by-row through the q4 decode path.

    Each prompt row goes through forward_layers (which resolves the q4
    linear primitives), so prefill needs no batched q4 GEMM yet; that
    is the speed follow-up (naive gguf_q4_k_prefill_* or the t16/wmma
    family after a load-time repack).
    """
    from hipengine.core.runtime import MemcpyKind

    width = runner.spec.hidden_size * 2
    for row in range(rows):
        runner.runtime.memcpy(
            runner._hidden.ptr, hidden_rows.ptr + row * width, width,
            MemcpyKind.DEVICE_TO_DEVICE,
        )
        runner.forward_layers(start + row)
        runner.runtime.memcpy(
            hidden_rows.ptr + row * width, runner._hidden.ptr, width,
            MemcpyKind.DEVICE_TO_DEVICE,
        )


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
