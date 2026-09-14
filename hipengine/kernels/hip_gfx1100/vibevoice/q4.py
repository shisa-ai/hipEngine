"""Q4_K_M backbone linear primitives for the VibeVoice Qwen2 runner.

Same signatures as the bf16 dense primitives the runtime already calls,
so the runner is unchanged apart from weight content (raw Q4_K block
bytes from the merged GGUF) and which kernels namespace is resolved.
The fused gate+up dual GEMV is emulated as two Q4 GEMVs (f32 out) plus
f32->bf16 casts into the fused dual layout silu_mul_dual consumes;
its f32 scratch is cached per output width.
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

# GGML K-quant type ids used by llama-quantize's Q4_K_M ruleset; the
# sensitive tensors (attn_v, ffn_down) are adaptively upgraded per layer.
GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 12, 13, 14


def _pick_gemv(w_ptr, bf16_in, bf16_out):
    """Select the raw-block GEMV family for the weight's GGML type."""
    weight_type = GGUF_WEIGHT_TYPES.get(int(w_ptr), GGML_Q4_K)
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

_F32_SCRATCH: dict[int, object] = {}


def _scratch_f32(key: tuple[int, int]):
    buf = _F32_SCRATCH.get(key)
    if buf is None:
        buf = malloc(key[1] * 4)
        _F32_SCRATCH[key] = buf
    return buf


def close_q4_scratch() -> None:
    for buf in _F32_SCRATCH.values():
        free(buf)
    _F32_SCRATCH.clear()


def q4_dense_gemv_bf16_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    _pick_gemv(w_ptr, bf16_in=True, bf16_out=False)(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_gemv_f32_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    _pick_gemv(w_ptr, bf16_in=False, bf16_out=False)(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_gemv_out_bf16(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    _pick_gemv(w_ptr, bf16_in=True, bf16_out=True)(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_dual_gemv_out_bf16(x_ptr, gate_w_ptr, up_w_ptr, out_ptr, rows,
                                in_features, out_features, out_width, **kwargs):
    """Two Q4 GEMVs + casts into the fused [gate | up] bf16 dual layout."""
    if rows != 1:
        raise ValueError("q4 dual gemv emulation is decode-only (rows=1)")
    runtime = kwargs.get("runtime") or get_hip_runtime()
    stream = kwargs.get("stream", 0)
    gate_f32 = _scratch_f32((0, out_features))
    up_f32 = _scratch_f32((1, out_features))
    _pick_gemv(gate_w_ptr, bf16_in=True, bf16_out=False)(
        x_ptr, gate_w_ptr, gate_f32.ptr, rows, in_features, out_features,
        stream=stream, runtime=runtime,
    )
    _pick_gemv(up_w_ptr, bf16_in=True, bf16_out=False)(
        x_ptr, up_w_ptr, up_f32.ptr, rows, in_features, out_features,
        stream=stream, runtime=runtime,
    )
    f32_to_bf16(gate_f32.ptr, out_ptr, out_features, stream=stream, runtime=runtime)
    f32_to_bf16(up_f32.ptr, out_ptr + out_features * 2, out_features, stream=stream, runtime=runtime)


def q4_dense_gemv_f32_bf16w_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    """o_proj: f32 activations against Q4 blocks, f32 out."""
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
