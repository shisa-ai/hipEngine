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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_gemv_bf16_bf16_out,
    gguf_q4_k_gemv_bf16_f32_out,
    gguf_q4_k_gemv_f32_f32_out,
)

_F32_SCRATCH: dict[int, object] = {}


def _scratch_f32(width: int):
    buf = _F32_SCRATCH.get(width)
    if buf is None:
        buf = malloc(width * 4)
        _F32_SCRATCH[width] = buf
    return buf


def close_q4_scratch() -> None:
    for buf in _F32_SCRATCH.values():
        free(buf)
    _F32_SCRATCH.clear()


def q4_dense_gemv_bf16_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    gguf_q4_k_gemv_bf16_f32_out(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_gemv_f32_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    gguf_q4_k_gemv_f32_f32_out(
        x_ptr, w_ptr, out_ptr, rows, in_features, out_features,
        stream=kwargs.get("stream", 0),
        runtime=kwargs.get("runtime") or get_hip_runtime(),
    )


def q4_dense_gemv_out_bf16(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    gguf_q4_k_gemv_bf16_bf16_out(
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
    gate_f32 = _scratch_f32(out_features)
    up_f32 = _scratch_f32(out_features)
    gguf_q4_k_gemv_bf16_f32_out(
        x_ptr, gate_w_ptr, gate_f32.ptr, rows, in_features, out_features,
        stream=stream, runtime=runtime,
    )
    gguf_q4_k_gemv_bf16_f32_out(
        x_ptr, up_w_ptr, up_f32.ptr, rows, in_features, out_features,
        stream=stream, runtime=runtime,
    )
    f32_to_bf16(gate_f32.ptr, out_ptr, out_features, stream=stream, runtime=runtime)
    f32_to_bf16(up_f32.ptr, out_ptr + out_features * 2, out_features, stream=stream, runtime=runtime)


def q4_dense_gemv_f32_bf16w_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
    """o_proj: f32 activations against Q4 blocks, f32 out."""
    q4_dense_gemv_f32_f32_out(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs)


def q4_prefill(runner, hidden_rows, rows, start):
    """Batched causal prefill through the naive Q4_K prefill GEMM."""
    runner._prefill_q4(hidden_rows, rows, start)


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
