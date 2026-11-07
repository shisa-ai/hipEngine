"""Muse Glimmer flash-decoding attention (key-split, GQA-packed, BF16 KV).

Registered decode-attention owner for the Glimmer geometry (32:2, head_dim
128). The gemma4 strict decode kernel remains the registered fallback; this
variant is a different association (online softmax) gated by the CPU-reference
outer floor. See the .hip header for the layout and the campaign record for
the measured prize (llama.cpp flash decode 0.45 ms/step vs strict ~20-134).
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_suffix('.hip')
_P, _I64, _F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float


def build_muse_glimmer_decode_attention(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family='muse_glimmer_decode_attention',
                     profile='decode', output_name='muse_glimmer_decode_attention.so',
                     load=load, require_cached=require_cached,
                     compiler_version=compiler_version)


def plan_muse_glimmer_decode_attention_build(**kwargs):
    return plan_hip_build(sources=[_SOURCE], family='muse_glimmer_decode_attention',
                          profile='decode', output_name='muse_glimmer_decode_attention.so',
                          **kwargs)


@lru_cache(maxsize=1)
def _library():
    return build_muse_glimmer_decode_attention()


def muse_glimmer_decode_attention(
    q_ptr: int, k_ptr: int, v_ptr: int, out_ptr: int, partials_ptr: int,
    *, n_q_heads: int, n_kv_heads: int, key_begin: int, key_end: int,
    n_splits: int, scale: float, tokens: int = 1, window: int = 0,
    key_end_dev: int = 0, gate_ptr: int = 0, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime=None,
) -> None:
    """Attention over [key_begin, key_end) of the contiguous KV cache.

    A nonzero ``gate_ptr`` (BF16, same shape as ``out``, requires
    ``n_splits > 1``) applies the Glimmer sigmoid attention gate inside the
    combine pass, bit-identical to a following ``gguf_gate_mul_bf16``.

    ``tokens==1`` is decode; ``tokens>1`` is a causal prefill block whose rows
    occupy the last ``tokens`` positions ending at ``key_end``. ``window>0``
    clamps each row to its sliding window. ``partials`` must hold
    n_splits * tokens * n_q_heads * 130 float32 values (unused when
    n_splits == 1, which writes output directly).
    """

    if n_q_heads % n_kv_heads or n_q_heads % 4:
        raise ValueError("Glimmer decode attention requires n_q_heads % n_kv_heads == 0 and % 4 == 0")
    if key_end <= key_begin:
        raise ValueError("key range must be non-empty")
    if n_splits <= 0 or tokens <= 0:
        raise ValueError("n_splits and tokens must be positive")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    if gate_ptr:
        if n_splits <= 1:
            raise ValueError("the gated combine requires n_splits > 1")
        fn = signed_kernel_fn(
            library, "hipengine_muse_glimmer_decode_attn_gated",
            (_P, _P, _P, _P, _P, _P, _I64, _I64, _I64, _I64, _I64, _I64, _I64, _F, _P, _P),
            ctypes.c_int,
        )
        runtime.check(int(fn(q_ptr, k_ptr, v_ptr, gate_ptr, out_ptr, partials_ptr,
                             n_q_heads, n_kv_heads, key_begin, key_end, n_splits, tokens,
                             window, float(scale), key_end_dev, stream)))
        return
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_decode_attn",
        (_P, _P, _P, _P, _P, _I64, _I64, _I64, _I64, _I64, _I64, _I64, _F, _P, _P),
        ctypes.c_int,
    )
    err = fn(q_ptr, k_ptr, v_ptr, out_ptr, partials_ptr,
             n_q_heads, n_kv_heads, key_begin, key_end, n_splits, tokens, window,
             float(scale), key_end_dev, stream)
    runtime.check(int(err))


def muse_glimmer_kv_append(dst_ptr: int, src_ptr: int, pos_ptr: int, *,
                           row_elems: int, n: int, stream: int = 0,
                           library: ctypes.CDLL | None = None, runtime=None) -> None:
    """Append n BF16 elements at row *pos (device int64) of a contiguous cache."""

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_kv_append",
        (_P, _P, _P, _I64, _I64, _P), ctypes.c_int,
    )
    runtime.check(int(fn(dst_ptr, src_ptr, pos_ptr, row_elems, n, stream)))


def muse_glimmer_qk_norm_rope_kv_append(
    q_ptr: int, k_ptr: int, v_ptr: int, q_weight_ptr: int, k_weight_ptr: int,
    cos_ptr: int, sin_ptr: int, pos_ptr: int, q_out_ptr: int,
    k_cache_ptr: int, v_cache_ptr: int, *, eps: float, n_q_heads: int,
    n_kv_heads: int, head_dim: int, max_positions: int, tokens: int = 1,
    stream: int = 0,
    library: ctypes.CDLL | None = None, runtime=None,
) -> None:
    """Fused Q/K head RMSNorm + rotary + K/V cache append (BF16 I/O).

    Bit-identical to bf16_to_f32 x2 -> head-norm partial rotary (256 threads,
    rotary_dim == head_dim) -> f32_to_bf16 x2 -> K/V append. ``tokens==1`` is
    decode (K/V appended at row ``*pos``); ``tokens>1`` runs each block row at
    ``pos[row]`` with row-offset Q/K/V, bit-identical to the rows-1 kernel per
    row (the speculative block-verify glue).
    """

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_qk_norm_rope_kv_append",
        (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _F, _I64, _I64, _I64, _I64, _I64, _P),
        ctypes.c_int,
    )
    runtime.check(int(fn(
        q_ptr, k_ptr, v_ptr, q_weight_ptr, k_weight_ptr, cos_ptr, sin_ptr, pos_ptr,
        q_out_ptr, k_cache_ptr, v_cache_ptr, float(eps), n_q_heads, n_kv_heads,
        head_dim, max_positions, tokens, stream,
    )))


def muse_glimmer_add_rmsnorm_rmsnorm(
    branch_ptr: int, residual_ptr: int, post_weight_ptr: int, pre_weight_ptr: int,
    residual_out_ptr: int, normed_out_ptr: int, *, rows: int, hidden_size: int,
    post_eps: float, pre_eps: float, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime=None,
) -> None:
    """Post-norm + residual add, then the next pre-norm, in one launch.

    Bit-identical to gemma4_add_rmsnorm_scale_bf16 (layer scalar 1) followed
    by gemma4_rmsnorm_f32w_bf16 on its output. ``residual_out`` must not alias
    the inputs.
    """

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_add_rmsnorm_rmsnorm",
        (_P, _P, _P, _P, _P, _P, _I64, _I64, _F, _F, _P), ctypes.c_int,
    )
    runtime.check(int(fn(
        branch_ptr, residual_ptr, post_weight_ptr, pre_weight_ptr, residual_out_ptr,
        normed_out_ptr, rows, hidden_size, float(post_eps), float(pre_eps), stream,
    )))


def register_muse_glimmer_decode_attention_kernels():
    for backend in ('hip_gfx1100', 'hip_gfx1151'):
        register(KernelKey(backend, 'attention_decode', 'bf16', 'muse_glimmer'),
                 muse_glimmer_decode_attention, replace=True)


register_muse_glimmer_decode_attention_kernels()