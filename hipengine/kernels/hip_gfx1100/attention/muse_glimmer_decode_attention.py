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
    n_splits: int, scale: float, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime=None,
) -> None:
    """One decode token over [key_begin, key_end) of the contiguous KV cache.

    ``partials`` must hold n_splits * n_q_heads * 130 float32 values.
    ``q``/``out`` are [n_q_heads, 128] BF16; KV is [max_ctx, n_kv_heads, 128].
    """

    if n_q_heads % n_kv_heads or n_q_heads % 4:
        raise ValueError("Glimmer decode attention requires n_q_heads % n_kv_heads == 0 and % 4 == 0")
    if key_end <= key_begin:
        raise ValueError("key range must be non-empty")
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_decode_attn",
        (_P, _P, _P, _P, _P, _I64, _I64, _I64, _I64, _I64, _F, _P),
        ctypes.c_int,
    )
    err = fn(q_ptr, k_ptr, v_ptr, out_ptr, partials_ptr,
             n_q_heads, n_kv_heads, key_begin, key_end, n_splits, float(scale), stream)
    runtime.check(int(err))


def register_muse_glimmer_decode_attention_kernels():
    for backend in ('hip_gfx1100', 'hip_gfx1151'):
        register(KernelKey(backend, 'attention_decode', 'bf16', 'muse_glimmer'),
                 muse_glimmer_decode_attention, replace=True)


register_muse_glimmer_decode_attention_kernels()