"""Muse Glimmer WMMA flash prefill attention (GQA-packed, contiguous KV)."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import get_hip_runtime

_SOURCE = Path(__file__).with_suffix(".hip")
_P, _I64, _F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float


def build_muse_glimmer_flash_prefill(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family="muse_glimmer_flash_prefill",
                     profile="prefill", output_name="muse_glimmer_flash_prefill.so",
                     load=load, require_cached=require_cached,
                     compiler_version=compiler_version)


@lru_cache(maxsize=1)
def _library():
    return build_muse_glimmer_flash_prefill()


def muse_glimmer_flash_prefill(
    q_ptr: int, k_ptr: int, v_ptr: int, out_ptr: int, *,
    n_q_heads: int, n_kv_heads: int, key_begin: int, key_end: int,
    tokens: int, window: int, scale: float, stream: int = 0,
    library: ctypes.CDLL | None = None, runtime=None,
) -> None:
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_flash_prefill_wmma",
        (_P, _P, _P, _P, _I64, _I64, _I64, _I64, _I64, _I64, _F, _P),
        ctypes.c_int,
    )
    err = fn(q_ptr, k_ptr, v_ptr, out_ptr, n_q_heads, n_kv_heads, key_begin,
             key_end, tokens, window, float(scale), stream)
    runtime.check(int(err))