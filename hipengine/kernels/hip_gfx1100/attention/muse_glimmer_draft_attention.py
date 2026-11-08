"""Muse Glimmer DFlash draft window attention (non-causal WMMA block-flash).

The drafter's block rows attend a sliding window of injected context plus the
whole block bidirectionally. This is a production flash route (online softmax),
not bit-exact to the scalar Gemma4 block kernel; the draft only proposes and the
exact target verify keeps DFlash IDs == true AR. See the .hip header.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import get_hip_runtime

_SOURCE = Path(__file__).with_suffix(".hip")
_P, _I64, _F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float


def build_muse_glimmer_draft_attention(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family="muse_glimmer_draft_attention",
                     profile="prefill", output_name="muse_glimmer_draft_attention.so",
                     load=load, require_cached=require_cached,
                     compiler_version=compiler_version)


@lru_cache(maxsize=1)
def _library():
    return build_muse_glimmer_draft_attention()


def muse_glimmer_draft_flash_attention(
    q_ptr: int, k_ptr: int, v_ptr: int, out_ptr: int, *,
    n_q_heads: int, n_kv_heads: int, keys: int,
    tokens: int, window: int, scale: float,
    stream: int = 0, library: ctypes.CDLL | None = None, runtime=None,
) -> None:
    """Non-causal windowed WMMA block-flash over the draft's contiguous KV.

    ``k``/``v`` point at the window start (key index 0 is the first kept key);
    ``keys`` is the span. Each block row's position is ``keys - tokens + token``
    in that frame, and the mask is ``key > position - window`` with no causal
    upper bound (block rows see later block rows). Requires head_dim 128 and
    n_q_heads a multiple of n_kv_heads (GQA <= 16).
    """

    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_muse_glimmer_draft_flash_wmma",
        (_P, _P, _P, _P, _I64, _I64, _I64, _I64, _I64, _F, _P),
        ctypes.c_int,
    )
    err = fn(q_ptr, k_ptr, v_ptr, out_ptr, n_q_heads, n_kv_heads, keys,
             tokens, window, float(scale), stream)
    runtime.check(int(err))