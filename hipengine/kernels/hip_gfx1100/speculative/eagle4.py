"""Raw-pointer GPU EAGLE-4 drafter kernels (clean-room): RoPE-only + top-K retrieve."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime

_SOURCE = Path(__file__).with_name("eagle4.hip")
_OUTPUT_NAME = "eagle4.so"
_P, _I64, _I, _F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float


def build_eagle4(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family="eagle4", profile="decode",
                     output_name=_OUTPUT_NAME, load=load,
                     require_cached=require_cached, compiler_version=compiler_version)


@lru_cache(maxsize=1)
def _library():
    return build_eagle4()


def _fn(name, argtypes):
    f = getattr(_library(), name)
    f.argtypes = argtypes
    f.restype = ctypes.c_int
    return f


def eagle4_rope(x_ptr, positions_ptr, rows, n_heads, head_dim, theta, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_eagle4_rope", [_P, _P, _I64, _I, _I, _F, _P])
    rt.check(int(f(x_ptr, positions_ptr, rows, n_heads, head_dim, theta, stream)))


def eagle4_topk(scores_ptr, out_ids_ptr, rows, vocab, K, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_eagle4_topk", [_P, _P, _I64, _I64, _I, _P])
    rt.check(int(f(scores_ptr, out_ids_ptr, rows, vocab, K, stream)))


def eagle4_concat(a_ptr, b_ptr, out_ptr, rows, H, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_eagle4_concat", [_P, _P, _P, _I64, _I64, _P])
    rt.check(int(f(a_ptr, b_ptr, out_ptr, rows, H, stream)))


def eagle4_concat3(a_ptr, b_ptr, c_ptr, out_ptr, rows, H, *, stream=0, runtime=None):
    """Row-concat three [rows, H] BF16 buffers into [rows, 3H] (EAGLE fc input)."""
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_eagle4_concat3", [_P, _P, _P, _P, _I64, _I64, _P])
    rt.check(int(f(a_ptr, b_ptr, c_ptr, out_ptr, rows, H, stream)))