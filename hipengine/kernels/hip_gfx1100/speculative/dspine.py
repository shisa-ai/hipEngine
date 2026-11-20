"""Raw-pointer GPU DSpine bridge kernels (clean-room).

Elementwise glue for the adjacent-causal-injection code bridge
(arXiv:2609.36173); the heavy read/packed/write projections run on the reused
DFlash dense WMMA kernels. RED-pinned against kernels/cpu_reference/dspine.py.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime

_SOURCE = Path(__file__).with_name("dspine.hip")
_OUTPUT_NAME = "dspine.so"
_P, _I64, _F = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float


def build_dspine(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family="dspine", profile="decode",
                     output_name=_OUTPUT_NAME, load=load,
                     require_cached=require_cached, compiler_version=compiler_version)


@lru_cache(maxsize=1)
def _library():
    return build_dspine()


def _fn(name, argtypes):
    lib = _library()
    f = getattr(lib, name)
    f.argtypes = argtypes
    f.restype = ctypes.c_int
    return f


def dspine_rms_normalize(raw_ptr, normalized_ptr, rms_ptr, T, H, eps, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_dspine_rms_normalize", [_P, _P, _P, _I64, _I64, _F, _P])
    rt.check(int(f(raw_ptr, normalized_ptr, rms_ptr, T, H, eps, stream)))


def dspine_codes(projected_ptr, codebook_ptr, codes_ptr, anchor, T, C, eps, sqrt_c, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_dspine_codes", [_P, _P, _P, _I64, _I64, _I64, _F, _F, _P])
    rt.check(int(f(projected_ptr, codebook_ptr, codes_ptr, anchor, T, C, eps, sqrt_c, stream)))


def dspine_gate_message(packed_ptr, bias_ptr, message_ptr, T, R, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_dspine_gate_message", [_P, _P, _P, _I64, _I64, _P])
    rt.check(int(f(packed_ptr, bias_ptr, message_ptr, T, R, stream)))


def dspine_residual(raw_ptr, rms_ptr, delta_ptr, out_ptr, T, H, *, stream=0, runtime=None):
    rt = runtime or get_hip_runtime()
    f = _fn("hipengine_dspine_residual", [_P, _P, _P, _P, _I64, _I64, _P])
    rt.check(int(f(raw_ptr, rms_ptr, delta_ptr, out_ptr, T, H, stream)))


def dspine_concat_taps(src_ptrs, out_ptr, rows, H, *, stream=0, runtime=None):
    """Cast+concat ``len(src_ptrs)`` (<=5) FP32 [rows, H] aux chunks into a BF16
    [rows, n*H] fc-input buffer. Fuses the f32->bf16 cast and feature concat."""
    rt = runtime or get_hip_runtime()
    n = len(src_ptrs)
    if not 1 <= n <= 5:
        raise ValueError(f"dspine_concat_taps supports 1..5 taps, got {n}")
    ptrs = list(src_ptrs) + [src_ptrs[0]] * (5 - n)
    f = _fn("hipengine_dspine_concat_taps", [_P, _P, _P, _P, _P, _P, _I64, _I64, _I64, _P])
    rt.check(int(f(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], out_ptr, rows, H, n, stream)))