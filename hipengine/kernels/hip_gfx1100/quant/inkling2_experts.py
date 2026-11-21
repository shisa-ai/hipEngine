"""gfx1100 packed expert projection with FP32 activations and output."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=1)
def _expert_library():
    path = Path(__file__).with_suffix(".hip")
    return build_hip(sources=[path], family="inkling2_experts", profile="decode",
                     output_name="inkling2_experts.so")


@lru_cache(maxsize=1)
def build_inkling2_expert():
    fn = _expert_library().hipengine_inkling2_expert_projection
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def launch_inkling2_expert(weights, scales, x, output, rows, columns, *, nvfp4, outer_scale=1.0, stream=0):
    status = build_inkling2_expert()(weights, scales, x, output, rows, columns, nvfp4, outer_scale, stream)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 packed expert projection failed with HIP status {status}")


@lru_cache(maxsize=1)
def _grouped_symbol():
    fn = _expert_library().hipengine_inkling2_grouped_expert_projection
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5
    fn.restype = ctypes.c_int
    return fn


def launch_grouped_experts(w, s, x, out, ids, outer, count, rows, columns, stride, *, nvfp4):
    status = _grouped_symbol()(w, s, x, out, ids, outer, count, rows, columns, stride, nvfp4)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 grouped expert projection failed with HIP status {status}")


@lru_cache(maxsize=1)
def _pointwise_symbol():
    fn = _expert_library().hipengine_inkling2_expert_pointwise
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_float] * 2
    fn.restype = ctypes.c_int
    return fn


def launch_expert_pointwise(x, mix, out, count, width, gate_cap=0.0, up_cap=0.0):
    status = _pointwise_symbol()(x, mix, out, count, width, gate_cap, up_cap)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 expert pointwise operation failed with HIP status {status}")


register(KernelKey("hip_gfx1100", "inkling2_expert_pointwise", "inkling2_mixed", "default"), launch_expert_pointwise)
register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "grouped"), launch_grouped_experts)
register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "default"), launch_inkling2_expert)
