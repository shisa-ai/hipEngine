"""gfx1100 packed expert projection with FP32 activations and output."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=1)
def build_inkling2_expert():
    path = Path(__file__).with_suffix(".hip")
    lib = build_hip(sources=[path], family="inkling2_experts", profile="decode",
                    output_name="inkling2_experts.so")
    fn = lib.hipengine_inkling2_expert_projection
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def launch_inkling2_expert(weights, scales, x, output, rows, columns, *, nvfp4, outer_scale=1.0, stream=0):
    status = build_inkling2_expert()(weights, scales, x, output, rows, columns, nvfp4, outer_scale, stream)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 packed expert projection failed with HIP status {status}")


register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "default"), launch_inkling2_expert)
