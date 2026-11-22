"""HIP F32 projections over raw GGUF NVFP4 weight rows."""

from __future__ import annotations

import ctypes
from functools import lru_cache, partial
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=None)
def _projection(target_arch):
    library = build_hip(sources=[Path(__file__).with_suffix(".hip")], family="gguf_nvfp4_gemv",
                        profile="decode", target_arch=target_arch, output_name="gguf_nvfp4_gemv.so")
    fn = library.hipengine_gguf_nvfp4_gemv_f32_f32_out
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _launch(x, weight, output, rows, in_features, out_features, *, target_arch, outer_scale=1.0, stream=0):
    status = _projection(target_arch)(x, weight, output, rows, in_features, out_features, outer_scale, stream)
    if status != HIP_SUCCESS:
        raise HipError(f"GGUF NVFP4 projection failed with HIP status {status}")


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "linear", "gguf_nvfp4", "gemv_f32_f32_out"),
             partial(_launch, target_arch=hip_target_arch_for_backend(_backend)))
