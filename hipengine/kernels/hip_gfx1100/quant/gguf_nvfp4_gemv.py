"""HIP F32 projections over raw GGUF NVFP4 weight rows."""

from __future__ import annotations

import ctypes
from functools import lru_cache, partial
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError, capture_stream
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register


def _ambient(stream):
    return capture_stream() if not stream else stream



@lru_cache(maxsize=None)
def _library(target_arch):
    return build_hip(sources=[Path(__file__).with_suffix(".hip")], family="gguf_nvfp4_gemv",
                     profile="decode", target_arch=target_arch, output_name="gguf_nvfp4_gemv.so")


@lru_cache(maxsize=None)
def _projection(target_arch):
    fn = _library(target_arch).hipengine_gguf_nvfp4_gemv_f32_f32_out
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _launch(x, weight, output, rows, in_features, out_features, *, target_arch, outer_scale=1.0, stream=0):
    status = _projection(target_arch)(x, weight, output, rows, in_features, out_features, outer_scale, _ambient(stream))
    if status != HIP_SUCCESS:
        raise HipError(status, "GGUF NVFP4 projection failed")


@lru_cache(maxsize=None)
def _selected(target_arch, register_fp4):
    variant = "selected_register_fp4" if register_fp4 else "selected"
    fn = getattr(_library(target_arch), f"hipengine_gguf_nvfp4_{variant}_f32_f32_out")
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _launch_selected(x, weight, output, ids, scales, rows, experts, in_features, out_features, *, target_arch,
                     stream=0, register_fp4=False):
    status = _selected(target_arch, register_fp4)(x, weight, output, ids, scales, rows, experts, in_features,
                                                  out_features, _ambient(stream))
    if status != HIP_SUCCESS:
        raise HipError(status, "GGUF NVFP4 selected projection failed")


@lru_cache(maxsize=None)
def _tiled(target_arch, shared):
    layout = "_shared_input" if shared else ""
    fn = getattr(_library(target_arch), f"hipengine_gguf_nvfp4_selected_tile4{layout}_f32_f32_out")
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * (5 + shared) + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _launch_tiled(x, weight, output, records, scales, tiles, rows, experts, n, m, *, target_arch, stream=0):
    status = _tiled(target_arch, False)(x, weight, output, records, scales, tiles, rows, experts, n, m, _ambient(stream))
    if status != HIP_SUCCESS:
        raise HipError(status, "GGUF NVFP4 tiled projection failed")


def _launch_shared_tiled(x, weight, output, records, scales, tiles, rows, experts, n, m, *, target_arch,
                         routes_per_input, stream=0):
    status = _tiled(target_arch, True)(x, weight, output, records, scales, tiles, rows, experts, n, m,
                                      routes_per_input, _ambient(stream))
    if status != HIP_SUCCESS:
        raise HipError(status, "GGUF NVFP4 shared-input tiled projection failed")


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    _arch = hip_target_arch_for_backend(_backend)
    register(KernelKey(_backend, "linear", "gguf_nvfp4", "gemv_f32_f32_out"), partial(_launch, target_arch=_arch))
    register(KernelKey(_backend, "moe_linear", "gguf_nvfp4", "selected_f32_f32_out"),
             partial(_launch_selected, target_arch=_arch))
    register(KernelKey(_backend, "moe_linear", "gguf_nvfp4", "selected_register_fp4_f32_f32_out"),
             partial(_launch_selected, target_arch=_arch, register_fp4=True))
    register(KernelKey(_backend, "moe_linear", "gguf_nvfp4", "selected_tile4_f32_f32_out"),
             partial(_launch_tiled, target_arch=_arch))
    register(KernelKey(_backend, "moe_linear", "gguf_nvfp4", "selected_tile4_shared_input_f32_f32_out"),
             partial(_launch_shared_tiled, target_arch=_arch))
