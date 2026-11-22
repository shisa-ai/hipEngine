"""gfx1100 packed expert projection with FP32 activations and output."""

from __future__ import annotations

import ctypes
from functools import lru_cache, partial
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.cpu_reference.inkling2_mini import split_expert_activation
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=2)
def _expert_library(target_arch=None):
    path = Path(__file__).with_suffix(".hip")
    return build_hip(sources=[path], family="inkling2_experts", profile="decode",
                     target_arch=target_arch, output_name="inkling2_experts.so")


@lru_cache(maxsize=1)
def build_inkling2_expert():
    fn = _expert_library().hipengine_inkling2_expert_projection
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def launch_inkling2_expert(weights, scales, x, output, rows, columns, *, nvfp4, outer_scale=1.0, stream=0,
                           expanded_scales=False):
    status = build_inkling2_expert()(weights, scales, x, output, rows, columns, nvfp4, outer_scale, stream,
                                    expanded_scales)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 packed expert projection failed with HIP status {status}")


@lru_cache(maxsize=1)
def _grouped_symbol():
    fn = _expert_library().hipengine_inkling2_grouped_expert_projection
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8
    fn.restype = ctypes.c_int
    return fn


def launch_grouped_experts(w, s, x, out, ids, outer, count, rows, columns, stride, *, nvfp4, wave_rows=False, tokens=1,
                           expanded_scales=False):
    status = _grouped_symbol()(w, s, x, out, ids, outer, count, rows, columns, stride, nvfp4, wave_rows, tokens,
                               expanded_scales)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 grouped expert projection failed with HIP status {status}")


@lru_cache(maxsize=1)
def _tiled_symbol():
    fn = _expert_library().hipengine_inkling2_tiled_experts
    fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8
    fn.restype = ctypes.c_int
    return fn


def launch_tiled_experts(w, s, x, out, routes, outer, count, rows, columns, stride, *,
                         nvfp4, tiles, tile_width, expanded_scales=False):
    status = _tiled_symbol()(w, s, x, out, routes, outer, count, rows, columns, stride, nvfp4,
                             expanded_scales, tiles, tile_width)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 tiled expert projection failed with HIP status {status}")


@lru_cache(maxsize=1)
def _pointwise_symbol():
    fn = _expert_library().hipengine_inkling2_expert_pointwise
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_float] * 2 + [ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def launch_expert_pointwise(x, mix, out, count, width, gate_cap=0.0, up_cap=0.0, *, tokens=1):
    status = _pointwise_symbol()(x, mix, out, count, width, gate_cap, up_cap, tokens)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 expert pointwise operation failed with HIP status {status}")


@lru_cache(maxsize=2)
def _mix_exact_symbol(target_arch):
    fn = _expert_library(target_arch).hipengine_inkling2_expert_mix_exact
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
    fn.restype = ctypes.c_int
    return fn


def launch_expert_mix_exact(down, mix, out, count, width, *, tokens=1, target_arch=None):
    status = _mix_exact_symbol(target_arch)(down, mix, out, count, width, tokens)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 exact expert mix failed with HIP status {status}")


@lru_cache(maxsize=2)
def _split_activation_symbol(target_arch):
    fn = _expert_library(target_arch).hipengine_inkling2_split_activation
    fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2 + [ctypes.c_float] * 2
    fn.restype = ctypes.c_int
    return fn


def launch_split_expert_activation(gate_up, out, routes, width, gate_cap, up_cap, *, target_arch=None):
    status = _split_activation_symbol(target_arch)(gate_up, out, routes, width, gate_cap, up_cap)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 split expert activation failed with HIP status {status}")


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "inkling2_expert_pointwise", "inkling2_mixed", "ordered_mix_exact"),
             launch_expert_mix_exact)
    register(KernelKey(_backend, "inkling2_expert_pointwise", "inkling2_mixed", "split_activation_device"),
             launch_split_expert_activation)
    register(KernelKey(_backend, "inkling2_expert_pointwise", "inkling2_mixed", "split_activation_host"),
             split_expert_activation)


register(KernelKey("hip_gfx1100", "inkling2_expert_pointwise", "inkling2_mixed", "default"), launch_expert_pointwise)
launch_wave_grouped_experts = partial(launch_grouped_experts, wave_rows=True)
register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "grouped_wave_rows"),
         launch_wave_grouped_experts)
register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "grouped"), launch_grouped_experts)
register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "tiled"), launch_tiled_experts)
register(KernelKey("hip_gfx1100", "inkling2_expert_projection", "inkling2_mixed", "default"), launch_inkling2_expert)
