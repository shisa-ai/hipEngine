"""Raw-pointer launcher for Inkling 2 target short convolution."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=2)
def _build(target_arch):
    return build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_short_conv",
                     profile="decode", target_arch=target_arch, output_name="inkling2_short_conv.so")


@lru_cache(maxsize=4)
def _symbol(block, target_arch):
    lib = _build(target_arch)
    launch = lib.hipengine_inkling2_short_conv_block if block else lib.hipengine_inkling2_short_conv
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    if block:
        launch.argtypes += [ctypes.c_int, ctypes.c_int]
    launch.restype = ctypes.c_int
    return launch


def launch_short_conv(history, x, taps, out, channels, width, scale, *, target_arch=None):
    status = _symbol(False, target_arch)(history, x, taps, out, channels, width, scale)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 short convolution failed with HIP status {status}")


def launch_short_conv_block(history, x, taps, out, channels, width, scale, rows, *, dynamic_taps=False, target_arch=None):
    status = _symbol(True, target_arch)(history, x, taps, out, channels, width, scale, rows, dynamic_taps)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 block short convolution failed with HIP status {status}")


@lru_cache(maxsize=2)
def _dynamic_symbol(target_arch):
    launch = _build(target_arch).hipengine_inkling2_short_conv_dynamic
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_float]
    launch.restype = ctypes.c_int
    return launch


def launch_short_conv_dynamic(history, x, fixed, dynamic, heads, dim, width, rows, scale, *, target_arch=None):
    status = _dynamic_symbol(target_arch)(history, x, fixed, dynamic, heads, dim, width, rows, scale)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 projected dynamic convolution failed with HIP status {status}")


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "inkling2_short_conv", "inkling2_mixed", "default"), launch_short_conv)
    register(KernelKey(_backend, "inkling2_short_conv", "inkling2_mixed", "block"), launch_short_conv_block)
    register(KernelKey(_backend, "inkling2_short_conv", "inkling2_mixed", "projected_dynamic"), launch_short_conv_dynamic)
