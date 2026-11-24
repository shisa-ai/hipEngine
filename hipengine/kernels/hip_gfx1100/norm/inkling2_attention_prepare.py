"""Raw-pointer launchers for Inkling 2 attention input preparation."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=2)
def _library(target_arch):
    return build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_attention_prepare",
                     profile="decode", target_arch=target_arch, output_name="inkling2_attention_prepare.so")


@lru_cache(maxsize=2)
def _build(clip, target_arch):
    lib = _library(target_arch)
    launch = lib.hipengine_inkling2_attention_prepare_clip if clip else lib.hipengine_inkling2_attention_prepare_rms
    launch.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float] * 2 + [ctypes.c_double]
                       + [ctypes.c_int] * 2) if clip else [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_float]
    launch.restype = ctypes.c_int
    return launch


def launch_prepare_clip(x, out, scale, rows, dim, heads, threshold, *, alpha=0.0, floor=1, first=0, logscale=False,
                        target_arch=None):
    """out = clip(x * scale, threshold), times 1 + alpha * log(max(1, position / floor)) per row."""
    status = _build(True, target_arch)(x, out, scale, rows, dim, heads, threshold, alpha, floor, first,
                                       1 if logscale else 0)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 attention clip failed with HIP status {status}")


def launch_prepare_rms(x, out, rows, dim, heads, eps, *, target_arch=None):
    """out = x / sqrt(mean(x * x) + eps), per head."""
    status = _build(False, target_arch)(x, out, rows, dim, heads, eps)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 attention normalization failed with HIP status {status}")


def clip_rows(x, scale, threshold):
    """Strict control: the host clip the reference session evaluates."""
    from hipengine.runtime.inkling2_mini_reference import _clip

    return _clip(x if scale is None else x * scale, threshold)


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "inkling2_attention_prepare", "inkling2_mixed", "host_clip"), clip_rows)
    register(KernelKey(_backend, "inkling2_attention_prepare", "inkling2_mixed", "clip"), launch_prepare_clip)
    register(KernelKey(_backend, "inkling2_attention_prepare", "inkling2_mixed", "rms"), launch_prepare_rms)
