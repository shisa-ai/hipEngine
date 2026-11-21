"""Raw-pointer launcher for Inkling 2 target short convolution."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=1)
def _build():
    lib = build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_short_conv",
                    profile="decode", output_name="inkling2_short_conv.so")
    launch = lib.hipengine_inkling2_short_conv
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    launch.restype = ctypes.c_int
    return launch


def launch_short_conv(history, x, taps, out, channels, width, scale):
    status = _build()(history, x, taps, out, channels, width, scale)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 short convolution failed with HIP status {status}")


register(KernelKey("hip_gfx1100", "inkling2_short_conv", "inkling2_mixed", "default"), launch_short_conv)
