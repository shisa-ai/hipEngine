"""Raw-pointer launchers for Inkling 2 loop-controller input, blend and normalization."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError, capture_stream
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=2)
def _build(target_arch):
    return build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_controller",
                     profile="decode", target_arch=target_arch, output_name="inkling2_controller.so")


@lru_cache(maxsize=4)
def _symbol(blend, target_arch):
    lib = _build(target_arch)
    launch = (lib.hipengine_inkling2_controller_blend_norm if blend
              else lib.hipengine_inkling2_controller_input)
    pointers = 6 if blend else 3
    launch.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return launch


def launch_controller_input(x, message, out, rows, width, count, eps, *, stream=None, target_arch=None):
    status = _symbol(False, target_arch)(x, message, out, rows, width, count, eps,
                                         capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 controller input failed")


def launch_controller_blend_norm(x, message, forget, gain, weight, out, rows, width, count, eps, *,
                                 stream=None, target_arch=None):
    status = _symbol(True, target_arch)(x, message, forget, gain, weight, out, rows, width, count, eps,
                                        capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 controller blend failed")


def blend_controller_norm(x, message, forget, gain, weight, *, eps=1e-6):
    """Strict control: the host blend and normalization the reference session evaluates."""
    from hipengine.runtime.inkling2_mini_reference import _blend_controller, _rms

    return _rms(_blend_controller(x, message, forget, gain), eps) * weight


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "inkling2_controller", "inkling2_mixed", "host_blend_norm"),
             blend_controller_norm)
    register(KernelKey(_backend, "inkling2_controller", "inkling2_mixed", "input"), launch_controller_input)
    register(KernelKey(_backend, "inkling2_controller", "inkling2_mixed", "blend_norm"),
             launch_controller_blend_norm)
