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


@lru_cache(maxsize=2)
def _direct_symbol(blend, target_arch):
    lib = _build(target_arch)
    launch = (lib.hipengine_inkling2_controller_blend_norm_direct if blend
              else lib.hipengine_inkling2_controller_input_direct)
    pointers = 12 if blend else 7
    launch.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return launch


def _triples(values):
    values = [int(value) for value in values]
    return values + [0] * (3 - len(values))


def launch_controller_input_direct(inputs, messages, out, rows, width, count, eps, *, stream=None,
                                   target_arch=None):
    """launch_controller_input over input and message rows that stay where they are (one pointer each)."""
    status = _direct_symbol(False, target_arch)(*_triples(inputs), *_triples(messages), out, rows, width, count, eps,
                                                capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 controller input failed")


def launch_controller_blend_norm_direct(inputs, messages, forget, gain, weights, out, rows, width, count, eps, *,
                                        stream=None, target_arch=None):
    """launch_controller_blend_norm over in-place input, message and weight rows."""
    status = _direct_symbol(True, target_arch)(
        *_triples(inputs), *_triples(messages), forget, gain, *_triples(weights), out, rows, width, count, eps,
        capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 controller blend failed")


@lru_cache(maxsize=2)
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
    register(KernelKey(_backend, "inkling2_controller", "inkling2_mixed", "input_direct"),
             launch_controller_input_direct)
    register(KernelKey(_backend, "inkling2_controller", "inkling2_mixed", "blend_norm_direct"),
             launch_controller_blend_norm_direct)
