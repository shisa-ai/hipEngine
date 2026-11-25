"""Selected raw NVFP4 projection with blockwise signed Q8 activations."""

import ctypes
from functools import lru_cache, partial
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError, capture_stream
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=10)
def _symbol(arch, *, fused=False, shared=False, counted=False):
    library = build_hip(sources=[Path(__file__).with_suffix('.hip')], family='gguf_nvfp4_q8_dot',
                        profile='decode', target_arch=arch, output_name='gguf_nvfp4_q8_dot.so')
    kind = 'split_activation_' if fused else 'shared_input_' if shared or counted else ''
    if counted:
        kind = 'counted_' + kind
    fn = getattr(library, f'hipengine_gguf_nvfp4_selected_{kind}q8_dot')
    fn.argtypes = ([ctypes.c_void_p] * 7 + [ctypes.c_int] * (5 if counted else 4) + ([ctypes.c_float] * 2 if fused else [])
                   + [ctypes.c_void_p])
    fn.restype = ctypes.c_int
    return fn


def launch_q8_dot(x, weight, out, ids, outer, routes, experts, n, m, *, scratch, target_arch, stream=None):
    status = _symbol(target_arch)(x, scratch, scratch + routes * n, weight, out,
                                  ids, outer, routes, n, m, experts,
                                  capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 Q8 dot projection failed')


def launch_shared_input_q8_dot(x, weight, out, ids, outer, routes, experts, n, m, *, scratch, target_arch,
                               stream=None):
    status = _symbol(target_arch, shared=True)(x, scratch, scratch + n, weight, out,
                                               ids, outer, routes, n, m, experts,
                                               capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 shared input Q8 dot projection failed')


def launch_counted_shared_input_q8_dot(x, weight, out, records, outer, tiles, rows, experts, n, m, *, scratch,
                                      target_arch, stream=None):
    """Project eight routes per row from stable same-expert records with trailing -1 slots."""
    status = _symbol(target_arch, counted=True)(x, scratch, scratch + rows * n, weight, out, records, outer,
                                                tiles, rows, n, m, experts,
                                                capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 counted Q8 dot projection failed')


def launch_split_activation_q8_dot(x, weight, out, ids, outer, routes, experts, n, m,
                                   gate_cap, up_cap, *, scratch, target_arch, stream=None):
    status = _symbol(target_arch, fused=True)(x, scratch, scratch + routes * n, weight, out,
                                              ids, outer, routes, n, m, experts, gate_cap, up_cap,
                                              capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 split activation Q8 dot projection failed')


def launch_counted_split_activation_q8_dot(x, weight, out, records, outer, tiles, rows, experts, n, m,
                                           gate_cap, up_cap, *, scratch, target_arch, stream=None):
    values = rows * 8 * n
    status = _symbol(target_arch, fused=True, counted=True)(x, scratch, scratch + values, weight, out, records, outer,
                                                            tiles, rows, n, m, experts, gate_cap, up_cap,
                                                            capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 counted split activation Q8 dot projection failed')


for _backend in ('hip_gfx1100', 'hip_gfx1151'):
    _arch = hip_target_arch_for_backend(_backend)
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_q8_dot_f32_f32_out'),
             partial(launch_q8_dot, target_arch=_arch))
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_shared_input_q8_dot_f32_f32_out'),
             partial(launch_shared_input_q8_dot, target_arch=_arch))
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_split_activation_q8_dot_f32_f32_out'),
             partial(launch_split_activation_q8_dot, target_arch=_arch))
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_counted_shared_input_q8_dot_f32_f32_out'),
             partial(launch_counted_shared_input_q8_dot, target_arch=_arch))
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_counted_split_activation_q8_dot_f32_f32_out'),
             partial(launch_counted_split_activation_q8_dot, target_arch=_arch))
