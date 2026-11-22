"""Selected raw NVFP4 projection with blockwise signed Q8 activations."""

import ctypes
from functools import lru_cache, partial
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=2)
def _symbol(arch):
    library = build_hip(sources=[Path(__file__).with_suffix('.hip')], family='gguf_nvfp4_q8_dot',
                        profile='decode', target_arch=arch, output_name='gguf_nvfp4_q8_dot.so')
    fn = library.hipengine_gguf_nvfp4_selected_q8_dot
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
    fn.restype = ctypes.c_int
    return fn


def launch_q8_dot(x, weight, out, ids, outer, routes, experts, n, m, *, scratch, target_arch):
    status = _symbol(target_arch)(x, scratch, scratch + routes * n, weight, out,
                                  ids, outer, routes, n, m, experts)
    if status != HIP_SUCCESS:
        raise HipError(f'GGUF NVFP4 Q8 dot projection failed with HIP status {status}')


for _backend in ('hip_gfx1100', 'hip_gfx1151'):
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_q8_dot_f32_f32_out'),
             partial(launch_q8_dot, target_arch=hip_target_arch_for_backend(_backend)))
