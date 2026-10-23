"""Raw-pointer FP32 Gemma 4 vision primitives for the gfx11 backends."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_suffix('.hip')
_P, _I, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float


def build_gemma4_vision(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family='gemma4_vision', profile='prefill',
                     output_name='gemma4_vision.so', load=load,
                     require_cached=require_cached, compiler_version=compiler_version)


def plan_gemma4_vision_build(**kwargs):
    return plan_hip_build(sources=[_SOURCE], family='gemma4_vision', profile='prefill',
                          output_name='gemma4_vision.so', **kwargs)


@lru_cache(maxsize=1)
def _library():
    return build_gemma4_vision()


def _launch(name, types, args, stream):
    fn = signed_kernel_fn(_library(), 'hipengine_gemma4_vision_' + name,
                          (*types, _P), ctypes.c_int)
    get_hip_runtime().check(int(fn(*args, stream)))


def gemma4_vision_norm(x, w, y, rows, width, eps, *, stream=0):
    _launch('norm', (_P, _P, _P, _I, _I, _F), (x, w, y, rows, width, eps), stream)


def gemma4_vision_elementwise(x, b, y, n, op, lo=0.0, hi=0.0, *, stream=0):
    _launch('elementwise', (_P, _P, _P, _I, _I, _F, _F), (x, b, y, n, op, lo, hi), stream)


def gemma4_vision_heads(x, positions, y, tokens, heads, dim, theta, mode, *, stream=0):
    _launch('heads', (_P, _P, _P, _I, _I, _I, _F, _I),
            (x, positions, y, tokens, heads, dim, theta, mode), stream)


def gemma4_vision_softmax(x, rows, cols, *, stream=0):
    _launch('softmax', (_P, _I, _I), (x, rows, cols), stream)


def gemma4_vision_pool(x, bias, scale, y, gh, gw, hidden, merge, *, stream=0):
    _launch('pool', (_P, _P, _P, _P, _I, _I, _I, _I),
            (x, bias, scale, y, gh, gw, hidden, merge), stream)


def register_gemma4_vision_kernels():
    for backend in ('hip_gfx1100', 'hip_gfx1151'):
        for layer, fn in (
            ('vision_rmsnorm', gemma4_vision_norm),
            ('vision_elementwise', gemma4_vision_elementwise),
            ('vision_heads', gemma4_vision_heads),
            ('vision_softmax', gemma4_vision_softmax),
            ('vision_pool', gemma4_vision_pool),
        ):
            register(KernelKey(backend, layer, 'fp32', 'gemma4'), fn, replace=True)


register_gemma4_vision_kernels()
