"""Raw-pointer FP32 Muse Glimmer vision primitives for the gfx11 backends.

Equations follow ``clip_graph_muse_glimmer`` (llama.cpp mtmd) and are pinned
against the f32-dequantized oracle in the live tier. Linear and attention
contractions use rocBLAS; these kernels cover the non-linear glue.
"""
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


def build_muse_glimmer_vision(*, load=True, require_cached=False, compiler_version=None):
    return build_hip(sources=[_SOURCE], family='muse_glimmer_vision', profile='prefill',
                     output_name='muse_glimmer_vision.so', load=load,
                     require_cached=require_cached, compiler_version=compiler_version)


def plan_muse_glimmer_vision_build(**kwargs):
    return plan_hip_build(sources=[_SOURCE], family='muse_glimmer_vision', profile='prefill',
                          output_name='muse_glimmer_vision.so', **kwargs)


@lru_cache(maxsize=1)
def _library():
    return build_muse_glimmer_vision()


def _launch(name, types, args, stream):
    fn = signed_kernel_fn(_library(), 'hipengine_muse_glimmer_vision_' + name,
                          (*types, _P), ctypes.c_int)
    get_hip_runtime().check(int(fn(*args, stream)))


def muse_glimmer_vision_layernorm(x, w, b, y, rows, width, eps, *, stream=0):
    _launch('layernorm', (_P, _P, _P, _P, _I, _I, _F), (x, w, b, y, rows, width, eps), stream)


def muse_glimmer_vision_elementwise(x, b, y, n, op, lo=0.0, *, stream=0):
    _launch('elementwise', (_P, _P, _P, _I, _I, _F), (x, b, y, n, op, lo), stream)


def muse_glimmer_vision_bias_add(x, b, y, rows, width, *, stream=0):
    _launch('bias_add', (_P, _P, _P, _I, _I), (x, b, y, rows, width), stream)


def muse_glimmer_vision_rope2d(x, pos_w, pos_h, y, tokens, heads, dim, theta, *, stream=0):
    _launch('rope2d', (_P, _P, _P, _P, _I, _I, _I, _F),
            (x, pos_w, pos_h, y, tokens, heads, dim, theta), stream)


def muse_glimmer_vision_transpose(x, y, tokens, heads, dim, dir, *, stream=0):
    _launch('transpose', (_P, _P, _I, _I, _I, _I), (x, y, tokens, heads, dim, dir), stream)


def muse_glimmer_vision_softmax(x, mask, rows, cols, nq, *, stream=0):
    _launch('softmax', (_P, _P, _I, _I, _I), (x, mask, rows, cols, nq), stream)


def muse_glimmer_vision_gather(x, perm, y, rows, width, *, stream=0):
    _launch('gather', (_P, _P, _P, _I, _I), (x, perm, y, rows, width), stream)


def muse_glimmer_vision_shuffle(x, y, gh, gw, hidden, f, *, stream=0):
    _launch('shuffle', (_P, _P, _I, _I, _I, _I), (x, y, gh, gw, hidden, f), stream)


def register_muse_glimmer_vision_kernels():
    for backend in ('hip_gfx1100', 'hip_gfx1151'):
        for layer, fn in (
            ('vision_layernorm', muse_glimmer_vision_layernorm),
            ('vision_elementwise', muse_glimmer_vision_elementwise),
            ('vision_bias_add', muse_glimmer_vision_bias_add),
            ('vision_rope2d', muse_glimmer_vision_rope2d),
            ('vision_transpose', muse_glimmer_vision_transpose),
            ('vision_softmax', muse_glimmer_vision_softmax),
            ('vision_gather', muse_glimmer_vision_gather),
            ('vision_shuffle', muse_glimmer_vision_shuffle),
        ):
            register(KernelKey(backend, layer, 'fp32', 'muse_glimmer'), fn, replace=True)


register_muse_glimmer_vision_kernels()