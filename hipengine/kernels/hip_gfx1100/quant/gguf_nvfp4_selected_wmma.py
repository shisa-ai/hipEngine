"""Grouped WMMA projection of selected raw NVFP4 experts for Inkling prompt blocks.

One wave32 computes 32 output columns by 64 compact rows of one expert. NVFP4 sub-blocks are
dequantized to FP16 in registers; activations are FP16, either one value or a high plus a residual
part that the same weights multiply; accumulation is FP32. Routes arrive grouped by expert.
"""

import ctypes
from functools import lru_cache, partial
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError, capture_stream
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=4)
def _library(arch):
    return build_hip(sources=[Path(__file__).with_suffix('.hip')], family='gguf_nvfp4_selected_wmma',
                     profile='prefill', target_arch=arch, output_name='gguf_nvfp4_selected_wmma.so')


# Weight layouts the kernel reads: NVFP4 in one FP16 part, Q5_K and Q8_0 in a high and a residual part.
WEIGHT_FORMATS = ('nvfp4', 'q5_k', 'q8_0')


@lru_cache(maxsize=16)
def _symbol(arch, parts, fmt='nvfp4'):
    kind = 'f16_parts' if parts == 2 else 'f16'
    fn = getattr(_library(arch), f'hipengine_gguf_{fmt}_selected_wmma_{kind}_f32_out')
    # (ten pointers, groups, in, out, group count pointer, stream)
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
    fn.restype = ctypes.c_int
    return fn


@lru_cache(maxsize=4)
def _schedule_symbol(arch):
    fn = _library(arch).hipengine_nvfp4_route_schedule
    # (ids, scales, routes, experts, top, seven pointers, capacity, stream)
    fn.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 7
                   + [ctypes.c_int, ctypes.c_void_p])
    fn.restype = ctypes.c_int
    return fn


@lru_cache(maxsize=4)
def _parts_symbol(arch):
    fn = getattr(_library(arch), 'hipengine_fp16_parts')
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def launch_wmma(x_hi, x_lo, expert_start, group_expert, group_first_row, row_source, row_route, outer, weight, out,
                groups, n, m, *, parts, target_arch, stream=None, group_count=None, fmt='nvfp4'):
    """Project grouped routes: `out[row_route[c]]` is expert `e(c)` applied to row `row_source[c]` of `x`.

    `expert_start` is int64 ``[experts + 1]`` over compact rows. Each of the `groups` pairs in
    `group_expert` and `group_first_row` names an expert and the first of 64 of its compact rows.
    A null `row_source` reads compact row `c` of `x`; a null `row_route` stores compact row `c`;
    a null `outer` applies no per-expert scale. `x_lo` is required for two parts.

    With `group_count` (a device int holding the number of groups of a schedule built on the
    device) `groups` is the grid's fixed height and the groups beyond the count are not read.
    """
    status = _symbol(target_arch, parts, fmt)(x_hi, x_lo, expert_start, group_expert, group_first_row, row_source,
                                             row_route, outer, weight, out, groups, n, m, group_count,
                                             capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, f'GGUF {fmt} grouped WMMA projection failed')


def launch_route_schedule(ids, scaled, routes, experts, top, start, group_expert, group_first, source, route, mix,
                          total_groups, group_capacity, *, target_arch, stream=None):
    """Group a block's routes by expert on the device.

    `ids` and `scaled` are the expert and the mix scale of each of `routes` routes, `top` per row.
    `start` is int64 ``[experts + 1]`` over the compact rows; `group_expert` and `group_first` take
    up to `group_capacity` groups of 64 compact rows; `source` takes the row of each compact row,
    `route` the slot where each compact row's output goes and `mix` the scales in slot order, a
    row's routes in expert order; `total_groups` takes the number of groups, or -1 when they do not
    fit the capacity.
    """
    status = _schedule_symbol(target_arch)(ids, scaled, routes, experts, top, start, group_expert, group_first,
                                           source, route, mix, total_groups, group_capacity,
                                           capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 route schedule failed')


def launch_fp16_parts(x, hi, lo, count, parts, *, target_arch, stream=None):
    """Split `count` FP32 values into an FP16 high part and, for two parts, an FP16 residual."""
    status = _parts_symbol(target_arch)(x, hi, lo, count, parts, capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, 'GGUF NVFP4 grouped WMMA FP16 split failed')


for _backend in ('hip_gfx1100', 'hip_gfx1151'):
    _arch = hip_target_arch_for_backend(_backend)
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_wmma_f16_f32_out'),
             partial(launch_wmma, parts=1, target_arch=_arch))
    register(KernelKey(_backend, 'moe_linear', 'gguf_nvfp4', 'selected_wmma_f16_parts_f32_out'),
             partial(launch_wmma, parts=2, target_arch=_arch))
    register(KernelKey(_backend, 'inkling2_expert_pointwise', 'inkling2_mixed', 'fp16_parts'),
             partial(launch_fp16_parts, target_arch=_arch))
    # Q5_K and Q8_0 experts: each weight in a high and a residual FP16 part.
    for _fmt, _quant in (('q5_k', 'gguf_q5_k'), ('q8_0', 'gguf_q8_0')):
        register(KernelKey(_backend, 'moe_linear', _quant, 'selected_wmma_f16_parts_f32_out'),
                 partial(launch_wmma, parts=2, fmt=_fmt, target_arch=_arch))
