"""Raw-pointer launchers for Inkling residual-stack reads and row normalization."""

import ctypes
from functools import lru_cache
from math import sqrt
from pathlib import Path

import numpy as np

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError, capture_stream
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=2)
def _library(target_arch):
    return build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_residual",
                     profile="decode", target_arch=target_arch, output_name="inkling2_residual.so")


@lru_cache(maxsize=8)
def _build(name, target_arch, pointers, ints, floats=0, streams=1):
    fn = getattr(_library(target_arch), "hipengine_inkling2_" + name)
    fn.argtypes = ([ctypes.c_void_p] * pointers + [ctypes.c_int] * ints + [ctypes.c_float] * floats
                   + [ctypes.c_void_p] * streams)
    fn.restype = ctypes.c_int
    return fn


def _check(status, what):
    if status != HIP_SUCCESS:
        raise HipError(status, f"Inkling 2 {what} failed")


def launch_row_rms(x, out, rows, width, eps, *, stream=None, target_arch=None):
    """out = x / sqrt(mean(x * x) + eps) per row; `out` may alias `x`."""
    _check(_build("row_rms", target_arch, 2, 2, 1)(x, out, rows, width, eps,
                                                   capture_stream() if stream is None else stream),
           "row normalization")


def launch_row_norm_mul(x, weight, out, rows, width, eps, *, stream=None, target_arch=None):
    """out = x / sqrt(mean(x * x) + eps) * weight per row."""
    _check(_build("row_norm_mul", target_arch, 3, 2, 1)(x, weight, out, rows, width, eps,
                                                        capture_stream() if stream is None else stream),
           "normalized row")


def launch_row_add(a, b, out, rows, width, *, stream=None, target_arch=None):
    """out = a + b per row."""
    _check(_build("row_add", target_arch, 3, 2)(a, b, out, rows, width,
                                                capture_stream() if stream is None else stream),
           "residual add")


def launch_residual_read(query, values, out, rows, blocks, queries, width, *, stream=None, target_arch=None):
    """out[i][r] = softmax((query[i] . values[b][r]) / sqrt(width)) . values[b][r].

    `values` is block-major, so block `b` holds the `rows` normalized rows every block row
    reads, and the result is query-major so one query's rows stay contiguous.
    """
    _check(_build("residual_read", target_arch, 3, 4)(query, values, out, rows, blocks, queries, width,
                                                      capture_stream() if stream is None else stream),
           "residual read")


def host_row_rms(x, out, rows, width, eps, **kwargs):
    """Strict control: the host normalization the reference session evaluates."""
    x = np.asarray(x, dtype=np.float32).reshape(rows, width)
    return (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)).astype(np.float32)


def host_row_norm_mul(x, weight, out, rows, width, eps, **kwargs):
    x = np.asarray(x, dtype=np.float32).reshape(rows, width)
    weight = np.asarray(weight, dtype=np.float32).reshape(width)
    return ((x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)) * weight).astype(np.float32)


def host_row_add(a, b, out, rows, width, **kwargs):
    return (np.asarray(a, np.float32).reshape(rows, width) + np.asarray(b, np.float32).reshape(rows, width))


def host_residual_read(query, values, out, rows, blocks, queries, width, **kwargs):
    from hipengine.runtime.inkling2_mini_reference import _mix_residual_rows

    query = np.asarray(query, np.float32).reshape(queries, width)
    values = np.asarray(values, np.float32).reshape(blocks, rows, width).transpose(1, 0, 2)
    return _mix_residual_rows(query, values)


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    for _variant, _launch in (("row_rms", launch_row_rms), ("row_norm_mul", launch_row_norm_mul),
                              ("row_add", launch_row_add), ("residual_read", launch_residual_read)):
        register(KernelKey(_backend, "inkling2_residual", "inkling2_mixed", _variant), _launch)
    for _variant, _launch in (("host_row_rms", host_row_rms), ("host_row_norm_mul", host_row_norm_mul),
                              ("host_row_add", host_row_add), ("host_residual_read", host_residual_read)):
        register(KernelKey(_backend, "inkling2_residual", "inkling2_mixed", _variant), _launch)



@lru_cache(maxsize=2)
def _tick_symbol(target_arch):
    fn = _library(target_arch).hipengine_inkling2_resident_tick
    fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def launch_resident_tick(headers, caps, counter, nvisits, *, stream=None, target_arch=None):
    """Fill per-visit (live, position) header pairs from one device counter; advance the counter."""
    status = _tick_symbol(target_arch)(headers, caps, counter, nvisits,
                                       capture_stream() if stream is None else stream)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 resident tick failed")
