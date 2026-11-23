"""gfx1100 Inkling 2 target attention over explicit FP32 KV spans."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache.spans import KVLiveSpans


@lru_cache(maxsize=2)
def _library(target_arch):
    return build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_target_attention",
                     profile="decode", target_arch=target_arch, output_name="inkling2_target_attention.so")


@lru_cache(maxsize=4)
def _build(symbol, target_arch):
    fn = getattr(_library(target_arch), symbol)
    fn.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int] * 4
    fn.restype = ctypes.c_int
    return fn


@lru_cache(maxsize=2)
def _build_cache_write(target_arch):
    fn = _library(target_arch).hipengine_inkling2_target_cache_write
    fn.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 2
    fn.restype = ctypes.c_int
    return fn


def _span_pointers(spans: KVLiveSpans):
    t = (spans.base_offsets, spans.live_counts, spans.token_positions, spans.evict_mask, spans.row_positions)
    if spans.storage_dtype != DType.FP32 or spans.spans_mode != "uniform" or any(x is None for x in t):
        raise ValueError("Inkling 2 target attention requires explicit uniform FP32 KV spans")
    dtypes = (DType.INT32, DType.INT32, DType.INT32, DType.BOOL, DType.INT32)
    sizes = (spans.max_live_count, 1, spans.max_live_count, spans.max_live_count, 1)
    if any(x.dtype != dtype or x.numel < n or not x.ptr or x.device.kind != "hip"
           for x, dtype, n in zip(t, dtypes, sizes)):
        raise ValueError("invalid gfx1100 Inkling 2 KV span layout")
    return tuple(x.ptr for x in t)


def _launch_attention(symbol, q, keys, values, bias, sink, out, spans: KVLiveSpans, nq, nk, dim, extent,
                      *, gain=0, target_arch=None):
    status = _build(symbol, target_arch)(q, keys, values, bias, sink, out, *_span_pointers(spans), gain, nq, nk, dim, extent)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 target attention failed with HIP status {status}")


def launch_target_attention(*args, gain=0, target_arch=None):
    _launch_attention("hipengine_inkling2_target_attention", *args, gain=gain, target_arch=target_arch)


def launch_tiled_attention(*args, gain=0, target_arch=None):
    _launch_attention("hipengine_inkling2_target_attention_tiled", *args, gain=gain, target_arch=target_arch)


@lru_cache(maxsize=2)
def _split_library(target_arch):
    path = Path(__file__).with_name("inkling2_split_attention.hip")
    return build_hip(sources=[path], family="inkling2_split_attention", profile="decode",
                     target_arch=target_arch, output_name="inkling2_split_attention.so")


@lru_cache(maxsize=2)
def _build_split(target_arch):
    fn = _split_library(target_arch).hipengine_inkling2_split_attention
    fn.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 5
    fn.restype = ctypes.c_int
    return fn


def launch_split_attention(q, keys, values, bias, sink, out, spans, nq, nk, dim, extent, scores, live, *, gain=0,
                           target_arch=None):
    if not scores or not 0 < live <= spans.max_live_count:
        raise ValueError("invalid Inkling 2 split attention score scratch or live span count")
    status = _build_split(target_arch)(q, keys, values, bias, sink, out, *_span_pointers(spans),
                            scores, gain, nq, nk, dim, extent, live)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 split attention launch failed")


def launch_split_chunked_attention(q, keys, values, bias, sink, out, spans, nq, nk, dim, extent, scores, live,
                                   *, partial, gain=0, target_arch=None):
    if not scores or not partial or not 0 < live <= spans.max_live_count:
        raise ValueError("invalid Inkling 2 chunked attention scratch or live span count")
    status = _build_split_chunked(target_arch)(q, keys, values, bias, sink, out, *_span_pointers(spans),
                                                scores, gain, partial, nq, nk, dim, extent, live)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 chunked attention launch failed")


@lru_cache(maxsize=2)
def _build_split_chunked(target_arch):
    fn = _split_library(target_arch).hipengine_inkling2_split_chunked_attention
    fn.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 5
    fn.restype = ctypes.c_int
    return fn


def launch_cache_write(key, value, keys, values, spans: KVLiveSpans, width, *, target_arch=None):
    status = _build_cache_write(target_arch)(key, value, keys, values, *_span_pointers(spans), spans.max_live_count, width)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 target KV write failed with HIP status {status}")


@lru_cache(maxsize=4)
def _build_block(target_arch, *, sigmoid=False):
    lib = _library(target_arch)
    fn = (lib.hipengine_inkling2_target_attention_block_sigmoid if sigmoid
          else lib.hipengine_inkling2_target_attention_block)
    fn.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 6
    fn.restype = ctypes.c_int
    return fn


def _launch_attention_block(fn, q, k, v, bias, sink, out, spans, nq, nk, dim, extent, block_k, block_v, rows, gain):
    status = fn(q, k, v, bias, sink, out, *_span_pointers(spans), block_k, block_v, gain,
                nq, nk, dim, extent, rows, spans.max_live_count)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 block attention launch failed")


def launch_attention_block(*args, gain=0, target_arch=None):
    _launch_attention_block(_build_block(target_arch), *args, gain)


def launch_attention_block_sigmoid(*args, gain=0, target_arch=None):
    _launch_attention_block(_build_block(target_arch, sigmoid=True), *args, gain)


for _backend in ("hip_gfx1100", "hip_gfx1151"):
    register(KernelKey(_backend, "inkling2_target_attention", "inkling2_mixed", "block"), launch_attention_block)
    register(KernelKey(_backend, "inkling2_target_attention", "inkling2_mixed", "block_sigmoid"),
             launch_attention_block_sigmoid)
    register(KernelKey(_backend, "inkling2_target_kv_write", "inkling2_mixed", "default"), launch_cache_write)
    register(KernelKey(_backend, "inkling2_target_attention", "inkling2_mixed", "default"), launch_target_attention)
    register(KernelKey(_backend, "inkling2_target_attention", "inkling2_mixed", "tiled"), launch_tiled_attention)
    register(KernelKey(_backend, "inkling2_target_attention", "inkling2_mixed", "split"), launch_split_attention)
    register(KernelKey(_backend, "inkling2_target_attention", "inkling2_mixed", "split_chunked"),
             launch_split_chunked_attention)
