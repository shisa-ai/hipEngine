"""gfx1100 Inkling 2 target attention over explicit FP32 KV spans."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache.spans import KVLiveSpans


@lru_cache(maxsize=1)
def _build():
    lib = build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_target_attention",
                    profile="decode", output_name="inkling2_target_attention.so")
    fn = lib.hipengine_inkling2_target_attention
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 4
    fn.restype = ctypes.c_int
    return fn


def launch_target_attention(q, keys, values, bias, sink, out, spans: KVLiveSpans, nq, nk, dim, extent):
    t = (spans.base_offsets, spans.live_counts, spans.token_positions, spans.evict_mask, spans.row_positions)
    if spans.storage_dtype != DType.FP32 or spans.spans_mode != "uniform" or any(x is None for x in t):
        raise ValueError("Inkling 2 target attention requires explicit uniform FP32 KV spans")
    dtypes = (DType.INT32, DType.INT32, DType.INT32, DType.BOOL, DType.INT32)
    sizes = (spans.max_live_count, 1, spans.max_live_count, spans.max_live_count, 1)
    if any(x.dtype != dtype or x.numel < n or not x.ptr or x.device.kind != "hip"
           for x, dtype, n in zip(t, dtypes, sizes)):
        raise ValueError("invalid gfx1100 Inkling 2 KV span layout")
    status = _build()(q, keys, values, bias, sink, out, *(x.ptr for x in t), nq, nk, dim, extent)
    if status != HIP_SUCCESS:
        raise HipError(f"Inkling 2 target attention failed with HIP status {status}")


register(KernelKey("hip_gfx1100", "inkling2_target_attention", "inkling2_mixed", "default"), launch_target_attention)
