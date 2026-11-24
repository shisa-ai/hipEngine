"""gfx1100 noncausal attention on a dense, preselected key block."""

import ctypes
from functools import lru_cache
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import HIP_SUCCESS, HipError
from hipengine.kernels.registry import KernelKey, register


@lru_cache(maxsize=1)
def _build():
    lib = build_hip(sources=[Path(__file__).with_suffix(".hip")], family="inkling2_block_attention",
                    profile="decode", output_name="inkling2_block_attention.so")
    fn = lib.hipengine_inkling2_block_attention
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
    fn.restype = ctypes.c_int
    return fn


def launch_block_attention(q, k, v, out, queries, keys, nq, nk, dim):
    status = _build()(q, k, v, out, queries, keys, nq, nk, dim)
    if status != HIP_SUCCESS:
        raise HipError(status, "Inkling 2 noncausal block attention failed")


register(KernelKey("hip_gfx1100", "inkling2_block_attention", "inkling2_mixed", "default"), launch_block_attention)
