"""Raw-pointer wrappers for the compact DMS kernel family (C2-7 device port).

U1 of the streaming no-shadow DMS port: ``dms_extract_decision_bf16`` reads
the borrowed decision neuron (last channel of the first query head of each
GQA group) from pre-RoPE Q, publishes per-KV-head eviction bits, and zeroes
the channel in place. The CPU oracle is the registered ``cpu_reference``
``dms_extract_decision``; the device kernel is bit-exact against it.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dms_compact.hip")
_OUTPUT_NAME = "dms_compact.so"
_SYMBOL_EXTRACT_DECISION = "hipengine_dms_extract_decision_bf16"


def plan_dms_compact_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dms_compact",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dms_compact(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="dms_compact",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dms_extract_decision_bf16(
    q_ptr: int,
    evict_ptr: int,
    alpha_scale: float,
    alpha_offset: float,
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Extract per-KV-head DMS eviction decisions from pre-RoPE Q (BF16).

    ``q`` is ``[tokens, q_heads, head_dim]`` BF16; ``evict`` is
    ``[tokens, kv_heads]`` uint8. The borrowed channel (first query head of
    each GQA group, last channel) is zeroed in place. The threshold
    arithmetic mirrors the CPU reference's float64 scalar comparisons.
    """
    if int(tokens) <= 0:
        raise ValueError("tokens must be positive")
    if int(kv_heads) <= 0:
        raise ValueError("kv_heads must be positive")
    if int(q_heads) <= 0 or int(q_heads) % int(kv_heads) != 0:
        raise ValueError("q_heads must be positive and divisible by kv_heads")
    if int(head_dim) <= 0:
        raise ValueError("head_dim must be positive")
    if not math.isfinite(float(alpha_scale)) or float(alpha_scale) == 0.0:
        raise ValueError("alpha_scale must be finite and non-zero")
    if not math.isfinite(float(alpha_offset)):
        raise ValueError("alpha_offset must be finite")

    library = library or build_dms_compact(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_EXTRACT_DECISION)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(q_ptr),
        ctypes.c_void_p(evict_ptr),
        ctypes.c_double(float(alpha_scale)),
        ctypes.c_double(float(alpha_offset)),
        ctypes.c_int64(tokens),
        ctypes.c_int64(q_heads),
        ctypes.c_int64(kv_heads),
        ctypes.c_int64(head_dim),
        ctypes.c_void_p(stream),
    )
    if err != HIP_SUCCESS:
        raise RuntimeError(f"dms_extract_decision_bf16 failed with HIP error {err}")


def register_dms_compact_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dms_extract_decision", "bf16", "corrected_mask"),
        dms_extract_decision_bf16,
        replace=replace,
    )


register_dms_compact_kernels()
