"""Raw-pointer wrapper for the dense Q8_0 q8_1+dp4a GEMV decode kernel.

Task #13: the dense Q8_0 attention projections are the top GPU cost of the GGUF
MTP verifier. This kernel consumes raw GGUF Q8_0 weights (``[out_features,
blocks * 34]``) and q8_1-quantized activations (``gguf_q8_1_block`` produced by
:func:`gguf_q4_k_quantize_bf16_q8_1`) and does the int8 dot via ``v_dot4``.
No weight repack is required -- raw Q8_0 stores 32 int8 contiguously.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("gguf_q8_0_dp4a_gemv.hip")
_OUTPUT_NAME = "gguf_q8_0_dp4a_gemv.so"
_Q8_0_DP4A_BF16 = "hipengine_gguf_q8_0_dp4a_gemv_bf16_bf16_out"
_Q8_0_DP4A_ROWTILE4_BF16 = "hipengine_gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out"
_Q8_0_DP4A_DUAL_ROWTILE4_BF16 = "hipengine_gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out"
_Q8_0_DP4A_TRIPLE_ROWTILE4_BF16 = "hipengine_gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out"
_Q8_0_BLOCK = 32

_CACHED_FNS: dict[tuple[int, str], "ctypes._CFuncPtr"] = {}
_ARGTYPES = [
    ctypes.c_void_p,  # xq (q8_1 blocks)
    ctypes.c_void_p,  # qweight (raw Q8_0)
    ctypes.c_void_p,  # out
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # in_features
    ctypes.c_int64,   # out_features
    ctypes.c_int64,   # threads (ignored)
    ctypes.c_void_p,  # stream
]
_ROWTILE_ARGTYPES = _ARGTYPES
_DUAL_ROWTILE_ARGTYPES = [
    ctypes.c_void_p,  # xq (q8_1 blocks)
    ctypes.c_void_p,  # qweight_a (raw Q8_0)
    ctypes.c_void_p,  # qweight_b (raw Q8_0)
    ctypes.c_void_p,  # out_a
    ctypes.c_void_p,  # out_b
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # in_features
    ctypes.c_int64,   # out_features_a
    ctypes.c_int64,   # out_features_b
    ctypes.c_int64,   # threads (ignored)
    ctypes.c_void_p,  # stream
]
_TRIPLE_ROWTILE_ARGTYPES = [
    ctypes.c_void_p,  # xq (q8_1 blocks)
    ctypes.c_void_p,  # qweight_a (raw Q8_0)
    ctypes.c_void_p,  # qweight_b (raw Q8_0)
    ctypes.c_void_p,  # qweight_c (raw Q8_0)
    ctypes.c_void_p,  # out_a
    ctypes.c_void_p,  # out_b
    ctypes.c_void_p,  # out_c
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # in_features
    ctypes.c_int64,   # out_features_a
    ctypes.c_int64,   # out_features_b
    ctypes.c_int64,   # out_features_c
    ctypes.c_int64,   # threads (ignored)
    ctypes.c_void_p,  # stream
]


def plan_gguf_q8_0_dp4a_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q8_0_dp4a_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q8_0_dp4a_gemv(
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
        family="gguf_q8_0_dp4a_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _cached_fn(library: ctypes.CDLL, symbol: str, argtypes: list[object]) -> "ctypes._CFuncPtr":
    key = (id(library), symbol)
    fn = _CACHED_FNS.get(key)
    if fn is None:
        fn = getattr(library, symbol)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        _CACHED_FNS[key] = fn
    return fn


def gguf_q8_0_dp4a_gemv_bf16_bf16_out(
    xq_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Dense Q8_0 GEMV via q8_1 activations + dp4a; bf16 output.

    ``xq_ptr`` points at ``gguf_q8_1_block``-quantized activations
    (``rows * in_features/32`` blocks); ``qweight_ptr`` at raw Q8_0 bytes.
    """

    if in_features % _Q8_0_BLOCK != 0:
        raise ValueError("in_features must be a multiple of 32 for Q8_0 dp4a")
    library = library or build_gguf_q8_0_dp4a_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, _Q8_0_DP4A_BF16, _ARGTYPES)
    err = fn(xq_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, 0, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out(
    xq_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split-output raw Q8_0 pair GEMV via q8_1 activations + dp4a.

    One wave computes one output column across up to four destination rows,
    reusing the raw Q8_0 row bytes across those rows. This mirrors llama.cpp
    MMVQ's small-B row economy more closely than launching two single GEMVs.
    """

    if in_features % _Q8_0_BLOCK != 0:
        raise ValueError("in_features must be a multiple of 32 for Q8_0 dp4a")
    library = library or build_gguf_q8_0_dp4a_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, _Q8_0_DP4A_DUAL_ROWTILE4_BF16, _DUAL_ROWTILE_ARGTYPES)
    err = fn(
        xq_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        0,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out(
    xq_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Row-tiled raw Q8_0 GEMV via q8_1 activations + dp4a.

    One wave computes one output column across up to four destination rows,
    matching the B2/B3 verifier economy used by the split wrappers.
    """

    if in_features % _Q8_0_BLOCK != 0:
        raise ValueError("in_features must be a multiple of 32 for Q8_0 dp4a")
    library = library or build_gguf_q8_0_dp4a_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, _Q8_0_DP4A_ROWTILE4_BF16, _ROWTILE_ARGTYPES)
    err = fn(xq_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, 0, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out(
    xq_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    qweight_c_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    out_features_c: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Split-output raw Q8_0 triple GEMV via q8_1 activations + dp4a."""

    if in_features % _Q8_0_BLOCK != 0:
        raise ValueError("in_features must be a multiple of 32 for Q8_0 dp4a")
    library = library or build_gguf_q8_0_dp4a_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, _Q8_0_DP4A_TRIPLE_ROWTILE4_BF16, _TRIPLE_ROWTILE_ARGTYPES)
    err = fn(
        xq_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        qweight_c_ptr,
        out_a_ptr,
        out_b_ptr,
        out_c_ptr,
        rows,
        in_features,
        out_features_a,
        out_features_b,
        out_features_c,
        0,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


__all__ = [
    "build_gguf_q8_0_dp4a_gemv",
    "gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out",
    "gguf_q8_0_dp4a_gemv_bf16_bf16_out",
    "gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out",
    "gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out",
    "plan_gguf_q8_0_dp4a_gemv_build",
]
