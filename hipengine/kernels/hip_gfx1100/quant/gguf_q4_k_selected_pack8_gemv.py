"""Raw-pointer wrappers for GGUF Q4_K selected compact pack8 GEMV decode.

P9.B1 grouped/selected MoE gate+up GEMV decode kernel. The wrapper consumes
the compact-MoE scheduler ABI introduced in P8.4 (``x`` compact slab,
``expert_start_compact[E+1]``, raw rank-3 Q4_K expert weights
``[E, out_features, raw_bytes_per_row]``) and produces one row-major
concatenated output ``[compact_rows, out_features_a + out_features_b]``.

Compared to the P8.4 WMMA prefill kernel, this decode-shaped pack8 GEMV:

* drops ``expert_start_wmma`` and ``tile_expert`` (WMMA-only fields) and
  instead recovers the per-row expert via a linear scan over
  ``expert_start_compact`` inside the kernel,
* uses PARO-style ``__launch_bounds__(128, 4)`` 4-wave32 reduction (one block
  per ``(output pack, compact_row)`` pair),
* hoists Q4_K per-block ``d``, ``dmin``, scale, and min into shared memory so
  the inner k loop stays in registers.

No new compact-MoE ABI and no resident weight sidecar/repack are introduced.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_selected_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_q4_k_selected_pack8_gemv.so"
_SYMBOL_DUAL_BF16 = "hipengine_gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out"
_SYMBOL_DUAL_FP16 = "hipengine_gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out"
_Q4_K_BLOCK = 256


def plan_gguf_q4_k_selected_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_selected_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_selected_pack8_gemv(
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
        family="gguf_q4_k_selected_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact raw-Q4_K dual gate+up pack8 GEMV decode."""

    _launch_dual(
        _SYMBOL_DUAL_BF16,
        x_ptr,
        expert_start_compact_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact raw-Q4_K dual gate+up pack8 GEMV decode."""

    _launch_dual(
        _SYMBOL_DUAL_FP16,
        x_ptr,
        expert_start_compact_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_dual(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
    )
    library = library or build_gguf_q4_k_selected_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    _check_positive(num_experts, "num_experts")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if out_features_a % 8 != 0:
        raise ValueError("out_features_a must be a multiple of 8 (pack8 lane)")
    if out_features_b % 8 != 0:
        raise ValueError("out_features_b must be a multiple of 8 (pack8 lane)")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_q4_k_selected_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.B1 compact selected raw-Q4_K pack8 GEMV decode kernels."""

    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_pack8_gemv_decode_compact_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_pack8_gemv_decode_compact_fp16_fp16_out",
        ),
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out,
        replace=replace,
    )
    # Shorthand aliases matching the docs/GGUF.md P9 pipeline language. The
    # runtime can choose either spelling without changing the wrapper ABI.
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_pack8_gemv_decode_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "moe_linear",
            "gguf_q4_k",
            "selected_dual_pack8_gemv_decode_fp16_fp16_out",
        ),
        gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out,
        replace=replace,
    )


register_gguf_q4_k_selected_pack8_gemv_kernels()


__all__ = [
    "build_gguf_q4_k_selected_pack8_gemv",
    "gguf_q4_k_selected_dual_pack8_gemv_decode_compact_bf16_bf16_out",
    "gguf_q4_k_selected_dual_pack8_gemv_decode_compact_fp16_fp16_out",
    "plan_gguf_q4_k_selected_pack8_gemv_build",
    "register_gguf_q4_k_selected_pack8_gemv_kernels",
]
