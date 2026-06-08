"""Raw-pointer wrappers for GGUF Q5_K/Q6_K selected compact pack8 GEMV decode.

P9.B2 grouped/selected MoE down-projection GEMV decode kernels. The wrappers
consume the compact-MoE scheduler ABI introduced in P8.5 (``x`` compact slab,
``expert_start_compact[E+1]``, raw rank-3 expert weights
``[E, out_features, raw_bytes_per_row]``) and produce row-major output
``[compact_rows, out_features]``.

Compared to the P8.5 WMMA prefill kernels, these decode-shaped pack8 GEMVs:

* drop ``expert_start_wmma`` and ``tile_expert`` (WMMA-only fields) and
  instead recover the per-row expert via a linear scan over
  ``expert_start_compact`` inside the kernel,
* use PARO-style ``__launch_bounds__(128, 4)`` 4-wave32 reduction (one block
  per ``(output pack, compact_row)`` pair),
* hoist Q5_K/Q6_K per-block scale/min into shared memory once per 256-element
  block so the inner k loop stays in registers.

No new compact-MoE ABI and no resident weight sidecar/repack are introduced.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_selected_pack8_gemv.hip")
_OUTPUT_NAME = "gguf_k_selected_pack8_gemv.so"
_Q5_K_BF16 = "hipengine_gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out"
_Q5_K_FP16 = "hipengine_gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out"
_Q6_K_BF16 = "hipengine_gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out"
_Q6_K_FP16 = "hipengine_gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out"
_QK_K = 256


def plan_gguf_k_selected_pack8_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_selected_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_selected_pack8_gemv(
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
        family="gguf_k_selected_pack8_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact raw-Q5_K pack8 GEMV decode."""

    _launch_single(
        _Q5_K_BF16,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact raw-Q5_K pack8 GEMV decode."""

    _launch_single(
        _Q5_K_FP16,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16 selected compact raw-Q6_K pack8 GEMV decode."""

    _launch_single(
        _Q6_K_BF16,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out(
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16 selected compact raw-Q6_K pack8 GEMV decode."""

    _launch_single(
        _Q6_K_FP16,
        x_ptr,
        expert_start_compact_ptr,
        qweight_ptr,
        out_ptr,
        compact_rows,
        in_features,
        out_features,
        num_experts,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _launch_single(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, out_features, num_experts)
    library = library or build_gguf_k_selected_pack8_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    if in_features % _QK_K != 0:
        raise ValueError("in_features must be divisible by GGUF Q5_K/Q6_K block size 256")
    if out_features % 8 != 0:
        raise ValueError("out_features must be a multiple of 8 (pack8 lane)")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_k_selected_pack8_gemv_kernels(*, replace: bool = True) -> None:
    """Register P9.B2 compact selected raw-Q5_K/Q6_K pack8 GEMV decode kernels."""

    for quant_key, fn_bf16, fn_fp16 in (
        (
            "gguf_q5_k",
            gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out,
            gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out,
        ),
        (
            "gguf_q6_k",
            gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out,
            gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_pack8_gemv_decode_compact_bf16_bf16_out",
            ),
            fn_bf16,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_pack8_gemv_decode_compact_fp16_fp16_out",
            ),
            fn_fp16,
            replace=replace,
        )
        # Shorthand aliases matching the docs/GGUF.md P9 pipeline language.
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_pack8_gemv_decode_bf16_bf16_out",
            ),
            fn_bf16,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_pack8_gemv_decode_fp16_fp16_out",
            ),
            fn_fp16,
            replace=replace,
        )


register_gguf_k_selected_pack8_gemv_kernels()


__all__ = [
    "build_gguf_k_selected_pack8_gemv",
    "gguf_q5_k_selected_pack8_gemv_decode_compact_bf16_bf16_out",
    "gguf_q5_k_selected_pack8_gemv_decode_compact_fp16_fp16_out",
    "gguf_q6_k_selected_pack8_gemv_decode_compact_bf16_bf16_out",
    "gguf_q6_k_selected_pack8_gemv_decode_compact_fp16_fp16_out",
    "plan_gguf_k_selected_pack8_gemv_build",
    "register_gguf_k_selected_pack8_gemv_kernels",
]
