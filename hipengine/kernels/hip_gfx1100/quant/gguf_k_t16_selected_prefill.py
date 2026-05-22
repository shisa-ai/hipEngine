"""Wrappers for compact selected-MoE WMMA prefill on GGUF Q4_K / Q5_K / Q6_K T16 tiles.

P10.B2 / P10.B3 / Q4_K_S follow-up: ports selected single-output WMMA prefill
kernels (``gguf_k_selected_prefill.hip``) to consume the T16 replacement
layout used by the decode-repack path. The exported callables share the
``selected_wmma_prefill_compact_bf16_bf16_out`` ABI used by the raw
versions so dispatch can swap quant keys without runtime / backend
branches.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_t16_selected_prefill.hip")
_OUTPUT_NAME = "gguf_k_t16_selected_prefill.so"
_ENV_LAUNCH_BOUNDS = "HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS"
_QK_K = 256

_SYMBOLS = {
    ("gguf_q4_k_t16", "bf16"): "hipengine_gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q4_k_t16", "fp16"): "hipengine_gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    ("gguf_q5_k_t16", "bf16"): "hipengine_gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q5_k_t16", "fp16"): "hipengine_gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    ("gguf_q6_k_t16", "bf16"): "hipengine_gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    ("gguf_q6_k_t16", "fp16"): "hipengine_gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
}


def _extra_flags() -> tuple[str, ...]:
    value = os.environ.get(_ENV_LAUNCH_BOUNDS)
    if not value:
        return ("-mcumode",)
    min_blocks = int(value)
    if min_blocks not in {1, 2, 4, 8}:
        raise ValueError(f"{_ENV_LAUNCH_BOUNDS} must be one of 1, 2, 4, 8")
    return ("-mcumode", f"-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS={min_blocks}")


def plan_gguf_k_t16_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_t16_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_t16_selected_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_k_t16_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make_wrapper(quant: str, dtype: str):
    symbol = _SYMBOLS[(quant, dtype)]

    def wrapper(
        x_ptr: int,
        expert_start_compact_ptr: int,
        expert_start_wmma_ptr: int,
        tile_expert_ptr: int,
        tiles_ptr: int,
        out_ptr: int,
        compact_rows: int,
        in_features: int,
        out_features: int,
        num_experts: int,
        wmma_total_rows: int,
        *,
        stream: int = 0,
        library: ctypes.CDLL | None = None,
        runtime: HipRuntime | None = None,
        # Accepted-but-ignored kwargs so dispatch can call this wrapper with
        # the same (tile_m, tile_n) keyword arguments accepted by the raw
        # ``gguf_k_selected_prefill`` wrappers. The T16 kernel ships a single
        # 16x16 tile shape; the per-quant tile sweep belongs to P10.C2.
        tile_m: int | None = None,
        tile_n: int | None = None,
    ) -> None:
        del tile_m, tile_n  # tile sweep is P10.C2, not P10.B2/B3
        _launch_k_t16(
            symbol,
            x_ptr,
            expert_start_compact_ptr,
            expert_start_wmma_ptr,
            tile_expert_ptr,
            tiles_ptr,
            out_ptr,
            compact_rows,
            in_features,
            out_features,
            num_experts,
            wmma_total_rows,
            stream=stream,
            library=library,
            runtime=runtime,
        )

    wrapper.__name__ = f"{quant}_selected_wmma_prefill_compact_{dtype}_{dtype}_out"
    wrapper.__qualname__ = wrapper.__name__
    wrapper.__doc__ = (
        f"Launch {quant} T16 selected compact single-output WMMA prefill "
        f"({dtype}->{dtype})."
    )
    return wrapper


# Public wrapper functions.
gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q4_k_t16", "bf16")
gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q4_k_t16", "fp16")
gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q5_k_t16", "bf16")
gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q5_k_t16", "fp16")
gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out = _make_wrapper("gguf_q6_k_t16", "bf16")
gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out = _make_wrapper("gguf_q6_k_t16", "fp16")


def _launch_k_t16(
    symbol: str,
    x_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_common(compact_rows, in_features, out_features, num_experts, wmma_total_rows)
    library = library or build_gguf_k_t16_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(tiles_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _check_common(
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features, "out_features")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if in_features % _QK_K != 0:
        raise ValueError(f"in_features must be divisible by GGUF K-family block size {_QK_K}")
    if out_features % 16 != 0:
        raise ValueError("out_features must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_k_t16_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register Q4T16/Q5T16/Q6T16 selected WMMA prefill kernels.

    Each kernel is registered under its native ``gguf_q*_k_t16_v1`` quant key
    using the shared ``selected_wmma_prefill_compact_*`` alias spelling so
    ``_COMPACT_MOE_DOWN_KEYS`` in the runner can route on quant key alone.
    """

    for quant_key, fn_bf16, fn_fp16 in (
        (
            "gguf_q4_k_t16_v1",
            gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
        (
            "gguf_q5_k_t16_v1",
            gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
        (
            "gguf_q6_k_t16_v1",
            gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out,
            gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out,
        ),
    ):
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_wmma_prefill_compact_bf16_bf16_out",
            ),
            fn_bf16,
            replace=replace,
        )
        register(
            KernelKey(
                "hip_gfx1100",
                "moe_linear",
                quant_key,
                "selected_wmma_prefill_compact_fp16_fp16_out",
            ),
            fn_fp16,
            replace=replace,
        )


register_gguf_k_t16_selected_prefill_kernels()


__all__ = [
    "build_gguf_k_t16_selected_prefill",
    "gguf_q4_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q4_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q5_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q5_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "gguf_q6_k_t16_selected_wmma_prefill_compact_bf16_bf16_out",
    "gguf_q6_k_t16_selected_wmma_prefill_compact_fp16_fp16_out",
    "plan_gguf_k_t16_selected_prefill_build",
    "register_gguf_k_t16_selected_prefill_kernels",
]
