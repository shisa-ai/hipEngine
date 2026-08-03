"""Source-shaped IQ3_XXS/IQ4_XS selected-down integer-MMQ wrappers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq_source_mmq_prefill.hip")
_PARENT_SOURCE = Path(__file__).with_name("gguf_iq_gemv.hip")
_OUTPUT_NAME = "gguf_iq_source_mmq_prefill.so"
_VARIANT = "selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out"
_D4X2_VARIANT = (
    "selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out"
)
_D4X2_SYMBOL = (
    "hipengine_gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_"
    "prefill_compact_bf16_bf16_out"
)
_SYMBOL_TEMPLATE = (
    "hipengine_{quant}_selected_mmq_i128_j128_k256_q8_1_ds4_"
    "prefill_compact_bf16_bf16_out"
)
_MMQ_ROWS = 128
_QK_K = 256
_SOURCE_FLAGS = (
    "-mcumode",
    "-funsafe-math-optimizations",
    "-ffast-math",
    "-fno-finite-math-only",
)


@dataclass(frozen=True)
class IQSourceMMQ128Metadata:
    """Expert-local 128-row padding and source-MMQ tile ownership."""

    expert_start_mmq: np.ndarray
    tile_expert: np.ndarray
    mmq_total_rows: int


def build_iq_source_mmq128_metadata(
    counts: Sequence[int],
) -> IQSourceMMQ128Metadata:
    """Build host metadata without materializing padded activation rows."""

    if not counts:
        raise ValueError("counts must be non-empty")
    normalized = np.asarray(counts, dtype=np.int64)
    if np.any(normalized < 0):
        raise ValueError("counts must be non-negative")
    padded = ((normalized + (_MMQ_ROWS - 1)) // _MMQ_ROWS) * _MMQ_ROWS
    starts = np.zeros(len(normalized) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(padded, dtype=np.int64)
    tiles = np.repeat(
        np.arange(len(normalized), dtype=np.int64),
        padded // _MMQ_ROWS,
    )
    return IQSourceMMQ128Metadata(
        expert_start_mmq=starts,
        tile_expert=np.ascontiguousarray(tiles, dtype=np.int64),
        mmq_total_rows=int(starts[-1]),
    )


def _extra_flags() -> tuple[str, ...]:
    parent_tag = int(hashlib.sha256(_PARENT_SOURCE.read_bytes()).hexdigest()[:8], 16)
    return (*_SOURCE_FLAGS, f"-DHIPENGINE_IQ_GEMV_SOURCE_TAG={parent_tag}")


def plan_gguf_iq_source_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq_source_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq_source_mmq_prefill(
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
        family="gguf_iq_source_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _launch_iq_source_mmq(
    quant: str,
    xq_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    mmq_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0 or out_features % _MMQ_ROWS != 0:
        raise ValueError("out_features must be a positive multiple of 128")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if mmq_total_rows <= 0 or mmq_total_rows % _MMQ_ROWS != 0:
        raise ValueError("mmq_total_rows must be positive and a multiple of 128")
    library = library or build_gguf_iq_source_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TEMPLATE.format(quant=quant))
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
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_iq_source_mmq("gguf_iq3_xxs", *args, **kwargs)


def gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(
    *args, **kwargs
) -> None:
    _launch_iq_source_mmq("gguf_iq4_xs", *args, **kwargs)


def gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out(
    xq_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq_ptr: int,
    tile_expert_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    *,
    compact_rows: int,
    in_features: int,
    out_features: int,
    num_experts: int,
    mmq_total_rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch IQ3 MMQ with primary and residual D4 activation planes."""

    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0 or out_features % _MMQ_ROWS != 0:
        raise ValueError("out_features must be a positive multiple of 128")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if mmq_total_rows <= 0 or mmq_total_rows % _MMQ_ROWS != 0:
        raise ValueError("mmq_total_rows must be positive and a multiple of 128")
    library = library or build_gguf_iq_source_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _D4X2_SYMBOL)
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
        ctypes.c_void_p(xq_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(mmq_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gguf_iq_source_mmq_prefill_kernels(*, replace: bool = True) -> None:
    for quant, function in (
        (
            "gguf_iq3_xxs",
            gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
        (
            "gguf_iq4_xs",
            gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out,
        ),
    ):
        register(
            KernelKey("hip_gfx1100", "moe_linear", quant, _VARIANT),
            function,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_iq3_xxs", _D4X2_VARIANT),
        gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out,
        replace=replace,
    )


register_gguf_iq_source_mmq_prefill_kernels()


__all__ = [
    "IQSourceMMQ128Metadata",
    "build_gguf_iq_source_mmq_prefill",
    "build_iq_source_mmq128_metadata",
    "gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out",
    "gguf_iq3_xxs_selected_mmq_i128_j128_k256_q8_1_ds4x2_prefill_compact_bf16_bf16_out",
    "gguf_iq4_xs_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out",
    "plan_gguf_iq_source_mmq_prefill_build",
    "register_gguf_iq_source_mmq_prefill_kernels",
]
