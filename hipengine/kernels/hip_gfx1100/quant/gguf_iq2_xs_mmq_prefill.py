"""Raw IQ2_XS x D4-Q8_1 integer-MMQ prefill candidate."""

from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_iq2_xs_mmq_prefill.hip")
_PARENT_SOURCE = Path(__file__).with_name("gguf_iq_gemv.hip")
_OUTPUT_NAME = "gguf_iq2_xs_mmq_prefill.so"
_SYMBOL = (
    "hipengine_gguf_iq2_xs_selected_dual_mmq32_prefill_"
    "q8_1_d4_bf16_bf16_out"
)
_VARIANT = "selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out"
_MMQ_ROWS = 32
_QK_K = 256


@dataclass(frozen=True)
class IQ2XSMMQ32Metadata:
    """Expert-local 32-row padding and tile ownership for selected MMQ."""

    expert_start_mmq: np.ndarray
    tile_expert: np.ndarray
    mmq_total_rows: int


def build_iq2_xs_mmq32_metadata(counts: Sequence[int]) -> IQ2XSMMQ32Metadata:
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
    return IQ2XSMMQ32Metadata(
        expert_start_mmq=starts,
        tile_expert=np.ascontiguousarray(tiles, dtype=np.int64),
        mmq_total_rows=int(starts[-1]),
    )


def _extra_flags() -> tuple[str, ...]:
    parent_tag = int(hashlib.sha256(_PARENT_SOURCE.read_bytes()).hexdigest()[:8], 16)
    return ("-mcumode", f"-DHIPENGINE_IQ_GEMV_SOURCE_TAG={parent_tag}")


def plan_gguf_iq2_xs_mmq_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_iq2_xs_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_iq2_xs_mmq_prefill(
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
        family="gguf_iq2_xs_mmq_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=_extra_flags(),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out(
    x_d4_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_mmq_ptr: int,
    tile_expert_ptr: int,
    gate_weight_ptr: int,
    up_weight_ptr: int,
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
    """Launch raw-IQ2 selected dual integer MMQ over 32-row expert tiles."""

    if compact_rows <= 0:
        raise ValueError("compact_rows must be positive")
    if in_features <= 0 or in_features % _QK_K != 0:
        raise ValueError("in_features must be positive and divisible by 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if mmq_total_rows <= 0 or mmq_total_rows % _MMQ_ROWS != 0:
        raise ValueError("mmq_total_rows must be positive and a multiple of 32")

    library = library or build_gguf_iq2_xs_mmq_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
    fn.argtypes = [
        ctypes.c_void_p,
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
        ctypes.c_void_p(x_d4_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_mmq_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(gate_weight_ptr),
        ctypes.c_void_p(up_weight_ptr),
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


def register_gguf_iq2_xs_mmq_prefill_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "moe_linear", "gguf_iq2_xs", _VARIANT),
        gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out,
        replace=replace,
    )


register_gguf_iq2_xs_mmq_prefill_kernels()


__all__ = [
    "IQ2XSMMQ32Metadata",
    "build_gguf_iq2_xs_mmq_prefill",
    "build_iq2_xs_mmq32_metadata",
    "gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out",
    "plan_gguf_iq2_xs_mmq_prefill_build",
    "register_gguf_iq2_xs_mmq_prefill_kernels",
]
