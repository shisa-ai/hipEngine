"""Raw Q5_1 selected-MoE MMQ prefill (Q8_1 ds4 activations, DP4A) wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q5_1_mmq_selected_prefill.hip")
_OUTPUT_NAME = "gguf_q5_1_mmq_selected_prefill.so"
_SYMBOL = "hipengine_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out"
VARIANT = "q5_1_mmq_ds4_selected_prefill_bf16_bf16_out"

_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
) + (ctypes.c_int64,) * 5


def ds4_workspace_nbytes(compact_rows: int, in_features: int, planes: int = 3) -> int:
    """Device bytes for the multi-plane ds4 activation workspace."""

    if compact_rows <= 0 or in_features <= 0 or in_features % 128:
        raise ValueError("compact_rows must be positive and in_features % 128 == 0")
    if planes <= 0 or planes > 3:
        raise ValueError("planes must be in 1..3")
    return planes * compact_rows * (in_features // 128) * 144


def plan_gguf_q5_1_mmq_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q5_1_mmq_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q5_1_mmq_selected_prefill(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "baseline",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gguf_q5_1_mmq_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out(
    x_ds4_ptr: int,
    expert_start_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    compact_rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    planes: int = 3,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the raw Q5_1 DP4A MMQ consumer over compact expert-sorted rows."""

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 128:
        raise ValueError("in_features must be a positive multiple of 128")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if planes <= 0 or planes > 3:
        raise ValueError("planes must be in 1..3")
    library = library or build_gguf_q5_1_mmq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
    fn.argtypes = list(_ARGS) + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    error = fn(
        ctypes.c_void_p(x_ds4_ptr),
        ctypes.c_void_p(expert_start_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(planes),
        ctypes.c_void_p(stream),
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


_KERNEL_KEY = KernelKey("hip_gfx1100", "moe_linear", "gguf_q5_1", VARIANT)


def register_gguf_q5_1_mmq_selected_prefill_kernels(*, replace: bool = True) -> None:
    register(
        _KERNEL_KEY,
        gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out,
        replace=replace,
    )


register_gguf_q5_1_mmq_selected_prefill_kernels()


__all__ = [
    "VARIANT",
    "build_gguf_q5_1_mmq_selected_prefill",
    "ds4_workspace_nbytes",
    "gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out",
    "plan_gguf_q5_1_mmq_selected_prefill_build",
    "register_gguf_q5_1_mmq_selected_prefill_kernels",
]
