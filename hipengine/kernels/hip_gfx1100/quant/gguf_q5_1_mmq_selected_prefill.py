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
_SYMBOL_F32 = "hipengine_q5_1_mmq_ds4_f32_selected_prefill_bf16_bf16_out"
VARIANT = "q5_1_mmq_ds4_selected_prefill_bf16_bf16_out"

# Activation lanes per DS4 block. Must match Q8_1_MMQ_BLOCK in the .hip.
Q8_1_MMQ_BLOCK = 128

_ARGS = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
) + (ctypes.c_int64,) * 5


def _ds4_blocks(in_features: int) -> int:
    """128-wide DS4 activation blocks covering ``in_features``, tail included.

    The DS4 Q8_1 MMQ block is 128 elements, and Gemma 4 26B-A4B's expert down
    projection is 704 wide -- five full blocks plus a 64-element tail. Rounding
    down here would silently drop that tail, so every size, loop bound and guard
    on this route rounds up and the kernel stops at the last block that holds
    real data.
    """

    return (in_features + 127) // 128


def ds4_workspace_nbytes(compact_rows: int, in_features: int, planes: int = 3) -> int:
    """Device bytes for the multi-plane ds4 activation workspace."""

    if compact_rows <= 0 or in_features <= 0:
        raise ValueError("compact_rows and in_features must be positive")
    if planes <= 0 or planes > 3:
        raise ValueError("planes must be in 1..3")
    # Ceil, not floor: the down projection's K=704 is 5.5 DS4 blocks, and the
    # trailing lanes of the partial block are zero-filled by the pack so the
    # padding contributes exactly 0 to dot4, to the block sum, and to the m
    # offset. Flooring here would allocate 640 of 704 inputs' worth of blocks.
    blocks = -(-in_features // Q8_1_MMQ_BLOCK)
    return planes * compact_rows * blocks * 144


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
    f32_scales: bool = False,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the raw Q5_1 DP4A MMQ consumer over compact expert-sorted rows.

    ``f32_scales`` selects the activation block layout. The fp16 one is for a
    hidden-state input; the fp32 one is for a post-SiLU input such as a down
    projection's, whose magnitude can exceed what an fp16 scale or block sum
    holds.
    """

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32:
        # Q5_1 weights are stored as 32-value blocks, so the weight row needs
        # an exact division by 32. A multiple of 128 is NOT required: the DS4
        # pack zero-fills the partial final block (the down projection is
        # K=704 = 22 Q5_1 blocks = 5.5 DS4 blocks).
        raise ValueError("in_features must be a positive multiple of 32")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if planes <= 0 or planes > 3:
        raise ValueError("planes must be in 1..3")
    library = library or build_gguf_q5_1_mmq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_F32 if f32_scales else _SYMBOL)
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

_WMMA_SYMBOL = "hipengine_q5_1_mmq_ds4_wmma_prefill_bf16_bf16_out"


def gguf_q5_1_mmq_ds4_wmma_prefill_bf16_bf16_out(
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
    """Run the P10 int8 matrix-core twin of the DP4A leaf above.

    Same arguments, same DS4 activation layout and same arithmetic contract,
    so an A/B against ``gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out``
    needs only this symbol swapped.
    """

    if compact_rows <= 0 or num_experts <= 0:
        raise ValueError("compact_rows and num_experts must be positive")
    if in_features <= 0 or in_features % 32:
        raise ValueError("in_features must be a positive multiple of 32")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if planes <= 0 or planes > 3:
        raise ValueError("planes must be in 1..3")
    if num_experts > 65535:
        raise ValueError("num_experts must fit the launcher's grid.y")
    library = library or build_gguf_q5_1_mmq_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _WMMA_SYMBOL)
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
    "gguf_q5_1_mmq_ds4_wmma_prefill_bf16_bf16_out",
    "plan_gguf_q5_1_mmq_selected_prefill_build",
    "register_gguf_q5_1_mmq_selected_prefill_kernels",
]
