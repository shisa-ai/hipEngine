"""Wrappers for diagnostic GGUF Q4_K x prequantized-Q8_1 selected prefill.

The kernel is a standalone microbench/prototype for the llama.cpp MMQ prefill
hypothesis.  It is not wired into model dispatch; callers provide Q8_1-style
prequantized activations plus raw Q4_K gate/up expert weights.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_q4_k_q8_1_selected_prefill.hip")
_OUTPUT_NAME = "gguf_q4_k_q8_1_selected_prefill.so"
_SYMBOL_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA32_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA64_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_PREVIEW_WMMA32_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA32_LDSPACK_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out"
_SYMBOL_DS4_WMMA32_LDS_BF16 = "hipengine_gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out"
_SYMBOL_WMMA_I8_PROBE = "hipengine_gguf_q4_k_q8_1_wmma_i8_probe_16x16"
_SYMBOL_DS4_PACK_BF16 = "hipengine_gguf_q8_1_mmq_ds4_pack_bf16"
_Q4_K_BLOCK = 256
_Q8_1_MMQ_BLOCK = 128


def plan_gguf_q4_k_q8_1_selected_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_q4_k_q8_1_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
    )


def build_gguf_q4_k_q8_1_selected_prefill(
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
        family="gguf_q4_k_q8_1_selected_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        extra_flags=("-mcumode",),
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gguf_q4_k_q8_1_wmma_i8_probe_16x16(
    a_rows_ptr: int,
    b_cols_ptr: int,
    out_ptr: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch a diagnostic RDNA3 int8/uint8 WMMA 16x16 probe.

    ``a_rows`` is row-major ``int8[16, 16]`` and ``b_cols`` is row-major
    ``uint8[16, 16]`` where each row represents one logical output column over
    K. The kernel writes ``int32[16, 16]`` equal to ``a_rows @ b_cols.T``.
    """

    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_WMMA_I8_PROBE)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(a_rows_ptr),
        ctypes.c_void_p(b_cols_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q8_1_mmq_ds4_pack_bf16(
    x_bf16_ptr: int,
    out_q8_ptr: int,
    rows: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pack BF16 activations to llama.cpp-style DS4 block_q8_1_mmq on GPU."""

    _check_positive(rows, "rows")
    _check_positive(hidden, "hidden")
    if hidden % _Q8_1_MMQ_BLOCK != 0:
        raise ValueError("hidden must be divisible by DS4 Q8_1 MMQ block size 128")
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_PACK_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_bf16_ptr),
        ctypes.c_void_p(out_q8_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out(
    x_qs_ptr: int,
    x_d_ptr: int,
    x_sum_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x prequantized-Q8_1 selected prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_qs_ptr),
        ctypes.c_void_p(x_d_ptr),
        ctypes.c_void_p(x_sum_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 integer-WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA_BF16)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 32-column WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA32_BF16)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 DS4 Q8_1 x raw Q4_K four-wave WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA64_BF16)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    q4_a_ptr: int,
    scale_a_ptr: int,
    min_a_ptr: int,
    q4_b_ptr: int,
    scale_b_ptr: int,
    min_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 DS4 Q8_1 x pre-unpacked Q4_K preview WMMA prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_PREVIEW_WMMA32_BF16)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(q4_a_ptr),
        ctypes.c_void_p(scale_a_ptr),
        ctypes.c_void_p(min_a_ptr),
        ctypes.c_void_p(q4_b_ptr),
        ctypes.c_void_p(scale_b_ptr),
        ctypes.c_void_p(min_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 32-column WMMA+packed-LDS prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA32_LDSPACK_BF16)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 Q8_1 32-column WMMA+LDS prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_WMMA32_LDS_BF16)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out(
    x_q8_ptr: int,
    expert_start_compact_ptr: int,
    expert_start_wmma_ptr: int,
    tile_expert_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_ptr: int,
    compact_rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    num_experts: int,
    wmma_total_rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the diagnostic BF16 Q4_K x DS4 block_q8_1_mmq selected prefill."""

    _check_common(
        compact_rows,
        in_features,
        out_features_a,
        out_features_b,
        num_experts,
        wmma_total_rows,
    )
    library = library or build_gguf_q4_k_q8_1_selected_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_DS4_BF16)
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
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_q8_ptr),
        ctypes.c_void_p(expert_start_compact_ptr),
        ctypes.c_void_p(expert_start_wmma_ptr),
        ctypes.c_void_p(tile_expert_ptr),
        ctypes.c_void_p(qweight_a_ptr),
        ctypes.c_void_p(qweight_b_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(compact_rows),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features_a),
        ctypes.c_int64(out_features_b),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(wmma_total_rows),
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
    wmma_total_rows: int,
) -> None:
    _check_positive(compact_rows, "compact_rows")
    _check_positive(in_features, "in_features")
    _check_positive(out_features_a, "out_features_a")
    _check_positive(out_features_b, "out_features_b")
    _check_positive(num_experts, "num_experts")
    _check_positive(wmma_total_rows, "wmma_total_rows")
    if in_features % _Q4_K_BLOCK != 0:
        raise ValueError("in_features must be divisible by GGUF Q4_K block size 256")
    if out_features_a % 16 != 0:
        raise ValueError("out_features_a must be a multiple of 16")
    if out_features_b % 16 != 0:
        raise ValueError("out_features_b must be a multiple of 16")
    if wmma_total_rows % 16 != 0:
        raise ValueError("wmma_total_rows must be a multiple of 16")


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def register_gguf_q4_k_q8_1_selected_prefill_kernels(*, replace: bool = True) -> None:
    """Register the standalone diagnostic Q8_1 selected-prefill prototype."""

    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_prefill_compact_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma64_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_preview_wmma32_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_ldspack_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_wmma32_lds_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k",
            variant="selected_dual_q8_1_ds4_prefill_compact_bf16_bf16_out",
        ),
        gguf_q4_k_selected_dual_q8_1_ds4_prefill_compact32_bf16_bf16_out,
        replace=replace,
    )


register_gguf_q4_k_q8_1_selected_prefill_kernels()
