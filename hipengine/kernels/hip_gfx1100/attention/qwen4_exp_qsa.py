"""Raw-pointer wrappers for strict Qwen4Exp QSA control primitives."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_qsa.hip")
_OUTPUT_NAME = "qwen4_exp_qsa.so"
_ARGS_POOL = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
)
_ARGS_SCORE = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_SELECT = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)


def plan_qwen4_exp_qsa_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_qsa",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_qsa(
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
        family="qwen4_exp_qsa",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_qsa_pool_norm_rope_f32(
    raw_keys_ptr: int,
    member_indices_ptr: int,
    block_starts_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    blocks: int,
    ratio: int,
    index_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float = 1e-6,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Pool complete raw index keys, normalize, and partial-RoPE at block starts."""

    if blocks <= 0 or ratio <= 0 or index_dim <= 0:
        raise ValueError("blocks, ratio, and index_dim must be positive")
    if rotary_dim <= 0 or rotary_dim > index_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= index_dim")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_pool_norm_rope_f32",
        _ARGS_POOL,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            raw_keys_ptr,
            member_indices_ptr,
            block_starts_ptr,
            weight_ptr,
            output_ptr,
            blocks,
            ratio,
            index_dim,
            rotary_dim,
            float(theta),
            float(eps),
            stream,
        ),
    )


def qwen4_exp_qsa_score_f32(
    queries_ptr: int,
    pooled_keys_ptr: int,
    scores_ptr: int,
    query_count: int,
    blocks: int,
    heads: int,
    index_dim: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Compute ReLU-per-index-head QSA block scores."""

    if query_count <= 0 or blocks <= 0 or heads <= 0 or index_dim <= 0:
        raise ValueError("query_count, blocks, heads, and index_dim must be positive")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_score_f32",
        _ARGS_SCORE,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            queries_ptr,
            pooled_keys_ptr,
            scores_ptr,
            query_count,
            blocks,
            heads,
            index_dim,
            stream,
        ),
    )


def qwen4_exp_qsa_select_blocks_f32_i64(
    scores_ptr: int,
    block_starts_ptr: int,
    query_positions_ptr: int,
    selected_starts_ptr: int,
    selected_counts_ptr: int,
    query_count: int,
    blocks: int,
    ratio: int,
    budget: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Strict deterministic complete-block selection with lower-start tie break."""

    if query_count <= 0 or blocks <= 0 or ratio <= 0 or budget <= 0:
        raise ValueError("query_count, blocks, ratio, and budget must be positive")
    library = library or build_qwen4_exp_qsa(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_select_blocks_f32_i64",
        _ARGS_SELECT,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            scores_ptr,
            block_starts_ptr,
            query_positions_ptr,
            selected_starts_ptr,
            selected_counts_ptr,
            query_count,
            blocks,
            ratio,
            budget,
            stream,
        ),
    )


def register_qwen4_exp_qsa_kernels(*, replace: bool = True) -> None:
    registrations = {
        KernelKey(
            "hip_gfx1100",
            "qsa_pool_norm_rope",
            "f32",
            "strict",
        ): qwen4_exp_qsa_pool_norm_rope_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_index_score",
            "f32",
            "strict",
        ): qwen4_exp_qsa_score_f32,
        KernelKey(
            "hip_gfx1100",
            "qsa_select_blocks",
            "f32_i64",
            "strict",
        ): qwen4_exp_qsa_select_blocks_f32_i64,
    }
    for key, function in registrations.items():
        register(key, function, replace=replace)


def _check_launch(runtime: HipRuntime, error: int) -> None:
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


register_qwen4_exp_qsa_kernels()


__all__ = [
    "build_qwen4_exp_qsa",
    "plan_qwen4_exp_qsa_build",
    "qwen4_exp_qsa_pool_norm_rope_f32",
    "qwen4_exp_qsa_score_f32",
    "qwen4_exp_qsa_select_blocks_f32_i64",
    "register_qwen4_exp_qsa_kernels",
]
