"""Raw-pointer GPU DFlash accept-summary wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("dflash_accept.hip")
_OUTPUT_NAME = "dflash_accept.so"
_SYMBOL_CHAIN_I32 = "hipengine_dflash_accept_chain_i32"


def plan_dflash_accept_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dflash_accept",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dflash_accept(
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
        family="dflash_accept",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dflash_accept_chain_i32(
    token_ids_i32_ptr: int,
    positions_i32_ptr: int,
    parent_rows_i32_ptr: int,
    draft_depths_i32_ptr: int,
    active_mask_u8_ptr: int,
    target_top1_i32_ptr: int,
    remaining_decode_i32_ptr: int | None,
    accepted_counts_i32_ptr: int,
    commit_rows_i32_ptr: int,
    commit_tokens_i32_ptr: int,
    commit_positions_i32_ptr: int,
    next_tokens_i32_ptr: int,
    full_accept_u8_ptr: int,
    committed_output_ids_i32_ptr: int,
    committed_output_lengths_i32_ptr: int,
    rows: int,
    request_count: int,
    output_stride: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Summarize DFlash chain acceptance on device.

    ``token_ids``/``positions``/``parent_rows``/``draft_depths`` describe the
    root-prefixed ``TargetVerifyBatch`` row layout where root rows occupy
    ``[0, request_count)``. ``target_top1[row]`` is produced by the target
    lm-head/argmax path. The kernel writes per-request accepted counts, selected
    commit rows/tokens/positions, next-token bonus/correction ids (``-1`` when
    a caller-supplied remaining budget is exhausted), full-accept flags, and
    committed output-id prefixes ``[root, accepted draft ...]``.
    """

    _check_accept_shape(rows, request_count, output_stride)
    library = library or build_dflash_accept(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_CHAIN_I32)
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
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(token_ids_i32_ptr),
        ctypes.c_void_p(positions_i32_ptr),
        ctypes.c_void_p(parent_rows_i32_ptr),
        ctypes.c_void_p(draft_depths_i32_ptr),
        ctypes.c_void_p(active_mask_u8_ptr),
        ctypes.c_void_p(target_top1_i32_ptr),
        ctypes.c_void_p(remaining_decode_i32_ptr) if remaining_decode_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(accepted_counts_i32_ptr),
        ctypes.c_void_p(commit_rows_i32_ptr),
        ctypes.c_void_p(commit_tokens_i32_ptr),
        ctypes.c_void_p(commit_positions_i32_ptr),
        ctypes.c_void_p(next_tokens_i32_ptr),
        ctypes.c_void_p(full_accept_u8_ptr),
        ctypes.c_void_p(committed_output_ids_i32_ptr),
        ctypes.c_void_p(committed_output_lengths_i32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(request_count),
        ctypes.c_int64(output_stride),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_dflash_accept_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "dflash_accept_chain", "w4_paro", "i32"),
        dflash_accept_chain_i32,
        replace=replace,
    )


def _check_accept_shape(rows: int, request_count: int, output_stride: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if request_count > rows:
        raise ValueError("request_count must be no larger than rows")
    if output_stride <= 0:
        raise ValueError("output_stride must be positive")


register_dflash_accept_kernels()
