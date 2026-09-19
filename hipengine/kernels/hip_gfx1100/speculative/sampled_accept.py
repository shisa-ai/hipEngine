"""Raw-pointer GPU sampled MTP accept-chain wrappers.

The argmax accept chain decides a drafted chain on device by comparing tokens
against the target's top-1 id. A sampled row instead decides with the coupled
acceptance ``min(1, p(x)/q(x))`` and, on rejection, draws from
``normalize(max(0, p - q))``. These wrappers run that decision from device
logits, so a sampled row no longer needs a full-vocabulary logits readback and a
host-side accept walk.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("sampled_accept.hip")
_OUTPUT_NAME = "sampled_accept.so"
_SYMBOL_ROW_STATS = "hipengine_sampled_accept_row_stats_f32"
_SYMBOL_CHAIN = "hipengine_sampled_accept_chain_i32"

# The accept-chain payload the native cycle path consumes; same seven fields as
# the argmax chain so a sampled commit is interchangeable with a greedy one.
SAMPLED_ACCEPT_PACKED_PAYLOAD_FIELDS = 7

# A row's processed logits are reduced to these two scalars for its softmax.
SAMPLED_ACCEPT_ROW_STATS_FIELDS = 2


def plan_sampled_accept_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="sampled_accept",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_sampled_accept(
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
        family="sampled_accept",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def sampled_accept_row_stats_f32(
    processed_logits_f32_ptr: int,
    temperatures_f32_ptr: int,
    row_max_f32_ptr: int,
    row_inv_sum_f32_ptr: int,
    rows: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Reduce each row's processed logits to ``(max, 1/sum)`` for its softmax.

    ``processed_logits`` is row-major FP32 *after* the sampler's logits
    processors, so the row statistics describe exactly the distribution the
    request's own sampler would draw from.
    """

    if rows <= 0 or vocab_size <= 0:
        raise ValueError("sampled accept row stats need positive rows and vocab")
    library = library or build_sampled_accept(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ROW_STATS)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(processed_logits_f32_ptr),
        ctypes.c_void_p(temperatures_f32_ptr),
        ctypes.c_void_p(row_max_f32_ptr),
        ctypes.c_void_p(row_inv_sum_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def sampled_accept_chain_i32(
    processed_logits_f32_ptr: int,
    temperatures_f32_ptr: int,
    row_max_f32_ptr: int,
    row_inv_sum_f32_ptr: int,
    row_seeds_u64_ptr: int,
    step_indices_u64_ptr: int,
    token_ids_i32_ptr: int,
    positions_i32_ptr: int,
    parent_rows_i32_ptr: int,
    draft_depths_i32_ptr: int,
    active_mask_u8_ptr: int,
    remaining_decode_i32_ptr: int | None,
    accepted_counts_i32_ptr: int,
    commit_rows_i32_ptr: int,
    commit_tokens_i32_ptr: int,
    commit_positions_i32_ptr: int,
    next_tokens_i32_ptr: int,
    full_accept_u8_ptr: int,
    committed_output_ids_i32_ptr: int,
    committed_output_lengths_i32_ptr: int,
    packed_payload_i32_ptr: int,
    visible_output_ids_i32_ptr: int,
    visible_output_lengths_i32_ptr: int,
    resident_positions_i64_ptr: int,
    resident_contexts_i64_ptr: int,
    cursor_offset: int,
    rows: int,
    request_count: int,
    output_stride: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Walk one drafted chain per request with a device sampled accept.

    The emitted payload matches ``dflash_accept_chain_i32_native_cycle`` field
    for field, so the resident commit path does not need a sampled variant. The
    accept test and the residual/bonus draw use the row's own counter-based
    sampler stream (seed, step index) exactly like the native row sampler, so a
    row's sampled decisions are reproducible from its seed.
    """

    if rows <= 0 or request_count <= 0 or request_count > rows:
        raise ValueError("sampled accept chain needs 1 <= request_count <= rows")
    if output_stride <= 0:
        raise ValueError("sampled accept chain needs a positive output stride")
    if vocab_size <= 0:
        raise ValueError("sampled accept chain needs a positive vocab size")
    if isinstance(cursor_offset, bool) or not isinstance(cursor_offset, int):
        raise TypeError("cursor_offset must be an integer")
    if cursor_offset < 0 or cursor_offset > (1 << 31) - 1:
        raise ValueError("cursor_offset must fit non-negative int32")
    library = library or build_sampled_accept(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_CHAIN)
    fn.argtypes = [
        *([ctypes.c_void_p] * 25),
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(processed_logits_f32_ptr),
        ctypes.c_void_p(temperatures_f32_ptr),
        ctypes.c_void_p(row_max_f32_ptr),
        ctypes.c_void_p(row_inv_sum_f32_ptr),
        ctypes.c_void_p(row_seeds_u64_ptr),
        ctypes.c_void_p(step_indices_u64_ptr),
        ctypes.c_void_p(token_ids_i32_ptr),
        ctypes.c_void_p(positions_i32_ptr),
        ctypes.c_void_p(parent_rows_i32_ptr),
        ctypes.c_void_p(draft_depths_i32_ptr),
        ctypes.c_void_p(active_mask_u8_ptr),
        ctypes.c_void_p(remaining_decode_i32_ptr)
        if remaining_decode_i32_ptr is not None
        else ctypes.c_void_p(),
        ctypes.c_void_p(accepted_counts_i32_ptr),
        ctypes.c_void_p(commit_rows_i32_ptr),
        ctypes.c_void_p(commit_tokens_i32_ptr),
        ctypes.c_void_p(commit_positions_i32_ptr),
        ctypes.c_void_p(next_tokens_i32_ptr),
        ctypes.c_void_p(full_accept_u8_ptr),
        ctypes.c_void_p(committed_output_ids_i32_ptr),
        ctypes.c_void_p(committed_output_lengths_i32_ptr),
        ctypes.c_void_p(packed_payload_i32_ptr),
        ctypes.c_void_p(visible_output_ids_i32_ptr),
        ctypes.c_void_p(visible_output_lengths_i32_ptr),
        ctypes.c_void_p(resident_positions_i64_ptr),
        ctypes.c_void_p(resident_contexts_i64_ptr),
        ctypes.c_int32(cursor_offset),
        ctypes.c_int64(rows),
        ctypes.c_int64(request_count),
        ctypes.c_int64(output_stride),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_sampled_accept_kernels(*, replace: bool = True) -> None:
    """Register the sampled accept chain on the speculative accept layer."""

    for quant in (
        "w4_paro",
        "w4_gguf",
        "gguf_ud_q3_k_m",
        "gguf_ud_q4_k_m",
        "gguf_ud_q4_k_s",
        "gguf_q4_k_m",
        "gguf_q4_k_s",
    ):
        register(
            KernelKey("hip_gfx1100", "sampled_accept_chain", quant, "native_v1_i32"),
            sampled_accept_chain_i32,
            replace=replace,
        )
    register(
        KernelKey("hip_gfx1100", "sampled_accept_chain", "f32", "native_v1_i32"),
        sampled_accept_chain_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "sampled_accept_row_stats", "f32", "rows"),
        sampled_accept_row_stats_f32,
        replace=replace,
    )


register_sampled_accept_kernels()
