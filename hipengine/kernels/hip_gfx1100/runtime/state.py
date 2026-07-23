"""Graph-friendly runtime state kernels for Qwen3.5/PARO decode."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("state.hip")
_OUTPUT_NAME = "runtime_state.so"
_SYMBOL_EMBEDDING_LOOKUP = "hipengine_embedding_lookup_bf16_i64"
_SYMBOL_EMBEDDING_LOOKUP_BATCH = "hipengine_embedding_lookup_batch_bf16_i64"
_SYMBOL_EMBEDDING_LOOKUP_BATCH_MAPPED = "hipengine_embedding_lookup_batch_mapped_bf16_i64"
_SYMBOL_EMBEDDING_LOOKUP_FP16 = "hipengine_embedding_lookup_fp16_i64"
_SYMBOL_EMBEDDING_LOOKUP_BATCH_FP16 = "hipengine_embedding_lookup_batch_fp16_i64"
_SYMBOL_EMBEDDING_LOOKUP_BATCH_MAPPED_FP16 = "hipengine_embedding_lookup_batch_mapped_fp16_i64"
_SYMBOL_SET_I64 = "hipengine_set_i64_scalar"
_SYMBOL_PREFILL_FLIGHT_RECORDER_MARK_I64 = "hipengine_prefill_flight_recorder_mark_i64"
_SYMBOL_SET_I64_VECTOR = "hipengine_set_i64_vector"
_SYMBOL_COPY_I32_TO_I64 = "hipengine_copy_i32_to_i64"
_SYMBOL_SET_POSITION = "hipengine_set_decode_position_i64"
_SYMBOL_SET_POSITIONS = "hipengine_set_decode_positions_i64"
_SYMBOL_PREPARE_PREFILL_CHUNK_METADATA = "hipengine_prepare_prefill_chunk_metadata"
_SYMBOL_PREPARE_PACKED_DECODE_METADATA = "hipengine_prepare_packed_decode_metadata"
_SYMBOL_PREPARE_PACKED_DECODE_METADATA_FROM_POSITIONS = (
    "hipengine_prepare_packed_decode_metadata_from_positions"
)
_SYMBOL_COMMIT_PACKED_DECODE_GRAPH_STEP = "hipengine_commit_packed_decode_graph_step"
_SYMBOL_RECORD_U16_ROWS_INDEXED = "hipengine_record_u16_rows_indexed"
_SYMBOL_ADVANCE_POSITION = "hipengine_advance_decode_position_i64"
_SYMBOL_ADVANCE_LAGUNA_POSITION_PAIR = "hipengine_advance_laguna_position_pair_i64"
_SYMBOL_ADVANCE_POSITIONS = "hipengine_advance_decode_positions_i64"
_SYMBOL_RECORD_I64_INDEXED = "hipengine_record_i64_scalar_indexed"
_SYMBOL_RECORD_F32_ROW_INDEXED = "hipengine_record_f32_row_indexed"
_SYMBOL_UNPACK_VERIFY_CHAIN_DYNAMIC_METADATA = "hipengine_unpack_verify_chain_dynamic_metadata_i64"


def plan_runtime_state_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="runtime_state",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_runtime_state(
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
        family="runtime_state",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def embedding_lookup_bf16_i64(
    embedding_bf16_ptr: int,
    token_id_i64_ptr: int,
    out_bf16_ptr: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy ``embedding[token_id[0], :]`` to ``out`` using device token state."""

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_EMBEDDING_LOOKUP)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(embedding_bf16_ptr),
        ctypes.c_void_p(token_id_i64_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def embedding_lookup_batch_bf16_i64(
    embedding_bf16_ptr: int,
    token_ids_i64_ptr: int,
    out_bf16_ptr: int,
    tokens: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy ``embedding[token_ids[row], :]`` for a batch of token ids."""

    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_EMBEDDING_LOOKUP_BATCH)
    fn.argtypes = [
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
        ctypes.c_void_p(embedding_bf16_ptr),
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def embedding_lookup_batch_mapped_bf16_i64(
    embedding_bf16_ptr: int,
    token_ids_i64_ptr: int,
    out_bf16_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    token_slots: int,
    *,
    row_map_i32_ptr: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy embeddings for output rows, optionally gathering token ids by row map."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if token_slots <= 0:
        raise ValueError("token_slots must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_EMBEDDING_LOOKUP_BATCH_MAPPED)
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
        ctypes.c_void_p(embedding_bf16_ptr),
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(row_map_i32_ptr) if row_map_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_bf16_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(token_slots),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def embedding_lookup_fp16_i64(
    embedding_fp16_ptr: int,
    token_id_i64_ptr: int,
    out_fp16_ptr: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy ``embedding[token_id[0], :]`` to an FP16 output row."""

    _launch_embedding_lookup(
        _SYMBOL_EMBEDDING_LOOKUP_FP16,
        embedding_fp16_ptr,
        token_id_i64_ptr,
        out_fp16_ptr,
        hidden_size,
        vocab_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def embedding_lookup_batch_fp16_i64(
    embedding_fp16_ptr: int,
    token_ids_i64_ptr: int,
    out_fp16_ptr: int,
    tokens: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy FP16 embeddings for a batch of token ids."""

    _launch_embedding_lookup_batch(
        _SYMBOL_EMBEDDING_LOOKUP_BATCH_FP16,
        embedding_fp16_ptr,
        token_ids_i64_ptr,
        out_fp16_ptr,
        tokens,
        hidden_size,
        vocab_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def embedding_lookup_batch_mapped_fp16_i64(
    embedding_fp16_ptr: int,
    token_ids_i64_ptr: int,
    out_fp16_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    token_slots: int,
    *,
    row_map_i32_ptr: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy FP16 embeddings for output rows, optionally gathering token ids by row map."""

    _launch_embedding_lookup_batch_mapped(
        _SYMBOL_EMBEDDING_LOOKUP_BATCH_MAPPED_FP16,
        embedding_fp16_ptr,
        token_ids_i64_ptr,
        out_fp16_ptr,
        rows,
        hidden_size,
        vocab_size,
        token_slots,
        row_map_i32_ptr=row_map_i32_ptr,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def set_i64_scalar(
    out_i64_ptr: int,
    value: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Set one device int64 scalar."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SET_I64)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(out_i64_ptr), ctypes.c_int64(value), ctypes.c_void_p(stream))
    _check_launch(runtime, err)


def flight_recorder_mark_i64(
    out_i64_ptr: int,
    value: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Publish a same-stream completion sequence to mapped host memory."""

    if out_i64_ptr <= 0:
        raise ValueError("out_i64_ptr must be positive")
    if value <= 0:
        raise ValueError("value must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_FLIGHT_RECORDER_MARK_I64)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(out_i64_ptr), ctypes.c_int64(value), ctypes.c_void_p(stream))
    _check_launch(runtime, err)


def set_i64_vector(
    out_i64_ptr: int,
    values_i64_ptr: int,
    rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Set ``out[row] = values[row]`` for a device int64 vector."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SET_I64_VECTOR)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(out_i64_ptr),
        ctypes.c_void_p(values_i64_ptr),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def copy_i32_to_i64(
    input_i32_ptr: int,
    output_i64_ptr: int,
    rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Exactly widen a device int32 row vector to int64."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_COPY_I32_TO_I64)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(input_i32_ptr),
        ctypes.c_void_p(output_i64_ptr),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def set_decode_position_i64(
    position_i64_ptr: int,
    context_i64_ptr: int,
    value: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Set decode append position and attention context count on device."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SET_POSITION)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(position_i64_ptr),
        ctypes.c_void_p(context_i64_ptr),
        ctypes.c_int64(value),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def set_decode_positions_i64(
    positions_i64_ptr: int,
    contexts_i64_ptr: int,
    values_i64_ptr: int,
    rows: int,
    *,
    active_mask_u8_ptr: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Set batched decode positions/contexts, optionally gated by active mask."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_SET_POSITIONS)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        ctypes.c_void_p(values_i64_ptr),
        ctypes.c_void_p(active_mask_u8_ptr) if active_mask_u8_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def prepare_prefill_chunk_metadata(
    cu_q_i32_ptr: int,
    cu_k_i32_ptr: int,
    atomic_i32_ptr: int,
    gdn_cu_seqlens_i32_ptr: int,
    positions_i64_ptr: int,
    contexts_i64_ptr: int,
    start: int,
    rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Prepare contiguous GGUF prefill metadata entirely on the device."""

    if start < 0:
        raise ValueError("start must be non-negative")
    if rows <= 0:
        raise ValueError("rows must be positive")
    if start + rows > 2**31 - 1:
        raise ValueError("start + rows must fit int32 CU metadata")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREPARE_PREFILL_CHUNK_METADATA)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
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
        ctypes.c_void_p(cu_q_i32_ptr),
        ctypes.c_void_p(cu_k_i32_ptr),
        ctypes.c_void_p(atomic_i32_ptr),
        ctypes.c_void_p(gdn_cu_seqlens_i32_ptr),
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        ctypes.c_int64(start),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def prepare_packed_decode_metadata(
    block_table_i32_ptr: int,
    positions_i64_ptr: int,
    contexts_i64_ptr: int,
    cu_q_i32_ptr: int,
    cu_k_i32_ptr: int,
    atomic_i32_ptr: int,
    gdn_cu_seqlens_i32_ptr: int,
    state_indices_i64_ptr: int,
    position_values: Sequence[int],
    blocks_per_slot: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Prepare singleton packed-decode metadata without host-to-device copies."""

    values = tuple(int(value) for value in position_values)
    if not values:
        raise ValueError("position_values must be non-empty")
    if len(values) > 4:
        raise ValueError("position_values supports at most four packed decode rows")
    if min(values) < 0:
        raise ValueError("position_values must be non-negative")
    if max(values) >= 2**31 - 1:
        raise ValueError("position_values + 1 must fit int32 CU metadata")
    block_count = int(blocks_per_slot)
    if block_count <= 0:
        raise ValueError("blocks_per_slot must be positive")
    if len(values) * block_count > 2**31 - 1:
        raise ValueError("packed block table size must fit int32")
    padded_positions = (*values, *(0 for _ in range(4 - len(values))))

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREPARE_PACKED_DECODE_METADATA)
    fn.argtypes = [
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
        ctypes.c_void_p(block_table_i32_ptr),
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        ctypes.c_void_p(cu_q_i32_ptr),
        ctypes.c_void_p(cu_k_i32_ptr),
        ctypes.c_void_p(atomic_i32_ptr),
        ctypes.c_void_p(gdn_cu_seqlens_i32_ptr),
        ctypes.c_void_p(state_indices_i64_ptr),
        *(ctypes.c_int64(value) for value in padded_positions),
        ctypes.c_int64(len(values)),
        ctypes.c_int64(block_count),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def prepare_packed_decode_metadata_from_positions(
    block_table_i32_ptr: int,
    positions_i64_ptr: int,
    contexts_i64_ptr: int,
    cu_q_i32_ptr: int,
    cu_k_i32_ptr: int,
    atomic_i32_ptr: int,
    gdn_cu_seqlens_i32_ptr: int,
    state_indices_i64_ptr: int,
    rows: int,
    blocks_per_slot: int,
    *,
    active_mask_u8_ptr: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Refresh replayable packed metadata from device-resident positions."""

    row_count = int(rows)
    block_count = int(blocks_per_slot)
    if row_count <= 0 or row_count > 8:
        raise ValueError("rows must be in [1, 8]")
    if block_count <= 0:
        raise ValueError("blocks_per_slot must be positive")
    if row_count * block_count > 2**31 - 1:
        raise ValueError("packed block table size must fit int32")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREPARE_PACKED_DECODE_METADATA_FROM_POSITIONS)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(block_table_i32_ptr),
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        ctypes.c_void_p(cu_q_i32_ptr),
        ctypes.c_void_p(cu_k_i32_ptr),
        ctypes.c_void_p(atomic_i32_ptr),
        ctypes.c_void_p(gdn_cu_seqlens_i32_ptr),
        ctypes.c_void_p(state_indices_i64_ptr),
        (
            ctypes.c_void_p(active_mask_u8_ptr)
            if active_mask_u8_ptr is not None
            else ctypes.c_void_p()
        ),
        ctypes.c_int64(row_count),
        ctypes.c_int64(block_count),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def commit_packed_decode_graph_step(
    token_ids_i32_ptr: int,
    token_ids_i64_ptr: int,
    positions_i64_ptr: int,
    contexts_i64_ptr: int,
    rows: int,
    *,
    active_mask_u8_ptr: int | None = None,
    recorded_token_ids_i32_ptr: int | None = None,
    record_index_i64_ptr: int | None = None,
    record_capacity: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Feed sampled row tokens back to embedding and advance replay state."""

    row_count = int(rows)
    if row_count <= 0 or row_count > 8:
        raise ValueError("rows must be in [1, 8]")
    recording = recorded_token_ids_i32_ptr is not None or record_index_i64_ptr is not None
    if recording and (
        recorded_token_ids_i32_ptr is None
        or record_index_i64_ptr is None
        or int(record_capacity) <= 0
    ):
        raise ValueError("token recording requires output, index, and positive capacity")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_COMMIT_PACKED_DECODE_GRAPH_STEP)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(token_ids_i32_ptr),
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        (
            ctypes.c_void_p(active_mask_u8_ptr)
            if active_mask_u8_ptr is not None
            else ctypes.c_void_p()
        ),
        (
            ctypes.c_void_p(recorded_token_ids_i32_ptr)
            if recorded_token_ids_i32_ptr is not None
            else ctypes.c_void_p()
        ),
        (
            ctypes.c_void_p(record_index_i64_ptr)
            if record_index_i64_ptr is not None
            else ctypes.c_void_p()
        ),
        ctypes.c_int64(int(record_capacity)),
        ctypes.c_int64(row_count),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def record_u16_rows_indexed(
    value_u16_ptr: int,
    out_u16_ptr: int,
    index_i64_ptr: int,
    elements: int,
    step_stride: int,
    capacity: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Record one packed uint16 slab at the current graph-step index."""

    element_count = int(elements)
    stride = int(step_stride)
    step_capacity = int(capacity)
    if element_count <= 0:
        raise ValueError("elements must be positive")
    if stride < element_count:
        raise ValueError("step_stride must cover elements")
    if step_capacity <= 0:
        raise ValueError("capacity must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_RECORD_U16_ROWS_INDEXED)
    fn.argtypes = [
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
        ctypes.c_void_p(value_u16_ptr),
        ctypes.c_void_p(out_u16_ptr),
        ctypes.c_void_p(index_i64_ptr),
        ctypes.c_int64(element_count),
        ctypes.c_int64(stride),
        ctypes.c_int64(step_capacity),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def advance_decode_position_i64(
    position_i64_ptr: int,
    context_i64_ptr: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Increment device decode position and refresh context count."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ADVANCE_POSITION)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(position_i64_ptr), ctypes.c_void_p(context_i64_ptr), ctypes.c_void_p(stream))
    _check_launch(runtime, err)


def advance_laguna_position_pair_i64(
    rope_position_i64_ptr: int,
    kv_position_i64_ptr: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Increment Laguna RoPE/KV position scalars to one identical value."""

    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ADVANCE_LAGUNA_POSITION_PAIR)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(rope_position_i64_ptr),
        ctypes.c_void_p(kv_position_i64_ptr),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def advance_decode_positions_i64(
    positions_i64_ptr: int,
    contexts_i64_ptr: int,
    rows: int,
    *,
    active_mask_u8_ptr: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Increment batched decode positions/contexts, optionally gated by active mask."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_ADVANCE_POSITIONS)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        ctypes.c_void_p(active_mask_u8_ptr) if active_mask_u8_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def record_i64_scalar_indexed(
    value_i64_ptr: int,
    out_i64_ptr: int,
    index_i64_ptr: int,
    capacity: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Append one int64 scalar to ``out[index[0]]`` and increment ``index`` on device."""

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_RECORD_I64_INDEXED)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(value_i64_ptr),
        ctypes.c_void_p(out_i64_ptr),
        ctypes.c_void_p(index_i64_ptr),
        ctypes.c_int64(capacity),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def record_f32_row_indexed(
    value_f32_ptr: int,
    out_f32_ptr: int,
    index_i64_ptr: int,
    cols: int,
    capacity: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Copy one FP32 row to ``out[index[0], :]`` without advancing ``index``.

    This is paired with :func:`record_i64_scalar_indexed` inside decode graphs:
    the hidden-seed row is recorded at the current generated-token index, then
    the token recorder appends the token and advances that shared index.
    """

    if cols <= 0:
        raise ValueError("cols must be positive")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_RECORD_F32_ROW_INDEXED)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(value_f32_ptr),
        ctypes.c_void_p(out_f32_ptr),
        ctypes.c_void_p(index_i64_ptr),
        ctypes.c_int64(cols),
        ctypes.c_int64(capacity),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def unpack_verify_chain_dynamic_metadata_i64(
    packed_i64_ptr: int,
    token_ids_i64_ptr: int,
    token_ids_i32_ptr: int,
    positions_i64_ptr: int,
    positions_i32_ptr: int,
    contexts_i64_ptr: int,
    rows: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Unpack verifier token/position/context metadata from one packed int64 buffer."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_UNPACK_VERIFY_CHAIN_DYNAMIC_METADATA)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(packed_i64_ptr),
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(token_ids_i32_ptr),
        ctypes.c_void_p(positions_i64_ptr),
        ctypes.c_void_p(positions_i32_ptr),
        ctypes.c_void_p(contexts_i64_ptr),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_runtime_state_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "embedding", "bf16", "lookup_bf16_out"),
        embedding_lookup_bf16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "bf16_i64"),
        embedding_lookup_bf16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "batch_bf16_i64"),
        embedding_lookup_batch_bf16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "batch_mapped_bf16_i64"),
        embedding_lookup_batch_mapped_bf16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "fp16_i64"),
        embedding_lookup_fp16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "batch_fp16_i64"),
        embedding_lookup_batch_fp16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "token_embedding", "w4_paro", "batch_mapped_fp16_i64"),
        embedding_lookup_batch_mapped_fp16_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "metadata_cast", "gguf_qwen35", "i32_to_i64"),
        copy_i32_to_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "w4_paro", "set_i64"),
        set_decode_position_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "w4_paro", "set_vector_i64"),
        set_decode_positions_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "prefill_metadata", "gguf_qwen35", "contiguous_chunk"),
        prepare_prefill_chunk_metadata,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_metadata", "gguf_qwen35", "packed_c4_i64"),
        prepare_packed_decode_metadata,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "decode_metadata",
            "gguf_qwen35",
            "packed_c4_device_positions_i64",
        ),
        prepare_packed_decode_metadata_from_positions,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "decode_graph_commit",
            "gguf_qwen35",
            "packed_c4_i32_i64",
        ),
        commit_packed_decode_graph_step,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "decode_metadata",
            "gguf_qwen35",
            "packed_c8_device_positions_i64",
        ),
        prepare_packed_decode_metadata_from_positions,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "decode_graph_commit",
            "gguf_qwen35",
            "packed_c8_i32_i64",
        ),
        commit_packed_decode_graph_step,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "decode_graph_record",
            "gguf_qwen35",
            "packed_u16_rows_indexed",
        ),
        record_u16_rows_indexed,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "w4_paro", "advance_i64"),
        advance_decode_position_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "w4_paro", "advance_vector_i64"),
        advance_decode_positions_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "decode_position", "laguna", "advance_pair_i64"),
        advance_laguna_position_pair_i64,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "scalar_state", "w4_paro", "set_i64"),
        set_i64_scalar,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "scalar_state", "w4_paro", "set_vector_i64"),
        set_i64_vector,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "scalar_state", "w4_paro", "record_i64_indexed"),
        record_i64_scalar_indexed,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "scalar_state", "w4_paro", "record_f32_row_indexed"),
        record_f32_row_indexed,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "verify_metadata", "w4_paro", "unpack_chain_dynamic_i64"),
        unpack_verify_chain_dynamic_metadata_i64,
        replace=replace,
    )


def _launch_embedding_lookup(
    symbol: str,
    embedding_ptr: int,
    token_id_i64_ptr: int,
    out_ptr: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(embedding_ptr),
        ctypes.c_void_p(token_id_i64_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_embedding_lookup_batch(
    symbol: str,
    embedding_ptr: int,
    token_ids_i64_ptr: int,
    out_ptr: int,
    tokens: int,
    hidden_size: int,
    vocab_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    library = library or build_runtime_state(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(embedding_ptr),
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_embedding_lookup_batch_mapped(
    symbol: str,
    embedding_ptr: int,
    token_ids_i64_ptr: int,
    out_ptr: int,
    rows: int,
    hidden_size: int,
    vocab_size: int,
    token_slots: int,
    *,
    row_map_i32_ptr: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if token_slots <= 0:
        raise ValueError("token_slots must be positive")
    library = library or build_runtime_state(load=True)
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
        ctypes.c_void_p(embedding_ptr),
        ctypes.c_void_p(token_ids_i64_ptr),
        ctypes.c_void_p(row_map_i32_ptr) if row_map_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(token_slots),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_runtime_state_kernels()
