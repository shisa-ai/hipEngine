"""Graph-friendly runtime state kernels for Qwen3.5/PARO decode."""

from __future__ import annotations

import ctypes
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
_SYMBOL_SET_I64_VECTOR = "hipengine_set_i64_vector"
_SYMBOL_SET_POSITION = "hipengine_set_decode_position_i64"
_SYMBOL_SET_POSITIONS = "hipengine_set_decode_positions_i64"
_SYMBOL_ADVANCE_POSITION = "hipengine_advance_decode_position_i64"
_SYMBOL_ADVANCE_POSITIONS = "hipengine_advance_decode_positions_i64"
_SYMBOL_RECORD_I64_INDEXED = "hipengine_record_i64_scalar_indexed"
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
