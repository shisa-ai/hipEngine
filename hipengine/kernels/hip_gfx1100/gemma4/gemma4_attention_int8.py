"""Raw-pointer wrappers for the Gemma 4 direct per-token/head INT8 decode consumer.

The strict consumer for the BF16-source INT8 KV writer. It reads the writer's
exact representation (INT8 payload plus per-token/per-KV-head FP16 or FP32
scale), reconstructs K/V as ``float32(int8) * float32(scale)`` in FP32, and
computes one masked decode row per query head over the ``KVLiveSpans`` paged
cache. The BF16 Gemma 4 attention kernels are untouched; this is a new sibling.

Span contract, checked before any build or launch:

* ``spans_mode == "uniform"`` -- the paged per-token/head INT8 layout is a fixed
  page table, so per-head-variable and sliding-ring spans are refused.
* ``storage_dtype == INT8_PER_TOKEN_HEAD`` with ``per_token_head`` scale
  metadata. Grouped granularities (``block16``, ``hadamard_group32``) and missing
  scale metadata are refused explicitly rather than silently misread.
* ``token_positions`` / ``evict_mask`` / ``row_positions`` are optional. When
  absent the dense fill is uniform: slot ``j`` is at position ``j`` and the query
  sits at ``context_len - 1``. When present they carry the sliding-window and
  eviction semantics.

Capability is the only admission gate: the two Gemma 4 GQA geometries
``(16, 8, 256)`` and ``(16, 2, 512)`` run; anything else fails with a named
capability miss. No model, path, or artifact identity is consulted.

Single-row contract: ``live_counts`` must hold exactly one count and
``row_positions`` (when present) exactly one position; ``base_offsets`` must be a
1-D page table. A multi-row span is refused before build.

Checked error-reporting path. This strict prerequisite validates the small
device metadata (the live count and the page-table prefix the launch will
dereference) with a synchronous device-to-host read before launch, and raises a
named ``ValueError`` for a negative or over-capacity live count, or a physical
page id outside ``[0, cache_blocks)``. It is therefore **not** the asynchronous
hot path: a production caller that needs async dispatch would validate once at
admission and use the kernel's own defensive guards. The readback is a
synchronous ``hipMemcpy``, which is not ordered against a non-blocking producer
stream, so a supplied ``stream`` is synchronized before the read; the checked
path makes no graph-capture or async promise. The kernel repeats every check and
fails loudly (NaN row) rather than reading out of bounds or truncating silently,
so a direct kernel caller is never left with a success-shaped wrong answer.

Host-versus-device split. The host rejects structural invalidity -- a malformed
count or an out-of-range page id **anywhere in the live prefix, even a masked
slot**, because it validates the whole prefix the kernel will index. The device
only fails a *visible* out-of-range page (a masked slot is never dereferenced),
and reports a numerical failure (a non-finite visible logit or reconstructed
value) as a NaN row rather than a Python error. Empty (``0``) is supported and
distinct from a negative count.

Failure policy: an empty span (``live_counts[0] == 0``) is supported and returns
zeros. A negative count, a count above ``max_context_len``, or an out-of-range
visible page id is a structural failure reported as a ``ValueError`` here and as
a NaN row by the kernel. A non-finite *visible* logit or reconstructed V is a
numerical failure reported as a NaN row by the kernel (the host does not inspect
values). A masked slot never participates, so a poisoned masked K/V scale cannot
leak into the result.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.dtype import DType
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, MemcpyKind, get_hip_runtime
from hipengine.core.memory import host_buffer_ptr
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("gemma4_attention_int8.hip")
_OUTPUT_NAME = "gemma4_attention_int8.so"

_SYMBOL_DECODE_SCALE_F32 = (
    "hipengine_gemma4_attention_decode_int8_per_token_head_scale_f32_spans"
)
_SYMBOL_DECODE_SCALE_FP16 = (
    "hipengine_gemma4_attention_decode_int8_per_token_head_scale_fp16_spans"
)

# Gemma 4 26B A4B GQA geometries: sliding layers (16 query / 8 KV heads, 256) and
# global layers (16 query / 2 KV heads, 512). A capability declaration, not an
# artifact allowlist.
_SUPPORTED_GEOMETRIES = frozenset({(16, 8, 256), (16, 2, 512)})

_GEMMA4_INT8_ATTENTION_QUANT = "int8_per_token_head"
_GEMMA4_INT8_ATTENTION_VARIANT = "gemma4_direct_spans"

_MAX_SHARED_BYTES = 64 * 1024
_THREADS = 256
_NUM_WARPS = _THREADS // 32

_ARGTYPES = (
    ctypes.c_void_p,  # query
    ctypes.c_void_p,  # key_cache
    ctypes.c_void_p,  # value_cache
    ctypes.c_void_p,  # k_scale
    ctypes.c_void_p,  # v_scale
    ctypes.c_void_p,  # out
    ctypes.c_void_p,  # base_offsets
    ctypes.c_void_p,  # live_counts
    ctypes.c_void_p,  # token_positions
    ctypes.c_void_p,  # evict_mask
    ctypes.c_void_p,  # row_positions
    ctypes.c_int64,  # capacity
    ctypes.c_int64,  # cache_blocks
    ctypes.c_int64,  # block_table_len
    ctypes.c_int64,  # block_size
    ctypes.c_int64,  # num_q_heads
    ctypes.c_int64,  # num_kv_heads
    ctypes.c_int64,  # head_dim
    ctypes.c_int64,  # sliding_window
    ctypes.c_float,  # scale
    ctypes.c_void_p,  # stream
)


def plan_gemma4_attention_int8_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_attention_int8",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_attention_int8(
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
        family="gemma4_attention_int8",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _check_positive(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_contiguous(tensor: object, name: str) -> None:
    """Refuse a non-contiguous view whose raw pointer the kernel would misread."""

    strides = getattr(tensor, "strides", None)
    if strides is None:
        return
    expected: list[int] = []
    running = 1
    for dim in reversed(tuple(getattr(tensor, "shape"))):
        expected.append(running)
        running *= int(dim)
    if tuple(int(stride) for stride in strides) != tuple(reversed(expected)):
        raise ValueError(
            f"Gemma 4 direct INT8 attention requires a contiguous {name} tensor"
        )


def _read_device_metadata(
    spans: KVLiveSpans, used_blocks: int, runtime: HipRuntime, stream: int = 0
) -> tuple[int, int | None, list[int]]:
    """Read the live count, row position, and page-table prefix from the device.

    Synchronous by design: this is the checked path, not the async hot path. The
    synchronous ``hipMemcpy`` used for the readback is **not** ordered against a
    non-blocking producer stream, so a supplied ``stream`` is synchronized first.
    The checked path makes no graph-capture or async promise.
    """

    if int(stream):
        runtime.stream_synchronize(int(stream))
    live = (ctypes.c_int64 * 1)()
    runtime.memcpy(
        host_buffer_ptr(live),
        int(spans.live_counts.ptr),
        ctypes.sizeof(live),
        MemcpyKind.DEVICE_TO_HOST,
    )
    row_position: int | None = None
    if spans.row_positions is not None:
        row = (ctypes.c_int64 * 1)()
        runtime.memcpy(
            host_buffer_ptr(row),
            int(spans.row_positions.ptr),
            ctypes.sizeof(row),
            MemcpyKind.DEVICE_TO_HOST,
        )
        row_position = int(row[0])
    table = (ctypes.c_int32 * used_blocks)()
    runtime.memcpy(
        host_buffer_ptr(table),
        int(spans.base_offsets.ptr),
        ctypes.sizeof(table),
        MemcpyKind.DEVICE_TO_HOST,
    )
    return int(live[0]), row_position, [int(table[i]) for i in range(used_blocks)]


def _check_device_metadata(
    spans: KVLiveSpans,
    required_blocks: int,
    cache_blocks: int,
    block_size: int,
    max_context_len: int,
    runtime: HipRuntime,
    stream: int = 0,
) -> None:
    """Reject invalid counts and out-of-range physical page ids before launch.

    A negative or over-capacity count is reported, never silently truncated. A
    negative or ``>= cache_blocks`` physical page id in the *live prefix* is
    reported, never read out of bounds -- even for a masked slot, because the
    host validates the prefix the kernel will index. Empty (``0``) is supported
    and distinct from a negative count. This is the host half of the contract:
    structural invalidity raises ``ValueError`` here; a numerical failure (a
    non-finite visible logit or value) is left to the kernel, which writes NaN.
    """

    live_count, _row_position, physical = _read_device_metadata(
        spans, required_blocks, runtime, stream
    )
    if live_count < 0:
        raise ValueError(
            f"live_counts[0]={live_count} is negative; a malformed count is not a "
            "supported empty span"
        )
    if live_count > int(max_context_len):
        raise ValueError(
            f"live_counts[0]={live_count} exceeds max_context_len={int(max_context_len)}; "
            "the count would be silently truncated"
        )
    used_blocks = (live_count + int(block_size) - 1) // int(block_size)
    for logical_block in range(used_blocks):
        physical_block = physical[logical_block]
        if physical_block < 0 or physical_block >= int(cache_blocks):
            raise ValueError(
                f"base_offsets[{logical_block}]={physical_block} is outside "
                f"[0, {int(cache_blocks)}); the page id would read out of bounds"
            )


def _decode_symbol(spans: KVLiveSpans) -> str:
    metadata = spans.scale_metadata
    if metadata is None:
        raise ValueError(
            "Gemma 4 direct INT8 attention requires int8_per_token_head scale metadata"
        )
    if metadata.granularity != "per_token_head":
        raise ValueError(
            "Gemma 4 direct INT8 attention requires per_token_head scale granularity; "
            f"got {metadata.granularity!r}"
        )
    if metadata.scale_dtype == DType.FP16:
        return _SYMBOL_DECODE_SCALE_FP16
    if metadata.scale_dtype == DType.FP32:
        return _SYMBOL_DECODE_SCALE_F32
    raise ValueError("Gemma 4 direct INT8 attention scales must be fp16 or fp32")


def _check_decode_shape(
    spans: KVLiveSpans,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int]:
    """Validate the span/scale/geometry contract before any build or launch."""

    if spans.spans_mode != "uniform":
        raise ValueError(
            "Gemma 4 direct INT8 attention requires uniform spans; "
            f"got {spans.spans_mode!r}"
        )
    if spans.storage_dtype != DType.INT8_PER_TOKEN_HEAD:
        raise ValueError(
            "Gemma 4 direct INT8 attention requires int8_per_token_head storage spans"
        )
    if spans.live_counts.dtype != DType.INT64:
        raise ValueError("Gemma 4 direct INT8 attention requires int64 live_counts")
    metadata = spans.scale_metadata
    if metadata is None:  # defensive; KVLiveSpans rejects this combination first.
        raise ValueError("int8_per_token_head spans require scale metadata")
    if metadata.granularity != "per_token_head":
        raise ValueError(
            "Gemma 4 direct INT8 attention requires per_token_head scale granularity; "
            f"got {metadata.granularity!r}"
        )
    if metadata.scale_dtype not in {DType.FP16, DType.FP32}:
        raise ValueError("Gemma 4 direct INT8 attention scales must be fp16 or fp32")
    if len(metadata.k_scale.shape) != 3 or len(metadata.v_scale.shape) != 3:
        raise ValueError(
            "Gemma 4 direct INT8 attention scale tensors must have shape "
            "[blocks, block_size, num_kv_heads]"
        )
    scale_blocks, scale_block_size, scale_heads = (int(dim) for dim in metadata.k_scale.shape)
    v_blocks, v_block_size, v_heads = (int(dim) for dim in metadata.v_scale.shape)
    if scale_block_size != block_size or scale_heads != num_kv_heads:
        raise ValueError(
            "Gemma 4 direct INT8 attention scale shape must match block_size and num_kv_heads"
        )
    if v_block_size != block_size or v_heads != num_kv_heads:
        raise ValueError(
            "Gemma 4 direct INT8 attention value-scale shape must match block_size "
            "and num_kv_heads"
        )
    if v_blocks != scale_blocks:
        raise ValueError("Gemma 4 direct INT8 attention K/V scales must have the same block count")

    _require_contiguous(metadata.k_scale, "k_scale")
    _require_contiguous(metadata.v_scale, "v_scale")
    _require_contiguous(spans.base_offsets, "base_offsets")
    _require_contiguous(spans.live_counts, "live_counts")
    if spans.token_positions is not None:
        _require_contiguous(spans.token_positions, "token_positions")
    if spans.evict_mask is not None:
        _require_contiguous(spans.evict_mask, "evict_mask")
    if spans.row_positions is not None:
        _require_contiguous(spans.row_positions, "row_positions")

    _check_positive(max_context_len, "max_context_len")
    _check_positive(block_size, "block_size")
    _check_positive(num_q_heads, "num_q_heads")
    _check_positive(num_kv_heads, "num_kv_heads")
    _check_positive(head_dim, "head_dim")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if (num_q_heads, num_kv_heads, head_dim) not in _SUPPORTED_GEOMETRIES:
        raise ValueError(
            "Gemma 4 direct INT8 attention supports GQA geometries "
            "(num_q_heads, num_kv_heads, head_dim) in "
            f"{sorted(_SUPPORTED_GEOMETRIES)}; got "
            f"({num_q_heads}, {num_kv_heads}, {head_dim})"
        )

    block_table_len = int(spans.base_offsets.numel)
    _check_positive(block_table_len, "block_table_len")
    if len(spans.base_offsets.shape) != 1:
        raise ValueError(
            "Gemma 4 direct INT8 attention requires a 1-D single-row base_offsets "
            f"page table; got rank {len(spans.base_offsets.shape)}"
        )
    required_blocks = (int(max_context_len) + int(block_size) - 1) // int(block_size)
    if required_blocks > block_table_len:
        raise ValueError(
            "Gemma 4 direct INT8 attention span block table is too short for "
            "max_context_len"
        )
    if required_blocks > scale_blocks:
        raise ValueError(
            "Gemma 4 direct INT8 attention scale tensors do not cover max_context_len"
        )
    if spans.live_counts.numel != 1:
        raise ValueError(
            "Gemma 4 direct INT8 attention is single-row: live_counts must hold "
            f"exactly one count; got {spans.live_counts.numel}"
        )
    if spans.token_positions is not None:
        if spans.token_positions.dtype != DType.INT64:
            raise ValueError("Gemma 4 direct INT8 attention token_positions must be int64")
        if spans.token_positions.numel < int(max_context_len):
            raise ValueError("token_positions must cover max_context_len slots")
    if spans.evict_mask is not None and spans.evict_mask.numel < int(max_context_len):
        raise ValueError("evict_mask must cover max_context_len slots")
    if spans.row_positions is not None:
        if spans.row_positions.dtype != DType.INT64:
            raise ValueError("Gemma 4 direct INT8 attention row_positions must be int64")
        if spans.row_positions.numel != 1:
            raise ValueError(
                "Gemma 4 direct INT8 attention is single-row: row_positions must "
                f"hold exactly one position; got {spans.row_positions.numel}"
            )

    shared_bytes = (int(max_context_len) + int(head_dim) + 2 * _NUM_WARPS) * 4
    if shared_bytes > _MAX_SHARED_BYTES:
        raise ValueError(
            f"Gemma 4 direct INT8 attention needs {shared_bytes} bytes of shared memory "
            f"for max_context_len={max_context_len}, head_dim={head_dim}; this kernel "
            f"supports at most {_MAX_SHARED_BYTES} bytes"
        )
    return block_table_len, scale_blocks


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gemma4_attention_decode_int8_per_token_head_spans(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    max_context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float = 1.0,
    sliding_window: int | None = None,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run one masked Gemma 4 INT8 decode row directly from a paged INT8 cache.

    ``query`` is ``[num_q_heads, head_dim]`` FP32, ``out`` is
    ``[num_q_heads, head_dim]`` FP32, and ``key_cache``/``value_cache`` are the
    writer's ``[blocks, block_size, num_kv_heads, head_dim]`` INT8 caches with
    ``k_scale``/``v_scale`` taken from ``spans.scale_metadata``. ``max_context_len``
    bounds ``spans.live_counts[0]`` and sizes the logit scratch.

    ``sliding_window`` enables the Gemma 4 sliding bound when positive; ``None``
    or a non-positive value is a global (causal-only) layer. ``scale`` is Gemma
    4's ``geometry.scale`` (1.0).

    The live count and page-table prefix are validated with a synchronous
    device-to-host read before launch (see the module docstring); invalid counts
    or physical page ids raise ``ValueError`` instead of launching.
    """

    block_table_len, cache_blocks = _check_decode_shape(
        spans,
        max_context_len,
        block_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
    symbol = _decode_symbol(spans)
    metadata = spans.scale_metadata
    assert metadata is not None  # validated above.
    runtime = runtime or get_hip_runtime()
    required_blocks = (int(max_context_len) + int(block_size) - 1) // int(block_size)
    _check_device_metadata(
        spans, required_blocks, cache_blocks, block_size, max_context_len, runtime, stream
    )
    library = library or build_gemma4_attention_int8(load=True)

    token_positions_ptr = 0 if spans.token_positions is None else int(spans.token_positions.ptr)
    evict_mask_ptr = 0 if spans.evict_mask is None else int(spans.evict_mask.ptr)
    row_positions_ptr = 0 if spans.row_positions is None else int(spans.row_positions.ptr)
    window = 0 if sliding_window is None else int(sliding_window)

    fn = getattr(library, symbol)
    fn.argtypes = list(_ARGTYPES)
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(query_ptr),
        ctypes.c_void_p(key_cache_ptr),
        ctypes.c_void_p(value_cache_ptr),
        ctypes.c_void_p(metadata.k_scale.ptr),
        ctypes.c_void_p(metadata.v_scale.ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(spans.base_offsets.ptr),
        ctypes.c_void_p(spans.live_counts.ptr),
        ctypes.c_void_p(token_positions_ptr),
        ctypes.c_void_p(evict_mask_ptr),
        ctypes.c_void_p(row_positions_ptr),
        ctypes.c_int64(int(max_context_len)),
        ctypes.c_int64(cache_blocks),
        ctypes.c_int64(block_table_len),
        ctypes.c_int64(int(block_size)),
        ctypes.c_int64(int(num_q_heads)),
        ctypes.c_int64(int(num_kv_heads)),
        ctypes.c_int64(int(head_dim)),
        ctypes.c_int64(window),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def register_gemma4_int8_attention_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 direct INT8 decode consumer against the four-axis registry.

    Storage quant is the key's quant axis (the cache is INT8), matching the
    Qwen3.5 INT8 consumers; the variant names the Gemma-specific direct route.
    """

    register(
        KernelKey(
            "hip_gfx1100",
            "paged_attn_decode",
            _GEMMA4_INT8_ATTENTION_QUANT,
            _GEMMA4_INT8_ATTENTION_VARIANT,
        ),
        gemma4_attention_decode_int8_per_token_head_spans,
        replace=replace,
    )
