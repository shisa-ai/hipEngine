"""Raw-pointer wrappers for Qwen3.5 linear-attention convolution kernels."""

from __future__ import annotations

import ctypes
from hipengine.kernels.indexed_state import indexed_state_indices
from pathlib import Path

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("conv.hip")
_OUTPUT_NAME = "qwen35_linear_attn_conv.so"
_SYMBOL_F32 = "hipengine_qwen35_linear_attn_conv_decode_f32"
_SYMBOL_BF16 = "hipengine_qwen35_linear_attn_conv_decode_bf16"
_SYMBOL_INDEXED_BF16 = "hipengine_qwen35_linear_attn_conv_decode_indexed_bf16"
_SYMBOL_FP16 = "hipengine_qwen35_linear_attn_conv_decode_fp16"
_SYMBOL_TREE_BF16_TLOOP = "hipengine_qwen35_linear_attn_tree_conv_decode_bf16_tloop"
_SYMBOL_TREE_FP16_TLOOP = "hipengine_qwen35_linear_attn_tree_conv_decode_fp16_tloop"
_SYMBOL_CHAIN_BF16_TLOOP = "hipengine_qwen35_linear_attn_chain_conv_decode_bf16_tloop"
_SYMBOL_CHAIN_BF16_SNAPSHOT_TLOOP = (
    "hipengine_qwen35_linear_attn_chain_conv_decode_bf16_snapshot_tloop"
)
_SYMBOL_ALPHA_BETA_CHAIN_CONV_SNAPSHOT_BF16_F32W = (
    "hipengine_qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w"
)
_SYMBOL_ALPHA_BETA_CONV_DECODE_BF16_F32W = (
    "hipengine_qwen35_linear_attn_alpha_beta_conv_decode_bf16_f32w"
)
_SYMBOL_CHAIN_F32_TLOOP = "hipengine_qwen35_linear_attn_chain_conv_decode_f32_tloop"
_SYMBOL_CHAIN_FP16_TLOOP = "hipengine_qwen35_linear_attn_chain_conv_decode_fp16_tloop"
_SYMBOL_PREFILL_F32 = "hipengine_qwen35_linear_attn_conv_prefill_f32"
_SYMBOL_QWEN4_EXP_PREFILL_EXACT_F32 = (
    "hipengine_qwen4_exp_linear_attn_conv_prefill_exact_f32"
)
_SYMBOL_PREFILL_F32_TILE32X128 = "hipengine_qwen35_linear_attn_conv_prefill_f32_tile32x128"
_SYMBOL_PREFILL_F32_STATE_ROWS = "hipengine_qwen35_linear_attn_conv_prefill_f32_state_rows"
_SYMBOL_PREFILL_FP16 = "hipengine_qwen35_linear_attn_conv_prefill_fp16"
_SYMBOL_PREFILL_SEGMENTS_F32 = "hipengine_qwen35_linear_attn_conv_prefill_segments_f32"
_SYMBOL_PREFILL_SEGMENTS_F32_STATE_ROWS = "hipengine_qwen35_linear_attn_conv_prefill_segments_f32_state_rows"
_SYMBOL_QWEN4_EXP_DECODE_INDEXED_F32 = (
    "hipengine_qwen4_exp_linear_attn_conv_decode_indexed_f32"
)

QWEN4_EXP_CONV_INDEXED_MAX_ROWS = 64
"""Admitted row width of the request-indexed F32 conv decode ABI.

Mirrors ``kQwen4ExpConvIndexedMaxRows`` in ``conv.hip``; the two must agree.
Wider than the serving campaign's C1..C8 range.
"""

QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS = 4096
"""Admitted Conv state slab width of the same ABI.

Mirrors ``kQwen4ExpConvIndexedMaxStateSlots``. The device rejects a
malformed batch with a shared-memory bitmap over the declared slab, so the slab
is bounded to keep that bitmap a fixed 512 bytes.
"""

QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS = 1 << 20
"""Admitted ``channels`` bound.

Together with the kernel-size and state-slot bounds, all derived element and
byte offsets fit in signed 64 bits; callers still own allocation-size checks.
"""

QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE = 1 << 10
"""Admitted ``kernel_size`` bound; see ``..._MAX_CHANNELS``."""

_ARGTYPES_QWEN4_EXP_DECODE_INDEXED_F32 = (
    ctypes.c_void_p,  # hidden_states
    ctypes.c_void_p,  # conv_state
    ctypes.c_void_p,  # conv_weight
    ctypes.c_void_p,  # out
    ctypes.c_void_p,  # state_indices
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # state_slots
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # kernel_size
    ctypes.c_void_p,  # stream
)
_ARGTYPES_ALPHA_BETA_CHAIN_CONV_SNAPSHOT = (
    ctypes.c_void_p,  # norm
    ctypes.c_void_p,  # weight_a
    ctypes.c_void_p,  # weight_b
    ctypes.c_void_p,  # out_a
    ctypes.c_void_p,  # out_b
    ctypes.c_void_p,  # hidden_states
    ctypes.c_void_p,  # base_conv_state
    ctypes.c_void_p,  # chain_conv_state
    ctypes.c_void_p,  # initial_conv_state_snapshot
    ctypes.c_void_p,  # conv_weight
    ctypes.c_void_p,  # conv_out
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # in_features
    ctypes.c_int64,   # out_features
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # kernel_size
    ctypes.c_void_p,  # stream
)
_ARGTYPES_ALPHA_BETA_CONV_DECODE = (
    ctypes.c_void_p,  # norm
    ctypes.c_void_p,  # weight_a
    ctypes.c_void_p,  # weight_b
    ctypes.c_void_p,  # out_a
    ctypes.c_void_p,  # out_b
    ctypes.c_void_p,  # hidden_states
    ctypes.c_void_p,  # conv_state
    ctypes.c_void_p,  # conv_weight
    ctypes.c_void_p,  # conv_out
    ctypes.c_int64,   # in_features
    ctypes.c_int64,   # out_features
    ctypes.c_int64,   # channels
    ctypes.c_int64,   # kernel_size
    ctypes.c_void_p,  # stream
)


def plan_qwen35_linear_attn_conv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen35_linear_attn_conv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen35_linear_attn_conv(
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
        family="qwen35_linear_attn_conv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen35_linear_attn_conv_decode_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 linear-attention decode convolution."""

    _launch_conv(
        _SYMBOL_F32,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_decode_bf16(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-input lowp linear-attention decode convolution."""

    from hipengine.loading.qwen35_gguf_consumer_surface import CONV_SINGLE
    _check_conv_shape(channels, kernel_size)
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    _check_launch(runtime, CONV_SINGLE.launch(library, locals()))


def qwen35_linear_attn_conv_decode_indexed_bf16(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    state_indices_ptr: int,
    rows: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch sparse indexed BF16-input batch decode convolution."""

    _check_positive(rows, "rows")
    _check_conv_shape(channels, kernel_size)
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    from hipengine.loading.qwen35_gguf_consumer_surface import CONV_DECODE
    err = CONV_DECODE.launch(library, locals())
    _check_launch(runtime, err)


def qwen4_exp_conv_indexed_state_indices(
    state_indices: object, *, rows: int, state_slots: int
) -> np.ndarray:
    """Validate per-row Conv state-slot indices for the indexed decode ABI.

    Returns a contiguous ``int64`` vector of ``rows`` entries in ``[0,
    state_slots)``. This is the authoritative admission check: the kernel sees
    only a device pointer and can reject a malformed batch but cannot report
    which row was wrong.

    **Slots must be distinct.** Each row is one independent request's decode
    step, and the decode body shifts and appends inside its slot as it runs, so
    two rows naming one slot would interleave their shifts and commit a state
    that is neither row's result. The kernel also rejects such a batch as a
    whole, but a caller that ignores this check has already lost the ability to
    tell which request was dropped.

    ``state_slots >= rows`` is a capacity bound and is not a uniqueness check;
    uniqueness is enforced separately below.
    """

    return indexed_state_indices(
        state_indices, rows=rows, state_slots=state_slots,
        max_rows=QWEN4_EXP_CONV_INDEXED_MAX_ROWS,
        max_state_slots=QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
        operation='conv', state_name='Conv state',
    )


def qwen4_exp_linear_attn_conv_decode_indexed_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    state_indices_ptr: int,
    *,
    rows: int,
    state_slots: int,
    channels: int,
    kernel_size: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Request-indexed F32 decode convolution for independent rows.

    The multi-row form of :func:`qwen35_linear_attn_conv_decode_f32`. Row ``r``
    of the packed ``hidden_states`` block is one independent request's decode
    step, and row ``r`` reads and writes exactly the Conv state slot
    ``state_indices[r]`` of a ``[state_slots][channels][kernel_size]`` FP32 slab.
    Slots no row selects are not touched.

    The ABI is packed-only, like the single-row owner: hidden and output rows
    stride by ``channels``, the weight by ``channels * kernel_size``, and the
    state by ``channels * kernel_size`` per slot. There are no stride arguments
    to bound; the shape bounds below are what keep the derived offsets in range.

    A malformed batch (a slot outside the slab, or two rows naming one slot) is
    rejected on the device as a whole, leaving every slot and every output row
    unchanged. Use :func:`qwen4_exp_conv_indexed_state_indices` to reject it on
    the host instead, with a diagnosable error.
    """

    count = int(rows)
    slots = int(state_slots)
    if not 1 <= count <= QWEN4_EXP_CONV_INDEXED_MAX_ROWS:
        raise ValueError(
            "indexed conv decode rows must be in "
            f"1..{QWEN4_EXP_CONV_INDEXED_MAX_ROWS}"
        )
    if slots <= 0:
        raise ValueError("state_slots must be positive")
    if slots > QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS:
        raise ValueError(
            "state_slots must be in "
            f"1..{QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS}"
        )
    if slots < count:
        raise ValueError(
            "state_slots must cover one slot per row (state slab too small)"
        )
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels > QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS:
        raise ValueError(
            f"channels must be in 1..{QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS}"
        )
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    if kernel_size > QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE:
        raise ValueError(
            "kernel_size must be in "
            f"1..{QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE}"
        )
    if state_indices_ptr % 8:
        raise ValueError("state_indices must be 8-byte aligned for int64 loads")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_QWEN4_EXP_DECODE_INDEXED_F32,
        _ARGTYPES_QWEN4_EXP_DECODE_INDEXED_F32,
        ctypes.c_int,
    )
    error = fn(
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        state_indices_ptr,
        count,
        slots,
        channels,
        kernel_size,
        stream,
    )
    _check_launch(runtime, error)


def qwen35_linear_attn_conv_decode_fp16(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-input lowp linear-attention decode convolution."""

    _launch_conv(
        _SYMBOL_FP16,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_tree_conv_decode_bf16_tloop(
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    tree_conv_state_ptr: int,
    conv_weight_ptr: int,
    parent_ids_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-input parent-indexed tree convolution t-loop."""

    _launch_tree_conv_tloop(
        _SYMBOL_TREE_BF16_TLOOP,
        hidden_states_ptr,
        base_conv_state_ptr,
        tree_conv_state_ptr,
        conv_weight_ptr,
        parent_ids_ptr,
        out_ptr,
        max_nodes,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_tree_conv_decode_fp16_tloop(
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    tree_conv_state_ptr: int,
    conv_weight_ptr: int,
    parent_ids_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-input parent-indexed tree convolution t-loop."""

    _launch_tree_conv_tloop(
        _SYMBOL_TREE_FP16_TLOOP,
        hidden_states_ptr,
        base_conv_state_ptr,
        tree_conv_state_ptr,
        conv_weight_ptr,
        parent_ids_ptr,
        out_ptr,
        max_nodes,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_chain_conv_decode_bf16_tloop(
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch BF16-input single-chain convolution t-loop with row-state output."""

    _launch_chain_conv_tloop(
        _SYMBOL_CHAIN_BF16_TLOOP,
        hidden_states_ptr,
        base_conv_state_ptr,
        chain_conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        max_nodes,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_chain_conv_decode_bf16_snapshot_tloop(
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    initial_conv_state_snapshot_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Write exact chain rows plus the immutable initial Conv state."""

    _launch_chain_conv_snapshot_tloop(
        _SYMBOL_CHAIN_BF16_SNAPSHOT_TLOOP,
        hidden_states_ptr,
        base_conv_state_ptr,
        chain_conv_state_ptr,
        initial_conv_state_snapshot_ptr,
        conv_weight_ptr,
        out_ptr,
        max_nodes,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w_tloop(
    norm_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    initial_conv_state_snapshot_ptr: int,
    conv_weight_ptr: int,
    conv_out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    channels: int,
    kernel_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact dense alpha/beta plus rollback-safe chain Conv once."""

    if rows < 1 or rows > 4:
        raise ValueError("rows must be between 1 and 4 for alpha/beta chain Conv")
    if in_features != 5120:
        raise ValueError("in_features must equal 5120 for alpha/beta chain Conv")
    if out_features != 48:
        raise ValueError("out_features must equal 48 for alpha/beta chain Conv")
    if channels != 10240:
        raise ValueError("channels must equal 10240 for alpha/beta chain Conv")
    if kernel_size != 4:
        raise ValueError("kernel_size must equal 4 for alpha/beta chain Conv")
    if threads != 256:
        raise ValueError("threads must equal 256 for alpha/beta chain Conv")
    if int(initial_conv_state_snapshot_ptr) <= 0:
        raise ValueError("initial_conv_state_snapshot_ptr must be non-zero")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_ALPHA_BETA_CHAIN_CONV_SNAPSHOT_BF16_F32W,
        _ARGTYPES_ALPHA_BETA_CHAIN_CONV_SNAPSHOT,
        ctypes.c_int,
    )
    err = fn(
        norm_ptr,
        weight_a_ptr,
        weight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        hidden_states_ptr,
        base_conv_state_ptr,
        chain_conv_state_ptr,
        initial_conv_state_snapshot_ptr,
        conv_weight_ptr,
        conv_out_ptr,
        rows,
        in_features,
        out_features,
        channels,
        kernel_size,
        stream,
    )
    _check_launch(runtime, err)


def qwen35_linear_attn_alpha_beta_conv_decode_bf16_f32w(
    norm_ptr: int,
    weight_a_ptr: int,
    weight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    conv_out_ptr: int,
    in_features: int,
    out_features: int,
    channels: int,
    kernel_size: int,
    *,
    threads: int = 256,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact scalar alpha/beta projections and in-place Conv once."""

    if in_features != 5120:
        raise ValueError("in_features must equal 5120 for alpha/beta serial Conv")
    if out_features != 48:
        raise ValueError("out_features must equal 48 for alpha/beta serial Conv")
    if channels != 10240:
        raise ValueError("channels must equal 10240 for alpha/beta serial Conv")
    if kernel_size != 4:
        raise ValueError("kernel_size must equal 4 for alpha/beta serial Conv")
    if threads != 256:
        raise ValueError("threads must equal 256 for alpha/beta serial Conv")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        _SYMBOL_ALPHA_BETA_CONV_DECODE_BF16_F32W,
        _ARGTYPES_ALPHA_BETA_CONV_DECODE,
        ctypes.c_int,
    )
    err = fn(
        norm_ptr,
        weight_a_ptr,
        weight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        conv_out_ptr,
        in_features,
        out_features,
        channels,
        kernel_size,
        stream,
    )
    _check_launch(runtime, err)


def qwen35_linear_attn_chain_conv_decode_f32_tloop(
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32-input single-chain convolution t-loop with row-state output."""

    _launch_chain_conv_tloop(
        _SYMBOL_CHAIN_F32_TLOOP,
        hidden_states_ptr,
        base_conv_state_ptr,
        chain_conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        max_nodes,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_chain_conv_decode_fp16_tloop(
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-input single-chain convolution t-loop with row-state output."""

    _launch_chain_conv_tloop(
        _SYMBOL_CHAIN_FP16_TLOOP,
        hidden_states_ptr,
        base_conv_state_ptr,
        chain_conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        max_nodes,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_prefill_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 native prefill convolution and update conv_state."""

    from hipengine.loading.qwen35_gguf_consumer_surface import CONV_PREFILL
    _check_positive(tokens, "tokens")
    _check_conv_shape(channels, kernel_size)
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    _check_launch(runtime, CONV_PREFILL.launch(library, locals()))


def qwen4_exp_linear_attn_conv_prefill_exact_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    state_rows_ptr: int = 0,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch bulk K4 Conv with strict decode arithmetic per prompt row.

    ``state_rows_ptr`` also receives the conv state after every token
    (``[tokens][channels][kernel_size]``), for speculative verification.
    """

    _check_conv_shape(channels, kernel_size)
    _check_positive(tokens, "tokens")
    if kernel_size != 4:
        raise ValueError("Qwen4Exp exact prefill requires kernel_size == 4")
    if state_rows_ptr:
        library = library or build_qwen35_linear_attn_conv(load=True)
        runtime = runtime or get_hip_runtime()
        fn = signed_kernel_fn(
            library,
            "hipengine_qwen4_exp_linear_attn_conv_prefill_exact_state_rows_f32",
            (ctypes.c_void_p,) * 5 + (ctypes.c_int64,) * 3 + (ctypes.c_void_p,),
            ctypes.c_int,
        )
        error = fn(hidden_states_ptr, conv_state_ptr, conv_weight_ptr, out_ptr,
                   state_rows_ptr, tokens, channels, kernel_size, stream)
        if int(error) != HIP_SUCCESS:
            runtime.check(int(error))
        return
    _launch_conv_prefill(
        _SYMBOL_QWEN4_EXP_PREFILL_EXACT_F32,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        tokens,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_prefill_f32_tile32x128(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the exact 32-token/128-channel FP32 prefill candidate."""

    _check_conv_shape(channels, kernel_size)
    _check_positive(tokens, "tokens")
    if kernel_size != 4:
        raise ValueError("tile32x128 prefill requires kernel_size == 4")
    _launch_conv_prefill(
        _SYMBOL_PREFILL_F32_TILE32X128,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        tokens,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_prefill_f32_state_rows(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_state_rows_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP32 prefill convolution and materialize per-row Conv states."""

    _check_conv_shape(channels, kernel_size)
    _check_positive(tokens, "tokens")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_F32_STATE_ROWS)
    fn.argtypes = [
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_state_rows_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def qwen35_linear_attn_conv_prefill_fp16(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch FP16-input native prefill convolution and update conv_state."""

    _launch_conv_prefill(
        _SYMBOL_PREFILL_FP16,
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        tokens,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_prefill_segments_f32(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch segment-aware FP32 prefill convolution over packed prompt rows.

    ``cu_seqlens`` has shape ``[segments + 1]`` and defines packed prompt
    ranges. ``state_indices`` maps each segment to the leading conv-state slot
    in a ``[state_slots, channels, kernel_size]`` state slab. Unlike the legacy
    one-request wrapper, short segments are allowed and update only their own
    mapped state slot.
    """

    _launch_conv_prefill_segments(
        hidden_states_ptr,
        conv_state_ptr,
        conv_weight_ptr,
        out_ptr,
        cu_seqlens_ptr,
        state_indices_ptr,
        total_tokens,
        segments,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def qwen35_linear_attn_conv_prefill_segments_f32_state_rows(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_state_rows_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch segment-aware FP32 prefill convolution and row-state capture.

    This is the packed-verifier sibling of
    :func:`qwen35_linear_attn_conv_prefill_f32_state_rows`: each segment reads
    its own initial state slot from ``conv_state`` via ``state_indices`` and
    writes per-row Conv states into the slot-major packed row slab without
    mutating the initial state.
    """

    _launch_conv_prefill_segments_state_rows(
        hidden_states_ptr,
        conv_state_ptr,
        conv_state_rows_ptr,
        conv_weight_ptr,
        out_ptr,
        cu_seqlens_ptr,
        state_indices_ptr,
        total_tokens,
        segments,
        channels,
        kernel_size,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_qwen35_linear_attn_conv_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "w4_paro", "f32"),
        qwen35_linear_attn_conv_decode_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "w4_paro", "bf16"),
        qwen35_linear_attn_conv_decode_bf16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "w4_paro", "fp16"),
        qwen35_linear_attn_conv_decode_fp16,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_decode", "gguf_qwen35", "bf16_indexed"),
        qwen35_linear_attn_conv_decode_indexed_bf16,
        replace=replace,
    )
    # Request-indexed F32 decode: row r advances exactly its own Conv state slot.
    # The single-row F32 owner above stays the strict fallback and the per-row
    # arithmetic oracle.
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_conv_decode",
            "f32_state",
            "qwen4exp_conv_indexed_decode",
        ),
        qwen4_exp_linear_attn_conv_decode_indexed_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_chain_conv_decode",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop",
        ),
        qwen35_linear_attn_chain_conv_decode_bf16_tloop,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_chain_conv_decode+snapshot",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop",
        ),
        qwen35_linear_attn_chain_conv_decode_bf16_snapshot_tloop,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_alpha_beta+chain_conv+snapshot",
            "f32",
            "bf16_k5120_n48_c10240_k4_exact_state_rows_tloop",
        ),
        qwen35_linear_attn_alpha_beta_chain_conv_snapshot_bf16_f32w_tloop,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_alpha_beta+conv_decode",
            "f32",
            "bf16_k5120_n48_c10240_k4_c1",
        ),
        qwen35_linear_attn_alpha_beta_conv_decode_bf16_f32w,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "w4_paro", "f32"),
        qwen35_linear_attn_conv_prefill_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "w4_paro", "fp16"),
        qwen35_linear_attn_conv_prefill_fp16,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_attn_conv_prefill",
            "qwen4_exp",
            "f32_decode_exact_k4",
        ),
        qwen4_exp_linear_attn_conv_prefill_exact_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "gguf_qwen35", "f32_baseline"),
        qwen35_linear_attn_conv_prefill_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "gguf_qwen35", "f32_tile32x128"),
        qwen35_linear_attn_conv_prefill_f32_tile32x128,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "w4_paro", "f32_segments"),
        qwen35_linear_attn_conv_prefill_segments_f32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "linear_attn_conv_prefill", "w4_paro", "f32_segments_state_rows"),
        qwen35_linear_attn_conv_prefill_segments_f32_state_rows,
        replace=replace,
    )
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        register(
            KernelKey(backend, "linear_attn_tree_conv_decode", "w4_paro", "bf16_tloop"),
            qwen35_linear_attn_tree_conv_decode_bf16_tloop,
            replace=replace,
        )
        register(
            KernelKey(backend, "linear_attn_tree_conv_decode", "w4_paro", "fp16_tloop"),
            qwen35_linear_attn_tree_conv_decode_fp16_tloop,
            replace=replace,
        )
        register(
            KernelKey(backend, "linear_attn_chain_conv_decode", "gguf_qwen35", "f32_tloop"),
            qwen35_linear_attn_chain_conv_decode_f32_tloop,
            replace=replace,
        )


def _launch_conv(
    symbol: str,
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_conv_shape(channels, kernel_size)
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_tree_conv_tloop(
    symbol: str,
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    tree_conv_state_ptr: int,
    conv_weight_ptr: int,
    parent_ids_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(max_nodes, "max_nodes")
    _check_conv_shape(channels, kernel_size)
    if kernel_size < 2:
        raise ValueError("tree conv requires kernel_size >= 2")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(base_conv_state_ptr),
        ctypes.c_void_p(tree_conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(parent_ids_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(max_nodes),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_chain_conv_tloop(
    symbol: str,
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(max_nodes, "max_nodes")
    _check_conv_shape(channels, kernel_size)
    if kernel_size < 2 or kernel_size > 8:
        raise ValueError("chain conv requires 2 <= kernel_size <= 8")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(base_conv_state_ptr),
        ctypes.c_void_p(chain_conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(max_nodes),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_chain_conv_snapshot_tloop(
    symbol: str,
    hidden_states_ptr: int,
    base_conv_state_ptr: int,
    chain_conv_state_ptr: int,
    initial_conv_state_snapshot_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    max_nodes: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_positive(max_nodes, "max_nodes")
    _check_conv_shape(channels, kernel_size)
    if kernel_size < 2 or kernel_size > 8:
        raise ValueError("chain conv requires 2 <= kernel_size <= 8")
    if int(initial_conv_state_snapshot_ptr) <= 0:
        raise ValueError("initial_conv_state_snapshot_ptr must be non-zero")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
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
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(base_conv_state_ptr),
        ctypes.c_void_p(chain_conv_state_ptr),
        ctypes.c_void_p(initial_conv_state_snapshot_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(max_nodes),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_conv_prefill(
    symbol: str,
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    tokens: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_conv_shape(channels, kernel_size)
    _check_positive(tokens, "tokens")
    library = library or build_qwen35_linear_attn_conv(load=True)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(tokens),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_conv_prefill_segments(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_conv_shape(channels, kernel_size)
    _check_positive(total_tokens, "total_tokens")
    _check_positive(segments, "segments")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_SEGMENTS_F32)
    fn.argtypes = [
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_conv_prefill_segments_state_rows(
    hidden_states_ptr: int,
    conv_state_ptr: int,
    conv_state_rows_ptr: int,
    conv_weight_ptr: int,
    out_ptr: int,
    cu_seqlens_ptr: int,
    state_indices_ptr: int,
    total_tokens: int,
    segments: int,
    channels: int,
    kernel_size: int,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
) -> None:
    _check_conv_shape(channels, kernel_size)
    _check_positive(total_tokens, "total_tokens")
    _check_positive(segments, "segments")
    library = library or build_qwen35_linear_attn_conv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PREFILL_SEGMENTS_F32_STATE_ROWS)
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
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(hidden_states_ptr),
        ctypes.c_void_p(conv_state_ptr),
        ctypes.c_void_p(conv_state_rows_ptr),
        ctypes.c_void_p(conv_weight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_void_p(cu_seqlens_ptr),
        ctypes.c_void_p(state_indices_ptr),
        ctypes.c_int64(total_tokens),
        ctypes.c_int64(segments),
        ctypes.c_int64(channels),
        ctypes.c_int64(kernel_size),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _check_conv_shape(channels: int, kernel_size: int) -> None:
    from hipengine.loading.qwen35_gguf_consumer_surface import validate_conv_geometry
    validate_conv_geometry(channels, kernel_size)


def _check_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_qwen35_linear_attn_conv_kernels()
