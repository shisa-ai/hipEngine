"""Raw-pointer wrappers for strict Qwen4Exp PLE HIP primitives."""

from __future__ import annotations

import ctypes
from hipengine.kernels.indexed_state import indexed_state_indices
from pathlib import Path

import numpy as np

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("qwen4_exp_ple.hip")
_OUTPUT_NAME = "qwen4_exp_ple.so"

# Bounds of the request-indexed PLE Conv decode ABI. Each is mirrored by a
# ``constexpr`` guard in ``qwen4_exp_ple.hip``; the host check runs first, so a
# drift that widens only this side would make the device guards dead code.
QWEN4_EXP_PLE_INDEXED_MAX_ROWS = 64
QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS = 4096
QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS = 1048576
QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE = 1024
QWEN4_EXP_PLE_INDEXED_MAX_DILATION = 1024
QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS = 65536
_ARGS_3PTR_3DIM = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_ADD = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_void_p,
)
_ARGS_CONV = (
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
_ARGS_CONV_INDEXED = (
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
    ctypes.c_void_p,
)


def plan_qwen4_exp_ple_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="qwen4_exp_ple",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_qwen4_exp_ple(
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
        family="qwen4_exp_ple",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_ple_signed_sqrt_gate_f32(
    key_ptr: int,
    query_ptr: int,
    gate_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Compute the per-branch signed-square-root PLE gate."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_signed_sqrt_gate_f32",
        _ARGS_3PTR_3DIM,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(key_ptr, query_ptr, gate_ptr, rows, branches, hidden, stream))


def qwen4_exp_ple_repeat_gated_value_f32(
    value_ptr: int,
    gate_ptr: int,
    output_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Broadcast one value row across branches and apply each branch gate."""

    _check_shape(rows, branches, hidden)
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_repeat_gated_value_f32",
        _ARGS_3PTR_3DIM,
        ctypes.c_int,
    )
    _check_launch(runtime, fn(value_ptr, gate_ptr, output_ptr, rows, branches, hidden, stream))


def qwen4_exp_ple_add_delta_bf16_f32(
    residual_ptr: int,
    gated_value_ptr: int,
    conv_output_ptr: int,
    output_ptr: int,
    elements: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if elements <= 0:
        raise ValueError("elements must be positive")
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_add_delta_bf16_f32",
        _ARGS_ADD,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            residual_ptr,
            gated_value_ptr,
            conv_output_ptr,
            output_ptr,
            elements,
            stream,
        ),
    )


def qwen4_exp_ple_dilated_depthwise_conv_f32(
    input_ptr: int,
    weights_ptr: int,
    history_ptr: int,
    output_ptr: int,
    rows: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Compute causal PLE Conv and update caller-owned history in place."""

    if rows <= 0 or channels <= 0 or kernel_size <= 0 or dilation <= 0:
        raise ValueError("rows, channels, kernel_size, and dilation must be positive")
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_dilated_depthwise_conv_f32",
        _ARGS_CONV,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            input_ptr,
            weights_ptr,
            history_ptr,
            output_ptr,
            rows,
            channels,
            kernel_size,
            dilation,
            stream,
        ),
    )


def qwen4_exp_ple_indexed_state_indices(
    state_indices: object, *, rows: int, state_slots: int
) -> np.ndarray:
    """Validate per-row PLE history-slot indices for the indexed decode ABI.

    Returns a contiguous ``int64`` vector of ``rows`` entries in ``[0,
    state_slots)``. This is the authoritative admission check: the kernel sees
    only a device pointer and can reject a malformed batch but cannot report which
    row was wrong.

    **Slots must be distinct.** Each row is one independent request's decode step,
    and the decode body shifts and appends inside its slot as it runs, so two rows
    naming one slot would interleave their shifts and commit a history that is
    neither request's result. The kernel also rejects such a batch as a whole, but
    a caller that ignores this check has already lost the ability to tell which
    request was dropped.

    ``state_slots >= rows`` is a capacity bound and is not a uniqueness check;
    uniqueness is enforced separately below.
    """

    return indexed_state_indices(
        state_indices, rows=rows, state_slots=state_slots,
        max_rows=QWEN4_EXP_PLE_INDEXED_MAX_ROWS,
        max_state_slots=QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS,
        operation='PLE', state_name='PLE history',
    )


def qwen4_exp_ple_dilated_depthwise_conv_indexed_f32(
    input_ptr: int,
    weights_ptr: int,
    history_ptr: int,
    output_ptr: int,
    state_indices_ptr: int,
    *,
    rows: int,
    state_slots: int,
    channels: int,
    kernel_size: int,
    dilation: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Request-indexed PLE dilated Conv decode for independent requests.

    The multi-request form of :func:`qwen4_exp_ple_dilated_depthwise_conv_f32`.
    Row ``r`` of the packed ``input`` block is one token of an independent
    request; row ``r`` reads and writes exactly PLE history slot
    ``state_indices[r]`` of a ``[state_slots][(kernel_size - 1) * dilation]
    [channels]`` FP32 slab. Slots no row selects are not touched.

    The ABI is packed-only, like the single-row owner: input and output rows
    stride by ``channels``, the weight by ``channels * kernel_size``, and the
    history by ``history_rows * channels`` per slot. There are no stride arguments
    to bound; the shape bounds below are what keep the derived offsets in range,
    and ``(kernel_size - 1) * dilation`` is bounded directly rather than through
    its operands.

    A malformed batch (a slot outside the slab, or two rows naming one slot) is
    rejected on the device as a whole, leaving every slot and every output row
    unchanged. Use :func:`qwen4_exp_ple_indexed_state_indices` to reject it on the
    host instead, with a diagnosable error.
    """

    count = int(rows)
    slots = int(state_slots)
    if not 1 <= count <= QWEN4_EXP_PLE_INDEXED_MAX_ROWS:
        raise ValueError(
            "indexed PLE decode rows must be in "
            f"1..{QWEN4_EXP_PLE_INDEXED_MAX_ROWS}"
        )
    if slots <= 0:
        raise ValueError("state_slots must be positive")
    if slots > QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS:
        raise ValueError(
            "state_slots must be in "
            f"1..{QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS}"
        )
    if slots < count:
        raise ValueError(
            "state_slots must cover one slot per row (state slab too small)"
        )
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels > QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS:
        raise ValueError(
            f"channels must be in 1..{QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS}"
        )
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    if kernel_size > QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE:
        raise ValueError(
            "kernel_size must be in "
            f"1..{QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE}"
        )
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    if dilation > QWEN4_EXP_PLE_INDEXED_MAX_DILATION:
        raise ValueError(
            f"dilation must be in 1..{QWEN4_EXP_PLE_INDEXED_MAX_DILATION}"
        )
    history_rows = (kernel_size - 1) * dilation
    if history_rows > QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS:
        raise ValueError(
            "history rows must be in "
            f"0..{QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS} "
            "((kernel_size - 1) * dilation)"
        )
    if state_indices_ptr % 8:
        raise ValueError("state_indices must be 8-byte aligned for int64 loads")
    library = library or build_qwen4_exp_ple(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_ple_dilated_depthwise_conv_indexed_f32",
        _ARGS_CONV_INDEXED,
        ctypes.c_int,
    )
    _check_launch(
        runtime,
        fn(
            input_ptr,
            weights_ptr,
            history_ptr,
            output_ptr,
            state_indices_ptr,
            count,
            slots,
            channels,
            kernel_size,
            dilation,
            stream,
        ),
    )


def register_qwen4_exp_ple_kernels(*, replace: bool = True) -> None:
    registrations = {
        KernelKey(
            "hip_gfx1100",
            "ple_signed_sqrt_gate",
            "f32",
            "strict",
        ): qwen4_exp_ple_signed_sqrt_gate_f32,
        KernelKey(
            "hip_gfx1100",
            "ple_repeat_gated_value",
            "f32",
            "strict",
        ): qwen4_exp_ple_repeat_gated_value_f32,
        KernelKey(
            "hip_gfx1100",
            "ple_add_delta",
            "bf16_f32",
            "strict",
        ): qwen4_exp_ple_add_delta_bf16_f32,
        KernelKey(
            "hip_gfx1100",
            "ple_dilated_depthwise_conv",
            "f32",
            "strict",
        ): qwen4_exp_ple_dilated_depthwise_conv_f32,
        # Request-indexed decode for independent rows. The ``strict`` variant above
        # stays the single-request owner and the registered fallback.
        KernelKey(
            "hip_gfx1100",
            "ple_dilated_depthwise_conv",
            "f32",
            "qwen4exp_ple_indexed_decode",
        ): qwen4_exp_ple_dilated_depthwise_conv_indexed_f32,
    }
    for key, function in registrations.items():
        register(key, function, replace=replace)


def _check_shape(rows: int, branches: int, hidden: int) -> None:
    if rows <= 0 or branches <= 0 or hidden <= 0:
        raise ValueError("rows, branches, and hidden must be positive")


def _check_launch(runtime: HipRuntime, error: int) -> None:
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


register_qwen4_exp_ple_kernels()


__all__ = [
    "build_qwen4_exp_ple",
    "plan_qwen4_exp_ple_build",
    "qwen4_exp_ple_add_delta_bf16_f32",
    "qwen4_exp_ple_dilated_depthwise_conv_f32",
    "qwen4_exp_ple_dilated_depthwise_conv_indexed_f32",
    "qwen4_exp_ple_indexed_state_indices",
    "qwen4_exp_ple_repeat_gated_value_f32",
    "qwen4_exp_ple_signed_sqrt_gate_f32",
    "register_qwen4_exp_ple_kernels",
]
