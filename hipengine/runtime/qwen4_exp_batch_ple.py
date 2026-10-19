"""Independent-request PLE stage over per-request Conv-history slots.

``run_qwen4_exp_ple`` in ``runtime/qwen4_exp_runner.py:6931`` is the
single-request PLE oracle: one ``[history_rows, channels]`` Conv history, one
token stream. Its Conv stage reads and shifts *that* history, so serving N
resident requests through it needs N full history copies (or a per-request
step loop).

This module is the N-request form:

* :class:`Qwen4ExpBatchPleState` owns one PLE stage's
  ``[state_slots][history_rows][channels]`` F32 history slab plus the
  ``int64[rows]`` row-to-slot index vector. Slots are physical requests.
* :func:`run_qwen4_exp_batch_ple_stage` runs one token for every admitted row,
  one dispatch per stage, and hands the Conv stage the row-to-slot vector so
  each row shifts only its own request's history.

Stages, in order, one dispatch each (the oracle's chain):

1. ``ple_key`` GGUF projection ``[rows, hidden] -> [rows, branches*hidden]``.
2. ``ple_value`` GGUF projection ``[rows, hidden] -> [rows, hidden]``.
3. ``qwen4_exp_grouped_rmsnorm_f32`` over the key projection, in place.
4. ``qwen4_exp_grouped_rmsnorm_bf16_f32`` from the BF16 residual to the query.
5. ``qwen4_exp_ple_signed_sqrt_gate_f32`` (key . query per branch).
6. ``qwen4_exp_ple_repeat_gated_value_f32`` (value broadcast per branch).
7. ``qwen4_exp_grouped_rmsnorm_f32`` over the gated value.
8. The dilated depthwise Conv, ``qwen4exp_ple_indexed_decode``: row ``r`` reads
   and shifts history slot ``indices[r]`` only.
9. ``qwen4_exp_scaled_silu_f32`` in place.
10. ``qwen4_exp_ple_add_delta_bf16_f32`` (residual + gated value + Conv, BF16).

Stages 1-7, 9, and 10 are already row-independent owners whose grid is the row
count; only stage 8 changes owner, from the strict single-row Conv to the
committed request-indexed decode. That variant is resolved through the registry
from the projection weight's backend, and the strict single-row owner stays
registered as the fallback: nothing here branches on backend or quant.

Scope: one token per row (a decode step). A multi-token prefill chunk is one
request's consecutive positions, not N independent requests, and must not be
admitted as active rows. PLE hash orchestration is outside this unit: the caller
supplies the staged embedding rows. Nothing here is wired into the runner, and
the c1 path is unchanged.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_grouped_rmsnorm_bf16_f32,
    qwen4_exp_grouped_rmsnorm_f32,
    qwen4_exp_scaled_silu_f32,
)
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS,
    QWEN4_EXP_PLE_INDEXED_MAX_DILATION,
    QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS,
    QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE,
    QWEN4_EXP_PLE_INDEXED_MAX_ROWS,
    QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_ple_add_delta_bf16_f32,
    qwen4_exp_ple_indexed_state_indices,
    qwen4_exp_ple_repeat_gated_value_f32,
    qwen4_exp_ple_signed_sqrt_gate_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.runtime.gguf_weight import GGUFDeviceWeight
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpPLEScratch

# The committed request-indexed Conv decode; the ``strict`` variant stays the
# single-request owner and the registered fallback.
_INDEXED_CONV_LAYER = "ple_dilated_depthwise_conv"
_INDEXED_CONV_QUANT = "f32"
_INDEXED_CONV_VARIANT = "qwen4exp_ple_indexed_decode"
_REQUIRED_PROJECTIONS = ("ple_key", "ple_value")
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class Qwen4ExpBatchPleActiveRows:
    """One admitted row set: physical history slots and the uploaded indices."""

    rows: int
    slot_ids: tuple[int, ...]
    indices: Tensor
    indices_host: np.ndarray


@dataclass
class Qwen4ExpBatchPleState:
    """One PLE stage's per-request dilated-Conv history slab.

    ``history`` is ``[state_slots, (conv_kernel - 1) * dilation, channels]`` F32,
    the layout the indexed decode reads and the same per-request layout the c1
    state holds. A slot no active row selects is never touched.
    """

    history: DeviceBuffer
    indices: DeviceBuffer
    indices_host: np.ndarray
    state_slots: int
    branches: int
    hidden: int
    conv_kernel: int
    dilation: int
    runtime: HipRuntime
    active: Qwen4ExpBatchPleActiveRows | None = field(default=None)
    closed: bool = field(default=False)
    # Set before the first free of a close attempt. Uploads are rejected from that
    # point on, because a partial close can leave earlier buffers released while
    # later ones are still owned; ``close`` itself may retry the remainder.
    closing: bool = field(default=False)
    # Buffers still owned, in allocation order; ``close`` frees the last entry
    # first and drops it only after its free returns, so a retry cannot double-free.
    _pending_free: list[DeviceBuffer] = field(default_factory=list)

    # -- construction -----------------------------------------------------

    @classmethod
    def allocate(
        cls,
        *,
        state_slots: int,
        branches: int,
        hidden: int,
        conv_kernel: int,
        dilation: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpBatchPleState":
        """Allocate ``state_slots`` request-owned Conv-history slots.

        The slab starts zeroed, which is the history of a request that has not
        produced a token yet. Geometry the indexed decode cannot address is
        rejected before a runtime is resolved or a byte is allocated.
        """

        slots = _positive_int(state_slots, "state_slots")
        parsed_branches = _positive_int(branches, "branches")
        parsed_hidden = _positive_int(hidden, "hidden")
        parsed_kernel = _positive_int(conv_kernel, "conv_kernel")
        parsed_dilation = _positive_int(dilation, "dilation")
        channels = parsed_branches * parsed_hidden
        history_rows = (parsed_kernel - 1) * parsed_dilation
        _check_indexed_bounds(
            state_slots=slots,
            channels=channels,
            conv_kernel=parsed_kernel,
            dilation=parsed_dilation,
            history_rows=history_rows,
        )
        _check_slab_products(
            state_slots=slots, history_rows=history_rows, channels=channels
        )
        active_runtime = runtime or get_hip_runtime()

        indices_host = np.zeros(slots, dtype=np.int64)
        buffers: list[DeviceBuffer] = []
        try:
            history = malloc(
                slots * max(history_rows, 0) * channels * 4, runtime=active_runtime
            )
            buffers.append(history)
            indices = malloc(indices_host.nbytes, runtime=active_runtime)
            buffers.append(indices)
            copy_host_to_device(
                indices, host_array_ptr(indices_host), runtime=active_runtime
            )
            # A slot no request owns reads as zero history, so "never written" and
            # "fresh request" are the same, checkable value.
            if history.nbytes:
                active_runtime.memset(history.ptr, 0, history.nbytes)
        except BaseException as error:
            try:
                _free_pending(buffers, active_runtime)
            except Exception as cleanup_error:  # pragma: no cover - defensive
                error.add_note(
                    f"cleanup after the failed allocation also failed: {cleanup_error!r}"
                )
            raise
        return cls(
            history=history,
            indices=indices,
            indices_host=indices_host,
            state_slots=slots,
            branches=parsed_branches,
            hidden=parsed_hidden,
            conv_kernel=parsed_kernel,
            dilation=parsed_dilation,
            runtime=active_runtime,
            _pending_free=list(buffers),
        )

    # -- geometry ---------------------------------------------------------

    @property
    def channels(self) -> int:
        return self.branches * self.hidden

    @property
    def history_rows(self) -> int:
        return (self.conv_kernel - 1) * self.dilation

    @property
    def history_bytes_per_slot(self) -> int:
        if self.state_slots == 0:  # pragma: no cover - allocate rejects zero slots
            return 0
        return self.history.nbytes // self.state_slots

    @property
    def active_rows(self) -> int:
        return 0 if self.active is None else self.active.rows

    @property
    def indices_ptr(self) -> int:
        return self._require_active().indices.ptr

    def history_slot_ptr(self, slot: int) -> int:
        """Device address of one request's ``[history_rows, channels]`` block."""

        self._require_open()
        parsed = _int_index(slot, "slot")
        if not 0 <= parsed < self.state_slots:
            raise ValueError(f"slot {parsed} is outside [0, {self.state_slots})")
        return self.history.ptr + parsed * self.history_bytes_per_slot

    # -- active rows ------------------------------------------------------

    def prepare(self, slot_ids: Sequence[int]) -> Qwen4ExpBatchPleActiveRows:
        """Admit ``slot_ids[r]`` as the history slot serving row ``r``.

        The slots must be distinct and inside the slab; the row order is the
        caller's, so a permuted or shrinking active set is supported and never
        remaps a slot. Validation is the primitive's own admission helper and
        runs in full before the index vector is uploaded, so a rejected call
        leaves the published rows and the device vector exactly as they were.
        """

        self._require_open()
        slots = _index_sequence(slot_ids, "slot_ids")
        if not slots:
            raise ValueError("prepare requires at least one active row")
        if len(slots) > self.state_slots:
            raise ValueError(
                f"prepare admits at most {self.state_slots} active rows, got {len(slots)}"
            )
        indices = qwen4_exp_ple_indexed_state_indices(
            slots, rows=len(slots), state_slots=self.state_slots
        )
        try:
            copy_host_to_device(
                self.indices,
                host_array_ptr(indices),
                indices.nbytes,
                runtime=self.runtime,
            )
        except BaseException:
            self.active = None
            raise
        self.indices_host[: indices.size] = indices
        active = Qwen4ExpBatchPleActiveRows(
            rows=len(slots),
            slot_ids=tuple(slots),
            indices=Tensor.from_handle(
                self.indices.ptr, (len(slots),), DType.INT64, Device("hip", 0)
            ),
            indices_host=indices,
        )
        self.active = active
        return active

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        # Poison uploads before the first free: a partial close leaves some
        # buffers released and some owned, and no upload may target either.
        self.closing = True
        self.active = None
        pending = self._pending_free
        if not pending:
            # Ownership was not handed over by ``allocate`` (direct construction);
            # take it now so closing cannot silently leak.
            pending.extend((self.history, self.indices))
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpBatchPleState":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp batch PLE state is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp batch PLE state is closing; its buffers may already be released"
            )

    def _require_active(self) -> Qwen4ExpBatchPleActiveRows:
        self._require_open()
        if self.active is None:
            raise RuntimeError(
                "Qwen4Exp batch PLE state has no active rows; call prepare first"
            )
        return self.active


def batch_ple_indexed_conv_key(backend: str) -> KernelKey:
    """The registry key the batch PLE stage resolves for its indexed Conv."""

    return KernelKey(
        str(backend),
        _INDEXED_CONV_LAYER,
        _INDEXED_CONV_QUANT,
        _INDEXED_CONV_VARIANT,
    )


def batch_ple_indexed_conv_fallback_key(backend: str) -> KernelKey:
    """The single-request strict Conv owner that remains the fallback."""

    return KernelKey(
        str(backend),
        _INDEXED_CONV_LAYER,
        _INDEXED_CONV_QUANT,
        "strict",
    )


def _resolve_indexed_conv(
    weights: Mapping[str, GGUFDeviceWeight],
) -> Callable[..., None]:
    """Resolve the request-indexed Conv decode for the weight's backend.

    Exact-key registration is checked first: ``resolve`` applies generic
    fallbacks (variant, quant, then the cpu_reference catch-all), and a silent
    fallback to the single-row owner would hand it a slot vector it does not
    accept. The backend comes from the projection weight, so this is the registry
    mechanism, not a backend branch.
    """

    key = batch_ple_indexed_conv_key(str(weights["ple_key"].backend))
    backend = key.backend
    # Refreshing the backend package re-walks every registration (milliseconds
    # per decode step), so do it only when the key is actually missing.
    if not is_registered(key):
        load_backend_kernel_package(backend)
    if not is_registered(key):
        raise RuntimeError(
            "the request-indexed PLE Conv decode "
            f"({_INDEXED_CONV_LAYER}/{_INDEXED_CONV_QUANT}/{_INDEXED_CONV_VARIANT}) "
            f"is not registered for backend {backend!r}; the single-request stage "
            "remains the fallback"
        )
    return resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )


@dataclass(frozen=True)
class Qwen4ExpBatchPlePlan:
    """Validated PLE geometry and indexed history update for one row set."""

    runtime: HipRuntime
    rows: int
    channels: int
    indices_ptr: int
    conv_stage: Callable[..., None]
    branches: int
    hidden: int
    eps: float


def validate_qwen4_exp_batch_ple_stage(
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchPleState,
    scratch: Qwen4ExpPLEScratch,
    residual_ptr: int,
    embedding_ptr: int,
    norm_key_ptr: int,
    norm_query_ptr: int,
    norm_conv_ptr: int,
    conv_weight_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    conv_kernel: int,
    dilation: int,
    eps: float = 1e-6,
    runtime: HipRuntime | None = None,
) -> Qwen4ExpBatchPlePlan:
    """Check the complete PLE call before dispatching any stage."""

    active_runtime, count, channels, conv_stage = _validate(
        weights, state=state, scratch=scratch, residual_ptr=residual_ptr,
        embedding_ptr=embedding_ptr, norm_key_ptr=norm_key_ptr,
        norm_query_ptr=norm_query_ptr, norm_conv_ptr=norm_conv_ptr,
        conv_weight_ptr=conv_weight_ptr, rows=rows, branches=branches,
        hidden=hidden, conv_kernel=conv_kernel, dilation=dilation, eps=eps,
        runtime=runtime,
    )
    return Qwen4ExpBatchPlePlan(
        active_runtime, count, channels, state.indices_ptr, conv_stage,
        int(branches), int(hidden), float(eps),
    )


def run_qwen4_exp_batch_ple_stage(
    residual_ptr: int,
    embedding_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchPleState,
    scratch: Qwen4ExpPLEScratch,
    norm_key_ptr: int,
    norm_query_ptr: int,
    norm_conv_ptr: int,
    conv_weight_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    conv_kernel: int,
    dilation: int,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Run one PLE token for every admitted row, one dispatch per stage.

    ``state.prepare(slot_ids)`` must already have admitted exactly ``rows`` rows;
    this call consumes that row-to-slot mapping and never re-uploads or reorders
    it. Row ``r`` reads and shifts only history slot ``slot_ids[r]``. Returns the
    ``[rows, branches, hidden]`` BF16 output buffer owned by ``scratch``.
    """

    plan = validate_qwen4_exp_batch_ple_stage(
        weights, state=state, scratch=scratch, residual_ptr=residual_ptr,
        embedding_ptr=embedding_ptr, norm_key_ptr=norm_key_ptr,
        norm_query_ptr=norm_query_ptr, norm_conv_ptr=norm_conv_ptr,
        conv_weight_ptr=conv_weight_ptr, rows=rows, branches=branches,
        hidden=hidden, conv_kernel=conv_kernel, dilation=dilation, eps=eps,
        runtime=runtime,
    )
    run_qwen4_exp_batch_ple_prepare(
        plan, residual_ptr, embedding_ptr, weights, scratch=scratch,
        norm_key_ptr=norm_key_ptr, norm_query_ptr=norm_query_ptr,
        norm_conv_ptr=norm_conv_ptr, stream=stream,
    )
    run_qwen4_exp_batch_ple_convolution(
        plan, state=state, scratch=scratch, conv_weight_ptr=conv_weight_ptr,
        stream=stream,
    )
    return run_qwen4_exp_batch_ple_finish(
        plan, residual_ptr, scratch=scratch, stream=stream,
    )


def run_qwen4_exp_batch_ple_prepare(
    plan: Qwen4ExpBatchPlePlan,
    residual_ptr: int,
    embedding_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    scratch: Qwen4ExpPLEScratch,
    norm_key_ptr: int,
    norm_query_ptr: int,
    norm_conv_ptr: int,
    stream: int = 0,
) -> None:
    """Project, normalize and gate every row without touching request history."""

    active_runtime, count, channels = plan.runtime, plan.rows, plan.channels
    branches, hidden, eps = plan.branches, plan.hidden, plan.eps
    launch_gguf_linear(
        weights["ple_key"],
        embedding_ptr,
        scratch.key.ptr,
        count,
        hidden,
        channels,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights["ple_value"],
        embedding_ptr,
        scratch.value.ptr,
        count,
        hidden,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_f32(
        scratch.key.ptr,
        norm_key_ptr,
        scratch.key.ptr,
        count,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_bf16_f32(
        residual_ptr,
        norm_query_ptr,
        scratch.query.ptr,
        count,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_signed_sqrt_gate_f32(
        scratch.key.ptr,
        scratch.query.ptr,
        scratch.gate.ptr,
        count,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_repeat_gated_value_f32(
        scratch.value.ptr,
        scratch.gate.ptr,
        scratch.gated_value.ptr,
        count,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_grouped_rmsnorm_f32(
        scratch.gated_value.ptr,
        norm_conv_ptr,
        scratch.normalized.ptr,
        count,
        branches,
        hidden,
        eps,
        stream=stream,
        runtime=active_runtime,
    )


def run_qwen4_exp_batch_ple_convolution(
    plan: Qwen4ExpBatchPlePlan,
    *,
    state: Qwen4ExpBatchPleState,
    scratch: Qwen4ExpPLEScratch,
    conv_weight_ptr: int,
    stream: int = 0,
) -> None:
    """Advance the validated row set's request-local convolution histories."""

    plan.conv_stage(
        scratch.normalized.ptr,
        conv_weight_ptr,
        state.history.ptr,
        scratch.conv.ptr,
        plan.indices_ptr,
        rows=plan.rows,
        state_slots=state.state_slots,
        channels=plan.channels,
        kernel_size=state.conv_kernel,
        dilation=state.dilation,
        stream=stream,
        runtime=plan.runtime,
    )


def run_qwen4_exp_batch_ple_finish(
    plan: Qwen4ExpBatchPlePlan,
    residual_ptr: int,
    *,
    scratch: Qwen4ExpPLEScratch,
    stream: int = 0,
) -> DeviceBuffer:
    """Apply the convolution result and residual delta without changing history."""

    active_runtime, count, channels = plan.runtime, plan.rows, plan.channels
    qwen4_exp_scaled_silu_f32(
        scratch.conv.ptr,
        scratch.conv.ptr,
        count * channels,
        1.0,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_ple_add_delta_bf16_f32(
        residual_ptr,
        scratch.gated_value.ptr,
        scratch.conv.ptr,
        scratch.output.ptr,
        count * channels,
        stream=stream,
        runtime=active_runtime,
    )
    return scratch.output


def _validate(
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchPleState,
    scratch: Qwen4ExpPLEScratch,
    residual_ptr: int,
    embedding_ptr: int,
    norm_key_ptr: int,
    norm_query_ptr: int,
    norm_conv_ptr: int,
    conv_weight_ptr: int,
    rows: int,
    branches: int,
    hidden: int,
    conv_kernel: int,
    dilation: int,
    eps: float,
    runtime: HipRuntime | None,
) -> tuple[HipRuntime, int, int, Callable[..., None]]:
    """Admit one batch stage call before any dispatch touches the history."""

    if scratch.closed or state.closed:
        raise RuntimeError("Qwen4Exp PLE scratch/state is closed")
    if state.closing:
        raise RuntimeError(
            "Qwen4Exp PLE state is closing; its buffers may already be released"
        )
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime or active_runtime is not state.runtime:
        raise ValueError("runtime must match PLE scratch and state owners")

    count = _positive_int(rows, "rows")
    parsed_branches = _positive_int(branches, "branches")
    parsed_hidden = _positive_int(hidden, "hidden")
    parsed_kernel = _positive_int(conv_kernel, "conv_kernel")
    parsed_dilation = _positive_int(dilation, "dilation")
    if not _is_finite_positive(eps, "eps"):
        raise ValueError("eps must be a finite positive number")
    if (parsed_branches, parsed_hidden, parsed_kernel, parsed_dilation) != (
        state.branches,
        state.hidden,
        state.conv_kernel,
        state.dilation,
    ):
        raise ValueError(
            "PLE geometry must match the state owner "
            f"({state.branches} branches x {state.hidden} hidden, kernel "
            f"{state.conv_kernel}, dilation {state.dilation})"
        )
    channels = parsed_branches * parsed_hidden
    _check_indexed_bounds(
        state_slots=state.state_slots,
        channels=channels,
        conv_kernel=parsed_kernel,
        dilation=parsed_dilation,
        history_rows=state.history_rows,
    )
    if count > QWEN4_EXP_PLE_INDEXED_MAX_ROWS:
        raise ValueError(
            f"rows {count} exceeds the indexed PLE decode row bound "
            f"{QWEN4_EXP_PLE_INDEXED_MAX_ROWS}"
        )

    missing = sorted(set(_REQUIRED_PROJECTIONS) - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp PLE weights: " + ", ".join(missing))
    for name, pointer in (
        ("residual", residual_ptr),
        ("embedding", embedding_ptr),
        ("norm_key", norm_key_ptr),
        ("norm_query", norm_query_ptr),
        ("norm_conv", norm_conv_ptr),
        ("conv_weight", conv_weight_ptr),
    ):
        _positive_int(pointer, f"{name}_ptr")

    active = state.active
    if active is None:
        raise RuntimeError(
            "Qwen4Exp batch PLE state has no active rows; call prepare first"
        )
    if count != active.rows:
        raise ValueError(
            f"stage rows {count} does not match the {active.rows} rows prepared on "
            "the PLE state"
        )
    # Re-admit the published row set: this rejects a caller that rewrote the
    # vector after ``prepare`` and keeps one authoritative rule for the ABI.
    qwen4_exp_ple_indexed_state_indices(
        active.indices_host, rows=count, state_slots=state.state_slots
    )
    if active.indices.ptr % 8:
        raise ValueError("state indices must be 8-byte aligned for int64 loads")

    _require_scratch(scratch, count, parsed_branches, parsed_hidden)
    _require_projection_geometry(
        weights,
        {"ple_key": (channels, parsed_hidden), "ple_value": (parsed_hidden, parsed_hidden)},
    )
    return active_runtime, count, channels, _resolve_indexed_conv(weights)


def _require_scratch(
    scratch: Qwen4ExpPLEScratch,
    rows: int,
    branches: int,
    hidden: int,
) -> None:
    """Reject a scratch that cannot hold ``rows`` of this PLE geometry."""

    channels = branches * hidden
    required = {
        "key": (rows * channels * 4, scratch.key),
        "value": (rows * hidden * 4, scratch.value),
        "query": (rows * channels * 4, scratch.query),
        "gate": (rows * branches * 4, scratch.gate),
        "gated_value": (rows * channels * 4, scratch.gated_value),
        "normalized": (rows * channels * 4, scratch.normalized),
        "conv": (rows * channels * 4, scratch.conv),
        # The residual-delta output is BF16, like the residual it updates.
        "output": (rows * channels * 2, scratch.output),
    }
    for name, (needed, buffer) in required.items():
        if buffer.nbytes < needed:
            raise ValueError(
                f"Qwen4Exp PLE scratch {name} holds {buffer.nbytes} bytes, needs "
                f"{needed} for {rows} rows"
            )


def _require_projection_geometry(
    weights: Mapping[str, GGUFDeviceWeight],
    expected: dict[str, tuple[int, int]],
) -> None:
    """Reject a projection whose stored shape cannot produce the staged layout.

    Weight objects that do not expose a source tensor shape keep the dispatch's
    own check.
    """

    for slot, (out_features, in_features) in expected.items():
        spec = getattr(weights[slot], "spec", None)
        shape = getattr(getattr(spec, "source", None), "shape", None)
        if shape is None:
            continue
        parsed = tuple(int(dimension) for dimension in shape)
        if parsed != (out_features, in_features):
            raise ValueError(
                f"{slot} weight shape {parsed} does not match the expected "
                f"({out_features}, {in_features}) projection geometry"
            )


def _check_indexed_bounds(
    *,
    state_slots: int,
    channels: int,
    conv_kernel: int,
    dilation: int,
    history_rows: int,
) -> None:
    """Mirror the indexed decode's own admission bounds before any allocation."""

    if state_slots > QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS:
        raise ValueError(
            f"state_slots {state_slots} exceeds the indexed PLE decode slab width "
            f"{QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS}"
        )
    if channels > QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS:
        raise ValueError(
            f"channels {channels} exceeds the indexed PLE decode channel bound "
            f"{QWEN4_EXP_PLE_INDEXED_MAX_CHANNELS}"
        )
    if conv_kernel > QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE:
        raise ValueError(
            f"conv_kernel {conv_kernel} exceeds the indexed PLE decode kernel bound "
            f"{QWEN4_EXP_PLE_INDEXED_MAX_KERNEL_SIZE}"
        )
    if dilation > QWEN4_EXP_PLE_INDEXED_MAX_DILATION:
        raise ValueError(
            f"dilation {dilation} exceeds the indexed PLE decode dilation bound "
            f"{QWEN4_EXP_PLE_INDEXED_MAX_DILATION}"
        )
    if history_rows > QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS:
        raise ValueError(
            f"history rows {history_rows} exceeds the indexed PLE decode history "
            f"bound {QWEN4_EXP_PLE_INDEXED_MAX_HISTORY_ROWS}"
        )


def _check_slab_products(*, state_slots: int, history_rows: int, channels: int) -> None:
    """Reject slab shapes whose products the kernels' integer types cannot hold."""

    elements = state_slots * history_rows * channels
    if elements > _INT64_MAX // 4:
        raise ValueError(f"{elements} PLE history elements exceed the int64 byte bound")


def _free_pending(pending: list[DeviceBuffer], runtime: HipRuntime) -> None:
    """Free ``pending`` newest-first (last entry first), popping after each free.

    A failed ``free`` leaves that buffer and everything older in ``pending``, so
    retrying releases exactly the remainder instead of double-freeing the
    buffers that are already gone.
    """

    while pending:
        free(pending[-1], runtime=runtime)
        pending.pop()


def _int_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None


def _positive_int(value: object, name: str) -> int:
    parsed = _int_index(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _is_finite_positive(value: object, name: str) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        parsed = float(value)
    except OverflowError:
        return False
    return math.isfinite(parsed) and parsed > 0.0


def _index_sequence(values: object, name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        items = list(values)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integers") from None
    return [_int_index(item, name) for item in items]


def batch_ple_call_table() -> dict[str, Callable[..., Any]]:
    """The exact direct callables this module uses for one batch PLE stage.

    The indexed Conv is resolved through the registry and is bound separately
    with its registry key and strict fallback. Projections are bound as
    row-dependent registry dispatch records.
    """

    return {
        "ple_stage": run_qwen4_exp_batch_ple_stage,
        "ple_prepare": run_qwen4_exp_batch_ple_prepare,
        "ple_convolution": run_qwen4_exp_batch_ple_convolution,
        "ple_finish": run_qwen4_exp_batch_ple_finish,
        "ple_grouped_rmsnorm": qwen4_exp_grouped_rmsnorm_f32,
        "ple_grouped_rmsnorm_bf16": qwen4_exp_grouped_rmsnorm_bf16_f32,
        "ple_signed_sqrt_gate": qwen4_exp_ple_signed_sqrt_gate_f32,
        "ple_repeat_gated_value": qwen4_exp_ple_repeat_gated_value_f32,
        "ple_scaled_silu": qwen4_exp_scaled_silu_f32,
        "ple_add_delta": qwen4_exp_ple_add_delta_bf16_f32,
    }
