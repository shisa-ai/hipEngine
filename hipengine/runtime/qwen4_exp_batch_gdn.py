"""Independent-request GDN token mixing over per-request recurrent/Conv slabs.

``run_qwen4_exp_gdn_token_mixer`` in ``runtime/qwen4_exp_runner.py:7080`` is the
strict single-row oracle: one request, one ``[num_v_heads, head_dim, head_dim]``
recurrent row, one ``[channels, kernel]`` Conv row, and one int64 position. This
module is the N-request form of the same decode arithmetic:

* :class:`Qwen4ExpBatchGdnState` owns one layer's ``[state_slots, ...]``
  recurrent and Conv slabs plus the ``int64[rows]`` row-to-slot index vector.
  Slots are physical *requests*; nothing here treats a slot as a layer ordinal.
* :func:`run_qwen4_exp_batch_gdn_token_mixer` runs one decode step for the
  active rows, one dispatch per stage instead of a per-request step loop.

Stages, in order, one dispatch each (the oracle's decode path):

1. ``attn_qkv`` and ``attn_gate`` GGUF projections over the N-row hidden.
2. ``ssm_alpha`` and ``ssm_beta`` GGUF projections to the per-row scalars.
3. ``qwen4_exp_linear_attn_conv_decode_indexed_f32``: row ``r`` shifts and
   appends inside Conv state slot ``indices[r]``.
4. ``qwen4_exp_gdn_decode_rowsplit_indexed_f32``: recurrence, output norm, and
   gate for row ``r`` over recurrent state slot ``indices[r]``.
5. ``ssm_out`` GGUF projection back to the N-row hidden.

The two indexed primitives are the committed request-indexed owners
(``kernels/hip_gfx1100/linear_attn/conv.py`` and ``.../qwen4_exp_gdn.py``). Both
admit a malformed batch as a whole on the device and leave every slot unchanged,
and their host admission helpers are the authoritative slot checks; this module
calls them rather than restating the rules.

Arithmetic scope: the indexed recurrence owner is the row-split form, which
computes the same algebra as the oracle's column-major decode with a different
FP32 summation order (and folds the two dot products into one state pass), and
the projection GEMM picks a row-count-dependent variant. N-row results therefore
match the oracle to FP32 round-off, not bit-for-bit; the tests measure the
deviation instead of assuming it.

Not covered here: prefill chunks (the recurrence is a per-token recurrence, and a
multi-token chunk needs the prefill owners plus per-token state rows), PLE, the
sparse QSA index path, and any runner wiring. Nothing in this module dispatches
by backend or quant: the projections go through ``launch_gguf_linear``'s registry
resolution and the rest are the two public indexed wrappers.
"""

from __future__ import annotations

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
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS,
    QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE,
    QWEN4_EXP_CONV_INDEXED_MAX_ROWS,
    QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_linear_attn_conv_decode_indexed_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    QWEN4_EXP_GDN_INDEXED_MAX_ROWS,
    QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS,
    qwen4_exp_gdn_decode_rowsplit_indexed_f32,
    qwen4_exp_gdn_indexed_state_indices,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.runtime.gguf_weight import GGUFDeviceWeight
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGDNScratch

# The indexed recurrence owner is compiled for this head geometry only; the
# conv owner has no such bound but is always paired with it here.
_INDEXED_HEAD_DIM = 128
# ``qwen4_exp_gdn_decode_rowsplit_indexed_f32`` loads Q/K as float4 and takes the
# Conv row stride in floats, so both operands need this alignment.
_FLOAT4_BYTES = 16
_REQUIRED_PROJECTIONS = ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out")
_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class Qwen4ExpBatchGdnActiveRows:
    """One admitted row set: physical slots and the uploaded index vector."""

    rows: int
    slot_ids: tuple[int, ...]
    indices: Tensor
    indices_host: np.ndarray


@dataclass
class Qwen4ExpBatchGdnState:
    """One GDN layer's per-request recurrent/Conv state slabs.

    ``recurrent_state`` is ``[state_slots, num_v_heads, head_dim, head_dim]`` F32
    and ``conv_state`` is ``[state_slots, qkv_width, conv_kernel]`` F32, both the
    layout the indexed primitives read and the same per-row layout the c1 state
    uses. Slots are independent requests: a slot no active row selects is never
    touched by a mixer step.
    """

    recurrent_state: DeviceBuffer
    conv_state: DeviceBuffer
    indices: DeviceBuffer
    indices_host: np.ndarray
    state_slots: int
    num_k_heads: int
    num_v_heads: int
    head_dim: int
    conv_kernel: int
    runtime: HipRuntime
    active: Qwen4ExpBatchGdnActiveRows | None = field(default=None)
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
        num_k_heads: int,
        num_v_heads: int,
        head_dim: int,
        conv_kernel: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpBatchGdnState":
        """Allocate ``state_slots`` request-owned recurrent and Conv slots.

        Both slabs start zeroed, which is the recurrent/Conv state of a request
        that has not produced a token yet. Geometry the primitives or the host
        cannot address is rejected before a runtime is resolved or a byte is
        allocated.
        """

        slots = _positive_int(state_slots, "state_slots")
        k_heads = _positive_int(num_k_heads, "num_k_heads")
        v_heads = _positive_int(num_v_heads, "num_v_heads")
        parsed_head_dim = _positive_int(head_dim, "head_dim")
        parsed_kernel = _positive_int(conv_kernel, "conv_kernel")
        if v_heads % k_heads:
            raise ValueError("num_v_heads must be divisible by num_k_heads")
        if slots > QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS:
            raise ValueError(
                f"state_slots {slots} exceeds the indexed decode slab width "
                f"{QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS}"
            )
        qkv_width = 2 * k_heads * parsed_head_dim + v_heads * parsed_head_dim
        if qkv_width > QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS:
            raise ValueError(
                f"qkv_width {qkv_width} exceeds the indexed conv channel bound "
                f"{QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS}"
            )
        if parsed_kernel > QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE:
            raise ValueError(
                f"conv_kernel {parsed_kernel} exceeds the indexed conv kernel bound "
                f"{QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE}"
            )
        _check_slab_products(
            state_slots=slots,
            qkv_width=qkv_width,
            conv_kernel=parsed_kernel,
            num_v_heads=v_heads,
            head_dim=parsed_head_dim,
        )
        active_runtime = runtime or get_hip_runtime()

        indices_host = np.zeros(slots, dtype=np.int64)
        buffers: list[DeviceBuffer] = []
        try:
            recurrent_state = malloc(
                slots * v_heads * parsed_head_dim * parsed_head_dim * 4,
                runtime=active_runtime,
            )
            buffers.append(recurrent_state)
            conv_state = malloc(
                slots * qkv_width * parsed_kernel * 4, runtime=active_runtime
            )
            buffers.append(conv_state)
            indices = malloc(indices_host.nbytes, runtime=active_runtime)
            buffers.append(indices)
            copy_host_to_device(
                indices, host_array_ptr(indices_host), runtime=active_runtime
            )
            # A slot that no request owns reads as zero state, so "never written"
            # and "fresh request" are the same, checkable value.
            active_runtime.memset(recurrent_state.ptr, 0, recurrent_state.nbytes)
            active_runtime.memset(conv_state.ptr, 0, conv_state.nbytes)
        except BaseException as error:
            try:
                _free_pending(buffers, active_runtime)
            except Exception as cleanup_error:  # pragma: no cover - defensive
                error.add_note(
                    f"cleanup after the failed allocation also failed: {cleanup_error!r}"
                )
            raise
        return cls(
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            indices=indices,
            indices_host=indices_host,
            state_slots=slots,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_dim=parsed_head_dim,
            conv_kernel=parsed_kernel,
            runtime=active_runtime,
            _pending_free=list(buffers),
        )

    # -- geometry ---------------------------------------------------------

    @property
    def qkv_width(self) -> int:
        return 2 * self.num_k_heads * self.head_dim + self.num_v_heads * self.head_dim

    @property
    def core_width(self) -> int:
        return self.num_v_heads * self.head_dim

    @property
    def recurrent_bytes_per_slot(self) -> int:
        return self.recurrent_state.nbytes // self.state_slots

    @property
    def conv_bytes_per_slot(self) -> int:
        return self.conv_state.nbytes // self.state_slots

    @property
    def active_rows(self) -> int:
        return 0 if self.active is None else self.active.rows

    @property
    def indices_ptr(self) -> int:
        return self._require_active().indices.ptr

    def recurrent_slot_ptr(self, slot: int) -> int:
        """Device address of one recurrent slot's ``[V, D, D]`` block."""

        parsed = self._require_slot(slot)
        return self.recurrent_state.ptr + parsed * self.recurrent_bytes_per_slot

    def conv_slot_ptr(self, slot: int) -> int:
        """Device address of one Conv slot's ``[qkv_width, kernel]`` block."""

        parsed = self._require_slot(slot)
        return self.conv_state.ptr + parsed * self.conv_bytes_per_slot

    # -- active rows ------------------------------------------------------

    def prepare(self, slot_ids: Sequence[int]) -> Qwen4ExpBatchGdnActiveRows:
        """Admit ``slot_ids[r]`` as the physical slot serving row ``r``.

        The slots must be distinct and inside the slab; the order is the
        caller's, so a permuted or shrinking active set is supported and never
        remaps a slot. Validation is the primitives' own admission helper, and
        it runs in full before the index vector is uploaded, so a rejected call
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
        indices = qwen4_exp_gdn_indexed_state_indices(
            slots, rows=len(slots), state_slots=self.state_slots
        )
        # ``indices_host`` mirrors the device vector, so an unchanged row set
        # (every decode step between admissions) needs no upload.
        if not np.array_equal(self.indices_host[: indices.size], indices):
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
        device = Device("hip", 0)
        active = Qwen4ExpBatchGdnActiveRows(
            rows=len(slots),
            slot_ids=tuple(slots),
            indices=Tensor.from_handle(
                self.indices.ptr, (len(slots),), DType.INT64, device
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
            pending.extend((self.recurrent_state, self.conv_state, self.indices))
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpBatchGdnState":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp batch GDN state is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp batch GDN state is closing; its buffers may already be released"
            )

    def _require_active(self) -> Qwen4ExpBatchGdnActiveRows:
        self._require_open()
        if self.active is None:
            raise RuntimeError(
                "Qwen4Exp batch GDN state has no active rows; call prepare first"
            )
        return self.active

    def _require_slot(self, slot: object) -> int:
        self._require_open()
        parsed = _int_index(slot, "slot")
        if not 0 <= parsed < self.state_slots:
            raise ValueError(f"slot {parsed} is outside [0, {self.state_slots})")
        return parsed


@dataclass(frozen=True)
class Qwen4ExpBatchGdnMixerPlan:
    """One admitted batch GDN mixer call, split into its three stages.

    ``validate_qwen4_exp_batch_gdn_mixer`` builds it once per layer step. The
    input and output projections depend only on ``rows`` and fixed scratch
    pointers, so the batch runner may graph-capture them; the indexed conv and
    recurrence update request state in place and always run eagerly.
    """

    runtime: HipRuntime
    rows: int
    qkv_width: int
    core_width: int
    indices_ptr: int


def validate_qwen4_exp_batch_gdn_mixer(
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchGdnState,
    scratch: Qwen4ExpGDNScratch,
    rows: int,
    hidden: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_kernel: int,
    conv_weight_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    runtime: HipRuntime | None = None,
) -> Qwen4ExpBatchGdnMixerPlan:
    """Admit one batch mixer call before any dispatch touches the state."""

    active_runtime, count, qkv_width, core_width, indices_ptr = _validate(
        weights,
        state=state,
        scratch=scratch,
        rows=rows,
        hidden=hidden,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel=conv_kernel,
        conv_weight_ptr=conv_weight_ptr,
        dt_bias_ptr=dt_bias_ptr,
        a_ptr=a_ptr,
        norm_weight_ptr=norm_weight_ptr,
        runtime=runtime,
    )
    return Qwen4ExpBatchGdnMixerPlan(
        active_runtime, count, qkv_width, core_width, indices_ptr
    )


def run_qwen4_exp_batch_gdn_input_projections(
    plan: Qwen4ExpBatchGdnMixerPlan,
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchGdnState,
    scratch: Qwen4ExpGDNScratch,
    hidden: int,
    stream: int = 0,
) -> None:
    """Project ``qkv``/``gate``/``alpha``/``beta`` for every row (stateless)."""

    for slot, output, out_features in (
        ("attn_qkv", scratch.qkv, plan.qkv_width),
        ("attn_gate", scratch.gate, plan.core_width),
        ("ssm_alpha", scratch.alpha, state.num_v_heads),
        ("ssm_beta", scratch.beta, state.num_v_heads),
    ):
        launch_gguf_linear(
            weights[slot],
            mixed_ptr,
            output.ptr,
            plan.rows,
            hidden,
            out_features,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=plan.runtime,
        )


def run_qwen4_exp_batch_gdn_recurrence(
    plan: Qwen4ExpBatchGdnMixerPlan,
    *,
    state: Qwen4ExpBatchGdnState,
    scratch: Qwen4ExpGDNScratch,
    conv_weight_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    stream: int = 0,
) -> None:
    """Advance each row's conv and recurrent state slot in place."""

    qwen4_exp_linear_attn_conv_decode_indexed_f32(
        scratch.qkv.ptr,
        state.conv_state.ptr,
        conv_weight_ptr,
        scratch.conv.ptr,
        plan.indices_ptr,
        rows=plan.rows,
        state_slots=state.state_slots,
        channels=plan.qkv_width,
        kernel_size=state.conv_kernel,
        stream=stream,
        runtime=plan.runtime,
    )
    qwen4_exp_gdn_decode_rowsplit_indexed_f32(
        scratch.conv.ptr,
        scratch.gate.ptr,
        scratch.alpha.ptr,
        scratch.beta.ptr,
        dt_bias_ptr,
        a_ptr,
        norm_weight_ptr,
        state.recurrent_state.ptr,
        scratch.core.ptr,
        plan.indices_ptr,
        rows=plan.rows,
        state_slots=state.state_slots,
        num_k_heads=state.num_k_heads,
        num_v_heads=state.num_v_heads,
        head_k_dim=state.head_dim,
        head_v_dim=state.head_dim,
        stream=stream,
        runtime=plan.runtime,
    )


def run_qwen4_exp_batch_gdn_output_projection(
    plan: Qwen4ExpBatchGdnMixerPlan,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    scratch: Qwen4ExpGDNScratch,
    hidden: int,
    stream: int = 0,
) -> DeviceBuffer:
    """Project the gated recurrence output back to ``hidden`` (stateless)."""

    launch_gguf_linear(
        weights["ssm_out"],
        scratch.core.ptr,
        scratch.output.ptr,
        plan.rows,
        plan.core_width,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=plan.runtime,
    )
    return scratch.output


def run_qwen4_exp_batch_gdn_token_mixer(
    mixed_ptr: int,
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchGdnState,
    scratch: Qwen4ExpGDNScratch,
    rows: int,
    hidden: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_kernel: int,
    conv_weight_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Run one N-row GDN decode step for independent requests.

    ``state.prepare(slot_ids)`` must already have admitted exactly ``rows`` rows;
    this call consumes that row-to-slot mapping and never re-uploads or reorders
    it. Row ``r`` reads and writes only state slot ``slot_ids[r]``, so a slot no
    row selects keeps its bytes. Returns the ``[rows, hidden]`` output buffer
    owned by ``scratch``.
    """

    plan = validate_qwen4_exp_batch_gdn_mixer(
        weights,
        state=state,
        scratch=scratch,
        rows=rows,
        hidden=hidden,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel=conv_kernel,
        conv_weight_ptr=conv_weight_ptr,
        dt_bias_ptr=dt_bias_ptr,
        a_ptr=a_ptr,
        norm_weight_ptr=norm_weight_ptr,
        runtime=runtime,
    )
    run_qwen4_exp_batch_gdn_input_projections(
        plan, mixed_ptr, weights, state=state, scratch=scratch, hidden=hidden,
        stream=stream,
    )
    run_qwen4_exp_batch_gdn_recurrence(
        plan,
        state=state,
        scratch=scratch,
        conv_weight_ptr=conv_weight_ptr,
        dt_bias_ptr=dt_bias_ptr,
        a_ptr=a_ptr,
        norm_weight_ptr=norm_weight_ptr,
        stream=stream,
    )
    return run_qwen4_exp_batch_gdn_output_projection(
        plan, weights, scratch=scratch, hidden=hidden, stream=stream
    )


def _validate(
    weights: Mapping[str, GGUFDeviceWeight],
    *,
    state: Qwen4ExpBatchGdnState,
    scratch: Qwen4ExpGDNScratch,
    rows: int,
    hidden: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_kernel: int,
    conv_weight_ptr: int,
    dt_bias_ptr: int,
    a_ptr: int,
    norm_weight_ptr: int,
    runtime: HipRuntime | None,
) -> tuple[HipRuntime, int, int, int, int]:
    """Admit one batch mixer call before any dispatch touches the state."""

    if scratch.closed or state.closed:
        raise RuntimeError("Qwen4Exp GDN scratch/state is closed")
    if state.closing:
        raise RuntimeError(
            "Qwen4Exp GDN state is closing; its buffers may already be released"
        )
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime or active_runtime is not state.runtime:
        raise ValueError("runtime must match GDN scratch and state owners")

    count = _positive_int(rows, "rows")
    parsed_hidden = _positive_int(hidden, "hidden")
    parsed_k = _positive_int(num_k_heads, "num_k_heads")
    parsed_v = _positive_int(num_v_heads, "num_v_heads")
    parsed_head_dim = _positive_int(head_dim, "head_dim")
    parsed_kernel = _positive_int(conv_kernel, "conv_kernel")
    if parsed_v % parsed_k:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    if parsed_head_dim != _INDEXED_HEAD_DIM:
        raise ValueError(
            f"indexed GDN decode requires head_dim {_INDEXED_HEAD_DIM}, got "
            f"{parsed_head_dim}"
        )
    if (parsed_k, parsed_v, parsed_head_dim, parsed_kernel) != (
        state.num_k_heads,
        state.num_v_heads,
        state.head_dim,
        state.conv_kernel,
    ):
        raise ValueError(
            "GDN geometry must match the state owner "
            f"({state.num_k_heads} k heads x {state.num_v_heads} v heads x "
            f"{state.head_dim}, kernel {state.conv_kernel})"
        )
    if count > QWEN4_EXP_GDN_INDEXED_MAX_ROWS or count > QWEN4_EXP_CONV_INDEXED_MAX_ROWS:
        raise ValueError(
            f"rows {count} exceeds the indexed decode row bound "
            f"{min(QWEN4_EXP_GDN_INDEXED_MAX_ROWS, QWEN4_EXP_CONV_INDEXED_MAX_ROWS)}"
        )
    if state.state_slots > QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS:
        raise ValueError(
            f"state_slots {state.state_slots} exceeds the indexed conv slab width "
            f"{QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS}"
        )
    qkv_width = 2 * parsed_k * parsed_head_dim + parsed_v * parsed_head_dim
    core_width = parsed_v * parsed_head_dim
    if qkv_width > QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS:
        raise ValueError(
            f"qkv_width {qkv_width} exceeds the indexed conv channel bound "
            f"{QWEN4_EXP_CONV_INDEXED_MAX_CHANNELS}"
        )
    if parsed_kernel > QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE:
        raise ValueError(
            f"conv_kernel {parsed_kernel} exceeds the indexed conv kernel bound "
            f"{QWEN4_EXP_CONV_INDEXED_MAX_KERNEL_SIZE}"
        )
    if qkv_width % 4:
        raise ValueError(
            "qkv_width must be a multiple of 4 floats so every Conv row base stays "
            "16-byte aligned for the float4 Q/K loads"
        )

    missing = sorted(set(_REQUIRED_PROJECTIONS) - set(weights))
    if missing:
        raise ValueError("missing Qwen4Exp GDN weights: " + ", ".join(missing))
    for name, pointer in (
        ("conv_weight", conv_weight_ptr),
        ("dt_bias", dt_bias_ptr),
        ("a", a_ptr),
        ("norm_weight", norm_weight_ptr),
    ):
        _positive_int(pointer, f"{name}_ptr")

    active = state.active
    if active is None:
        raise RuntimeError(
            "Qwen4Exp batch GDN state has no active rows; call prepare first"
        )
    if count != active.rows:
        raise ValueError(
            f"mixer rows {count} does not match the {active.rows} rows prepared on "
            "the GDN state"
        )
    # Re-admit the published row set: this rejects a caller that rewrote the
    # vector after ``prepare`` and keeps one authoritative rule for the ABI.
    qwen4_exp_gdn_indexed_state_indices(
        active.indices_host, rows=count, state_slots=state.state_slots
    )
    indices_ptr = active.indices.ptr
    if indices_ptr % 8:
        raise ValueError("state indices must be 8-byte aligned for int64 loads")
    if scratch.conv.ptr % _FLOAT4_BYTES:
        raise ValueError(
            "GDN scratch conv must be 16-byte aligned for the float4 Q/K loads"
        )

    _require_scratch(scratch, count, qkv_width, core_width, parsed_v, parsed_hidden)
    _require_projection_geometry(
        weights,
        {
            "attn_qkv": (qkv_width, parsed_hidden),
            "attn_gate": (core_width, parsed_hidden),
            "ssm_alpha": (parsed_v, parsed_hidden),
            "ssm_beta": (parsed_v, parsed_hidden),
            "ssm_out": (parsed_hidden, core_width),
        },
    )
    return active_runtime, count, qkv_width, core_width, indices_ptr


def _require_scratch(
    scratch: Qwen4ExpGDNScratch,
    rows: int,
    qkv_width: int,
    core_width: int,
    scalar_width: int,
    hidden: int,
) -> None:
    """Reject a scratch that cannot hold ``rows`` of this GDN geometry."""

    required = {
        "qkv": (rows * qkv_width, scratch.qkv),
        "gate": (rows * core_width, scratch.gate),
        "alpha": (rows * scalar_width, scratch.alpha),
        "beta": (rows * scalar_width, scratch.beta),
        "conv": (rows * qkv_width, scratch.conv),
        "core": (rows * core_width, scratch.core),
        "output": (rows * hidden, scratch.output),
    }
    for name, (elements, buffer) in required.items():
        needed = elements * 4
        if buffer.nbytes < needed:
            raise ValueError(
                f"Qwen4Exp GDN scratch {name} holds {buffer.nbytes} bytes, needs "
                f"{needed} for {rows} rows"
            )


def _require_projection_geometry(
    weights: Mapping[str, GGUFDeviceWeight],
    expected: dict[str, tuple[int, int]],
) -> None:
    """Reject a projection whose stored shape cannot produce the staged layout.

    The output projection runs last, after both indexed kernels have committed
    state, so a shape mismatch there would otherwise advance the recurrent and
    Conv slots with no result. Weight objects that do not expose a source tensor
    shape keep the dispatch's own check.
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


def _check_slab_products(
    *,
    state_slots: int,
    qkv_width: int,
    conv_kernel: int,
    num_v_heads: int,
    head_dim: int,
) -> None:
    """Reject slab shapes whose products the kernels' integer types cannot hold."""

    conv_elements = state_slots * qkv_width * conv_kernel
    if conv_elements > _INT64_MAX // 4:
        raise ValueError(
            f"{conv_elements} Conv state elements exceed the int64 byte bound"
        )
    recurrent_elements = state_slots * num_v_heads * head_dim * head_dim
    if recurrent_elements > _INT64_MAX // 4:
        raise ValueError(
            f"{recurrent_elements} recurrent state elements exceed the int64 byte bound"
        )
    if num_v_heads > _INT32_MAX or qkv_width > _INT32_MAX:
        raise ValueError("GDN head/width geometry exceeds the int32 kernel arguments")


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


def _index_sequence(values: object, name: str) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        items = list(values)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(f"{name} must be a sequence of integers") from None
    return [_int_index(item, name) for item in items]


def batch_gdn_call_table() -> dict[str, Callable[..., Any]]:
    """The exact direct callables this module uses for one batch GDN step.

    The projections are not listed here: they go through
    :func:`launch_gguf_linear`'s row-dependent registry dispatch and are bound
    separately as a dispatch record. These two indexed owners are imported and
    called directly by this module, so the manifest must not describe them as
    registry-resolved even though matching registry keys exist.
    """

    return {
        "gdn_conv_indexed": qwen4_exp_linear_attn_conv_decode_indexed_f32,
        "gdn_recurrence_norm_gate": qwen4_exp_gdn_decode_rowsplit_indexed_f32,
    }
