"""Independent-request QSA token mixers over the batch attention owner.

``run_qwen4_exp_dense_qsa_token_mixer`` in ``runtime/qwen4_exp_runner.py`` is the
strict single-row oracle: one request, one ``Qwen4ExpDenseAttentionState``, one
int64 position scalar. This module is the N-request form of the same dense
sub-budget arithmetic on top of ``Qwen4ExpBatchAttentionState``: every row is an
independent request with its own slot, page table, and position, and each stage
is a single N-row dispatch instead of a loop of single-row mixers.

Stages, in order, one dispatch each (the dense-only path is identical to the
oracle's dense path; the mixed path replaces stage 4 with the index pipeline and
one variable-selection rows dispatch):

1. ``attn_q`` / ``attn_k`` / ``attn_v`` GGUF projections over the N-row hidden.
2. ``qwen4_exp_qsa_split_norm_rope_rows_f32`` for Q/K norm, split-half RoPE,
   and the gate split, reading the positions ``prepare`` uploaded.
3. ``qwen35_write_paged_kv_f32_batch_spans`` into the per-row page tables.
4. ``qwen35_paged_full_attn_decode_context_bf16_batch_spans`` with per-row
   context lengths.
5. ``qwen4_exp_qsa_gate_context_f32`` over ``rows * query_heads * head_dim``.
6. ``attn_output`` GGUF projection back to the N-row hidden.

Two entry points share that stage list:

* ``run_qwen4_exp_batch_dense_qsa_token_mixer`` is the dense-only mixer. It takes
  the caller's ``dense_equivalent_limit`` and rejects any prepared row past it
  before the first dispatch, so an integration cannot silently compute dense
  attention where sparse selection is required.
* ``run_qwen4_exp_batch_qsa_mixed_token_mixer`` adds the index pipeline
  (``index_k`` projected and appended for every row on every step, ``index_q``
  and its norm/rope only when a row is past the bound) and attends every row
  through one variable-selection rows dispatch. Rows inside the bound carry all
  visible positions as their selection, so dense and sparse rows share one
  representation and one kernel launch; the bound is the index owner's own
  ``dense_equivalent_limit``. Dense rows therefore match the dense-only mixer to
  FP32 round-off rather than bit-exactly, and a caller that needs the dense
  decode kernel's arithmetic keeps those rows on the dense entry point.

Nothing here dispatches by backend or quant: the projections go through
``launch_gguf_linear``'s registry resolution and the rest are the same public
wrappers the oracle calls.

Every argument the later wrappers check is validated here, before the KV write:
row count and slot/position agreement with ``prepare``, capacity, scratch
sizing, block size, head geometry including the decode's ``head_dim <= 256``
bound, finite positive ``theta``/``eps``, the optional MRoPE position pointer,
and the four projection geometries. A malformed call therefore cannot write a
token into the KV cache.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.runtime import MemcpyKind
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_f32_batch_spans,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_gate_context_f32,
    qwen4_exp_qsa_norm_mrope_rows_f32,
    qwen4_exp_qsa_norm_rope_rows_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32,
    qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32,
    qwen4_exp_qsa_split_norm_mrope_rows_f32,
    qwen4_exp_qsa_split_norm_rope_rows_f32,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpQSAMixerDeviceWeights,
    Qwen4ExpQSAScratch,
)

# The dense paged decode wrapper requires this block size; checking it here keeps
# a misconfigured state from writing a token that the decode can never read.
_DENSE_DECODE_BLOCK_SIZE = 256
# ``paged_attn_decode._check_decode_shape`` rejects head_dim > 256, but only after
# the KV write has already committed the token, so bound it before the write too.
_DENSE_DECODE_MAX_HEAD_DIM = 256
_REQUIRED_PROJECTIONS = ("attn_q", "attn_k", "attn_v", "attn_output")
_INDEX_PROJECTIONS = ("index_q", "index_k")


@dataclass
class Qwen4ExpBatchQsaSelectionScratch:
    """Per-row selection staging for the mixed dense/sparse QSA mixer.

    ``positions`` is ``int64[rows, stride]`` and ``counts`` is ``int32[rows]``,
    the exact layout ``qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32``
    reads: row ``r`` attends ``positions[r, :counts[r]]`` against row ``r``'s own
    page table. One buffer therefore serves both row kinds of a mixed batch, and
    every row of one call is covered by a single dispatch.

    ``stride`` must cover the widest row selection. The index owner's
    ``output_stride`` bounds both kinds: the budgeted rule emits at most that
    many positions, and a dense row's all-visible selection is at most the same
    length because a row is dense exactly when its context is within
    ``dense_equivalent_limit == output_stride``.

    ``queries`` stages the sparse group's prepared index queries in that group's
    own row order, which is what ``select_active`` indexes by. It is a gather of
    at most ``rows`` small device-to-device copies and holds no host mirror.
    """

    positions: DeviceBuffer
    counts: DeviceBuffer
    queries: DeviceBuffer
    positions_host: np.ndarray
    counts_host: np.ndarray
    rows: int
    stride: int
    index_heads: int
    index_dim: int
    runtime: HipRuntime
    closed: bool = False
    closing: bool = False
    _pending_free: list[DeviceBuffer] = field(default_factory=list)
    # What the device buffers hold. Every row starts as ``arange(stride)``,
    # which is a dense row's selection for any context (the kernel reads only
    # ``positions[r, :counts[r]]``), so a dense row is never re-uploaded and a
    # step whose counts match the last upload uploads nothing. A row a sparse
    # selection overwrote is restored to the arange when it is dense again.
    row_is_arange: np.ndarray | None = None
    device_counts: np.ndarray | None = None

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        stride: int,
        index_heads: int,
        index_dim: int,
        runtime: HipRuntime | None = None,
    ) -> "Qwen4ExpBatchQsaSelectionScratch":
        parsed_rows = _positive(rows, "rows")
        parsed_stride = _positive(stride, "stride")
        parsed_heads = _positive(index_heads, "index_heads")
        parsed_dim = _positive(index_dim, "index_dim")
        # Byte products are checked, then the host mirrors are built, before the
        # runtime is touched: a shape whose slabs are unaddressable, or a host
        # allocation that fails, must not leave device buffers allocated.
        sizes = _selection_products(
            rows=parsed_rows,
            stride=parsed_stride,
            index_heads=parsed_heads,
            index_dim=parsed_dim,
        )
        positions_host = np.array(
            np.broadcast_to(
                np.arange(parsed_stride, dtype=np.int64), (parsed_rows, parsed_stride)
            ),
            copy=True, order='C',
        )
        counts_host = np.zeros(parsed_rows, dtype=np.int32)
        active_runtime = runtime or get_hip_runtime()
        buffers: list[DeviceBuffer] = []
        try:
            for name in ("positions", "counts", "queries"):
                buffers.append(malloc(sizes[name], runtime=active_runtime))
            copy_host_to_device(
                buffers[0], host_array_ptr(positions_host), positions_host.nbytes,
                runtime=active_runtime,
            )
            copy_host_to_device(
                buffers[1], host_array_ptr(counts_host), counts_host.nbytes,
                runtime=active_runtime,
            )
        except BaseException:
            _free_pending(buffers, active_runtime)
            raise
        return cls(
            positions=buffers[0],
            counts=buffers[1],
            queries=buffers[2],
            positions_host=positions_host,
            counts_host=counts_host,
            rows=parsed_rows,
            stride=parsed_stride,
            index_heads=parsed_heads,
            index_dim=parsed_dim,
            runtime=active_runtime,
            _pending_free=buffers,
            row_is_arange=np.ones(parsed_rows, dtype=bool),
            device_counts=counts_host.copy(),
        )

    def close(self) -> None:
        """Release the slabs, tolerating a free that fails part-way.

        ``closing`` is set before the first free, so a partial close poisons every
        accessor instead of handing out an address that may already be released.
        Each buffer is popped only after its free returns, so a retry frees the
        remainder exactly once rather than leaking or double-freeing it.
        """

        if self.closed:
            return
        self.closing = True
        pending = self._pending_free
        if not pending:
            pending.extend((self.positions, self.counts, self.queries))
        _free_pending(pending, self.runtime)
        self.closed = True

    def __enter__(self) -> "Qwen4ExpBatchQsaSelectionScratch":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def query_row_ptr(self, row: int) -> int:
        """Device address of staged index query ``row`` in the sparse row order."""

        self._require_open()
        if not 0 <= row < self.rows:
            raise IndexError(f"row {row} is outside [0, {self.rows})")
        return self.queries.ptr + row * self.index_heads * self.index_dim * 4

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp QSA selection scratch is closed")
        if self.closing:
            raise RuntimeError(
                "Qwen4Exp QSA selection scratch is closing; its buffers may already "
                "be released"
            )


def run_qwen4_exp_batch_dense_qsa_token_mixer(
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpBatchAttentionState,
    scratch: Qwen4ExpQSAScratch,
    rows: int,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    dense_equivalent_limit: int,
    rope_positions_ptr: int | None = None,
    attention_context_limit: int | None = None,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Run one N-row dense QSA decode step for independent requests.

    ``attention_state.prepare(slot_ids, positions)`` must already have admitted
    exactly ``rows`` active rows; this call consumes those positions and page
    tables and never re-uploads or reorders them. ``dense_equivalent_limit`` is
    the largest context length for which dense attention is equivalent to the
    sparse path; every prepared row's context length must be within it, which is
    the conservative test when the caller's index count is smaller than that row's
    context. Returns the ``[rows, hidden]`` output buffer owned by ``scratch``.
    """

    active_runtime, count, q_width, kv_width, context_limit = _validate(
        weights,
        attention_state=attention_state,
        scratch=scratch,
        rows=rows,
        hidden=hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        eps=eps,
        dense_equivalent_limit=dense_equivalent_limit,
        rope_positions_ptr=rope_positions_ptr,
        attention_context_limit=attention_context_limit,
        runtime=runtime,
    )
    active = attention_state.active
    assert active is not None  # _validate admits only a prepared state
    positions_ptr = (
        active.position.ptr if rope_positions_ptr is None else int(rope_positions_ptr)
    )

    for slot, output, out_features in (
        ("attn_q", scratch.q_projected, q_width * 2),
        ("attn_k", scratch.key_projected, kv_width),
        ("attn_v", scratch.value_projected, kv_width),
    ):
        launch_gguf_linear(
            weights.projections[slot],
            mixed_ptr,
            output.ptr,
            count,
            hidden,
            out_features,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
    split_rope_rows = (
        qwen4_exp_qsa_split_norm_mrope_rows_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_split_norm_rope_rows_f32
    )
    split_rope_rows(
        scratch.q_projected.ptr,
        scratch.key_projected.ptr,
        weights.q_norm_weight_ptr,
        weights.k_norm_weight_ptr,
        positions_ptr,
        scratch.query.ptr,
        scratch.key.ptr,
        scratch.gate.ptr,
        count,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen35_write_paged_kv_f32_batch_spans(
        scratch.key.ptr,
        scratch.value_projected.ptr,
        attention_state.key_cache.ptr,
        attention_state.value_cache.ptr,
        active.append_spans,
        count,
        attention_state.block_size,
        kv_heads,
        head_dim,
        stream=stream,
        runtime=active_runtime,
    )
    qwen35_paged_full_attn_decode_context_bf16_batch_spans(
        scratch.query.ptr,
        attention_state.key_cache.ptr,
        attention_state.value_cache.ptr,
        scratch.context.ptr,
        active.decode_spans,
        count,
        context_limit,
        attention_state.block_size,
        query_heads,
        kv_heads,
        head_dim,
        head_dim**-0.5,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_qsa_gate_context_f32(
        scratch.context.ptr,
        scratch.gate.ptr,
        scratch.gated.ptr,
        count * q_width,
        stream=stream,
        runtime=active_runtime,
    )
    launch_gguf_linear(
        weights.projections["attn_output"],
        scratch.gated.ptr,
        scratch.output.ptr,
        count,
        q_width,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    return scratch.output


def run_qwen4_exp_batch_qsa_mixed_token_mixer(
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    selection: Qwen4ExpBatchQsaSelectionScratch,
    scratch: Qwen4ExpQSAScratch,
    rows: int,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    rope_positions_ptr: int | None = None,
    attention_context_limit: int | None = None,
    eps: float = 1e-6,
    stream: int = 0,
    runtime: HipRuntime | None = None,
) -> DeviceBuffer:
    """Run one N-row QSA decode step with mixed dense and sparse rows.

    Same stages as the dense mixer, plus the index pipeline the sparse path
    needs: ``index_k`` is projected and appended for **every** row on **every**
    step, including rows still inside the dense-equivalence bound, so a request
    that crosses the threshold keeps its complete index history. ``index_q`` is
    projected and normalized only when at least one row is past the bound.

    Row kinds are decided by the index owner's own rule
    (``partition_rows``): a row is dense while its complete-block count is within
    ``block_budget``, and dense rows are attended with **all visible positions**
    as their selection. That is the one unified representation this mixer uses:
    every row carries a ``(positions, count)`` entry in ``selection`` and the
    whole batch is attended by one
    ``qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32`` dispatch over the
    per-row page tables. Dense rows therefore match the dense mixer to FP32
    round-off rather than bit-exactly, because the online-softmax rows kernel is
    not the dense decode kernel; a caller that needs the dense decode arithmetic
    keeps those rows on ``run_qwen4_exp_batch_dense_qsa_token_mixer``, which this
    module leaves unchanged.

    Serial work, accepted as the current fallback and not hidden: the index
    append is one scatter per row, ``select_active`` is one pooled-key score and
    one selection per sparse row, and a mixed row set's metadata uploads stage
    host selections. Every projection, the KV write, the attention, the gate, and
    the output projection are single N-row dispatches; no stage loops the c1
    mixer.

    An **all-sparse** row set takes the device form instead. The sparse group is
    then the whole row set, so the index owner's own ``selected_positions`` and
    ``selected_counts`` already hold one selection per row in this step's row
    order, at the owner's own ``output_stride``. Those device buffers go straight
    to the attention kernel -- the same device-resident selection the c1 owner
    hands its kernel -- and no selection metadata is copied to or from the host.
    A mixed or dense-only row set keeps the host-staged path unchanged, because
    only a sparse row has a device selection to reuse and the staging buffer is
    what carries a dense row's all-visible selection.

    ``attention_state.prepare`` and ``index_state.prepare`` must already have
    admitted the same slot list in the same row order for the current positions.
    The call mutates the index cursors (append) and the KV cache before the
    attention reads them, so it is not transactional: capture
    ``index_state.begin_rollback()`` before the call and, on any failure,
    ``index_state.rollback(token)`` and re-run the same step. A rollback restores
    the cursors, and the re-run rewrites the same KV bytes for the same
    positions, which is what makes the retry equivalent to a clean run. On
    success the caller discards the token. The index owner's active set is
    restored to the full row list before returning, so the next step's
    ``append_active`` sees every row again.
    """

    plan = validate_qwen4_exp_batch_qsa_mixed_mixer(
        weights,
        attention_state=attention_state,
        index_state=index_state,
        selection=selection,
        scratch=scratch,
        rows=rows,
        hidden=hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        rope_positions_ptr=rope_positions_ptr,
        attention_context_limit=attention_context_limit,
        eps=eps,
        runtime=runtime,
    )
    run_qwen4_exp_batch_qsa_input_projections(
        plan, mixed_ptr, weights, scratch=scratch, hidden=hidden,
        index_dim=index_dim, stream=stream,
    )
    run_qwen4_exp_batch_qsa_attention(
        plan,
        mixed_ptr,
        weights,
        attention_state=attention_state,
        index_state=index_state,
        selection=selection,
        scratch=scratch,
        hidden=hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
        rope_positions_ptr=rope_positions_ptr,
        eps=eps,
        stream=stream,
    )
    return run_qwen4_exp_batch_qsa_output_projection(
        plan, weights, scratch=scratch, hidden=hidden, stream=stream
    )


@dataclass(frozen=True)
class Qwen4ExpBatchQsaMixerPlan:
    """One admitted mixed QSA mixer call, split into its stages.

    The input and output projections depend only on ``rows`` and fixed scratch
    pointers, so the batch runner may graph-capture them. The attention stage
    appends index keys, writes KV, stages host selections, and reads per-row
    positions, so it always runs eagerly.
    """

    runtime: HipRuntime
    rows: int
    q_width: int
    kv_width: int
    dense_rows: tuple[int, ...]
    sparse_rows: tuple[int, ...]
    positions_ptr: int


def validate_qwen4_exp_batch_qsa_mixed_mixer(
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    selection: Qwen4ExpBatchQsaSelectionScratch,
    scratch: Qwen4ExpQSAScratch,
    rows: int,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    rope_positions_ptr: int | None = None,
    attention_context_limit: int | None = None,
    eps: float = 1e-6,
    runtime: HipRuntime | None = None,
) -> Qwen4ExpBatchQsaMixerPlan:
    """Admit one mixed mixer call before any dispatch touches the state."""


    active_runtime, count, q_width, kv_width, context_limit = _validate(
        weights,
        attention_state=attention_state,
        scratch=scratch,
        rows=rows,
        hidden=hidden,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        theta=theta,
        eps=eps,
        dense_equivalent_limit=index_state.dense_equivalent_limit,
        rope_positions_ptr=rope_positions_ptr,
        attention_context_limit=attention_context_limit,
        runtime=runtime,
        require_dense=False,
    )
    dense_rows, sparse_rows = _validate_index(
        weights,
        active_runtime=active_runtime,
        attention_state=attention_state,
        index_state=index_state,
        selection=selection,
        scratch=scratch,
        rows=count,
        hidden=hidden,
        index_heads=index_heads,
        index_dim=index_dim,
        index_rotary_dim=index_rotary_dim,
    )
    active = attention_state.active
    assert active is not None  # _validate admits only a prepared state
    positions_ptr = (
        active.position.ptr if rope_positions_ptr is None else int(rope_positions_ptr)
    )

    return Qwen4ExpBatchQsaMixerPlan(
        active_runtime, count, q_width, kv_width, dense_rows, sparse_rows,
        positions_ptr,
    )


def run_qwen4_exp_batch_qsa_input_projections(
    plan: Qwen4ExpBatchQsaMixerPlan,
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    scratch: Qwen4ExpQSAScratch,
    hidden: int,
    index_dim: int,
    stream: int = 0,
) -> None:
    """Project Q/K/V and the raw index key for every row (stateless)."""

    active_runtime, count = plan.runtime, plan.rows
    q_width, kv_width = plan.q_width, plan.kv_width
    for slot, output, out_features in (
        ("attn_q", scratch.q_projected, q_width * 2),
        ("attn_k", scratch.key_projected, kv_width),
        ("attn_v", scratch.value_projected, kv_width),
    ):
        launch_gguf_linear(
            weights.projections[slot],
            mixed_ptr,
            output.ptr,
            count,
            hidden,
            out_features,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
    launch_gguf_linear(
        weights.projections["index_k"],
        mixed_ptr,
        scratch.index_k_projected.ptr,
        count,
        hidden,
        index_dim,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )


def run_qwen4_exp_batch_qsa_attention(
    plan: Qwen4ExpBatchQsaMixerPlan,
    mixed_ptr: int,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    selection: Qwen4ExpBatchQsaSelectionScratch,
    scratch: Qwen4ExpQSAScratch,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    rope_positions_ptr: int | None = None,
    eps: float = 1e-6,
    stream: int = 0,
) -> None:
    """Append index keys, write KV, select, attend, and gate (stateful)."""

    active_runtime, count = plan.runtime, plan.rows
    q_width = plan.q_width
    dense_rows, sparse_rows = plan.dense_rows, plan.sparse_rows
    positions_ptr = plan.positions_ptr
    active = attention_state.active
    assert active is not None  # validate admitted only a prepared state
    # Every row, every step: a request that is dense now still needs the keys it
    # saw while it was dense once it crosses the bound.
    index_state.append_active(scratch.index_k_projected.ptr, stream=stream)
    if index_state.partition_rows() != (dense_rows, sparse_rows):
        raise RuntimeError(
            "QSA index row kinds changed between preflight and append; the index "
            "cursors were mutated concurrently"
        )
    split_rope_rows = (
        qwen4_exp_qsa_split_norm_mrope_rows_f32
        if rope_positions_ptr is not None
        else qwen4_exp_qsa_split_norm_rope_rows_f32
    )
    split_rope_rows(
        scratch.q_projected.ptr,
        scratch.key_projected.ptr,
        weights.q_norm_weight_ptr,
        weights.k_norm_weight_ptr,
        positions_ptr,
        scratch.query.ptr,
        scratch.key.ptr,
        scratch.gate.ptr,
        count,
        query_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        theta,
        eps,
        stream=stream,
        runtime=active_runtime,
    )
    qwen35_write_paged_kv_f32_batch_spans(
        scratch.key.ptr,
        scratch.value_projected.ptr,
        attention_state.key_cache.ptr,
        attention_state.value_cache.ptr,
        active.append_spans,
        count,
        attention_state.block_size,
        kv_heads,
        head_dim,
        stream=stream,
        runtime=active_runtime,
    )
    slot_ids = active.slot_ids
    sparse_counts: np.ndarray | None = None
    sparse_positions: np.ndarray | None = None
    if sparse_rows:
        index_rope = (
            qwen4_exp_qsa_norm_mrope_rows_f32
            if rope_positions_ptr is not None
            else qwen4_exp_qsa_norm_rope_rows_f32
        )
        launch_gguf_linear(
            weights.projections["index_q"],
            mixed_ptr,
            scratch.index_q_projected.ptr,
            count,
            hidden,
            index_heads * index_dim,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            stream=stream,
            runtime=active_runtime,
        )
        index_rope(
            scratch.index_q_projected.ptr,
            weights.index_q_norm_weight_ptr,
            positions_ptr,
            scratch.index_query.ptr,
            count,
            index_heads,
            index_dim,
            index_rotary_dim,
            theta,
            eps,
            stream=stream,
            runtime=active_runtime,
        )
        # ``select_active`` scores the sparse group only, so it is published as
        # its own row set; the full set is restored before returning. It indexes
        # its query rows by that group's own order, so each sparse row's prepared
        # query is gathered into it first.
        query_bytes = index_heads * index_dim * 4
        for subset_row, row in enumerate(sparse_rows):
            active_runtime.memcpy(
                selection.query_row_ptr(subset_row),
                scratch.index_query.ptr + row * query_bytes,
                query_bytes,
                MemcpyKind.DEVICE_TO_DEVICE,
            )
        subset_slots = [slot_ids[row] for row in sparse_rows]
        index_state.prepare(subset_slots)
        if sparse_rows == tuple(range(count)):
            # Every row is sparse, so the sparse group *is* the row set: the
            # owner's own buffers already hold one selection per row in this
            # step's row order, at the owner's own stride. Handing them to the
            # attention kernel is what the c1 owner does, and it keeps the whole
            # selection on device instead of staging it through host memory.
            device_selection = index_state.select_active_device(
                selection.queries.ptr,
                key_norm_weight_ptr=weights.index_k_norm_weight_ptr,
                rotary_dim=index_rotary_dim,
                theta=theta,
                eps=eps,
                stream=stream,
            )
            if device_selection.slot_ids != tuple(subset_slots):
                raise RuntimeError(
                    "QSA device selection covers slots "
                    f"{list(device_selection.slot_ids)} for the prepared sparse "
                    f"group {subset_slots}; the selection cannot be attributed "
                    "to these rows"
                )
        else:
            device_selection = None
            sparse_counts, sparse_positions = index_state.select_active(
                selection.queries.ptr,
                key_norm_weight_ptr=weights.index_k_norm_weight_ptr,
                rotary_dim=index_rotary_dim,
                theta=theta,
                eps=eps,
                stream=stream,
            )
        index_state.prepare(slot_ids)
    else:
        device_selection = None
    if device_selection is None:
        _stage_selection(
            selection,
            rows=count,
            contexts=attention_state.context_host,
            sparse_rows=sparse_rows,
            sparse_counts=sparse_counts,
            sparse_positions=sparse_positions,
            runtime=active_runtime,
        )
        selected_positions_ptr = selection.positions.ptr
        selected_counts_ptr = selection.counts.ptr
        selected_stride = selection.stride
    else:
        # ``stride`` is the index owner's, which is the stride its selection
        # kernel wrote; the selection scratch only has to cover it.
        selected_positions_ptr = device_selection.positions_ptr
        selected_counts_ptr = device_selection.counts_ptr
        selected_stride = device_selection.stride
    attention_fn, _ = batch_qsa_attention_implementation(
        head_dim=head_dim,
        block_size=attention_state.block_size,
        query_heads=query_heads,
        kv_heads=kv_heads,
    )
    attention_fn(
        scratch.query.ptr,
        attention_state.key_cache.ptr,
        attention_state.value_cache.ptr,
        selected_positions_ptr,
        selected_counts_ptr,
        scratch.context.ptr,
        active.decode_spans,
        rows=count,
        selected_stride=selected_stride,
        block_size=attention_state.block_size,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_qsa_gate_context_f32(
        scratch.context.ptr,
        scratch.gate.ptr,
        scratch.gated.ptr,
        count * q_width,
        stream=stream,
        runtime=active_runtime,
    )


def run_qwen4_exp_batch_qsa_output_projection(
    plan: Qwen4ExpBatchQsaMixerPlan,
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    scratch: Qwen4ExpQSAScratch,
    hidden: int,
    stream: int = 0,
) -> DeviceBuffer:
    """Project the gated attention context back to ``hidden`` (stateless)."""

    active_runtime, count, q_width = plan.runtime, plan.rows, plan.q_width
    launch_gguf_linear(
        weights.projections["attn_output"],
        scratch.gated.ptr,
        scratch.output.ptr,
        count,
        q_width,
        hidden,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        stream=stream,
        runtime=active_runtime,
    )
    return scratch.output


def _stage_selection(
    selection: Qwen4ExpBatchQsaSelectionScratch,
    *,
    rows: int,
    contexts: np.ndarray,
    sparse_rows: tuple[int, ...],
    sparse_counts: np.ndarray | None,
    sparse_positions: np.ndarray | None,
    runtime: HipRuntime,
) -> None:
    """Publish one ``(positions, counts)`` row selection per row.

    Dense rows get every visible position, so the same rows kernel attends them
    as a variable-length selection with no separate code path. A dense row's
    device positions are already ``arange(stride)``, so only its count changes;
    the counts upload is skipped when the device already holds them, which is
    every QSA layer after the first in one step. Sparse rows upload their own
    positions, and a row they overwrote is restored to the arange when it next
    runs dense.

    This is the mixed and dense-only path. An all-sparse row set does not call it:
    that step hands the index owner's device selection to the kernel directly, so
    the staging mirrors keep whatever a previous staged step left in them and are
    recomputed by the next call.
    """

    counts = selection.counts_host
    counts.fill(0)
    sparse = set(sparse_rows)
    row_bytes = selection.stride * np.dtype(np.int64).itemsize
    dirty: list[int] = []
    for row in range(rows):
        if row in sparse:
            continue
        # A dense row is attended as a variable-length selection of every visible
        # position, so the count is its context length.
        context = int(contexts[row])
        if context > selection.stride:
            raise ValueError(
                f"row {row} dense selection of {context} positions does not fit the "
                f"selection stride {selection.stride}"
            )
        counts[row] = context
        if not selection.row_is_arange[row]:
            selection.positions_host[row] = np.arange(selection.stride, dtype=np.int64)
            selection.row_is_arange[row] = True
            dirty.append(row)
    for subset_row, row in enumerate(sparse_rows):
        assert sparse_counts is not None
        count = int(sparse_counts[subset_row])
        if not 0 < count <= selection.stride:
            raise ValueError(
                f"row {row} selected {count} positions, which the selection stride "
                f"{selection.stride} cannot hold"
            )
        assert sparse_positions is not None
        counts[row] = count
        selection.positions_host[row, :count] = sparse_positions[subset_row, :count]
        selection.row_is_arange[row] = False
        dirty.append(row)
    for row in dirty:
        runtime.memcpy(
            selection.positions.ptr + row * row_bytes,
            host_array_ptr(selection.positions_host[row]),
            row_bytes,
            MemcpyKind.HOST_TO_DEVICE,
        )
    if not np.array_equal(counts, selection.device_counts):
        copy_host_to_device(
            selection.counts,
            host_array_ptr(counts),
            selection.counts.nbytes,
            runtime=runtime,
        )
        selection.device_counts[:] = counts


def _validate_index(
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    active_runtime: HipRuntime,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    selection: Qwen4ExpBatchQsaSelectionScratch,
    scratch: Qwen4ExpQSAScratch,
    rows: int,
    hidden: int,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Admit the index side and return the ``(dense, sparse)`` row split.

    Runs before the first dispatch, so an index mismatch cannot leave a token in
    the KV cache or an extra key in the index slab. The split is predicted from
    the cursors the append will produce (``cursor + 1``); the caller re-reads it
    from ``partition_rows`` after the append and this prediction is what the
    preflight capacity and stride checks are sized against.
    """

    if index_state.closed or selection.closed:
        raise RuntimeError("Qwen4Exp QSA index/selection owner is closed")
    if index_state.closing or selection.closing:
        raise RuntimeError(
            "Qwen4Exp QSA index/selection owner is closing; its buffers may already "
            "be released"
        )
    if index_state.runtime is not active_runtime or selection.runtime is not active_runtime:
        raise ValueError("runtime must match the QSA index and selection owners")
    if index_state.attention is not attention_state:
        raise ValueError(
            "QSA index state must mirror the batch attention state passed to the mixer"
        )
    parsed_heads = _positive(index_heads, "index_heads")
    parsed_dim = _positive(index_dim, "index_dim")
    parsed_rotary = _positive(index_rotary_dim, "index_rotary_dim")
    if parsed_rotary > parsed_dim or parsed_rotary % 2:
        raise ValueError("index_rotary_dim must be even and <= index_dim")
    if parsed_heads != index_state.index_heads or parsed_dim != index_state.index_dim:
        raise ValueError(
            "QSA index geometry must match its state owner "
            f"({index_state.index_heads} heads x {index_state.index_dim})"
        )
    missing = sorted(set(_INDEX_PROJECTIONS) - set(weights.projections))
    if missing:
        raise ValueError("missing Qwen4Exp QSA index weights: " + ", ".join(missing))
    if weights.index_q_norm_weight_ptr == 0 or weights.index_k_norm_weight_ptr == 0:
        raise ValueError("Qwen4Exp QSA index Q/K norm weights are required")
    parsed_hidden = _positive(hidden, "hidden")
    _require_projection_geometry(
        weights,
        {
            "index_q": (parsed_heads * parsed_dim, parsed_hidden),
            "index_k": (parsed_dim, parsed_hidden),
        },
    )
    required = {
        "index_q_projected": rows * parsed_heads * parsed_dim * 4,
        "index_k_projected": rows * parsed_dim * 4,
        "index_query": rows * parsed_heads * parsed_dim * 4,
    }
    for name, needed in required.items():
        buffer = getattr(scratch, name)
        if buffer.nbytes < needed:
            raise ValueError(
                f"Qwen4Exp QSA scratch {name} holds {buffer.nbytes} bytes, needs "
                f"{needed} for {rows} index rows"
            )
    if selection.rows < rows:
        raise ValueError(
            f"selection scratch holds {selection.rows} rows, needs {rows}"
        )
    if (
        selection.index_heads != parsed_heads
        or selection.index_dim != parsed_dim
    ):
        raise ValueError(
            "selection scratch query geometry does not match the index state "
            f"({parsed_heads} heads x {parsed_dim})"
        )
    if selection.stride < index_state.output_stride:
        raise ValueError(
            f"selection stride {selection.stride} does not cover the index owner's "
            f"output stride {index_state.output_stride}"
        )

    active = attention_state.active
    index_active = index_state.active
    if index_active is None:
        raise RuntimeError(
            "Qwen4Exp QSA index state has no active rows; prepare the same slots as "
            "the attention state first"
        )
    if index_active.slot_ids != active.slot_ids:
        raise ValueError(
            "QSA index state active slots do not match the attention state rows; "
            "prepare both owners with the same slot list in the same order"
        )
    slots = index_active.slots
    cursors = index_state.cursor_host[slots]
    expected = attention_state.position_host[:rows].astype(np.int64)
    if not np.array_equal(cursors, expected):
        first = int(np.nonzero(cursors != expected)[0][0])
        raise ValueError(
            f"row {first} slot {int(slots[first])} index cursor {int(cursors[first])} "
            f"does not match the attention position {int(expected[first])}; append the "
            "index key in the same step as the KV write"
        )
    beyond = np.nonzero(cursors >= index_state.capacity)[0]
    if beyond.size:
        first = int(beyond[0])
        raise ValueError(
            f"row {first} slot {int(slots[first])} index cursor "
            f"{int(cursors[first])} reached the per-request capacity "
            f"{index_state.capacity}"
        )
    blocks = (cursors + 1) // index_state.compression_ratio
    sparse = tuple(int(row) for row in np.nonzero(blocks > index_state.block_budget)[0])
    dense = tuple(int(row) for row in np.nonzero(blocks <= index_state.block_budget)[0])
    for row in dense:
        context = int(attention_state.context_host[row])
        if context > index_state.dense_equivalent_limit:
            raise ValueError(
                f"row {row} context {context} is inside the index budget but exceeds "
                f"the dense-equivalence bound {index_state.dense_equivalent_limit}"
            )
        if context > selection.stride:
            raise ValueError(
                f"row {row} dense selection of {context} positions does not fit the "
                f"selection stride {selection.stride}"
            )
    return dense, sparse


def _validate(
    weights: Qwen4ExpQSAMixerDeviceWeights,
    *,
    attention_state: Qwen4ExpBatchAttentionState,
    scratch: Qwen4ExpQSAScratch,
    rows: int,
    hidden: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    eps: float,
    dense_equivalent_limit: int,
    rope_positions_ptr: int | None,
    attention_context_limit: int | None,
    runtime: HipRuntime | None,
    require_dense: bool = True,
) -> tuple[HipRuntime, int, int, int, int]:
    """Admit one batch mixer call before any dispatch touches the cache."""

    if scratch.closed or attention_state.closed:
        raise RuntimeError("Qwen4Exp QSA scratch/state is closed")
    if attention_state.closing:
        raise RuntimeError(
            "Qwen4Exp QSA attention state is closing; its buffers may already be released"
        )
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime or active_runtime is not attention_state.runtime:
        raise ValueError("runtime must match QSA scratch and attention state owners")

    count = _positive(rows, "rows")
    parsed_hidden = _positive(hidden, "hidden")
    parsed_query_heads = _positive(query_heads, "query_heads")
    parsed_kv_heads = _positive(kv_heads, "kv_heads")
    parsed_head_dim = _positive(head_dim, "head_dim")
    parsed_rotary = _positive(rotary_dim, "rotary_dim")
    if parsed_rotary > parsed_head_dim or parsed_rotary % 2:
        raise ValueError("rotary_dim must be even and <= head_dim")
    if not _is_finite_positive(theta, "theta"):
        raise ValueError("theta must be a finite positive number")
    if not _is_finite_positive(eps, "eps"):
        raise ValueError("eps must be a finite positive number")
    if parsed_query_heads % parsed_kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if parsed_head_dim > _DENSE_DECODE_MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim {parsed_head_dim} exceeds the dense decode bound "
            f"{_DENSE_DECODE_MAX_HEAD_DIM}"
        )
    if parsed_kv_heads != attention_state.kv_heads or parsed_head_dim != attention_state.head_dim:
        raise ValueError(
            "QSA head geometry must match the attention state cache "
            f"({attention_state.kv_heads} kv heads x {attention_state.head_dim})"
        )
    if attention_state.block_size != _DENSE_DECODE_BLOCK_SIZE:
        raise ValueError(
            "dense QSA decode requires block_size "
            f"{_DENSE_DECODE_BLOCK_SIZE}, state has {attention_state.block_size}"
        )

    missing = sorted(set(_REQUIRED_PROJECTIONS) - set(weights.projections))
    if missing:
        raise ValueError("missing Qwen4Exp QSA weights: " + ", ".join(missing))
    if weights.q_norm_weight_ptr == 0 or weights.k_norm_weight_ptr == 0:
        raise ValueError("Qwen4Exp QSA Q/K norm weights are required")
    if rope_positions_ptr is not None:
        _require_int64_positions(rope_positions_ptr)

    active = attention_state.active
    if active is None:
        raise RuntimeError(
            "Qwen4Exp batch attention state has no active rows; call prepare first"
        )
    if count != active.rows:
        raise ValueError(
            f"mixer rows {count} does not match the {active.rows} rows prepared on "
            "the attention state"
        )
    if count > attention_state.state_slots:
        raise ValueError(
            f"mixer rows {count} exceeds the {attention_state.state_slots} state slots"
        )
    context_limit = (
        active.max_context_len
        if attention_context_limit is None
        else _positive(attention_context_limit, "attention_context_limit")
    )
    if context_limit < active.max_context_len:
        raise ValueError(
            f"attention context limit {context_limit} does not cover the prepared "
            f"context {active.max_context_len}"
        )
    if context_limit > attention_state.max_positions:
        raise ValueError("QSA attention context limit must fit attention capacity")
    needed_blocks = (context_limit + attention_state.block_size - 1) // attention_state.block_size
    if needed_blocks > attention_state.blocks_per_request:
        raise ValueError(
            f"context {context_limit} needs {needed_blocks} blocks but each row table "
            f"holds {attention_state.blocks_per_request}"
        )
    if active.decode_spans.live_counts.numel != count:
        raise ValueError("attention state context vector does not cover the mixer rows")
    if active.append_spans.base_offsets.numel != count * attention_state.blocks_per_request:
        raise ValueError("attention state page table does not cover the mixer rows")

    q_width = parsed_query_heads * parsed_head_dim
    kv_width = parsed_kv_heads * parsed_head_dim
    limit = _positive(dense_equivalent_limit, "dense_equivalent_limit")
    # ``prepare`` publishes exactly ``count`` contexts in row order into the host
    # mirror, so this is the same vector the decode reads as live counts. The
    # mixed dense/sparse mixer admits rows past the bound, so it disables only
    # this check and keeps every other admission step identical.
    if require_dense:
        for row, context in enumerate(attention_state.context_host[:count]):
            if int(context) > limit:
                raise ValueError(
                    f"row {row} context {int(context)} exceeds the dense-equivalent "
                    f"limit {limit}; sparse index selection is not implemented in "
                    "this mixer, so that request must be routed through the sparse path"
                )
    _require_scratch(scratch, count, q_width, kv_width, parsed_hidden)
    _require_projection_geometry(
        weights,
        {
            "attn_q": (q_width * 2, parsed_hidden),
            "attn_k": (kv_width, parsed_hidden),
            "attn_v": (kv_width, parsed_hidden),
            "attn_output": (parsed_hidden, q_width),
        },
    )
    return active_runtime, count, q_width, kv_width, context_limit


def _require_projection_geometry(
    weights: Qwen4ExpQSAMixerDeviceWeights,
    expected: dict[str, tuple[int, int]],
) -> None:
    """Reject a projection whose stored shape cannot produce the staged layout.

    The output projection runs last, after the KV write, so a shape mismatch there
    would otherwise leave a token in the cache with no result. Weight objects that
    do not expose a source tensor shape keep the dispatch's own check.
    """

    for slot, (out_features, in_features) in expected.items():
        spec = getattr(weights.projections[slot], "spec", None)
        shape = getattr(getattr(spec, "source", None), "shape", None)
        if shape is None:
            continue
        parsed = tuple(int(dimension) for dimension in shape)
        if parsed != (out_features, in_features):
            raise ValueError(
                f"{slot} weight shape {parsed} does not match the expected "
                f"({out_features}, {in_features}) projection geometry"
            )


def _require_int64_positions(pointer: object) -> None:
    """Reject an MRoPE position pointer the kernel could not read.

    ``..._split_norm_mrope_rows_f32`` reads ``positions[axis * rows + row]`` for
    three axes, so the caller's allocation must hold ``3 * rows`` int64 values.
    Only the pointer's own validity is checkable here.
    """

    value = _positive(pointer, "rope_positions_ptr")
    if value % 8:
        raise ValueError("rope_positions_ptr must be 8-byte aligned for int64 positions")


def _require_scratch(
    scratch: Qwen4ExpQSAScratch,
    rows: int,
    q_width: int,
    kv_width: int,
    hidden: int,
) -> None:
    """Reject a scratch that cannot hold ``rows`` of this head geometry."""

    required = {
        "q_projected": (rows * q_width * 2, scratch.q_projected),
        "key_projected": (rows * kv_width, scratch.key_projected),
        "value_projected": (rows * kv_width, scratch.value_projected),
        "query": (rows * q_width, scratch.query),
        "key": (rows * kv_width, scratch.key),
        "gate": (rows * q_width, scratch.gate),
        "context": (rows * q_width, scratch.context),
        "gated": (rows * q_width, scratch.gated),
        "output": (rows * hidden, scratch.output),
    }
    for name, (elements, buffer) in required.items():
        needed = elements * 4
        if buffer.nbytes < needed:
            raise ValueError(
                f"Qwen4Exp QSA scratch {name} holds {buffer.nbytes} bytes, needs "
                f"{needed} for {rows} rows"
            )


def _selection_products(
    *, rows: int, stride: int, index_heads: int, index_dim: int
) -> dict[str, int]:
    """Byte sizes of the selection slabs, checked before any allocation.

    ``rows * stride * 8`` also bounds the per-row offsets ``query_row_ptr`` and
    the kernel's row addressing, so one addressability check covers both.
    """

    limit = np.iinfo(np.int64).max
    products = {
        "positions": rows * stride * 8,
        "counts": rows * 4,
        "queries": rows * index_heads * index_dim * 4,
    }
    for name, nbytes in products.items():
        if nbytes <= 0 or nbytes > limit:
            raise ValueError(
                f"selection {name} byte size {nbytes} is not addressable"
            )
    return products


def _free_pending(pending: list[DeviceBuffer], runtime: HipRuntime) -> None:
    """Free ``pending`` newest-first, popping only after each free returns."""

    while pending:
        free(pending[-1], runtime=runtime)
        pending.pop()


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
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


def batch_qsa_attention_implementation(
    *, head_dim: int = 0, block_size: int = 0,
    query_heads: int = 0, kv_heads: int = 0,
) -> tuple[Callable[..., Any], str]:
    """Select an exact rows kernel and its existing registry variant by shape."""

    if (
        head_dim == 256 and block_size == 256
        and query_heads > 0 and kv_heads > 0
        and query_heads % (2 * kv_heads) == 0
    ):
        return (
            qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32,
            "strict_h256_head_pair_rows_spans",
        )
    return qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32, "strict_rows_spans"


def batch_qsa_call_table(
    *, attention_geometry: dict[str, int] | None = None,
) -> dict[str, Callable[..., Any]]:
    """The exact direct callables this module uses for one mixed QSA step.

    The mixed path attends every row through the variable-selection rows
    kernel. The dense paged decode kernel is not called by this module and is
    therefore not part of the batch claim. Projections are bound separately as
    row-dependent registry dispatch records.
    """

    return {
        "qsa_mixed_mixer": run_qwen4_exp_batch_qsa_mixed_token_mixer,
        "qsa_split_norm_rope": qwen4_exp_qsa_split_norm_rope_rows_f32,
        "qsa_split_norm_mrope": qwen4_exp_qsa_split_norm_mrope_rows_f32,
        "qsa_paged_kv_write": qwen35_write_paged_kv_f32_batch_spans,
        "qsa_rows_attention": batch_qsa_attention_implementation(
            **(attention_geometry or {})
        )[0],
        "qsa_gate_context": qwen4_exp_qsa_gate_context_f32,
    }
