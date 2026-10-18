"""Independent-request dense QSA token mixer over the batch attention owner.

``run_qwen4_exp_dense_qsa_token_mixer`` in ``runtime/qwen4_exp_runner.py`` is the
strict single-row oracle: one request, one ``Qwen4ExpDenseAttentionState``, one
int64 position scalar. This module is the N-request form of the same dense
sub-budget arithmetic on top of ``Qwen4ExpBatchAttentionState``: every row is an
independent request with its own slot, page table, and position, and each stage
is a single N-row dispatch instead of a loop of single-row mixers.

Stages, in order, one dispatch each (identical to the oracle's dense path):

1. ``attn_q`` / ``attn_k`` / ``attn_v`` GGUF projections over the N-row hidden.
2. ``qwen4_exp_qsa_split_norm_rope_rows_f32`` for Q/K norm, split-half RoPE,
   and the gate split, reading the positions ``prepare`` uploaded.
3. ``qwen35_write_paged_kv_f32_batch_spans`` into the per-row page tables.
4. ``qwen35_paged_full_attn_decode_context_bf16_batch_spans`` with per-row
   context lengths.
5. ``qwen4_exp_qsa_gate_context_f32`` over ``rows * query_heads * head_dim``.
6. ``attn_output`` GGUF projection back to the N-row hidden.

Scope: the *dense* path only. Sparse index selection is not implemented here;
it is a single-position construction (one query position against a global key
space) with no N-row form yet. ``dense_equivalent_limit`` is therefore a
required argument rather than an optional hint: the caller states the context
length up to which dense attention is known to match the sparse path (the bound
the runner's index state carries), and every prepared row must fall inside it.
A call whose rows exceed the bound is rejected before the first dispatch, so an
integration cannot silently compute dense attention where sparse selection is
required. Passing a bound that covers the rows is how a caller explicitly asks
for diagnostic dense attention.

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

from hipengine.core.hip import HipRuntime
from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_batch_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_f32_batch_spans,
)
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_gate_context_f32,
    qwen4_exp_qsa_split_norm_mrope_rows_f32,
    qwen4_exp_qsa_split_norm_rope_rows_f32,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
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
    # mirror, so this is the same vector the decode reads as live counts.
    for row, context in enumerate(attention_state.context_host[:count]):
        if int(context) > limit:
            raise ValueError(
                f"row {row} context {int(context)} exceeds the dense-equivalent limit "
                f"{limit}; sparse index selection is not implemented in this mixer, so "
                "that request must be routed through the sparse path"
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
