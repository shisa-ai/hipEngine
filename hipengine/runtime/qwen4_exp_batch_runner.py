"""Independent-request full-model decode for the Qwen4Exp GGUF target.

One :class:`Qwen4ExpGGUFBatchRunner` owns a fixed set of physical state slots
and executes one complete Qwen4Exp decode step for N independent requests in
one layer traversal: embedding lookup, every PLE/GDN/QSA layer, and the
norm/head projection. It shares the immutable resident weights with the c1
runner, reuses the same GR/MoE/projection helpers, and adds no serving path.

What this module is not
-----------------------

* It does not touch the generation engine, the server, or any default. The
  committed ``decode_batch`` path still loops c1 runners.
* It does not claim production numerics or speed. The QSA mixer here is the
  mixed dense/sparse rows mixer, the GDN mixer is the indexed decode owner, and
  the PLE stage is the indexed-conv owner, so every stage is a *different*
  arithmetic path from the c1 single-request kernels. Deviations against c1 are
  measured by ``tests/test_gpu_qwen4exp_batch_runner.py``; the calibrated
  production-profile gate (``docs/EXECUTION-PROFILES.md``) is separate work.
* It does not express MTP, speculative rows, or prefill. A slot holds exactly
  one live token per step, and prefill stays on the c1 runner, which is what
  :meth:`Qwen4ExpGGUFBatchRunner.import_prefilled` consumes.

Ownership
---------

State is addressed by *physical slot*, not by request or batch row. ``slot_ids``
is the caller's row order and may be any permutation of the admitted slots, so
row ``r`` reads and writes only slot ``slot_ids[r]``. A slot must be admitted
before any step touches it, and release requires the owner token that admission
returned, so a stale handle cannot free another request's state. Admission and
release both zero the slot, which is the only state a fresh request may see.

Step semantics
--------------

``step`` validates the whole row set (slots, ownership, positions, token ids,
paged and index capacity) before the first mutation, and a rejected call leaves
every slot healthy. The stages that follow are not transactional: a failure
after the index append leaves the affected slots' device state partially
advanced and their PLE n-gram history advanced. So a failed step *poisons* its
rows, and a poisoned slot refuses ``step`` and ``import_prefilled`` until the
caller resets it (:meth:`reset_slot`) or releases and re-admits it. Slots that
were not rows of the failed call stay usable. A retry of the failed row set
therefore cannot silently continue corrupted state.

A row set that names slots explicitly must also name each slot's owner token,
so a scheduler holding a stale handle cannot advance a request that was
released and re-admitted into the same physical slot.

Per-step metadata
-----------------

Every QSA layer needs its own position, context, and index query position per
active row. Those are packed into one buffer (:class:`Qwen4ExpBatchStepMetadata`)
that each owner publishes through stable, non-owning views, so a step performs
one host-to-device copy for all of them instead of three per layer. The views
have a fixed address for the buffer's lifetime, which is what a captured graph
replays against. The owners' own block tables and slot-index vectors keep their
unchanged-content upload suppression.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.cpu_reference.qwen4_exp import PLEHashState, ple_hash_rows
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
    qwen4_exp_repeat_bf16_branches,
)
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32_rows_i32,
    lm_head_argmax_stage1_blocks,
)
from hipengine.kernels.registry import resolve
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
)
from hipengine.runtime.moe_graph import MoeGraphCache
from hipengine.runtime.qwen4_exp_batch_gdn import (
    Qwen4ExpBatchGdnState,
    run_qwen4_exp_batch_gdn_input_projections,
    run_qwen4_exp_batch_gdn_output_projection,
    run_qwen4_exp_batch_gdn_recurrence,
    validate_qwen4_exp_batch_gdn_mixer,
)
from hipengine.runtime.qwen4_exp_batch_ple import (
    Qwen4ExpBatchPleState,
    run_qwen4_exp_batch_ple_stage,
)
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_qsa_attention,
    run_qwen4_exp_batch_qsa_input_projections,
    run_qwen4_exp_batch_qsa_output_projection,
    validate_qwen4_exp_batch_qsa_mixed_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import (
    Qwen4ExpBatchQsaIndexRowPlan,
    Qwen4ExpBatchQsaIndexState,
)
from hipengine.runtime.qwen4_exp_batch_state import (
    Qwen4ExpBatchAttentionRowPlan,
    Qwen4ExpBatchAttentionState,
    Qwen4ExpBatchStepMetadata,
)
from hipengine.runtime.qwen4_exp_runner import (
    _qwen4_exp_q8_tile_scope,
    Qwen4ExpGDNLayerDeviceWeights,
    Qwen4ExpGDNLayerScratch,
    Qwen4ExpGRScratch,
    Qwen4ExpPLEScratch,
    Qwen4ExpQSALayerDeviceWeights,
    Qwen4ExpQSALayerScratch,
    bind_qwen4_exp_gdn_layer,
    bind_qwen4_exp_qsa_layer,
    qwen4_exp_gr_write_bf16_bf16,
    qwen4_exp_gr_write_bf16_f32,
    run_qwen4_exp_gr_read,
    run_qwen4_exp_moe,
    stage_qwen4_exp_ple_rows,
)

#: Paged KV block size, matching the c1 runner and the production page table.
QWEN4_EXP_BATCH_BLOCK_SIZE = 256


@dataclass(frozen=True)
class Qwen4ExpBatchDecodeResult:
    """One N-row decode step's output, indexed by batch row.

    ``tokens[r]`` is row ``r``'s greedy top-1 id and ``logits`` is the
    ``[rows, vocab_size]`` FP32 logit matrix when the caller asked for it. Rows
    are in the caller's ``slot_ids`` order, so ``slot_ids[r]`` names the slot
    whose state this row advanced.
    """

    slot_ids: tuple[int, ...]
    positions: np.ndarray
    tokens: np.ndarray
    logits: np.ndarray | None

    @property
    def rows(self) -> int:
        return len(self.slot_ids)


@dataclass(frozen=True)
class _StepPlan:
    rows: int
    slots: tuple[int, ...]
    positions: np.ndarray


@dataclass(frozen=True)
class _StepOwners:
    """Every QSA owner's validated row set for one step, in layer order.

    Built before the step's first device write, so a rejected row set cannot
    leave a packed metadata buffer uploaded or an owner's active rows published.
    """

    attention: tuple[Qwen4ExpBatchAttentionRowPlan, ...]
    index: tuple[Qwen4ExpBatchQsaIndexRowPlan, ...]


def _release_owners(owners: list[object], runtime: HipRuntime) -> None:
    """Free buffers and close owners from the end, leaving a failure pending."""

    while owners:
        owner = owners[-1]
        if isinstance(owner, DeviceBuffer):
            free(owner, runtime=runtime)
        else:
            owner.close()
        owners.pop()


class Qwen4ExpMixedTargetScratch:
    """Reusable stateless scratch for one mixed target pass.

    A mixed target pass is built and closed once per serving cycle. Its stateless
    half -- the token/embedding/residual/logits slabs and the per-layer GDN, QSA
    and head stage scratch -- depends only on the physical row count and the
    model geometry, never on which requests a layout holds, and every byte of it
    is rewritten before it is read. Allocating that set per pass charged every
    cycle a device allocation and free for buffers whose contents were already
    dead.

    The batch runner keeps one such set alive and every pass borrows it. A pass
    wider than ``capacity_rows`` replaces the set with a larger one, so retained
    memory is a single capacity-bounded set rather than a per-layout cache. The
    pass's provisional state -- depth-wave adapters, their per-row state slabs,
    and the QSA runs -- stays with the pass and is never cached here.

    ``borrow``/``release`` make that reuse a single-owner lease: the row slabs
    hold one pass's rows, so a second live pass would overwrite the first's
    token, residual, logit and stage bytes mid-verification. A second borrow is
    therefore refused whatever its row count, and growth is refused while the
    lease is held, because replacing the set would free buffers that pass still
    names. A borrow is dropped even by a close that leaves the pass retryable,
    because a pass that refuses work can never read those buffers again. A pass
    is expected to close before the runner: the runner's close releases the set
    regardless, and a pass over a closed runner refuses work.
    """

    def __init__(
        self,
        *,
        capacity_rows: int,
        token_ids: DeviceBuffer,
        embedding: DeviceBuffer,
        ple_embedding: DeviceBuffer,
        residual: DeviceBuffer,
        logits: DeviceBuffer,
        gdn: Qwen4ExpGDNLayerScratch,
        qsa: Qwen4ExpQSALayerScratch,
        head: Qwen4ExpGRScratch,
        runtime: HipRuntime,
    ) -> None:
        self.capacity_rows = int(capacity_rows)
        self.token_ids = token_ids
        self.embedding = embedding
        self.ple_embedding = ple_embedding
        self.residual = residual
        self.logits = logits
        self.gdn = gdn
        self.qsa = qsa
        self.head = head
        self.runtime = runtime
        self.borrowers = 0
        self.closed = False
        self.closing = False
        self._owned: list[object] = [
            token_ids, embedding, ple_embedding, residual, logits, gdn, qsa, head,
        ]

    @classmethod
    def allocate(
        cls, *, rows: int, config: object, runtime: HipRuntime
    ) -> "Qwen4ExpMixedTargetScratch":
        """Allocate one capacity-``rows`` set, releasing partials on failure."""

        capacity = int(rows)
        if capacity <= 0:
            raise ValueError("mixed target scratch rows must be positive")
        owners: list[object] = []
        try:
            token_ids = malloc(capacity * 8, runtime=runtime)
            owners.append(token_ids)
            embedding = malloc(capacity * config.hidden_size * 2, runtime=runtime)
            owners.append(embedding)
            ple_embedding = malloc(capacity * config.hidden_size * 4, runtime=runtime)
            owners.append(ple_embedding)
            residual = malloc(capacity * config.residual_width * 2, runtime=runtime)
            owners.append(residual)
            logits = malloc(capacity * config.vocab_size * 4, runtime=runtime)
            owners.append(logits)
            gdn = Qwen4ExpGDNLayerScratch.allocate(
                rows=capacity,
                branches=config.residual_branch_count,
                hidden=config.hidden_size,
                low_rank=config.residual_low_rank,
                qkv_width=(
                    2 * config.gdn_group_count * config.gdn_state_size
                    + config.gdn_inner_size
                ),
                core_width=config.gdn_inner_size,
                scalar_width=config.gdn_time_step_rank,
                ffn=config.expert_feed_forward_length,
                experts=config.expert_count,
                top_k=config.expert_used_count,
                runtime=runtime,
            )
            owners.append(gdn)
            qsa = Qwen4ExpQSALayerScratch.allocate(
                rows=capacity,
                branches=config.residual_branch_count,
                hidden=config.hidden_size,
                low_rank=config.residual_low_rank,
                query_heads=config.attention_head_count,
                kv_heads=config.attention_kv_head_count,
                head_dim=config.attention_key_length,
                index_heads=config.indexer_head_count,
                index_dim=config.indexer_key_length,
                ffn=config.expert_feed_forward_length,
                experts=config.expert_count,
                top_k=config.expert_used_count,
                runtime=runtime,
            )
            owners.append(qsa)
            head = Qwen4ExpGRScratch.allocate(
                rows=capacity,
                branches=config.residual_branch_count,
                hidden=config.hidden_size,
                low_rank=config.residual_low_rank,
                runtime=runtime,
            )
            owners.append(head)
        except BaseException as error:
            try:
                _release_owners(owners, runtime)
            except Exception as cleanup:
                error.add_note(
                    f"mixed target scratch allocation cleanup failed: {cleanup!r}"
                )
            raise
        return cls(
            capacity_rows=capacity,
            token_ids=token_ids,
            embedding=embedding,
            ple_embedding=ple_embedding,
            residual=residual,
            logits=logits,
            gdn=gdn,
            qsa=qsa,
            head=head,
            runtime=runtime,
        )

    def borrow(self) -> "Qwen4ExpMixedTargetScratch":
        """Take the single-owner lease on the row slabs, or refuse.

        The set holds one pass's rows at a time, so a second live pass is
        rejected before it allocates or writes anything: sharing the slabs would
        have both passes reading and overwriting the same token, residual, logit
        and stage bytes. The refusal does not depend on the requested row count,
        because a narrower pass aliases the same buffers.
        """

        if self.closed or self.closing:
            raise RuntimeError("mixed target scratch is closed")
        if self.borrowers:
            raise RuntimeError(
                "mixed target scratch is already leased by a live mixed target "
                "pass; a second pass would overwrite the row slabs it is using"
            )
        self.borrowers = 1
        return self

    def release(self) -> None:
        """Drop the lease; the batch runner keeps the buffers for the next pass."""

        if self.borrowers <= 0:
            raise RuntimeError("mixed target scratch released without a borrow")
        self.borrowers = 0

    def close(self) -> None:
        """Release every buffer, retryable: a failed free stays pending."""

        if self.closed:
            return
        self.closing = True
        _release_owners(self._owned, self.runtime)
        self.closed = True


class _BufferGroup:
    """Own plain device buffers with a retry-safe reverse free."""

    def __init__(self, buffers: Sequence[DeviceBuffer], runtime: HipRuntime) -> None:
        self.buffers = list(buffers)
        self.runtime = runtime
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        # Pop only after the free returns: a raising free leaves the remaining
        # slabs reachable, and a retry frees each exactly once.
        while self.buffers:
            free(self.buffers[-1], runtime=self.runtime)
            self.buffers.pop()
        self.closed = True


def run_qwen4_exp_batch_gdn_layer(
    residual_ptr: int,
    weights: Qwen4ExpGDNLayerDeviceWeights,
    *,
    state: Qwen4ExpBatchGdnState,
    scratch: Qwen4ExpGDNLayerScratch,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    conv_kernel: int,
    ffn: int,
    experts: int,
    top_k: int,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    graph_cache: MoeGraphCache | None = None,
    graph_key: Hashable | None = None,
    q51_down_m1: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
) -> DeviceBuffer:
    """Execute one complete N-row GDN+MoE layer over request-owned state slots.

    The GR reads/writes and the MoE are the committed row-generic helpers the c1
    runner uses; only the mixer owner changes (indexed decode instead of the
    single-request owner). ``state.prepare(slot_ids)`` must already have admitted
    the rows.

    With ``graph_cache`` the layer runs as two captured segments around the
    eager indexed conv/recurrence: the GR read plus the four input projections,
    then the output projection, GR write, FFN read, MoE, and GR write. Both
    segments read only fixed scratch and weight pointers and their launch
    geometry depends only on ``rows``, which is part of ``graph_key``; the
    stateful mixer core never enters a graph. The explicit residual write below
    is the same arithmetic as the fused prefill write
    (``tests/test_gpu_qwen4exp_gr_write_norm.py``), and the batch runner's slot
    bound is far below the 1024-row threshold that selects it.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GDN layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match GDN layer scratch owner")
    plan = validate_qwen4_exp_batch_gdn_mixer(
        weights.mixer.projections,
        state=state,
        scratch=scratch.gdn,
        rows=rows,
        hidden=hidden,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim=head_dim,
        conv_kernel=conv_kernel,
        conv_weight_ptr=weights.mixer.conv_weight_ptr,
        dt_bias_ptr=weights.mixer.dt_bias_ptr,
        a_ptr=weights.mixer.a_ptr,
        norm_weight_ptr=weights.mixer.norm_weight_ptr,
        runtime=active_runtime,
    )
    graph = _layer_graph(graph_cache, graph_key, weights)

    def pre_mixer(segment_stream: int) -> None:
        run_qwen4_exp_gr_read(
            residual_ptr,
            weights.attention_gr.norm_weight_ptr,
            weights.attention_gr.down,
            weights.attention_gr.up,
            weights.attention_gr.inject,
            scratch.attention_gr,
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            stream=segment_stream, runtime=active_runtime,
        )
        run_qwen4_exp_batch_gdn_input_projections(
            plan,
            scratch.attention_gr.mixed.ptr,
            weights.mixer.projections,
            state=state,
            scratch=scratch.gdn,
            hidden=hidden,
            stream=segment_stream,
        )

    def post_mixer(segment_stream: int) -> None:
        mixer_output = run_qwen4_exp_batch_gdn_output_projection(
            plan,
            weights.mixer.projections,
            scratch=scratch.gdn,
            hidden=hidden,
            stream=segment_stream,
        )
        _run_layer_ffn(
            residual_ptr,
            mixer_output.ptr,
            weights,
            scratch,
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            ffn=ffn, experts=experts, top_k=top_k,
            stream=segment_stream, runtime=active_runtime,
            q51_down_m1=q51_down_m1,
            q51_fold128_prefill=q51_fold128_prefill,
            q51_fold_pair_prefill=q51_fold_pair_prefill,
            q51_exact_grouped_down=q51_exact_grouped_down,
            q51_wmma_down=q51_wmma_down,
            q51_out8=q51_out8,
            q4_exact_grouped_gate=q4_exact_grouped_gate,
            q4_exact_grouped_all=q4_exact_grouped_all,
            q4_out4=q4_out4,
            q4_tile_m=q4_tile_m,
            q4_tile_n=q4_tile_n,
            forkb_grouped_down=forkb_grouped_down,
            q8_down_row4_prefill=q8_down_row4_prefill,
            q8_down_bundle_prefill=q8_down_bundle_prefill,
        )

    _run_segment(
        graph, graph_key, "pre_mixer", pre_mixer,
        outputs=(
            (scratch.gdn.qkv.ptr, rows * plan.qkv_width * 4),
            (scratch.gdn.gate.ptr, rows * plan.core_width * 4),
            (scratch.gdn.alpha.ptr, rows * state.num_v_heads * 4),
            (scratch.gdn.beta.ptr, rows * state.num_v_heads * 4),
            (scratch.attention_gr.inject_logits.ptr, rows * branches * 4),
        ),
        stream=stream,
    )
    run_qwen4_exp_batch_gdn_recurrence(
        plan,
        state=state,
        scratch=scratch.gdn,
        conv_weight_ptr=weights.mixer.conv_weight_ptr,
        dt_bias_ptr=weights.mixer.dt_bias_ptr,
        a_ptr=weights.mixer.a_ptr,
        norm_weight_ptr=weights.mixer.norm_weight_ptr,
        stream=stream,
    )
    _run_segment(
        graph, graph_key, "post_mixer", post_mixer,
        outputs=((scratch.output.ptr, rows * branches * hidden * DType.BF16.itemsize),),
        mutable_inputs=((residual_ptr, rows * branches * hidden * DType.BF16.itemsize),),
        stream=stream,
    )
    return scratch.output


def _layer_graph(
    graph_cache: MoeGraphCache | None,
    graph_key: Hashable | None,
    weights: object,
) -> MoeGraphCache | None:
    """The cache a layer may capture into, or ``None`` for the eager path.

    An inject-free GR read memsets its inject logits on the null stream, which
    a capture would not record, so that rare layout stays eager exactly as the
    c1 layer graph keeps it.
    """

    if graph_cache is None or graph_key is None or not graph_cache.enabled:
        return None
    if weights.attention_gr.inject is None or weights.ffn_gr.inject is None:
        return None
    return graph_cache


def _run_segment(
    graph: MoeGraphCache | None,
    graph_key: Hashable | None,
    name: str,
    launch: Callable[[int], None],
    *,
    outputs: Sequence[tuple[int, int]],
    mutable_inputs: Sequence[tuple[int, int]] = (),
    stream: int,
    key: Hashable | None = None,
) -> None:
    """Replay a stateless segment, capturing and validating on first use.

    ``key`` names a unit that is not a layer (the output head); it defaults to
    ``(graph_key, name)``, the layer-segment convention.
    """

    if graph is None:
        launch(stream)
        return
    (first_ptr, first_bytes), *rest = outputs
    graph.run(
        (graph_key, name) if key is None else key,
        eager=launch,
        out_ptr=first_ptr,
        out_nbytes=first_bytes,
        out_regions=tuple(rest),
        mutable_inputs=tuple(mutable_inputs),
        stream=stream,
    )


def _run_layer_ffn(
    residual_ptr: int,
    mixer_output_ptr: int,
    weights: object,
    scratch: object,
    *,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    ffn: int,
    experts: int,
    top_k: int,
    stream: int,
    runtime: HipRuntime,
    q51_down_m1: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
) -> None:
    """GR write after the mixer, FFN GR read, MoE, and the final GR write.

    The eight expert-down decisions (``q51_down_m1``, the two Q5_1 fold rungs,
    the Q5_1 exact grouped down chain gate, its out8 fallback rung,
    ``forkb_grouped_down`` and the two Q8_0 row4 rungs) are the owning runner's
    plan-published selections, forwarded to the MoE dispatch unchanged. A batch
    tops out at ``QWEN4_EXP_MAX_BATCH_ROWS`` rows, so fold128 is reachable at
    exactly its own threshold and the rows>=512 rungs stay on their parents; the
    Q5_1 down M1
    and PF-1 fork-b decisions are reachable at every multi-row batch width, and
    the out8 fallback rung is live wherever the exact grouped Q5_1 down chain
    runs with the expert-grid family off.
    """

    attention_inject = scratch.attention_gr.inject_logits
    qwen4_exp_gr_write_bf16_f32(
        residual_ptr,
        mixer_output_ptr,
        attention_inject.ptr,
        scratch.after_attention.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=runtime,
    )
    ffn_read = run_qwen4_exp_gr_read(
        scratch.after_attention.ptr,
        weights.ffn_gr.norm_weight_ptr,
        weights.ffn_gr.down,
        weights.ffn_gr.up,
        weights.ffn_gr.inject,
        scratch.ffn_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=runtime,
    )
    moe_output = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        scratch=scratch.moe,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=runtime,
        q51_down_m1=q51_down_m1,
        q51_fold128_prefill=q51_fold128_prefill,
        q51_fold_pair_prefill=q51_fold_pair_prefill,
        q51_exact_grouped_down=q51_exact_grouped_down,
        q51_wmma_down=q51_wmma_down,
        q51_out8=q51_out8,
        q4_exact_grouped_gate=q4_exact_grouped_gate,
        q4_exact_grouped_all=q4_exact_grouped_all,
        q4_out4=q4_out4,
        q4_tile_m=q4_tile_m,
        q4_tile_n=q4_tile_n,
        forkb_grouped_down=forkb_grouped_down,
        q8_down_row4_prefill=q8_down_row4_prefill,
        q8_down_bundle_prefill=q8_down_bundle_prefill,
    )
    qwen4_exp_gr_write_bf16_bf16(
        scratch.after_attention.ptr,
        moe_output.output.ptr,
        ffn_read.inject_logits.ptr,
        scratch.output.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=runtime,
    )


def run_qwen4_exp_batch_qsa_layer(
    residual_ptr: int,
    weights: Qwen4ExpQSALayerDeviceWeights,
    *,
    attention_state: Qwen4ExpBatchAttentionState,
    index_state: Qwen4ExpBatchQsaIndexState,
    selection: Qwen4ExpBatchQsaSelectionScratch,
    scratch: Qwen4ExpQSALayerScratch,
    rows: int,
    branches: int,
    hidden: int,
    low_rank: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    theta: float,
    index_heads: int,
    index_dim: int,
    index_rotary_dim: int,
    ffn: int,
    experts: int,
    top_k: int,
    rope_positions_ptr: int | None = None,
    attention_context_limit: int | None = None,
    stream: int = 0,
    runtime: HipRuntime | None = None,
    graph_cache: MoeGraphCache | None = None,
    graph_key: Hashable | None = None,
    q51_down_m1: bool = False,
    q51_fold128_prefill: bool = False,
    q51_fold_pair_prefill: bool = False,
    q51_exact_grouped_down: bool = True,
    q51_wmma_down: bool = False,
    q51_out8: bool = True,
    q4_exact_grouped_gate: bool = True,
    q4_exact_grouped_all: bool = True,
    q4_out4: bool = True,
    q4_tile_m: int | None = None,
    q4_tile_n: int | None = None,
    forkb_grouped_down: bool = False,
    q8_down_row4_prefill: bool = False,
    q8_down_bundle_prefill: bool = False,
) -> DeviceBuffer:
    """Execute one complete N-row QSA+MoE layer over request-owned slots.

    The mixer is the mixed dense/sparse rows owner: every row carries its own
    ``(positions, count)`` selection, so a batch may hold requests on either
    side of the index bound. ``attention_state.prepare`` and
    ``index_state.prepare`` must already have admitted the same slots in the
    same row order.

    With ``graph_cache`` the GR read plus the Q/K/V/index-key projections, and
    the output projection plus the FFN half, replay as captured segments. The
    index append, KV write, selection, and attention read per-row positions and
    host selections, so they stay eager on every step.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp QSA layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match QSA layer scratch owner")
    plan = validate_qwen4_exp_batch_qsa_mixed_mixer(
        weights.mixer,
        attention_state=attention_state,
        index_state=index_state,
        selection=selection,
        scratch=scratch.qsa,
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
        runtime=active_runtime,
    )
    graph = _layer_graph(graph_cache, graph_key, weights)

    def pre_mixer(segment_stream: int) -> None:
        run_qwen4_exp_gr_read(
            residual_ptr,
            weights.attention_gr.norm_weight_ptr,
            weights.attention_gr.down,
            weights.attention_gr.up,
            weights.attention_gr.inject,
            scratch.attention_gr,
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            stream=segment_stream, runtime=active_runtime,
        )
        run_qwen4_exp_batch_qsa_input_projections(
            plan,
            scratch.attention_gr.mixed.ptr,
            weights.mixer,
            scratch=scratch.qsa,
            hidden=hidden,
            index_dim=index_dim,
            stream=segment_stream,
        )

    def post_mixer(segment_stream: int) -> None:
        mixer_output = run_qwen4_exp_batch_qsa_output_projection(
            plan,
            weights.mixer,
            scratch=scratch.qsa,
            hidden=hidden,
            stream=segment_stream,
        )
        _run_layer_ffn(
            residual_ptr,
            mixer_output.ptr,
            weights,
            scratch,
            rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
            ffn=ffn, experts=experts, top_k=top_k,
            stream=segment_stream, runtime=active_runtime,
            q51_down_m1=q51_down_m1,
            q51_fold128_prefill=q51_fold128_prefill,
            q51_fold_pair_prefill=q51_fold_pair_prefill,
            q51_exact_grouped_down=q51_exact_grouped_down,
            q51_wmma_down=q51_wmma_down,
            q51_out8=q51_out8,
            q4_exact_grouped_gate=q4_exact_grouped_gate,
            q4_exact_grouped_all=q4_exact_grouped_all,
            q4_out4=q4_out4,
            q4_tile_m=q4_tile_m,
            q4_tile_n=q4_tile_n,
            forkb_grouped_down=forkb_grouped_down,
            q8_down_row4_prefill=q8_down_row4_prefill,
            q8_down_bundle_prefill=q8_down_bundle_prefill,
        )

    _run_segment(
        graph, graph_key, "pre_mixer", pre_mixer,
        outputs=(
            (scratch.qsa.q_projected.ptr, rows * plan.q_width * 2 * 4),
            (scratch.qsa.key_projected.ptr, rows * plan.kv_width * 4),
            (scratch.qsa.value_projected.ptr, rows * plan.kv_width * 4),
            (scratch.qsa.index_k_projected.ptr, rows * index_dim * 4),
            (scratch.attention_gr.mixed.ptr, rows * hidden * 4),
            (scratch.attention_gr.inject_logits.ptr, rows * branches * 4),
        ),
        stream=stream,
    )
    run_qwen4_exp_batch_qsa_attention(
        plan,
        scratch.attention_gr.mixed.ptr,
        weights.mixer,
        attention_state=attention_state,
        index_state=index_state,
        selection=selection,
        scratch=scratch.qsa,
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
        stream=stream,
    )
    _run_segment(
        graph, graph_key, "post_mixer", post_mixer,
        outputs=((scratch.output.ptr, rows * branches * hidden * DType.BF16.itemsize),),
        mutable_inputs=((residual_ptr, rows * branches * hidden * DType.BF16.itemsize),),
        stream=stream,
    )
    return scratch.output


class Qwen4ExpGGUFBatchRunner:
    """One full-model decode step for N independent Qwen4Exp requests.

    ``resident`` is the immutable weight residency shared with the c1 runner
    (``Qwen4ExpGGUFTextGenerator._resident`` or
    ``Qwen4ExpGGUFResidentModelRunner.resident``). This runner owns only state:
    the paged KV slabs, per-layer GDN recurrent/conv slots, PLE conv-history
    slots, per-layer index slabs, the PLE n-gram history, and N-row scratch.

    ``state_slots`` is the number of concurrently resident requests. Every slot
    costs one request's KV cache, so it is the serving-side capacity knob;
    ``max_positions`` is the per-request decode capacity and must be at least
    the index compression ratio (one complete index block).
    """

    # This owner can execute one shared target pass holding autoregressive rows
    # and prepared speculative chains, so serving may route a speculative batch
    # row here. Owners that do not declare it keep the c1 MTP path.
    supports_mixed_target = True

    # The expert-down decisions are the owning runner's plan-published
    # selections. They are class defaults so an owner built without the binder's
    # publish keeps every parent, exactly like an unbound c1 runner. A batch
    # tops out at ``QWEN4_EXP_MAX_BATCH_ROWS`` (64) rows: fold128 is reachable at
    # exactly its own threshold while the rows>=512 rungs stay on their parents,
    # and the Q5_1 down M1 and PF-1 fork-b decisions select from rows>=2, so they
    # are reachable here too. The exact grouped Q5_1 down chain gate and its
    # non-expertgrid fallback rung are the decisions every plan publishes on, so
    # their defaults are the production chain and the out8 owner.
    q51_exact_grouped_down: bool = True
    # The F16 WMMA down owner's admission, published off by the profile
    # binder: it is the down-only MX-1 diagnostic below that chain, and the
    # production grouped MoE route admits the owner on its own. Decode never
    # reaches the production term, so a captured segment that ran this owner
    # is keyed on it.
    q51_wmma_down: bool = False
    # The chain's non-expertgrid fallback rung: every plan publishes the out8
    # owner, and the output1 base is a diagnostic binding on one runner.
    q51_out8: bool = True
    # The Q4 gate/up chain's own gate and its `_ALL` joint term, published by
    # the profile binder: the chain runs on every Q4_K multi-row chunk unless
    # an arm closes the gate, and both decide what the captured segment
    # launches.
    q4_exact_grouped_gate: bool = True
    q4_exact_grouped_all: bool = True
    # The Q4 gate/up ladder's non-expertgrid fallback rung, published by the
    # profile binder: the captured MoE segment bakes its gate/up launch in, so
    # this belongs in the capture key like the Q5_1 rungs above.
    q4_out4: bool = True
    # The compact F16 Q4_K dual gate/up tile pair, published 16/16 by the
    # profile binder and ``None`` (the dispatch heuristic) on an unbound owner.
    # The MoE runs eager inside the captured segment, so a decode capture
    # records this launch and the pair belongs in the capture key.
    q4_tile_m: int | None = None
    q4_tile_n: int | None = None
    # The Q8_0 dense WMMA prefill tile pair, published (64, 32) by the profile
    # binder and ``None`` on an unbound owner. A step at rows>1 can reach that
    # family, so the pair is bound around the step and carried in the capture
    # key of the segment it builds.
    q8_wmma_tiles: tuple[int, int] | None = None
    q51_down_m1: bool = False
    q51_fold128_prefill: bool = False
    q51_fold_pair_prefill: bool = False
    forkb_grouped_down: bool = False
    q8_down_row4_prefill: bool = False
    q8_down_bundle_prefill: bool = False

    def __init__(
        self,
        resident: object,
        *,
        state_slots: int,
        max_positions: int,
        backend: str = "hip_gfx1151",
        block_size: int = QWEN4_EXP_BATCH_BLOCK_SIZE,
        runtime: HipRuntime | None = None,
        graph_segments: bool = True,
        pool_blocks: int | str | None = None,
        pool_reserve_bytes: int = 3 * 1024**3,
    ) -> None:
        plan = getattr(resident, "plan")
        config = plan.config
        slots = int(state_slots)
        positions = int(max_positions)
        if slots <= 0:
            raise ValueError("state_slots must be positive")
        if not 0 < positions <= int(config.context_length):
            raise ValueError(
                f"max_positions must be in 1..{int(config.context_length)}, got {positions}"
            )
        if positions < int(config.qsa_compression_ratio):
            raise ValueError(
                "max_positions must cover one complete index block "
                f"({int(config.qsa_compression_ratio)} positions)"
            )
        if int(block_size) <= 0:
            raise ValueError("block_size must be positive")
        self.resident = resident
        self.config = config
        self.backend = str(backend)
        self.runtime = runtime or get_hip_runtime()
        self.state_slots = slots
        self.max_positions = positions
        self.block_size = int(block_size)
        # ``None`` keeps the static layout (one full-context window per slot).
        # An integer or ``"auto"`` shares one physical block pool between the
        # slots; ``"auto"`` sizes it from the device memory left after every
        # other allocation, keeping ``pool_reserve_bytes`` free.
        if pool_blocks is not None and pool_blocks != "auto":
            pool_blocks = int(pool_blocks)
            if pool_blocks <= 0:
                raise ValueError("pool_blocks must be positive")
        self._pool_request = pool_blocks
        self._pool_reserve_bytes = int(pool_reserve_bytes)
        self.pooled = pool_blocks is not None
        blocks_per_request = (positions + int(block_size) - 1) // int(block_size)
        self._block_host = np.zeros((slots, blocks_per_request), dtype=np.int32)
        self._free_blocks: list[int] = []
        self._slot_blocks = [0] * slots
        self.closed = False
        self.closing = False
        self._owned: list[Any] = []
        self._buffers: _BufferGroup | None = None
        # The stateless half of a mixed target pass, kept at the widest capacity
        # admitted so far. Created on the first pass and released by ``close``.
        self._mixed_target_scratch: Qwen4ExpMixedTargetScratch | None = None
        self._owners = [0] * slots
        self._poisoned: set[int] = set()
        self._next_owner = 0
        self._positions = np.zeros(slots, dtype=np.int64)
        self._ple_hash_states: dict[int, PLEHashState] = {}
        self.gdn_bindings: Mapping[int, Qwen4ExpGDNLayerDeviceWeights] = {}
        self.qsa_bindings: Mapping[int, Qwen4ExpQSALayerDeviceWeights] = {}
        self.attention_states: tuple[Qwen4ExpBatchAttentionState, ...] = ()
        self.index_states: tuple[Qwen4ExpBatchQsaIndexState, ...] = ()
        # One packed buffer carries every QSA owner's dynamic per-step metadata,
        # so a step publishes it with one upload instead of three per layer.
        self.step_metadata: Qwen4ExpBatchStepMetadata | None = None
        self.gdn_states: tuple[Qwen4ExpBatchGdnState, ...] = ()
        self.ple_state: Qwen4ExpBatchPleState | None = None
        self.selection: Qwen4ExpBatchQsaSelectionScratch | None = None
        self.gdn_scratch: Qwen4ExpGDNLayerScratch | None = None
        self.qsa_scratch: Qwen4ExpQSALayerScratch | None = None
        self.ple_scratch: Qwen4ExpPLEScratch | None = None
        self.head_scratch: Qwen4ExpGRScratch | None = None
        self.token_ids: DeviceBuffer | None = None
        self.embedding: DeviceBuffer | None = None
        self.ple_embedding: DeviceBuffer | None = None
        self.residual: DeviceBuffer | None = None
        self.logits: DeviceBuffer | None = None
        self.argmax_ids: DeviceBuffer | None = None
        self.argmax_values: DeviceBuffer | None = None
        self.argmax_blocks_i32: DeviceBuffer | None = None
        self._ple_layers = frozenset(int(layer) for layer in config.ple_layers)
        # Stateless segments replay from captured graphs: per-layer segments are
        # keyed by ``(layer, rows)`` and the output head by
        # ``("head", rows, residual_ptr)``. The cache validates every capture
        # bit-exact against an eager run and keeps a rejected key eager.
        self._graphs: MoeGraphCache | None = None
        try:
            load_backend_kernel_package(self.backend)
            self.gdn_bindings = MappingProxyType(
                {
                    layer: bind_qwen4_exp_gdn_layer(resident, layer)
                    for layer, kind in enumerate(config.layer_types)
                    if kind == "gdn"
                }
            )
            self.qsa_bindings = MappingProxyType(
                {
                    layer: bind_qwen4_exp_qsa_layer(resident, layer)
                    for layer, kind in enumerate(config.layer_types)
                    if kind == "qsa"
                }
            )
            self._allocate()
            self._graphs = MoeGraphCache(self.runtime, enabled=bool(graph_segments))
        except BaseException:
            self.close()
            raise

    # -- construction -----------------------------------------------------

    def _allocate(self) -> None:
        cfg = self.config
        qkv_width = 2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size
        # One paged K/V cache per QSA layer, exactly as the c1 runner keeps one
        # ``Qwen4ExpDenseAttentionState`` per QSA layer. A single shared cache
        # would have every QSA layer append over the previous layer's keys at the
        # same physical block and position, which the one-QSA-layer synthetic
        # fixture cannot show and the real 12-QSA-layer model hits immediately.
        gdn_states = tuple(
            Qwen4ExpBatchGdnState.allocate(
                state_slots=self.state_slots,
                num_k_heads=cfg.gdn_group_count,
                num_v_heads=cfg.gdn_time_step_rank,
                head_dim=cfg.gdn_state_size,
                conv_kernel=cfg.gdn_conv_kernel,
                runtime=self.runtime,
            )
            for _ in range(cfg.layer_types.count("gdn"))
        )
        self.gdn_states = gdn_states
        self._owned.extend(gdn_states)
        if self._ple_layers:
            ple_state = Qwen4ExpBatchPleState.allocate(
                state_slots=self.state_slots,
                branches=cfg.residual_branch_count,
                hidden=cfg.hidden_size,
                conv_kernel=cfg.ple_conv_kernel,
                dilation=cfg.ple_ngram_size,
                runtime=self.runtime,
            )
            self.ple_state = ple_state
            self._owned.append(ple_state)
        selection = Qwen4ExpBatchQsaSelectionScratch.allocate(
            rows=self.state_slots,
            stride=(
                cfg.qsa_block_budget * cfg.qsa_compression_ratio
                + cfg.qsa_compression_ratio - 1
            ) if cfg.qsa_layer_count else 1,
            index_heads=cfg.indexer_head_count,
            index_dim=cfg.indexer_key_length,
            runtime=self.runtime,
        )
        self.selection = selection
        self._owned.append(selection)
        gdn_scratch = Qwen4ExpGDNLayerScratch.allocate(
            rows=self.state_slots,
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            qkv_width=qkv_width,
            core_width=cfg.gdn_inner_size,
            scalar_width=cfg.gdn_time_step_rank,
            ffn=cfg.expert_feed_forward_length,
            experts=cfg.expert_count,
            top_k=cfg.expert_used_count,
            runtime=self.runtime,
        )
        self.gdn_scratch = gdn_scratch
        self._owned.append(gdn_scratch)
        qsa_scratch = Qwen4ExpQSALayerScratch.allocate(
            rows=self.state_slots,
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            query_heads=cfg.attention_head_count,
            kv_heads=cfg.attention_kv_head_count,
            head_dim=cfg.attention_key_length,
            ffn=cfg.expert_feed_forward_length,
            experts=cfg.expert_count,
            top_k=cfg.expert_used_count,
            index_heads=cfg.indexer_head_count,
            index_dim=cfg.indexer_key_length,
            runtime=self.runtime,
        )
        self.qsa_scratch = qsa_scratch
        self._owned.append(qsa_scratch)
        if self._ple_layers:
            ple_scratch = Qwen4ExpPLEScratch.allocate(
                rows=self.state_slots,
                branches=cfg.residual_branch_count,
                hidden=cfg.hidden_size,
                runtime=self.runtime,
            )
            self.ple_scratch = ple_scratch
            self._owned.append(ple_scratch)
        head_scratch = Qwen4ExpGRScratch.allocate(
            rows=self.state_slots,
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            runtime=self.runtime,
        )
        self.head_scratch = head_scratch
        self._owned.append(head_scratch)
        # Allocated before the pooled K/V slabs so the pool is sized from the
        # memory every other fixed allocation left, and before the QSA owners so
        # its single buffer outlives every view they publish through.
        step_metadata = Qwen4ExpBatchStepMetadata.allocate(
            layers=cfg.qsa_layer_count,
            state_slots=self.state_slots,
            runtime=self.runtime,
        )
        self.step_metadata = step_metadata
        self._owned.append(step_metadata)
        argmax_blocks = lm_head_argmax_stage1_blocks(cfg.vocab_size, threads=256)
        buffers = _BufferGroup(
            [
                malloc(
                    self.state_slots * np.dtype(np.int64).itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    self.state_slots * cfg.hidden_size * DType.BF16.itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    self.state_slots * cfg.hidden_size * DType.FP32.itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    self.state_slots
                    * cfg.residual_branch_count
                    * cfg.hidden_size
                    * DType.BF16.itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    self.state_slots * cfg.vocab_size * DType.FP32.itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    self.state_slots * DType.INT32.itemsize,
                    runtime=self.runtime,
                ),
                # The row-wise argmax writes one stage-1 partial per row and
                # block, so the scratch scales with the slot count.
                malloc(
                    self.state_slots * argmax_blocks * DType.FP32.itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    self.state_slots * argmax_blocks * DType.INT32.itemsize,
                    runtime=self.runtime,
                ),
            ],
            self.runtime,
        )
        self._buffers = buffers
        (
            self.token_ids,
            self.embedding,
            self.ple_embedding,
            self.residual,
            self.logits,
            self.argmax_ids,
            self.argmax_values,
            self.argmax_blocks_i32,
        ) = buffers.buffers
        # The K/V and raw index-key slabs come last so a pooled layout can be
        # sized from the memory every other allocation left.
        pool_blocks = self._resolve_pool_blocks()
        attention_states = tuple(
            Qwen4ExpBatchAttentionState.allocate(
                state_slots=self.state_slots,
                max_positions=self.max_positions,
                block_size=self.block_size,
                kv_heads=cfg.attention_kv_head_count,
                head_dim=cfg.attention_key_length,
                runtime=self.runtime,
                pool_blocks=pool_blocks,
                block_host=self._block_host if pool_blocks is not None else None,
                position_view=step_metadata.views(layer).position,
                context_view=step_metadata.views(layer).context,
            )
            for layer in range(cfg.qsa_layer_count)
        )
        self.attention_states = attention_states
        self._owned.extend(attention_states)
        index_states = tuple(
            Qwen4ExpBatchQsaIndexState.allocate(
                attention_state=attention,
                index_heads=cfg.indexer_head_count,
                index_dim=cfg.indexer_key_length,
                compression_ratio=cfg.qsa_compression_ratio,
                block_budget=cfg.qsa_block_budget,
                runtime=self.runtime,
                position_view=step_metadata.views(layer).index_position,
            )
            for layer, attention in enumerate(attention_states)
        )
        self.index_states = index_states
        self._owned.extend(index_states)
        if self.pooled:
            self._free_blocks = list(range(int(attention_states[0].pool_blocks)))
        else:
            self._block_host = attention_states[0].block_host if attention_states else self._block_host
            self._slot_blocks = [self._blocks_per_request] * self.state_slots

    @property
    def _blocks_per_request(self) -> int:
        return int(self._block_host.shape[1])

    @staticmethod
    def pool_block_bytes(config: object, block_size: int = QWEN4_EXP_BATCH_BLOCK_SIZE) -> int:
        """Device bytes one pooled block costs across every QSA layer.

        One block holds ``block_size`` positions of BF16 K and V per QSA layer
        plus the matching FP32 raw index keys.
        """

        kv = 2 * int(config.attention_kv_head_count) * int(config.attention_key_length) * 2
        index = int(config.indexer_key_length) * DType.FP32.itemsize
        return int(block_size) * (kv + index) * int(config.qsa_layer_count)

    def _index_fixed_bytes(self) -> int:
        """Per-slot index slabs that do not live in the block pool."""

        cfg = self.config
        ratio = int(cfg.qsa_compression_ratio)
        complete = self.max_positions // ratio
        stride = int(cfg.qsa_block_budget) * ratio + ratio - 1
        per_layer = self.state_slots * (
            complete * ratio * 4
            + complete * 8
            + complete * int(cfg.indexer_key_length) * 4
            + complete * 4
            + 8
            + stride * 8
            + 4
        )
        return per_layer * int(cfg.qsa_layer_count)

    def _resolve_pool_blocks(self) -> int | None:
        request = self._pool_request
        if request is None:
            return None
        maximum = self.state_slots * self._blocks_per_request
        if request != "auto":
            return min(int(request), maximum)
        free_bytes, _total = self.runtime.mem_get_info()
        budget = int(free_bytes) - self._index_fixed_bytes() - self._pool_reserve_bytes
        blocks = budget // self.pool_block_bytes(self.config, self.block_size)
        if blocks < 1:
            raise MemoryError(
                "Qwen4Exp batch KV pool does not fit: "
                f"{int(free_bytes)} bytes free, {self._index_fixed_bytes()} needed "
                f"for {self.state_slots} slots' index state and "
                f"{self._pool_reserve_bytes} kept in reserve; lower the active "
                "request count or the context"
            )
        return int(min(blocks, maximum))

    @property
    def pool_blocks(self) -> int:
        """Physical blocks in the K/V slabs (shared in pooled mode)."""

        states = self.attention_states
        return int(states[0].pool_blocks) if states else 0

    @property
    def free_block_count(self) -> int:
        """Blocks a new admission can still take (pooled mode)."""

        if not self.pooled:
            return sum(
                self._blocks_per_request for owner in self._owners if not owner
            )
        return len(self._free_blocks)

    def blocks_for_positions(self, positions: int) -> int:
        """Paged blocks a request of ``positions`` positions holds."""

        needed = int(positions)
        if needed <= 0:
            raise ValueError("positions must be positive")
        return (needed + self.block_size - 1) // self.block_size

    def slot_capacity(self, slot: int) -> int:
        """Positions ``slot``'s assigned blocks hold (0 when not admitted)."""

        return self._slot_blocks[self._require_slot(slot)] * self.block_size

    @property
    def graph_stats(self) -> dict[str, int]:
        """Capture/replay/eager/reject counts of the stateless graph segments."""

        return {} if self._graphs is None else self._graphs.stats

    # -- slot ownership ---------------------------------------------------

    @property
    def active_slots(self) -> tuple[int, ...]:
        """Admitted slots in ascending slot order.

        This is the row set a ``step`` without an explicit ``slots`` argument
        advances, and its order is the physical slot order rather than the order
        the slots were admitted in.
        """

        return tuple(slot for slot, owner in enumerate(self._owners) if owner)

    @property
    def free_slots(self) -> tuple[int, ...]:
        return tuple(slot for slot, owner in enumerate(self._owners) if not owner)

    def owner_of(self, slot: int) -> int:
        """Owner token of ``slot``, or 0 when it is free."""

        return self._owners[self._require_slot(slot)]

    def position(self, slot: int) -> int:
        """Next append position of an admitted slot."""

        parsed = self._require_slot(slot)
        self._require_admitted(parsed)
        return int(self._positions[parsed])

    @property
    def ple_hash_states(self) -> Mapping[int, PLEHashState]:
        """Per-slot PLE n-gram history, keyed by physical slot."""

        return MappingProxyType(dict(self._ple_hash_states))

    @property
    def poisoned_slots(self) -> tuple[int, ...]:
        """Slots whose device state a failed call left partially advanced."""

        return tuple(sorted(self._poisoned))

    def is_poisoned(self, slot: int) -> bool:
        """Whether ``slot`` needs an explicit reset before it can be used."""

        return self._require_slot(slot) in self._poisoned

    def _require_healthy(self, slot: int) -> None:
        if slot in self._poisoned:
            raise RuntimeError(
                f"slot {slot} is poisoned by a failed step: its device state is "
                "partially advanced, so advancing it again would continue "
                "corrupted state. Call reset_slot(slot, owner) or release the "
                "slot and admit it again first."
            )

    def admit(self, slot: int, *, positions: int | None = None) -> int:
        """Take ownership of ``slot`` and zero the request state it holds.

        Returns a fresh owner token. The token is the only handle that can
        release the slot again, and it is never reused, so a handle from a
        previous request cannot free the current one.

        In pooled mode ``positions`` is the request's whole extent (prompt plus
        decode budget); that many positions of blocks are taken from the pool
        and held until release, so an admitted request can never run out of KV
        space. The static layout ignores it and gives every slot its window.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        if self._owners[parsed]:
            raise RuntimeError(
                f"slot {parsed} is already owned by request {self._owners[parsed]}"
            )
        if self.pooled:
            if positions is None:
                raise ValueError("a pooled batch runner admits a slot with its positions")
            needed = self.blocks_for_positions(positions)
            if needed > self._blocks_per_request:
                raise ValueError(
                    f"{int(positions)} positions exceed the per-slot capacity "
                    f"{self.max_positions}"
                )
            if needed > len(self._free_blocks):
                raise RuntimeError(
                    f"Qwen4Exp batch KV pool has {len(self._free_blocks)} free "
                    f"blocks; {int(positions)} positions need {needed}"
                )
            taken = self._free_blocks[:needed]
            del self._free_blocks[:needed]
            self._block_host[parsed, :] = 0
            self._block_host[parsed, :needed] = taken
            self._slot_blocks[parsed] = needed
            for index_state in self.index_states:
                index_state.map_slot(parsed, needed)
        elif positions is not None and int(positions) > self.max_positions:
            raise ValueError(
                f"{int(positions)} positions exceed the per-slot capacity "
                f"{self.max_positions}"
            )
        self._zero_slot(parsed)
        self._poisoned.discard(parsed)
        self._next_owner += 1
        self._owners[parsed] = self._next_owner
        self._positions[parsed] = 0
        self._ple_hash_states.pop(parsed, None)
        return self._next_owner

    def release(self, slot: int, owner: int) -> None:
        """Drop ownership of ``slot`` and zero its request state."""

        self._require_open()
        parsed = self._require_slot(slot)
        self._require_owner(parsed, owner)
        self._zero_slot(parsed)
        self._poisoned.discard(parsed)
        self._owners[parsed] = 0
        self._positions[parsed] = 0
        self._ple_hash_states.pop(parsed, None)
        if self.pooled:
            held = self._slot_blocks[parsed]
            self._free_blocks.extend(int(b) for b in self._block_host[parsed, :held])
            self._free_blocks.sort()
            self._block_host[parsed, :] = 0
            self._slot_blocks[parsed] = 0

    def reset_slot(self, slot: int, owner: int) -> None:
        """Zero an owned slot in place, keeping the owner token and position 0.

        This is the slot-reuse path for a request that is replaced without
        releasing the slot, and it is also the recovery path for a poisoned
        slot: the owner token stays valid because the physical owner did not
        change. The poison mark is cleared only after the zeroing succeeded, so
        a reset that fails leaves the slot refusing work.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        self._require_owner(parsed, owner)
        self._zero_slot(parsed)
        self._poisoned.discard(parsed)
        self._positions[parsed] = 0
        self._ple_hash_states.pop(parsed, None)

    def _zero_slot(self, slot: int) -> None:
        attention_states = self._require_attention_states()
        held = self._slot_blocks[slot]
        blocks = [int(block) for block in self._block_host[slot, :held]]
        for state in (*attention_states, *self.index_states, *self.gdn_states):
            state.reset_slot(slot, blocks=blocks)
        if self.ple_state is not None:
            self.ple_state.reset_slot(slot, blocks=blocks)

    # -- c1 hand-off ------------------------------------------------------

    def import_prefilled(self, slot: int, owner: int, *, source: object) -> int:
        """Seed ``slot`` from a prefilled c1 runner and return its position.

        ``source`` is a live :class:`Qwen4ExpGGUFResidentModelRunner` whose
        request was prefilled by the committed c1 path. The import copies, for
        this slot only, the paged K/V blocks, the raw index keys, every GDN
        layer's recurrent/conv row, the PLE conv history, and the PLE n-gram
        history. Pooled index keys are *not* copied: the slot's pooled count is
        reset so the next selection re-pools from the copied raw keys, which is
        the same derivation the c1 owner performs.

        The slot must already be admitted (``owner``) and untouched, the source
        must be open with an identical model config, and every copied source
        buffer must be a distinct allocation from its destination. All checks
        run before the first write, so a rejected import leaves the slot exactly
        as admission left it.

        Not supported, by design: a source with a different paged block size, a
        source whose index count disagrees with its position, and a source with
        a pending index rollback. Each raises a concrete error instead of
        approximating the copy.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        self._require_owner(parsed, owner)
        self._require_healthy(parsed)
        if int(self._positions[parsed]) != 0 or any(
            state.cursor_host[parsed] for state in self.index_states
        ):
            raise ValueError(
                f"slot {parsed} already holds request history; import requires a "
                "fresh slot"
            )
        if getattr(source, "closed", True):
            raise ValueError("import source runner is closed")
        source_config = getattr(source, "config", None)
        if source_config is None:
            raise ValueError("import source has no model config")
        self._require_matching_config(source_config)
        position = int(getattr(source, "position", -1))
        if position <= 0:
            raise ValueError(
                "import source has no prefilled position; prefill it before importing"
            )
        if position > self.max_positions:
            raise ValueError(
                f"import position {position} exceeds max_positions {self.max_positions}"
            )
        attention_states = self._require_attention_states()
        needed_blocks = (position + self.block_size - 1) // self.block_size
        if needed_blocks > self._slot_blocks[parsed]:
            raise ValueError(
                f"import position {position} needs {needed_blocks} paged blocks but "
                f"slot {parsed} holds {self._slot_blocks[parsed]}"
            )
        source_attention = tuple(getattr(source, "attention_states", ()))
        if len(source_attention) != len(attention_states):
            raise ValueError(
                f"import source has {len(source_attention)} QSA attention states, "
                f"expected {len(attention_states)}"
            )
        source_blocks: list[np.ndarray] = []
        for layer, state in enumerate(source_attention):
            if int(state.block_size) != self.block_size:
                raise ValueError(
                    f"import source QSA layer {layer} uses paged block size "
                    f"{int(state.block_size)}, this runner uses {self.block_size}; "
                    "a cross-block-size copy is not implemented"
                )
            blocks = np.asarray(state.block_host)
            if blocks.ndim == 2:
                if blocks.shape[0] != 1:
                    raise ValueError(
                        f"import source QSA layer {layer} holds {blocks.shape[0]} "
                        "requests; import needs a single-request source"
                    )
                blocks = blocks[0]
            if blocks.ndim != 1 or blocks.size < needed_blocks:
                raise ValueError(
                    f"import source QSA layer {layer} block table does not cover "
                    f"{needed_blocks} blocks"
                )
            source_blocks.append(blocks)
        source_index = tuple(getattr(source, "index_states", ()))
        if len(source_index) != len(self.index_states):
            raise ValueError(
                f"import source has {len(source_index)} QSA index states, "
                f"expected {len(self.index_states)}"
            )
        for layer, index_state in enumerate(source_index):
            if getattr(index_state, "pending_rollback", None) is not None:
                raise ValueError(
                    f"import source QSA layer {layer} has a pending index rollback"
                )
            if int(index_state.count) != position:
                raise ValueError(
                    f"import source QSA layer {layer} index count "
                    f"{int(index_state.count)} does not match position {position}"
                )
            if int(index_state.capacity) < position:
                raise ValueError(
                    f"import source QSA layer {layer} index capacity "
                    f"{int(index_state.capacity)} is below position {position}"
                )
        source_decode = getattr(source, "state", None)
        if source_decode is None:
            raise ValueError("import source has no decode state")
        if self.gdn_states:
            conv_bytes = int(self.gdn_states[0].conv_bytes_per_slot)
            matrix_bytes = int(self.gdn_states[0].recurrent_bytes_per_slot)
            if int(source_decode.gdn_conv.nbytes) != conv_bytes * len(self.gdn_states):
                raise ValueError(
                    "import source GDN conv bytes do not match this runner's "
                    "slot layout"
                )
            if (
                int(source_decode.gdn_matrix.nbytes)
                != matrix_bytes * len(self.gdn_states)
            ):
                raise ValueError(
                    "import source GDN recurrent bytes do not match this "
                    "runner's slot layout"
                )
        if self.ple_state is not None:
            if int(source_decode.ple_conv.nbytes) != int(
                self.ple_state.history_bytes_per_slot
            ):
                raise ValueError(
                    "import source PLE conv history bytes do not match this "
                    "runner's slot layout"
                )
            source_hashes = getattr(source, "ple_hash_states", None)
            if source_hashes is None:
                raise ValueError(
                    "import source does not expose its PLE n-gram history"
                )
            if 0 not in source_hashes:
                raise ValueError(
                    "import source has no PLE n-gram history for its request; "
                    "prefill it with the c1 runner before importing"
                )
        for layer, state in enumerate(source_attention):
            _require_disjoint(
                state.key_cache,
                attention_states[layer].key_cache,
                name=f"QSA layer {layer} key cache",
            )
            _require_disjoint(
                state.value_cache,
                attention_states[layer].value_cache,
                name=f"QSA layer {layer} value cache",
            )
            _require_disjoint(
                source_index[layer].raw_keys,
                self.index_states[layer].raw_keys,
                name=f"QSA layer {layer} index keys",
            )
        for gdn_state in self.gdn_states:
            _require_disjoint(
                source_decode.gdn_conv,
                gdn_state.conv_state,
                name="GDN conv state",
            )
            _require_disjoint(
                source_decode.gdn_matrix,
                gdn_state.recurrent_state,
                name="GDN recurrent state",
            )
        if self.ple_state is not None:
            _require_disjoint(
                source_decode.ple_conv,
                self.ple_state.history,
                name="PLE conv history",
            )

        # Copy phase. The slot is admitted and zeroed, so every write below is
        # the slot's first state. A failure part-way through leaves the slot
        # holding a mix of copied and zeroed state, so it is poisoned rather
        # than left usable.
        try:
            self._copy_prefilled_state(
                parsed,
                position=position,
                needed_blocks=needed_blocks,
                source_attention=source_attention,
                source_blocks=source_blocks,
                source_index=source_index,
                source_decode=source_decode,
                ple_hash_state=(
                    source.ple_hash_states[0] if self.ple_state is not None else None
                ),
            )
        except BaseException:
            self._poisoned.add(parsed)
            raise
        self._positions[parsed] = position
        return position

    def _copy_prefilled_state(
        self,
        slot: int,
        *,
        position: int,
        needed_blocks: int,
        source_attention: Sequence[object],
        source_blocks: Sequence[np.ndarray],
        source_index: Sequence[object],
        source_decode: object,
        ple_hash_state: PLEHashState | None,
    ) -> None:
        """Copy a validated source's per-slot state into ``slot``.

        Only the valid prefix of the last paged block is copied: the source's
        unwritten tail must not land in this slot.
        """

        attention_states = self._require_attention_states()
        attention = attention_states[0]
        row_bytes = attention.kv_heads * attention.head_dim * DType.BF16.itemsize
        block_bytes = attention.block_size * row_bytes
        slot_blocks = self._block_host[slot]
        for layer, state in enumerate(source_attention):
            destination_attention = attention_states[layer]
            index_state = self.index_states[layer]
            index_row_bytes = index_state.index_dim * DType.FP32.itemsize
            index_block_bytes = index_state.block_size * index_row_bytes
            rows_left = position
            for logical in range(needed_blocks):
                rows_in_block = min(self.block_size, rows_left)
                rows_left -= rows_in_block
                source_block = int(source_blocks[layer][logical])
                destination_block = int(slot_blocks[logical])
                for source_cache, destination_cache in (
                    (state.key_cache, destination_attention.key_cache),
                    (state.value_cache, destination_attention.value_cache),
                ):
                    self.runtime.memcpy(
                        destination_cache.ptr
                        + destination_block * block_bytes,
                        source_cache.ptr + source_block * block_bytes,
                        rows_in_block * row_bytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                    )
                self.runtime.memcpy(
                    index_state.raw_keys.ptr
                    + destination_block * index_block_bytes,
                    source_index[layer].raw_keys.ptr
                    + source_block * index_block_bytes,
                    rows_in_block * index_row_bytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
            # Pooled keys are re-derived by the next selection, not copied.
            index_state.cursor_host[slot] = position
            index_state.pooled_count_host[slot] = 0
        if self.gdn_states:
            layers = len(self.gdn_states)
            conv_bytes = int(self.gdn_states[0].conv_bytes_per_slot)
            matrix_bytes = int(self.gdn_states[0].recurrent_bytes_per_slot)
            source_conv_stride = int(source_decode.gdn_conv.nbytes) // layers
            source_matrix_stride = int(source_decode.gdn_matrix.nbytes) // layers
            for layer, gdn_state in enumerate(self.gdn_states):
                self.runtime.memcpy(
                    gdn_state.conv_slot_ptr(slot),
                    source_decode.gdn_conv.ptr + layer * source_conv_stride,
                    conv_bytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
                self.runtime.memcpy(
                    gdn_state.recurrent_slot_ptr(slot),
                    source_decode.gdn_matrix.ptr + layer * source_matrix_stride,
                    matrix_bytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
        if self.ple_state is not None:
            self.runtime.memcpy(
                self.ple_state.history_slot_ptr(slot),
                source_decode.ple_conv.ptr,
                int(self.ple_state.history_bytes_per_slot),
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            if ple_hash_state is None:
                raise RuntimeError(
                    "a PLE layer needs the source request's n-gram history"
                )
            self._ple_hash_states[slot] = _copy_ple_hash_state(ple_hash_state)

    # -- decode step ------------------------------------------------------

    def _q4_tile_policy(self) -> tuple[int | None, int | None]:
        """The compact F16 Q4_K gate/up tile pair a captured segment bakes in.

        The pair is request-local runner state and the owner resolves it at
        capture time, so a segment built under one pair must not replay under
        another. ``None`` is the unbound heuristic and is part of the identity.
        """

        return (self.q4_tile_m, self.q4_tile_n)

    def _q8_tile_policy(self) -> tuple[tuple[int, int] | None]:
        """The Q8_0 dense tile pair a captured layer segment bakes in.

        A step at ``rows>1`` can launch the Q8_0 dense WMMA prefill inside the
        captured segment, and that launch resolves this pair. ``None`` is the
        unbound heuristic and is part of the identity.
        """

        return (self.q8_wmma_tiles,)

    def _expert_down_policy(self) -> tuple[bool, ...]:
        """The expert-down decisions a captured layer segment depends on.

        The MoE runs eager inside the layer's captured ``post_mixer`` segment,
        so the expert-path selectors resolve at capture time and their launches
        are baked into the graph. Every capture key therefore carries these
        twelve plan-published decisions: a runner whose selection changes
        captures a new segment instead of replaying one built under the old
        selection. ``q51_exact_grouped_down`` is the exact grouped Q5_1 down
        chain's own gate, and every Q5_1 ladder rung sits inside the chain it
        gates; ``q51_wmma_down`` admits the F16 WMMA down owner below that
        chain, and a decode row cannot reach the production admission that also
        selects it; ``q51_out8`` is that chain's non-expertgrid fallback rung, so a
        captured segment built on the out8 owner must not replay under the
        output1 base. ``q4_exact_grouped_gate`` and ``q4_exact_grouped_all`` are
        the Q4_K gate/up chain's own gate and its `_ALL` joint term, and
        ``q4_out4`` is that chain's non-expertgrid fallback rung; all three
        select launches inside the segment.
        """

        return (
            bool(self.q51_down_m1),
            bool(self.q51_fold128_prefill),
            bool(self.q51_fold_pair_prefill),
            bool(self.q51_exact_grouped_down),
            bool(self.q51_wmma_down),
            bool(self.q51_out8),
            bool(self.q4_exact_grouped_gate),
            bool(self.q4_exact_grouped_all),
            bool(self.q4_out4),
            bool(self.forkb_grouped_down),
            bool(self.q8_down_row4_prefill),
            bool(self.q8_down_bundle_prefill),
        )

    def step(
        self,
        token_ids: Sequence[int],
        *,
        slots: Sequence[int] | None = None,
        owners: Sequence[int] | None = None,
        capture_logits: bool = True,
    ) -> Qwen4ExpBatchDecodeResult:
        """Advance every row by one token and return its logits.

        ``slots`` is the row-to-slot mapping: row ``r`` advances slot
        ``slots[r]``, in any order, and the returned result keeps that order.
        It defaults to every admitted slot in ascending slot order, and that
        default is the only form that does not need ``owners``.

        Naming slots explicitly requires ``owners``: the owner token each slot's
        admission returned. A token that no longer matches means the request
        that held the slot was released and another one was admitted into the
        same physical slot, so the call is refused instead of advancing a
        stranger's state.

        Token ids and slot indices must be integers (``operator.index``), so a
        float or a bool is rejected instead of being truncated. The whole row
        set, including every QSA owner's row plan, is validated before the first
        device write; a rejected call leaves every slot healthy. A call that
        fails *after* it started mutating poisons its rows (see
        :attr:`poisoned_slots`).

        ``capture_logits=False`` skips the ``[rows, vocab]`` readback and
        returns only greedy ids from the committed row-wise argmax kernel.
        """

        self._require_open()
        tokens = _token_ids(token_ids)
        plan = self._plan_step(tokens, slots, owners)
        if (
            self.selection is None
            or self.gdn_scratch is None
            or self.qsa_scratch is None
            or self.head_scratch is None
            or (self._ple_layers and self.ple_scratch is None)
        ):
            raise RuntimeError("Qwen4Exp batch runner has no live decode scratch")
        token_weight = self.resident.weight("root.token_embedding")
        embedding_kernel = resolve(
            backend=self.backend,
            layer="embedding",
            quant=token_weight.spec.quant_key,
            variant="lookup_bf16_out",
        )
        owner_plans = self._plan_owners(plan)
        # Everything above is host-side validation and lookup. From here on the
        # slots' device state changes, so a failure poisons the rows: a retry
        # would append and advance on top of a half-applied step.
        try:
            with _qwen4_exp_q8_tile_scope(self):
                return self._execute_step(
                    plan,
                    tokens,
                    owner_plans=owner_plans,
                    embedding_kernel=embedding_kernel,
                    token_weight=token_weight,
                    capture_logits=capture_logits,
                )
        except BaseException:
            self._poisoned.update(plan.slots)
            raise

    def _plan_owners(self, plan: _StepPlan) -> _StepOwners:
        """Validate every QSA owner's row set without touching a device buffer.

        The attention and index owners read the packed per-step metadata, so
        their rows are validated before that buffer's single upload: a rejected
        row set cannot leave uploaded metadata behind for a step that never ran.
        """

        attention_states = self._require_attention_states()
        self._require_step_metadata()
        index_states = self.index_states
        if len(index_states) != len(attention_states):
            raise RuntimeError(
                f"{len(attention_states)} attention owners do not match "
                f"{len(index_states)} index owners"
            )
        return _StepOwners(
            attention=tuple(
                state.validate_rows(plan.slots, plan.positions)
                for state in attention_states
            ),
            index=tuple(state.validate_rows(plan.slots) for state in index_states),
        )

    def _require_step_metadata(self) -> Qwen4ExpBatchStepMetadata:
        metadata = self.step_metadata
        if metadata is None:
            raise RuntimeError(
                "Qwen4Exp batch runner has no packed step metadata buffer"
            )
        if metadata.layers != len(self.attention_states):
            raise RuntimeError(
                f"the packed step metadata holds {metadata.layers} layers for "
                f"{len(self.attention_states)} attention owners"
            )
        if metadata.state_slots != self.state_slots:
            raise RuntimeError(
                f"the packed step metadata holds {metadata.state_slots} slots for "
                f"a {self.state_slots}-slot runner"
            )
        return metadata

    def _execute_step(
        self,
        plan: _StepPlan,
        tokens: np.ndarray,
        *,
        owner_plans: _StepOwners,
        embedding_kernel: object,
        token_weight: object,
        capture_logits: bool,
    ) -> Qwen4ExpBatchDecodeResult:
        """Run one validated step. Callers must poison ``plan.slots`` on failure.

        Every owner's row set arrives already validated in ``owner_plans``. This
        method stages all of them into the packed metadata buffer, publishes it
        with one copy, and only then admits the rows on the owners, so no owner
        publishes a row set whose siblings were rejected.
        """

        cfg = self.config
        attention_states = self._require_attention_states()
        step_metadata = self._require_step_metadata()
        selection = self.selection
        gdn_scratch = self.gdn_scratch
        qsa_scratch = self.qsa_scratch
        ple_scratch = self.ple_scratch
        head_scratch = self.head_scratch
        rows = plan.rows
        position_host = np.ascontiguousarray(plan.positions)
        copy_host_array_to_device(
            self.token_ids,
            tokens,
            tokens.nbytes,
            runtime=self.runtime,
        )
        embedding_kernel(
            self.token_ids.ptr,
            token_weight.allocation("raw").tensor.ptr,
            self.embedding.ptr,
            rows,
            cfg.hidden_size,
            cfg.vocab_size,
            runtime=self.runtime,
        )
        qwen4_exp_repeat_bf16_branches(
            self.embedding.ptr,
            self.residual.ptr,
            cfg.residual_branch_count,
            cfg.hidden_size,
            rows=rows,
            runtime=self.runtime,
        )
        if self._ple_layers:
            ple_rows, ple_states = ple_hash_rows(
                tokens.tolist(),
                positions=position_host.tolist(),
                sequence_ids=[int(slot) for slot in plan.slots],
                states=self._ple_hash_states,
                eos_token_id=cfg.ple_eos_token_id,
                layer_multipliers=cfg.ple_layer_multipliers,
                head_offsets=cfg.ple_head_offsets,
                head_vocab_sizes=cfg.ple_head_vocab_sizes,
                heads_per_ngram=cfg.ple_heads_per_ngram,
                ngram_size=cfg.ple_ngram_size,
            )
            staged = stage_qwen4_exp_ple_rows(
                self.resident.ple_staging, ple_rows, hidden=cfg.hidden_size
            )
            ple_upload = np.ascontiguousarray(staged)
            copy_host_array_to_device(
                self.ple_embedding,
                ple_upload,
                staged.nbytes,
                runtime=self.runtime,
            )
            self._ple_hash_states = ple_states
        for layer, attention_plan in enumerate(owner_plans.attention):
            index_plan = owner_plans.index[layer]
            step_metadata.stage(
                layer,
                positions=attention_plan.positions,
                contexts=attention_plan.contexts,
                index_positions=index_plan.positions,
            )
        # One upload for every QSA owner's dynamic per-step metadata. Staging is
        # host-only, so a step that cannot stage its rows publishes nothing.
        step_metadata.upload()
        for layer, attention_plan in enumerate(owner_plans.attention):
            attention_states[layer].publish(
                attention_plan, packed=step_metadata.views(layer)
            )
        for layer, index_plan in enumerate(owner_plans.index):
            self.index_states[layer].publish(
                index_plan, packed=step_metadata.views(layer)
            )
        for gdn_state in self.gdn_states:
            gdn_state.prepare(plan.slots)
        if self.ple_state is not None:
            self.ple_state.prepare(plan.slots)
        residual_ptr = self.residual.ptr
        for layer, kind in enumerate(cfg.layer_types):
            if layer in self._ple_layers:
                layer_prefix = f"layers.{layer}."
                residual_ptr = run_qwen4_exp_batch_ple_stage(
                    residual_ptr,
                    self.ple_embedding.ptr,
                    {
                        "ple_key": self.resident.weight(layer_prefix + "ple_key"),
                        "ple_value": self.resident.weight(layer_prefix + "ple_value"),
                    },
                    state=self.ple_state,
                    scratch=ple_scratch,
                    norm_key_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_key"
                    ).allocation("raw").tensor.ptr,
                    norm_query_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_query"
                    ).allocation("raw").tensor.ptr,
                    norm_conv_ptr=self.resident.weight(
                        layer_prefix + "ple_norm_conv"
                    ).allocation("raw").tensor.ptr,
                    conv_weight_ptr=self.resident.weight(
                        layer_prefix + "ple_conv1d"
                    ).allocation("raw").tensor.ptr,
                    rows=rows,
                    branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size,
                    conv_kernel=cfg.ple_conv_kernel,
                    dilation=cfg.ple_ngram_size,
                    runtime=self.runtime,
                ).ptr
            if kind == "gdn":
                gdn_binding = self.gdn_bindings[layer]
                residual_ptr = run_qwen4_exp_batch_gdn_layer(
                    residual_ptr,
                    gdn_binding,
                    state=self.gdn_states[gdn_binding.gdn_state_index],
                    scratch=gdn_scratch,
                    rows=rows,
                    branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank,
                    num_k_heads=cfg.gdn_group_count,
                    num_v_heads=cfg.gdn_time_step_rank,
                    head_dim=cfg.gdn_state_size,
                    conv_kernel=cfg.gdn_conv_kernel,
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                    graph_cache=self._graphs,
                    graph_key=(layer, rows) + self._expert_down_policy()
                    + self._q4_tile_policy() + self._q8_tile_policy(),
                    q51_down_m1=self.q51_down_m1,
                    q51_fold128_prefill=self.q51_fold128_prefill,
                    q51_fold_pair_prefill=self.q51_fold_pair_prefill,
                    q51_exact_grouped_down=self.q51_exact_grouped_down,
                    q51_wmma_down=self.q51_wmma_down,
                    q51_out8=self.q51_out8,
                    q4_exact_grouped_gate=self.q4_exact_grouped_gate,
                    q4_exact_grouped_all=self.q4_exact_grouped_all,
                    q4_out4=self.q4_out4,
                    q4_tile_m=self.q4_tile_m,
                    q4_tile_n=self.q4_tile_n,
                    forkb_grouped_down=self.forkb_grouped_down,
                    q8_down_row4_prefill=self.q8_down_row4_prefill,
                    q8_down_bundle_prefill=self.q8_down_bundle_prefill,
                ).ptr
            else:
                qsa_binding = self.qsa_bindings[layer]
                residual_ptr = run_qwen4_exp_batch_qsa_layer(
                    residual_ptr,
                    qsa_binding,
                    attention_state=attention_states[qsa_binding.qsa_state_index],
                    index_state=self.index_states[qsa_binding.qsa_state_index],
                    selection=selection,
                    scratch=qsa_scratch,
                    rows=rows,
                    branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank,
                    query_heads=cfg.attention_head_count,
                    kv_heads=cfg.attention_kv_head_count,
                    head_dim=cfg.attention_key_length,
                    rotary_dim=cfg.rope_dimension_count,
                    theta=cfg.rope_freq_base,
                    index_heads=cfg.indexer_head_count,
                    index_dim=cfg.indexer_key_length,
                    index_rotary_dim=cfg.rope_dimension_count,
                    ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count,
                    top_k=cfg.expert_used_count,
                    runtime=self.runtime,
                    graph_cache=self._graphs,
                    graph_key=(layer, rows) + self._expert_down_policy()
                    + self._q4_tile_policy() + self._q8_tile_policy(),
                    q51_down_m1=self.q51_down_m1,
                    q51_fold128_prefill=self.q51_fold128_prefill,
                    q51_fold_pair_prefill=self.q51_fold_pair_prefill,
                    q51_exact_grouped_down=self.q51_exact_grouped_down,
                    q51_wmma_down=self.q51_wmma_down,
                    q51_out8=self.q51_out8,
                    q4_exact_grouped_gate=self.q4_exact_grouped_gate,
                    q4_exact_grouped_all=self.q4_exact_grouped_all,
                    q4_out4=self.q4_out4,
                    q4_tile_m=self.q4_tile_m,
                    q4_tile_n=self.q4_tile_n,
                    forkb_grouped_down=self.forkb_grouped_down,
                    q8_down_row4_prefill=self.q8_down_row4_prefill,
                    q8_down_bundle_prefill=self.q8_down_bundle_prefill,
                ).ptr
        head_norm_ptr = self.resident.weight(
            "root.head_hc_norm"
        ).allocation("raw").tensor.ptr
        head_down = self.resident.weight("root.head_hc_down")
        head_up = self.resident.weight("root.head_hc_up")
        lm_head = self.resident.weight("root.lm_head")

        def head_unit(segment_stream: int) -> None:
            """Final GR read plus the tied lm_head projection, as one unit."""

            head_read = run_qwen4_exp_gr_read(
                residual_ptr,
                head_norm_ptr,
                head_down,
                head_up,
                None,
                head_scratch,
                rows=rows,
                branches=cfg.residual_branch_count,
                hidden=cfg.hidden_size,
                low_rank=cfg.residual_low_rank,
                stream=segment_stream,
                runtime=self.runtime,
            )
            launch_gguf_linear(
                lm_head,
                head_read.mixed.ptr,
                self.logits.ptr,
                rows,
                cfg.hidden_size,
                cfg.vocab_size,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32,
                stream=segment_stream,
                runtime=self.runtime,
            )

        # The head is stateless: it reads the final residual, writes its own
        # session-resident scratch and ``logits``, and keeps no state across
        # steps, so it is graph-safe exactly like a layer segment. Its launch
        # geometry is fixed by ``rows`` and a captured graph bakes in the
        # residual pointer, so the key carries both: a width or residual-pointer
        # change captures a new graph instead of replaying one that would read a
        # stale row set. The logits/argmax readback stays outside this unit, so
        # no host copy enters the capture.
        #
        # Unlike a layer segment this head needs no inject-free guard: the GR
        # read zero-fills its inject buffer on the stream it is handed
        # (``memset_async``), so the capture records that fill instead of
        # leaving it on the null stream.
        _run_segment(
            self._graphs,
            None,
            "head",
            head_unit,
            outputs=(
                (self.logits.ptr, rows * cfg.vocab_size * DType.FP32.itemsize),
            ),
            stream=0,
            key=("head", rows, residual_ptr),
        )
        if capture_logits:
            self.runtime.device_synchronize()
            logits = np.empty((rows, cfg.vocab_size), dtype=np.float32)
            copy_device_to_host(
                host_array_ptr(logits),
                self.logits,
                logits.nbytes,
                runtime=self.runtime,
            )
            tokens_out = np.ascontiguousarray(
                np.argmax(logits, axis=1), dtype=np.int64
            )
        else:
            logits = None
            argmax_f32_rows_i32(
                self.logits.ptr,
                self.argmax_values.ptr,
                self.argmax_blocks_i32.ptr,
                self.argmax_ids.ptr,
                None,
                rows,
                cfg.vocab_size,
                runtime=self.runtime,
            )
            ids = np.empty(rows, dtype=np.int32)
            copy_device_to_host(
                host_array_ptr(ids),
                self.argmax_ids,
                ids.nbytes,
                runtime=self.runtime,
            )
            tokens_out = np.ascontiguousarray(ids, dtype=np.int64)
        self._positions[list(plan.slots)] = position_host + 1
        return Qwen4ExpBatchDecodeResult(
            slot_ids=plan.slots,
            positions=position_host,
            tokens=tokens_out,
            logits=logits,
        )

    # -- validation -------------------------------------------------------

    def _plan_step(
        self,
        tokens: np.ndarray,
        slots: Sequence[int] | None,
        owners: Sequence[int] | None,
    ) -> _StepPlan:
        """Validate the whole row set before the step mutates anything.

        Nothing here writes device or host state, so a rejection leaves every
        slot healthy.
        """

        self._require_attention_states()
        cfg = self.config
        rows = int(tokens.size)
        if rows <= 0:
            raise ValueError("a batch decode step needs at least one token")
        if rows > self.state_slots:
            raise ValueError(
                f"step admits at most {self.state_slots} rows, got {rows}"
            )
        if slots is None:
            if owners is not None:
                raise ValueError(
                    "owners is only meaningful with explicit slots; a default "
                    "step advances the currently admitted slots"
                )
            chosen = self.active_slots
            if not chosen:
                raise RuntimeError(
                    "no slots are admitted; admit a slot before stepping"
                )
            granted = None
        else:
            if owners is None:
                raise ValueError(
                    "step with explicit slots requires the owner token of every "
                    "named slot; pass owners= (the tokens admit() returned) or "
                    "omit slots to advance the currently admitted slots"
                )
            chosen = tuple(
                self._row_slot(row, value) for row, value in enumerate(slots)
            )
            granted = tuple(
                _row_owner_token(row, value) for row, value in enumerate(owners)
            )
        if len(chosen) != rows:
            raise ValueError(
                f"slots and token_ids must have equal length, got {len(chosen)} "
                f"and {rows}"
            )
        if granted is not None and len(granted) != rows:
            raise ValueError(
                f"owners and token_ids must have equal length, got "
                f"{len(granted)} and {rows}"
            )
        if len(set(chosen)) != len(chosen):
            raise ValueError("a slot appears twice in one step's row set")
        for row, slot in enumerate(chosen):
            if not 0 <= slot < self.state_slots:
                raise ValueError(
                    f"row {row} slot {slot} is outside [0, {self.state_slots})"
                )
            self._require_admitted(slot)
            if granted is not None:
                self._require_owner(slot, granted[row])
            self._require_healthy(slot)
        if np.any(tokens < 0) or np.any(tokens >= int(cfg.vocab_size)):
            raise ValueError("token ids must be inside the model vocabulary")
        positions = np.ascontiguousarray(self._positions[list(chosen)])
        if np.any(positions < 0) or np.any(positions >= self.max_positions):
            raise ValueError(
                "a step position reached the per-slot decode capacity "
                f"{self.max_positions}"
            )
        for slot, position in zip(chosen, positions.tolist()):
            if int(position) + 1 > self.slot_capacity(slot):
                raise ValueError(
                    f"slot {slot} context {int(position) + 1} exceeds the "
                    f"{self.slot_capacity(slot)} positions its paged blocks hold"
                )
        for index_state in self.index_states:
            if int(positions.max()) + 1 > int(index_state.capacity):
                raise ValueError(
                    f"context {int(positions.max()) + 1} exceeds the index "
                    f"capacity {int(index_state.capacity)}"
                )
        return _StepPlan(rows=rows, slots=chosen, positions=positions)

    def _require_matching_config(self, source_config: object) -> None:
        if source_config is self.config:
            return
        if source_config == self.config:
            return
        declared = getattr(type(self.config), "__dataclass_fields__", None)
        if declared is None:
            raise ValueError(
                "import source model config differs from this runner"
            )
        mismatched = [
            field.name
            for field in fields(self.config)
            if getattr(source_config, field.name, object())
            != getattr(self.config, field.name)
        ]
        raise ValueError(
            "import source model config differs from this runner: "
            + ", ".join(mismatched or ["unknown field"])
        )

    def _require_open(self) -> None:
        if self.closed or self.closing:
            raise RuntimeError("Qwen4Exp batch runner is closed")

    def _row_slot(self, row: int, value: object) -> int:
        """Parse one row's slot index, rejecting a non-integral value."""

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"row {row} slot must be an integer, got a bool")
        try:
            parsed = operator.index(value)
        except TypeError:
            raise ValueError(
                f"row {row} slot must be an integer, got {type(value).__name__}"
            ) from None
        return int(parsed)

    def _require_slot(self, slot: object) -> int:
        if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)):
            raise ValueError(f"slot must be an integer, got {type(slot).__name__}")
        parsed = int(slot)
        if not 0 <= parsed < self.state_slots:
            raise ValueError(
                f"slot {parsed} is outside [0, {self.state_slots})"
            )
        return parsed

    def _require_admitted(self, slot: int) -> None:
        if not self._owners[slot]:
            raise RuntimeError(
                f"slot {slot} is not admitted; admit it before using it in a step"
            )

    def _require_owner(self, slot: int, owner: object) -> None:
        if isinstance(owner, bool) or not isinstance(owner, (int, np.integer)):
            raise ValueError(
                f"owner token must be an integer, got {type(owner).__name__}"
            )
        parsed = int(owner)
        if not self._owners[slot]:
            raise RuntimeError(f"slot {slot} is not admitted")
        if parsed != self._owners[slot]:
            raise RuntimeError(
                f"slot {slot} is owned by request {self._owners[slot]}, not {parsed}"
            )

    def _require_attention_states(self) -> tuple[Qwen4ExpBatchAttentionState, ...]:
        states = self.attention_states
        if not states:
            raise RuntimeError("Qwen4Exp batch runner has no attention state")
        return states

    # -- mixed target scratch ---------------------------------------------

    def mixed_target_scratch(self, rows: int) -> Qwen4ExpMixedTargetScratch:
        """The batch-owned stateless scratch for a ``rows``-row mixed pass.

        A pass at or below the installed capacity reuses it unchanged. A wider
        pass allocates the larger set first and releases the smaller one after,
        so a failed allocation leaves the previous capacity installed and the
        retry starts from it. Growth is refused while the set is leased by a live
        pass, because replacing it would free buffers that pass names.
        """

        self._require_open()
        count = int(rows)
        if count <= 0:
            raise ValueError("mixed target scratch rows must be positive")
        installed = self._mixed_target_scratch
        if (
            installed is not None
            and not installed.closing
            and installed.capacity_rows >= count
        ):
            return installed
        if installed is not None and installed.borrowers:
            raise RuntimeError(
                f"mixed target scratch is still leased by a live mixed target pass, "
                f"so it cannot grow from {installed.capacity_rows} to {count} rows"
            )
        grown = Qwen4ExpMixedTargetScratch.allocate(
            rows=count, config=self.config, runtime=self.runtime
        )
        self._mixed_target_scratch = grown
        self._owned.append(grown)
        if installed is not None:
            # Pop the replaced set only after its free returns, so a failed
            # release stays reachable for the runner's own retry.
            installed.close()
            self._owned.remove(installed)
        return grown

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        self.closing = True
        # Graphs reference the scratch below, so destroy them first.
        if self._graphs is not None:
            self._graphs.close()
            self._graphs = None
        # Pop only after a successful free/close, so a raising owner stays
        # reachable and a retry closes the remainder exactly once.
        while self._owned:
            owner = self._owned[-1]
            owner.close()
            self._owned.pop()
        if self._buffers is not None:
            self._buffers.close()
        self.attention_states = ()
        self.index_states = ()
        self.gdn_states = ()
        self.ple_state = None
        self.step_metadata = None
        self.selection = None
        self.gdn_scratch = None
        self.qsa_scratch = None
        self.ple_scratch = None
        self.head_scratch = None
        self._mixed_target_scratch = None
        self.token_ids = None
        self.embedding = None
        self.ple_embedding = None
        self.residual = None
        self.logits = None
        self.argmax_ids = None
        self.argmax_values = None
        self.argmax_blocks_i32 = None
        self._poisoned.clear()
        self.closed = True

    def __enter__(self) -> "Qwen4ExpGGUFBatchRunner":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _token_ids(token_ids: object) -> np.ndarray:
    """Parse a row set of token ids, rejecting anything non-integral.

    ``np.asarray(..., dtype=np.int64)`` would silently truncate ``3.7`` to ``3``
    and accept ``True`` as ``1``, which turns a caller's bug into a wrong token.
    """

    try:
        values = list(token_ids)
    except TypeError:
        raise ValueError("token_ids must be a sequence of integers") from None
    parsed = []
    for row, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"row {row} token id must be an integer, got a bool")
        try:
            parsed.append(int(operator.index(value)))
        except TypeError:
            raise ValueError(
                f"row {row} token id must be an integer, got "
                f"{type(value).__name__}"
            ) from None
    return np.ascontiguousarray(np.asarray(parsed, dtype=np.int64))


def _row_owner_token(row: int, value: object) -> int:
    """Parse one row's owner token, rejecting a non-integral value."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"row {row} owner token must be an integer, got a bool")
    try:
        return int(operator.index(value))
    except TypeError:
        raise ValueError(
            f"row {row} owner token must be an integer, got {type(value).__name__}"
        ) from None


def _require_disjoint(
    source: DeviceBuffer, destination: DeviceBuffer, *, name: str
) -> None:
    """Reject an import whose source and destination share device bytes."""

    source_start = int(source.ptr)
    source_end = source_start + int(source.nbytes)
    destination_start = int(destination.ptr)
    destination_end = destination_start + int(destination.nbytes)
    if source_start < destination_end and destination_start < source_end:
        raise ValueError(
            f"{name} source and destination device ranges overlap; import "
            "requires two distinct state owners"
        )


def _copy_ple_hash_state(state: PLEHashState) -> PLEHashState:
    """Copy the n-gram history so the two runners cannot alias it."""

    return PLEHashState(
        tokens=tuple(int(token) for token in state.tokens),
        next_position=int(state.next_position),
    )


def batch_runner_call_table() -> dict[str, Callable[..., Any]]:
    """The exact row-generic helpers the batch runner calls outside mixers.

    These helpers are shared with the c1 path, so they are recorded as shared
    bindings rather than as batch-specific arithmetic. The batch-specific
    mixers, indexed kernels, and projection dispatch are bound by the batch
    manifest from their own modules.
    """

    return {
        "gr_read": run_qwen4_exp_gr_read,
        "gr_write_f32": qwen4_exp_gr_write_bf16_f32,
        "gr_write_bf16": qwen4_exp_gr_write_bf16_bf16,
        "moe": run_qwen4_exp_moe,
        "repeat_branches": qwen4_exp_repeat_bf16_branches,
        "argmax": argmax_f32_rows_i32,
    }


__all__ = [
    "QWEN4_EXP_BATCH_BLOCK_SIZE",
    "Qwen4ExpBatchDecodeResult",
    "Qwen4ExpGGUFBatchRunner",
    "Qwen4ExpMixedTargetScratch",
    "batch_runner_call_table",
    "run_qwen4_exp_batch_gdn_layer",
    "run_qwen4_exp_batch_qsa_layer",
]
