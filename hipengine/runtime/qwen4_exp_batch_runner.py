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
paged and index capacity) before the first mutation. The stages that follow are
not transactional: a device failure after the index append leaves the affected
slots' device state partially advanced and their PLE n-gram history advanced.
Recovery is :meth:`release` (or :meth:`reset_slot`) and re-admission, not a
retry of the failed step.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
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
from hipengine.runtime.qwen4_exp_batch_gdn import (
    Qwen4ExpBatchGdnState,
    run_qwen4_exp_batch_gdn_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_ple import (
    Qwen4ExpBatchPleState,
    run_qwen4_exp_batch_ple_stage,
)
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaSelectionScratch,
    run_qwen4_exp_batch_qsa_mixed_token_mixer,
)
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_runner import (
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
) -> DeviceBuffer:
    """Execute one complete N-row GDN+MoE layer over request-owned state slots.

    The GR reads/writes and the MoE are the committed row-generic helpers the c1
    runner uses; only the mixer owner changes (indexed decode instead of the
    single-request owner). ``state.prepare(slot_ids)`` must already have admitted
    the rows.

    The c1 layer's graph-capture segments are deliberately absent: they require
    ``rows == 1`` and a captured single-request decode, which is the c1 runner's
    job. The explicit residual write below is the same arithmetic as the fused
    prefill write (``tests/test_gpu_qwen4exp_gr_write_norm.py``), and the batch
    runner's slot bound is far below the 1024-row threshold that selects it.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp GDN layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match GDN layer scratch owner")
    attention_read = run_qwen4_exp_gr_read(
        residual_ptr,
        weights.attention_gr.norm_weight_ptr,
        weights.attention_gr.down,
        weights.attention_gr.up,
        weights.attention_gr.inject,
        scratch.attention_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=active_runtime,
    )
    mixer_output = run_qwen4_exp_batch_gdn_token_mixer(
        attention_read.mixed.ptr,
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
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_gr_write_bf16_f32(
        residual_ptr,
        mixer_output.ptr,
        attention_read.inject_logits.ptr,
        scratch.after_attention.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    ffn_read = run_qwen4_exp_gr_read(
        scratch.after_attention.ptr,
        weights.ffn_gr.norm_weight_ptr,
        weights.ffn_gr.down,
        weights.ffn_gr.up,
        weights.ffn_gr.inject,
        scratch.ffn_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=active_runtime,
    )
    moe_output = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        scratch=scratch.moe,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=active_runtime,
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
        runtime=active_runtime,
    )
    return scratch.output


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
) -> DeviceBuffer:
    """Execute one complete N-row QSA+MoE layer over request-owned slots.

    The mixer is the mixed dense/sparse rows owner: every row carries its own
    ``(positions, count)`` selection, so a batch may hold requests on either
    side of the index bound. ``attention_state.prepare`` and
    ``index_state.prepare`` must already have admitted the same slots in the
    same row order.

    The c1 layer's graph-capture path is absent for the same reason as the GDN
    layer: capture is a ``rows == 1`` single-request optimization.
    """

    if scratch.closed:
        raise RuntimeError("Qwen4Exp QSA layer scratch is closed")
    active_runtime = runtime or scratch.runtime
    if active_runtime is not scratch.runtime:
        raise ValueError("runtime must match QSA layer scratch owner")
    attention_read = run_qwen4_exp_gr_read(
        residual_ptr,
        weights.attention_gr.norm_weight_ptr,
        weights.attention_gr.down,
        weights.attention_gr.up,
        weights.attention_gr.inject,
        scratch.attention_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=active_runtime,
    )
    mixer = run_qwen4_exp_batch_qsa_mixed_token_mixer(
        attention_read.mixed.ptr,
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
        stream=stream,
        runtime=active_runtime,
    )
    qwen4_exp_gr_write_bf16_f32(
        residual_ptr,
        mixer.ptr,
        attention_read.inject_logits.ptr,
        scratch.after_attention.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
    )
    ffn_read = run_qwen4_exp_gr_read(
        scratch.after_attention.ptr,
        weights.ffn_gr.norm_weight_ptr,
        weights.ffn_gr.down,
        weights.ffn_gr.up,
        weights.ffn_gr.inject,
        scratch.ffn_gr,
        rows=rows, branches=branches, hidden=hidden, low_rank=low_rank,
        stream=stream, runtime=active_runtime,
    )
    moe = run_qwen4_exp_moe(
        ffn_read.mixed.ptr,
        weights.moe,
        scratch=scratch.moe,
        rows=rows, hidden=hidden, ffn=ffn, experts=experts, top_k=top_k,
        stream=stream, runtime=active_runtime,
    )
    qwen4_exp_gr_write_bf16_bf16(
        scratch.after_attention.ptr,
        moe.output.ptr,
        ffn_read.inject_logits.ptr,
        scratch.output.ptr,
        rows,
        branches,
        hidden,
        stream=stream,
        runtime=active_runtime,
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

    def __init__(
        self,
        resident: object,
        *,
        state_slots: int,
        max_positions: int,
        backend: str = "hip_gfx1151",
        block_size: int = QWEN4_EXP_BATCH_BLOCK_SIZE,
        runtime: HipRuntime | None = None,
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
        self.closed = False
        self.closing = False
        self._owned: list[Any] = []
        self._buffers: _BufferGroup | None = None
        self._owners = [0] * slots
        self._next_owner = 0
        self._positions = np.zeros(slots, dtype=np.int64)
        self._ple_hash_states: dict[int, PLEHashState] = {}
        self.gdn_bindings: Mapping[int, Qwen4ExpGDNLayerDeviceWeights] = {}
        self.qsa_bindings: Mapping[int, Qwen4ExpQSALayerDeviceWeights] = {}
        self.attention_state: Qwen4ExpBatchAttentionState | None = None
        self.index_states: tuple[Qwen4ExpBatchQsaIndexState, ...] = ()
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
        except BaseException:
            self.close()
            raise

    # -- construction -----------------------------------------------------

    def _allocate(self) -> None:
        cfg = self.config
        qkv_width = 2 * cfg.gdn_group_count * cfg.gdn_state_size + cfg.gdn_inner_size
        attention = Qwen4ExpBatchAttentionState.allocate(
            state_slots=self.state_slots,
            max_positions=self.max_positions,
            block_size=self.block_size,
            kv_heads=cfg.attention_kv_head_count,
            head_dim=cfg.attention_key_length,
            runtime=self.runtime,
        )
        self.attention_state = attention
        self._owned.append(attention)
        index_states = tuple(
            Qwen4ExpBatchQsaIndexState.allocate(
                attention_state=attention,
                index_heads=cfg.indexer_head_count,
                index_dim=cfg.indexer_key_length,
                compression_ratio=cfg.qsa_compression_ratio,
                block_budget=cfg.qsa_block_budget,
                runtime=self.runtime,
            )
            for _ in range(cfg.layer_types.count("qsa"))
        )
        self.index_states = index_states
        self._owned.extend(index_states)
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
            stride=index_states[0].output_stride if index_states else 1,
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
                malloc(
                    argmax_blocks * DType.FP32.itemsize,
                    runtime=self.runtime,
                ),
                malloc(
                    argmax_blocks * DType.INT32.itemsize,
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

    # -- slot ownership ---------------------------------------------------

    @property
    def active_slots(self) -> tuple[int, ...]:
        """Admitted slots in admission order."""

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

    def admit(self, slot: int) -> int:
        """Take ownership of ``slot`` and zero the request state it holds.

        Returns a fresh owner token. The token is the only handle that can
        release the slot again, and it is never reused, so a handle from a
        previous request cannot free the current one.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        if self._owners[parsed]:
            raise RuntimeError(
                f"slot {parsed} is already owned by request {self._owners[parsed]}"
            )
        self._zero_slot(parsed)
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
        self._owners[parsed] = 0
        self._positions[parsed] = 0
        self._ple_hash_states.pop(parsed, None)

    def reset_slot(self, slot: int, owner: int) -> None:
        """Zero an owned slot in place, keeping the owner token and position 0.

        This is the slot-reuse path for a request that is replaced without
        releasing the slot; the owner token stays valid because the physical
        owner did not change.
        """

        self._require_open()
        parsed = self._require_slot(slot)
        self._require_owner(parsed, owner)
        self._zero_slot(parsed)
        self._positions[parsed] = 0
        self._ple_hash_states.pop(parsed, None)

    def _zero_slot(self, slot: int) -> None:
        attention = self._require_attention()
        blocks = attention.block_host[slot]
        row_bytes = (
            attention.kv_heads * attention.head_dim * DType.BF16.itemsize
        )
        block_bytes = attention.block_size * row_bytes
        for physical in blocks.tolist():
            offset = int(physical) * block_bytes
            self.runtime.memset(
                attention.key_cache.ptr + offset, 0, block_bytes
            )
            self.runtime.memset(
                attention.value_cache.ptr + offset, 0, block_bytes
            )
        for index_state in self.index_states:
            base = index_state.raw_keys.ptr
            index_row_bytes = index_state.index_dim * DType.FP32.itemsize
            index_block_bytes = index_state.block_size * index_row_bytes
            for logical in range(attention.blocks_per_request):
                self.runtime.memset(
                    base + int(blocks[logical]) * index_block_bytes,
                    0,
                    index_block_bytes,
                )
            # The owner's own reset advances its generation, which invalidates
            # any rollback token captured by the previous request.
            index_state.reset_slot(slot)
        for gdn_state in self.gdn_states:
            self.runtime.memset(
                gdn_state.conv_slot_ptr(slot), 0, gdn_state.conv_bytes_per_slot
            )
            self.runtime.memset(
                gdn_state.recurrent_slot_ptr(slot),
                0,
                gdn_state.recurrent_bytes_per_slot,
            )
        if self.ple_state is not None:
            self.runtime.memset(
                self.ple_state.history_slot_ptr(slot),
                0,
                self.ple_state.history_bytes_per_slot,
            )

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
        attention = self._require_attention()
        needed_blocks = (position + self.block_size - 1) // self.block_size
        if needed_blocks > attention.blocks_per_request:
            raise ValueError(
                f"import position {position} needs {needed_blocks} paged blocks but "
                f"each slot holds {attention.blocks_per_request}"
            )
        source_attention = tuple(getattr(source, "attention_states", ()))
        if len(source_attention) != len(self.index_states):
            raise ValueError(
                f"import source has {len(source_attention)} QSA attention states, "
                f"expected {len(self.index_states)}"
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
                attention.key_cache,
                name=f"QSA layer {layer} key cache",
            )
            _require_disjoint(
                state.value_cache,
                attention.value_cache,
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
        # the slot's first state. Only the valid prefix of the last block is
        # copied: the source's unwritten tail must not land in this slot.
        row_bytes = attention.kv_heads * attention.head_dim * DType.BF16.itemsize
        block_bytes = attention.block_size * row_bytes
        slot_blocks = attention.block_host[parsed]
        for layer, state in enumerate(source_attention):
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
                    (state.key_cache, attention.key_cache),
                    (state.value_cache, attention.value_cache),
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
            index_state.cursor_host[parsed] = position
            index_state.pooled_count_host[parsed] = 0
        if self.gdn_states:
            layers = len(self.gdn_states)
            conv_bytes = int(self.gdn_states[0].conv_bytes_per_slot)
            matrix_bytes = int(self.gdn_states[0].recurrent_bytes_per_slot)
            source_conv_stride = int(source_decode.gdn_conv.nbytes) // layers
            source_matrix_stride = int(source_decode.gdn_matrix.nbytes) // layers
            for layer, gdn_state in enumerate(self.gdn_states):
                self.runtime.memcpy(
                    gdn_state.conv_slot_ptr(parsed),
                    source_decode.gdn_conv.ptr + layer * source_conv_stride,
                    conv_bytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
                self.runtime.memcpy(
                    gdn_state.recurrent_slot_ptr(parsed),
                    source_decode.gdn_matrix.ptr + layer * source_matrix_stride,
                    matrix_bytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
        if self.ple_state is not None:
            self.runtime.memcpy(
                self.ple_state.history_slot_ptr(parsed),
                source_decode.ple_conv.ptr,
                int(self.ple_state.history_bytes_per_slot),
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            self._ple_hash_states[parsed] = _copy_ple_hash_state(
                source.ple_hash_states[0]
            )
        self._positions[parsed] = position
        return position

    # -- decode step ------------------------------------------------------

    def step(
        self,
        token_ids: Sequence[int],
        *,
        slots: Sequence[int] | None = None,
        capture_logits: bool = True,
    ) -> Qwen4ExpBatchDecodeResult:
        """Advance every admitted row by one token and return its logits.

        ``slots`` is the row-to-slot mapping: row ``r`` advances slot
        ``slots[r]``, in any order, and the returned result keeps that order.
        It defaults to the admitted slots in admission order. All rows must
        already be admitted; the whole row set is validated before the first
        device write.

        ``capture_logits=False`` skips the ``[rows, vocab]`` readback and
        returns only greedy ids from the committed row-wise argmax kernel.
        """

        self._require_open()
        tokens = np.ascontiguousarray(
            np.asarray(token_ids, dtype=np.int64).reshape(-1)
        )
        plan = self._plan_step(tokens, slots)
        cfg = self.config
        attention = self._require_attention()
        selection = self.selection
        gdn_scratch = self.gdn_scratch
        qsa_scratch = self.qsa_scratch
        ple_scratch = self.ple_scratch
        head_scratch = self.head_scratch
        if (
            selection is None
            or gdn_scratch is None
            or qsa_scratch is None
            or head_scratch is None
            or (self._ple_layers and ple_scratch is None)
        ):
            raise RuntimeError("Qwen4Exp batch runner has no live decode scratch")
        rows = plan.rows
        position_host = np.ascontiguousarray(plan.positions)
        token_weight = self.resident.weight("root.token_embedding")
        embedding_kernel = resolve(
            backend=self.backend,
            layer="embedding",
            quant=token_weight.spec.quant_key,
            variant="lookup_bf16_out",
        )
        copy_host_to_device(
            self.token_ids,
            host_array_ptr(tokens),
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
            copy_host_to_device(
                self.ple_embedding,
                host_array_ptr(np.ascontiguousarray(staged)),
                staged.nbytes,
                runtime=self.runtime,
            )
            self._ple_hash_states = ple_states
        attention.prepare(plan.slots, position_host)
        for index_state in self.index_states:
            index_state.prepare(plan.slots)
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
                ).ptr
            else:
                qsa_binding = self.qsa_bindings[layer]
                residual_ptr = run_qwen4_exp_batch_qsa_layer(
                    residual_ptr,
                    qsa_binding,
                    attention_state=attention,
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
                ).ptr
        head_read = run_qwen4_exp_gr_read(
            residual_ptr,
            self.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
            self.resident.weight("root.head_hc_down"),
            self.resident.weight("root.head_hc_up"),
            None,
            head_scratch,
            rows=rows,
            branches=cfg.residual_branch_count,
            hidden=cfg.hidden_size,
            low_rank=cfg.residual_low_rank,
            runtime=self.runtime,
        )
        launch_gguf_linear(
            self.resident.weight("root.lm_head"),
            head_read.mixed.ptr,
            self.logits.ptr,
            rows,
            cfg.hidden_size,
            cfg.vocab_size,
            activation_dtype=GGUF_ACTIVATION_F32,
            output_dtype=GGUF_OUTPUT_F32,
            runtime=self.runtime,
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
        self, tokens: np.ndarray, slots: Sequence[int] | None
    ) -> _StepPlan:
        """Validate the whole row set before the step mutates anything."""

        attention = self._require_attention()
        cfg = self.config
        rows = int(tokens.size)
        if rows <= 0:
            raise ValueError("a batch decode step needs at least one token")
        if rows > self.state_slots:
            raise ValueError(
                f"step admits at most {self.state_slots} rows, got {rows}"
            )
        if slots is None:
            chosen = self.active_slots
            if not chosen:
                raise RuntimeError(
                    "no slots are admitted; admit a slot before stepping"
                )
        else:
            chosen = tuple(int(slot) for slot in slots)
        if len(chosen) != rows:
            raise ValueError(
                f"slots and token_ids must have equal length, got {len(chosen)} "
                f"and {rows}"
            )
        if len(set(chosen)) != len(chosen):
            raise ValueError("a slot appears twice in one step's row set")
        for row, slot in enumerate(chosen):
            if not 0 <= slot < self.state_slots:
                raise ValueError(
                    f"row {row} slot {slot} is outside [0, {self.state_slots})"
                )
            self._require_admitted(slot)
        if np.any(tokens < 0) or np.any(tokens >= int(cfg.vocab_size)):
            raise ValueError("token ids must be inside the model vocabulary")
        positions = np.ascontiguousarray(self._positions[list(chosen)])
        if np.any(positions < 0) or np.any(positions >= self.max_positions):
            raise ValueError(
                "a step position reached the per-slot decode capacity "
                f"{self.max_positions}"
            )
        needed_blocks = (int(positions.max()) + 1 + self.block_size - 1) // self.block_size
        if needed_blocks > attention.blocks_per_request:
            raise ValueError(
                f"context {int(positions.max()) + 1} needs {needed_blocks} paged "
                f"blocks but each slot holds {attention.blocks_per_request}"
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

    def _require_attention(self) -> Qwen4ExpBatchAttentionState:
        attention = self.attention_state
        if attention is None:
            raise RuntimeError("Qwen4Exp batch runner has no attention state")
        return attention

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self.closed:
            return
        self.closing = True
        # Pop only after a successful free/close, so a raising owner stays
        # reachable and a retry closes the remainder exactly once.
        while self._owned:
            owner = self._owned[-1]
            owner.close()
            self._owned.pop()
        if self._buffers is not None:
            self._buffers.close()
        self.attention_state = None
        self.index_states = ()
        self.gdn_states = ()
        self.ple_state = None
        self.selection = None
        self.gdn_scratch = None
        self.qsa_scratch = None
        self.ple_scratch = None
        self.head_scratch = None
        self.token_ids = None
        self.embedding = None
        self.ple_embedding = None
        self.residual = None
        self.logits = None
        self.argmax_ids = None
        self.argmax_values = None
        self.argmax_blocks_i32 = None
        self.closed = True

    def __enter__(self) -> "Qwen4ExpGGUFBatchRunner":
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


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


__all__ = [
    "QWEN4_EXP_BATCH_BLOCK_SIZE",
    "Qwen4ExpBatchDecodeResult",
    "Qwen4ExpGGUFBatchRunner",
    "run_qwen4_exp_batch_gdn_layer",
    "run_qwen4_exp_batch_qsa_layer",
]
