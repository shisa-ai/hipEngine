"""One provisional full-model target pass over MTP chains and AR peers.

The existing batch runner owns live request slots. This executor projects every
physical row together, then uses depth-ordered state adapters. Verification does
not publish positions, recurrent state, PLE history or n-gram history. Attention
appends remain hidden after abort by restoring their index cursors. A commit
publishes one selected prefix per request; every commit precondition is checked on
the host before the first live write, so a rejected commit leaves the pass
abortable, while a device failure during commit poisons all participating slots
rather than permitting a partially applied retry. The stateless row slabs and
per-layer stage scratch belong to the batch runner and are borrowed here, so
closing a pass releases its provisional state and the borrow while those buffers
survive for the next pass; the pass still releases its own state even when the
live runner it borrowed has already gone away.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer, copy_device_to_host, copy_host_array_to_device, free,
    host_array_ptr,
)
from hipengine.kernels.cpu_reference.qwen4_exp import ple_hash_rows
from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
    QWEN4_EXP_PLE_INDEXED_MAX_ROWS, QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    QWEN4_EXP_CONV_INDEXED_MAX_ROWS, QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
)
from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
    QWEN4_EXP_GDN_INDEXED_MAX_ROWS, QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS,
)
from hipengine.kernels.registry import resolve
from hipengine.runtime.gguf_linear import GGUF_ACTIVATION_F32, GGUF_OUTPUT_F32, launch_gguf_linear
from hipengine.runtime.qwen4_exp_batch_gdn import (
    Qwen4ExpBatchGdnMixerPlan, run_qwen4_exp_batch_gdn_input_projections,
    run_qwen4_exp_batch_gdn_output_projection,
)
from hipengine.runtime.qwen4_exp_batch_qsa import (
    Qwen4ExpBatchQsaMixerPlan, run_qwen4_exp_batch_qsa_input_projections,
)
from hipengine.runtime.qwen4_exp_batch_runner import _run_layer_ffn
from hipengine.runtime.qwen4_exp_mixed_gdn import (
    Qwen4ExpMixedGdnAdapter, Qwen4ExpMixedGdnProjectedRows,
)
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedLayout
from hipengine.runtime.qwen4_exp_mixed_ple import Qwen4ExpMixedPleAdapter
from hipengine.runtime.qwen4_exp_mixed_qsa import (
    Qwen4ExpMixedQsaScratch, _commit_counters, _require_commit_owner,
    abort_qwen4_exp_mixed_qsa_attention, commit_qwen4_exp_mixed_qsa_attention,
    run_qwen4_exp_mixed_qsa_attention,
)
from hipengine.runtime.qwen4_exp_runner import (
    qwen4_exp_repeat_bf16_branches, run_qwen4_exp_gr_read, stage_qwen4_exp_ple_rows,
)


@dataclass(frozen=True)
class Qwen4ExpMixedTargetResult:
    """Target scores and residual seeds in the layout's physical row order."""

    layout: Qwen4ExpMixedLayout
    token_ids: np.ndarray
    logits: np.ndarray
    hidden_seeds: np.ndarray

    def request_rows(self, request_id):
        indices = [row.physical_row for row in self.layout.rows_for_request(request_id)]
        from hipengine.generation.qwen4_exp_mtp_stages import Qwen4ExpMTPTargetRows
        return Qwen4ExpMTPTargetRows(
            tuple(int(x) for x in self.token_ids[indices]),
            tuple(self.logits[indices]), self.hidden_seeds[indices],
        )


class Qwen4ExpMixedTargetPass:
    """Own scratch and provisional state for exactly one validated layout."""

    def __init__(self, batch, layout: Qwen4ExpMixedLayout):
        self.batch, self.layout, self.runtime = batch, layout, batch.runtime
        self._owned = []
        self.scratch = None
        self._scratch_released = False
        self.gdn = {}
        self.qsa = {}
        self.qsa_runs = []
        self.ple = {}
        self.hash_snapshots = {}
        self.verified = self.resolved = self.closed = self.closing = False
        self._validate()
        rows = layout.total_rows
        try:
            # Stateless row slabs and stage scratch are the batch runner's and
            # outlive this pass; only the provisional state below is allocated
            # here and released by ``close``.
            self._borrow_scratch(rows)
            cfg = batch.config
            for layer, binding in batch.gdn_bindings.items():
                self.gdn[layer] = self._own(Qwen4ExpMixedGdnAdapter.allocate(
                    layout=layout, live_state=batch.gdn_states[binding.gdn_state_index],
                    owner_of=self._owner, runtime=self.runtime,
                ))
            for layer, binding in batch.qsa_bindings.items():
                index = binding.qsa_state_index
                self.qsa[layer] = self._own(Qwen4ExpMixedQsaScratch.allocate(
                    layout=layout, attention_state=batch.attention_states[index],
                    index_state=batch.index_states[index], hidden=cfg.hidden_size,
                    query_heads=cfg.attention_head_count, kv_heads=cfg.attention_kv_head_count,
                    head_dim=cfg.attention_key_length, runtime=self.runtime,
                ))
            for layer in batch._ple_layers:
                self.ple[layer] = self._own(Qwen4ExpMixedPleAdapter(
                    layout, batch.ple_state, owner_of=self._owner,
                ))
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup:
                error.add_note(f'mixed target allocation cleanup failed: {cleanup!r}')
            raise

    def _owner(self, slot):
        return self.batch._owners[slot]

    def _validate(self):
        self.batch._require_open()
        self.layout.require_verifier_compatible()
        self._require_row_capacity()
        if len(self.batch._ple_layers) > 1:
            raise NotImplementedError('mixed target does not implement multiple shared PLE histories')
        for request in self.layout.requests:
            self.batch._require_slot(request.slot)
            self.batch._require_owner(request.slot, request.owner_token)
            self.batch._require_healthy(request.slot)
            if int(self.batch._positions[request.slot]) != request.start_position:
                raise ValueError('mixed target layout start does not match the live slot cursor')
            if self.batch._ple_layers:
                self._require_ple_hash_cursor(request)
            end = request.start_position + len(request.candidates) + 1
            if end > self.batch.max_positions:
                raise ValueError('mixed target chain exceeds the slot position capacity')
            if self.batch.blocks_for_positions(end) > self.batch._slot_blocks[request.slot]:
                raise ValueError('mixed target chain exceeds the slot block reservation')
        if any(row.token >= self.batch.config.vocab_size for row in self.layout.rows):
            raise ValueError('mixed target token is outside the vocabulary')
        for index in self.batch.index_states:
            if index.pending_rollback is not None:
                raise RuntimeError('mixed target cannot enter a pending index transaction')
            for request in self.layout.requests:
                if int(index.cursor_host[request.slot]) != request.start_position:
                    raise ValueError('mixed target index cursor does not match the layout')

    def _require_row_capacity(self):
        """Refuse a pass wider than the indexed primitives it will dispatch.

        Every indexed state primitive in this pass runs one dispatch per depth
        wave, and wave ``0`` holds every request, so the widest dispatch is the
        layout's physical request width — never its total row count. A chain of
        ``d`` candidates costs ``d + 1`` physical rows but still one row per
        dispatch, which is why a deep pass is legal at a narrow request width.

        The provisional state slabs the adapters materialise are per physical
        row instead, so those are bounded by the primitives' state-slot
        constants. Only the families this batch actually dispatches are checked:
        a pass that runs no GDN, Conv or PLE stage is bounded by its attention
        owners' own admission instead.
        """

        width = self.layout.physical_request_width
        rows = self.layout.total_rows
        bounds = []
        if self.batch.gdn_bindings:
            bounds.append((
                'indexed GDN decode',
                QWEN4_EXP_GDN_INDEXED_MAX_ROWS, QWEN4_EXP_GDN_INDEXED_MAX_STATE_SLOTS,
            ))
            bounds.append((
                'indexed Conv decode',
                QWEN4_EXP_CONV_INDEXED_MAX_ROWS, QWEN4_EXP_CONV_INDEXED_MAX_STATE_SLOTS,
            ))
        if self.batch._ple_layers:
            bounds.append((
                'indexed PLE decode',
                QWEN4_EXP_PLE_INDEXED_MAX_ROWS, QWEN4_EXP_PLE_INDEXED_MAX_STATE_SLOTS,
            ))
        for name, max_rows, max_slots in bounds:
            if width > max_rows:
                raise ValueError(
                    f'mixed target widest depth wave holds {width} row(s), past the '
                    f'{name} dispatch bound of {max_rows}; a wave carries one row per '
                    'request, so this is the physical request width'
                )
            if rows > max_slots:
                raise ValueError(
                    f'mixed target pass holds {rows} physical row(s), past the {name} '
                    f'state slab bound of {max_slots}'
                )

    def _require_ple_hash_cursor(self, request):
        """Refuse a slot whose n-gram history does not start where its chain does.

        ``ple_hash_rows`` silently substitutes an EOS-padded history when the
        stored state's ``next_position`` is not the row's position, so an
        out-of-sync history would produce wrong PLE embeddings instead of an
        error. A slot with no stored state is a request that has produced no
        token yet, which is consistent only with a start position of zero.
        """
        state = self.batch._ple_hash_states.get(request.slot)
        if state is None:
            if request.start_position:
                raise ValueError(
                    f'mixed target slot {request.slot} holds no PLE n-gram history '
                    f'but request {request.request_id} starts at position '
                    f'{request.start_position}; its PLE embeddings would be built '
                    'from an EOS-padded history'
                )
            return
        stored = int(state.next_position)
        if stored != request.start_position:
            raise ValueError(
                f'mixed target slot {request.slot} PLE n-gram history starts at '
                f'position {stored} but request {request.request_id} starts at '
                f'{request.start_position}; the history does not describe this chain'
            )

    def _borrow_scratch(self, rows: int) -> None:
        """Name the batch-owned stateless scratch this pass runs on.

        The set depends only on the row count and the model geometry, so the
        batch runner keeps one at the widest capacity admitted so far and every
        pass borrows it instead of allocating its own. The pass never frees it:
        ``close`` drops the borrow and the batch runner releases the bytes.
        """

        scratch = self.batch.mixed_target_scratch(rows).borrow()
        self.scratch = scratch
        self.token_ids = scratch.token_ids
        self.embedding = scratch.embedding
        self.ple_embedding = scratch.ple_embedding
        self.residual = scratch.residual
        self.logits = scratch.logits
        self.gdn_scratch = scratch.gdn
        self.qsa_scratch = scratch.qsa
        self.head = scratch.head

    def _own(self, owner):
        self._owned.append(owner)
        return owner

    def _stage_hashes(self):
        cfg = self.batch.config
        states = dict(self.batch._ple_hash_states)
        embeddings = np.empty((self.layout.total_rows, cfg.hidden_size), dtype=np.float32)
        for depth in range(self.layout.max_chain_rows):
            wave = [row for row in self.layout.rows if row.depth == depth]
            hashes, states = ple_hash_rows(
                [row.token for row in wave], positions=[row.position for row in wave],
                sequence_ids=[row.slot for row in wave], states=states,
                eos_token_id=cfg.ple_eos_token_id,
                layer_multipliers=cfg.ple_layer_multipliers, head_offsets=cfg.ple_head_offsets,
                head_vocab_sizes=cfg.ple_head_vocab_sizes, heads_per_ngram=cfg.ple_heads_per_ngram,
                ngram_size=cfg.ple_ngram_size,
            )
            # The pinned gather buffer must be consumed before the next gather.
            staged = stage_qwen4_exp_ple_rows(self.batch.resident.ple_staging, hashes, hidden=cfg.hidden_size)
            embeddings[[row.physical_row for row in wave]] = staged
            for row in wave:
                self.hash_snapshots[row.physical_row] = states[row.slot]
        copy_host_array_to_device(
            self.ple_embedding, embeddings, embeddings.nbytes,
            runtime=self.runtime,
        )

    def verify(self) -> Qwen4ExpMixedTargetResult:
        """Execute shared row projections and provisional depth-wave state work."""
        if self.closed or self.closing or self.resolved or self.verified:
            raise RuntimeError('mixed target verification requires a fresh open pass')
        self._validate()
        batch, cfg, rows = self.batch, self.batch.config, self.layout.total_rows
        tokens = np.ascontiguousarray([row.token for row in self.layout.rows], dtype=np.int64)
        token_weight = batch.resident.weight('root.token_embedding')
        embedding = resolve(backend=batch.backend, layer='embedding',
                            quant=token_weight.spec.quant_key, variant='lookup_bf16_out')
        try:
            copy_host_array_to_device(
                self.token_ids, tokens, tokens.nbytes, runtime=self.runtime,
            )
            embedding(self.token_ids.ptr, token_weight.allocation('raw').tensor.ptr,
                      self.embedding.ptr, rows, cfg.hidden_size, cfg.vocab_size, runtime=self.runtime)
            qwen4_exp_repeat_bf16_branches(
                self.embedding.ptr, self.residual.ptr, cfg.residual_branch_count,
                cfg.hidden_size, rows=rows, runtime=self.runtime,
            )
            if self.ple:
                self._stage_hashes()
            residual = self.residual
            for layer, kind in enumerate(cfg.layer_types):
                if layer in self.ple:
                    prefix = f'layers.{layer}.'
                    residual = self.ple[layer].verify(
                        residual, self.ple_embedding,
                        {name: batch.resident.weight(prefix + name) for name in ('ple_key', 'ple_value')},
                        **{name + '_ptr': batch.resident.weight(prefix + slot).allocation('raw').tensor.ptr
                           for name, slot in (('norm_key', 'ple_norm_key'), ('norm_query', 'ple_norm_query'),
                                              ('norm_conv', 'ple_norm_conv'), ('conv_weight', 'ple_conv1d'))},
                    )
                binding = batch.gdn_bindings[layer] if kind == 'gdn' else batch.qsa_bindings[layer]
                scratch = self.gdn_scratch if kind == 'gdn' else self.qsa_scratch
                read = run_qwen4_exp_gr_read(
                    residual.ptr, binding.attention_gr.norm_weight_ptr,
                    binding.attention_gr.down, binding.attention_gr.up, binding.attention_gr.inject,
                    scratch.attention_gr, rows=rows, branches=cfg.residual_branch_count,
                    hidden=cfg.hidden_size, low_rank=cfg.residual_low_rank, runtime=self.runtime,
                )
                if kind == 'gdn':
                    adapter = self.gdn[layer]
                    plan = Qwen4ExpBatchGdnMixerPlan(self.runtime, rows, adapter.geometry.qkv_width,
                                                    adapter.geometry.core_width, 0)
                    run_qwen4_exp_batch_gdn_input_projections(
                        plan, read.mixed.ptr, binding.mixer.projections,
                        state=adapter.live_state, scratch=scratch.gdn, hidden=cfg.hidden_size,
                    )
                    core = adapter.run(
                        Qwen4ExpMixedGdnProjectedRows(
                            scratch.gdn.qkv, scratch.gdn.gate, scratch.gdn.alpha, scratch.gdn.beta),
                        conv_weight_ptr=binding.mixer.conv_weight_ptr,
                        dt_bias_ptr=binding.mixer.dt_bias_ptr, a_ptr=binding.mixer.a_ptr,
                        norm_weight_ptr=binding.mixer.norm_weight_ptr,
                    )
                    view = SimpleNamespace(core=core, output=scratch.gdn.output)
                    mixed = run_qwen4_exp_batch_gdn_output_projection(
                        plan, binding.mixer.projections, scratch=view, hidden=cfg.hidden_size,
                    )
                else:
                    plan = Qwen4ExpBatchQsaMixerPlan(
                        self.runtime, rows, cfg.attention_head_count * cfg.attention_key_length,
                        cfg.attention_kv_head_count * cfg.attention_key_length, (), (), 0,
                    )
                    run_qwen4_exp_batch_qsa_input_projections(
                        plan, read.mixed.ptr, binding.mixer, scratch=scratch.qsa,
                        hidden=cfg.hidden_size, index_dim=cfg.indexer_key_length,
                    )
                    index = binding.qsa_state_index
                    run = run_qwen4_exp_mixed_qsa_attention(
                        read.mixed.ptr, binding.mixer, layout=self.layout, shared=scratch.qsa,
                        attention_state=batch.attention_states[index], index_state=batch.index_states[index],
                        scratch=self.qsa[layer], owner_of=self._owner, hidden=cfg.hidden_size,
                        query_heads=cfg.attention_head_count, kv_heads=cfg.attention_kv_head_count,
                        head_dim=cfg.attention_key_length, rotary_dim=cfg.rope_dimension_count,
                        theta=cfg.rope_freq_base, index_heads=cfg.indexer_head_count,
                        index_dim=cfg.indexer_key_length, index_rotary_dim=cfg.rope_dimension_count,
                        runtime=self.runtime,
                    )
                    self.qsa_runs.append((index, run))
                    mixed = run.output
                _run_layer_ffn(
                    residual.ptr, mixed.ptr, binding, scratch, rows=rows,
                    branches=cfg.residual_branch_count, hidden=cfg.hidden_size,
                    low_rank=cfg.residual_low_rank, ffn=cfg.expert_feed_forward_length,
                    experts=cfg.expert_count, top_k=cfg.expert_used_count, stream=0, runtime=self.runtime,
                    q51_exact_grouped_down=getattr(
                        batch, "q51_exact_grouped_down", True),
                    q51_down_m1=getattr(batch, "q51_down_m1", False),
                    q51_fold128_prefill=getattr(
                        batch, "q51_fold128_prefill", False),
                    q51_fold_pair_prefill=getattr(
                        batch, "q51_fold_pair_prefill", False),
                    forkb_grouped_down=getattr(
                        batch, "forkb_grouped_down", False),
                    q8_down_row4_prefill=getattr(
                        batch, "q8_down_row4_prefill", False),
                    q8_down_bundle_prefill=getattr(
                        batch, "q8_down_bundle_prefill", False),
                )
                residual = scratch.output
            head = run_qwen4_exp_gr_read(
                residual.ptr, batch.resident.weight('root.head_hc_norm').allocation('raw').tensor.ptr,
                batch.resident.weight('root.head_hc_down'), batch.resident.weight('root.head_hc_up'),
                None, self.head, rows=rows, branches=cfg.residual_branch_count,
                hidden=cfg.hidden_size, low_rank=cfg.residual_low_rank, runtime=self.runtime,
            )
            launch_gguf_linear(
                batch.resident.weight('root.lm_head'), head.mixed.ptr, self.logits.ptr,
                rows, cfg.hidden_size, cfg.vocab_size, activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32, runtime=self.runtime,
            )
            logits = np.empty((rows, cfg.vocab_size), dtype=np.float32)
            bits = np.empty((rows, cfg.residual_width), dtype=np.uint16)
            copy_device_to_host(
                host_array_ptr(logits), self.logits, logits.nbytes, runtime=self.runtime,
            )
            copy_device_to_host(host_array_ptr(bits), residual, bits.nbytes, runtime=self.runtime)
            hidden = np.ascontiguousarray((bits.astype(np.uint32) << 16).view(np.float32))
            self.verified = True
            return Qwen4ExpMixedTargetResult(self.layout, np.argmax(logits, axis=1), logits, hidden)
        except BaseException as error:
            try:
                self.abort()
            except BaseException as cleanup:
                error.add_note(
                    'discarding the failed pass also failed, so the attention and '
                    f'index cursors may still cover its rows: {cleanup!r}'
                )
            raise

    def commit(self, accepted_counts: Sequence[int]):
        """Publish selected prefixes once; device failure poisons participating slots."""
        if self.closed or self.closing or self.resolved or not self.verified:
            raise RuntimeError('mixed target commit requires an unresolved verification')
        plan = self._preflight_commit(accepted_counts)
        self.resolved = True
        try:
            for adapter in self.gdn.values():
                adapter.commit(accepted_counts)
            for adapter in self.ple.values():
                adapter.commit(accepted_counts)
            for index, run in self.qsa_runs:
                commit_qwen4_exp_mixed_qsa_attention(
                    run, accepted_counts, index_state=self.batch.index_states[index], owner_of=self._owner,
                )
            for request, row in zip(self.layout.requests, plan.commit_rows, strict=True):
                self.batch._positions[request.slot] = self.layout.rows[row].position + 1
                if self.ple:
                    self.batch._ple_hash_states[request.slot] = self.hash_snapshots[row]
        except BaseException as error:
            self.batch._poisoned.update(request.slot for request in self.layout.requests)
            try:
                self._abort_indexes()
            except BaseException as cleanup:
                error.add_note(
                    'aborting the failed commit also failed, so the index cursors '
                    f'may still cover its rejected rows: {cleanup!r}'
                )
            raise
        return plan

    def _preflight_commit(self, accepted_counts: Sequence[int]):
        """Validate the whole commit on the host before any live byte is written.

        Every check a committing owner performs is repeated here first: the live
        runner is open, each slot is still the same admission at the same live
        position, and each QSA run still holds a pending index transaction whose
        cursors sit exactly where the pass left them. A refusal therefore leaves
        the live state, the index owners and this pass unchanged, so the caller
        can still abort or correct the call — which is what separates a rejected
        commit from a poisoned one.
        """

        self.batch._require_open()
        plan = self.layout.commit_destinations(accepted_counts)
        for request in self.layout.requests:
            self.batch._require_slot(request.slot)
            self.batch._require_owner(request.slot, request.owner_token)
            self.batch._require_healthy(request.slot)
            if int(self.batch._positions[request.slot]) != request.start_position:
                raise ValueError('mixed target live position changed before commit')
        # ``_require_commit_owner`` and ``_commit_counters`` are the read-only half
        # of ``commit_qwen4_exp_mixed_qsa_attention``: reusing them here means this
        # preflight cannot drift from the checks that commit re-runs, and both are
        # pure host work.
        for index, run in self.qsa_runs:
            state = self.batch.index_states[index]
            _require_commit_owner(run, state)
            _commit_counters(run, state, plan)
        return plan

    def _abort_indexes(self):
        """Rewind every unpublished attention suffix that is still ours to rewind.

        A run whose index owner the live runner has already released — the owner
        is gone from ``index_states``, or it is closed or closing — has no
        counters left to restore, so it is skipped instead of failing. Every
        other failure propagates, and a caller that is already handling an error
        records this one as a note rather than letting it replace the cause.
        """

        for index, run in reversed(self.qsa_runs):
            states = self.batch.index_states
            if not 0 <= index < len(states):
                continue
            state = states[index]
            if state.pending_rollback is not run.rollback:
                continue
            if state.closed or state.closing:
                continue
            abort_qwen4_exp_mixed_qsa_attention(run, index_state=state)
        self.qsa_runs.clear()

    def abort(self):
        """Discard unpublished state; only unreachable attention suffix bytes remain."""
        if not self.resolved:
            self._abort_indexes()
            self.resolved = True

    def _release_borrowed_scratch(self):
        """Drop the borrow even when a failed free keeps the pass retryable.

        A pass that refuses work can never read those buffers again, so the batch
        runner may replace them; keeping the borrow would let one failed close
        block every wider pass for the life of the runner. The reference stays
        for diagnostics, so the release is guarded rather than cleared.
        """

        scratch = getattr(self, 'scratch', None)
        if scratch is not None and not self._scratch_released:
            scratch.release()
            self._scratch_released = True

    def close(self):
        """Release this pass's provisional state and its borrowed scratch.

        ``closing`` is set before the first free, so a partially closed pass
        refuses verification and commit instead of handing out a released
        address. A failure to discard unpublished state does not stop the frees:
        the state is this pass's own, and leaking it would outlive the cause. A
        failed free leaves its owner pending and raises, exactly as the state
        owners in this tree do; the discard failure is then reported as a note on
        that error, or raised once cleanup completed.

        The stateless scratch is not freed here. It belongs to the batch runner
        and outlives the pass, so only the borrow is dropped.
        """

        if self.closed:
            return
        self.closing = True
        try:
            abort_error = None
            try:
                self.abort()
            except BaseException as error:
                abort_error = error
            while self._owned:
                owner = self._owned[-1]
                try:
                    if isinstance(owner, DeviceBuffer):
                        free(owner, runtime=self.runtime)
                    else:
                        owner.close()
                except BaseException as error:
                    if abort_error is not None:
                        error.add_note(
                            'releasing the pass also failed to discard its unpublished '
                            f'state: {abort_error!r}'
                        )
                    raise
                self._owned.pop()
            self.closed = True
            if abort_error is not None:
                raise abort_error
        finally:
            self._release_borrowed_scratch()
