"""Scheduler-owned proposal, shared target execution and acceptance for Flash-Next."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import time

from hipengine.generation.qwen4_exp_mtp_stages import prepare_proposal, consume_proposal_rows
from hipengine.generation.registry import DecodeState, GenerationTelemetry
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.runtime.qwen4_exp_mixed_layout import Qwen4ExpMixedRequest, compile_mixed_layout
from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass
from hipengine.speculative import (
    AcceptResult, ProviderAttachment, ProviderCatchupMode, SpecCycleResult,
    SpecCycleStage, SpecCycleTransaction, SpecTransactionMode, SpeculativeCapability,
)


class Qwen4ExpMixedCycleError(RuntimeError):
    """A quiesced mixed pass failed.

    ``mutation`` names how much of the pass could have reached device state.
    ``'partial'`` means the pass provably never reached its canonical commit, so
    ``abort()`` rewound every provisional row and only its participating rows
    must retire. ``'unknown'`` means the pass could not establish that window;
    it is never a successful containment claim, so ``contain_execution_failure``
    refuses it and the failure stays fatal for the whole service.
    """

    def __init__(self, request_ids, cause, *, mutation='unknown'):
        self.request_ids = tuple(request_ids)
        self.cause = cause
        self.mutation = mutation
        super().__init__(f'Flash-Next mixed target failed for requests {list(self.request_ids)}: {cause}')


def _device_fault(error):
    """Return the ``HipError`` in ``error``'s cause chain, or ``None``.

    Only ``__cause__`` is followed. A wrapper that reports a device fault keeps
    the original error there by construction, while ``__context__`` can name an
    unrelated exception the same frame happened to handle earlier.
    """

    from hipengine.core.hip import HipError
    seen = set()
    cause = error
    while cause is not None and id(cause) not in seen:
        if isinstance(cause, HipError):
            return cause
        seen.add(id(cause))
        cause = cause.__cause__
    return None


def _mixed_failure_mutation(*, commit_started, error):
    """Name a mixed-pass failure's mutation window, or ``'unknown'`` for none.

    ``'partial'`` is claimed only for a pass that provably never reached
    ``target.commit``: every row it wrote is provisional and ``abort()`` rewound
    it. A ``HipError`` anywhere in the chain means the device itself reported a
    fault, so shared device state is unproven even before the commit.

    A failure raised *by* ``commit`` is never narrowable from here. The pass runs
    its host preflight and its device copies inside that one call, so this layer
    cannot tell a refused commit from a partially applied one. The only signal
    that separates them is the pass's own ``resolved`` flag, which this layer
    does not own: making a containment decision depend on it would turn into an
    over-narrow claim the moment that ordering changes. Unresolved stays fatal.
    """

    if _device_fault(error) is not None:
        return 'unknown'
    return 'unknown' if commit_started else 'partial'


class Qwen4ExpServingMixed:
    """Use the existing scheduler plan for MTP chains and ordinary batch peers."""

    def _mixed_rows_ready(self, request_ids):
        owner = self._ar_batch
        return bool(owner is not None and request_ids and all(
            owner.imported(rid) and self._row(rid).generated_ids
            and (not self._row(rid).mtp_budget or self._row(rid).mixed_mtp)
            for rid in request_ids
        ))

    def _mixed_capability(self, semantics):
        if not self._mixed_rows_ready(tuple(item.request_id for item in semantics)):
            return None
        if not any(self._row(item.request_id).mixed_mtp for item in semantics):
            return None
        max_candidates = min(4, self.generator._speculative_provider.candidate_budget)
        width = min(self.capacity, 64)
        max_rows = width * (max_candidates + 1)
        return SpeculativeCapability(
            capability_key='qwen4exp:shared-mtp-ar', target_key='qwen4_exp_gguf',
            provider_key='qwen4_exp_mtp', method_key='verify_chain',
            policy_fingerprint='qwen4exp:shared-mtp-ar:v1',
            execution_profile=str(getattr(self.generator, 'execution_profile', None) or 'unprofiled'),
            kv_backend_key='bf16', attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
            supported_modes=('verify_chain',), supported_sampling_modes=('greedy', 'processed'),
            max_requests=width, max_candidates_per_request=max_candidates,
            max_frontier_rows=max_rows, proposal_widths=(1,),
            target_row_buckets=tuple(range(1, max_rows + 1)),
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            graph_supported=False, eager_supported=True, strict_fallback_key='qwen4_exp_gguf:ar',
            max_context_tokens=self._ar_batch.max_positions,
        )

    def _mixed_transaction(self, plan, rows, checkpoints):
        spec = [row for row in rows if row.mixed_mtp]
        batch = self._ar_batch.runner
        return SpecCycleTransaction(
            operation_id=plan.operation_id, transaction_id=plan.cycle_id, cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=ResourceClaimSet.from_mapping(
                plan.operation_id,
                {**{f'qwen4exp.slot.{id(batch)}.{self._ar_batch._leases[row.request_id].slot}': 1
                    for row in rows},
                 **{f'qwen4exp.draft.{id(row.draft)}': 1 for row in spec}},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            pre_target_cursors=tuple(int(batch._positions[self._ar_batch._leases[row.request_id].slot]) for row in rows),
            pre_rng_counters=tuple(0 if state is None else state.rng_draw_count for state, _, _, _ in checkpoints),
            target_transaction_mode=plan.target_transaction_mode, target_owner=f'qwen4exp.mixed.{id(batch)}',
            target_checkpoint_ids=tuple(f'{plan.cycle_id}:{row.request_id}' for row in rows),
            pre_provider_cursors=tuple(int(row.draft.position) for row in spec),
            provider_transaction_mode=plan.provider_transaction_mode,
            provider_owner=f'qwen4exp.drafts.{id(self)}',
            provider_request_ids=tuple(row.request_id for row in spec),
            provider_checkpoint_ids=tuple(f'draft:{plan.cycle_id}:{row.request_id}' for row in spec),
            target_open=True, provider_open=True,
        )

    def _execute_mixed_cycle(self, plan):
        from hipengine.generation.qwen4_exp_gguf import _token_finish
        started = stage_started = time.perf_counter()
        stages_ms = {}

        def finish_stage(name):
            nonlocal stage_started
            now = time.perf_counter()
            stages_ms[name] = (now - stage_started) * 1000
            stage_started = now

        rows = [self._row(rid) for rid in plan.request_ids]
        batch = self._ar_batch.runner
        checkpoints = [(
            None if row.sampling_state is None else row.sampling_state.clone(),
            None if row.grammar is None else row.grammar.clone(),
            len(row.token_logprobs), None if not row.mixed_mtp else row.draft.snapshot(),
        ) for row in rows]
        transaction = self._mixed_transaction(plan, rows, checkpoints)
        finish_stage('checkpoint')
        proposals, groups, ar_tokens, ar_finishes = {}, [], {}, {}
        lookup_copies = set()
        lookup_checkpoints = {}
        for row in rows:
            lookup = getattr(row, 'prompt_lookup', None)
            lookup_checkpoints[row.request_id] = (lookup, None if lookup is None else lookup.snapshot())
        target = layout = None
        primary_error = None
        commit_started = False
        try:
            for row, count in zip(rows, plan.candidate_counts, strict=True):
                self._check_request_live(row)
                lease = self._ar_batch._leases[row.request_id]
                position = int(batch._positions[lease.slot])
                if row.mixed_mtp:
                    prepare_vocab = getattr(row.draft, 'prepare_draft_vocabulary', None)
                    if count and callable(prepare_vocab):
                        prepare_vocab([*row.prompt_ids, *row.generated_ids], anchor=row.generated_ids[-1])
                    copied = ()
                    if count and callable(getattr(row.draft, 'propose_lookup_chain', None)):
                        from hipengine.speculative.prompt_lookup import PromptLookup
                        if getattr(row, 'prompt_lookup', None) is None:
                            row.prompt_lookup = PromptLookup(sampled=row.request.temperature > 0)
                        copied = row.prompt_lookup.propose([*row.prompt_ids, *row.generated_ids], count, require_full=True)
                        if copied:
                            lookup_copies.add(row.request_id)
                    proposal = prepare_proposal(
                        row.draft, start_position=position, root_token=row.generated_ids[-1],
                        lookup_candidates=copied,
                        target_hidden_seed=row.mixed_root_hidden, draft_n_max=count,
                        compact_output=True,
                    ) if count else self._mixed_zero_proposal(position, row.generated_ids[-1])
                    proposals[row.request_id] = proposal
                    root, candidates = proposal.root_token, proposal.candidates
                else:
                    root = self.generator._controlled_token(
                        row.next_result, row.request, row.grammar, row.sampling_state, row.token_logprobs,
                    )
                    ar_tokens[row.request_id] = root
                    candidates = ()
                    ar_finishes[row.request_id] = _token_finish(
                        row.request, (*row.generated_ids, root), row.grammar,
                        self.generator.tokenizer.eos_token_id,
                    )
                    if ar_finishes[row.request_id] is not None:
                        continue  # A terminal AR token never needs another target step.
                groups.append(Qwen4ExpMixedRequest(row.request_id, lease.slot, lease.owner,
                                                   position, root, candidates))
            finish_stage('proposal')
            layout = compile_mixed_layout(tuple(groups))
            target = Qwen4ExpMixedTargetPass(batch, layout)
            finish_stage('target_setup')
            scores = target.verify()
            finish_stage('verify')
            accepted, depths = {}, []
            for row in rows:
                if row.mixed_mtp:
                    row_scores = scores.request_rows(row.request_id)
                    acceptance = consume_proposal_rows(
                        proposals[row.request_id], row_scores, prior_tokens=tuple(row.generated_ids),
                        select=lambda result, row=row: self.generator._controlled_token(
                            result, row.request, row.grammar, row.sampling_state, row.token_logprobs,
                        ),
                        finish_for=lambda ids, row=row: _token_finish(
                            row.request, ids, row.grammar, self.generator.tokenizer.eos_token_id,
                        ),
                    )
                    accepted[row.request_id] = acceptance
                    depths.append(acceptance.consumed_rows - 1)
                    proposal = proposals[row.request_id]
                    if acceptance.consumed_rows == proposal.rows:
                        if proposal.candidates:
                            row.draft.forward(
                                proposal.candidates[-1], row_scores.hidden_seeds[-2],
                                capture_logits=False, capture_hidden_seed=False,
                                capture_token_id=False, run_head=False,
                            )
                        else:
                            row.draft.forward(
                                proposal.root_token, row.mixed_root_hidden,
                                capture_logits=False, capture_hidden_seed=False,
                                capture_token_id=False, run_head=False,
                            )
                    row.draft.trim(proposal.start_position + acceptance.consumed_rows)
                else:
                    if ar_finishes[row.request_id] is None:
                        depths.append(0)
            finish_stage('acceptance_catchup')
            for row in rows:
                self._check_request_live(row)
            commit_started = True
            target.commit(tuple(depths))
            finish_stage('commit')
        except BaseException as error:
            primary_error = error
            for row, (state, grammar, log_count, draft) in zip(rows, checkpoints, strict=True):
                lookup, lookup_state = lookup_checkpoints[row.request_id]
                row.prompt_lookup = lookup
                if lookup is not None:
                    lookup.restore(lookup_state)
                row.sampling_state, row.grammar = state, grammar
                del row.token_logprobs[log_count:]
                if draft is not None:
                    row.draft.restore(draft)
            if target is not None:
                target.abort()
            from hipengine.generation.qwen4_exp_serving_mtp import _RequestRetired
            if isinstance(error, _RequestRetired):
                cancelled = tuple(row.request_id for row in rows if self.request_cancel_reason(row.request_id))
                return SpecCycleResult(stage=SpecCycleStage.CANCELLED,
                                       transaction=replace(transaction, rolled_back=True),
                                       cancelled_request_ids=cancelled)
            # Only a successful synchronization permits request-scoped failure
            # containment. A device-wide error remains fatal to the service.
            batch.runtime.device_synchronize()
            affected = plan.request_ids if layout is None else tuple(request.request_id for request in layout.requests)
            raise Qwen4ExpMixedCycleError(
                affected, error,
                mutation=_mixed_failure_mutation(commit_started=commit_started, error=error),
            ) from error
        finally:
            if target is not None:
                try:
                    target.close()
                except BaseException as cleanup:
                    if primary_error is not None:
                        cleanup.add_note(f'original mixed target failure: {primary_error!r}')
                    raise
        finish_stage('cleanup')
        committed_tokens, accept_counts, finish_reasons = [], [], []
        corrections, target_deltas, provider_deltas = [], [], []
        depth_by_request = {request.request_id: depth for request, depth in zip(layout.requests, depths, strict=True)}
        for row in rows:
            depth = depth_by_request.get(row.request_id, -1)
            if row.mixed_mtp:
                acceptance = accepted[row.request_id]
                tokens, count = acceptance.tokens, acceptance.accepted
                row.mixed_root_hidden = acceptance.root_hidden
                row.generated_ids.extend(tokens)
                finish = self._finish(row)
                proposal = proposals[row.request_id]
                row.mixed_proposed += len(proposal.candidates)
                row.mixed_accepted += count
                if row.request_id in lookup_copies:
                    row.prompt_lookup.observe(count)
                row.mixed_cycles += 1
                if finish is not None:
                    from hipengine.generation.registry import GenerationOutput
                    row.mtp_output = GenerationOutput(
                        text=self.generator.tokenizer.decode(row.generated_ids, skip_special=True),
                        generated_token_ids=tuple(row.generated_ids), finish_details=finish,
                        token_logprobs=tuple(row.token_logprobs), telemetry=self._mixed_telemetry(row),
                    )
            else:
                tokens, count = (ar_tokens[row.request_id],), 0
                row.generated_ids.extend(tokens)
                finish = self._finish(row)
                if ar_finishes[row.request_id] is None:
                    physical = layout.physical_row(row.request_id, 0)
                    row.next_result = SimpleNamespace(token_id=int(scores.token_ids[physical]),
                                                     logits=scores.logits[physical])
            committed_tokens.append(tokens[:count])
            accept_counts.append(count)
            corrections.append(tokens[count] if len(tokens) > count else None)
            finish_reasons.append(None if finish is None else finish.reason)
            target_deltas.append(depth + 1)
            provider_deltas.append(depth + 1 if row.mixed_mtp else 0)
            if depth >= 0:
                row.mixed_widths[layout.physical_request_width] = row.mixed_widths.get(layout.physical_request_width, 0) + 1
                row.mixed_target_rows += layout.total_rows
        self._mixed_passes = getattr(self, '_mixed_passes', 0) + 1
        physical_ids = [request.request_id for request in layout.requests]
        counts_by_request = dict(zip(plan.request_ids, plan.candidate_counts, strict=True))
        accepted_by_request = dict(zip(plan.request_ids, accept_counts, strict=True))
        self._last_mixed_pass = dict(
            request_width=layout.physical_request_width, target_rows=layout.total_rows,
            request_ids=physical_ids,
            candidate_counts=[counts_by_request[rid] for rid in physical_ids],
            accepted_draft_tokens=[accepted_by_request[rid] for rid in physical_ids],
            committed_input_rows=[depth + 1 for depth in depths],
            planned_request_ids=list(plan.request_ids),
            planned_candidate_counts=list(plan.candidate_counts),
            stages_ms=stages_ms, timing_scope='host wall; verify includes result readback',
            wall_ms=(time.perf_counter() - started) * 1000,
        )
        samples = getattr(self, '_mixed_pass_samples', None)
        if samples is not None:
            samples.append(self._last_mixed_pass)
        return SpecCycleResult.committed(
            replace(transaction, target_committed=True, provider_committed=True),
            AcceptResult(
                request_ids=plan.request_ids, accepted_counts=tuple(accept_counts),
                accepted_tokens=tuple(committed_tokens), transaction_id=plan.cycle_id,
                correction_or_bonus_tokens=tuple(corrections), target_cursor_deltas=tuple(target_deltas),
                provider_cursor_deltas=tuple(provider_deltas), finish_reasons=tuple(finish_reasons),
            ),
        )

    def contain_execution_failure(self, error, *, phase, request_ids, work_kind):
        """Retire a quiesced mixed pass without disabling unrelated batch slots."""
        if not isinstance(error, Qwen4ExpMixedCycleError):
            return None
        if (
            error.mutation == 'unknown'
            or _device_fault(error) is not None
            or _device_fault(error.cause) is not None
        ):
            # A device fault or unresolved mutation window cannot support a
            # request-local containment claim, regardless of wrapper metadata.
            return None
        if not set(error.request_ids) <= set(request_ids):
            return None
        from hipengine.generation.engine_loop import ExecutionFailure
        return ExecutionFailure(request_ids=error.request_ids, phase=phase, work_kind=work_kind,
                                mutation=error.mutation, reason=str(error), error=error)

    @staticmethod
    def _mixed_zero_proposal(position, root):
        from hipengine.generation.qwen4_exp_mtp_stages import Qwen4ExpMTPProposal
        return Qwen4ExpMTPProposal(position, root, ())

    def _mixed_k0_decode(self, work):
        """Advance one target input per due row when no proposal fits the plan."""
        cancelled = tuple(rid for rid in work.request_ids if self.request_cancel_reason(rid))
        if cancelled:
            survivors = tuple(rid for rid in work.request_ids if rid not in cancelled)
            if not survivors:
                return ()
            from hipengine.generation.engine_loop import ResidentEngineLoop
            return self._mixed_k0_decode(ResidentEngineLoop._decode_work_subset(work, survivors))
        if not self._mixed_rows_ready(work.request_ids):
            ready = tuple(rid for rid in work.request_ids
                          if self._ar_batch is not None and self._ar_batch.imported(rid)
                          and self._row(rid).generated_ids
                          and (not self._row(rid).mtp_budget or self._row(rid).mixed_mtp))
            if not any(self._row(rid).mixed_mtp for rid in ready):
                return None
            from hipengine.generation.engine_loop import ResidentEngineLoop
            ready_set = set(ready)
            mixed = self._mixed_k0_decode(ResidentEngineLoop._decode_work_subset(work, ready))
            remaining = tuple(rid for rid in work.request_ids if rid not in ready_set)
            rest = self.decode_batch(ResidentEngineLoop._decode_work_subset(work, remaining), commit=True)
            return (*mixed, *rest)
        if not any(self._row(rid).mixed_mtp for rid in work.request_ids):
            return None
        cycle = getattr(self, '_mixed_k0_sequence', 0) + 1
        self._mixed_k0_sequence = cycle
        result = self._execute_mixed_cycle(SimpleNamespace(
            request_ids=tuple(work.request_ids), candidate_counts=(0,) * len(work.request_ids),
            operation_id=f'qwen4exp-mixed-k0:{cycle}', cycle_id=cycle,
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        ))
        if result.stage is SpecCycleStage.CANCELLED:
            survivors = tuple(rid for rid in work.request_ids if rid not in result.cancelled_request_ids)
            if not survivors:
                return ()
            from hipengine.generation.engine_loop import ResidentEngineLoop
            return self._mixed_k0_decode(ResidentEngineLoop._decode_work_subset(work, survivors))
        return tuple(self._published_token(self._row(rid), token, self._finish(self._row(rid)))
                     for rid, token in zip(work.request_ids, result.accept_result.correction_or_bonus_tokens, strict=True))

    def _mixed_telemetry(self, row):
        execution_path = ('qwen4exp_shared_mtp_ar_target' if row.mtp_budget
                          else 'qwen4exp_shared_target_ar_peer')
        provider = ({'speculative_provider': 'qwen4_exp_mtp',
                     'cycle_count': row.mixed_cycles} if row.mtp_budget else {})
        return GenerationTelemetry(
            decode_state=DecodeState(prompt_tokens=len(row.prompt_ids), generated_tokens=len(row.generated_ids),
                                     execution_path=execution_path),
            diagnostics=dict(execution_path=execution_path, **provider,
                             candidate_budget=row.mtp_budget, proposed_draft_tokens=row.mixed_proposed,
                             accepted_draft_tokens=row.mixed_accepted, cycles_count=row.mixed_cycles,
                             prompt_lookup=(row.prompt_lookup.diagnostics()
                                 if getattr(row, 'prompt_lookup', None) is not None else {}),
                             draft_vocabulary=(row.draft.draft_vocabulary_diagnostics()
                                 if callable(getattr(getattr(row, 'draft', None), 'draft_vocabulary_diagnostics', None)) else {}),
                             physical_request_width_histogram=dict(row.mixed_widths),
                             physical_target_rows=row.mixed_target_rows),
        )
