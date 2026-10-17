"""Request-owned Flash-Next MTP steps for the resident scheduler.

Physical execution is C1. Concurrent logical requests use independent pooled
runners and the scheduler's existing serial partition hook, not candidate rows
masquerading as multi-request execution.
"""
from __future__ import annotations

from dataclasses import replace

from hipengine.generation.deadline import generation_deadline_expired
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner
from hipengine.speculative import (
    AcceptResult, ProviderAttachment, ProviderCatchupMode, SpecCycleResult,
    SpecCycleStage, SpecCycleTransaction, SpecTransactionMode, SpeculativeCapability,
)


class _RequestRetired(Exception):
    """Local unwind only; the scheduler owns cancellation and reclamation."""


class Qwen4ExpServingMTP:
    """Mixin using the resident runner's row table and target pool."""

    def _init_mtp(self):
        self._all_drafts = []
        self._available_drafts = []
        self._mtp_snapshot_ready = set()

    def _prepare_mtp(self):
        provider = self.generator._speculative_provider
        if provider is None:
            return
        if not callable(getattr(provider, "iter_generate", None)):
            raise NotImplementedError("Flash-Next serving requires a cooperative MTP provider")
        if not self._all_drafts:
            self._all_drafts.append(provider.draft)
            self._available_drafts.append(provider.draft)
        while len(self._all_drafts) < self.capacity:
            draft = Qwen4ExpGGUFMTPDraftRunner(
                provider._resident, target_config=self.generator.runner.config,
                max_sequence_length=self.generator.runner.max_sequence_length,
                backend=self.generator.backend, runtime=self.generator.runner.runtime,
            )
            self._all_drafts.append(draft)
            self._available_drafts.append(draft)
        # Allocate reusable recurrent snapshots at model load, not admission.
        for target in self._all_runners:
            if id(target) not in self._mtp_snapshot_ready:
                checkpoint = target.begin_device_transaction(reuse_snapshot=True)
                target.commit_device_transaction(checkpoint)
                self._mtp_snapshot_ready.add(id(target))

    def bind_request_lifetime(self, request_ids, original_request):
        for request_id in request_ids:
            self._row(request_id).lifetime_request = original_request

    def request_cancel_reason(self, request_id):
        row = self._row(request_id)
        request = row.lifetime_request or row.request
        if generation_deadline_expired(request.deadline_at):
            return "timeout"
        token = request.cancellation_token
        if token is not None and (token.cancel_requested or token.cancelled):
            return "timeout" if token.finish_details.deadline_exceeded else "disconnect"
        return None

    def _check_request_live(self, row):
        if self.request_cancel_reason(row.request_id) is not None:
            raise _RequestRetired()

    def register_speculative_request(self, request_id, candidate_budget, *, static_eligibility=None):
        del static_eligibility
        row = self._row(request_id)
        request = row.request
        provider = self.generator._speculative_provider
        if provider is None:
            raise NotImplementedError("Flash-Next MTP provider is not attached")
        if (request.grammar is not None or request.stop_token_ids or request.stop_token_sequences
                or request.min_tokens or request.suppress_token_ids or request.logit_bias
                or request.forced_tokens_pending or request.post_thinking_forced_tokens_pending
                or request.force_sequence_completion_token_sequences
                or request.json_object_close_forcing or request.tool_call_constraint
                or request.thinking_hard_token_cap is not None
                or request.thinking_soft_close_window
                or request.resident_session_key
                or request.repetition_penalty != 1.0 or request.presence_penalty
                or request.frequency_penalty or request.logprobs or request.top_logprobs):
            raise NotImplementedError("Flash-Next staged MTP does not yet support these token controls")
        if (request.temperature != 0.0 or request.top_k not in (0, 1)
                or request.top_p != 1.0 or request.min_p != 0.0):
            raise NotImplementedError("Flash-Next staged MTP supports raw greedy controls only")
        if request.eos_token_id not in (None, self.generator.tokenizer.eos_token_id):
            raise NotImplementedError("Flash-Next MTP does not support an alternate EOS control")
        if request.kv_storage not in ("auto", "bf16"):
            raise NotImplementedError("Flash-Next MTP requires BF16 KV")
        if len(row.prompt_ids) + request.max_tokens > min(
            self.generator.runner.max_sequence_length, provider.draft.max_sequence_length
        ):
            raise ValueError("Flash-Next MTP request exceeds target/draft capacity")
        row.mtp_budget = min(int(candidate_budget), provider.candidate_budget)

    def speculative_desired_candidate_count(self, request):
        return self.generator._speculative_provider.candidate_budget

    @property
    def speculative_capability(self):
        # Method presence is the submit/poll capability probe. Do not advertise
        # staged MTP when this target was loaded without a sidecar.
        return self._mtp_capability if self.generator._speculative_provider is not None else None

    def _mtp_capability(self, semantics):
        if len(semantics) != 1:
            return None
        row = self._row(semantics[0].request_id)
        if not row.mtp_budget or not row.generated_ids:
            return None  # The prompt's initial root is emitted by decode_batch.
        return SpeculativeCapability(
            capability_key="qwen4exp:request-owned-mtp:c1",
            target_key="qwen4_exp_gguf", provider_key="qwen4_exp_mtp",
            method_key="verify_chain", policy_fingerprint="qwen4exp:bounded-mtp:v1",
            execution_profile=str(getattr(self.generator, "execution_profile", None) or "unprofiled"),
            kv_backend_key="bf16", attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
            supported_modes=("verify_chain",), supported_sampling_modes=("greedy",),
            max_requests=1, max_candidates_per_request=row.mtp_budget,
            max_frontier_rows=row.mtp_budget + 1, proposal_widths=(1,),
            target_row_buckets=tuple(range(1, row.mtp_budget + 2)),
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
            graph_supported=False, eager_supported=True,
            strict_fallback_key="qwen4_exp_gguf:ar",
            max_context_tokens=row.runner.max_sequence_length,
        )

    def speculative_eos_supported(self, request_id):
        return bool(self._row(request_id).mtp_budget)

    def speculative_partition_max_requests(self, work):
        return 1

    def speculative_claims_fit(self, plan):
        if len(plan.request_ids) != 1:
            return False
        row = self._row(plan.request_ids[0])
        return row.runner is not None and row.draft is not None and row.mtp_steps is not None

    def _reserve_mtp(self, row):
        if row.mtp_budget and row.request.max_tokens:
            if not self._available_drafts:
                raise RuntimeError("Flash-Next scheduler has no free draft owner")
            row.draft = self._available_drafts.pop()
            row.draft.reset()

    def _prefill_mtp(self, row, chunk, result):
        row.draft.prime_prompt_chunk(
            chunk, result.hidden_seeds, previous_hidden=row.previous_hidden,
            check_cancelled=lambda: self._check_request_live(row),
        )
        self._check_request_live(row)
        row.previous_hidden = result.hidden_seed
        if row.prefill_tokens_seen + len(chunk) == len(row.prompt_ids):
            # Lifetime is checked here and around each bounded step. Never let
            # a row's cancellation exception escape through a shared poll.
            request = replace(row.request, deadline_at=None, cancellation_token=None)
            row.mtp_steps = self.generator._speculative_provider.iter_generate(
                row.prompt_ids, request, target=row.runner, draft=row.draft,
                prefill_result=result, draft_primed=True,
            )

    def _advance_mtp(self, row, budget, *, plan=None):
        """Return (progress, transaction) after commit or cancellation rollback."""
        target, draft = row.runner, row.draft
        target_checkpoint = target.begin_device_transaction(reuse_snapshot=True)
        draft_checkpoint = draft.snapshot()
        transaction = None
        if plan is not None:
            transaction = SpecCycleTransaction(
                operation_id=plan.operation_id, transaction_id=plan.cycle_id,
                cycle_id=plan.cycle_id, request_ids=plan.request_ids,
                reserved_claims=ResourceClaimSet.from_mapping(
                    plan.operation_id,
                    {f"qwen4exp.target.{id(target)}": 1, f"qwen4exp.draft.{id(draft)}": 1},
                    request_id=row.request_id, lifetime=ClaimLifetime.TRANSACTION,
                ),
                pre_target_cursors=(target_checkpoint.position,), pre_rng_counters=(0,),
                target_transaction_mode=plan.target_transaction_mode,
                target_owner=f"qwen4exp.target.{id(target)}",
                target_checkpoint_ids=(f"device:{id(target_checkpoint)}",),
                pre_provider_cursors=(draft_checkpoint.position,),
                provider_transaction_mode=plan.provider_transaction_mode,
                provider_owner=f"qwen4exp.draft.{id(draft)}",
                provider_request_ids=plan.request_ids,
                provider_checkpoint_ids=(f"cursor:{id(draft_checkpoint)}",),
                target_open=True, provider_open=True,
            )
        try:
            self._check_request_live(row)
            progress = (next(row.mtp_steps) if not row.generated_ids
                        else row.mtp_steps.send(budget))
            terminal = None
            if progress.finished:
                try:
                    next(row.mtp_steps)
                except StopIteration as stopped:
                    terminal = stopped.value
                else:
                    raise RuntimeError("Flash-Next finished iterator emitted another step")
            if (not progress.token_ids or len(row.generated_ids) + len(progress.token_ids) > row.request.max_tokens
                    or int(target.position) != int(draft.position)):
                raise RuntimeError("Flash-Next MTP step violated output/cursor bounds")
            if plan is not None:
                cycle = progress.cycle
                if (cycle is None or len(cycle.candidates) != budget
                        or not 0 <= cycle.accepted <= len(progress.token_ids)
                        or len(progress.token_ids) - cycle.accepted > 1):
                    raise RuntimeError("Flash-Next MTP step differs from the planned cycle")
            self._check_request_live(row)
        except BaseException as error:
            target.rollback_device_transaction(target_checkpoint)
            draft.restore(draft_checkpoint)
            row.mtp_steps.close()
            if not isinstance(error, _RequestRetired):
                raise
            return None, None if transaction is None else replace(transaction, rolled_back=True)
        target.commit_device_transaction(target_checkpoint)
        row.generated_ids.extend(progress.token_ids)
        row.mtp_output = terminal
        if transaction is not None:
            transaction = replace(transaction, target_committed=True, provider_committed=True)
        return progress, transaction

    def execute_speculative_cycle(self, plan, *, commit):
        if not commit or len(plan.request_ids) != 1 or not self.speculative_claims_fit(plan):
            raise ValueError("Flash-Next MTP requires one reserved request and commit=True")
        row = self._row(plan.request_ids[0])
        progress, transaction = self._advance_mtp(row, plan.candidate_counts[0], plan=plan)
        if progress is None:
            return SpecCycleResult(stage=SpecCycleStage.CANCELLED, transaction=transaction,
                                   cancelled_request_ids=plan.request_ids)
        cycle = progress.cycle
        if cycle is None:
            raise RuntimeError("Flash-Next speculative step did not execute a cycle")
        ids = progress.token_ids
        accepted = cycle.accepted
        accept = AcceptResult(
            request_ids=plan.request_ids, accepted_counts=(accepted,),
            accepted_tokens=(ids[:accepted],), transaction_id=transaction.transaction_id,
            correction_or_bonus_tokens=(ids[accepted] if len(ids) > accepted else None,),
            target_cursor_deltas=(row.runner.position - transaction.pre_target_cursors[0],),
            provider_cursor_deltas=(row.draft.position - transaction.pre_provider_cursors[0],),
            finish_reasons=(row.mtp_output.finish_details.reason if row.mtp_output else None,),
        )
        return SpecCycleResult.committed(transaction, accept)

    def _release_mtp(self, row):
        if row.mtp_steps is not None:
            row.mtp_steps.close()
            row.mtp_steps = None
        draft, row.draft = row.draft, None
        if draft is not None and draft not in self._available_drafts:
            self._available_drafts.append(draft)

    def _close_mtp(self):
        provider = self.generator._speculative_provider
        for draft in reversed(self._all_drafts):
            if provider is None or draft is not provider.draft:
                draft.close()
        self._all_drafts.clear()
        self._available_drafts.clear()
