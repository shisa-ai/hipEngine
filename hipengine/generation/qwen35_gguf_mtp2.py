"""Staged dense GGUF NextN/MTP2 adapter for the Generation-2 resident owner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
from hipengine.speculative.frontier import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    TargetFrontier,
)
from hipengine.speculative.interfaces import AcceptResult, TargetCommitPlan
from hipengine.speculative.mtp import MtpProposalContext
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
)


@dataclass(slots=True)
class _MTP2RequestState:
    request_id: int
    provider: Any
    provider_pool_key: Any | None
    verifier: Qwen35GGUFTransactionalVerifier
    root_hidden_buffer: DeviceBuffer
    last_proposal_seconds: float = 0.0


class Qwen35GGUFMTP2Adapter:
    """One-request staged adapter over the retained exact dense components."""

    def __init__(
        self,
        owner: Any,
        *,
        enabled: bool,
        target_verify_mode: str,
        candidate_budget: int,
        quant: str = "gguf_q4_k_m",
    ) -> None:
        self.owner = owner
        self.generator = owner.generator
        self.enabled = bool(enabled)
        self.target_verify_mode = str(target_verify_mode)
        self.candidate_budget = int(candidate_budget)
        self.quant = str(quant)
        if self.candidate_budget not in {1, 2, 3}:
            raise ValueError("MTP2 candidate budget must be 1, 2, or 3")
        self._intents: dict[int, int] = {}
        self._prompt_hidden_rows: dict[int, np.ndarray] = {}
        self._states: dict[int, _MTP2RequestState] = {}
        self._disabled_requests: set[int] = set()
        self._active_claims: ResourceClaimSet | None = None
        self._transaction_sequence = 0

    def register_request(self, request_id: int, candidate_budget: int) -> None:
        rid = int(request_id)
        budget = min(self.candidate_budget, max(1, int(candidate_budget)))
        self._intents[rid] = budget
        self._disabled_requests.discard(rid)

    def observe_prefill_result(self, request_id: int, prompt_ids: Sequence[int], result: Any) -> None:
        rid = int(request_id)
        if rid not in self._intents:
            return
        hidden = getattr(result, "hidden_seeds", None)
        if hidden is None:
            self._prompt_hidden_rows.pop(rid, None)
            return
        rows = np.ascontiguousarray(hidden, dtype=np.float32)
        expected_rows = len(tuple(prompt_ids))
        hidden_size = int(getattr(self.owner._shared_runner, "hidden_size", 0))
        if rows.shape != (expected_rows, hidden_size):
            raise RuntimeError(
                "GGUF MTP2 prefill hidden rows have shape "
                f"{rows.shape}, expected {(expected_rows, hidden_size)}"
            )
        self._prompt_hidden_rows[rid] = rows

    def capability(
        self,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability | None:
        semantics = tuple(request_semantics)
        if not self.enabled or len(semantics) != 1:
            return None
        rid = int(semantics[0].request_id)
        if rid not in self._intents or rid in self._disabled_requests:
            return None
        row = self.owner._row(rid)
        if (
            not row.native_greedy
            or not row.first_token_emitted
            or row.lease is None
            or row.slot is None
            or rid not in self._prompt_hidden_rows
        ):
            return None
        target = row.lease.session
        if bool(
            getattr(getattr(target, "runner", None), "fp16_recurrent_state", False)
        ):
            return None
        max_context = int(target.target_layout.max_sequence_length)
        profile = str(getattr(self.generator, "execution_profile", None) or "legacy_exact")
        return SpeculativeCapability(
            capability_key=(
                f"gguf_mtp2_c1:{self.generator.backend}:{self.quant}:"
                f"{self.target_verify_mode}:{self.candidate_budget}"
            ),
            target_key="qwen_dense_gguf",
            provider_key="qwen_nextn_dense",
            method_key="mtp2",
            policy_fingerprint=(
                f"dense-nextn:{self.target_verify_mode}:b{self.candidate_budget}"
            ),
            execution_profile=profile,
            kv_backend_key=str(getattr(target, "kv_storage_dtype", "bf16")),
            attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
            supported_modes=("verify_chain",),
            supported_sampling_modes=("greedy",),
            max_requests=1,
            max_candidates_per_request=self.candidate_budget,
            max_frontier_rows=self.candidate_budget + 1,
            proposal_widths=(1,),
            target_row_buckets=tuple(range(2, self.candidate_budget + 2)),
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            graph_supported=True,
            eager_supported=True,
            strict_fallback_key="gguf_target_ar",
            max_context_tokens=max_context,
        )

    def claims_fit(self, plan: SpecRequestPlan) -> bool:
        return bool(
            self.enabled
            and self._active_claims is None
            and len(plan.speculative_request_ids) == 1
            and plan.speculative_request_ids[0] not in self._disabled_requests
        )

    def component_claims(
        self,
        plan: SpecRequestPlan,
    ) -> Mapping[str, ResourceClaimSet]:
        rows = int(plan.logical_frontier_rows)
        candidates = int(sum(plan.candidate_counts))
        requests = len(plan.request_ids)
        return {
            "target": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:target",
                {"gguf_mtp2.target_rows": rows},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"gguf_mtp2.provider_rows": candidates},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "transient": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:transient",
                {"gguf_mtp2.result_rows": requests},
                lifetime=ClaimLifetime.WORK_ITEM,
            ),
        }

    def reserve_claims(self, claims: ResourceClaimSet) -> str:
        if self._active_claims is not None:
            raise RuntimeError("GGUF MTP2 claims are already reserved")
        self._active_claims = claims
        return claims.claim_id

    def release_claims(self, reservation: str) -> None:
        if self._active_claims is None or self._active_claims.claim_id != str(reservation):
            raise RuntimeError("GGUF MTP2 claim release does not match active ownership")
        self._active_claims = None

    def prepare_requests(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del stream
        for request_id in plan.speculative_request_ids:
            if request_id not in self._states:
                row = self.owner._row(request_id)
                # Generation-2 packed prefill/decode may still own deferred
                # state in the shared execution owner. The standalone exact c1
                # verifier consumes request-session pointers, so scatter the
                # complete row back only after the operation claims are held.
                self.owner._flush_row_owner(row)
                self._states[request_id] = self._open_request(request_id)

    def propose_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph:
        del stream
        if len(plan.speculative_request_ids) != 1:
            raise NotImplementedError("GGUF MTP2 S3 supports one speculative request")
        rid = int(plan.speculative_request_ids[0])
        state = self._states[rid]
        row = self.owner._row(rid)
        target = row.lease.session
        slot = row.slot
        if slot is None:
            raise RuntimeError("GGUF MTP2 row has no committed root token")
        budget = int(plan.candidate_counts[plan.request_ids.index(rid)])
        context = MtpProposalContext(
            request_ids=(rid,),
            root_tokens=(int(slot.generated_ids[-1]),),
            root_positions=(int(target.position),),
            target_hidden=target.last_target_hidden,
        )
        proposal_started = time.perf_counter()
        draft = state.provider.propose(
            context,
            candidate_budget=budget,
            return_logits=False,
        )
        state.last_proposal_seconds = time.perf_counter() - proposal_started
        parents = draft.tree_parents or tuple(
            -1 if depth == 1 else index - 1
            for index, depth in enumerate(draft.draft_depths)
        )
        counts = tuple(
            sum(1 for owner in draft.row_to_request if owner == request_id)
            for request_id in plan.request_ids
        )
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        semantics_by_id = {item.request_id: item for item in request_semantics}
        return CandidateGraph(
            provider_key=str(plan.provider_key),
            method_key="mtp2",
            policy_fingerprint="dense-nextn-strict",
            cycle_id=plan.cycle_id,
            transaction_id=plan.cycle_id,
            request_ids=plan.request_ids,
            resident_slots=plan.resident_slots,
            root_positions=tuple(
                int(semantics_by_id[request_id].context_tokens) - 1
                for request_id in plan.request_ids
            ),
            row_offsets=tuple(offsets),
            row_to_request=draft.row_to_request,
            parent_candidate_rows=parents,
            draft_depths=draft.draft_depths,
            active_mask=draft.active_mask or (True,) * draft.draft_rows,
            candidate_tokens=draft.candidate_tokens,
            candidate_ids=draft.candidate_ids,
            mode=draft.mode,
            provider_metadata=draft.provider_metadata,
        )

    def execute_target_frontier(
        self,
        plan: SpecRequestPlan,
        frontier: TargetFrontier,
        complete_claims: ResourceClaimSet,
        *,
        commit: bool,
        cancelled_request_ids: Callable[[], Sequence[int]],
    ) -> SpecCycleResult:
        if not commit:
            raise ValueError("GGUF MTP2 target frontier requires commit=True")
        if self._active_claims != complete_claims:
            raise RuntimeError("GGUF MTP2 target does not own complete claims")
        if len(plan.speculative_request_ids) != 1 or frontier.target_batch is None:
            raise NotImplementedError("GGUF MTP2 S3 requires one host-visible chain")
        rid = int(plan.speculative_request_ids[0])
        state = self._states[rid]
        row = self.owner._row(rid)
        target = row.lease.session
        slot = row.slot
        if slot is None:
            raise RuntimeError("GGUF MTP2 target row has no AR root")
        self._transaction_sequence += 1
        transaction_id = self._transaction_sequence
        transaction = SpecCycleTransaction(
            operation_id=plan.operation_id,
            transaction_id=transaction_id,
            cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=complete_claims,
            pre_target_cursors=tuple(
                int(self.owner._row(request_id).lease.session.position)
                for request_id in plan.request_ids
            ),
            pre_provider_cursors=(int(target.position),),
            pre_rng_counters=(0,) * len(plan.request_ids),
            target_transaction_mode=plan.target_transaction_mode,
            provider_transaction_mode=plan.provider_transaction_mode,
            target_owner=f"{plan.operation_id}:gguf-target",
            provider_owner=f"{plan.operation_id}:nextn",
            provider_request_ids=(rid,),
            target_checkpoint_ids=tuple(
                f"target:{request_id}:{self.owner._row(request_id).lease.session.position}"
                for request_id in plan.request_ids
            ),
            provider_checkpoint_ids=(f"provider:{rid}:{target.position}",),
            target_open=True,
            provider_open=True,
        )
        if tuple(int(value) for value in cancelled_request_ids()):
            self._drop_request(rid, disable=True)
            return SpecCycleResult(
                stage=SpecCycleStage.CANCELLED,
                transaction=replace(
                    transaction,
                    target_open=False,
                    provider_open=False,
                    rolled_back=True,
                ),
                cancelled_request_ids=(rid,),
            )
        batch = frontier.target_batch
        remaining = max(0, int(row.request.max_tokens) - len(slot.generated_ids))
        graph_key = (
            "specdec2",
            batch.mode,
            batch.rows,
            plan.execution_route,
        )
        bucket = state.verifier.graph_bucket(graph_key, batch)
        prepared = None
        target_seconds = 0.0
        provider_update_seconds = 0.0
        try:
            target_started = time.perf_counter()
            prepared = state.verifier.prepare(
                batch,
                transaction_id=transaction_id,
                graph_bucket=bucket,
                remaining_decode=(remaining,),
                return_logits=False,
            )
            target_seconds = time.perf_counter() - target_started
            cancelled = tuple(int(value) for value in cancelled_request_ids())
            if cancelled:
                state.verifier.rollback(prepared)
                prepared = None
                self._drop_request(rid, disable=True)
                return SpecCycleResult(
                    stage=SpecCycleStage.CANCELLED,
                    transaction=replace(
                        transaction,
                        target_open=False,
                        provider_open=False,
                        rolled_back=True,
                    ),
                    cancelled_request_ids=cancelled,
                )
            summary = prepared.summary
            commit_plan = TargetCommitPlan(
                transaction_id=transaction_id,
                request_ids=summary.request_ids,
                accepted_counts=summary.accepted_counts,
                commit_rows=summary.commit_rows,
                commit_tokens=summary.commit_tokens,
                commit_positions=summary.commit_positions,
                next_tokens=summary.next_tokens,
                candidate_counts=summary.candidate_counts,
                draft_depth=summary.draft_depth,
                tree_shape=summary.tree_shape,
                mode=summary.mode,
            )
            state.verifier.commit(prepared, commit_plan)
            native_graph_submitted = bool(prepared.native_graph_submitted)
            state.verifier.finish(prepared)
            prepared = None
            accepted = int(summary.accepted_counts[0])
            provider_update_started = time.perf_counter()
            state.provider.advance_full_accept_tail(
                rid,
                accepted_count=accepted,
            )
            provider_update_seconds = time.perf_counter() - provider_update_started
            next_token = None if summary.next_tokens is None else summary.next_tokens[0]
            output_ids = (
                *tuple(int(token) for token in summary.accepted_tokens[0]),
                *(() if next_token is None else (int(next_token),)),
            )
            if not output_ids:
                raise RuntimeError("GGUF MTP2 committed cycle produced no visible token")
            slot.generated_ids.extend(output_ids)
            slot.prev_token = int(output_ids[-1])
            slot.seq_position = int(target.position)
            slot.native_decode_steps += 1
            slot.done = len(slot.generated_ids) >= int(row.request.max_tokens)
            row.mtp2_cycles += 1
            row.mtp2_candidate_counts.append(int(plan.candidate_counts[0]))
            row.mtp2_accepted_counts.append(accepted)
            row.mtp2_proposal_ms += float(state.last_proposal_seconds) * 1000.0
            row.mtp2_target_ms += float(target_seconds) * 1000.0
            row.mtp2_provider_update_ms += float(provider_update_seconds) * 1000.0
            accept = AcceptResult(
                request_ids=plan.request_ids,
                accepted_counts=summary.accepted_counts,
                accepted_tokens=summary.accepted_tokens,
                transaction_id=transaction_id,
                selected_candidate_rows=summary.commit_rows,
                correction_or_bonus_tokens=(
                    None if next_token is None else int(next_token),
                ),
                target_cursor_deltas=(len(output_ids),),
                provider_cursor_deltas=summary.accepted_counts,
                finish_reasons=(None,),
            )
            actual_execution_route = (
                "graph" if native_graph_submitted else "eager"
            )
            telemetry = SpecCycleTelemetry(
                operation_id=plan.operation_id,
                request_ids=plan.request_ids,
                candidate_counts=plan.candidate_counts,
                plan_reasons=plan.reasons,
                proposal_widths=plan.proposal_widths,
                target_row_decomposition=plan.target_row_decomposition,
                execution_route=actual_execution_route,
                proposal_seconds=max(0.0, state.last_proposal_seconds),
                target_seconds=target_seconds,
                provider_update_seconds=provider_update_seconds,
                weight_sweeps=len(plan.target_row_decomposition),
            )
            committed = replace(
                transaction,
                target_committed=True,
                provider_committed=True,
            )
            return SpecCycleResult.committed(
                committed,
                accept,
                telemetry=telemetry,
            )
        except Exception:
            if prepared is not None:
                state.verifier.rollback(prepared)
            raise

    def rollback_cycle(
        self,
        plan: SpecRequestPlan,
        candidate_graph: CandidateGraph | None,
        error: BaseException,
    ) -> None:
        del candidate_graph, error
        for request_id in plan.speculative_request_ids:
            self._drop_request(int(request_id), disable=True)

    def release_request(self, request_id: int) -> None:
        rid = int(request_id)
        self._drop_request(rid, disable=False)
        self._intents.pop(rid, None)
        self._prompt_hidden_rows.pop(rid, None)
        self._disabled_requests.discard(rid)

    def close(self) -> None:
        for request_id in tuple(self._states):
            self._drop_request(request_id, disable=False)
        self._intents.clear()
        self._prompt_hidden_rows.clear()
        self._disabled_requests.clear()
        self._active_claims = None

    def _open_request(self, request_id: int) -> _MTP2RequestState:
        rid = int(request_id)
        row = self.owner._row(rid)
        if row.lease is None:
            raise RuntimeError("GGUF MTP2 request has no target session")
        target = row.lease.session
        if bool(
            getattr(getattr(target, "runner", None), "fp16_recurrent_state", False)
        ):
            raise RuntimeError(
                "GGUF MTP2 strict c1 requires FP32 recurrent state; "
                "disable HIPENGINE_GGUF_FP16_RECURRENT_STATE"
            )
        max_positions = int(target.target_layout.max_sequence_length)
        provider, pool_key, _reused = self.generator._acquire_dense_mtp_draft_provider(
            target,
            max_positions=max_positions,
            pool_enabled=self.owner._shared_runner is not None,
        )
        verifier = None
        root_hidden_buffer = None
        try:
            provider.reset_request(rid)
            root_hidden_buffer = self._catch_up_provider(
                provider,
                rid,
                row.prompt_ids,
                self._prompt_hidden_rows[rid],
                target,
            )
            target._last_target_hidden_ptr = int(root_hidden_buffer.ptr)
            verifier = Qwen35GGUFTransactionalVerifier(
                target,
                max_candidate_budget=self.candidate_budget,
                quant=self.quant,
                target_verify_mode=self.target_verify_mode,
            )
            return _MTP2RequestState(
                request_id=rid,
                provider=provider,
                provider_pool_key=pool_key,
                verifier=verifier,
                root_hidden_buffer=root_hidden_buffer,
            )
        except Exception:
            if verifier is not None:
                verifier.close()
            if root_hidden_buffer is not None:
                free(root_hidden_buffer, runtime=target.runtime)
            provider.release_request(rid)
            self.generator._release_mtp_draft_runner(pool_key, provider)
            raise

    def _catch_up_provider(
        self,
        provider: Any,
        request_id: int,
        prompt_ids: Sequence[int],
        hidden_rows: np.ndarray,
        target: Any,
    ) -> DeviceBuffer:
        rows = np.ascontiguousarray(hidden_rows, dtype=np.float32)
        hidden_size = int(provider.executor.hidden_size)
        if rows.shape != (len(tuple(prompt_ids)), hidden_size):
            raise ValueError("GGUF MTP2 prompt hidden rows do not align")
        hidden_buffer = malloc(hidden_size * DType.BF16.itemsize, runtime=target.runtime)
        try:
            zero_bits = np.zeros((hidden_size,), dtype=np.uint16)
            for position, token in enumerate(prompt_ids):
                hidden_bits = (
                    zero_bits
                    if position == 0
                    else np.ascontiguousarray(
                        float_array_to_bf16_bits(rows[position - 1]),
                        dtype=np.uint16,
                    )
                )
                copy_host_to_device(
                    DeviceBuffer(hidden_buffer.ptr, hidden_bits.nbytes),
                    host_array_ptr(hidden_bits),
                    hidden_bits.nbytes,
                    runtime=target.runtime,
                )
                provider.executor.run_step(
                    int(request_id),
                    int(token),
                    int(position),
                    Tensor.from_handle(
                        hidden_buffer.ptr,
                        (1, hidden_size),
                        DType.BF16,
                        Device("hip", 0),
                    ),
                    return_logits=False,
                )
            final_hidden_bits = np.ascontiguousarray(
                float_array_to_bf16_bits(rows[-1]),
                dtype=np.uint16,
            )
            copy_host_to_device(
                DeviceBuffer(hidden_buffer.ptr, final_hidden_bits.nbytes),
                host_array_ptr(final_hidden_bits),
                final_hidden_bits.nbytes,
                runtime=target.runtime,
            )
            return hidden_buffer
        except Exception:
            free(hidden_buffer, runtime=target.runtime)
            raise

    def _drop_request(self, request_id: int, *, disable: bool) -> None:
        rid = int(request_id)
        state = self._states.pop(rid, None)
        if state is not None:
            target = self.owner._row(rid).lease.session
            state.verifier.close()
            if int(getattr(target, "_last_target_hidden_ptr", 0)) == int(
                state.root_hidden_buffer.ptr
            ):
                target._last_target_hidden_ptr = 0
            free(state.root_hidden_buffer, runtime=target.runtime)
            state.provider.release_request(rid)
            self.generator._release_mtp_draft_runner(
                state.provider_pool_key,
                state.provider,
            )
        if disable:
            self._disabled_requests.add(rid)


__all__ = ["Qwen35GGUFMTP2Adapter"]
