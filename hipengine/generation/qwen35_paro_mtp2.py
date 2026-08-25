"""Staged packed-PARO MTP2 adapter for gfx1100 SPECDEC2 C1/K1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import time
from typing import Any, Mapping, Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import memory_stats
from hipengine.core.tensor import Tensor
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    _decode_token_cached,
)
from hipengine.speculative.frontier import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    TargetFrontier,
)
from hipengine.speculative.interfaces import AcceptResult
from hipengine.speculative.mtp_native import (
    NativeMtpChainProposer,
    NativeMtpStateSnapshot,
    NativeMtpW8A16Head,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
)


_PROPOSER_CAPACITY_FLOOR = 256
_CYCLE_ALLOCATION_TELEMETRY_ENV = "HIPENGINE_SPECDEC2_CYCLE_ALLOC_TELEMETRY"


def _cycle_allocation_telemetry_enabled() -> bool:
    return os.environ.get(_CYCLE_ALLOCATION_TELEMETRY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _memory_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    return {
        f"delta_{name}": int(after[name]) - int(before[name])
        for name in (
            "active_allocations",
            "current_allocated_bytes",
            "total_allocated_bytes",
            "total_freed_bytes",
        )
    }


def _proposer_capacity_bucket(required_tokens: int) -> int:
    required = int(required_tokens)
    if required <= 0:
        raise ValueError("PARO MTP2 proposer capacity requirement must be positive")
    return 1 << (max(_PROPOSER_CAPACITY_FLOOR, required) - 1).bit_length()


@dataclass(slots=True)
class _ParoMTP2RequestState:
    request_id: int
    proposer: NativeMtpChainProposer
    prompt_rows_consumed: int = 0
    checkpoint: NativeMtpStateSnapshot | None = None
    last_proposal_seconds: float = 0.0
    prompt_prime_seconds: float = 0.0


class Qwen35ParoMTP2Adapter:
    """One bounded C1/K1 provider+target cycle on the qualified PARO route.

    The first S7 capability deliberately exposes no C>1 shape.  Provider state
    is checkpointed before proposal, target state is copied into the spare
    resident slot before verify, and either both owners commit or both restore.
    """

    provider_name = "paro_mtp2"

    def __init__(self, owner: Any, *, enabled: bool = True) -> None:
        self.owner = owner
        self.generator = owner.generator
        self.enabled = bool(enabled)
        self._intents: dict[int, int] = {}
        self._states: dict[int, _ParoMTP2RequestState] = {}
        self._disabled_requests: set[int] = set()
        self._proposer_pool: list[NativeMtpChainProposer] = []
        self._active_claims: ResourceClaimSet | None = None
        self._transaction_sequence = 0
        self._proposer_builds = 0
        self._proposer_reuses = 0

    def register_request(self, request_id: int, candidate_budget: int) -> None:
        rid = int(request_id)
        if int(candidate_budget) <= 0:
            raise ValueError("PARO MTP2 candidate budget must be positive")
        self._intents[rid] = 1
        self._disabled_requests.discard(rid)

    def begin_prompt(self, request_id: int) -> None:
        rid = int(request_id)
        if rid not in self._intents or rid in self._disabled_requests:
            return
        if rid in self._states:
            return
        row = self.owner._row(rid)
        session = self.owner._session
        if session is None:
            raise RuntimeError("PARO MTP2 target session is unavailable")
        started = time.perf_counter()
        required_tokens = len(row.prompt_ids) + 2 * int(row.request.max_tokens) + 8
        proposer_capacity = _proposer_capacity_bucket(required_tokens)
        proposer = None
        while self._proposer_pool:
            candidate = self._proposer_pool.pop()
            if (
                not candidate.closed
                and int(candidate.max_positions) >= int(session.max_sequence_length)
                and int(candidate.max_mtp_tokens) >= required_tokens
                and candidate.scoring_head is not None
                and candidate.scoring_head.owner is session
            ):
                proposer = candidate
                self._proposer_reuses += 1
                break
            candidate.close()
        if proposer is None:
            scoring_head = NativeMtpW8A16Head(
                weight_int8_ptr=int(session.lm_head_weight.tensor.ptr),
                scale_f32_ptr=int(session.lm_head_scale.tensor.ptr),
                vocab_size=int(session.vocab_size),
                threads=int(session.lm_head_threads),
                owner=session,
            )
            proposer = NativeMtpChainProposer(
                self.generator.model_path,
                max_positions=int(session.max_sequence_length),
                max_mtp_tokens=proposer_capacity,
                runtime=session.runtime,
                compiler_version=session.compiler_version,
                scoring_head=scoring_head,
            )
            self._proposer_builds += 1
        proposer.reset()
        self._states[rid] = _ParoMTP2RequestState(rid, proposer)
        row.mtp2_provider_open_ms += (time.perf_counter() - started) * 1000.0

    def consume_prompt_row(
        self,
        request_id: int,
        *,
        prompt_index: int,
        target_hidden_ptr: int,
        seed_token: int | None,
    ) -> None:
        rid = int(request_id)
        self.begin_prompt(rid)
        state = self._states.get(rid)
        if state is None:
            return
        row = self.owner._row(rid)
        index = int(prompt_index)
        if index != state.prompt_rows_consumed:
            raise RuntimeError("PARO MTP2 prompt rows must arrive in order")
        final = index == len(row.prompt_ids) - 1
        if final != (seed_token is not None):
            raise RuntimeError("PARO MTP2 final prompt row must carry target root")
        input_token = (
            int(seed_token)
            if final
            else int(row.prompt_ids[index + 1])
        )
        started = time.perf_counter()
        state.proposer.advance_with_target_hidden(
            input_token=input_token,
            target_hidden_ptr=int(target_hidden_ptr),
            position=index + 1,
            need_result=final,
            read_token_id=False,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        elapsed = time.perf_counter() - started
        state.prompt_prime_seconds += elapsed
        row.mtp2_prompt_prime_ms += elapsed * 1000.0
        state.prompt_rows_consumed += 1

    def observe_prefill_result(self, request_id: int) -> None:
        rid = int(request_id)
        state = self._states.get(rid)
        if state is None:
            return
        row = self.owner._row(rid)
        if state.prompt_rows_consumed != len(row.prompt_ids):
            raise RuntimeError("PARO MTP2 streaming prompt priming is incomplete")
        session = self.owner._session
        if session is None:
            raise RuntimeError("PARO MTP2 target session is unavailable after prefill")
        session.prepare_specdec2_verify_scratch(
            rows=2,
            chain_attn_mode=(
                "c1_loop"
                if str(getattr(self.generator, "execution_profile", "production"))
                == "strict"
                else "decode_batched"
            ),
            max_context_tokens=len(row.prompt_ids) + int(row.request.max_tokens),
        )
        self.owner._release_mtp2_prompt_capture(row)

    def capability(
        self,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability | None:
        semantics = tuple(request_semantics)
        if not self.enabled or len(semantics) != 1:
            return None
        row_semantics = semantics[0]
        rid = int(row_semantics.request_id)
        row = self.owner._row(rid)
        session = self.owner._session
        if (
            rid not in self._intents
            or rid in self._disabled_requests
            or rid not in self._states
            or session is None
            or int(row.model_slot if row.model_slot is not None else -1) != 0
            or not row.native_greedy
            or not row.first_token_emitted
        ):
            return None
        profile = str(
            getattr(self.generator, "execution_profile", None) or "production"
        )
        target_variant = (
            "b1_graph_off_strict_exact"
            if profile == "strict"
            else "b1_graph_off_fast_d64_candidate"
        )
        return SpeculativeCapability(
            capability_key=(
                f"paro_mtp2_c1:{self.generator.backend}:w4_paro:{profile}:{target_variant}"
            ),
            target_key="qwen_paro_w4",
            provider_key="qwen_paro_mtp_bf16",
            method_key="mtp2",
            policy_fingerprint=f"paro-mtp2:c1:k1:{profile}:{target_variant}",
            execution_profile=profile,
            kv_backend_key="paged_bf16_kv_live_spans",
            attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.ONE_WAY_AR,
            supported_modes=("verify_chain",),
            supported_sampling_modes=("greedy",),
            max_requests=1,
            max_candidates_per_request=1,
            max_frontier_rows=2,
            proposal_widths=(1,),
            target_row_buckets=(2,),
            target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            graph_supported=False,
            eager_supported=True,
            strict_fallback_key="paro_target_c1_loop_exact",
            max_context_tokens=int(session.max_sequence_length) - 2,
        )

    def claims_fit(self, plan: SpecRequestPlan) -> bool:
        return bool(
            self.enabled
            and self._active_claims is None
            and plan.request_ids == plan.speculative_request_ids
            and len(plan.request_ids) == 1
            and plan.candidate_counts == (1,)
            and int(plan.request_ids[0]) in self._states
        )

    def component_claims(self, plan: SpecRequestPlan) -> Mapping[str, ResourceClaimSet]:
        rid = int(plan.request_ids[0])
        return {
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"paro_mtp2.provider_rows": 1},
                request_id=rid,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "target": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:target",
                {"paro_mtp2.target_rows": 2},
                request_id=rid,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "result": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:result",
                {"paro_mtp2.result_rows": 1},
                request_id=rid,
                lifetime=ClaimLifetime.TRANSACTION,
            ),
        }

    def reserve_claims(self, claims: ResourceClaimSet) -> ResourceClaimSet:
        if self._active_claims is not None:
            raise RuntimeError("PARO MTP2 already owns an open claim")
        self._active_claims = claims
        return claims

    def release_claims(self, reservation: ResourceClaimSet) -> None:
        if self._active_claims != reservation:
            raise RuntimeError("PARO MTP2 reservation does not own active claims")
        self._active_claims = None

    def prepare_k0(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        for request_id in plan.request_ids:
            rid = int(request_id)
            if rid not in self._states:
                continue
            row = self.owner._row(rid)
            # The first scheduler decode transition only publishes the root
            # sampled by target prefill. The provider is already primed from
            # that prompt/root and must remain live for the following cycle.
            if not bool(row.first_token_emitted):
                continue
            self._drop_request(rid, disable=True, recycle=True)

    def prepare_requests(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        if (
            self._active_claims is None
            or len(plan.request_ids) != 1
            or plan.request_ids != plan.speculative_request_ids
            or plan.candidate_counts != (1,)
            or int(plan.request_ids[0]) not in self._states
        ):
            raise RuntimeError("PARO MTP2 plan is not prepared/claim-fit")

    def propose_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph:
        del stream
        semantics = tuple(request_semantics)
        if len(semantics) != 1:
            raise ValueError("PARO MTP2 proposal requires one request semantic row")
        rid = int(plan.request_ids[0])
        row = self.owner._row(rid)
        state = self._states[rid]
        started = time.perf_counter()
        state.checkpoint = state.proposer.save_state(0)
        candidate_token_ids = state.proposer.device_candidate_token_ids()
        state.last_proposal_seconds = time.perf_counter() - started
        row.mtp2_candidate_device_handoffs += 1
        return CandidateGraph(
            provider_key=plan.provider_key or "qwen_paro_mtp_bf16",
            method_key="mtp2",
            policy_fingerprint="paro-mtp2:c1:k1",
            cycle_id=plan.cycle_id,
            transaction_id=self._next_transaction_id(),
            request_ids=plan.request_ids,
            resident_slots=plan.resident_slots,
            root_positions=(int(semantics[0].context_tokens) - 1,),
            row_offsets=(0, 1),
            row_to_request=(rid,),
            parent_candidate_rows=(-1,),
            draft_depths=(1,),
            active_mask=(True,),
            candidate_tokens=(0,),
            token_ids=candidate_token_ids,
            mode="verify_chain",
            provider_metadata=(("candidate_handoff", "device_i32"),),
        )

    def kv_live_spans_owner(self, plan: SpecRequestPlan) -> str:
        return f"paro-resident:{id(self.owner)}:{plan.operation_id}"

    @staticmethod
    def _verify_target(
        session: Any,
        batch: Any,
        *,
        chain_attn_mode: str,
        candidate_token_ids_i32: Tensor | None,
    ) -> Any:
        return session.verify_chain_bulk_and_commit(
            batch,
            base_slot=0,
            capture_layer_ids=(),
            capture_hidden_concat=Tensor.from_handle(
                0, (2, 0), DType.BF16, Device("hip", 0)
            ),
            capture_row_start=0,
            chain_attn_mode=str(chain_attn_mode),
            graph_mode="off",
            canonicalize_after=False,
            synchronize_after_commit=False,
            candidate_token_ids_i32=candidate_token_ids_i32,
        )

    def execute_target_frontier(
        self,
        plan: SpecRequestPlan,
        frontier: TargetFrontier,
        complete_claims: ResourceClaimSet,
        *,
        commit: bool,
        cancelled_request_ids: Any,
    ) -> SpecCycleResult:
        if not commit:
            raise ValueError("PARO MTP2 target execution requires commit=True")
        if frontier.target_batch is None:
            raise RuntimeError("PARO MTP2 C1 requires host-visible target batch")
        rid = int(plan.request_ids[0])
        row = self.owner._row(rid)
        state = self._states[rid]
        session = self.owner._session
        if session is None or row.model_slot != 0:
            raise RuntimeError("PARO MTP2 C1 requires resident target slot 0")
        cycle_memory_before = (
            memory_stats() if _cycle_allocation_telemetry_enabled() else None
        )
        transaction = SpecCycleTransaction(
            operation_id=plan.operation_id,
            transaction_id=int(frontier.provider_transaction_id or 0),
            cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=complete_claims,
            pre_target_cursors=(int(session.position_arr[0]),),
            pre_rng_counters=(0,),
            target_transaction_mode=plan.target_transaction_mode,
            target_owner=f"paro-target:{id(session)}",
            target_checkpoint_ids=(f"target-slot1:{plan.cycle_id}",),
            pre_provider_cursors=(int(state.proposer.cache_len),),
            provider_transaction_mode=plan.provider_transaction_mode,
            provider_owner=f"paro-provider:{id(state.proposer)}",
            provider_checkpoint_ids=(f"provider-snapshot0:{plan.cycle_id}",),
            target_open=True,
            provider_open=True,
        )
        root_position = int(frontier.root_positions[0])
        session.copy_slot_state(0, 1, kv_rows=root_position)
        session._record_slot_position_host(int(session.position_arr[0]), slot=1)
        if tuple(int(value) for value in cancelled_request_ids()):
            return self._cancelled_result(transaction, state, session)
        target_started = time.perf_counter()
        verify = self._verify_target(
            session,
            frontier.target_batch,
            chain_attn_mode=(
                "c1_loop"
                if str(getattr(self.generator, "execution_profile", "production"))
                == "strict"
                else "decode_batched"
            ),
            candidate_token_ids_i32=(
                frontier.candidate_graph.token_ids
                if frontier.candidate_graph is not None
                else None
            ),
        )
        target_seconds = time.perf_counter() - target_started
        row.mtp2_candidate_d2h_after_target += int(
            frontier.candidate_graph.candidate_rows
        )
        accepted = int(verify.accepted_count)
        bonus = int(
            verify.next_token
            if verify.next_token is not None
            else verify.target_top1[accepted]
        )
        update_started = time.perf_counter()
        try:
            if accepted:
                state.proposer.advance_with_previous_hidden(
                    input_token=int(verify.accepted_tokens[0]),
                    position=state.proposer.position + 1,
                    need_result=False,
                    read_expert_topk=False,
                    read_lm_head_value=False,
                )
            state.proposer.advance_with_target_hidden(
                input_token=bonus,
                target_hidden_ptr=int(verify.selected_target_hidden_ptr),
                position=state.proposer.position + 1,
                read_token_id=False,
                read_expert_topk=False,
                read_lm_head_value=False,
            )
        except Exception:
            self._restore_target_and_provider(state, session)
            raise
        provider_update_seconds = time.perf_counter() - update_started
        cancelled = tuple(int(value) for value in cancelled_request_ids())
        if cancelled:
            return self._cancelled_result(transaction, state, session)
        state.checkpoint = None
        output_ids = (
            *tuple(int(token) for token in verify.accepted_tokens),
            bonus,
        )
        for token in output_ids:
            self.owner._record_step(
                row,
                Qwen35ParoAutoregressiveStepResult(
                    token_id=token,
                    token_text=_decode_token_cached(session.tokenizer, token),
                    logit=float("nan"),
                ),
            )
        row.mtp2_cycles += 1
        row.mtp2_candidate_counts.append(1)
        row.mtp2_accepted_counts.append(accepted)
        row.mtp2_execution_routes.append("eager")
        if cycle_memory_before is not None:
            row.mtp2_cycle_allocation_deltas.append(
                _memory_delta(cycle_memory_before, memory_stats())
            )
        accept = AcceptResult(
            request_ids=plan.request_ids,
            accepted_counts=(accepted,),
            accepted_tokens=(tuple(int(token) for token in verify.accepted_tokens),),
            transaction_id=transaction.transaction_id,
            selected_candidate_rows=(int(verify.commit_row),),
            correction_or_bonus_tokens=(bonus,),
            target_cursor_deltas=(len(output_ids),),
            provider_cursor_deltas=(accepted,),
            finish_reasons=(None,),
        )
        telemetry = SpecCycleTelemetry(
            operation_id=plan.operation_id,
            request_ids=plan.request_ids,
            candidate_counts=plan.candidate_counts,
            plan_reasons=plan.reasons,
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route="eager",
            proposal_seconds=max(0.0, state.last_proposal_seconds),
            target_seconds=target_seconds,
            provider_update_seconds=provider_update_seconds,
            weight_sweeps=1,
        )
        committed = replace(
            transaction,
            target_committed=True,
            provider_committed=True,
        )
        return SpecCycleResult.committed(committed, accept, telemetry=telemetry)

    def rollback_cycle(
        self,
        plan: SpecRequestPlan,
        candidate_graph: CandidateGraph | None,
        error: BaseException,
    ) -> None:
        del candidate_graph, error
        rid = int(plan.request_ids[0])
        state = self._states.get(rid)
        session = self.owner._session
        if state is not None and session is not None:
            self._restore_target_and_provider(state, session)

    def recover_cycle_failure(self, plan: SpecRequestPlan, error: BaseException) -> bool:
        del error
        for request_id in plan.request_ids:
            self._drop_request(int(request_id), disable=True)
        return True

    def close_requests(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            self._drop_request(int(request_id), disable=False, recycle=True)
            self._intents.pop(int(request_id), None)
            self._disabled_requests.discard(int(request_id))

    def _restore_target_and_provider(
        self,
        state: _ParoMTP2RequestState,
        session: Any,
    ) -> None:
        session.copy_slot_state(1, 0, kv_rows=int(session.context_arr[1]))
        session._record_slot_position_host(int(session.position_arr[1]), slot=0)
        if state.checkpoint is not None:
            state.proposer.restore_state(state.checkpoint)
        state.checkpoint = None
        session.runtime.device_synchronize()

    def _cancelled_result(
        self,
        transaction: SpecCycleTransaction,
        state: _ParoMTP2RequestState,
        session: Any,
    ) -> SpecCycleResult:
        self._restore_target_and_provider(state, session)
        return SpecCycleResult(
            stage=SpecCycleStage.CANCELLED,
            transaction=replace(
                transaction,
                target_open=False,
                provider_open=False,
                rolled_back=True,
            ),
            cancelled_request_ids=transaction.request_ids,
        )

    def _drop_request(
        self,
        request_id: int,
        *,
        disable: bool,
        recycle: bool = False,
    ) -> None:
        state = self._states.pop(int(request_id), None)
        if state is not None:
            if recycle and not state.proposer.closed and state.checkpoint is None:
                state.proposer.reset()
                self._proposer_pool.append(state.proposer)
            else:
                state.proposer.close()
        if disable:
            self._disabled_requests.add(int(request_id))

    def _next_transaction_id(self) -> int:
        self._transaction_sequence += 1
        return self._transaction_sequence

    def observability_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "intents": len(self._intents),
            "states": len(self._states),
            "pooled_proposers": len(self._proposer_pool),
            "proposer_builds": self._proposer_builds,
            "proposer_reuses": self._proposer_reuses,
            "active_claim": self._active_claims is not None,
            "disabled_requests": sorted(self._disabled_requests),
        }

    def close(self) -> None:
        for request_id in tuple(self._states):
            self._drop_request(request_id, disable=False)
        for proposer in self._proposer_pool:
            proposer.close()
        self._proposer_pool.clear()
        self._intents.clear()
        self._disabled_requests.clear()
        self._active_claims = None


__all__ = ["Qwen35ParoMTP2Adapter"]
