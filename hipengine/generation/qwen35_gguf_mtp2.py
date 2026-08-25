"""Staged dense GGUF NextN/MTP2 adapter for the Generation-2 resident owner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (
    ACCEPT_PACKED_PAYLOAD_FIELDS,
    build_dflash_accept,
    dflash_accept_chain_i32_packed,
)
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.runtime.qwen35_gguf_mtp import (
    Qwen35GGUFTransactionalVerifier,
    _StreamingNextNPromptSink,
)
from hipengine.runtime.qwen35_gguf_nextn import (
    Qwen35GGUFNextNBatchDeviceProposal,
)
from hipengine.speculative.frontier import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    TargetFrontier,
)
from hipengine.speculative.buffers import (
    TargetVerifyBufferOwner,
    TargetVerifyBufferSpec,
)
from hipengine.speculative.interfaces import (
    AcceptResult,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetVerifyBatch,
    TargetVerifyBuffers,
)
from hipengine.speculative.mtp import MtpProposalContext
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
)


@dataclass(slots=True)
class _MTP2ProviderGroup:
    key: tuple[int, ...]
    provider: Any
    provider_pool_key: Any | None
    request_ids: set[int]


@dataclass(slots=True)
class _MTP2RequestState:
    request_id: int
    provider: Any
    provider_pool_key: Any | None
    provider_group_key: tuple[int, ...]
    verifier: Qwen35GGUFTransactionalVerifier | None
    root_hidden_buffer: DeviceBuffer
    last_proposal_seconds: float = 0.0
    proposal_checkpoint: Any | None = None
    proposal_context: MtpProposalContext | None = None
    proposal_device_batch: Qwen35GGUFNextNBatchDeviceProposal | None = None


class Qwen35GGUFMTP2Adapter:
    """Staged C1/C2/C4 adapter over the retained exact dense components."""

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
        self._prompt_streaming_sinks: dict[int, _StreamingNextNPromptSink] = {}
        self._prompt_streaming_group_keys: dict[int, tuple[int, ...]] = {}
        self._states: dict[int, _MTP2RequestState] = {}
        self._provider_groups: dict[tuple[int, ...], _MTP2ProviderGroup] = {}
        self._disabled_requests: set[int] = set()
        self._active_claims: ResourceClaimSet | None = None
        self._transaction_sequence = 0
        self._batch_accept_workspace: RuntimeWorkspace | None = None
        self._batch_accept_owner: TargetVerifyBufferOwner | None = None
        self._batch_accept_remaining: Tensor | None = None
        self._batch_accept_payload: Tensor | None = None
        self._batch_accept_library: Any | None = None

    def register_request(self, request_id: int, candidate_budget: int) -> None:
        rid = int(request_id)
        budget = min(self.candidate_budget, max(1, int(candidate_budget)))
        self._intents[rid] = budget
        self._disabled_requests.discard(rid)

    def begin_prompt_streaming(
        self,
        request_ids: Sequence[int],
        *,
        checkpoints: Mapping[int, Callable[[], None] | None] | None = None,
    ) -> tuple[_StreamingNextNPromptSink, ...] | None:
        """Open exact shifted NextN sinks before target prompt prefill."""

        ids = tuple(int(value) for value in request_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("streaming prompt request IDs must be non-empty and unique")
        if any(request_id not in self._intents for request_id in ids):
            return None
        if any(request_id in self._disabled_requests for request_id in ids):
            return None
        existing = tuple(self._prompt_streaming_sinks.get(request_id) for request_id in ids)
        if all(sink is not None for sink in existing):
            return tuple(sink for sink in existing if sink is not None)
        if any(sink is not None for sink in existing) or any(
            request_id in self._states for request_id in ids
        ):
            raise RuntimeError("streaming prompt ownership is only opened once per request")
        rows = tuple(self.owner._row(request_id) for request_id in ids)
        if any(row.lease is None or int(row.prefix_reused_tokens) > 0 for row in rows):
            for row in rows:
                if int(getattr(row, "prefix_reused_tokens", 0)) > 0:
                    row.mtp2_prompt_fallback_reason = "prefix_reuse_k0"
            return None
        targets = tuple(row.lease.session for row in rows)
        missing = len(ids)
        group = next(
            (
                candidate
                for candidate in self._provider_groups.values()
                if len(candidate.request_ids) + missing
                <= int(candidate.provider.executor.max_requests)
            ),
            None,
        )
        acquired = group is None
        if group is None:
            max_positions = min(
                int(target.target_layout.max_sequence_length) for target in targets
            )
            provider_capacity = max(
                len(ids),
                min(4, int(getattr(self.owner, "capacity", len(ids)))),
            )
            provider, pool_key, _reused = self.generator._acquire_dense_mtp_draft_provider(
                targets[0],
                max_positions=max_positions,
                pool_enabled=self.owner._shared_runner is not None,
                max_requests=provider_capacity,
            )
            if not callable(getattr(provider.executor, "enqueue_prompt_rows", None)):
                self.generator._release_mtp_draft_runner(pool_key, provider)
                for row in rows:
                    row.mtp2_prompt_fallback_reason = "provider_no_streaming_prompt_abi"
                return None
            group = _MTP2ProviderGroup(
                key=tuple(sorted(ids)),
                provider=provider,
                provider_pool_key=pool_key,
                request_ids=set(),
            )
            self._provider_groups[group.key] = group
        created: list[int] = []
        checkpoint_by_id = {} if checkpoints is None else dict(checkpoints)
        try:
            for request_id, row, target in zip(ids, rows, targets, strict=True):
                group.provider.reset_request(request_id)
                checkpoint = checkpoint_by_id.get(request_id)
                if checkpoint is None:
                    checkpoint = lambda row=row: raise_if_generation_deadline_expired(
                        row.request
                    )
                sink = _StreamingNextNPromptSink(
                    request_id=request_id,
                    prompt_tokens=row.prompt_ids,
                    hidden_size=int(group.provider.executor.hidden_size),
                    executor=group.provider.executor,
                    runtime=target.runtime,
                    checkpoint=checkpoint,
                )
                self._prompt_streaming_sinks[request_id] = sink
                self._prompt_streaming_group_keys[request_id] = group.key
                group.request_ids.add(request_id)
                created.append(request_id)
        except Exception:
            self._abort_prompt_streaming(tuple(created), stream=0)
            if acquired and not group.request_ids:
                self._provider_groups.pop(group.key, None)
            raise
        return tuple(self._prompt_streaming_sinks[request_id] for request_id in ids)

    def finish_prompt_streaming(
        self,
        request_ids: Sequence[int],
        *,
        success: bool,
        stream: int = 0,
    ) -> None:
        """Commit carried prompt rows or roll back every provider/sink owner."""

        ids = tuple(int(value) for value in request_ids)
        if not success:
            self._abort_prompt_streaming(ids, stream=int(stream))
            return
        buffers: dict[int, DeviceBuffer] = {}
        try:
            for request_id in ids:
                sink = self._prompt_streaming_sinks[request_id]
                group = self._provider_groups[
                    self._prompt_streaming_group_keys[request_id]
                ]
                finish = getattr(group.provider.executor, "finish_prompt_priming", None)
                if callable(finish):
                    finish(request_id, stream=int(stream), synchronize=False)
                buffers[request_id] = sink.take_final_pending_buffer()
            pending_states: dict[int, _MTP2RequestState] = {}
            for request_id in ids:
                row = self.owner._row(request_id)
                target = row.lease.session
                group_key = self._prompt_streaming_group_keys[request_id]
                group = self._provider_groups[group_key]
                verifier = (
                    Qwen35GGUFTransactionalVerifier(
                        target,
                        max_candidate_budget=self.candidate_budget,
                        quant=self.quant,
                        target_verify_mode=self.target_verify_mode,
                    )
                    if len(ids) == 1 and int(getattr(self.owner, "capacity", 1)) == 1
                    else None
                )
                pending_states[request_id] = _MTP2RequestState(
                    request_id=request_id,
                    provider=group.provider,
                    provider_pool_key=group.provider_pool_key,
                    provider_group_key=group_key,
                    verifier=verifier,
                    root_hidden_buffer=buffers[request_id],
                )
            for request_id, state in pending_states.items():
                row = self.owner._row(request_id)
                target = row.lease.session
                target._last_target_hidden_ptr = int(state.root_hidden_buffer.ptr)
                self._states[request_id] = state
                row.mtp2_prompt_streaming = True
                row.mtp2_prompt_prime_rows = len(row.prompt_ids)
                row.mtp2_prompt_carried_bytes = int(state.root_hidden_buffer.nbytes)
                row.mtp2_prompt_fallback_reason = None
                self._prompt_hidden_rows.pop(request_id, None)
        except Exception:
            for state in locals().get("pending_states", {}).values():
                if state.verifier is not None:
                    state.verifier.close()
            for request_id, buffer in buffers.items():
                target = self.owner._row(request_id).lease.session
                if request_id in self._states:
                    self._states.pop(request_id, None)
                    if int(getattr(target, "_last_target_hidden_ptr", 0)) == int(
                        buffer.ptr
                    ):
                        target._last_target_hidden_ptr = 0
                free(buffer, runtime=target.runtime)
            self._abort_prompt_streaming(ids, stream=int(stream))
            raise
        finally:
            for request_id in ids:
                sink = self._prompt_streaming_sinks.pop(request_id, None)
                self._prompt_streaming_group_keys.pop(request_id, None)
                if sink is not None:
                    sink.close()

    def _abort_prompt_streaming(
        self,
        request_ids: Sequence[int],
        *,
        stream: int,
    ) -> None:
        groups: set[tuple[int, ...]] = set()
        for request_id in tuple(int(value) for value in request_ids):
            sink = self._prompt_streaming_sinks.pop(request_id, None)
            group_key = self._prompt_streaming_group_keys.pop(request_id, None)
            if group_key is None:
                if sink is not None:
                    sink.close()
                continue
            group = self._provider_groups.get(group_key)
            if group is not None:
                finish = getattr(group.provider.executor, "finish_prompt_priming", None)
                if callable(finish):
                    finish(request_id, stream=int(stream), synchronize=True)
                group.provider.release_request(request_id)
                group.request_ids.discard(request_id)
                groups.add(group_key)
            if sink is not None:
                sink.close()
        for group_key in groups:
            group = self._provider_groups.get(group_key)
            if group is not None and not group.request_ids:
                self._provider_groups.pop(group_key, None)
                self.generator._release_mtp_draft_runner(
                    group.provider_pool_key,
                    group.provider,
                )

    def observe_prefill_result(self, request_id: int, prompt_ids: Sequence[int], result: Any) -> None:
        rid = int(request_id)
        if rid not in self._intents:
            return
        hidden = getattr(result, "hidden_seeds", None)
        if hidden is None:
            self._prompt_hidden_rows.pop(rid, None)
            if rid in self._states:
                return
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
        if not self.enabled or not (1 <= len(semantics) <= 4):
            return None
        targets = []
        for item in semantics:
            rid = int(item.request_id)
            if rid not in self._intents or rid in self._disabled_requests:
                return None
            row = self.owner._row(rid)
            if (
                not row.native_greedy
                or not row.first_token_emitted
                or row.lease is None
                or row.slot is None
                or (
                    rid not in self._states
                    and rid not in self._prompt_hidden_rows
                )
            ):
                return None
            target = row.lease.session
            if bool(
                getattr(getattr(target, "runner", None), "fp16_recurrent_state", False)
            ):
                return None
            targets.append(target)
        if len(semantics) == 1:
            existing = self._states.get(int(semantics[0].request_id))
            owner = getattr(targets[0], "_target_scratch_owner", None)
            if (
                (existing is not None and existing.verifier is None)
                or int(getattr(owner, "slot_count", 1)) > 1
                or int(getattr(self.owner, "capacity", 1)) > 1
            ):
                return None
        # Streaming activation is retained only through the already-qualified
        # short target context. Longer requests stay K0 until an exact shifted-
        # page eager target owner is qualified independently of graph capture.
        max_context = min(
            1023,
            *(int(target.target_layout.max_sequence_length) for target in targets),
        )
        profile = str(getattr(self.generator, "execution_profile", None) or "legacy_exact")
        max_requests = min(4, max(1, int(getattr(self.owner, "capacity", 4))))
        max_frontier_rows = max_requests * (self.candidate_budget + 1)
        return SpeculativeCapability(
            capability_key=(
                f"gguf_mtp2_c{max_requests}:{self.generator.backend}:{self.quant}:"
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
            max_requests=max_requests,
            max_candidates_per_request=self.candidate_budget,
            max_frontier_rows=max_frontier_rows,
            proposal_widths=tuple(
                width for width in (1, 2, 4) if width <= max_requests
            ),
            target_row_buckets=tuple(range(2, max_frontier_rows + 1)),
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
            and 1 <= len(plan.speculative_request_ids) <= 4
            and not (
                len(plan.speculative_request_ids) == 1
                and int(getattr(self.owner, "capacity", 1)) > 1
            )
            and not any(
                request_id in self._disabled_requests
                for request_id in plan.speculative_request_ids
            )
        )

    def prepare_k0(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        reason_by_id = dict(zip(plan.request_ids, plan.reasons, strict=True))
        ids = tuple(
            int(request_id)
            for request_id in plan.request_ids
            if int(request_id) in self._intents
            and int(request_id) not in self._disabled_requests
            and (
                int(request_id) in self._states
                or int(request_id) in self._prompt_hidden_rows
            )
        )
        attach = tuple(
            request_id
            for request_id in ids
            if request_id not in self._states
            and reason_by_id[request_id] is SpecPlanReason.NO_PROVIDER
        )
        if attach:
            for request_id in attach:
                self.owner._flush_row_owner(self.owner._row(request_id))
            self._ensure_request_states(attach)
        for rid in ids:
            if reason_by_id[rid] not in {
                SpecPlanReason.NO_PROVIDER,
                SpecPlanReason.POLICY_SELECTED_AR,
            }:
                continue
            state = self._states.get(rid)
            if state is None:
                continue
            row = self.owner._row(rid)
            if row.lease is None or row.slot is None or not row.first_token_emitted:
                continue
            target = row.lease.session
            root_token = int(row.slot.generated_ids[-1])
            root_position = int(target.position)
            state.provider.executor.advance_state_only(
                rid,
                root_token,
                root_position,
                target.last_target_hidden,
            )
            row.mtp2_k0_catchups += 1

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
        ids = tuple(int(value) for value in plan.speculative_request_ids)
        for request_id in ids:
            if request_id not in self._states:
                self.owner._flush_row_owner(self.owner._row(request_id))
        self._ensure_request_states(ids)

    def _ensure_request_states(self, ids: tuple[int, ...]) -> None:
        missing = tuple(request_id for request_id in ids if request_id not in self._states)
        if not missing:
            return
        existing_groups = {
            self._states[request_id].provider_group_key
            for request_id in ids
            if request_id in self._states
        }
        if len(existing_groups) > 1:
            raise RuntimeError("GGUF MTP2 requests belong to incompatible provider groups")
        if existing_groups:
            group = self._provider_groups[next(iter(existing_groups))]
            if len(group.request_ids) + len(missing) > int(group.provider.executor.max_requests):
                raise RuntimeError("GGUF MTP2 provider group has no refill capacity")
            for request_id in missing:
                self._states[request_id] = self._attach_request_to_group(
                    request_id,
                    group,
                )
            return
        if len(ids) == 1 and int(getattr(self.owner, "capacity", 1)) == 1:
            self._states[ids[0]] = self._open_request(ids[0])
        else:
            self._open_batch_requests(ids)

    def propose_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph:
        del stream
        ids = tuple(int(value) for value in plan.speculative_request_ids)
        if not (1 <= len(ids) <= 4):
            raise NotImplementedError("GGUF MTP2 supports c1/c2/c4 proposal")
        states = tuple(self._states[request_id] for request_id in ids)
        if len({state.provider_group_key for state in states}) != 1:
            raise RuntimeError("physical NextN proposal requires one provider group")
        rows = tuple(self.owner._row(request_id) for request_id in ids)
        targets = tuple(row.lease.session for row in rows)
        slots = tuple(row.slot for row in rows)
        if any(slot is None for slot in slots):
            raise RuntimeError("GGUF MTP2 row has no committed root token")
        counts_by_id = dict(zip(plan.request_ids, plan.candidate_counts, strict=True))
        budgets = tuple(int(counts_by_id[request_id]) for request_id in ids)
        hidden_size = int(states[0].provider.executor.hidden_size)
        hidden_batch = malloc(
            len(ids) * hidden_size * DType.BF16.itemsize,
            runtime=targets[0].runtime,
        )
        try:
            row_nbytes = hidden_size * DType.BF16.itemsize
            for index, target in enumerate(targets):
                self.owner._flush_row_owner(rows[index])
                if int(target.position) != int(slots[index].seq_position):
                    raise RuntimeError(
                        "GGUF MTP2 target session cursor is stale after owner flush: "
                        f"request={ids[index]} target={int(target.position)} "
                        f"slot={int(slots[index].seq_position)}"
                    )
                target.runtime.memcpy(
                    hidden_batch.ptr + index * row_nbytes,
                    target.last_target_hidden.ptr,
                    row_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
            context = MtpProposalContext(
                request_ids=ids,
                root_tokens=tuple(int(slot.generated_ids[-1]) for slot in slots),
                root_positions=tuple(int(target.position) for target in targets),
                target_hidden=Tensor.from_handle(
                    hidden_batch.ptr,
                    (len(ids), hidden_size),
                    DType.BF16,
                    Device("hip", 0),
                ),
            )
            checkpoints = {}
            for request_id, state in zip(ids, states, strict=True):
                if state.proposal_checkpoint is not None:
                    raise RuntimeError("GGUF MTP2 proposal checkpoint is already open")
                checkpoints[request_id] = (
                    state.provider.executor.capture_request_checkpoint(request_id)
                )
            proposal_started = time.perf_counter()
            device_draft = None
            if len(ids) == 1:
                draft = states[0].provider.propose(
                    context,
                    candidate_budget=budgets[0],
                    return_logits=False,
                    allow_graph=False,
                )
            else:
                draft = None
                device_draft = states[0].provider.propose_batch_device(
                    context,
                    candidate_counts=budgets,
                )
                physical_shapes = tuple(
                    sum(1 for count in budgets if depth < count)
                    for depth in range(max(budgets))
                )
                batched_shapes = tuple(shape for shape in physical_shapes if shape > 1)
                for row in rows:
                    row.mtp2_proposal_batch_calls += len(batched_shapes)
                    row.mtp2_proposal_physical_rows.extend(batched_shapes)
                    row.mtp2_candidate_device_handoffs += 1
            proposal_seconds = time.perf_counter() - proposal_started
            for index, (request_id, state) in enumerate(zip(ids, states, strict=True)):
                state.last_proposal_seconds = proposal_seconds
                state.proposal_checkpoint = checkpoints[request_id]
                state.proposal_context = MtpProposalContext(
                    request_ids=(request_id,),
                    root_tokens=(int(context.root_tokens[index]),),
                    root_positions=(int(context.root_positions[index]),),
                    target_hidden=targets[index].last_target_hidden,
                )
                state.proposal_device_batch = device_draft
        except Exception:
            for request_id, checkpoint in locals().get("checkpoints", {}).items():
                state = self._states[request_id]
                state.provider.executor.restore_request_checkpoint(checkpoint)
                state.provider.executor.release_request_checkpoint(checkpoint)
            raise
        finally:
            free(hidden_batch, runtime=targets[0].runtime)
        if device_draft is None:
            assert draft is not None
            enabled = draft.active_mask or (True,) * draft.draft_rows
            active_indices = tuple(
                index for index, active in enumerate(enabled) if bool(active)
            )
            candidate_tokens = tuple(
                draft.candidate_tokens[index] for index in active_indices
            )
            draft_depths = tuple(
                draft.draft_depths[index] for index in active_indices
            )
            row_to_request = tuple(
                draft.row_to_request[index] for index in active_indices
            )
            candidate_token_ids = None
            draft_mode = draft.mode
            draft_metadata = draft.provider_metadata
        else:
            candidate_tokens = ()
            draft_depths = tuple(
                depth
                for count in budgets
                for depth in range(1, count + 1)
            )
            row_to_request = tuple(
                request_id
                for request_id, count in zip(ids, budgets, strict=True)
                for _ in range(count)
            )
            candidate_token_ids = device_draft.token_ids
            draft_mode = "verify_chain"
            draft_metadata = (
                ("candidate_handoff", "device_i32"),
                ("candidate_rows", sum(budgets)),
            )
        parents_list: list[int] = []
        last_row_by_request: dict[int, int] = {}
        for row_index, (request_id, depth) in enumerate(
            zip(row_to_request, draft_depths, strict=True)
        ):
            parents_list.append(
                -1 if depth == 1 else last_row_by_request[request_id]
            )
            last_row_by_request[request_id] = row_index
        parents = tuple(parents_list)
        counts = tuple(
            sum(1 for owner in row_to_request if owner == request_id)
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
            row_to_request=row_to_request,
            parent_candidate_rows=parents,
            draft_depths=draft_depths,
            active_mask=(True,) * len(row_to_request),
            candidate_tokens=candidate_tokens,
            token_ids=candidate_token_ids,
            candidate_ids=(),
            mode=draft_mode,
            provider_metadata=draft_metadata,
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
        if len(plan.speculative_request_ids) > 1 and (
            frontier.target_batch is not None
            or (
                frontier.candidate_graph is not None
                and frontier.candidate_graph.token_ids is not None
            )
        ):
            return self._execute_target_frontier_batch(
                plan,
                frontier,
                complete_claims,
                cancelled_request_ids=cancelled_request_ids,
            )
        if frontier.target_batch is None:
            raise NotImplementedError("GGUF MTP2 requires a host or device chain")
        if len(plan.speculative_request_ids) != 1:
            raise NotImplementedError("GGUF MTP2 target has no speculative rows")
        rid = int(plan.speculative_request_ids[0])
        state = self._states[rid]
        if state.verifier is None:
            raise NotImplementedError(
                "a survivor of a physical provider group requires physical target padding"
            )
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
            self._restore_provider_checkpoint(state)
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
                self._restore_provider_checkpoint(state)
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
            accepted = int(summary.accepted_counts[0])
            provider_update_started = time.perf_counter()
            self._repair_provider_state(
                state,
                accepted_count=accepted,
                candidate_count=int(plan.candidate_counts[0]),
            )
            provider_update_seconds = time.perf_counter() - provider_update_started
            cancelled = tuple(int(value) for value in cancelled_request_ids())
            if cancelled:
                state.verifier.rollback(prepared)
                prepared = None
                self._restore_provider_checkpoint(state)
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
            state.verifier.commit(prepared, commit_plan)
            native_graph_submitted = bool(prepared.native_graph_submitted)
            state.verifier.finish(prepared)
            prepared = None
            self._release_provider_checkpoint(state)
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
            row.mtp2_execution_routes.append(actual_execution_route)
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
            self._restore_provider_checkpoint(state)
            raise

    def _batch_accept_resources(
        self,
        runtime: Any,
    ) -> tuple[TargetVerifyBufferOwner, Tensor, Tensor]:
        if self._batch_accept_workspace is None:
            workspace = RuntimeWorkspace(device=Device("hip", 0), runtime=runtime)
            spec = TargetVerifyBufferSpec(
                backend=str(self.generator.backend),
                bucket="gguf-mtp2-physical-r16-c4",
                device=Device("hip", 0),
                max_rows=16,
                max_requests=4,
                mode="verify_chain",
            )
            self._batch_accept_workspace = workspace
            self._batch_accept_owner = TargetVerifyBufferOwner.allocate(
                spec,
                workspace=workspace,
            )
            self._batch_accept_remaining = workspace.reserve_tensor(
                "target_verify/gguf-mtp2-physical-r16-c4/remaining_decode",
                (4,),
                DType.INT32,
            )
            self._batch_accept_payload = workspace.reserve_tensor(
                "target_verify/gguf-mtp2-physical-r16-c4/packed_accept_payload",
                (4, ACCEPT_PACKED_PAYLOAD_FIELDS),
                DType.INT32,
            )
        if (
            self._batch_accept_owner is None
            or self._batch_accept_remaining is None
            or self._batch_accept_payload is None
        ):
            raise RuntimeError("physical accept workspace is incomplete")
        return (
            self._batch_accept_owner,
            self._batch_accept_remaining,
            self._batch_accept_payload,
        )

    @staticmethod
    def _upload_accept_array(tensor: Tensor, values: np.ndarray, runtime: Any) -> None:
        array = np.ascontiguousarray(values)
        if array.nbytes > tensor.numel * tensor.dtype.itemsize:
            raise ValueError("physical accept upload exceeds tensor capacity")
        copy_host_to_device(
            DeviceBuffer(tensor.ptr, array.nbytes),
            host_array_ptr(array),
            array.nbytes,
            runtime=runtime,
        )

    def _accept_target_batch_on_device(
        self,
        batch: TargetVerifyBatch,
        target_top1: Sequence[int],
        remaining_decode: Sequence[int],
        *,
        transaction_id: int,
        runtime: Any,
    ) -> tuple[TargetAcceptSummary, TargetVerifyBuffers]:
        """Emit one GPU accept payload for the whole physical target group."""

        owner, remaining_owner, payload_owner = self._batch_accept_resources(runtime)
        buffers = owner.bind(batch, transaction_id=int(transaction_id))
        request_count = len(batch.request_ids)
        remaining = Tensor.from_handle(
            remaining_owner.ptr,
            (request_count,),
            DType.INT32,
            remaining_owner.device,
        )
        payload = Tensor.from_handle(
            payload_owner.ptr,
            (request_count, ACCEPT_PACKED_PAYLOAD_FIELDS),
            DType.INT32,
            payload_owner.device,
        )
        for tensor, values in (
            (buffers.token_ids, np.asarray(batch.tokens, dtype=np.int32)),
            (buffers.positions, np.asarray(batch.positions, dtype=np.int32)),
            (buffers.parent_rows, np.asarray(batch.parent_rows, dtype=np.int32)),
            (buffers.draft_depths, np.asarray(batch.draft_depths, dtype=np.int32)),
            (buffers.row_to_request, np.asarray(batch.row_to_request, dtype=np.int32)),
            (buffers.active_mask, np.asarray(batch.active_mask, dtype=np.uint8)),
            (buffers.target_top1, np.asarray(tuple(target_top1), dtype=np.int32)),
            (remaining, np.asarray(tuple(remaining_decode), dtype=np.int32)),
        ):
            self._upload_accept_array(tensor, values, runtime)
        if self._batch_accept_library is None:
            self._batch_accept_library = build_dflash_accept(
                load=True,
                compiler_version=getattr(self.generator, "compiler_version", None),
                require_cached=bool(
                    getattr(self.generator, "require_cached_build", False)
                ),
            )
        dflash_accept_chain_i32_packed(
            buffers.token_ids.ptr,
            buffers.positions.ptr,
            buffers.parent_rows.ptr,
            buffers.draft_depths.ptr,
            buffers.active_mask.ptr,
            buffers.target_top1.ptr,
            remaining.ptr,
            buffers.accepted_counts.ptr,
            buffers.commit_rows.ptr,
            buffers.commit_tokens.ptr,
            buffers.commit_positions.ptr,
            buffers.next_tokens.ptr if buffers.next_tokens is not None else 0,
            buffers.full_accept.ptr if buffers.full_accept is not None else 0,
            (
                buffers.committed_output_ids.ptr
                if buffers.committed_output_ids is not None
                else 0
            ),
            (
                buffers.committed_output_lengths.ptr
                if buffers.committed_output_lengths is not None
                else 0
            ),
            payload.ptr,
            batch.rows,
            request_count,
            batch.rows,
            library=self._batch_accept_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        payload_host = np.empty(
            (request_count, ACCEPT_PACKED_PAYLOAD_FIELDS),
            dtype=np.int32,
        )
        copy_device_to_host(
            host_array_ptr(payload_host),
            DeviceBuffer(payload.ptr, payload_host.nbytes),
            payload_host.nbytes,
            runtime=runtime,
        )
        summary = TargetAcceptSummary.from_gpu_payload(
            batch,
            {
                "accepted_counts": tuple(int(value) for value in payload_host[:, 0]),
                "commit_rows": tuple(int(value) for value in payload_host[:, 1]),
                "commit_tokens": tuple(int(value) for value in payload_host[:, 2]),
                "commit_positions": tuple(int(value) for value in payload_host[:, 3]),
                "next_tokens": tuple(int(value) for value in payload_host[:, 4]),
                "full_accept": tuple(bool(value) for value in payload_host[:, 5]),
            },
        )
        return replace(summary, transaction_id=int(transaction_id)), buffers

    def _execute_target_frontier_batch(
        self,
        plan: SpecRequestPlan,
        frontier: TargetFrontier,
        complete_claims: ResourceClaimSet,
        *,
        cancelled_request_ids: Callable[[], Sequence[int]],
    ) -> SpecCycleResult:
        ids = tuple(int(value) for value in plan.speculative_request_ids)
        states = tuple(self._states[request_id] for request_id in ids)
        rows = tuple(self.owner._row(request_id) for request_id in ids)
        targets = tuple(row.lease.session for row in rows)
        batch = frontier.target_batch
        device_draft = states[0].proposal_device_batch
        if batch is None:
            if device_draft is None or frontier.candidate_graph is None:
                raise RuntimeError("device target frontier lost its proposal descriptor")
            if any(state.proposal_device_batch is not device_draft for state in states):
                raise RuntimeError("physical requests do not share one device proposal")
            if frontier.candidate_graph.token_ids is not device_draft.token_ids:
                raise RuntimeError("candidate graph does not own the device proposal tokens")
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
            pre_provider_cursors=tuple(int(target.position) for target in targets),
            pre_rng_counters=(0,) * len(plan.request_ids),
            target_transaction_mode=plan.target_transaction_mode,
            provider_transaction_mode=plan.provider_transaction_mode,
            target_owner=f"{plan.operation_id}:gguf-target-batch",
            provider_owner=f"{plan.operation_id}:nextn-batch",
            provider_request_ids=ids,
            target_checkpoint_ids=tuple(
                f"target:{request_id}:{self.owner._row(request_id).lease.session.position}"
                for request_id in plan.request_ids
            ),
            provider_checkpoint_ids=tuple(
                f"provider:{request_id}:{target.position}"
                for request_id, target in zip(ids, targets, strict=True)
            ),
            target_open=True,
            provider_open=True,
        )
        cancelled = tuple(int(value) for value in cancelled_request_ids())
        if cancelled:
            for state in states:
                self._restore_provider_checkpoint(state)
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
        root_row_by_id: dict[int, int] = {}
        candidate_rows_by_id: dict[int, tuple[int, ...]] = {}
        jobs = []
        if batch is not None:
            root_row_by_id = dict(
                zip(batch.request_ids, batch.root_rows, strict=True)
            )
            candidate_rows_by_id = {
                request_id: tuple(
                    sorted(
                        (
                            row_index
                            for row_index in batch.candidate_rows
                            if batch.row_to_request[row_index] == request_id
                        ),
                        key=lambda row_index: batch.draft_depths[row_index],
                    )
                )
                for request_id in ids
            }
        root_token_by_id = dict(
            zip(frontier.request_ids, frontier.root_tokens, strict=True)
        )
        device_offsets: dict[int, tuple[int, int]] = {}
        if device_draft is not None:
            offset = 0
            for request_id, count in zip(
                device_draft.request_ids,
                device_draft.candidate_counts,
                strict=True,
            ):
                device_offsets[int(request_id)] = (offset, int(count))
                offset += int(count)
        for request_id, target in zip(ids, targets, strict=True):
            job = {
                "session": target,
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": True,
                "defer_linear_state_commit": True,
                "defer_state_scatter": True,
            }
            if batch is not None:
                root_row = root_row_by_id[request_id]
                candidate_rows = candidate_rows_by_id[request_id]
                job["input_token_ids"] = (
                    int(batch.tokens[root_row]),
                    *tuple(int(batch.tokens[row]) for row in candidate_rows),
                )
            else:
                assert device_draft is not None
                offset, count = device_offsets[request_id]
                job["input_token_ids"] = (
                    int(root_token_by_id[request_id]),
                    *((0,) * count),
                )
                job["candidate_token_ids_device"] = Tensor.from_handle(
                    device_draft.token_ids.ptr + offset * DType.INT32.itemsize,
                    (count,),
                    DType.INT32,
                    device_draft.token_ids.device,
                )
            jobs.append(job)
        owner = self.owner._packed_execution_owner(targets[0])
        verify_batch = getattr(owner, "verify_target_blocks_batch", None)
        if not callable(verify_batch):
            raise RuntimeError("physical target owner has no packed verifier")
        target_started = time.perf_counter()
        results = list(verify_batch(jobs))
        target_seconds = time.perf_counter() - target_started
        physical_target_rows = sum(len(job["input_token_ids"]) for job in jobs)
        for row in rows:
            row.mtp2_target_batch_calls += 1
            row.mtp2_target_physical_rows.append(physical_target_rows)
        if len(results) != len(ids):
            raise RuntimeError("physical target verifier returned wrong result count")
        candidate_readback_seconds = 0.0
        if batch is None:
            assert device_draft is not None
            readback_started = time.perf_counter()
            materialized = states[0].provider.materialize_batch_device_proposal(
                device_draft
            )
            candidate_readback_seconds = time.perf_counter() - readback_started
            for row in rows:
                row.mtp2_candidate_d2h_after_target += 1
            enabled = materialized.active_mask or (True,) * materialized.draft_rows
            candidate_tokens = tuple(
                int(token)
                for token, active in zip(
                    materialized.candidate_tokens,
                    enabled,
                    strict=True,
                )
                if bool(active)
            )
            graph = frontier.candidate_graph
            assert graph is not None
            host_graph = replace(graph, candidate_tokens=candidate_tokens)
            batch = TargetVerifyBatch.from_draft(
                host_graph.to_draft_batch(),
                root_tokens=frontier.root_tokens,
                root_positions=frontier.root_positions,
            )
            root_row_by_id = dict(
                zip(batch.request_ids, batch.root_rows, strict=True)
            )
            candidate_rows_by_id = {
                request_id: tuple(
                    sorted(
                        (
                            row_index
                            for row_index in batch.candidate_rows
                            if batch.row_to_request[row_index] == request_id
                        ),
                        key=lambda row_index: batch.draft_depths[row_index],
                    )
                )
                for request_id in ids
            }
        target_top1 = [0] * batch.rows
        for request_id, result in zip(ids, results, strict=True):
            rows_for_request = (
                root_row_by_id[request_id],
                *candidate_rows_by_id[request_id],
            )
            if len(result.token_ids) != len(rows_for_request):
                raise RuntimeError("physical target verifier omitted request rows")
            for row_index, token in zip(
                rows_for_request,
                result.token_ids,
                strict=True,
            ):
                target_top1[row_index] = int(token)
        remaining = tuple(
            max(0, int(row.request.max_tokens) - len(row.slot.generated_ids))
            for row in rows
        )
        accept_started = time.perf_counter()
        gpu_summary, accept_buffers = self._accept_target_batch_on_device(
            batch,
            target_top1,
            remaining,
            transaction_id=transaction_id,
            runtime=targets[0].runtime,
        )
        accept = batch.accept_from_top1(
            target_top1,
            transaction_id=transaction_id,
            remaining_decode=remaining,
        )
        cpu_summary = replace(
            TargetAcceptSummary.from_accept_result(batch, accept),
            transaction_id=transaction_id,
        )
        if any(
            getattr(gpu_summary, field) != getattr(cpu_summary, field)
            for field in (
                "request_ids",
                "accepted_counts",
                "accepted_tokens",
                "commit_rows",
                "commit_tokens",
                "commit_positions",
                "next_tokens",
                "full_accept",
            )
        ):
            raise RuntimeError(
                "physical GPU accept payload does not match the CPU oracle"
            )
        accept_seconds = time.perf_counter() - accept_started
        for row in rows:
            row.mtp2_device_accept_calls += 1
        provider_update_started = time.perf_counter()
        self._repair_provider_states_batch(
            states,
            accepted_counts=accept.accepted_counts,
            candidate_counts=tuple(
                plan.candidate_counts[plan.request_ids.index(request_id)]
                for request_id in ids
            ),
        )
        provider_update_seconds = time.perf_counter() - provider_update_started
        cancelled = tuple(int(value) for value in cancelled_request_ids())
        if cancelled:
            for state in states:
                self._restore_provider_checkpoint(state)
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
        commit_batch = getattr(
            owner,
            "_commit_deferred_packed_verify_states_batch",
            None,
        )
        if not callable(commit_batch):
            raise RuntimeError("physical target owner has no batch selected-state commit")
        commit_started = time.perf_counter()
        commit_contract = commit_batch(
            results,
            targets,
            accepted_counts=gpu_summary.accepted_counts,
            accept_buffers=accept_buffers,
        )
        commit_seconds = time.perf_counter() - commit_started
        if int(commit_contract.get("requests", 0)) != len(ids):
            raise RuntimeError("physical selected-state commit omitted requests")
        for row in rows:
            row.mtp2_selected_commit_batch_calls += 1
            row.mtp2_execution_routes.append("eager")
        output_ids: list[tuple[int, ...]] = []
        next_tokens = accept.next_tokens or (None,) * len(ids)
        for index, (request_id, target, row, accepted, accepted_tokens, next_token) in enumerate(
            zip(
                ids,
                targets,
                rows,
                accept.accepted_counts,
                accept.accepted_tokens,
                next_tokens,
                strict=True,
            )
        ):
            visible = (
                *tuple(int(token) for token in accepted_tokens),
                *(() if next_token is None else (int(next_token),)),
            )
            if not visible:
                raise RuntimeError("physical target cycle produced no visible token")
            row.slot.generated_ids.extend(visible)
            row.slot.prev_token = int(visible[-1])
            row.slot.seq_position = int(target.position)
            row.slot.native_decode_steps += 1
            row.slot.done = len(row.slot.generated_ids) >= int(row.request.max_tokens)
            row.mtp2_cycles += 1
            row.mtp2_candidate_counts.append(
                int(plan.candidate_counts[plan.request_ids.index(request_id)])
            )
            row.mtp2_accepted_counts.append(int(accepted))
            row.mtp2_proposal_ms += float(states[index].last_proposal_seconds) * 1000.0
            row.mtp2_target_ms += float(target_seconds) * 1000.0
            row.mtp2_provider_update_ms += float(provider_update_seconds) * 1000.0
            row.mtp2_accept_ms += float(accept_seconds) * 1000.0
            row.mtp2_selected_commit_ms += float(commit_seconds) * 1000.0
            row.mtp2_candidate_readback_ms += (
                float(candidate_readback_seconds) * 1000.0
            )
            output_ids.append(visible)
            self._release_provider_checkpoint(states[index])
        committed_accept = AcceptResult(
            request_ids=ids,
            accepted_counts=accept.accepted_counts,
            accepted_tokens=accept.accepted_tokens,
            transaction_id=transaction_id,
            selected_candidate_rows=accept.selected_candidate_rows,
            correction_or_bonus_tokens=tuple(next_tokens),
            target_cursor_deltas=tuple(len(tokens) for tokens in output_ids),
            provider_cursor_deltas=accept.accepted_counts,
            finish_reasons=(None,) * len(ids),
        )
        telemetry = SpecCycleTelemetry(
            operation_id=plan.operation_id,
            request_ids=plan.request_ids,
            candidate_counts=plan.candidate_counts,
            plan_reasons=plan.reasons,
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route="eager",
            proposal_seconds=max(state.last_proposal_seconds for state in states),
            target_seconds=target_seconds,
            accept_commit_seconds=accept_seconds + commit_seconds,
            provider_update_seconds=provider_update_seconds,
            scheduler_readback_seconds=candidate_readback_seconds,
            weight_sweeps=1,
        )
        return SpecCycleResult.committed(
            replace(
                transaction,
                target_committed=True,
                provider_committed=True,
            ),
            committed_accept,
            telemetry=telemetry,
        )

    def _repair_provider_states_batch(
        self,
        states: tuple[_MTP2RequestState, ...],
        *,
        accepted_counts: Sequence[int],
        candidate_counts: Sequence[int],
    ) -> None:
        accepted = tuple(int(value) for value in accepted_counts)
        counts = tuple(int(value) for value in candidate_counts)
        if len(states) <= 1 or len(accepted) != len(states) or len(counts) != len(states):
            raise ValueError("physical provider repair requires aligned C>1 rows")
        if len({state.provider_group_key for state in states}) != 1:
            raise RuntimeError("physical provider repair requires one provider group")
        executor = states[0].provider.executor
        operations: list[list[tuple[int, int, Tensor]]] = []
        for state, accepted_count, candidate_count in zip(
            states,
            accepted,
            counts,
            strict=True,
        ):
            checkpoint = state.proposal_checkpoint
            context = state.proposal_context
            results = state.provider.last_results.get(int(state.request_id))
            if checkpoint is None or context is None:
                raise RuntimeError("GGUF MTP2 provider repair has no proposal checkpoint")
            if results is None or len(results) != candidate_count:
                raise RuntimeError("GGUF MTP2 provider repair lost proposal rows")
            if accepted_count < 0 or accepted_count > len(results):
                raise ValueError("accepted_count is outside proposal rows")
            if accepted_count == len(results):
                tail = results[-1]
                operations.append(
                    [
                        (
                            int(tail.token_id),
                            int(tail.position) + 1,
                            tail.hidden,
                        )
                    ]
                )
                continue
            executor.restore_request_checkpoint(checkpoint)
            replay = [
                (
                    int(context.root_tokens[0]),
                    int(context.root_positions[0]),
                    context.target_hidden,
                )
            ]
            replay.extend(
                (
                    int(results[index].token_id),
                    int(context.root_positions[0]) + index + 1,
                    results[index].hidden,
                )
                for index in range(accepted_count)
            )
            operations.append(replay)
        hidden_size = int(executor.hidden_size)
        hidden_nbytes = hidden_size * DType.BF16.itemsize
        hidden_batch = malloc(
            len(states) * hidden_nbytes,
            runtime=executor.runtime,
        )
        try:
            for depth in range(max(len(rows) for rows in operations)):
                active = tuple(
                    index for index, rows in enumerate(operations) if depth < len(rows)
                )
                for packed_row, state_index in enumerate(active):
                    executor.runtime.memcpy(
                        hidden_batch.ptr + packed_row * hidden_nbytes,
                        operations[state_index][depth][2].ptr,
                        hidden_nbytes,
                        HipMemcpyKind.DEVICE_TO_DEVICE,
                    )
                ids = tuple(states[index].request_id for index in active)
                tokens = tuple(operations[index][depth][0] for index in active)
                positions = tuple(operations[index][depth][1] for index in active)
                hidden = Tensor.from_handle(
                    hidden_batch.ptr,
                    (len(active), hidden_size),
                    DType.BF16,
                    Device("hip", 0),
                )
                if len(active) == 1:
                    executor.advance_state_only(
                        ids[0],
                        tokens[0],
                        positions[0],
                        Tensor.from_handle(
                            hidden.ptr,
                            (1, hidden_size),
                            DType.BF16,
                            Device("hip", 0),
                        ),
                    )
                else:
                    executor.advance_state_batch_only(
                        ids,
                        tokens,
                        positions,
                        hidden,
                    )
        finally:
            free(hidden_batch, runtime=executor.runtime)

    def _repair_provider_state(
        self,
        state: _MTP2RequestState,
        *,
        accepted_count: int,
        candidate_count: int,
    ) -> None:
        checkpoint = state.proposal_checkpoint
        context = state.proposal_context
        if checkpoint is None or context is None:
            raise RuntimeError("GGUF MTP2 provider repair has no proposal checkpoint")
        results = state.provider.last_results.get(int(state.request_id))
        if results is None or len(results) != int(candidate_count):
            raise RuntimeError("GGUF MTP2 provider repair lost proposal rows")
        accepted = int(accepted_count)
        if accepted < 0 or accepted > len(results):
            raise ValueError("accepted_count is outside proposal rows")
        executor = state.provider.executor
        if accepted == len(results):
            state.provider.advance_full_accept_tail(
                state.request_id,
                accepted_count=accepted,
            )
            return
        executor.restore_request_checkpoint(checkpoint)
        executor.advance_state_only(
            state.request_id,
            int(context.root_tokens[0]),
            int(context.root_positions[0]),
            context.target_hidden,
        )
        for index in range(accepted):
            proposal_row = results[index]
            executor.advance_state_only(
                state.request_id,
                int(proposal_row.token_id),
                int(context.root_positions[0]) + index + 1,
                proposal_row.hidden,
            )

    def _restore_provider_checkpoint(self, state: _MTP2RequestState) -> None:
        checkpoint = state.proposal_checkpoint
        if checkpoint is None:
            return
        try:
            state.provider.executor.restore_request_checkpoint(checkpoint)
        finally:
            state.provider.executor.release_request_checkpoint(checkpoint)
            state.proposal_checkpoint = None
            state.proposal_context = None
            state.proposal_device_batch = None

    def _release_provider_checkpoint(self, state: _MTP2RequestState) -> None:
        checkpoint = state.proposal_checkpoint
        if checkpoint is None:
            return
        state.provider.executor.release_request_checkpoint(checkpoint)
        state.proposal_checkpoint = None
        state.proposal_context = None
        state.proposal_device_batch = None

    def rollback_cycle(
        self,
        plan: SpecRequestPlan,
        candidate_graph: CandidateGraph | None,
        error: BaseException,
    ) -> None:
        del candidate_graph, error
        for request_id in plan.speculative_request_ids:
            state = self._states.get(int(request_id))
            if state is not None:
                self._restore_provider_checkpoint(state)
            self._drop_request(int(request_id), disable=True)

    def recover_cycle_failure(
        self,
        plan: SpecRequestPlan,
        error: BaseException,
    ) -> bool:
        """Fall back to AR only while every target cursor is still canonical."""

        del error
        rows = tuple(
            self.owner._row(int(request_id))
            for request_id in plan.speculative_request_ids
        )
        if not rows:
            return False
        for row in rows:
            if row.slot is None or row.lease is None:
                return False
            if int(row.lease.session.position) != int(row.slot.seq_position):
                return False
        for row in rows:
            row.mtp2_recoverable_failures += 1
            row.mtp2_failure_reasons.append("precommit_failure_ar_fallback")
        return True

    def release_request(self, request_id: int) -> None:
        rid = int(request_id)
        if rid in self._prompt_streaming_sinks:
            self._abort_prompt_streaming((rid,), stream=0)
        self._drop_request(rid, disable=False)
        self._intents.pop(rid, None)
        self._prompt_hidden_rows.pop(rid, None)
        self._disabled_requests.discard(rid)

    def close(self) -> None:
        if self._prompt_streaming_sinks:
            self._abort_prompt_streaming(
                tuple(self._prompt_streaming_sinks),
                stream=0,
            )
        for request_id in tuple(self._states):
            self._drop_request(request_id, disable=False)
        self._intents.clear()
        self._prompt_hidden_rows.clear()
        self._disabled_requests.clear()
        self._active_claims = None
        if self._batch_accept_workspace is not None:
            self._batch_accept_workspace.free()
            self._batch_accept_workspace = None
            self._batch_accept_owner = None
            self._batch_accept_remaining = None
            self._batch_accept_payload = None
            self._batch_accept_library = None

    def _open_batch_requests(self, request_ids: tuple[int, ...]) -> None:
        rows = [self.owner._row(request_id) for request_id in request_ids]
        targets = [row.lease.session for row in rows]
        max_positions = min(
            int(target.target_layout.max_sequence_length) for target in targets
        )
        provider_capacity = max(
            len(request_ids),
            min(4, int(getattr(self.owner, "capacity", len(request_ids)))),
        )
        provider, pool_key, _reused = self.generator._acquire_dense_mtp_draft_provider(
            targets[0],
            max_positions=max_positions,
            pool_enabled=self.owner._shared_runner is not None,
            max_requests=provider_capacity,
        )
        group_key = tuple(sorted(request_ids))
        group = _MTP2ProviderGroup(
            key=group_key,
            provider=provider,
            provider_pool_key=pool_key,
            request_ids=set(request_ids),
        )
        self._provider_groups[group_key] = group
        root_buffers: dict[int, DeviceBuffer] = {}
        verifiers: dict[int, Qwen35GGUFTransactionalVerifier] = {}
        try:
            for request_id in request_ids:
                provider.reset_request(request_id)
            root_buffers = self._catch_up_provider_batch(
                provider,
                request_ids,
                rows,
                targets,
            )
            for request_id, target in zip(request_ids, targets, strict=True):
                target._last_target_hidden_ptr = int(root_buffers[request_id].ptr)
                self._states[request_id] = _MTP2RequestState(
                    request_id=request_id,
                    provider=provider,
                    provider_pool_key=pool_key,
                    provider_group_key=group_key,
                    verifier=None,
                    root_hidden_buffer=root_buffers[request_id],
                )
        except Exception:
            for verifier in verifiers.values():
                verifier.close()
            for request_id, buffer in root_buffers.items():
                target = targets[request_ids.index(request_id)]
                free(buffer, runtime=target.runtime)
            for request_id in request_ids:
                provider.release_request(request_id)
                self._states.pop(request_id, None)
            self._provider_groups.pop(group_key, None)
            self.generator._release_mtp_draft_runner(pool_key, provider)
            raise

    def _attach_request_to_group(
        self,
        request_id: int,
        group: _MTP2ProviderGroup,
    ) -> _MTP2RequestState:
        row = self.owner._row(request_id)
        target = row.lease.session
        group.provider.reset_request(request_id)
        root_hidden = self._catch_up_provider(
            group.provider,
            request_id,
            row.prompt_ids,
            self._prompt_hidden_rows[request_id],
            target,
        )
        target._last_target_hidden_ptr = int(root_hidden.ptr)
        group.request_ids.add(request_id)
        return _MTP2RequestState(
            request_id=request_id,
            provider=group.provider,
            provider_pool_key=group.provider_pool_key,
            provider_group_key=group.key,
            verifier=None,
            root_hidden_buffer=root_hidden,
        )

    def _catch_up_provider_batch(
        self,
        provider: Any,
        request_ids: tuple[int, ...],
        rows: Sequence[Any],
        targets: Sequence[Any],
    ) -> dict[int, DeviceBuffer]:
        prompt_lengths = tuple(len(row.prompt_ids) for row in rows)
        hidden_size = int(provider.executor.hidden_size)
        count = len(rows)
        hidden_batch = malloc(
            count * hidden_size * DType.BF16.itemsize,
            runtime=targets[0].runtime,
        )
        root_buffers: dict[int, DeviceBuffer] = {}
        try:
            zero = np.zeros((hidden_size,), dtype=np.uint16)
            for position in range(max(prompt_lengths)):
                active = tuple(
                    index
                    for index, prompt_length in enumerate(prompt_lengths)
                    if position < prompt_length
                )
                hidden_rows = []
                for index in active:
                    request_id = request_ids[index]
                    prompt_hidden = self._prompt_hidden_rows[request_id]
                    hidden_rows.append(
                        zero
                        if position == 0
                        else np.ascontiguousarray(
                            float_array_to_bf16_bits(prompt_hidden[position - 1]),
                            dtype=np.uint16,
                        )
                    )
                hidden_bits = np.ascontiguousarray(np.stack(hidden_rows), dtype=np.uint16)
                copy_host_to_device(
                    hidden_batch,
                    host_array_ptr(hidden_bits),
                    hidden_bits.nbytes,
                    runtime=targets[0].runtime,
                )
                active_ids = tuple(request_ids[index] for index in active)
                active_tokens = tuple(
                    int(rows[index].prompt_ids[position]) for index in active
                )
                hidden = Tensor.from_handle(
                    hidden_batch.ptr,
                    (len(active), hidden_size),
                    DType.BF16,
                    Device("hip", 0),
                )
                if len(active) == 1:
                    provider.executor.run_step(
                        active_ids[0],
                        active_tokens[0],
                        position,
                        Tensor.from_handle(
                            hidden.ptr,
                            (1, hidden_size),
                            DType.BF16,
                            Device("hip", 0),
                        ),
                        return_logits=False,
                    )
                else:
                    provider.executor.run_step_batch(
                        active_ids,
                        active_tokens,
                        (position,) * len(active),
                        hidden,
                    )
            for request_id, row, target in zip(request_ids, rows, targets, strict=True):
                bits = np.ascontiguousarray(
                    float_array_to_bf16_bits(
                        self._prompt_hidden_rows[request_id][-1]
                    ),
                    dtype=np.uint16,
                )
                buffer = malloc(bits.nbytes, runtime=target.runtime)
                copy_host_to_device(
                    buffer,
                    host_array_ptr(bits),
                    bits.nbytes,
                    runtime=target.runtime,
                )
                root_buffers[request_id] = buffer
            return root_buffers
        except Exception:
            for request_id, buffer in root_buffers.items():
                target = targets[request_ids.index(request_id)]
                free(buffer, runtime=target.runtime)
            raise
        finally:
            free(hidden_batch, runtime=targets[0].runtime)

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
            max_requests=1,
        )
        group_key = (rid,)
        group = _MTP2ProviderGroup(
            key=group_key,
            provider=provider,
            provider_pool_key=pool_key,
            request_ids={rid},
        )
        self._provider_groups[group_key] = group
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
                provider_group_key=group_key,
                verifier=verifier,
                root_hidden_buffer=root_hidden_buffer,
            )
        except Exception:
            if verifier is not None:
                verifier.close()
            if root_hidden_buffer is not None:
                free(root_hidden_buffer, runtime=target.runtime)
            provider.release_request(rid)
            self._provider_groups.pop(group_key, None)
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
            self._release_provider_checkpoint(state)
            if state.verifier is not None:
                state.verifier.close()
            if int(getattr(target, "_last_target_hidden_ptr", 0)) == int(
                state.root_hidden_buffer.ptr
            ):
                target._last_target_hidden_ptr = 0
            free(state.root_hidden_buffer, runtime=target.runtime)
            state.provider.release_request(rid)
            group = self._provider_groups.get(state.provider_group_key)
            if group is not None:
                group.request_ids.discard(rid)
                if not group.request_ids:
                    self._provider_groups.pop(group.key, None)
                    self.generator._release_mtp_draft_runner(
                        group.provider_pool_key,
                        group.provider,
                    )
        if disable:
            self._disabled_requests.add(rid)


__all__ = ["Qwen35GGUFMTP2Adapter"]
