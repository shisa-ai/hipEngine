"""Staged C1 MoE GGUF NextN adapter for the Generation-2 resident owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, free, host_array_ptr, malloc
from hipengine.kernels.backends import backend_package_capability
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative.frontier import (
    ProviderAttachment,
    ProviderCatchupMode,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
)
from hipengine.speculative.interfaces import AcceptResult
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleTelemetry,
    SpecCycleTransaction,
    compose_speculative_claims,
)


@dataclass
class _MoeTargetHiddenSink:
    """Request-owned prompt trunk rows plus shifted NextN normalization."""

    request_id: int
    hidden_size: int
    total_rows: int
    target: Any
    buffer: DeviceBuffer
    normalized: DeviceBuffer
    normalized_bf16: DeviceBuffer
    finished: bool = False

    def consume(
        self,
        *,
        request_id: int,
        chunk_start: int,
        hidden_ptr: int,
        rows: int,
        stream: int,
    ) -> None:
        if int(request_id) != self.request_id:
            raise RuntimeError("MoE target-hidden sink request owner changed")
        count = int(rows)
        start = int(chunk_start)
        if count <= 0 or start < 0 or start + count > self.total_rows:
            raise ValueError("MoE target-hidden sink chunk is invalid")
        row_nbytes = self.hidden_size * DType.BF16.itemsize
        self.target.runtime.memcpy_async(
            self.buffer.ptr + start * row_nbytes,
            int(hidden_ptr),
            count * row_nbytes,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )

    def finish(self, *, request_id: int, total_rows: int, stream: int) -> None:
        if int(request_id) != self.request_id or int(total_rows) != self.total_rows:
            raise RuntimeError("MoE target-hidden sink completion identity changed")
        row_bf16 = self.hidden_size * DType.BF16.itemsize
        row_f32 = self.hidden_size * DType.FP32.itemsize
        for row in range(self.total_rows):
            self.target._run_output_norm_hidden(
                self.buffer.ptr + row * row_bf16,
                self.normalized_bf16.ptr,
                stream=int(stream),
                capture_hidden_seed_fp32=True,
            )
            self.target.runtime.memcpy_async(
                self.normalized.ptr + row * row_f32,
                self.target.scratch.hidden_seed_fp32.ptr,
                row_f32,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                int(stream),
            )
        self.target.runtime.memcpy_async(
            self.target.scratch.hidden_seed_fp32.ptr,
            self.normalized.ptr + (self.total_rows - 1) * row_f32,
            row_f32,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            int(stream),
        )
        self.target._hidden_seed_fp32_populated = True
        self.target._last_target_hidden_ptr = int(
            self.buffer.ptr + (self.total_rows - 1) * row_bf16
        )
        self.finished = True

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _MoeMTP2RequestState:
    request_id: int
    draft: Any
    draft_pool_key: Any | None
    context: Any
    key_cache: DeviceBuffer
    value_cache: DeviceBuffer
    target_hidden_buffer: DeviceBuffer
    target_hidden_ptr: int
    cache_len: int


class Qwen35GGUFMoEMTP2Adapter:
    """Real C1/K1-K2 complete cycles under Generation-2 request ownership.

    The retained native cycle is the strict provider+target transaction. The
    EngineService/ResidentEngineLoop still owns admission, one bounded cycle per
    tick, committed output publication, cancellation boundaries, and teardown.
    No whole-request legacy generation call is used.
    """

    staged_frontier = False

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
        self.candidate_budget = min(2, max(1, int(candidate_budget)))
        self.quant = str(quant)
        self._intents: dict[int, int] = {}
        self._states: dict[int, _MoeMTP2RequestState] = {}
        self._prompt_sinks: dict[int, _MoeTargetHiddenSink] = {}
        self._target_hidden_slabs: dict[
            int,
            tuple[Any, DeviceBuffer, DeviceBuffer, DeviceBuffer],
        ] = {}
        self._draft_kv_slabs: dict[
            int,
            tuple[Any, DeviceBuffer, DeviceBuffer],
        ] = {}
        self._disabled_requests: set[int] = set()
        self._transaction_sequence = 0
        self._assets: Any | None = None

    def register_request(self, request_id: int, candidate_budget: int) -> None:
        rid = int(request_id)
        self._intents[rid] = min(
            self.candidate_budget,
            max(1, int(candidate_budget)),
        )
        self._disabled_requests.discard(rid)

    def begin_prompt_streaming(
        self,
        request_ids: Sequence[int],
        *,
        checkpoints: Mapping[int, Callable[[], None] | None] | None = None,
    ) -> tuple[_MoeTargetHiddenSink, ...] | None:
        del checkpoints
        ids = tuple(int(value) for value in request_ids)
        if not ids or len(ids) != 1 or int(getattr(self.owner, "capacity", 1)) != 1:
            return None
        rid = ids[0]
        if rid not in self._intents or rid in self._disabled_requests:
            return None
        row = self.owner._row(rid)
        if row.lease is None:
            return None
        target = row.lease.session
        hidden_size = int(target.runner.hidden_size)
        rows = len(row.prompt_ids)
        target_key = id(target)
        slab = self._target_hidden_slabs.get(target_key)
        if slab is None:
            capacity = 95
            slab = (
                target,
                malloc(
                    capacity * hidden_size * DType.BF16.itemsize,
                    runtime=target.runtime,
                ),
                malloc(
                    capacity * hidden_size * DType.FP32.itemsize,
                    runtime=target.runtime,
                ),
                malloc(
                    hidden_size * DType.BF16.itemsize,
                    runtime=target.runtime,
                ),
            )
            self._target_hidden_slabs[target_key] = slab
        if rows > int(slab[1].nbytes) // (hidden_size * DType.BF16.itemsize):
            return None
        sink = _MoeTargetHiddenSink(
            request_id=rid,
            hidden_size=hidden_size,
            total_rows=rows,
            target=target,
            buffer=slab[1],
            normalized=slab[2],
            normalized_bf16=slab[3],
        )
        self._prompt_sinks[rid] = sink
        return (sink,)

    def finish_prompt_streaming(
        self,
        request_ids: Sequence[int],
        *,
        success: bool,
        stream: int = 0,
    ) -> None:
        del stream
        for request_id in tuple(int(value) for value in request_ids):
            sink = self._prompt_sinks.get(request_id)
            if sink is None:
                continue
            if success and sink.finished:
                continue
            self._prompt_sinks.pop(request_id, None)
            self._detach_sink(sink)
            if success:
                raise RuntimeError("MoE target-hidden sink did not finish")

    def observe_prefill_result(
        self,
        request_id: int,
        prompt_ids: Sequence[int],
        result: Any,
    ) -> None:
        rid = int(request_id)
        if rid not in self._intents or rid in self._disabled_requests:
            return
        row = self.owner._row(rid)
        if row.lease is None:
            raise RuntimeError("MoE MTP2 prefill has no target session")
        target = row.lease.session
        prompt = tuple(int(token) for token in prompt_ids)
        if not prompt:
            raise RuntimeError("MoE MTP2 prompt must be non-empty")
        if rid in self._states:
            self._release_state(rid)
        sink = self._prompt_sinks.pop(rid, None)
        if sink is None:
            if len(prompt) > 95:
                self._disabled_requests.add(rid)
                return
            raise RuntimeError("MoE MTP2 prefill lost its target-hidden sink")
        if not sink.finished:
            raise RuntimeError("MoE MTP2 prefill target-hidden sink is incomplete")
        from hipengine.generation.qwen35_gguf import (
            _allocate_mtp_dense_kv,
            _new_mtp_context,
        )

        assets = self._assets or self.generator._load_mtp_serving_assets()
        self._assets = assets
        draft, pool_key, _reused = self.generator._acquire_mtp_draft_runner(
            assets,
            runtime=target.runtime,
            pool_enabled=self.owner._shared_runner is not None,
        )
        shifted = None
        target_key = id(target)
        kv_slab = self._draft_kv_slabs.get(target_key)
        if kv_slab is None:
            key_cache, value_cache, _buffers = _allocate_mtp_dense_kv(
                runtime=target.runtime,
                capacity=min(1024, int(target.target_layout.max_sequence_length)),
                qk_head_dim=int(draft.qk_head_dim),
                kv_heads=int(draft.num_kv_heads),
            )
            kv_slab = (target, key_cache, value_cache)
            self._draft_kv_slabs[target_key] = kv_slab
        key_cache, value_cache = kv_slab[1], kv_slab[2]
        try:
            context = _new_mtp_context(
                target,
                token_id=int(result.token_id),
                position=int(target.position) - 1,
                mtp_block=draft,
            )
            hidden_size = int(target.runner.hidden_size)
            row_f32 = hidden_size * DType.FP32.itemsize
            shifted = malloc(len(prompt) * row_f32, runtime=target.runtime)
            zero = np.zeros((hidden_size,), dtype=np.float32)
            target.runtime.memcpy_async(
                shifted.ptr,
                host_array_ptr(zero),
                zero.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
                0,
            )
            if len(prompt) > 1:
                target.runtime.memcpy_async(
                    shifted.ptr + row_f32,
                    sink.normalized.ptr,
                    (len(prompt) - 1) * row_f32,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    0,
                )
            target.runtime.stream_synchronize(0)
            cache_len = draft.write_kv_rows_from_device_seed_base(
                shifted.ptr,
                np.asarray(prompt, dtype=np.int64),
                positions=np.arange(len(prompt), dtype=np.int64),
                rope_cos=assets.rope_cos,
                rope_sin=assets.rope_sin,
                dense_key_cache=key_cache,
                dense_value_cache=value_cache,
                dense_cache_len=0,
            )
            self._states[rid] = _MoeMTP2RequestState(
                request_id=rid,
                draft=draft,
                draft_pool_key=pool_key,
                context=context,
                key_cache=key_cache,
                value_cache=value_cache,
                target_hidden_buffer=sink.buffer,
                target_hidden_ptr=int(target._last_target_hidden_ptr),
                cache_len=int(cache_len),
            )
            free(shifted, runtime=target.runtime)
            shifted = None
        except Exception:
            if shifted is not None:
                free(shifted, runtime=target.runtime)
            self.generator._release_mtp_draft_runner(pool_key, draft)
            self._detach_sink(sink)
            raise

    def _target_mode(self, target: Any) -> str:
        if self.target_verify_mode != "native":
            return self.target_verify_mode
        limit = int(
            backend_package_capability(
                self.generator.backend,
                "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
                int(target.position),
            )
        )
        return "native" if int(target.position) <= limit else "serial_exact"

    def capability(
        self,
        request_semantics: Sequence[SpeculativeRequestSemantics],
    ) -> SpeculativeCapability | None:
        semantics = tuple(request_semantics)
        if (
            not self.enabled
            or len(semantics) != 1
            or int(getattr(self.owner, "capacity", 1)) != 1
        ):
            return None
        item = semantics[0]
        rid = int(item.request_id)
        state = self._states.get(rid)
        if state is None or rid in self._disabled_requests:
            return None
        row = self.owner._row(rid)
        if (
            not row.native_greedy
            or not row.first_token_emitted
            or row.lease is None
            or row.slot is None
        ):
            return None
        target = row.lease.session
        max_context = min(
            95,
            int(target.target_layout.max_sequence_length) - self.candidate_budget - 1,
        )
        if int(item.context_tokens) > max_context:
            return None
        profile = getattr(self.generator, "execution_profile", None)
        profile = getattr(profile, "value", profile)
        mode = self._target_mode(target)
        return SpeculativeCapability(
            capability_key=(
                f"gguf_moe_mtp2_c1:{self.generator.backend}:{self.quant}:"
                f"{mode}:b{self.candidate_budget}"
            ),
            target_key="qwen_moe_gguf",
            provider_key="qwen_nextn_moe",
            method_key="mtp2",
            policy_fingerprint=f"moe-nextn:{mode}:b{self.candidate_budget}",
            execution_profile=str(profile or "legacy_exact"),
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
            graph_supported=mode == "native",
            eager_supported=True,
            strict_fallback_key="gguf_target_ar",
            max_context_tokens=max_context,
            terminal_zero_accept_supported=True,
        )

    def claims_fit(self, plan: SpecRequestPlan) -> bool:
        return bool(
            self.enabled
            and len(plan.speculative_request_ids) == 1
            and int(plan.speculative_request_ids[0]) in self._states
            and int(plan.speculative_request_ids[0]) not in self._disabled_requests
        )

    def prepare_k0(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del stream
        reasons = dict(zip(plan.request_ids, plan.reasons, strict=True))
        semantics = {
            int(item.request_id): item for item in request_semantics
        }
        for request_id in plan.request_ids:
            rid = int(request_id)
            state = self._states.get(rid)
            if state is None or reasons[rid] not in {
                SpecPlanReason.NO_PROVIDER,
                SpecPlanReason.POLICY_SELECTED_AR,
                SpecPlanReason.TARGET_GRAPH_OUTPUT_ROOM_MISS,
            }:
                continue
            row = self.owner._row(rid)
            if row.lease is None or row.slot is None or not row.first_token_emitted:
                continue
            item = semantics.get(rid)
            if (
                item is not None
                and int(item.remaining_decode) <= 1
                and int(row.mtp2_cycles) > 0
            ):
                # The retained native complete-cycle graph owns selected target
                # state. Stable target-hidden slabs keep its captured pointers
                # valid across requests, so the ordinary one-token AR tail may
                # consume that state directly.
                continue
            target = row.lease.session
            pending = state.context.pending_seed
            if pending is None:
                continue
            state.cache_len = state.draft.write_kv_rows_from_device_seed_base(
                int(pending.hidden_ptr),
                np.asarray([int(row.slot.generated_ids[-1])], dtype=np.int64),
                positions=np.asarray([int(target.position)], dtype=np.int64),
                rope_cos=self._assets.rope_cos,
                rope_sin=self._assets.rope_sin,
                dense_key_cache=state.key_cache,
                dense_value_cache=state.value_cache,
                dense_cache_len=int(state.cache_len),
            )
            row.mtp2_k0_catchups += 1

    def execute_cycle(
        self,
        plan: SpecRequestPlan,
        *,
        commit: bool,
    ) -> SpecCycleResult:
        """Execute one bounded native complete cycle under Generation-2 ownership."""

        if not commit:
            raise ValueError("MoE MTP2 complete cycle requires commit=True")
        if len(plan.speculative_request_ids) != 1:
            raise NotImplementedError("MoE MTP2 complete cycle is C1-only")
        rid = int(plan.speculative_request_ids[0])
        state = self._states[rid]
        row = self.owner._row(rid)
        if row.lease is None or row.slot is None:
            raise RuntimeError("MoE MTP2 complete cycle has no resident target root")
        target = row.lease.session
        slot = row.slot
        self.owner._flush_row_owner(row)
        if int(target.position) != int(slot.seq_position):
            raise RuntimeError("MoE MTP2 complete-cycle target cursor is stale")
        budget = int(plan.candidate_counts[0])
        remaining = max(0, int(row.request.max_tokens) - len(slot.generated_ids))
        if remaining < 1:
            raise RuntimeError("MoE MTP2 complete cycle has no output room")
        complete_claims = compose_speculative_claims(
            plan.operation_id,
            self.component_claims(plan),
        )
        self._transaction_sequence += 1
        transaction_id = self._transaction_sequence
        start_position = int(target.position)
        cache_before = int(state.cache_len)
        native = target.run_native_spec_mtp_cycle(
            state.draft,
            state.context,
            root_token=int(slot.generated_ids[-1]),
            root_position=start_position,
            candidate_budget=budget,
            remaining_decode=remaining,
            rope_cos=self._assets.rope_cos,
            rope_sin=self._assets.rope_sin,
            draft_key_cache=state.key_cache,
            draft_value_cache=state.value_cache,
            draft_cache_len=cache_before,
            cycle_id=int(plan.cycle_id),
            transaction_id=transaction_id,
            target_bulk_attention_mode="bulk",
            k1_disable_f32_verifier=True,
        )
        state.cache_len = int(native.draft_cache_len_after)
        output_ids = tuple(int(token) for token in native.output_token_ids)
        accepted = int(native.accepted_draft_tokens)
        if not output_ids or int(native.end_position) != int(target.position):
            raise RuntimeError("MoE MTP2 native cycle returned inconsistent target state")
        slot.generated_ids.extend(output_ids)
        slot.prev_token = int(output_ids[-1])
        slot.seq_position = int(target.position)
        slot.native_decode_steps += 1
        slot.done = len(slot.generated_ids) >= int(row.request.max_tokens)
        row.mtp2_cycles += 1
        row.mtp2_candidate_counts.append(budget)
        row.mtp2_accepted_counts.append(accepted)
        row.mtp2_proposal_ms += float(native.proposal_wall_ms)
        row.mtp2_target_ms += float(native.target_wall_ms)
        row.mtp2_provider_update_ms += float(native.mtp_kv_commit_wall_ms)
        actual_route = (
            "graph"
            if bool(getattr(target, "last_native_spec_target_submitted", False))
            else "eager"
        )
        row.mtp2_execution_routes.append(actual_route)
        transaction = SpecCycleTransaction(
            operation_id=plan.operation_id,
            transaction_id=transaction_id,
            cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=complete_claims,
            pre_target_cursors=(start_position,),
            pre_provider_cursors=(cache_before,),
            pre_rng_counters=(0,),
            target_transaction_mode=plan.target_transaction_mode,
            provider_transaction_mode=plan.provider_transaction_mode,
            target_owner=f"{plan.operation_id}:gguf-moe-target",
            provider_owner=f"{plan.operation_id}:nextn-moe",
            provider_request_ids=(rid,),
            target_checkpoint_ids=(f"target:{rid}:{start_position}",),
            provider_checkpoint_ids=(f"provider:{rid}:{cache_before}",),
            target_open=True,
            provider_open=True,
            target_committed=True,
            provider_committed=True,
        )
        accept = AcceptResult(
            request_ids=plan.request_ids,
            accepted_counts=(accepted,),
            accepted_tokens=(tuple(output_ids[:accepted]),),
            transaction_id=transaction_id,
            correction_or_bonus_tokens=(int(output_ids[accepted]),),
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
            execution_route=actual_route,
            proposal_seconds=float(native.proposal_wall_ms) / 1000.0,
            target_seconds=float(native.target_wall_ms) / 1000.0,
            provider_update_seconds=float(native.mtp_kv_commit_wall_ms) / 1000.0,
            weight_sweeps=len(plan.target_row_decomposition),
        )
        return SpecCycleResult.committed(transaction, accept, telemetry=telemetry)

    def component_claims(
        self,
        plan: SpecRequestPlan,
    ) -> Mapping[str, ResourceClaimSet]:
        return {
            "target": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:target",
                {"gguf_moe_mtp2.target_rows": int(plan.logical_frontier_rows)},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"gguf_moe_mtp2.provider_rows": int(sum(plan.candidate_counts))},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "transient": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:transient",
                {"gguf_moe_mtp2.result_rows": 1},
                lifetime=ClaimLifetime.WORK_ITEM,
            ),
        }

    def rollback_cycle(self, plan: SpecRequestPlan, *args: Any) -> None:
        del args
        self._disabled_requests.update(int(value) for value in plan.speculative_request_ids)

    def recover_cycle_failure(
        self,
        plan: SpecRequestPlan,
        error: BaseException,
    ) -> bool:
        reason = f"{type(error).__name__}:{error}"
        rows = tuple(self.owner._row(int(value)) for value in plan.speculative_request_ids)
        if any(
            row.slot is None
            or row.lease is None
            or int(row.lease.session.position) != int(row.slot.seq_position)
            for row in rows
        ):
            return False
        for row in rows:
            row.mtp2_recoverable_failures += 1
            row.mtp2_failure_reasons.extend(("precommit_failure_ar_fallback", reason))
            self._disabled_requests.add(int(row.request_id))
        return True

    @staticmethod
    def _detach_sink(sink: _MoeTargetHiddenSink) -> None:
        target = sink.target
        current = int(getattr(target, "_last_target_hidden_ptr", 0))
        if int(sink.buffer.ptr) <= current < int(sink.buffer.ptr + sink.buffer.nbytes):
            target._last_target_hidden_ptr = 0

    def _release_state(self, request_id: int) -> None:
        rid = int(request_id)
        state = self._states.pop(rid, None)
        if state is None:
            return
        row = self.owner._row(rid)
        target = None if row.lease is None else row.lease.session
        if target is not None:
            if int(getattr(target, "_last_target_hidden_ptr", 0)) == int(
                state.target_hidden_ptr
            ):
                target._last_target_hidden_ptr = 0
        self.generator._release_mtp_draft_runner(
            state.draft_pool_key,
            state.draft,
        )

    def release_request(self, request_id: int) -> None:
        rid = int(request_id)
        self._release_state(rid)
        sink = self._prompt_sinks.pop(rid, None)
        if sink is not None:
            self._detach_sink(sink)
        self._intents.pop(rid, None)
        self._disabled_requests.discard(rid)

    def close(self) -> None:
        for request_id in tuple(set(self._states) | set(self._prompt_sinks)):
            self.release_request(request_id)
        self._intents.clear()
        self._disabled_requests.clear()
        for target, buffer, normalized, normalized_bf16 in self._target_hidden_slabs.values():
            self._detach_sink(
                _MoeTargetHiddenSink(
                    request_id=-1,
                    hidden_size=int(target.runner.hidden_size),
                    total_rows=1,
                    target=target,
                    buffer=buffer,
                    normalized=normalized,
                    normalized_bf16=normalized_bf16,
                )
            )
            free(normalized_bf16, runtime=target.runtime)
            free(normalized, runtime=target.runtime)
            free(buffer, runtime=target.runtime)
        self._target_hidden_slabs.clear()
        for target, key_cache, value_cache in self._draft_kv_slabs.values():
            free(value_cache, runtime=target.runtime)
            free(key_cache, runtime=target.runtime)
        self._draft_kv_slabs.clear()


__all__ = ["Qwen35GGUFMoEMTP2Adapter"]
