"""Staged C1 MoE GGUF NextN adapter for the Generation-2 resident owner."""

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
from hipengine.core.specdec2_scope import moe_physical_c2_numerics_session
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100.speculative.dflash_accept import (
    ACCEPT_PACKED_PAYLOAD_FIELDS,
    build_dflash_accept,
    dflash_accept_chain_i32_packed,
)
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
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
    TargetVerifyBatch,
    TargetVerifyBuffers,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
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
    proposal_tokens: tuple[int, ...] = ()
    proposed_cache_len: int | None = None
    last_proposal_seconds: float = 0.0


class Qwen35GGUFMoEMTP2Adapter:
    """Real C1/K1-K2 complete cycles under Generation-2 request ownership.

    The retained native cycle is the strict provider+target transaction. The
    EngineService/ResidentEngineLoop still owns admission, one bounded cycle per
    tick, committed output publication, cancellation boundaries, and teardown.
    No whole-request legacy generation call is used.
    """

    @property
    def staged_frontier(self) -> bool:
        return int(getattr(self.owner, "capacity", 1)) > 1

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
        self._active_claims: ResourceClaimSet | None = None
        self._batch_accept_workspace: RuntimeWorkspace | None = None
        self._batch_accept_owner: TargetVerifyBufferOwner | None = None
        self._batch_accept_remaining: Tensor | None = None
        self._batch_accept_payload: Tensor | None = None
        self._batch_accept_library: Any | None = None

    def register_request(
        self,
        request_id: int,
        candidate_budget: int,
        *,
        static_eligibility: Any | None = None,
    ) -> None:
        del static_eligibility  # Static policy is enforced by the model key and due width.
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
        capacity = int(getattr(self.owner, "capacity", 1))
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(ids) > 2
            or capacity not in {1, 2}
        ):
            return None
        if any(
            rid not in self._intents or rid in self._disabled_requests
            for rid in ids
        ):
            return None
        existing = tuple(self._prompt_sinks.get(rid) for rid in ids)
        if all(sink is not None for sink in existing):
            return tuple(sink for sink in existing if sink is not None)
        if any(sink is not None for sink in existing):
            raise RuntimeError("MoE prompt streaming ownership is only opened once")
        created: list[_MoeTargetHiddenSink] = []
        try:
            for rid in ids:
                row = self.owner._row(rid)
                if row.lease is None:
                    for created_sink in created:
                        self._prompt_sinks.pop(created_sink.request_id, None)
                        self._detach_sink(created_sink)
                    return None
                target = row.lease.session
                hidden_size = int(target.runner.hidden_size)
                rows = len(row.prompt_ids)
                target_key = id(target)
                slab = self._target_hidden_slabs.get(target_key)
                if slab is None:
                    slab_rows = 95
                    slab = (
                        target,
                        malloc(
                            slab_rows * hidden_size * DType.BF16.itemsize,
                            runtime=target.runtime,
                        ),
                        malloc(
                            slab_rows * hidden_size * DType.FP32.itemsize,
                            runtime=target.runtime,
                        ),
                        malloc(
                            hidden_size * DType.BF16.itemsize,
                            runtime=target.runtime,
                        ),
                    )
                    self._target_hidden_slabs[target_key] = slab
                if rows > int(slab[1].nbytes) // (
                    hidden_size * DType.BF16.itemsize
                ):
                    for created_sink in created:
                        self._prompt_sinks.pop(created_sink.request_id, None)
                        self._detach_sink(created_sink)
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
                created.append(sink)
            return tuple(created)
        except Exception:
            for sink in created:
                self._prompt_sinks.pop(sink.request_id, None)
                self._detach_sink(sink)
            raise

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
            row.mtp2_prompt_streaming = True
            row.mtp2_prompt_prime_rows = len(prompt)
            row.mtp2_prompt_carried_bytes = (
                int(target.runner.hidden_size) * DType.BF16.itemsize
            )
            row.mtp2_prompt_fallback_reason = None
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
        capacity = int(getattr(self.owner, "capacity", 1))
        if (
            not self.enabled
            or capacity not in {1, 2}
            or len(semantics) != capacity
        ):
            return None
        targets = []
        max_context = 95
        modes = set()
        for item in semantics:
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
            request_max_context = min(
                95,
                int(target.target_layout.max_sequence_length)
                - self.candidate_budget
                - 1,
            )
            if int(item.context_tokens) > request_max_context:
                return None
            max_context = min(max_context, request_max_context)
            modes.add(self._target_mode(target))
            targets.append(target)
        if len(modes) != 1:
            return None
        profile = getattr(self.generator, "execution_profile", None)
        profile = getattr(profile, "value", profile)
        if capacity > 1 and str(profile) != "production":
            return None
        mode = next(iter(modes))
        return SpeculativeCapability(
            capability_key=(
                f"gguf_moe_mtp2_c{capacity}:{self.generator.backend}:{self.quant}:"
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
            max_requests=capacity,
            max_candidates_per_request=self.candidate_budget,
            max_frontier_rows=capacity * (self.candidate_budget + 1),
            proposal_widths=tuple(range(1, capacity + 1)),
            target_row_buckets=tuple(
                range(capacity * 2, capacity * (self.candidate_budget + 1) + 1)
            ),
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            graph_supported=mode == "native",
            eager_supported=True,
            strict_fallback_key="gguf_target_ar",
            max_context_tokens=max_context,
            terminal_zero_accept_supported=True,
        )

    def claims_fit(self, plan: SpecRequestPlan) -> bool:
        capacity = int(getattr(self.owner, "capacity", 1))
        return bool(
            self.enabled
            and capacity in {1, 2}
            and len(plan.speculative_request_ids) == capacity
            and all(
                int(request_id) in self._states
                and int(request_id) not in self._disabled_requests
                for request_id in plan.speculative_request_ids
            )
        )

    def reserve_claims(self, claims: ResourceClaimSet) -> str:
        if self._active_claims is not None:
            raise RuntimeError("MoE MTP2 claims are already reserved")
        self._active_claims = claims
        return claims.claim_id

    def release_claims(self, reservation: str) -> None:
        if (
            self._active_claims is None
            or self._active_claims.claim_id != str(reservation)
        ):
            raise RuntimeError("MoE MTP2 claim release does not match ownership")
        self._active_claims = None

    def prepare_requests(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        ids = tuple(int(value) for value in plan.speculative_request_ids)
        if len(ids) != 2 or any(request_id not in self._states for request_id in ids):
            raise RuntimeError("MoE physical C2 requests are not prompt-primed")
        for request_id in ids:
            self.owner._flush_row_owner(self.owner._row(request_id))

    def propose_batch(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> CandidateGraph:
        del stream
        ids = tuple(int(value) for value in plan.speculative_request_ids)
        if len(ids) != 2:
            raise NotImplementedError("MoE staged proposal currently requires C2")
        semantics = {
            int(item.request_id): item for item in request_semantics
        }
        counts = tuple(
            int(plan.candidate_counts[plan.request_ids.index(request_id)])
            for request_id in ids
        )
        if any(count < 1 or count > self.candidate_budget for count in counts):
            raise ValueError("MoE C2 proposal count is outside the adapter budget")
        rows = tuple(self.owner._row(request_id) for request_id in ids)
        states = tuple(self._states[request_id] for request_id in ids)
        roots = []
        positions = []
        candidate_rows: list[tuple[int, ...]] = []
        started = time.perf_counter()
        for request_id, count, row, state in zip(
            ids, counts, rows, states, strict=True
        ):
            if row.lease is None or row.slot is None:
                raise RuntimeError("MoE C2 proposal lost resident target ownership")
            target = row.lease.session
            if int(target.position) != int(row.slot.seq_position):
                raise RuntimeError("MoE C2 proposal target cursor is stale")
            pending = state.context.pending_seed
            if pending is None or int(getattr(pending, "hidden_ptr", 0)) <= 0:
                raise RuntimeError("MoE C2 proposal has no resident pending seed")
            root_token = int(row.slot.generated_ids[-1])
            root_position = int(target.position)
            tokens, topk, proposed_cache_len = (
                state.draft.propose_chain_from_device_seed(
                    int(pending.hidden_ptr),
                    start_token=root_token,
                    start_position=root_position,
                    draft_n_max=count,
                    top_k=1,
                    rope_cos=self._assets.rope_cos,
                    rope_sin=self._assets.rope_sin,
                    dense_key_cache=state.key_cache,
                    dense_value_cache=state.value_cache,
                    dense_cache_len=int(state.cache_len),
                    draft_p_min=0.0,
                )
            )
            candidate_tokens = tuple(int(token) for token in tokens)
            if (
                len(candidate_tokens) != count
                or len(topk) != count
                or any(tuple(int(value) for value in values) != (candidate_tokens[index],)
                       for index, values in enumerate(topk))
            ):
                raise RuntimeError("MoE C2 provider did not produce one top-1 chain")
            if int(proposed_cache_len) != int(state.cache_len) + count:
                raise RuntimeError("MoE C2 provider returned an invalid speculative cursor")
            state.proposal_tokens = candidate_tokens
            state.proposed_cache_len = int(proposed_cache_len)
            roots.append(root_token)
            positions.append(root_position)
            candidate_rows.append(candidate_tokens)
        proposal_seconds = time.perf_counter() - started
        for state in states:
            state.last_proposal_seconds = proposal_seconds
        offsets = [0]
        for values in candidate_rows:
            offsets.append(offsets[-1] + len(values))
        row_to_request = tuple(
            request_id
            for request_id, values in zip(ids, candidate_rows, strict=True)
            for _ in values
        )
        draft_depths = tuple(
            depth
            for values in candidate_rows
            for depth in range(1, len(values) + 1)
        )
        parents: list[int] = []
        cursor = 0
        for values in candidate_rows:
            for index in range(len(values)):
                parents.append(-1 if index == 0 else cursor + index - 1)
            cursor += len(values)
        return CandidateGraph(
            provider_key=str(plan.provider_key),
            method_key="mtp2",
            policy_fingerprint="moe-nextn-request-major-c2",
            cycle_id=int(plan.cycle_id),
            transaction_id=int(plan.cycle_id),
            request_ids=plan.request_ids,
            resident_slots=plan.resident_slots,
            root_positions=tuple(
                int(semantics[request_id].context_tokens) - 1
                for request_id in plan.request_ids
            ),
            row_offsets=tuple(offsets),
            row_to_request=row_to_request,
            parent_candidate_rows=tuple(parents),
            draft_depths=draft_depths,
            active_mask=(True,) * sum(counts),
            candidate_tokens=tuple(
                token for values in candidate_rows for token in values
            ),
            mode="verify_chain",
            provider_metadata=(
                ("physical_provider_rows", len(ids)),
                ("provider_weight_sweeps", len(ids)),
            ),
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

    def _batch_accept_resources(
        self,
        runtime: Any,
    ) -> tuple[TargetVerifyBufferOwner, Tensor, Tensor]:
        if self._batch_accept_workspace is None:
            workspace = RuntimeWorkspace(device=Device("hip", 0), runtime=runtime)
            spec = TargetVerifyBufferSpec(
                backend=str(self.generator.backend),
                bucket="gguf-moe-mtp2-physical-r6-c2",
                device=Device("hip", 0),
                max_rows=6,
                max_requests=2,
                mode="verify_chain",
            )
            self._batch_accept_workspace = workspace
            self._batch_accept_owner = TargetVerifyBufferOwner.allocate(
                spec,
                workspace=workspace,
            )
            self._batch_accept_remaining = workspace.reserve_tensor(
                "target_verify/gguf-moe-mtp2-r6-c2/remaining_decode",
                (2,),
                DType.INT32,
            )
            self._batch_accept_payload = workspace.reserve_tensor(
                "target_verify/gguf-moe-mtp2-r6-c2/packed_accept_payload",
                (2, ACCEPT_PACKED_PAYLOAD_FIELDS),
                DType.INT32,
            )
        if (
            self._batch_accept_owner is None
            or self._batch_accept_remaining is None
            or self._batch_accept_payload is None
        ):
            raise RuntimeError("MoE physical accept workspace is incomplete")
        return (
            self._batch_accept_owner,
            self._batch_accept_remaining,
            self._batch_accept_payload,
        )

    @staticmethod
    def _upload_accept_array(
        tensor: Tensor,
        values: np.ndarray,
        runtime: Any,
    ) -> None:
        array = np.ascontiguousarray(values)
        if array.nbytes > tensor.numel * tensor.dtype.itemsize:
            raise ValueError("MoE physical accept upload exceeds capacity")
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
        owner, remaining_owner, payload_owner = self._batch_accept_resources(runtime)
        buffers = owner.bind(batch, transaction_id=int(transaction_id))
        requests = len(batch.request_ids)
        remaining = Tensor.from_handle(
            remaining_owner.ptr,
            (requests,),
            DType.INT32,
            remaining_owner.device,
        )
        payload = Tensor.from_handle(
            payload_owner.ptr,
            (requests, ACCEPT_PACKED_PAYLOAD_FIELDS),
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
            requests,
            batch.rows,
            library=self._batch_accept_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        payload_host = np.empty(
            (requests, ACCEPT_PACKED_PAYLOAD_FIELDS),
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

    def _repair_physical_provider(
        self,
        state: _MoeMTP2RequestState,
        target: Any,
        result: Any,
        *,
        accepted_count: int,
        output_ids: tuple[int, ...],
        start_position: int,
    ) -> None:
        accepted = int(accepted_count)
        proposed = state.proposed_cache_len
        if proposed is None or proposed != int(state.cache_len) + len(
            state.proposal_tokens
        ):
            raise RuntimeError("MoE physical provider lost its proposal cursor")
        consumed_rows = accepted + 1
        if len(output_ids) != consumed_rows:
            raise RuntimeError("MoE physical provider repair output count changed")
        hidden = getattr(target, "_verify_hidden_seed_buf", None)
        if hidden is None or int(getattr(target, "_verify_hidden_seed_rows_populated", 0)) < consumed_rows:
            raise RuntimeError("MoE physical target omitted verifier hidden rows")
        verify_seeds = tuple(
            target.mtp_verify_seed(
                row,
                token_id=int(output_ids[row]),
                position=int(start_position) + row,
                hidden_seed_base_ptr=int(hidden.ptr),
                hidden_seed_row_count=int(
                    getattr(target, "_verify_hidden_seed_rows_populated", 0)
                ),
            )
            for row in range(consumed_rows)
        )
        state.context.record_verify_seeds(verify_seeds)
        state.context.accept(accepted)
        committed_cache_len = int(state.cache_len) + 1
        if accepted:
            committed_cache_len = state.draft.write_kv_rows_from_device_seed_base(
                int(hidden.ptr),
                np.ascontiguousarray(output_ids[:accepted], dtype=np.int64),
                positions=np.arange(
                    int(start_position) + 1,
                    int(start_position) + 1 + accepted,
                    dtype=np.int64,
                ),
                rope_cos=self._assets.rope_cos,
                rope_sin=self._assets.rope_sin,
                dense_key_cache=state.key_cache,
                dense_value_cache=state.value_cache,
                dense_cache_len=committed_cache_len,
            )
        expected = int(state.cache_len) + 1 + accepted
        if int(committed_cache_len) != expected:
            raise RuntimeError("MoE physical provider commit cursor diverged")
        state.cache_len = expected
        state.proposal_tokens = ()
        state.proposed_cache_len = None

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
            raise ValueError("MoE physical target frontier requires commit=True")
        if self._active_claims != complete_claims:
            raise RuntimeError("MoE physical target does not own complete claims")
        ids = tuple(int(value) for value in plan.speculative_request_ids)
        if len(ids) != 2 or frontier.target_batch is None:
            raise NotImplementedError("MoE staged target currently requires physical C2")
        batch = frontier.target_batch
        states = tuple(self._states[request_id] for request_id in ids)
        rows = tuple(self.owner._row(request_id) for request_id in ids)
        if any(row.lease is None or row.slot is None for row in rows):
            raise RuntimeError("MoE physical target lost resident rows")
        targets = tuple(row.lease.session for row in rows)
        starts = tuple(int(target.position) for target in targets)
        caches = tuple(int(state.cache_len) for state in states)
        self._transaction_sequence += 1
        transaction_id = self._transaction_sequence
        transaction = SpecCycleTransaction(
            operation_id=plan.operation_id,
            transaction_id=transaction_id,
            cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=complete_claims,
            pre_target_cursors=starts,
            pre_provider_cursors=caches,
            pre_rng_counters=(0, 0),
            target_transaction_mode=plan.target_transaction_mode,
            provider_transaction_mode=plan.provider_transaction_mode,
            target_owner=f"{plan.operation_id}:gguf-moe-target-c2",
            provider_owner=f"{plan.operation_id}:nextn-moe-c2",
            provider_request_ids=ids,
            target_checkpoint_ids=tuple(
                f"target:{request_id}:{position}"
                for request_id, position in zip(ids, starts, strict=True)
            ),
            provider_checkpoint_ids=tuple(
                f"provider:{request_id}:{cursor}"
                for request_id, cursor in zip(ids, caches, strict=True)
            ),
            target_open=True,
            provider_open=True,
        )
        cancelled = tuple(int(value) for value in cancelled_request_ids())
        if cancelled:
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
        root_rows = dict(zip(batch.request_ids, batch.root_rows, strict=True))
        candidate_rows = {
            request_id: tuple(
                sorted(
                    (
                        row
                        for row in batch.candidate_rows
                        if batch.row_to_request[row] == request_id
                    ),
                    key=lambda row: batch.draft_depths[row],
                )
            )
            for request_id in ids
        }
        jobs = []
        for request_id, target in zip(ids, targets, strict=True):
            root = root_rows[request_id]
            candidates = candidate_rows[request_id]
            jobs.append(
                {
                    "session": target,
                    "request_id": request_id,
                    "resident_slot": int(
                        plan.resident_slots[plan.request_ids.index(request_id)]
                    ),
                    "transaction_id": transaction_id,
                    "bulk_attention_mode": "bulk",
                    "use_wmma_prefill": False,
                    "capture_linear_state_rows": True,
                    "defer_linear_state_commit": True,
                    "defer_state_scatter": True,
                    "input_token_ids": (
                        int(batch.tokens[root]),
                        *tuple(int(batch.tokens[row]) for row in candidates),
                    ),
                }
            )
        owner = self.owner._packed_execution_owner(targets[0])
        verify_batch = getattr(owner, "verify_target_blocks_batch", None)
        if not callable(verify_batch):
            raise RuntimeError("MoE physical target owner has no packed verifier")
        target_started = time.perf_counter()
        with moe_physical_c2_numerics_session(True):
            results = list(verify_batch(jobs, device_result=False))
        target_seconds = time.perf_counter() - target_started
        if len(results) != 2:
            raise RuntimeError("MoE physical target returned the wrong request count")
        physical_rows = sum(len(job["input_token_ids"]) for job in jobs)
        for row in rows:
            row.mtp2_target_batch_calls += 1
            row.mtp2_target_physical_rows.append(physical_rows)
        target_top1 = [0] * batch.rows
        for request_id, result in zip(ids, results, strict=True):
            destination = (root_rows[request_id], *candidate_rows[request_id])
            if len(result.token_ids) != len(destination):
                raise RuntimeError("MoE physical target omitted verifier rows")
            for row_index, token in zip(destination, result.token_ids, strict=True):
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
            raise RuntimeError("MoE physical GPU accept differs from CPU oracle")
        accept_seconds = time.perf_counter() - accept_started
        cancelled = tuple(int(value) for value in cancelled_request_ids())
        if cancelled:
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
            raise RuntimeError("MoE physical target owner has no selected commit")
        commit_started = time.perf_counter()
        commit_contract = commit_batch(
            results,
            targets,
            accepted_counts=gpu_summary.accepted_counts,
            accept_buffers=accept_buffers,
        )
        commit_seconds = time.perf_counter() - commit_started
        if int(commit_contract.get("requests", 0)) != 2:
            raise RuntimeError("MoE physical selected commit omitted a request")
        provider_started = time.perf_counter()
        next_tokens = gpu_summary.next_tokens
        visible_rows: list[tuple[int, ...]] = []
        for index, (
            request_id,
            state,
            target,
            result,
            row,
            accepted_count,
            accepted_tokens,
            next_token,
        ) in enumerate(
            zip(
                ids,
                states,
                targets,
                results,
                rows,
                gpu_summary.accepted_counts,
                gpu_summary.accepted_tokens,
                next_tokens,
                strict=True,
            )
        ):
            visible = (
                *tuple(int(token) for token in accepted_tokens),
                int(next_token),
            )
            self._repair_physical_provider(
                state,
                target,
                result,
                accepted_count=int(accepted_count),
                output_ids=visible,
                start_position=starts[index],
            )
            row.slot.generated_ids.extend(visible)
            row.slot.prev_token = int(visible[-1])
            row.slot.seq_position = int(target.position)
            row.slot.native_decode_steps += 1
            row.slot.done = len(row.slot.generated_ids) >= int(row.request.max_tokens)
            row.mtp2_cycles += 1
            row.mtp2_candidate_counts.append(
                int(plan.candidate_counts[plan.request_ids.index(request_id)])
            )
            row.mtp2_accepted_counts.append(int(accepted_count))
            row.mtp2_proposal_ms += float(state.last_proposal_seconds) * 1000.0
            row.mtp2_target_ms += float(target_seconds) * 1000.0
            row.mtp2_accept_ms += float(accept_seconds) * 1000.0
            row.mtp2_selected_commit_ms += float(commit_seconds) * 1000.0
            row.mtp2_device_accept_calls += 1
            row.mtp2_selected_commit_batch_calls += 1
            row.mtp2_execution_routes.append("eager")
            visible_rows.append(visible)
        provider_seconds = time.perf_counter() - provider_started
        for row in rows:
            row.mtp2_provider_update_ms += provider_seconds * 1000.0
        committed_accept = AcceptResult(
            request_ids=ids,
            accepted_counts=gpu_summary.accepted_counts,
            accepted_tokens=gpu_summary.accepted_tokens,
            transaction_id=transaction_id,
            selected_candidate_rows=gpu_summary.commit_rows,
            next_tokens=tuple(int(value) for value in next_tokens),
            correction_or_bonus_tokens=tuple(int(value) for value in next_tokens),
            target_cursor_deltas=tuple(len(values) for values in visible_rows),
            provider_cursor_deltas=gpu_summary.accepted_counts,
            finish_reasons=(None, None),
        )
        telemetry = SpecCycleTelemetry(
            operation_id=plan.operation_id,
            request_ids=plan.request_ids,
            candidate_counts=plan.candidate_counts,
            plan_reasons=plan.reasons,
            k0_classes=plan.k0_classes,
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route="eager",
            proposal_seconds=max(state.last_proposal_seconds for state in states),
            target_seconds=target_seconds,
            accept_commit_seconds=accept_seconds + commit_seconds,
            provider_update_seconds=provider_seconds,
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
                {
                    "gguf_moe_mtp2.result_rows": len(
                        plan.speculative_request_ids
                    )
                },
                lifetime=ClaimLifetime.WORK_ITEM,
            ),
        }

    def rollback_cycle(self, plan: SpecRequestPlan, *args: Any) -> None:
        del args
        for request_id in plan.speculative_request_ids:
            state = self._states.get(int(request_id))
            if state is not None:
                state.proposal_tokens = ()
                state.proposed_cache_len = None
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
        self._active_claims = None
        if self._batch_accept_workspace is not None:
            self._batch_accept_workspace.free()
            self._batch_accept_workspace = None
            self._batch_accept_owner = None
            self._batch_accept_remaining = None
            self._batch_accept_payload = None
            self._batch_accept_library = None
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


def create_qwen35_gguf_moe_mtp2_adapter(
    owner: Any,
    **kwargs: Any,
) -> Any:
    """Select retained C1 or architecture-shaped physical C2 ownership."""

    if int(getattr(owner, "capacity", 1)) == 1:
        return Qwen35GGUFMoEMTP2Adapter(owner, **kwargs)
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter

    adapter = Qwen35GGUFMTP2Adapter(owner, **kwargs)
    # Dense production rowtile arithmetic is independently qualified and must
    # not transfer to the MoE target. Prompt normalization and the packed C2
    # target numerical boundary are qualified in the MoE campaign instead.
    adapter.production_physical_extra_rowtiles = False
    adapter.production_physical_q5_rowtile = False
    adapter.production_physical_q6_rowtile = False
    adapter.moe_physical_c2_numerics = True
    adapter.target_key = "qwen_moe_gguf"
    adapter.provider_key = "qwen_nextn_moe"
    adapter.policy_prefix = "moe-nextn"
    return adapter


__all__ = [
    "Qwen35GGUFMoEMTP2Adapter",
    "create_qwen35_gguf_moe_mtp2_adapter",
]
