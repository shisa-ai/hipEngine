"""Staged dense GGUF NextN/MTP2 adapter for the Generation-2 resident owner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
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
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_rmsnorm_bf16_f32_weight,
)
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
    Qwen35GGUFNextNDeviceProposal,
)
from hipengine.speculative.frontier import (
    CandidateGraph,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecK0Class,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
    SpeculativeCapability,
    TargetFrontier,
    pad_candidate_graph_rows,
    physical_group_pad_rows,
)
from hipengine.speculative.buffers import (
    TargetVerifyBufferOwner,
    TargetVerifyBufferSpec,
)
from hipengine.speculative.interfaces import (
    AcceptResult,
    DraftBatch,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetVerifyBatch,
    TargetVerifyBuffers,
)
from hipengine.speculative.mtp import MtpProposalContext
from hipengine.speculative.ngram_mod import (
    NgramModConfig,
    NgramModProposal,
    RequestLocalNgramMod,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.speculative.serving import SpeculativeMTPStaticEligibility
from hipengine.runtime.workspace import RuntimeWorkspace
from hipengine.core.specdec2_scope import (
    moe_physical_c2_exact_linear_session,
    moe_physical_c2_numerics_session,
    moe_physical_c2_pairreuse_session,
    q4_t16_physical_extra_rowtiles_session,
    q5_t16_physical_rowtile_session,
    q6_t16_physical_rowtile_session,
)
from hipengine.speculative.transaction import (
    SpecCycleResult,
    SpecCycleStage,
    SpecCycleTelemetry,
    SpecCycleTransaction,
)


_NGRAM_MOD_ENV = "HIPENGINE_GGUF_SPECDEC2_NGRAM_MOD"
_NGRAM_MOD_N_MATCH_ENV = "HIPENGINE_GGUF_SPECDEC2_NGRAM_MATCH"
# Physical group accept buffers must hold any padded row multiple the
# backend admits (gfx1100 rows6 multiples up to 24 rows for C4/K3 groups).
_PHYSICAL_ACCEPT_MAX_ROWS = 24
_NGRAM_MOD_N_MIN_ENV = "HIPENGINE_GGUF_SPECDEC2_NGRAM_MIN"
_NGRAM_MOD_PROBE_MAX_ENV = "HIPENGINE_GGUF_SPECDEC2_NGRAM_PROBE_MAX"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        value = int(str(raw).strip())
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _ngram_mod_config_from_env() -> NgramModConfig:
    minimum = _positive_env(_NGRAM_MOD_N_MIN_ENV, 24)
    return NgramModConfig(
        n_match=_positive_env(_NGRAM_MOD_N_MATCH_ENV, 24),
        min_draft_tokens=minimum,
        max_probe_tokens=max(
            minimum,
            _positive_env(_NGRAM_MOD_PROBE_MAX_ENV, 64),
        ),
    )


_FP16_SPECDEC2_PROFILE_SELECTIONS = {
    (
        "gdn_chain_recurrent_rmsnorm_gate",
        "specdec2_mtp2_target_state_rows",
    ): (
        "bf16_c1_exact_state_rows_tloop_fp16state",
        "bf16_c1_exact_state_rows_tloop",
    ),
}


def _fp16_specdec2_profile_authorized(generator: Any) -> bool:
    """Require a complete non-fallback production manifest before FP16 mutation."""

    profile = getattr(generator, "execution_profile", None)
    profile = getattr(profile, "value", profile)
    if str(profile) != "production" or bool(
        getattr(generator, "execution_profile_fell_back_to_strict", True)
    ):
        return False
    if not str(getattr(generator, "execution_profile_manifest_sha256", "") or ""):
        return False
    manifest = getattr(generator, "execution_profile_manifest", None)
    if not isinstance(manifest, Mapping):
        return False
    selections = manifest.get("selections", ())
    if not isinstance(selections, Sequence) or isinstance(selections, (str, bytes)):
        return False
    selected = {}
    for row in selections:
        if not isinstance(row, Mapping):
            return False
        key = (str(row.get("layer", "")), str(row.get("scope", "")))
        selected[key] = (
            str(row.get("selected_variant", "")),
            str(row.get("strict_fallback_variant", "")),
        )
    return all(selected.get(key) == variants for key, variants in _FP16_SPECDEC2_PROFILE_SELECTIONS.items())


def _target_verify_mode_for_context(
    requested: str,
    *,
    backend: str,
    end_position: int,
) -> str:
    selected = str(requested)
    if selected != "native":
        return selected
    native_context_limit = int(
        backend_package_capability(
            str(backend),
            "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
            int(end_position),
        )
    )
    return "native" if int(end_position) <= native_context_limit else "serial_exact"


@dataclass(slots=True)
class _MTP2ProviderGroup:
    key: tuple[int, ...]
    provider: Any
    provider_pool_key: Any | None
    request_ids: set[int]


@dataclass(frozen=True, slots=True)
class _NgramBatchDeviceProposal:
    """Host-selected n-gram IDs staged for the existing device accept path."""

    request_ids: tuple[int, ...]
    root_tokens: tuple[int, ...]
    root_positions: tuple[int, ...]
    candidate_counts: tuple[int, ...]
    token_ids: Tensor
    probed_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        requests = tuple(int(value) for value in self.request_ids)
        counts = tuple(int(value) for value in self.candidate_counts)
        if not requests or len(requests) != len(set(requests)):
            raise ValueError("n-gram proposal request_ids must be non-empty and unique")
        if any(
            len(values) != len(requests)
            for values in (self.root_tokens, self.root_positions, counts, self.probed_tokens)
        ):
            raise ValueError("n-gram proposal metadata must align with requests")
        if any(value <= 0 for value in counts):
            raise ValueError("n-gram proposal candidate counts must be positive")
        if self.token_ids.dtype != DType.INT32 or self.token_ids.shape != (sum(counts),):
            raise ValueError("n-gram proposal token_ids must be packed INT32 candidates")


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
    proposal_device: Qwen35GGUFNextNDeviceProposal | None = None
    proposal_device_batch: Qwen35GGUFNextNBatchDeviceProposal | None = None
    proposal_ngram: _NgramBatchDeviceProposal | None = None
    ngram_candidate_tokens: tuple[int, ...] = ()
    proposal_source: str = "mtp2"
    device_chain_prepare_error: str | None = None


@dataclass(frozen=True, slots=True)
class _PhysicalAcceptPending:
    batch: TargetVerifyBatch
    buffers: TargetVerifyBuffers
    payload: Tensor
    request_count: int
    output_stride: int
    upload_seconds: float = 0.0
    tail_seconds: float = 0.0


class _PhysicalTargetCommitError(RuntimeError):
    """Target state may be committed; AR fallback requires canonical rebuild."""


def _device_chain_oracle_trace_rows(
    batch: TargetVerifyBatch,
    target_top1: Sequence[int],
    summary: TargetAcceptSummary,
    *,
    cycle_id: int,
) -> tuple[dict[str, Any], ...]:
    """Render bounded per-request proposal/target evidence after device accept."""

    top1 = tuple(int(value) for value in target_top1)
    if len(top1) != batch.rows:
        raise ValueError("device-chain oracle top-1 rows do not match target batch")
    if summary.request_ids != batch.request_ids:
        raise ValueError("device-chain oracle summary request IDs do not match target batch")
    traces: list[dict[str, Any]] = []
    for index, request_id in enumerate(batch.request_ids):
        candidate_rows = tuple(
            sorted(
                (
                    row
                    for row in batch.candidate_rows
                    if int(batch.row_to_request[row]) == int(request_id)
                ),
                key=lambda row: int(batch.draft_depths[row]),
            )
        )
        root_row = int(batch.root_rows[index])
        traces.append(
            {
                "cycle_id": int(cycle_id),
                "request_id": int(request_id),
                "root_token": int(batch.tokens[root_row]),
                "root_position": int(batch.positions[root_row]),
                "candidate_tokens": [int(batch.tokens[row]) for row in candidate_rows],
                "target_top1": [int(top1[row]) for row in (root_row, *candidate_rows)],
                "accepted_count": int(summary.accepted_counts[index]),
                "accepted_tokens": [
                    int(token) for token in summary.accepted_tokens[index]
                ],
                "next_token": (
                    None
                    if summary.next_tokens is None
                    else summary.next_tokens[index]
                ),
            }
        )
    return tuple(traces)


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
        ngram_enabled: bool | None = None,
        ngram_config: NgramModConfig | None = None,
    ) -> None:
        self.owner = owner
        self.generator = owner.generator
        self.enabled = bool(enabled)
        self.target_verify_mode = str(target_verify_mode)
        self.candidate_budget = int(candidate_budget)
        self.quant = str(quant)
        self.target_key = "qwen_dense_gguf"
        self.provider_key = "qwen_nextn_dense"
        self.policy_prefix = "dense-nextn"
        profile = getattr(self.generator, "execution_profile", None)
        profile = getattr(profile, "value", profile)
        self.physical_prompt_streaming = bool(
            str(profile) == "production"
            and backend_package_capability(
                str(self.generator.backend),
                "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_PROMPT_STREAMING",
                False,
            )
        )
        self.production_physical_extra_rowtiles = bool(
            str(profile) == "production"
            and backend_package_capability(
                str(self.generator.backend),
                "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXTRA_ROWTILE_SHAPES",
                (),
            )
        )
        self.production_physical_q5_rowtile = bool(
            str(profile) == "production"
            and backend_package_capability(
                str(self.generator.backend),
                "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q5_ROWTILE_ROWS",
                (),
            )
        )
        self.production_physical_q6_rowtile = bool(
            str(profile) == "production"
            and backend_package_capability(
                str(self.generator.backend),
                "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS",
                (),
            )
        )
        self.production_target_pad_row_counts = (
            tuple(
                int(value)
                for value in backend_package_capability(
                    str(self.generator.backend),
                    "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS",
                    (),
                )
            )
            if str(profile) == "production"
            else ()
        )
        self._target_pad_token_scratch: DeviceBuffer | None = None
        self._target_pad_token_capacity = 0
        use_ngram = (
            _env_enabled(_NGRAM_MOD_ENV)
            if ngram_enabled is None
            else bool(ngram_enabled)
        )
        self._ngram = (
            RequestLocalNgramMod(ngram_config or _ngram_mod_config_from_env())
            if use_ngram
            else None
        )
        self.device_chain_qualification_oracle = os.environ.get(
            "HIPENGINE_SPECDEC2_DEVICE_CHAIN_ORACLE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.post_reject_cooldown_enabled = _env_enabled(
            "HIPENGINE_SPECDEC2_POST_REJECT_COOLDOWN"
        )
        self._post_reject_pending: set[int] = set()
        if self.candidate_budget not in {1, 2, 3}:
            raise ValueError("MTP2 candidate budget must be 1, 2, or 3")
        self._intents: dict[int, int] = {}
        self._static_eligibility_by_request: dict[
            int, SpeculativeMTPStaticEligibility
        ] = {}
        self._prompt_hidden_rows: dict[int, np.ndarray] = {}
        self._prompt_streaming_sinks: dict[int, _StreamingNextNPromptSink] = {}
        self._prompt_streaming_group_keys: dict[int, tuple[int, ...]] = {}
        self._prompt_streaming_norm_buffers: dict[int, DeviceBuffer] = {}
        self._states: dict[int, _MTP2RequestState] = {}
        self._provider_groups: dict[tuple[int, ...], _MTP2ProviderGroup] = {}
        self._disabled_requests: set[int] = set()
        self._active_claims: ResourceClaimSet | None = None
        self._active_prompt_claims: ResourceClaimSet | None = None
        self._transaction_sequence = 0
        self._batch_accept_workspace: RuntimeWorkspace | None = None
        self._batch_accept_owner: TargetVerifyBufferOwner | None = None
        self._batch_accept_remaining: Tensor | None = None
        self._batch_accept_payload: Tensor | None = None
        self._batch_accept_library: Any | None = None
        self._cycle_workspace: RuntimeWorkspace | None = None
        self._cycle_proposal_hidden: Tensor | None = None
        self._cycle_repair_hidden: Tensor | None = None
        self._cycle_ngram_tokens: Tensor | None = None
        self._cycle_workspace_shape: tuple[int, int] | None = None

    def _target_profile_supported(self, target: Any) -> bool:
        runner = getattr(target, "runner", None)
        return bool(
            not bool(getattr(runner, "fp16_recurrent_state", False))
            or _fp16_specdec2_profile_authorized(self.generator)
        )

    def _bind_target_profile_metadata(self, target: Any) -> None:
        target._specdec2_execution_profile_manifest_sha256 = str(
            getattr(self.generator, "execution_profile_manifest_sha256", "legacy")
            or "legacy"
        )
        runner = getattr(target, "runner", None)
        target._specdec2_recurrent_state_dtype = (
            "fp16" if bool(getattr(runner, "fp16_recurrent_state", False)) else "fp32"
        )

    @staticmethod
    def _target_graph_supported(target: Any) -> bool:
        return not bool(
            getattr(getattr(target, "runner", None), "fp16_recurrent_state", False)
        )

    def _static_eligibility(
        self,
        request_id: int,
    ) -> SpeculativeMTPStaticEligibility | None:
        return getattr(self, "_static_eligibility_by_request", {}).get(
            int(request_id)
        )

    def _singleton_only(self, request_id: int) -> bool:
        eligibility = self._static_eligibility(request_id)
        return bool(
            eligibility is not None
            and eligibility.eligible
            and int(eligibility.max_realized_group_rows) == 1
            and int(getattr(self.owner, "capacity", 1)) > 1
        )

    def register_request(
        self,
        request_id: int,
        candidate_budget: int,
        *,
        static_eligibility: SpeculativeMTPStaticEligibility | None = None,
    ) -> None:
        rid = int(request_id)
        budget = min(self.candidate_budget, max(1, int(candidate_budget)))
        self._intents[rid] = budget
        eligibility_by_request = getattr(
            self,
            "_static_eligibility_by_request",
            None,
        )
        if eligibility_by_request is None:
            eligibility_by_request = {}
            self._static_eligibility_by_request = eligibility_by_request
        if static_eligibility is None:
            eligibility_by_request.pop(rid, None)
        else:
            if not isinstance(static_eligibility, SpeculativeMTPStaticEligibility):
                raise TypeError(
                    "static_eligibility must be SpeculativeMTPStaticEligibility"
                )
            if not static_eligibility.eligible:
                raise ValueError("permanent-AR eligibility cannot register a provider")
            eligibility_by_request[rid] = static_eligibility
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
        automatic_singleton = bool(
            len(ids) == 1 and self._singleton_only(ids[0])
        )
        if (
            int(getattr(self.owner, "capacity", 1)) > 1
            and not automatic_singleton
            and not bool(self.physical_prompt_streaming)
        ):
            for row in rows:
                row.mtp2_prompt_fallback_reason = "physical_streaming_category_rejected"
            return None
        if any(row.lease is None or int(row.prefix_reused_tokens) > 0 for row in rows):
            for row in rows:
                if int(getattr(row, "prefix_reused_tokens", 0)) > 0:
                    row.mtp2_prompt_fallback_reason = "prefix_reuse_k0"
            return None
        targets = tuple(row.lease.session for row in rows)
        if any(not self._target_profile_supported(target) for target in targets):
            for row in rows:
                row.mtp2_prompt_fallback_reason = "target_profile_k0"
            return None
        context_misses = tuple(
            (row, target)
            for row, target in zip(rows, targets, strict=True)
            if len(row.prompt_ids) + 1
            >= min(1023, int(target.target_layout.max_sequence_length))
        )
        if context_misses:
            for row, _target in context_misses:
                row.mtp2_candidate_budget = 0
                row.mtp2_prompt_fallback_reason = "target_context_k0"
            return None
        if self._active_prompt_claims is not None:
            raise RuntimeError("GGUF MTP2 prompt activation claims are already reserved")
        self._active_prompt_claims = ResourceClaimSet.from_mapping(
            "gguf-mtp2-prompt:" + ",".join(str(request_id) for request_id in ids),
            {
                "gguf_mtp2.prompt_rows": sum(len(row.prompt_ids) for row in rows),
                "gguf_mtp2.carried_hidden_rows": len(rows),
                "gguf_mtp2.provider_request_slots": len(rows),
            },
            lifetime=ClaimLifetime.WORK_ITEM,
        )
        missing = len(ids)
        group = (
            None
            if automatic_singleton
            else next(
                (
                    candidate
                    for candidate in self._provider_groups.values()
                    if len(candidate.request_ids) + missing
                    <= int(candidate.provider.executor.max_requests)
                ),
                None,
            )
        )
        acquired = group is None
        if group is None:
            max_positions = min(
                int(target.target_layout.max_sequence_length) for target in targets
            )
            provider_capacity = (
                1
                if automatic_singleton
                else max(
                    len(ids),
                    min(4, int(getattr(self.owner, "capacity", len(ids)))),
                )
            )
            provider, pool_key, _reused = self.generator._acquire_dense_mtp_draft_provider(
                targets[0],
                max_positions=max_positions,
                pool_enabled=self.owner._shared_runner is not None,
                max_requests=provider_capacity,
            )
            if not callable(getattr(provider.executor, "enqueue_prompt_rows", None)):
                self.generator._release_mtp_draft_runner(pool_key, provider)
                self._active_prompt_claims = None
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
        if not hasattr(self, "_prompt_streaming_norm_buffers"):
            self._prompt_streaming_norm_buffers = {}
        checkpoint_by_id = {} if checkpoints is None else dict(checkpoints)
        try:
            for request_id, row, target in zip(ids, rows, targets, strict=True):
                group.provider.reset_request(request_id)
                checkpoint = checkpoint_by_id.get(request_id)
                if checkpoint is None:
                    checkpoint = lambda row=row: raise_if_generation_deadline_expired(
                        row.request
                    )
                hidden_size = int(group.provider.executor.hidden_size)
                hidden_nbytes = hidden_size * DType.BF16.itemsize
                target_hidden = getattr(target, "_prefill_hidden_a", None)
                if target_hidden is None or target.runner is None:
                    raise RuntimeError("GGUF target prefill hidden storage is closed")
                capacity = min(
                    len(row.prompt_ids),
                    int(target_hidden.nbytes) // hidden_nbytes,
                )
                if capacity <= 0:
                    raise RuntimeError(
                        "GGUF target prefill hidden storage has no row capacity"
                    )
                norm_buffer = malloc(
                    capacity * hidden_nbytes,
                    runtime=target.runtime,
                )
                self._prompt_streaming_norm_buffers[request_id] = norm_buffer

                def transform_hidden_rows(
                    src_ptr: int,
                    count: int,
                    stream: int,
                    *,
                    target=target,
                    norm_buffer=norm_buffer,
                    capacity=capacity,
                ) -> int:
                    if int(count) > int(capacity):
                        raise RuntimeError(
                            "streaming normalized hidden rows exceed capacity"
                        )
                    gguf_rmsnorm_bf16_f32_weight(
                        int(src_ptr),
                        target.runner.weights.root("output_norm").allocation().tensor.ptr,
                        norm_buffer.ptr,
                        rows=int(count),
                        hidden_size=int(target.runner.hidden_size),
                        eps=target.runner.weights.config.rms_norm_eps,
                        stream=int(stream),
                        runtime=target.runtime,
                    )
                    return int(norm_buffer.ptr)

                sink = _StreamingNextNPromptSink(
                    request_id=request_id,
                    prompt_tokens=row.prompt_ids,
                    hidden_size=hidden_size,
                    executor=group.provider.executor,
                    runtime=target.runtime,
                    checkpoint=checkpoint,
                    transform_hidden_rows=transform_hidden_rows,
                )
                self._prompt_streaming_sinks[request_id] = sink
                self._prompt_streaming_group_keys[request_id] = group.key
                group.request_ids.add(request_id)
                created.append(request_id)
        except Exception:
            self._abort_prompt_streaming(tuple(created), stream=0)
            for request_id in ids:
                self._free_prompt_streaming_norm_buffer(request_id, target=None)
            if acquired and not group.request_ids:
                self._provider_groups.pop(group.key, None)
            self._active_prompt_claims = None
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
                        target_verify_mode=_target_verify_mode_for_context(
                            self.target_verify_mode,
                            backend=self.generator.backend,
                            end_position=(
                                len(row.prompt_ids) + self.candidate_budget + 1
                            ),
                        ),
                    )
                    if len(ids) == 1
                    and (
                        int(getattr(self.owner, "capacity", 1)) == 1
                        or self._singleton_only(request_id)
                    )
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
                self._free_prompt_streaming_norm_buffer(request_id, target=None)
            self._active_prompt_claims = None

    def _free_prompt_streaming_norm_buffer(
        self,
        request_id: int,
        *,
        target: Any | None,
    ) -> None:
        rid = int(request_id)
        norm_buffers = getattr(self, "_prompt_streaming_norm_buffers", None)
        if not norm_buffers:
            return
        buffer = norm_buffers.pop(rid, None)
        if buffer is None:
            return
        if target is None:
            try:
                row = self.owner._row(rid)
                target = None if row.lease is None else row.lease.session
            except (KeyError, AttributeError):
                target = None
        if target is None:
            raise RuntimeError(
                "streaming hidden normalization buffer lost its target owner"
            )
        free(buffer, runtime=target.runtime)

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
            self._free_prompt_streaming_norm_buffer(request_id, target=None)
        for group_key in groups:
            group = self._provider_groups.get(group_key)
            if group is not None and not group.request_ids:
                self._provider_groups.pop(group_key, None)
                self.generator._release_mtp_draft_runner(
                    group.provider_pool_key,
                    group.provider,
                )
        self._active_prompt_claims = None

    def observe_prefill_result(self, request_id: int, prompt_ids: Sequence[int], result: Any) -> None:
        rid = int(request_id)
        if rid not in self._intents:
            return
        if int(getattr(self.owner, "capacity", 1)) > 1:
            row = self.owner._row(rid)
            target = None if row.lease is None else row.lease.session
            prepare_device_commit = getattr(
                target,
                "prepare_external_verify_state_commit",
                None,
            )
            if callable(prepare_device_commit):
                prepare_device_commit()
        if int(getattr(self.owner, "capacity", 1)) == 1:
            state = self._states.get(rid)
            row = self.owner._row(rid)
            target = None if row.lease is None else row.lease.session
            token_id = getattr(result, "token_id", None)
            budget = int(self._intents[rid])
            if (
                state is not None
                and state.verifier is not None
                and target is not None
                and token_id is not None
            ):
                try:
                    prepare_proposal = getattr(
                        state.provider,
                        "prepare_device_proposal",
                        None,
                    )
                    prepare_target = getattr(
                        target,
                        "prepare_native_spec_target_graph",
                        None,
                    )
                    if callable(prepare_proposal):
                        prepare_proposal(rid, candidate_budget=budget)
                    if callable(prepare_target):
                        prepare_target(
                            (int(token_id), *((0,) * budget)),
                            request_id=rid,
                        )
                    for bucket_budget in range(1, budget + 1):
                        draft = DraftBatch(
                            request_ids=(rid,),
                            candidate_tokens=(0,) * bucket_budget,
                            parent_positions=tuple(
                                int(target.position) + depth - 1
                                for depth in range(1, bucket_budget + 1)
                            ),
                            draft_depths=tuple(
                                range(1, bucket_budget + 1)
                            ),
                            row_to_request=(rid,) * bucket_budget,
                            mode="verify_chain",
                        )
                        verify_batch = TargetVerifyBatch.from_draft(
                            draft,
                            root_tokens=(int(token_id),),
                            root_positions=(int(target.position),),
                        )
                        for execution_route in ("graph", "eager"):
                            state.verifier.graph_bucket(
                                (
                                    "specdec2",
                                    "verify_chain",
                                    bucket_budget + 1,
                                    execution_route,
                                ),
                                verify_batch,
                            )
                    state.device_chain_prepare_error = None
                except Exception as exc:
                    state.device_chain_prepare_error = (
                        f"{type(exc).__name__}:{exc}"
                    )
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
        static_eligibilities = tuple(
            self._static_eligibility(item.request_id)
            for item in semantics
        )
        static_bounds = tuple(
            int(eligibility.max_realized_group_rows)
            for eligibility in static_eligibilities
            if eligibility is not None
        )
        if static_bounds and len(semantics) > min(static_bounds):
            return None
        static_candidate_bounds = tuple(
            int(eligibility.max_candidate_count)
            for eligibility in static_eligibilities
            if eligibility is not None
        )
        max_candidate_count = min(
            self.candidate_budget,
            *(static_candidate_bounds or (self.candidate_budget,)),
        )
        singleton_only = tuple(
            self._singleton_only(item.request_id)
            for item in semantics
        )
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
            if not self._target_profile_supported(target):
                return None
            targets.append(target)
        if len(semantics) == 1:
            existing = self._states.get(int(semantics[0].request_id))
            owner = getattr(targets[0], "_target_scratch_owner", None)
            automatic_singleton = bool(singleton_only[0])
            eligibility = static_eligibilities[0]
            physical_singleton = bool(
                eligibility is not None
                and eligibility.eligible
                and int(eligibility.max_realized_group_rows) > 1
            )
            if (
                existing is not None
                and existing.verifier is None
                and not physical_singleton
            ) or (
                not automatic_singleton
                and not physical_singleton
                and (
                    int(getattr(owner, "slot_count", 1)) > 1
                    or int(getattr(self.owner, "capacity", 1)) > 1
                )
            ):
                return None
        # Streaming activation is retained only through the already-qualified
        # short target context. Longer requests stay K0 until an exact shifted-
        # page eager target owner is qualified independently of graph capture.
        max_context = min(
            1023,
            *(int(target.target_layout.max_sequence_length) for target in targets),
        )
        realized_verify_modes = {
            str(state.verifier.target_verify_mode)
            for item in semantics
            if (state := self._states.get(int(item.request_id))) is not None
            and state.verifier is not None
        }
        if len(realized_verify_modes) > 1:
            return None
        realized_verify_mode = (
            next(iter(realized_verify_modes))
            if realized_verify_modes
            else self.target_verify_mode
        )
        profile = str(getattr(self.generator, "execution_profile", None) or "legacy_exact")
        max_requests = min(4, max(1, int(getattr(self.owner, "capacity", 4))))
        max_frontier_rows = max_requests * (max_candidate_count + 1)
        return SpeculativeCapability(
            capability_key=(
                f"gguf_mtp2_c{max_requests}:{self.generator.backend}:{self.quant}:"
                f"{realized_verify_mode}:{max_candidate_count}"
            ),
            target_key=str(getattr(self, "target_key", "qwen_dense_gguf")),
            provider_key=str(
                getattr(self, "provider_key", "qwen_nextn_dense")
            ),
            method_key="mtp2",
            policy_fingerprint=(
                f"{getattr(self, 'policy_prefix', 'dense-nextn')}:"
                f"{realized_verify_mode}:b{max_candidate_count}:"
                "prompt-streaming"
                f"{int(getattr(self, 'physical_prompt_streaming', False))}:"
                "extra-rowtiles"
                f"{int(getattr(self, 'production_physical_extra_rowtiles', False))}:"
                "q5-rowtile"
                f"{int(getattr(self, 'production_physical_q5_rowtile', False))}:"
                "q6-rowtile"
                f"{int(getattr(self, 'production_physical_q6_rowtile', False))}"
            ),
            execution_profile=profile,
            kv_backend_key=str(getattr(target, "kv_storage_dtype", "bf16")),
            attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
            supported_modes=("verify_chain",),
            supported_sampling_modes=("greedy",),
            max_requests=max_requests,
            max_candidates_per_request=max_candidate_count,
            max_frontier_rows=max_frontier_rows,
            proposal_widths=tuple(
                width for width in (1, 2, 4) if width <= max_requests
            ),
            target_row_buckets=tuple(range(2, max_frontier_rows + 1)),
            target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            graph_supported=realized_verify_mode == "native",
            eager_supported=True,
            strict_fallback_key="gguf_target_ar",
            max_context_tokens=max_context,
        )

    def partition_max_requests(self, request_ids: Sequence[int]) -> int:
        """Return a physical subgroup bound without selecting a future due C/K."""

        ids = tuple(int(value) for value in request_ids)
        if not self.enabled or not ids:
            return 0
        eligibility = tuple(self._static_eligibility(request_id) for request_id in ids)
        if any(row is None or not row.eligible for row in eligibility):
            return 0
        bound = min(
            4,
            max(1, int(getattr(self.owner, "capacity", 1))),
            *(int(row.max_realized_group_rows) for row in eligibility if row is not None),
        )
        # Exact automatic-singleton evidence must fail the composed due group to
        # K0; it may not be reinterpreted as many independently profitable C1s.
        return bound if bound > 1 else 0

    def claims_fit(self, plan: SpecRequestPlan) -> bool:
        request_ids = tuple(int(value) for value in plan.speculative_request_ids)
        physical_singleton = bool(
            len(request_ids) == 1
            and (eligibility := self._static_eligibility(request_ids[0])) is not None
            and eligibility.eligible
            and int(eligibility.max_realized_group_rows) > 1
        )
        return bool(
            self.enabled
            and self._active_claims is None
            # The staged target result owns every due request in the plan. A
            # mixed positive-K/K0 group needs a separately declared partition;
            # this adapter must fail it closed before provider mutation.
            and tuple(int(value) for value in plan.request_ids) == request_ids
            and 1 <= len(request_ids) <= 4
            and not (
                len(request_ids) == 1
                and int(getattr(self.owner, "capacity", 1)) > 1
                and not self._singleton_only(request_ids[0])
                and not physical_singleton
            )
            and not any(
                request_id in self._disabled_requests
                for request_id in plan.speculative_request_ids
            )
        )

    def post_reject_cooldown(self, request_ids: Sequence[int]) -> tuple[bool, ...]:
        """Report a one-shot catchup need for each request.

        A request is suppressed exactly once per fully rejected physical
        cycle: the pending flag is set at reject commit and cleared when the
        K0-transitional catchup advance repairs the provider state.
        """

        if not self.post_reject_cooldown_enabled:
            return tuple(False for _ in request_ids)
        return tuple(int(request_id) in self._post_reject_pending for request_id in request_ids)

    def prepare_k0(
        self,
        plan: SpecRequestPlan,
        request_semantics: Sequence[SpeculativeRequestSemantics],
        *,
        stream: int | None = None,
    ) -> None:
        del request_semantics, stream
        reason_by_id = dict(zip(plan.request_ids, plan.reasons, strict=True))
        k0_by_id = dict(
            zip(
                plan.request_ids,
                getattr(
                    plan,
                    "k0_classes",
                    tuple(
                        SpecK0Class.TRANSITIONAL
                        if reason in {
                            SpecPlanReason.NO_PROVIDER,
                            SpecPlanReason.POLICY_SELECTED_AR,
                        }
                        else SpecK0Class.PURE
                        for reason in plan.reasons
                    ),
                ),
                strict=True,
            )
        )
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
            and self.owner._row(request_id).lease is not None
            and self._target_profile_supported(
                self.owner._row(request_id).lease.session
            )
        )
        if attach:
            for request_id in attach:
                self.owner._flush_row_owner(self.owner._row(request_id))
            self._ensure_request_states(attach)
        for rid in ids:
            if k0_by_id[rid] is not SpecK0Class.TRANSITIONAL:
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
            self._post_reject_pending.discard(int(rid))

    def _cycle_hidden_tensors(
        self,
        runtime: Any,
        *,
        hidden_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Return stable max-width proposal and repair BF16 slabs."""

        shape = (
            min(4, max(1, int(getattr(self.owner, "capacity", 1)))),
            int(hidden_size),
        )
        if shape[1] <= 0:
            raise ValueError("cycle hidden workspace requires positive hidden_size")
        if getattr(self, "_cycle_workspace", None) is None:
            workspace = RuntimeWorkspace(
                device=Device("hip", 0),
                runtime=runtime,
            )
            proposal = workspace.reserve_tensor(
                "gguf_mtp2/cycle/proposal_hidden",
                shape,
                DType.BF16,
            )
            repair = workspace.reserve_tensor(
                "gguf_mtp2/cycle/repair_hidden",
                shape,
                DType.BF16,
            )
            ngram_tokens = (
                workspace.reserve_tensor(
                    "gguf_mtp2/cycle/ngram_tokens",
                    (
                        min(4, max(1, int(getattr(self.owner, "capacity", 1))))
                        * self.candidate_budget,
                    ),
                    DType.INT32,
                )
                if getattr(self, "_ngram", None) is not None
                else None
            )
            self._cycle_workspace = workspace
            self._cycle_proposal_hidden = proposal
            self._cycle_repair_hidden = repair
            self._cycle_ngram_tokens = ngram_tokens
            self._cycle_workspace_shape = shape
        elif getattr(self, "_cycle_workspace_shape", None) != shape:
            raise RuntimeError(
                "GGUF MTP2 cycle workspace shape changed after allocation"
            )
        if (
            self._cycle_proposal_hidden is None
            or self._cycle_repair_hidden is None
        ):
            raise RuntimeError("GGUF MTP2 cycle hidden workspace is incomplete")
        return self._cycle_proposal_hidden, self._cycle_repair_hidden

    def _stage_ngram_tokens(
        self,
        tokens: Sequence[int],
        *,
        runtime: Any,
    ) -> Tensor:
        values = np.ascontiguousarray(tuple(int(token) for token in tokens), dtype=np.int32)
        if values.size <= 0 or np.any(values < 0):
            raise ValueError("n-gram candidate tokens must be non-empty and non-negative")
        workspace = getattr(self, "_cycle_ngram_tokens", None)
        if workspace is None or values.size > workspace.numel:
            raise RuntimeError("n-gram candidate tokens exceed the fixed cycle workspace")
        copy_host_to_device(
            DeviceBuffer(workspace.ptr, values.nbytes),
            host_array_ptr(values),
            values.nbytes,
            runtime=runtime,
        )
        return Tensor.from_handle(
            workspace.ptr,
            (int(values.size),),
            DType.INT32,
            workspace.device,
        )

    def _close_cycle_workspace(self) -> None:
        workspace = getattr(self, "_cycle_workspace", None)
        if workspace is not None:
            workspace.free()
        self._cycle_workspace = None
        self._cycle_proposal_hidden = None
        self._cycle_repair_hidden = None
        self._cycle_ngram_tokens = None
        self._cycle_workspace_shape = None

    def cycle_workspace_contract(self) -> dict[str, Any]:
        return {
            "allocated": self._cycle_workspace is not None,
            "shape": (
                None
                if self._cycle_workspace_shape is None
                else list(self._cycle_workspace_shape)
            ),
            "proposal_ptr": (
                0
                if self._cycle_proposal_hidden is None
                else int(self._cycle_proposal_hidden.ptr)
            ),
            "repair_ptr": (
                0
                if self._cycle_repair_hidden is None
                else int(self._cycle_repair_hidden.ptr)
            ),
            "ngram_tokens_ptr": (
                0
                if getattr(self, "_cycle_ngram_tokens", None) is None
                else int(self._cycle_ngram_tokens.ptr)
            ),
        }

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
                {
                    "gguf_mtp2.result_rows": requests,
                    "gguf_mtp2.cycle_hidden_rows": 2
                    * min(4, max(1, int(getattr(self.owner, "capacity", 1)))),
                },
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
        if len(missing) == 1 and self._singleton_only(missing[0]):
            self._states[missing[0]] = self._open_request(missing[0])
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
        reusable_group = next(
            (
                group
                for group in self._provider_groups.values()
                if len(group.request_ids) + len(missing)
                <= int(group.provider.executor.max_requests)
            ),
            None,
        )
        if reusable_group is not None:
            for request_id in missing:
                self._states[request_id] = self._attach_request_to_group(
                    request_id,
                    reusable_group,
                )
            return
        if len(ids) == 1 and int(getattr(self.owner, "capacity", 1)) == 1:
            self._states[ids[0]] = self._open_request(ids[0])
        else:
            self._open_batch_requests(ids)

    @staticmethod
    def _add_row_counter(row: Any, name: str, amount: int = 1) -> None:
        setattr(row, name, int(getattr(row, name, 0)) + int(amount))

    def _try_ngram_proposal(
        self,
        ids: tuple[int, ...],
        rows: tuple[Any, ...],
        budgets: tuple[int, ...],
        context: MtpProposalContext,
    ) -> tuple[_NgramBatchDeviceProposal, tuple[tuple[int, ...], ...]] | None:
        composer = getattr(self, "_ngram", None)
        if composer is None:
            return None
        proposals: list[NgramModProposal | None] = []
        for request_id, row, budget, root_token in zip(
            ids,
            rows,
            budgets,
            context.root_tokens,
            strict=True,
        ):
            self._add_row_counter(row, "mtp2_ngram_lookup_calls")
            prompt = tuple(int(token) for token in getattr(row, "prompt_ids", ()))
            generated = tuple(
                int(token)
                for token in getattr(getattr(row, "slot", None), "generated_ids", ())
            )
            history = (*prompt, *generated)
            if not history or history[-1] != int(root_token):
                proposals.append(None)
                continue
            proposal = composer.propose(
                request_id,
                history,
                max_candidates=int(budget),
            )
            if proposal is not None:
                self._add_row_counter(row, "mtp2_ngram_lookup_hits")
            proposals.append(proposal)
        # Preserve one physical provider source per fairness group. A mixed
        # hit/miss group falls back wholesale to the ordinary batched MTP path;
        # it never materializes MTP device candidates just to splice host rows.
        if any(proposal is None for proposal in proposals):
            return None
        selected = tuple(proposal for proposal in proposals if proposal is not None)
        candidate_rows = tuple(proposal.candidate_tokens for proposal in selected)
        flat_tokens = tuple(token for values in candidate_rows for token in values)
        token_ids = self._stage_ngram_tokens(
            flat_tokens,
            runtime=rows[0].lease.session.runtime,
        )
        descriptor = _NgramBatchDeviceProposal(
            request_ids=ids,
            root_tokens=tuple(int(token) for token in context.root_tokens),
            root_positions=tuple(int(position) for position in context.root_positions),
            candidate_counts=budgets,
            token_ids=token_ids,
            probed_tokens=tuple(int(proposal.probed_tokens) for proposal in selected),
        )
        for row, proposal in zip(rows, selected, strict=True):
            self._add_row_counter(row, "mtp2_ngram_cycles")
            self._add_row_counter(
                row,
                "mtp2_ngram_probed_tokens",
                int(proposal.probed_tokens),
            )
            self._add_row_counter(row, "mtp2_candidate_device_handoffs")
        return descriptor, candidate_rows

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
        hidden_batch, _repair_hidden = self._cycle_hidden_tensors(
            targets[0].runtime,
            hidden_size=hidden_size,
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
            device_proposal = None
            ngram_proposal = None
            ngram_candidate_rows: tuple[tuple[int, ...], ...] = ()
            ngram_selection = self._try_ngram_proposal(
                ids,
                rows,
                budgets,
                context,
            )
            if ngram_selection is not None:
                ngram_proposal, ngram_candidate_rows = ngram_selection
                draft = None
            elif len(ids) == 1 and states[0].verifier is not None:
                verifier = states[0].verifier
                remaining_by_id = {
                    int(item.request_id): int(item.remaining_decode)
                    for item in request_semantics
                }
                device_ready = getattr(verifier, "device_proposal_ready", None)
                launch_device = getattr(
                    states[0].provider, "launch_device_proposal", None
                )
                target_device_ready = bool(
                    self._target_graph_supported(targets[0])
                    and callable(device_ready)
                    and device_ready(
                        budgets[0],
                        remaining_decode=remaining_by_id[ids[0]],
                    )
                )
                if target_device_ready and callable(launch_device):
                    device_proposal = launch_device(
                        context,
                        candidate_budget=budgets[0],
                    )
                if device_proposal is None:
                    draft = states[0].provider.propose(
                        context,
                        candidate_budget=budgets[0],
                        return_logits=False,
                        allow_graph=target_device_ready,
                    )
                else:
                    draft = None
                    rows[0].mtp2_candidate_device_handoffs += 1
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
                for row in rows:
                    row.mtp2_proposal_batch_calls += len(physical_shapes)
                    row.mtp2_proposal_physical_rows.extend(physical_shapes)
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
                state.proposal_device = device_proposal
                state.proposal_device_batch = device_draft
                state.proposal_ngram = ngram_proposal
                state.ngram_candidate_tokens = (
                    () if ngram_proposal is None else ngram_candidate_rows[index]
                )
                state.proposal_source = (
                    "ngram_mod" if ngram_proposal is not None else "mtp2"
                )
        except Exception:
            for request_id, checkpoint in locals().get("checkpoints", {}).items():
                state = self._states[request_id]
                state.provider.executor.restore_request_checkpoint(checkpoint)
                state.provider.executor.release_request_checkpoint(checkpoint)
            raise
        if ngram_proposal is not None:
            candidate_tokens = tuple(
                token for values in ngram_candidate_rows for token in values
            )
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
            candidate_token_ids = ngram_proposal.token_ids
            draft_mode = "verify_chain"
            draft_metadata = (
                ("candidate_handoff", "host_exact_device_i32"),
                ("candidate_rows", sum(budgets)),
                ("proposal_source", "request_local_ngram_mod"),
                ("ngram_match", int(getattr(self._ngram.config, "n_match", 0))),
            )
            method_key = "ngram_mod+mtp2"
            policy_fingerprint = (
                f"request-local-ngram:n{self._ngram.config.n_match}:"
                f"min{self._ngram.config.min_draft_tokens}:mtp2-catchup"
            )
        elif device_draft is None and device_proposal is None:
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
            method_key = "mtp2"
            policy_fingerprint = (
                f"{getattr(self, 'policy_prefix', 'dense-nextn')}-strict"
            )
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
            if device_draft is not None:
                candidate_token_ids = device_draft.token_ids
                handoff = "device_i32"
            else:
                assert device_proposal is not None
                candidate_token_ids = Tensor.from_handle(
                    device_proposal.result_ptr,
                    (device_proposal.budget,),
                    DType.INT32,
                    device_proposal.final_hidden.device,
                    strides=(2,),
                )
                handoff = "device_graph_i32x2"
            draft_mode = "verify_chain"
            draft_metadata = (
                ("candidate_handoff", handoff),
                ("candidate_rows", sum(budgets)),
            )
            method_key = "mtp2"
            policy_fingerprint = (
                f"{getattr(self, 'policy_prefix', 'dense-nextn')}-strict"
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
            method_key=method_key,
            policy_fingerprint=policy_fingerprint,
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
        physical_provider = any(
            self._states[int(request_id)].verifier is None
            for request_id in plan.speculative_request_ids
        )
        if (len(plan.speculative_request_ids) > 1 or physical_provider) and (
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
        device_proposal = state.proposal_device
        if batch is None:
            graph = frontier.candidate_graph
            if device_proposal is None or graph is None or graph.token_ids is None:
                raise NotImplementedError("GGUF MTP2 requires a host or device chain")
            if (
                graph.token_ids.ptr != device_proposal.result_ptr
                or graph.token_ids.strides != (2,)
            ):
                raise RuntimeError("C1 candidate graph lost its device proposal")
            placeholder = getattr(
                state.provider,
                "placeholder_device_proposal",
                None,
            )
            if not callable(placeholder):
                raise RuntimeError("C1 provider omitted its device placeholder")
            shape_draft = placeholder(device_proposal)
            batch = TargetVerifyBatch.from_draft(
                shape_draft,
                root_tokens=frontier.root_tokens,
                root_positions=frontier.root_positions,
            )
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
                device_proposal=device_proposal,
                qualification_oracle=bool(
                    getattr(
                        self,
                        "device_chain_qualification_oracle",
                        False,
                    )
                ),
                allow_graph=(
                    int(batch.candidate_count) == int(self.candidate_budget)
                ),
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
            if state.proposal_source == "ngram_mod":
                hidden_source = (
                    None
                    if prepared.device_state_commit_buffers is None
                    else prepared.device_state_commit_buffers.hidden_taps_src
                )
                if hidden_source is None:
                    hidden_source = state.verifier.journal.hidden_rows_tensor(
                        batch.rows
                    )
                self._repair_provider_states_from_ngram_target_rows(
                    (state,),
                    (hidden_source,),
                    accepted_counts=(accepted,),
                )
            elif device_proposal is None:
                self._repair_provider_state(
                    state,
                    accepted_count=accepted,
                    candidate_count=int(plan.candidate_counts[0]),
                )
            else:
                self._repair_provider_state_device(
                    state,
                    device_proposal,
                    accepted_count=accepted,
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
            if prepared.native_device_accept_commit:
                row.mtp2_device_accept_calls += 1
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
                k0_classes=plan.k0_classes,
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

    def _target_group_pad_rows(self, *, request_count: int, candidate_rows: int) -> int:
        """Return inactive pad rows lifting a physical group to admitted multiples."""

        return physical_group_pad_rows(
            self.production_target_pad_row_counts,
            request_count,
            candidate_rows,
            _PHYSICAL_ACCEPT_MAX_ROWS,
        )

    def _target_pad_token_tensor(
        self,
        proposal: Qwen35GGUFNextNBatchDeviceProposal,
        *,
        pad_rows: int,
        runtime: Any,
    ) -> Tensor:
        """Materialize ``[proposal tokens | pad tokens]`` in adapter scratch."""

        real = int(proposal.token_ids.numel)
        total = real + int(pad_rows)
        nbytes = total * DType.INT32.itemsize
        if (
            self._target_pad_token_scratch is None
            or self._target_pad_token_capacity < nbytes
        ):
            capacity = max(nbytes, 64)
            if self._target_pad_token_scratch is not None:
                free(self._target_pad_token_scratch)
                self._target_pad_token_scratch = None
            self._target_pad_token_scratch = malloc(capacity, runtime=runtime)
            self._target_pad_token_capacity = capacity
        scratch = self._target_pad_token_scratch
        runtime.memcpy_async(
            scratch.ptr,
            proposal.token_ids.ptr,
            real * DType.INT32.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            0,
        )
        pads = np.zeros((int(pad_rows),), dtype=np.int32)
        copy_host_to_device(
            DeviceBuffer(scratch.ptr + real * DType.INT32.itemsize, pads.nbytes),
            host_array_ptr(pads),
            pads.nbytes,
            runtime=runtime,
        )
        return Tensor.from_handle(
            scratch.ptr,
            (total,),
            DType.INT32,
            proposal.token_ids.device,
        )

    def _batch_accept_resources(
        self,
        runtime: Any,
    ) -> tuple[TargetVerifyBufferOwner, Tensor, Tensor]:
        if self._batch_accept_workspace is None:
            workspace = RuntimeWorkspace(device=Device("hip", 0), runtime=runtime)
            spec = TargetVerifyBufferSpec(
                backend=str(self.generator.backend),
                bucket="gguf-mtp2-physical-r24-c4",
                device=Device("hip", 0),
                max_rows=_PHYSICAL_ACCEPT_MAX_ROWS,
                max_requests=4,
                mode="verify_chain",
            )
            self._batch_accept_workspace = workspace
            self._batch_accept_owner = TargetVerifyBufferOwner.allocate(
                spec,
                workspace=workspace,
            )
            self._batch_accept_remaining = workspace.reserve_tensor(
                "target_verify/gguf-mtp2-physical-r24-c4/remaining_decode",
                (4,),
                DType.INT32,
            )
            self._batch_accept_payload = workspace.reserve_tensor(
                "target_verify/gguf-mtp2-physical-r24-c4/packed_accept_payload",
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

    def _enqueue_target_batch_accept(
        self,
        batch: TargetVerifyBatch,
        *,
        proposal: Qwen35GGUFNextNBatchDeviceProposal | _NgramBatchDeviceProposal,
        target_results: Sequence[Any],
        remaining_decode: Sequence[int],
        transaction_id: int,
        runtime: Any,
        pad_rows: int = 0,
    ) -> _PhysicalAcceptPending:
        """Chain physical proposal/target IDs into GPU accept without D2H."""

        results = tuple(target_results)
        if proposal.request_ids != batch.request_ids or len(results) != len(
            batch.request_ids
        ):
            raise ValueError("physical accept identities do not align")
        if tuple(int(result.request_id) for result in results) != batch.request_ids:
            raise ValueError("physical target result request IDs changed")
        if any(
            int(result.transaction_id) != int(transaction_id)
            for result in results
        ):
            raise ValueError("physical target result transaction changed")
        slot_counts = tuple(int(count) for count in proposal.candidate_counts)
        if int(pad_rows) > 0:
            if not slot_counts:
                raise ValueError("target row padding requires at least one request")
            slot_counts = (
                slot_counts[:-1]
                + (slot_counts[-1] + int(pad_rows),)
            )
        if tuple(int(result.rows) - 1 for result in results) != slot_counts:
            raise ValueError("physical target rows do not match proposal counts")
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
        upload_seconds = 0.0

        def _upload_timed(tensor: Tensor, values: np.ndarray, rt: Any) -> None:
            nonlocal upload_seconds
            started = time.perf_counter()
            self._upload_accept_array(tensor, values, rt)
            upload_seconds += time.perf_counter() - started

        roots = np.asarray(tuple(proposal.root_tokens), dtype=np.int32)
        _upload_timed(
            Tensor.from_handle(
                buffers.token_ids.ptr,
                (request_count,),
                DType.INT32,
                buffers.token_ids.device,
            ),
            roots,
            runtime,
        )
        for tensor, values in (
            (buffers.positions, np.asarray(batch.positions, dtype=np.int32)),
            (buffers.parent_rows, np.asarray(batch.parent_rows, dtype=np.int32)),
            (buffers.draft_depths, np.asarray(batch.draft_depths, dtype=np.int32)),
            (buffers.row_to_request, np.asarray(batch.row_to_request, dtype=np.int32)),
            (buffers.active_mask, np.asarray(batch.active_mask, dtype=np.uint8)),
            (remaining, np.asarray(tuple(remaining_decode), dtype=np.int32)),
        ):
            _upload_timed(tensor, values, runtime)
        runtime.memcpy_async(
            buffers.token_ids.ptr + request_count * DType.INT32.itemsize,
            proposal.token_ids.ptr,
            proposal.token_ids.numel * DType.INT32.itemsize,
            HipMemcpyKind.DEVICE_TO_DEVICE,
            0,
        )
        if int(pad_rows) > 0:
            pads = np.zeros((int(pad_rows),), dtype=np.int32)
            _upload_timed(
                Tensor.from_handle(
                    buffers.token_ids.ptr
                    + (request_count + int(proposal.token_ids.numel))
                    * DType.INT32.itemsize,
                    (int(pad_rows),),
                    DType.INT32,
                    buffers.token_ids.device,
                ),
                pads,
                runtime,
            )
        tail_started = time.perf_counter()
        candidate_offset = request_count
        for request_index, (result, candidate_count) in enumerate(
            zip(results, slot_counts, strict=True)
        ):
            runtime.memcpy_async(
                buffers.target_top1.ptr
                + request_index * DType.INT32.itemsize,
                result.target_top1.ptr,
                DType.INT32.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
            runtime.memcpy_async(
                buffers.target_top1.ptr
                + candidate_offset * DType.INT32.itemsize,
                result.target_top1.ptr + DType.INT32.itemsize,
                int(candidate_count) * DType.INT32.itemsize,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            )
            candidate_offset += int(candidate_count)
        if self._batch_accept_library is None:
            self._batch_accept_library = build_dflash_accept(
                load=True,
                compiler_version=getattr(self.generator, "compiler_version", None),
                require_cached=bool(
                    getattr(self.generator, "require_cached_build", False)
                ),
            )
        output_stride = (
            int(buffers.committed_output_ids.strides[0])
            if buffers.committed_output_ids is not None
            and buffers.committed_output_ids.strides is not None
            else batch.rows
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
            output_stride,
            library=self._batch_accept_library,
            runtime=runtime,
        )
        return _PhysicalAcceptPending(
            batch=batch,
            buffers=buffers,
            payload=payload,
            request_count=request_count,
            output_stride=output_stride,
            upload_seconds=upload_seconds,
            tail_seconds=time.perf_counter() - tail_started,
        )

    @staticmethod
    def _read_target_batch_accept(
        pending: _PhysicalAcceptPending,
        *,
        runtime: Any,
    ) -> TargetAcceptSummary:
        """Read one bounded committed-output/status result after device commit.

        The first blocking default-stream D2H copy is the required producer-to-
        host dependency. Do not synchronize the whole device first: unrelated
        streams need not retire before this transaction's bounded payload.
        """

        payload_host = np.empty(
            (pending.request_count, ACCEPT_PACKED_PAYLOAD_FIELDS),
            dtype=np.int32,
        )
        copy_device_to_host(
            host_array_ptr(payload_host),
            DeviceBuffer(pending.payload.ptr, payload_host.nbytes),
            payload_host.nbytes,
            runtime=runtime,
        )
        committed = np.full(
            (pending.request_count, pending.output_stride),
            -1,
            dtype=np.int32,
        )
        output = pending.buffers.committed_output_ids
        if output is None:
            raise RuntimeError("physical accept omitted committed output IDs")
        for request_index, candidate_count in enumerate(
            pending.batch.candidate_counts
        ):
            count = int(candidate_count) + 1
            copy_device_to_host(
                host_array_ptr(committed)
                + request_index * pending.output_stride * DType.INT32.itemsize,
                DeviceBuffer(
                    output.ptr
                    + request_index
                    * pending.output_stride
                    * DType.INT32.itemsize,
                    count * DType.INT32.itemsize,
                ),
                count * DType.INT32.itemsize,
                runtime=runtime,
            )
        accepted_counts = tuple(int(value) for value in payload_host[:, 0])
        accepted_tokens = tuple(
            tuple(
                int(token)
                for token in committed[index, 1 : accepted + 1].tolist()
            )
            for index, accepted in enumerate(accepted_counts)
        )
        return TargetAcceptSummary(
            request_ids=pending.batch.request_ids,
            accepted_counts=accepted_counts,
            accepted_tokens=accepted_tokens,
            commit_rows=tuple(int(value) for value in payload_host[:, 1]),
            commit_tokens=tuple(int(value) for value in payload_host[:, 2]),
            commit_positions=tuple(int(value) for value in payload_host[:, 3]),
            next_tokens=tuple(
                None if int(value) < 0 else int(value)
                for value in payload_host[:, 4]
            ),
            full_accept=tuple(bool(value) for value in payload_host[:, 5]),
            candidate_counts=pending.batch.candidate_counts,
            transaction_id=pending.buffers.transaction_id,
            draft_depth=pending.batch.draft_depth,
            tree_shape=pending.batch.tree_shape,
            mode=pending.batch.mode,
        )

    def _qualify_target_batch_device_accept(
        self,
        frontier: TargetFrontier,
        proposal: Qwen35GGUFNextNBatchDeviceProposal,
        target_results: Sequence[Any],
        summary: TargetAcceptSummary,
        remaining_decode: Sequence[int],
        *,
        provider: Any,
        runtime: Any,
    ) -> float:
        """Run the post-commit CPU oracle outside the promoted device chain."""

        started = time.perf_counter()
        materialized = provider.materialize_batch_device_proposal(proposal)
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
        if graph is None:
            raise RuntimeError("device-chain oracle lost its candidate graph")
        host_graph = replace(graph, candidate_tokens=candidate_tokens)
        batch = TargetVerifyBatch.from_draft(
            host_graph.to_draft_batch(),
            root_tokens=frontier.root_tokens,
            root_positions=frontier.root_positions,
        )
        top1: list[int] = [0] * batch.rows
        root_by_id = dict(zip(batch.request_ids, batch.root_rows, strict=True))
        candidate_by_id = {
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
            for request_id in batch.request_ids
        }
        for result in target_results:
            values = np.empty((int(result.rows),), dtype=np.int32)
            copy_device_to_host(
                host_array_ptr(values),
                DeviceBuffer(result.target_top1.ptr, values.nbytes),
                values.nbytes,
                runtime=runtime,
            )
            destination_rows = (
                root_by_id[int(result.request_id)],
                *candidate_by_id[int(result.request_id)],
            )
            if len(destination_rows) != values.size:
                raise RuntimeError("device-chain oracle target rows changed")
            for row, value in zip(destination_rows, values, strict=True):
                top1[row] = int(value)
        cpu_accept = batch.accept_from_top1(
            tuple(top1),
            transaction_id=summary.transaction_id,
            remaining_decode=tuple(int(value) for value in remaining_decode),
        )
        cpu_summary = replace(
            TargetAcceptSummary.from_accept_result(batch, cpu_accept),
            transaction_id=summary.transaction_id,
        )
        if any(
            getattr(summary, field) != getattr(cpu_summary, field)
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
                "physical GPU device-chain accept does not match CPU oracle"
            )
        for trace in _device_chain_oracle_trace_rows(
            batch,
            top1,
            summary,
            cycle_id=frontier.cycle_id,
        ):
            row = self.owner._row(int(trace["request_id"]))
            traces = getattr(row, "mtp2_device_chain_oracle_trace", None)
            if traces is not None and len(traces) < 64:
                traces.append(trace)
        return time.perf_counter() - started

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
        ngram_proposal = states[0].proposal_ngram
        device_candidates = (
            ngram_proposal if ngram_proposal is not None else device_draft
        )
        if any(state.proposal_ngram is not ngram_proposal for state in states):
            raise RuntimeError("physical requests do not share one n-gram proposal")
        if batch is None:
            if device_candidates is None or frontier.candidate_graph is None:
                raise RuntimeError("device target frontier lost its proposal descriptor")
            if device_draft is not None and any(
                state.proposal_device_batch is not device_draft for state in states
            ):
                raise RuntimeError("physical requests do not share one device proposal")
            if frontier.candidate_graph.token_ids is not device_candidates.token_ids:
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
        pad_rows = 0
        pad_token_tensor: Tensor | None = None
        if device_draft is not None:
            offset = 0
            for request_id, count in zip(
                device_draft.request_ids,
                device_draft.candidate_counts,
                strict=True,
            ):
                device_offsets[int(request_id)] = (offset, int(count))
                offset += int(count)
            if (
                batch is None
                and ngram_proposal is None
                and device_draft.request_ids
            ):
                pad_rows = self._target_group_pad_rows(
                    request_count=len(device_draft.request_ids),
                    candidate_rows=sum(
                        int(count) for count in device_draft.candidate_counts
                    ),
                )
            if pad_rows:
                pad_token_tensor = self._target_pad_token_tensor(
                    device_draft,
                    pad_rows=pad_rows,
                    runtime=targets[0].runtime,
                )
                last_request = int(device_draft.request_ids[-1])
                last_offset, last_count = device_offsets[last_request]
                device_offsets[last_request] = (last_offset, last_count + pad_rows)
        for request_id, target in zip(ids, targets, strict=True):
            job = {
                "session": target,
                "request_id": int(request_id),
                "resident_slot": int(plan.resident_slots[plan.request_ids.index(request_id)]),
                "transaction_id": int(transaction_id),
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
                candidate_source = (
                    pad_token_tensor
                    if pad_token_tensor is not None
                    else device_draft.token_ids
                )
                job["candidate_token_ids_device"] = Tensor.from_handle(
                    candidate_source.ptr + offset * DType.INT32.itemsize,
                    (count,),
                    DType.INT32,
                    candidate_source.device,
                )
            jobs.append(job)
        owner = self.owner._packed_execution_owner(targets[0])
        verify_batch = getattr(owner, "verify_target_blocks_batch", None)
        if not callable(verify_batch):
            raise RuntimeError("physical target owner has no packed verifier")
        target_started = time.perf_counter()
        device_result = batch is None or ngram_proposal is not None
        with (
            q4_t16_physical_extra_rowtiles_session(
                bool(getattr(self, "production_physical_extra_rowtiles", False))
            ),
            q5_t16_physical_rowtile_session(
                bool(getattr(self, "production_physical_q5_rowtile", False))
            ),
            q6_t16_physical_rowtile_session(
                bool(getattr(self, "production_physical_q6_rowtile", False))
            ),
            moe_physical_c2_numerics_session(
                bool(getattr(self, "moe_physical_c2_numerics", False))
            ),
            moe_physical_c2_pairreuse_session(
                bool(getattr(self, "moe_physical_c2_pairreuse", False))
            ),
            moe_physical_c2_exact_linear_session(
                bool(getattr(self, "moe_physical_c2_exact_linear", False))
            ),
        ):
            results = list(verify_batch(jobs, device_result=device_result))
        target_seconds = time.perf_counter() - target_started
        physical_target_rows = sum(len(job["input_token_ids"]) for job in jobs)
        for row in rows:
            row.mtp2_target_batch_calls += 1
            row.mtp2_target_physical_rows.append(physical_target_rows)
        if len(results) != len(ids):
            raise RuntimeError("physical target verifier returned wrong result count")
        candidate_readback_seconds = 0.0
        bounded_readback_seconds = 0.0
        accept_upload_seconds = 0.0
        accept_tail_seconds = 0.0
        remaining = tuple(
            max(0, int(row.request.max_tokens) - len(row.slot.generated_ids))
            for row in rows
        )
        if device_result:
            assert device_candidates is not None
            graph = frontier.candidate_graph
            assert graph is not None
            if batch is None:
                if pad_rows:
                    if pad_token_tensor is None or graph.token_ids is None:
                        raise RuntimeError(
                            "target row padding lost its device token tensor"
                        )
                    graph = pad_candidate_graph_rows(
                        graph,
                        pad_rows=pad_rows,
                        pad_token_id=0,
                        token_ids=pad_token_tensor,
                    )
                shape_graph = replace(
                    graph,
                    candidate_tokens=(0,) * graph.candidate_rows,
                )
                batch = TargetVerifyBatch.from_draft(
                    shape_graph.to_draft_batch(),
                    root_tokens=frontier.root_tokens,
                    root_positions=frontier.root_positions,
                )
            cancelled = tuple(int(value) for value in cancelled_request_ids())
            if cancelled:
                targets[0].runtime.device_synchronize()
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
            accept_started = time.perf_counter()
            pending = self._enqueue_target_batch_accept(
                batch,
                proposal=device_candidates,
                target_results=results,
                remaining_decode=remaining,
                transaction_id=transaction_id,
                runtime=targets[0].runtime,
                pad_rows=pad_rows,
            )
            accept_upload_seconds = float(pending.upload_seconds)
            accept_tail_seconds = float(pending.tail_seconds)
            commit_batch = getattr(
                owner,
                "_commit_deferred_packed_verify_states_batch_device",
                None,
            )
            if not callable(commit_batch):
                raise RuntimeError(
                    "physical target owner has no device selected-state commit"
                )
            commit_started = time.perf_counter()
            try:
                commit_contract = commit_batch(
                    results,
                    targets,
                    accept_buffers=pending.buffers,
                )
                commit_seconds = time.perf_counter() - commit_started
                readback_started = time.perf_counter()
                gpu_summary = self._read_target_batch_accept(
                    pending,
                    runtime=targets[0].runtime,
                )
                bounded_readback_seconds = time.perf_counter() - readback_started
                if device_draft is not None and bool(
                    getattr(self, "device_chain_qualification_oracle", False)
                ):
                    candidate_readback_seconds = (
                        self._qualify_target_batch_device_accept(
                            frontier,
                            device_draft,
                            results,
                            gpu_summary,
                            remaining,
                            provider=states[0].provider,
                            runtime=targets[0].runtime,
                        )
                    )
                    for row in rows:
                        row.mtp2_candidate_d2h_after_target += 1
                accept_seconds = time.perf_counter() - accept_started
                if int(commit_contract.get("requests", 0)) != len(ids):
                    raise RuntimeError(
                        "physical selected-state commit omitted requests"
                    )
                for target, summary_position in zip(
                    targets, gpu_summary.commit_positions, strict=True
                ):
                    next_position = int(summary_position) + 1
                    target._position = next_position
                    target.scratch.position_host[0] = next_position
                    target.scratch.context_host[0] = next_position + 1
                accept = AcceptResult(
                    request_ids=ids,
                    accepted_counts=gpu_summary.accepted_counts,
                    accepted_tokens=gpu_summary.accepted_tokens,
                    transaction_id=transaction_id,
                    selected_candidate_rows=gpu_summary.commit_rows,
                    next_tokens=gpu_summary.next_tokens,
                    correction_or_bonus_tokens=gpu_summary.next_tokens,
                    target_cursor_deltas=tuple(
                        int(count) + (1 if next_token is not None else 0)
                        for count, next_token in zip(
                            gpu_summary.accepted_counts,
                            gpu_summary.next_tokens or (None,) * len(ids),
                            strict=True,
                        )
                    ),
                    provider_cursor_deltas=gpu_summary.accepted_counts,
                    finish_reasons=(None,) * len(ids),
                )
                provider_update_started = time.perf_counter()
                if ngram_proposal is not None:
                    hidden_sources = tuple(
                        result.pre_output_norm_hidden for result in results
                    )
                    if any(source is None for source in hidden_sources):
                        raise RuntimeError(
                            "n-gram target verifier omitted pre-output hidden rows"
                        )
                    self._repair_provider_states_from_ngram_target_rows(
                        states,
                        tuple(source for source in hidden_sources if source is not None),
                        accepted_counts=gpu_summary.accepted_counts,
                    )
                else:
                    assert device_draft is not None
                    self._repair_provider_states_batch_device(
                        states,
                        device_draft,
                        accepted_counts=gpu_summary.accepted_counts,
                    )
                provider_update_seconds = (
                    time.perf_counter() - provider_update_started
                )
            except BaseException as error:
                raise _PhysicalTargetCommitError(
                    "physical target selected commit path failed: "
                    f"{type(error).__name__}:{error}"
                ) from error
        else:
            assert batch is not None
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
            provider_update_started = time.perf_counter()
            self._repair_provider_states_batch(
                states,
                accepted_counts=accept.accepted_counts,
                candidate_counts=tuple(
                    plan.candidate_counts[plan.request_ids.index(request_id)]
                    for request_id in ids
                ),
            )
            provider_update_seconds = (
                time.perf_counter() - provider_update_started
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
            commit_batch = getattr(
                owner,
                "_commit_deferred_packed_verify_states_batch",
                None,
            )
            if not callable(commit_batch):
                raise RuntimeError(
                    "physical target owner has no batch selected-state commit"
                )
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
            row.mtp2_device_accept_calls += 1
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
                raise RuntimeError(
                    "physical target cycle produced no visible token: "
                    f"request={request_id} accepted={accepted} "
                    f"next={next_token} remaining={remaining[index]} "
                    f"summary={gpu_summary!r}"
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
            row.mtp2_accepted_counts.append(int(accepted))
            if (
                self.post_reject_cooldown_enabled
                and int(accepted) == 0
                and int(
                    plan.candidate_counts[plan.request_ids.index(request_id)]
                )
                > 0
            ):
                self._post_reject_pending.add(int(request_id))
            row.mtp2_proposal_ms += float(states[index].last_proposal_seconds) * 1000.0
            row.mtp2_target_ms += float(target_seconds) * 1000.0
            row.mtp2_provider_update_ms += float(provider_update_seconds) * 1000.0
            row.mtp2_accept_ms += float(accept_seconds) * 1000.0
            row.mtp2_accept_upload_ms += accept_upload_seconds * 1000.0
            row.mtp2_accept_tail_ms += accept_tail_seconds * 1000.0
            row.mtp2_target_readback_ms += float(bounded_readback_seconds) * 1000.0
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
            next_tokens=tuple(next_tokens),
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
            k0_classes=plan.k0_classes,
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route="eager",
            proposal_seconds=max(state.last_proposal_seconds for state in states),
            target_seconds=target_seconds,
            accept_commit_seconds=accept_seconds + commit_seconds,
            provider_update_seconds=provider_update_seconds,
            scheduler_readback_seconds=(
                candidate_readback_seconds + bounded_readback_seconds
            ),
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

    def _repair_provider_states_from_ngram_target_rows(
        self,
        states: tuple[_MTP2RequestState, ...],
        hidden_rows: Sequence[Tensor],
        *,
        accepted_counts: Sequence[int],
    ) -> None:
        """Catch target-attached MTP state up after a model-free proposal.

        The MTP checkpoint is captured before provider selection. N-gram lookup
        does not mutate it, while target verification produces the exact trunk
        hidden row for the root and every candidate. Replaying only root plus
        the accepted candidate prefix leaves MTP at the same boundary it would
        own after an ordinary proposal/repair cycle.
        """

        accepted = tuple(int(value) for value in accepted_counts)
        sources = tuple(hidden_rows)
        if not states or len(states) != len(accepted) or len(states) != len(sources):
            raise ValueError("n-gram MTP catch-up requires aligned request rows")
        if len({state.provider_group_key for state in states}) != 1:
            raise RuntimeError("n-gram MTP catch-up requires one provider group")
        executor = states[0].provider.executor
        hidden_size = int(executor.hidden_size)
        normalized: list[Tensor] = []
        for state, count, source in zip(states, accepted, sources, strict=True):
            if state.proposal_source != "ngram_mod":
                raise RuntimeError("n-gram MTP catch-up received another provider source")
            if count < 0 or count > len(state.ngram_candidate_tokens):
                raise ValueError("accepted n-gram count is outside the candidate chain")
            if source.dtype != DType.BF16:
                raise ValueError("n-gram target hidden rows must use BF16")
            if source.ndim == 3:
                if source.shape[0] != 1:
                    raise ValueError("n-gram target hidden batch must have one owner axis")
                source = Tensor.from_handle(
                    source.ptr,
                    (source.shape[1], source.shape[2]),
                    source.dtype,
                    source.device,
                )
            if source.shape != (len(state.ngram_candidate_tokens) + 1, hidden_size):
                raise ValueError("n-gram target hidden rows do not match the verifier chain")
            checkpoint = state.proposal_checkpoint
            if checkpoint is None or state.proposal_context is None:
                raise RuntimeError("n-gram MTP catch-up has no provider checkpoint")
            executor.restore_request_checkpoint(checkpoint)
            normalized.append(source)

        _proposal_hidden, packed_hidden = self._cycle_hidden_tensors(
            executor.runtime,
            hidden_size=hidden_size,
        )
        hidden_nbytes = hidden_size * DType.BF16.itemsize
        maximum_depth = max(count + 1 for count in accepted)
        for depth in range(maximum_depth):
            active = tuple(
                index for index, count in enumerate(accepted) if depth <= count
            )
            tokens: list[int] = []
            positions: list[int] = []
            for packed_row, state_index in enumerate(active):
                state = states[state_index]
                context = state.proposal_context
                assert context is not None
                token = (
                    int(context.root_tokens[0])
                    if depth == 0
                    else int(state.ngram_candidate_tokens[depth - 1])
                )
                tokens.append(token)
                positions.append(int(context.root_positions[0]) + depth)
                executor.runtime.memcpy(
                    packed_hidden.ptr + packed_row * hidden_nbytes,
                    normalized[state_index].ptr + depth * hidden_nbytes,
                    hidden_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
            hidden = Tensor.from_handle(
                packed_hidden.ptr,
                (len(active), hidden_size),
                DType.BF16,
                packed_hidden.device,
            )
            ids = tuple(states[index].request_id for index in active)
            if len(active) == 1:
                executor.advance_state_only(
                    ids[0],
                    tokens[0],
                    positions[0],
                    Tensor.from_handle(
                        hidden.ptr,
                        (1, hidden_size),
                        DType.BF16,
                        hidden.device,
                    ),
                )
            else:
                executor.advance_state_batch_only(
                    ids,
                    tuple(tokens),
                    tuple(positions),
                    hidden,
                )
        for state, count in zip(states, accepted, strict=True):
            row = self.owner._row(state.request_id)
            self._add_row_counter(row, "mtp2_ngram_accepted_tokens", count)

    def _repair_provider_states_batch_device(
        self,
        states: tuple[_MTP2RequestState, ...],
        proposal: Qwen35GGUFNextNBatchDeviceProposal,
        *,
        accepted_counts: Sequence[int],
    ) -> None:
        """Repair physical provider state without materializing candidate IDs."""

        accepted = tuple(int(value) for value in accepted_counts)
        if (
            not states
            or len(states) != len(accepted)
            or proposal.request_ids
            != tuple(int(state.request_id) for state in states)
        ):
            raise ValueError("physical device repair requires aligned request rows")
        if any(
            value < 0 or value > count
            for value, count in zip(
                accepted, proposal.candidate_counts, strict=True
            )
        ):
            raise ValueError("accepted count is outside the device proposal")
        executor = states[0].provider.executor
        hidden_size = int(executor.hidden_size)
        hidden_nbytes = hidden_size * DType.BF16.itemsize
        _proposal_hidden, repair_hidden = self._cycle_hidden_tensors(
            executor.runtime,
            hidden_size=hidden_size,
        )

        def packed_hidden(values: Sequence[Tensor]) -> Tensor:
            rows = tuple(values)
            for row, hidden in enumerate(rows):
                executor.runtime.memcpy(
                    repair_hidden.ptr + row * hidden_nbytes,
                    hidden.ptr,
                    hidden_nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
            return Tensor.from_handle(
                repair_hidden.ptr,
                (len(rows), hidden_size),
                DType.BF16,
                Device("hip", 0),
            )

        restored: list[tuple[_MTP2RequestState, MtpProposalContext]] = []
        root_snapshot_restore = getattr(executor, "restore_request_root_state", None)
        root_snapshot_requests: set[int] = set()
        keep_proposal_state: set[int] = set()
        for state, accepted_count, candidate_count in zip(
            states,
            accepted,
            proposal.candidate_counts,
            strict=True,
        ):
            checkpoint = state.proposal_checkpoint
            context = state.proposal_context
            if checkpoint is None or context is None:
                raise RuntimeError("GGUF MTP2 device repair has no checkpoint")
            count = int(candidate_count)
            if accepted_count == 0 and callable(root_snapshot_restore):
                root_snapshot_restore(state.request_id)
                root_snapshot_requests.add(int(state.request_id))
            elif count > 1 and accepted_count == count - 1:
                # A K-token proposal has already advanced provider state through
                # candidate K-1. Publishing that state is exact and needs no
                # restore/replay.
                keep_proposal_state.add(int(state.request_id))
            elif accepted_count < count:
                executor.restore_request_checkpoint(checkpoint)
                restored.append((state, context))
        if restored:
            ids = tuple(state.request_id for state, _context in restored)
            tokens = tuple(int(context.root_tokens[0]) for _state, context in restored)
            positions = tuple(
                int(context.root_positions[0]) for _state, context in restored
            )
            hidden = packed_hidden(
                tuple(context.target_hidden for _state, context in restored)
            )
            advance_one = getattr(executor, "advance_state_only", None)
            if len(restored) == 1 and callable(advance_one):
                advance_one(
                    ids[0],
                    tokens[0],
                    positions[0],
                    Tensor.from_handle(
                        hidden.ptr,
                        (1, hidden_size),
                        DType.BF16,
                        hidden.device,
                    ),
                )
            else:
                executor.advance_state_batch_only(ids, tokens, positions, hidden)

        offsets: dict[int, int] = {}
        cursor = 0
        for request_id, count in zip(
            proposal.request_ids, proposal.candidate_counts, strict=True
        ):
            offsets[int(request_id)] = cursor
            cursor += int(count)
        operations: list[list[tuple[Tensor, int, Tensor]]] = []
        for state, accepted_count, candidate_count, hidden_rows in zip(
            states,
            accepted,
            proposal.candidate_counts,
            proposal.hidden_rows,
            strict=True,
        ):
            root_position = int(proposal.root_positions[
                proposal.request_ids.index(state.request_id)
            ])
            request_id = int(state.request_id)
            if request_id in root_snapshot_requests or request_id in keep_proposal_state:
                candidate_indices = ()
            elif accepted_count == int(candidate_count):
                candidate_indices = (int(candidate_count) - 1,)
            else:
                candidate_indices = tuple(range(accepted_count))
            base = offsets[state.request_id]
            operations.append(
                [
                    (
                        Tensor.from_handle(
                            proposal.token_ids.ptr
                            + (base + candidate_index) * DType.INT32.itemsize,
                            (1,),
                            DType.INT32,
                            proposal.token_ids.device,
                        ),
                        root_position + candidate_index + 1,
                        hidden_rows[candidate_index],
                    )
                    for candidate_index in candidate_indices
                ]
            )
        for depth in range(max((len(rows) for rows in operations), default=0)):
            active = tuple(
                index for index, rows in enumerate(operations) if depth < len(rows)
            )
            ids = tuple(states[index].request_id for index in active)
            tokens = tuple(operations[index][depth][0] for index in active)
            positions = tuple(operations[index][depth][1] for index in active)
            hidden = packed_hidden(
                tuple(operations[index][depth][2] for index in active)
            )
            if len(active) == 1:
                advance_one = getattr(executor, "advance_state_only_device", None)
                if callable(advance_one):
                    advance_one(
                        ids[0],
                        tokens[0],
                        positions[0],
                        Tensor.from_handle(
                            hidden.ptr,
                            (1, hidden_size),
                            DType.BF16,
                            hidden.device,
                        ),
                    )
                else:
                    executor.advance_state_batch_only_device(
                        ids, tokens, positions, hidden
                    )
            else:
                executor.advance_state_batch_only_device(
                    ids, tokens, positions, hidden
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
        if not states or len(accepted) != len(states) or len(counts) != len(states):
            raise ValueError("physical provider repair requires aligned request rows")
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
        _proposal_hidden, hidden_batch = self._cycle_hidden_tensors(
            executor.runtime,
            hidden_size=hidden_size,
        )
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

    def _repair_provider_state_device(
        self,
        state: _MTP2RequestState,
        proposal: Qwen35GGUFNextNDeviceProposal,
        *,
        accepted_count: int,
    ) -> None:
        """Repair C1 provider state from proposal device rows only."""

        checkpoint = state.proposal_checkpoint
        context = state.proposal_context
        hidden_rows = proposal.hidden_rows
        if checkpoint is None or context is None or hidden_rows is None:
            raise RuntimeError("GGUF MTP2 C1 device repair lost proposal ownership")
        accepted = int(accepted_count)
        if accepted < 0 or accepted > int(proposal.budget):
            raise ValueError("accepted_count is outside the device proposal")
        executor = state.provider.executor
        if accepted < int(proposal.budget):
            executor.restore_request_checkpoint(checkpoint)
            executor.advance_state_only(
                state.request_id,
                int(context.root_tokens[0]),
                int(context.root_positions[0]),
                context.target_hidden,
            )
            candidate_indices = tuple(range(accepted))
        else:
            candidate_indices = (int(proposal.budget) - 1,)
        hidden_size = int(hidden_rows.shape[1])
        hidden_nbytes = hidden_size * DType.BF16.itemsize
        for candidate_index in candidate_indices:
            executor.advance_state_only_device(
                state.request_id,
                Tensor.from_handle(
                    proposal.result_ptr
                    + candidate_index * 2 * DType.INT32.itemsize,
                    (1,),
                    DType.INT32,
                    proposal.final_hidden.device,
                ),
                int(context.root_positions[0]) + candidate_index + 1,
                Tensor.from_handle(
                    hidden_rows.ptr + candidate_index * hidden_nbytes,
                    (1, hidden_size),
                    DType.BF16,
                    hidden_rows.device,
                ),
            )

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
            state.proposal_device = None
            state.proposal_device_batch = None
            state.proposal_ngram = None
            state.ngram_candidate_tokens = ()
            state.proposal_source = "mtp2"

    def _release_provider_checkpoint(self, state: _MTP2RequestState) -> None:
        checkpoint = state.proposal_checkpoint
        if checkpoint is None:
            return
        state.provider.executor.release_request_checkpoint(checkpoint)
        state.proposal_checkpoint = None
        state.proposal_context = None
        state.proposal_device = None
        state.proposal_device_batch = None
        state.proposal_ngram = None
        state.ngram_candidate_tokens = ()
        state.proposal_source = "mtp2"

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

        reason = f"{type(error).__name__}:{error}"
        rows = tuple(
            self.owner._row(int(request_id))
            for request_id in plan.speculative_request_ids
        )
        if not rows:
            return False
        if isinstance(error, _PhysicalTargetCommitError):
            rebuild = getattr(
                self.owner,
                "restore_speculative_target_rows",
                None,
            )
            if not callable(rebuild) or not bool(rebuild(plan)):
                for row in rows:
                    row.mtp2_failure_reasons.extend(
                        ("postcommit_failure_fatal", reason)
                    )
                return False
            for row in rows:
                row.mtp2_recoverable_failures += 1
                row.mtp2_failure_reasons.extend(
                    ("postcommit_target_rebuild_ar_fallback", reason)
                )
            return True
        for row in rows:
            if row.slot is None or row.lease is None:
                return False
            if int(row.lease.session.position) != int(row.slot.seq_position):
                return False
        for row in rows:
            row.mtp2_recoverable_failures += 1
            row.mtp2_failure_reasons.extend(
                ("precommit_failure_ar_fallback", reason)
            )
        return True

    def release_request(self, request_id: int) -> None:
        rid = int(request_id)
        ngram = getattr(self, "_ngram", None)
        if ngram is not None:
            ngram.release_request(rid)
        if rid in self._prompt_streaming_sinks:
            self._abort_prompt_streaming((rid,), stream=0)
        self._drop_request(rid, disable=False)
        self._intents.pop(rid, None)
        getattr(self, "_static_eligibility_by_request", {}).pop(rid, None)
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
        getattr(self, "_static_eligibility_by_request", {}).clear()
        self._prompt_hidden_rows.clear()
        self._disabled_requests.clear()
        self._active_claims = None
        self._active_prompt_claims = None
        for request_id in tuple(
            getattr(self, "_prompt_streaming_norm_buffers", {})
        ):
            self._free_prompt_streaming_norm_buffer(request_id, target=None)
        self._close_cycle_workspace()
        ngram = getattr(self, "_ngram", None)
        if ngram is not None:
            ngram.close()
        if self._batch_accept_workspace is not None:
            self._batch_accept_workspace.free()
            self._batch_accept_workspace = None
            self._batch_accept_owner = None
            self._batch_accept_remaining = None
            self._batch_accept_payload = None
            self._batch_accept_library = None
        if self._target_pad_token_scratch is not None:
            free(self._target_pad_token_scratch)
            self._target_pad_token_scratch = None
            self._target_pad_token_capacity = 0

    def _open_batch_requests(self, request_ids: tuple[int, ...]) -> None:
        rows = [self.owner._row(request_id) for request_id in request_ids]
        targets = [row.lease.session for row in rows]
        if any(not self._target_profile_supported(target) for target in targets):
            raise RuntimeError("GGUF MTP2 target execution profile is unsupported")
        for target in targets:
            self._bind_target_profile_metadata(target)
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
        if not self._target_profile_supported(target):
            raise RuntimeError("GGUF MTP2 target execution profile is unsupported")
        self._bind_target_profile_metadata(target)
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
                    # Prompt catch-up consumes provider state only. Scoring a
                    # full LM head here performs one unnecessary vocabulary
                    # projection and D2H readback per prompt position.
                    provider.executor.advance_state_batch_only(
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
        if not self._target_profile_supported(target):
            raise RuntimeError("GGUF MTP2 target execution profile is unsupported")
        self._bind_target_profile_metadata(target)
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
                target_verify_mode=_target_verify_mode_for_context(
                    self.target_verify_mode,
                    backend=self.generator.backend,
                    end_position=len(row.prompt_ids) + self.candidate_budget + 1,
                ),
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
        self._post_reject_pending.discard(rid)
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
