"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import time
from typing import Any, ClassVar
import uuid

from hipengine.dispatch import (
    BatchWidthGroup,
    NativeBatchWidthProfile,
    RequestState,
    SlotMove,
    WorkItem,
    plan_batch_width_partition,
)
from hipengine.generation.batch_scheduler import (
    CompactPromptSlab,
    CompletedRequest,
    GeneratedToken,
    PerRowSamplingParams,
    ResidentBatchScheduler,
)
from hipengine.generation.constraints import token_sequence_state_for_tokens
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
    PromptInput,
    TokenLogprob,
    register_text_generator,
)
from hipengine.generation.sampling import (
    RowSamplingState,
    SamplingMode,
    clone_thinking_budget_state,
    plan_sampler,
    row_seed_for_index,
    thinking_budget_state_from_params,
)
from hipengine.kernels.backends import backend_package_capability
from hipengine.kvcache import resolve_kv_policy
from hipengine.loading import WeightIndex
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
    _decode_token_cached,
    _select_token,
)
from hipengine.runtime.qwen35_paro_batch_width import (
    DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE,
    QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_ENV,
    load_qwen35_paro_native_batch_width_profile,
)


def _new_timing_batch_id(kind: str) -> str:
    return f"paro-{str(kind)}-{uuid.uuid4().hex}"


def _prompt_ids(model_path: Path, prompt: PromptInput) -> tuple[int, ...]:
    if not isinstance(prompt, str):
        return tuple(int(token) for token in prompt)
    _last_token_id, prompt_ids = _select_token(model_path, prompt, None)
    return tuple(int(token) for token in prompt_ids)


_EXACT_HYBRID_C2_MAX_CONTEXT = 1023


def _exact_hybrid_c2_route_blockers(
    *,
    model_path: Path,
    prompts: tuple[PromptInput, ...],
    max_tokens: int,
    kv_policy: Any,
    target_arch: str | None,
) -> tuple[str, ...]:
    """Return conservative blockers for the correctness-backed c2 hybrid."""

    blockers: list[str] = []
    normalized_arch = "" if target_arch is None else str(target_arch).split(":", 1)[0]
    if normalized_arch != "gfx1151":
        blockers.append("exact PARO c2 hybrid is currently certified only on gfx1151")
    if len(prompts) != 2:
        blockers.append("exact PARO hybrid currently requires exactly two requests")
    storage_dtype = getattr(getattr(kv_policy, "storage_dtype", None), "value", None)
    if storage_dtype != "bf16":
        blockers.append("exact PARO c2 hybrid currently requires BF16 KV")
    if not blockers:
        prompt_lengths = tuple(len(_prompt_ids(model_path, prompt)) for prompt in prompts)
        if any(length <= 0 for length in prompt_lengths):
            blockers.append("exact PARO c2 hybrid requires non-empty prompts")
        max_context = max(prompt_lengths) + max(0, int(max_tokens))
        if max_context > _EXACT_HYBRID_C2_MAX_CONTEXT:
            blockers.append(
                "exact PARO c2 hybrid is certified only below split-K context "
                f"({_EXACT_HYBRID_C2_MAX_CONTEXT} tokens)"
            )
    return tuple(blockers)


@dataclass
class Qwen35ParoOneTokenGenerator:
    """Greedy Qwen3.5/PARO generator backed by resident c=1 execution.

    The implementation is still serial across prompts, but each prompt uses the
    resident single-request native prefill path followed by multi-token
    autoregressive decode using the resident HIP layer chain.
    """

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "auto"
    lm_head_chunk: int = 4096
    _runner: Qwen35ParoNextTokenRunner | None = field(default=None, init=False, repr=False)
    _session: Qwen35ParoResidentSession | None = field(default=None, init=False, repr=False)
    _session_capacity: int = field(default=0, init=False, repr=False)
    _session_batch_size: int = field(default=0, init=False, repr=False)
    _session_kv_key: tuple[str, str, str, int] | None = field(default=None, init=False, repr=False)
    _resident_model_runner: "Qwen35ParoResidentModelRunner | None" = field(default=None, init=False, repr=False)
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)
    supports_stream_logprobs: ClassVar[bool] = True

    def create_resident_model_runner(
        self,
        *,
        capacity: int | None = None,
    ) -> "Qwen35ParoResidentModelRunner":
        """Create one fixed-capacity scheduler-facing PARO model owner."""

        if self._resident_model_runner is not None:
            raise RuntimeError("Qwen3.5/PARO resident model runner already exists")
        owner = Qwen35ParoResidentModelRunner(
            self,
            capacity=(8 if capacity is None else int(capacity)),
        )
        self._resident_model_runner = owner
        return owner

    def generate(self, request: GenerationRequest) -> list[str]:
        outputs = self.generate_detailed(request)
        return [output.text for output in outputs]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        native_gpu_available = _native_gpu_sampler_route_available(prompt_count=len(request.prompts))
        plan = plan_sampler(
            request,
            native_gpu_available=native_gpu_available,
            native_gpu_requested=_native_gpu_sampler_requested(),
        )
        if request.max_tokens == 0:
            self.last_batch_generation = None
            self.last_generation_outputs = tuple(
                GenerationOutput(
                    text="",
                    generated_token_ids=(),
                    finish_details=_finish_details_for_tokens(
                        None,
                        (),
                        ignore_eos=request.ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=request.max_tokens,
                        sampler_mode=plan.mode.value,
                    ),
                )
                for _prompt in request.prompts
            )
            return list(self.last_generation_outputs)
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        if len(request.prompts) == 1:
            self.last_batch_generation = None
            if plan.mode is SamplingMode.GREEDY_FAST:
                output = self._generate_one(
                    runner,
                    request.prompts[0],
                    request.max_tokens,
                    ignore_eos=request.ignore_eos,
                    kv_policy=kv_policy,
                    sampler_mode=plan.mode.value,
                    deadline_at=request.deadline_at,
                    cancellation_token=request.cancellation_token,
                )
                self.last_generation_outputs = (output,)
                return list(self.last_generation_outputs)
            output = self._generate_one_sampled(
                runner,
                request.prompts[0],
                request.max_tokens,
                request=request,
                row_index=0,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
                plan=plan,
            )
            self.last_generation_outputs = (output,)
            return [output]
        if plan.mode is not SamplingMode.GREEDY_FAST:
            c1_plan = plan_sampler(
                request,
                native_gpu_available=_native_gpu_sampler_route_available(prompt_count=1),
                native_gpu_requested=_native_gpu_sampler_requested(),
            )
            self.last_generation_outputs = tuple(
                self._generate_batch_sampled_true_c1_fallback(
                    runner,
                    request.prompts,
                    request.max_tokens,
                    request=request,
                    ignore_eos=request.ignore_eos,
                    kv_policy=kv_policy,
                    plan=c1_plan,
                )
            )
            return list(self.last_generation_outputs)
        width_profile = _native_batch_width_profile_for_runner(runner, kv_policy)
        profile_position_blockers = (
            ()
            if width_profile is None or width_profile.blockers
            else _native_profile_prompt_position_blockers(
                width_profile,
                model_path=Path(self.model_path),
                prompts=request.prompts,
                max_tokens=request.max_tokens,
            )
        )
        if width_profile is None or width_profile.blockers or profile_position_blockers:
            exact_hybrid_blockers = _exact_hybrid_c2_route_blockers(
                model_path=Path(self.model_path),
                prompts=request.prompts,
                max_tokens=request.max_tokens,
                kv_policy=kv_policy,
                target_arch=getattr(runner, "target_arch", None),
            )
            if not exact_hybrid_blockers:
                outputs = self._generate_batch(
                    runner,
                    request.prompts,
                    request.max_tokens,
                    ignore_eos=request.ignore_eos,
                    kv_policy=kv_policy,
                    sampler_mode=plan.mode.value,
                    deadline_at=request.deadline_at,
                    cancellation_token=request.cancellation_token,
                    exact_hybrid_c2=True,
                )
                self.last_generation_outputs = tuple(outputs)
                return list(self.last_generation_outputs)
            outputs = self._generate_batch_true_c1_fallback(
                runner,
                request.prompts,
                request.max_tokens,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
                sampler_mode=plan.mode.value,
                deadline_at=request.deadline_at,
                cancellation_token=request.cancellation_token,
                profile=width_profile,
                route_blockers=(
                    ("no accepted native batch width profile",)
                    if width_profile is None
                    else (*width_profile.blockers, *profile_position_blockers)
                ),
            )
            self.last_generation_outputs = tuple(outputs)
            return list(self.last_generation_outputs)
        width_plan = plan_batch_width_partition(
            len(request.prompts),
            profile=width_profile,
        )
        direct_native = (
            len(width_plan.groups) == 1
            and width_plan.groups[0].mode == "native"
            and width_plan.groups[0].width == len(request.prompts)
        )
        if not direct_native:
            outputs = self._generate_batch_isolated_width_groups(
                runner,
                request.prompts,
                request.max_tokens,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
                sampler_mode=plan.mode.value,
                deadline_at=request.deadline_at,
                cancellation_token=request.cancellation_token,
                profile=width_profile,
            )
            self.last_generation_outputs = tuple(outputs)
            return list(self.last_generation_outputs)
        outputs = self._generate_batch(
            runner,
            request.prompts,
            request.max_tokens,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
            sampler_mode=plan.mode.value,
            deadline_at=request.deadline_at,
            cancellation_token=request.cancellation_token,
        )
        self.last_generation_outputs = tuple(outputs)
        return list(self.last_generation_outputs)

    def _generate_batch_true_c1_fallback(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[PromptInput, ...],
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        sampler_mode: str,
        deadline_at: float | None,
        cancellation_token: Any | None,
        profile: NativeBatchWidthProfile | None,
        route_blockers: tuple[str, ...],
    ) -> list[GenerationOutput]:
        """Run each request through the current single-request graph contract."""

        parent_path = "scheduler_true_c1_fallback"
        prompt_rows: list[list[int]] = []
        outputs: list[GenerationOutput] = []
        generated_ids: dict[int, list[int]] = {}
        output_parts: dict[int, list[str]] = {}
        groups: list[dict[str, Any]] = []
        started_at = time.perf_counter()
        batch_id = _new_timing_batch_id("true-c1")
        for row_index, prompt in enumerate(prompts):
            prompt_ids = _prompt_ids(Path(self.model_path), prompt)
            prompt_row = [int(token) for token in prompt_ids]
            if not prompt_row:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append(prompt_row)
            group_started_at = time.perf_counter()
            output = self._generate_one(
                runner,
                prompt,
                max_tokens,
                ignore_eos=ignore_eos,
                kv_policy=kv_policy,
                sampler_mode=sampler_mode,
                deadline_at=deadline_at,
                cancellation_token=cancellation_token,
            )
            group_wall_s = time.perf_counter() - group_started_at
            if output.generated_token_ids is None:
                raise RuntimeError("true c1 fallback did not expose generated token ids")
            token_ids = list(output.generated_token_ids)
            generated_ids[row_index] = token_ids
            tokenizer = self._session.tokenizer if self._session is not None else None
            output_parts[row_index] = [
                _decode_token_cached(tokenizer, token_id)
                for token_id in token_ids
            ]
            relabeled = _relabel_isolated_group_output(
                output,
                row_index=row_index,
                group_index=row_index,
                group_width=1,
                parent_path=parent_path,
                native_compact_prefill=False,
                native_caware_decode=False,
                serial_decode_fallback=True,
            )
            outputs.append(relabeled)
            groups.append(
                {
                    "group_index": row_index,
                    "request_offset": row_index,
                    "planned_mode": "true_c1_graph",
                    "width": 1,
                    "wall_ms": group_wall_s * 1000.0,
                }
            )

        total_wall_s = time.perf_counter() - started_at
        request_ids = tuple(range(len(prompts)))
        prompt_rows_by_request = dict(zip(request_ids, prompt_rows, strict=True))
        self.last_batch_generation = {
            "path": parent_path,
            "batch_id": batch_id,
            "group_rows": len(prompts),
            "timing_owner": True,
            "batch_size": len(prompts),
            "request_ids": list(request_ids),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "group_widths": [1] * len(prompts),
            "group_modes": ["true_c1_graph"] * len(prompts),
            "max_session_width": 1,
            "groups": groups,
            "group_count": len(groups),
            "total_wall_ms": total_wall_s * 1000.0,
            "timing_scope": "batch",
            "batch_timing": {"batch_total_ms": total_wall_s * 1000.0},
            "native_width_profile": None if profile is None else profile.to_json_dict(),
            "route_blockers": list(dict.fromkeys(route_blockers)),
            "serial_decode_fallback": True,
            "native_compact_prefill": False,
            "native_caware_decode": False,
            "throughput_claim_eligible": False,
            "scheduler_token_chunks": _batch_scheduler_token_chunks(
                request_ids,
                prompt_rows_by_request,
                generated_ids,
                output_parts,
                tokenizer=self._session.tokenizer if self._session is not None else None,
                ignore_eos=ignore_eos,
                stop_token_ids=(),
                stop_token_sequences=(),
                max_tokens=max_tokens,
                sampler_mode=sampler_mode,
                execution_path=parent_path,
                native_compact_prefill=False,
                native_caware_decode=False,
                serial_decode_fallback=True,
            ),
        }
        return outputs

    def _generate_batch_sampled_true_c1_fallback(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[PromptInput, ...],
        max_tokens: int,
        *,
        request: GenerationRequest,
        ignore_eos: bool,
        kv_policy,
        plan,
    ) -> list[GenerationOutput]:
        """Run sampled rows independently until packed prefill is c1-certified."""

        parent_path = "scheduler_sampled_true_c1_fallback"
        prompt_rows: list[list[int]] = []
        outputs: list[GenerationOutput] = []
        groups: list[dict[str, Any]] = []
        scheduler_chunks: list[dict[str, Any]] = []
        native_sampler_rows = plan.mode is SamplingMode.GPU_SAMPLE
        started_at = time.perf_counter()
        batch_id = _new_timing_batch_id("sampled-true-c1")
        for row_index, prompt in enumerate(prompts):
            prompt_ids = _prompt_ids(Path(self.model_path), prompt)
            prompt_row = [int(token) for token in prompt_ids]
            if not prompt_row:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append(prompt_row)
            group_started_at = time.perf_counter()
            chunks = list(
                self._stream_one_sampled(
                    runner,
                    prompt,
                    max_tokens,
                    request=request,
                    row_index=row_index,
                    ignore_eos=ignore_eos,
                    kv_policy=kv_policy,
                    plan=plan,
                    include_internal_token_logprobs=True,
                )
            )
            group_wall_s = time.perf_counter() - group_started_at
            if not chunks:
                raise RuntimeError("sampled true c1 fallback produced no token chunks")

            output_tokens: list[TokenLogprob] = []
            relabeled_chunks: list[GenerationStreamChunk] = []
            for token_index, chunk in enumerate(chunks):
                if len(chunk.token_logprobs) != 1:
                    raise RuntimeError("sampled true c1 fallback did not retain its token metadata")
                token = chunk.token_logprobs[0]
                output_tokens.append(token)
                telemetry = _relabel_isolated_group_telemetry(
                    chunk.telemetry,
                    row_index=row_index,
                    group_index=row_index,
                    group_width=1,
                    parent_path=parent_path,
                    native_compact_prefill=False,
                    native_caware_decode=False,
                    serial_decode_fallback=True,
                    native_sampler_rows=native_sampler_rows,
                )
                public_token_logprobs = (
                    chunk.token_logprobs
                    if request.logprobs or int(request.top_logprobs) > 0
                    else ()
                )
                relabeled_chunk = replace(
                    chunk,
                    token_logprobs=public_token_logprobs,
                    telemetry=telemetry,
                )
                relabeled_chunks.append(relabeled_chunk)
                scheduler_chunks.append(
                    _scheduler_token_chunk_payload(
                        row_index,
                        token_index,
                        token.token_id,
                        relabeled_chunk,
                    )
                )
            final_chunk = relabeled_chunks[-1]
            if final_chunk.finish_details is None:
                raise RuntimeError("sampled true c1 fallback ended without finish details")
            outputs.append(
                GenerationOutput(
                    text="".join(chunk.text for chunk in relabeled_chunks),
                    token_logprobs=tuple(output_tokens),
                    generated_token_ids=tuple(token.token_id for token in output_tokens),
                    finish_details=final_chunk.finish_details,
                    telemetry=final_chunk.telemetry,
                )
            )
            groups.append(
                {
                    "group_index": row_index,
                    "request_offset": row_index,
                    "planned_mode": "sampled_true_c1",
                    "width": 1,
                    "wall_ms": group_wall_s * 1000.0,
                }
            )

        total_wall_s = time.perf_counter() - started_at
        sampler_plan_metadata = [
            {
                "request_id": row_index,
                "mode": plan.mode.value,
                "active_processors": list(plan.active_processors),
                "sampler_fast_path_blockers": list(plan.fast_path_blockers),
                "native_gpu_available": bool(plan.native_gpu_available),
                "uses_host_logits": bool(plan.uses_host_logits),
                **(
                    {"sampler_fallback_reason": plan.fallback_reason}
                    if plan.fallback_reason is not None
                    else {}
                ),
            }
            for row_index in range(len(prompts))
        ]
        self.last_batch_generation = {
            "path": parent_path,
            "batch_id": batch_id,
            "group_rows": len(prompts),
            "timing_owner": True,
            "batch_size": len(prompts),
            "request_ids": list(range(len(prompts))),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "group_widths": [1] * len(prompts),
            "group_modes": ["sampled_true_c1"] * len(prompts),
            "max_session_width": 1,
            "groups": groups,
            "group_count": len(groups),
            "total_wall_ms": total_wall_s * 1000.0,
            "timing_scope": "batch",
            "batch_timing": {"batch_total_ms": total_wall_s * 1000.0},
            "route_blockers": [
                "sampled packed prefill and c>N decode are not certified against true c1"
            ],
            "sampler_plan_metadata": sampler_plan_metadata,
            "serial_decode_fallback": True,
            "native_compact_prefill": False,
            "native_caware_decode": False,
            "native_sampler_rows": native_sampler_rows,
            "throughput_claim_eligible": False,
            "scheduler_token_chunks": scheduler_chunks,
        }
        return outputs

    def _generate_batch_isolated_width_groups(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[PromptInput, ...],
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        sampler_mode: str,
        deadline_at: float | None,
        cancellation_token: Any | None,
        profile: NativeBatchWidthProfile,
    ) -> list[GenerationOutput]:
        """Complete c>N request groups without creating an over-width session."""

        prompt_rows: list[list[int]] = []
        for prompt in prompts:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            prompt_ids = _prompt_ids(Path(self.model_path), prompt)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append([int(token) for token in prompt_ids])

        start_positions = tuple(len(row) for row in prompt_rows)
        end_positions = tuple(
            len(row) + max(0, int(max_tokens) - 2)
            for row in prompt_rows
        )
        position_blockers = tuple(
            dict.fromkeys(
                (*profile.position_blockers(start_positions), *profile.position_blockers(end_positions))
            )
        )
        if profile.blockers or position_blockers:
            raise RuntimeError("isolated native width execution requires an accepted in-range profile")
        requested_plan = plan_batch_width_partition(
            len(prompts),
            profile=profile,
        )

        execution_groups: list[tuple[str, int]] = []
        for group in requested_plan.groups:
            if group.mode == "native":
                execution_groups.append(("native", int(group.width)))
            else:
                execution_groups.extend(("serial", 1) for _ in range(int(group.width)))
        if sum(width for _mode, width in execution_groups) != len(prompts):
            raise RuntimeError("isolated width plan did not cover every prompt")

        parent_path = "scheduler_isolated_width_groups"
        outputs: list[GenerationOutput] = []
        group_metadata: list[dict[str, Any]] = []
        scheduler_chunks: list[dict[str, Any]] = []
        cursor = 0
        started_at = time.perf_counter()
        batch_id = _new_timing_batch_id("isolated-width-groups")
        for group_index, (planned_mode, width) in enumerate(execution_groups):
            group_prompts = prompts[cursor : cursor + width]
            group_offset = cursor
            cursor += width
            group_started_at = time.perf_counter()
            if planned_mode == "native":
                group_outputs = self._generate_batch(
                    runner,
                    group_prompts,
                    max_tokens,
                    ignore_eos=ignore_eos,
                    kv_policy=kv_policy,
                    sampler_mode=sampler_mode,
                    deadline_at=deadline_at,
                    cancellation_token=cancellation_token,
                )
            else:
                group_outputs = self._generate_batch_true_c1_fallback(
                    runner,
                    group_prompts,
                    max_tokens,
                    ignore_eos=ignore_eos,
                    kv_policy=kv_policy,
                    sampler_mode=sampler_mode,
                    deadline_at=deadline_at,
                    cancellation_token=cancellation_token,
                    profile=profile,
                    route_blockers=("width partition selected a true-c1 serial remainder",),
                )
            group_wall_s = time.perf_counter() - group_started_at
            subgroup = dict(self.last_batch_generation or {})
            group_timing = (
                dict(group_outputs[0].telemetry.timing or {})
                if group_outputs and group_outputs[0].telemetry is not None
                else {}
            )
            group_metadata.append(
                {
                    "group_index": group_index,
                    "request_offset": group_offset,
                    "planned_mode": planned_mode,
                    "width": width,
                    "wall_ms": group_wall_s * 1000.0,
                    "timing": group_timing,
                    "execution_path": subgroup.get("path"),
                    "decode_partition_histogram": subgroup.get("decode_partition_histogram", {}),
                    "native_caware_decode": bool(subgroup.get("native_caware_decode", False)),
                    "serial_decode_fallback": bool(subgroup.get("serial_decode_fallback", False)),
                }
            )
            for local_index, output in enumerate(group_outputs):
                global_index = group_offset + local_index
                outputs.append(
                    _relabel_isolated_group_output(
                        output,
                        row_index=global_index,
                        group_index=group_index,
                        group_width=width,
                        parent_path=parent_path,
                    )
                )
            scheduler_chunks.extend(
                _relabel_isolated_scheduler_chunks(
                    subgroup.get("scheduler_token_chunks", ()),
                    request_offset=group_offset,
                    parent_path=parent_path,
                )
            )
        if cursor != len(prompts):
            raise RuntimeError("isolated width execution did not consume every prompt")

        total_wall_s = time.perf_counter() - started_at
        total_prefill_ms = sum(
            float(group["timing"].get("batch_prefill_ms", 0.0))
            for group in group_metadata
        )
        total_decode_ms = sum(
            float(group["timing"].get("batch_decode_ms", 0.0))
            for group in group_metadata
        )
        serial_fallback = any(
            group["planned_mode"] == "serial" or group["serial_decode_fallback"]
            for group in group_metadata
        )
        native_complete = bool(group_metadata) and all(
            group["planned_mode"] == "native" and group["native_caware_decode"]
            for group in group_metadata
        )
        self.last_batch_generation = {
            "path": parent_path,
            "batch_id": batch_id,
            "group_rows": len(prompts),
            "timing_owner": True,
            "batch_size": len(prompts),
            "request_ids": list(range(len(prompts))),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "group_widths": [width for _mode, width in execution_groups],
            "group_modes": [mode for mode, _width in execution_groups],
            "max_session_width": max(width for _mode, width in execution_groups),
            "requested_width_plan": requested_plan.to_json_dict(),
            "position_blockers": list(position_blockers),
            "groups": group_metadata,
            "group_count": len(group_metadata),
            "total_wall_ms": total_wall_s * 1000.0,
            "timing_scope": "batch",
            "batch_timing": {
                "batch_total_ms": total_wall_s * 1000.0,
                "batch_prefill_ms": total_prefill_ms,
                "batch_decode_ms": total_decode_ms,
                "batch_decode_steps": float(max(
                    (
                        int(group["timing"].get("batch_decode_steps", 0.0))
                        for group in group_metadata
                    ),
                    default=0,
                )),
                "batch_group_decode_steps": float(sum(
                    int(group["timing"].get("batch_decode_steps", 0.0))
                    for group in group_metadata
                )),
            },
            "native_width_profile": profile.to_json_dict(),
            "serial_decode_fallback": serial_fallback,
            "native_compact_prefill": all(
                bool(group.get("execution_path", "").startswith("scheduler_native_packed_prefill"))
                for group in group_metadata
            ),
            "native_caware_decode": native_complete,
            "throughput_claim_eligible": False,
            "scheduler_token_chunks": scheduler_chunks,
        }
        return outputs

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int:
        params = sampling_params
        resident_owner = self._resident_model_runner
        if resident_owner is not None:
            return resident_owner.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=params,
            )
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            getattr(params, "kv_storage", "auto"),
            scale_dtype=getattr(params, "kv_scale_dtype", "fp16"),
            scale_granularity=getattr(params, "kv_scale_granularity", "per_token_head"),
        )
        auto_context_length = max_sequence_length is None
        if auto_context_length:
            requested_length = int(getattr(runner.config, "max_position_embeddings", 0) or 0)
            if requested_length <= 0:
                requested_length = _session_capacity_for(1)
        else:
            if int(max_sequence_length) <= 0:
                raise ValueError("max_sequence_length must be positive")
            requested_length = int(max_sequence_length)
        session_capacity = _session_capacity_for(requested_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
            auto_context_length=auto_context_length,
        )
        return int(getattr(session, "max_sequence_length", self._session_capacity))

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        params = sampling_params
        resident_owner = self._resident_model_runner
        if resident_owner is not None:
            return resident_owner.prepare_request_scratch(
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                sampling_params=params,
                max_batch_size=max_batch_size,
                release_after_probe=release_after_probe,
            )
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            getattr(params, "kv_storage", "auto"),
            scale_dtype=getattr(params, "kv_scale_dtype", "fp16"),
            scale_granularity=getattr(params, "kv_scale_granularity", "per_token_head"),
        )
        required_sequence_length = max(1, int(max_prompt_tokens)) + max(0, int(max_new_tokens)) + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
            max_batch_size=max_batch_size,
        )
        return session.prepare_request_scratch(
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            max_batch_size=max_batch_size,
            release_after_probe=release_after_probe,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> tuple[int, ...]:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), str(text), None)
        return tuple(int(token) for token in prompt_ids)

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield chunk.text

    def stream_detailed(self, request: GenerationRequest) -> Iterator[GenerationStreamChunk]:
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        native_gpu_available = _native_gpu_sampler_route_available(prompt_count=1)
        plan = plan_sampler(
            request,
            native_gpu_available=native_gpu_available,
            native_gpu_requested=_native_gpu_sampler_requested(),
        )
        if request.max_tokens == 0:
            return
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        if plan.mode is SamplingMode.GREEDY_FAST:
            yield from self._stream_one(
                runner,
                request.prompts[0],
                request.max_tokens,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
                deadline_at=request.deadline_at,
                cancellation_token=request.cancellation_token,
            )
            return
        yield from self._stream_one_sampled(
            runner,
            request.prompts[0],
            request.max_tokens,
            request=request,
            row_index=0,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
            plan=plan,
        )

    def _generate_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: PromptInput,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        sampler_mode: str,
        deadline_at: float | None,
        cancellation_token: Any | None,
    ) -> GenerationOutput:
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        prompt_ids = _prompt_ids(Path(self.model_path), prompt)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        generated_text: list[str] = []
        generated_token_ids: list[int] = []
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        next_result = _prefill_prompt(session, prompt_ids, sample=True)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        generated_text.append(next_result.token_text)
        generated_token_ids.append(int(next_result.token_id))
        if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
            return GenerationOutput(
                text="".join(generated_text),
                generated_token_ids=generated_token_ids,
                finish_details=_finish_details_for_tokens(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=(),
                    stop_token_sequences=(),
                    max_tokens=max_tokens,
                    sampler_mode=sampler_mode,
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=0,
                    sampler_mode=sampler_mode,
                    stop_token_sequences=(),
                    diagnostics={"generated_token_ids": list(generated_token_ids)},
                ),
            )

        remaining = max_tokens - 1
        if remaining:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            graph_policy = getattr(session, "greedy_decode_graph_eligible", None)
            graph_eligible = True if not callable(graph_policy) else bool(graph_policy())
            if graph_eligible:
                with session.capture_decode_graph(
                    position=len(prompt_ids),
                    steps_per_replay=1,
                    max_replay_steps=remaining,
                    record_steps=remaining,
                ) as graph:
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    graph.replay(remaining)
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    token_ids = graph.read_generated_token_ids(remaining)
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                for token_id in token_ids:
                    generated_text.append(_decode_token_cached(session.tokenizer, token_id))
                    generated_token_ids.append(int(token_id))
                    if not ignore_eos and _is_eos(session.tokenizer, token_id):
                        break
            else:
                current_result = next_result
                for offset in range(remaining):
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    current_result = session.step(
                        current_result.token_id,
                        position=len(prompt_ids) + offset,
                        sample=True,
                    )
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    if current_result is None:
                        raise RuntimeError("eager decode did not produce next-token logits")
                    generated_text.append(current_result.token_text)
                    generated_token_ids.append(int(current_result.token_id))
                    if not ignore_eos and _is_eos(session.tokenizer, current_result.token_id):
                        break
        return GenerationOutput(
            text="".join(generated_text),
            generated_token_ids=generated_token_ids,
            finish_details=_finish_details_for_tokens(
                session.tokenizer,
                generated_token_ids,
                ignore_eos=ignore_eos,
                stop_token_ids=(),
                stop_token_sequences=(),
                max_tokens=max_tokens,
                sampler_mode=sampler_mode,
            ),
            telemetry=_telemetry_for_tokens(
                prompt_ids,
                generated_token_ids,
                row_index=0,
                sampler_mode=sampler_mode,
                stop_token_sequences=(),
                diagnostics={"generated_token_ids": list(generated_token_ids)},
            ),
        )

    def _generate_one_sampled(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: PromptInput,
        max_tokens: int,
        *,
        request: GenerationRequest,
        row_index: int,
        ignore_eos: bool,
        kv_policy,
        plan,
    ) -> GenerationOutput:
        raise_if_generation_deadline_expired(request)
        prompt_ids = _prompt_ids(Path(self.model_path), prompt)
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        sampling_request = _request_with_tokenizer_eos(request, session.tokenizer)
        state = _row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        _configure_sampled_session(session, sampling_request, state, plan=plan)
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            plan,
            vocab_size=getattr(session, "vocab_size", None),
        )
        generated_text: list[str] = []
        generated_token_ids: list[int] = []
        generated_steps: list[Qwen35ParoAutoregressiveStepResult] = []
        try:
            raise_if_generation_deadline_expired(request)
            next_result = _prefill_prompt(session, prompt_ids, sample=True)
            raise_if_generation_deadline_expired(request)
            if next_result is None:
                raise RuntimeError("native prefill did not produce next-token logits")
            generated_text.append(next_result.token_text)
            generated_token_ids.append(int(next_result.token_id))
            generated_steps.append(next_result)
            _queue_json_object_close_if_needed(
                state,
                session.tokenizer,
                next_result.token_text,
                remaining_tokens=max_tokens - len(generated_token_ids),
            )
            if _is_finished(
                session.tokenizer,
                generated_token_ids,
                ignore_eos=ignore_eos,
                stop_token_ids=request.stop_token_ids,
                stop_token_sequences=request.stop_token_sequences,
            ):
                return _generation_output_from_steps(
                    session.tokenizer,
                    generated_steps,
                    finish_details=_finish_details_for_tokens(
                        session.tokenizer,
                        generated_token_ids,
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=max_tokens,
                        sampler_mode=plan.mode.value,
                        sampling_state=state,
                    ),
                    telemetry=_telemetry_for_tokens(
                        prompt_ids,
                        generated_token_ids,
                        row_index=row_index,
                        sampler_mode=plan.mode.value,
                        stop_token_sequences=request.stop_token_sequences,
                        active_processors=plan.active_processors,
                        sampler_fast_path_blockers=plan.fast_path_blockers,
                        sampler_fallback_reason=plan.fallback_reason,
                        sampling_state=state,
                        forced_sample=next_result,
                        full_vocab_logits_d2h=full_vocab_logits_d2h,
                        logits_d2h_bytes=logits_d2h_bytes,
                    ),
                )

            current_token_id = int(next_result.token_id)
            for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
                raise_if_generation_deadline_expired(request)
                result = session.step(current_token_id, position=position, sample=True)
                raise_if_generation_deadline_expired(request)
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                generated_text.append(result.token_text)
                generated_token_ids.append(int(result.token_id))
                generated_steps.append(result)
                _queue_json_object_close_if_needed(
                    state,
                    session.tokenizer,
                    result.token_text,
                    remaining_tokens=max_tokens - len(generated_token_ids),
                )
                current_token_id = int(result.token_id)
                if _is_finished(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=request.stop_token_ids,
                    stop_token_sequences=request.stop_token_sequences,
                ):
                    break
            return _generation_output_from_steps(
                session.tokenizer,
                generated_steps,
                finish_details=_finish_details_for_tokens(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=request.stop_token_ids,
                    stop_token_sequences=request.stop_token_sequences,
                    max_tokens=max_tokens,
                    sampler_mode=plan.mode.value,
                    sampling_state=state,
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=row_index,
                    sampler_mode=plan.mode.value,
                    stop_token_sequences=request.stop_token_sequences,
                    active_processors=plan.active_processors,
                    sampler_fast_path_blockers=plan.fast_path_blockers,
                    sampler_fallback_reason=plan.fallback_reason,
                    sampling_state=state,
                    forced_sample=generated_steps[-1] if generated_steps else None,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                ),
            )
        finally:
            _configure_sampled_session(session, None, None, plan=plan)

    def _generate_batch(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[PromptInput, ...],
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        sampler_mode: str,
        deadline_at: float | None,
        cancellation_token: Any | None,
        exact_hybrid_c2: bool = False,
    ) -> list[GenerationOutput]:
        """Generate a prompt list through the scheduler-owned c>N path.

        Native compact prefill runs all admitted rows together when their block
        table shapes permit it. Decode uses only identity-matched native widths;
        unsupported live widths are covered by native subgroups plus an exact
        serial remainder. ``last_batch_generation`` records the effective route.
        """

        prompt_rows: list[list[int]] = []
        for prompt in prompts:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            prompt_ids = _prompt_ids(Path(self.model_path), prompt)
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append([int(token) for token in prompt_ids])
        batch_size = len(prompt_rows)
        if exact_hybrid_c2 and batch_size != 2:
            raise ValueError("exact PARO hybrid decode requires batch_size=2")
        required_sequence_length = max(len(row) for row in prompt_rows) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            max_batch_size=batch_size,
            kv_policy=kv_policy,
        )
        scheduler = ResidentBatchScheduler(capacity=batch_size)
        request_ids = tuple(
            scheduler.submit(row, max_new_tokens=max(0, max_tokens - 1))
            for row in prompt_rows
        )
        prompt_rows_by_request = dict(zip(request_ids, prompt_rows, strict=True))
        admitted = scheduler.admit_pending()
        if admitted != request_ids:
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")

        output_parts: dict[int, list[str]] = {request_id: [] for request_id in request_ids}
        generated_ids: dict[int, list[int]] = {request_id: [] for request_id in request_ids}
        next_token_by_request: dict[int, int] = {}
        batch_started_at = time.perf_counter()
        batch_id = _new_timing_batch_id("decode")
        prefill_wall_s = 0.0
        decode_wall_s = 0.0
        packed_slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(len(row) for row in prompt_rows),
            block_size=getattr(session, "block_size", 256),
        )
        prefill_slab_shapes: list[dict[str, Any]] = []
        for slab in packed_slabs:
            prefill_slab_shapes.append(
                {
                    "request_ids": list(slab.request_ids),
                    "slot_ids": list(slab.physical_slot_ids),
                    "rows": slab.rows,
                    "request_count": slab.request_count,
                    "block_count": slab.block_count,
                }
            )
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            prefill_started_at = time.perf_counter()
            results = session.prefill_native_packed(slab, sample=True)
            prefill_wall_s += time.perf_counter() - prefill_started_at
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            if len(results) != slab.request_count:
                raise RuntimeError(
                    "packed prefill returned "
                    f"{len(results)} results for {slab.request_count} requests"
                )
            for request_id, result in zip(slab.request_ids, results, strict=True):
                if result is None:
                    raise RuntimeError("packed native prefill did not produce next-token logits")
                output_parts[request_id].append(result.token_text)
                generated_ids[request_id].append(int(result.token_id))
                seed_finished = (
                    not ignore_eos and _is_eos(session.tokenizer, result.token_id)
                ) or max_tokens <= 1
                if seed_finished:
                    scheduler.record_generated(
                        (GeneratedToken(request_id, result.token_id, finished=True),)
                    )
                else:
                    next_token_by_request[request_id] = int(result.token_id)

        decode_steps = 0
        native_decode_steps = 0
        native_decode_group_calls = 0
        exact_hybrid_decode_steps = 0
        exact_hybrid_decode_group_calls = 0
        serial_decode_row_calls = 0
        partitioned_decode_steps = 0
        decode_partition_histogram: Counter[str] = Counter()
        native_width_profile_payload: dict[str, Any] | None = None
        serial_decode_fallback = False
        while next_token_by_request:
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            work = scheduler.next_decode_work()
            if work is None:
                raise RuntimeError("scheduler did not emit decode work")
            request_ids_for_step = tuple(
                request_id for request_id in work.request_ids if request_id in next_token_by_request
            )
            if not request_ids_for_step:
                raise RuntimeError("scheduler decode work did not include runnable requests")
            token_ids_for_step = [next_token_by_request[request_id] for request_id in request_ids_for_step]
            positions_for_step = [
                scheduler.active_batch.requests[request_id].context_len
                for request_id in request_ids_for_step
            ]
            slots_for_step = [
                scheduler.active_batch.slot_for(request_id)
                for request_id in request_ids_for_step
            ]
            sorted_step_rows = sorted(
                zip(
                    slots_for_step,
                    request_ids_for_step,
                    token_ids_for_step,
                    positions_for_step,
                    strict=True,
                ),
                key=lambda item: item[0],
            )
            profile_provider = getattr(session, "native_batch_width_profile", None)
            profile = profile_provider() if callable(profile_provider) else None
            if profile is not None and not isinstance(profile, NativeBatchWidthProfile):
                raise TypeError("native_batch_width_profile() must return NativeBatchWidthProfile or None")
            if profile is not None:
                native_width_profile_payload = profile.to_json_dict()
            if exact_hybrid_c2 and len(sorted_step_rows) == 2:
                decode_groups = (BatchWidthGroup("native", 2, 1.0),)
            else:
                decode_groups = plan_batch_width_partition(
                    len(sorted_step_rows),
                    profile=profile if hasattr(session, "step_batch_native") else None,
                    positions=tuple(int(item[3]) for item in sorted_step_rows),
                ).groups
            decode_started_at = time.perf_counter()
            result_by_request: dict[int, Qwen35ParoAutoregressiveStepResult | None] = {}
            effective_groups: list[tuple[str, int]] = []
            cursor = 0
            step_native_rows = 0
            step_exact_hybrid_rows = 0
            step_serial_rows = 0
            for group in decode_groups:
                group_rows = sorted_step_rows[cursor : cursor + group.width]
                cursor += group.width
                group_slots = [int(item[0]) for item in group_rows]
                group_request_ids = [int(item[1]) for item in group_rows]
                group_token_ids = [int(item[2]) for item in group_rows]
                group_positions = [int(item[3]) for item in group_rows]
                results: tuple[Qwen35ParoAutoregressiveStepResult | None, ...]
                if group.mode == "native":
                    use_exact_hybrid = exact_hybrid_c2 and group.width == 2
                    try:
                        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                        if use_exact_hybrid:
                            results = session.step_batch_native(
                                group_token_ids,
                                positions=group_positions,
                                slots=group_slots,
                                sample=True,
                                exact_hybrid=True,
                            )
                        else:
                            results = session.step_batch_native(
                                group_token_ids,
                                positions=group_positions,
                                slots=group_slots,
                                sample=True,
                            )
                        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                        if use_exact_hybrid:
                            exact_hybrid_decode_group_calls += 1
                            step_exact_hybrid_rows += group.width
                            effective_groups.append(("exact_hybrid", group.width))
                        else:
                            native_decode_group_calls += 1
                            step_native_rows += group.width
                            effective_groups.append(("native", group.width))
                    except NotImplementedError:
                        serial_decode_fallback = True
                        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                        results = session.step_batch_serial(
                            group_token_ids,
                            positions=group_positions,
                            slots=group_slots,
                            sample=True,
                        )
                        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                        serial_decode_row_calls += group.width
                        step_serial_rows += group.width
                        effective_groups.append(("serial", group.width))
                else:
                    serial_decode_fallback = True
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    results = session.step_batch_serial(
                        group_token_ids,
                        positions=group_positions,
                        slots=group_slots,
                        sample=True,
                    )
                    raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
                    serial_decode_row_calls += group.width
                    step_serial_rows += group.width
                    effective_groups.append(("serial", group.width))
                if len(results) != len(group_request_ids):
                    raise RuntimeError(
                        f"decode group returned {len(results)} results for {len(group_request_ids)} requests"
                    )
                result_by_request.update(zip(group_request_ids, results, strict=True))
            if cursor != len(sorted_step_rows):
                raise RuntimeError("decode width partition did not consume every live row")
            results = tuple(result_by_request[request_id] for request_id in request_ids_for_step)
            if step_native_rows and step_serial_rows == 0:
                native_decode_steps += 1
            if step_exact_hybrid_rows and step_serial_rows == 0:
                exact_hybrid_decode_steps += 1
            if len(effective_groups) > 1:
                partitioned_decode_steps += 1
            signature = "+".join(f"{mode}:{width}" for mode, width in effective_groups)
            decode_partition_histogram[signature] += 1
            decode_wall_s += time.perf_counter() - decode_started_at
            generated: list[GeneratedToken] = []
            for request_id, result in zip(request_ids_for_step, results, strict=True):
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                output_parts[request_id].append(result.token_text)
                generated_ids[request_id].append(int(result.token_id))
                next_token_by_request[request_id] = int(result.token_id)
                finished = not ignore_eos and _is_eos(session.tokenizer, result.token_id)
                generated.append(GeneratedToken(request_id, result.token_id, finished=finished))
            completed = scheduler.record_generated(generated)
            for done in completed:
                next_token_by_request.pop(done.request_id, None)
            decode_steps += 1

        native_decode_complete = decode_steps > 0 and native_decode_steps == decode_steps and not serial_decode_fallback
        exact_hybrid_decode_complete = (
            exact_hybrid_c2
            and decode_steps > 0
            and exact_hybrid_decode_steps == decode_steps
            and not serial_decode_fallback
        )
        batch_execution = session.batch_execution_metadata(
            scheduler_owned=True,
            native_decode=native_decode_complete or exact_hybrid_decode_complete,
        )
        batch_execution_payload = (
            batch_execution.to_json_dict()
            if callable(getattr(batch_execution, "to_json_dict", None))
            else None
        )
        total_wall_s = time.perf_counter() - batch_started_at
        batch_timing = {
            "batch_total_ms": total_wall_s * 1000.0,
            "batch_prefill_ms": prefill_wall_s * 1000.0,
            "batch_decode_ms": decode_wall_s * 1000.0,
            "batch_decode_step_ms_avg": (decode_wall_s * 1000.0 / decode_steps) if decode_steps else 0.0,
            "batch_decode_steps": float(decode_steps),
            "batch_native_decode_steps": float(native_decode_steps),
            "batch_native_decode_group_calls": float(native_decode_group_calls),
            "batch_exact_hybrid_decode_steps": float(exact_hybrid_decode_steps),
            "batch_exact_hybrid_decode_group_calls": float(exact_hybrid_decode_group_calls),
            "batch_serial_decode_rows": float(serial_decode_row_calls),
        }
        if exact_hybrid_c2 and partitioned_decode_steps:
            execution_path = "scheduler_native_packed_prefill_exact_hybrid_partitioned_decode"
        elif exact_hybrid_decode_complete:
            execution_path = "scheduler_native_packed_prefill_exact_hybrid_decode"
        elif partitioned_decode_steps:
            execution_path = "scheduler_native_packed_prefill_partitioned_decode"
        elif native_decode_complete:
            execution_path = "scheduler_native_packed_prefill_native_decode"
        else:
            execution_path = "scheduler_native_packed_prefill_serial_decode"
        self.last_batch_generation = {
            "path": execution_path,
            "batch_id": batch_id,
            "group_rows": batch_size,
            "timing_scope": "batch",
            "timing_owner": True,
            "batch_size": batch_size,
            "request_ids": list(request_ids),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "packed_prefill_slabs": prefill_slab_shapes,
            "decode_steps": decode_steps,
            "native_decode_steps": native_decode_steps,
            "native_decode_group_calls": native_decode_group_calls,
            "serial_decode_row_calls": serial_decode_row_calls,
            "partitioned_decode_steps": partitioned_decode_steps,
            "decode_partition_histogram": dict(sorted(decode_partition_histogram.items())),
            "native_width_profile": native_width_profile_payload,
            "serial_decode_fallback": serial_decode_fallback,
            "native_compact_prefill": bool(
                getattr(batch_execution, "native_compact_prefill", False)
            ),
            "native_caware_decode": bool(getattr(batch_execution, "native_caware_decode", False)),
            "throughput_claim_eligible": bool(
                getattr(batch_execution, "throughput_claim_eligible", False)
            ),
            "batch_execution": batch_execution_payload,
        }
        if exact_hybrid_c2:
            self.last_batch_generation.update(
                {
                    "exact_hybrid_c2": True,
                    "exact_hybrid_decode_steps": exact_hybrid_decode_steps,
                    "exact_hybrid_decode_group_calls": exact_hybrid_decode_group_calls,
                }
            )
        self.last_batch_generation["scheduler_token_chunks"] = _batch_scheduler_token_chunks(
            request_ids,
            prompt_rows_by_request,
            generated_ids,
            output_parts,
            tokenizer=session.tokenizer,
            ignore_eos=ignore_eos,
            stop_token_ids=(),
            stop_token_sequences=(),
            max_tokens=max_tokens,
            sampler_mode=sampler_mode,
            execution_path=self.last_batch_generation["path"],
            native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
            native_caware_decode=self.last_batch_generation["native_caware_decode"],
            serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
        )
        return [
            GenerationOutput(
                text="".join(output_parts[request_id]),
                generated_token_ids=generated_ids[request_id],
                finish_details=_finish_details_for_tokens(
                    session.tokenizer,
                    generated_ids[request_id],
                    ignore_eos=ignore_eos,
                    stop_token_ids=(),
                    stop_token_sequences=(),
                    max_tokens=max_tokens,
                    sampler_mode=sampler_mode,
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_rows_by_request[request_id],
                    generated_ids[request_id],
                    row_index=request_id,
                    request_id=str(request_id),
                    sampler_mode=sampler_mode,
                    stop_token_sequences=(),
                    execution_path=self.last_batch_generation["path"],
                    native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
                    native_caware_decode=self.last_batch_generation["native_caware_decode"],
                    serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
                    timing=batch_timing,
                    timing_scope="batch",
                    batch_id=batch_id,
                    group_rows=batch_size,
                    timing_owner=request_id == request_ids[0],
                    diagnostics={"batch_execution": batch_execution_payload}
                    if batch_execution_payload is not None
                    else None,
                ),
            )
            for request_id in request_ids
        ]

    def _generate_batch_sampled(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompts: tuple[PromptInput, ...],
        max_tokens: int,
        *,
        request: GenerationRequest,
        ignore_eos: bool,
        kv_policy,
    ) -> list[GenerationOutput]:
        """Generate a sampled prompt list through scheduler-owned c>N state.

        Native packed prefill handles the prompt rows together, while decode uses
        the explicit serial slot bridge with per-slot host sampler state clones.
        The scheduler remains the owner of persistent row history.
        """

        prompt_rows: list[list[int]] = []
        for prompt in prompts:
            raise_if_generation_deadline_expired(request)
            prompt_ids = _prompt_ids(Path(self.model_path), prompt)
            raise_if_generation_deadline_expired(request)
            if not prompt_ids:
                raise ValueError("prompt produced no tokens")
            prompt_rows.append([int(token) for token in prompt_ids])
        batch_size = len(prompt_rows)
        required_sequence_length = max(len(row) for row in prompt_rows) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            max_batch_size=batch_size,
            kv_policy=kv_policy,
        )
        sampling_request = _request_with_tokenizer_eos(request, session.tokenizer)
        scheduler = ResidentBatchScheduler(capacity=batch_size)
        sampling = _per_row_sampling_params(sampling_request)
        request_ids = tuple(
            scheduler.submit(
                row,
                max_new_tokens=max(0, max_tokens - 1),
                sampling=sampling,
                sampling_row_index=index,
            )
            for index, row in enumerate(prompt_rows)
        )
        prompt_rows_by_request = dict(zip(request_ids, prompt_rows, strict=True))
        admitted = scheduler.admit_pending()
        if admitted != request_ids:
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
        native_sampler_requested = _native_gpu_sampler_requested()
        configure_native_rows = getattr(session, "configure_native_sampler_rows", None)
        native_sampler_rows_available = native_sampler_requested and callable(configure_native_rows)
        sampler_block = scheduler.sampler_params_block(request_ids)
        sampler_plans = dict(
            zip(
                request_ids,
                sampler_block.sampler_plans(
                    native_gpu_available=native_sampler_rows_available,
                    native_gpu_requested=native_sampler_requested,
                ),
                strict=True,
            )
        )
        sampler_plan_metadata = sampler_block.sampler_plan_metadata(
            native_gpu_available=native_sampler_rows_available,
            native_gpu_requested=native_sampler_requested
        )
        use_native_sampler_rows = native_sampler_rows_available and all(
            plan.mode is SamplingMode.GPU_SAMPLE for plan in sampler_plans.values()
        )

        output_steps: dict[int, list[Qwen35ParoAutoregressiveStepResult]] = {request_id: [] for request_id in request_ids}
        generated_ids: dict[int, list[int]] = {request_id: [] for request_id in request_ids}
        sampling_state_snapshots: dict[int, RowSamplingState] = {
            request_id: _clone_row_sampling_state(scheduler.sampler_state(request_id))
            for request_id in request_ids
        }
        sampling_state_step_snapshots: dict[int, list[RowSamplingState]] = {
            request_id: [] for request_id in request_ids
        }
        next_token_by_request: dict[int, int] = {}
        packed_slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(len(row) for row in prompt_rows),
            block_size=getattr(session, "block_size", 256),
        )
        prefill_slab_shapes: list[dict[str, Any]] = []
        if use_native_sampler_rows:
            configure_rows = configure_native_rows
            sampled_path = "scheduler_native_packed_prefill_serial_native_sampler_decode"
        else:
            configure_rows = getattr(session, "configure_host_sampler_rows", None)
            sampled_path = "scheduler_native_packed_prefill_serial_host_sampler_decode"
        if not callable(configure_rows):
            raise NotImplementedError("c>N sampled PARO batches require per-slot host sampler state")
        try:
            for slab in packed_slabs:
                prefill_slab_shapes.append(
                    {
                        "request_ids": list(slab.request_ids),
                        "slot_ids": list(slab.physical_slot_ids),
                        "rows": slab.rows,
                        "request_count": slab.request_count,
                        "block_count": slab.block_count,
                    }
                )
                configure_rows(sampling_request, _slot_sampler_state_clones(scheduler, slab.request_ids, slab.physical_slot_ids))
                raise_if_generation_deadline_expired(request)
                results = session.prefill_native_packed(slab, sample=True)
                raise_if_generation_deadline_expired(request)
                if len(results) != slab.request_count:
                    raise RuntimeError(
                        "packed prefill returned "
                        f"{len(results)} results for {slab.request_count} requests"
                    )
                generated: list[GeneratedToken] = []
                for request_id, result in zip(slab.request_ids, results, strict=True):
                    if result is None:
                        raise RuntimeError("packed native prefill did not produce next-token logits")
                    output_steps[request_id].append(result)
                    generated_ids[request_id].append(int(result.token_id))
                    snapshot = _clone_row_sampling_state(scheduler.sampler_state(request_id))
                    snapshot.observe(result.token_id)
                    sampling_state_snapshots[request_id] = snapshot
                    finished = max_tokens <= 1 or _is_finished(
                        session.tokenizer,
                        generated_ids[request_id],
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                    )
                    if finished:
                        sampling_state_step_snapshots[request_id].append(snapshot)
                        generated.append(GeneratedToken(request_id, result.token_id, finished=True))
                    else:
                        owner_state = scheduler.sampler_state(request_id)
                        owner_state.observe(result.token_id)
                        _queue_json_object_close_if_needed(
                            owner_state,
                            session.tokenizer,
                            result.token_text,
                            remaining_tokens=max_tokens - len(generated_ids[request_id]),
                        )
                        sampling_state_snapshots[request_id] = _clone_row_sampling_state(owner_state)
                        sampling_state_step_snapshots[request_id].append(sampling_state_snapshots[request_id])
                        next_token_by_request[request_id] = int(result.token_id)
                if generated:
                    completed_ids = {done.request_id for done in scheduler.record_generated(generated)}
                    for done in completed_ids:
                        next_token_by_request.pop(done, None)

            decode_steps = 0
            serial_decode_fallback = False
            while next_token_by_request:
                raise_if_generation_deadline_expired(request)
                work = scheduler.next_decode_work()
                if work is None:
                    raise RuntimeError("scheduler did not emit decode work")
                request_ids_for_step = tuple(
                    request_id for request_id in work.request_ids if request_id in next_token_by_request
                )
                if not request_ids_for_step:
                    raise RuntimeError("scheduler decode work did not include runnable requests")
                token_ids_for_step = [next_token_by_request[request_id] for request_id in request_ids_for_step]
                positions_for_step = [
                    scheduler.active_batch.requests[request_id].context_len
                    for request_id in request_ids_for_step
                ]
                slots_for_step = [scheduler.active_batch.slot_for(request_id) for request_id in request_ids_for_step]
                configure_rows(
                    sampling_request,
                    _slot_sampler_state_clones(scheduler, request_ids_for_step, slots_for_step),
                )
                raise_if_generation_deadline_expired(request)
                results = session.step_batch_serial(
                    token_ids_for_step,
                    positions=positions_for_step,
                    slots=slots_for_step,
                    sample=True,
                )
                raise_if_generation_deadline_expired(request)
                serial_decode_fallback = serial_decode_fallback or len(slots_for_step) > 1
                generated = []
                decode_results_by_request: dict[int, Qwen35ParoAutoregressiveStepResult] = {}
                for request_id, result in zip(request_ids_for_step, results, strict=True):
                    if result is None:
                        raise RuntimeError("decode step did not produce next-token logits")
                    output_steps[request_id].append(result)
                    generated_ids[request_id].append(int(result.token_id))
                    snapshot = _clone_row_sampling_state(scheduler.sampler_state(request_id))
                    snapshot.observe(result.token_id)
                    sampling_state_snapshots[request_id] = snapshot
                    finished = _is_finished(
                        session.tokenizer,
                        generated_ids[request_id],
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                    )
                    generated.append(GeneratedToken(request_id, result.token_id, finished=finished))
                    decode_results_by_request[int(request_id)] = result
                completed = scheduler.record_generated(generated)
                completed_ids = {done.request_id for done in completed}
                for done in completed_ids:
                    next_token_by_request.pop(done, None)
                for request_id, result in decode_results_by_request.items():
                    if request_id in completed_ids:
                        sampling_state_step_snapshots[request_id].append(sampling_state_snapshots[request_id])
                        continue
                    owner_state = scheduler.sampler_state(request_id)
                    _queue_json_object_close_if_needed(
                        owner_state,
                        session.tokenizer,
                        result.token_text,
                        remaining_tokens=max_tokens - len(generated_ids[request_id]),
                    )
                    sampling_state_snapshots[request_id] = _clone_row_sampling_state(owner_state)
                    sampling_state_step_snapshots[request_id].append(sampling_state_snapshots[request_id])
                    next_token_by_request[request_id] = int(result.token_id)
                decode_steps += 1
        finally:
            configure_rows(None, None)

        batch_execution = session.batch_execution_metadata(scheduler_owned=True, native_decode=False)
        batch_execution_payload = (
            batch_execution.to_json_dict()
            if callable(getattr(batch_execution, "to_json_dict", None))
            else None
        )
        self.last_batch_generation = {
            "path": sampled_path,
            "batch_size": batch_size,
            "request_ids": list(request_ids),
            "prompt_lengths": [len(row) for row in prompt_rows],
            "packed_prefill_slabs": prefill_slab_shapes,
            "decode_steps": decode_steps,
            "native_decode_steps": 0,
            "serial_decode_fallback": serial_decode_fallback,
            "native_compact_prefill": bool(getattr(batch_execution, "native_compact_prefill", False)),
            "native_caware_decode": False,
            "native_sampler_rows": use_native_sampler_rows,
            "throughput_claim_eligible": False,
            "batch_execution": batch_execution_payload,
            "sampler_plan_metadata": [dict(row) for row in sampler_plan_metadata],
        }
        self.last_batch_generation["scheduler_token_chunks"] = _sampled_batch_scheduler_token_chunks(
            request_ids,
            prompt_rows_by_request,
            output_steps,
            sampling_state_step_snapshots,
            tokenizer=session.tokenizer,
            vocab_size=getattr(session, "vocab_size", None),
            request=sampling_request,
            plans=sampler_plans,
            execution_path=self.last_batch_generation["path"],
            native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
            native_caware_decode=self.last_batch_generation["native_caware_decode"],
            serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
            native_sampler_rows=self.last_batch_generation["native_sampler_rows"],
        )
        outputs: list[GenerationOutput] = []
        for request_id in request_ids:
            plan = sampler_plans[request_id]
            sampler_mode = plan.mode.value
            full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
                plan,
                vocab_size=getattr(session, "vocab_size", None),
            )
            outputs.append(
                _generation_output_from_steps(
                    session.tokenizer,
                    output_steps[request_id],
                    finish_details=_finish_details_for_tokens(
                        session.tokenizer,
                        generated_ids[request_id],
                        ignore_eos=ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=request.max_tokens,
                        sampler_mode=sampler_mode,
                        sampling_state=sampling_state_snapshots.get(request_id),
                    ),
                    telemetry=_telemetry_for_tokens(
                        prompt_rows_by_request[request_id],
                        generated_ids[request_id],
                        row_index=request_id,
                        request_id=str(request_id),
                        sampler_mode=sampler_mode,
                        stop_token_sequences=request.stop_token_sequences,
                        active_processors=plan.active_processors,
                        sampler_fast_path_blockers=plan.fast_path_blockers,
                        sampler_fallback_reason=plan.fallback_reason,
                        sampling_state=sampling_state_snapshots.get(request_id),
                        forced_sample=output_steps[request_id][-1] if output_steps[request_id] else None,
                        full_vocab_logits_d2h=full_vocab_logits_d2h,
                        logits_d2h_bytes=logits_d2h_bytes,
                        execution_path=self.last_batch_generation["path"],
                        native_compact_prefill=self.last_batch_generation["native_compact_prefill"],
                        native_caware_decode=self.last_batch_generation["native_caware_decode"],
                        serial_decode_fallback=self.last_batch_generation["serial_decode_fallback"],
                        native_sampler_rows=self.last_batch_generation["native_sampler_rows"],
                        diagnostics={"batch_execution": batch_execution_payload}
                        if batch_execution_payload is not None
                        else None,
                    ),
                )
            )
        return outputs

    def _stream_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: PromptInput,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
        deadline_at: float | None,
        cancellation_token: Any | None,
    ) -> Iterator[GenerationStreamChunk]:
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        prompt_ids = _prompt_ids(Path(self.model_path), prompt)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        next_result = _prefill_prompt(session, prompt_ids, sample=True)
        raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        generated_token_ids = [int(next_result.token_id)]
        finished = not ignore_eos and _is_eos(session.tokenizer, next_result.token_id)
        yield GenerationStreamChunk(
            next_result.token_text,
            finish_details=(
                _finish_details_for_tokens(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=(),
                    stop_token_sequences=(),
                    max_tokens=max_tokens,
                    sampler_mode=SamplingMode.GREEDY_FAST.value,
                )
                if finished or len(generated_token_ids) >= max_tokens
                else None
            ),
            telemetry=_telemetry_for_tokens(
                prompt_ids,
                generated_token_ids,
                row_index=0,
                sampler_mode=SamplingMode.GREEDY_FAST.value,
                phase="answer",
                stop_token_sequences=(),
            ),
        )
        if finished:
            return

        current_token_id = next_result.token_id
        for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            result = session.step(current_token_id, position=position, sample=True)
            raise_if_generation_deadline_expired(deadline_at, cancellation_token=cancellation_token)
            if result is None:
                raise RuntimeError("decode step did not produce next-token logits")
            generated_token_ids.append(int(result.token_id))
            finished = not ignore_eos and _is_eos(session.tokenizer, result.token_id)
            yield GenerationStreamChunk(
                result.token_text,
                finish_details=(
                    _finish_details_for_tokens(
                        session.tokenizer,
                        generated_token_ids,
                        ignore_eos=ignore_eos,
                        stop_token_ids=(),
                        stop_token_sequences=(),
                        max_tokens=max_tokens,
                        sampler_mode=SamplingMode.GREEDY_FAST.value,
                    )
                    if finished or len(generated_token_ids) >= max_tokens
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=0,
                    sampler_mode=SamplingMode.GREEDY_FAST.value,
                    phase="answer",
                    stop_token_sequences=(),
                ),
            )
            current_token_id = result.token_id
            if finished:
                return

    def _stream_one_sampled(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: PromptInput,
        max_tokens: int,
        *,
        request: GenerationRequest,
        row_index: int,
        ignore_eos: bool,
        kv_policy,
        plan,
        include_internal_token_logprobs: bool = False,
    ) -> Iterator[GenerationStreamChunk]:
        raise_if_generation_deadline_expired(request)
        prompt_ids = _prompt_ids(Path(self.model_path), prompt)
        raise_if_generation_deadline_expired(request)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        sampling_request = _request_with_tokenizer_eos(request, session.tokenizer)
        state = _row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
        _configure_sampled_session(session, sampling_request, state, plan=plan)
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            plan,
            vocab_size=getattr(session, "vocab_size", None),
        )
        generated_token_ids: list[int] = []
        live_phase = None if state.thinking_budget is not None else "answer"
        try:
            raise_if_generation_deadline_expired(request)
            next_result = _prefill_prompt(session, prompt_ids, sample=True)
            raise_if_generation_deadline_expired(request)
            if next_result is None:
                raise RuntimeError("native prefill did not produce next-token logits")
            generated_token_ids.append(int(next_result.token_id))
            _queue_json_object_close_if_needed(
                state,
                session.tokenizer,
                next_result.token_text,
                remaining_tokens=max_tokens - len(generated_token_ids),
            )
            finished = _is_finished(
                session.tokenizer,
                generated_token_ids,
                ignore_eos=ignore_eos,
                stop_token_ids=sampling_request.stop_token_ids,
                stop_token_sequences=sampling_request.stop_token_sequences,
            )
            yield GenerationStreamChunk(
                next_result.token_text,
                token_logprobs=(
                    (_token_logprob_from_step(session.tokenizer, next_result),)
                    if include_internal_token_logprobs
                    else _stream_token_logprobs_from_step(session.tokenizer, next_result, sampling_request)
                ),
                finish_details=(
                    _finish_details_for_tokens(
                        session.tokenizer,
                        generated_token_ids,
                        ignore_eos=ignore_eos,
                        stop_token_ids=sampling_request.stop_token_ids,
                        stop_token_sequences=sampling_request.stop_token_sequences,
                        max_tokens=max_tokens,
                        sampler_mode=plan.mode.value,
                        sampling_state=state,
                    )
                    if finished or len(generated_token_ids) >= max_tokens
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_ids,
                    generated_token_ids,
                    row_index=row_index,
                    sampler_mode=plan.mode.value,
                    stop_token_sequences=sampling_request.stop_token_sequences,
                    phase=live_phase,
                    active_processors=plan.active_processors,
                    sampler_fast_path_blockers=plan.fast_path_blockers,
                    sampler_fallback_reason=plan.fallback_reason,
                    sampling_state=state,
                    forced_sample=next_result,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                ),
            )
            if finished:
                return

            current_token_id = int(next_result.token_id)
            for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
                raise_if_generation_deadline_expired(request)
                result = session.step(current_token_id, position=position, sample=True)
                raise_if_generation_deadline_expired(request)
                if result is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                generated_token_ids.append(int(result.token_id))
                _queue_json_object_close_if_needed(
                    state,
                    session.tokenizer,
                    result.token_text,
                    remaining_tokens=max_tokens - len(generated_token_ids),
                )
                finished = _is_finished(
                    session.tokenizer,
                    generated_token_ids,
                    ignore_eos=ignore_eos,
                    stop_token_ids=sampling_request.stop_token_ids,
                    stop_token_sequences=sampling_request.stop_token_sequences,
                )
                yield GenerationStreamChunk(
                    result.token_text,
                    token_logprobs=(
                        (_token_logprob_from_step(session.tokenizer, result),)
                        if include_internal_token_logprobs
                        else _stream_token_logprobs_from_step(session.tokenizer, result, sampling_request)
                    ),
                    finish_details=(
                        _finish_details_for_tokens(
                            session.tokenizer,
                            generated_token_ids,
                            ignore_eos=ignore_eos,
                            stop_token_ids=sampling_request.stop_token_ids,
                            stop_token_sequences=sampling_request.stop_token_sequences,
                            max_tokens=max_tokens,
                            sampler_mode=plan.mode.value,
                            sampling_state=state,
                        )
                        if finished or len(generated_token_ids) >= max_tokens
                        else None
                    ),
                    telemetry=_telemetry_for_tokens(
                        prompt_ids,
                        generated_token_ids,
                        row_index=row_index,
                        sampler_mode=plan.mode.value,
                        stop_token_sequences=sampling_request.stop_token_sequences,
                        phase=live_phase,
                        active_processors=plan.active_processors,
                        sampler_fast_path_blockers=plan.fast_path_blockers,
                        sampler_fallback_reason=plan.fallback_reason,
                        sampling_state=state,
                        forced_sample=result,
                        full_vocab_logits_d2h=full_vocab_logits_d2h,
                        logits_d2h_bytes=logits_d2h_bytes,
                    ),
                )
                current_token_id = int(result.token_id)
                if finished:
                    return
        finally:
            _configure_sampled_session(session, None, None, plan=plan)

    def _get_runner(self) -> Qwen35ParoNextTokenRunner:
        if self._runner is None:
            self._runner = Qwen35ParoNextTokenRunner(
                self.model_path,
                index=self.weight_index,
                backend=self.backend,
            )
        return self._runner

    def _get_session(
        self,
        runner: Qwen35ParoNextTokenRunner,
        *,
        max_sequence_length: int,
        kv_policy,
        auto_context_length: bool = False,
        max_batch_size: int = 1,
    ) -> Qwen35ParoResidentSession:
        kv_key = (
            kv_policy.storage_dtype.value,
            kv_policy.scale_dtype.value,
            kv_policy.scale_granularity,
            int(kv_policy.block_size),
        )
        batch_size = max(1, int(max_batch_size))
        capacity_ok = self._session_capacity >= max_sequence_length or bool(auto_context_length)
        batch_ok = self._session_batch_size == batch_size
        if (
            self._session is None
            or not capacity_ok
            or not batch_ok
            or self._session_kv_key != kv_key
        ):
            self.close()
            session_kwargs = {
                "max_sequence_length": max_sequence_length,
                "kv_policy": kv_policy.create_policy(),
                "kv_scale_dtype": kv_policy.scale_dtype,
                "kv_scale_granularity": kv_policy.scale_granularity,
            }
            if auto_context_length:
                session_kwargs["auto_context_length"] = True
            if batch_size > 1:
                session_kwargs["max_batch_size"] = batch_size
            self._session = Qwen35ParoResidentSession(runner, **session_kwargs)
            self._session_capacity = int(
                getattr(self._session, "max_sequence_length", max_sequence_length)
            )
            self._session_batch_size = int(getattr(self._session, "max_batch_size", batch_size))
            self._session_kv_key = kv_key
        else:
            self._session.reset()
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._session_capacity = 0
        self._session_batch_size = 0
        self._session_kv_key = None


@dataclass(slots=True)
class _ParoResidentLoopRow:
    request_id: int
    batch_id: int
    row_index: int
    request: GenerationRequest
    sampling_request: GenerationRequest
    prompt_ids: tuple[int, ...]
    sampler_plan: Any
    native_greedy: bool
    submitted_at: float
    sampling_state: RowSamplingState | None = None
    model_slot: int | None = None
    prefill_tokens_seen: int = 0
    generated_steps: list[Qwen35ParoAutoregressiveStepResult] = field(default_factory=list)
    scheduler_chunks: list[dict[str, Any]] = field(default_factory=list)
    first_token_emitted: bool = False
    native_prefill: bool = False
    native_decode_steps: int = 0
    serial_decode_steps: int = 0
    last_execution_path: str = "paro_resident_model_loop"


class Qwen35ParoResidentModelRunner:
    """Single fixed-capacity PARO session owned by the shared engine loop.

    Scheduler slots are transient placement metadata. Each admitted request gets
    one stable model slot in this owner's batch-shaped recurrent/KV allocation;
    scheduler compaction therefore never copies or aliases model state. Greedy
    decode uses only identity- and position-matched retained widths, while every
    unsupported width, position, or sampler shape stays on the exact resident
    row-serial path.
    """

    def __init__(
        self,
        generator: Qwen35ParoOneTokenGenerator,
        *,
        capacity: int = 8,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.generator = generator
        self.capacity = int(capacity)
        self._runner = generator._get_runner()
        self._session: Qwen35ParoResidentSession | None = None
        self._session_kv_policy: Any | None = None
        self._session_kv_key: tuple[str, str, str, int] | None = None
        self._rows: dict[int, _ParoResidentLoopRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._completed_metadata: dict[int, dict[str, Any]] = {}
        self._available_model_slots: list[int] = list(range(self.capacity))
        self._next_batch_id = 0
        self._route_counts: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()
        self._last_width_plan: dict[str, Any] = {}
        self._last_execution_manifest: dict[str, Any] = {}
        self._recent_completed_routes: deque[dict[str, Any]] = deque(maxlen=1024)
        self._closed = False

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    @property
    def available_model_slots(self) -> tuple[int, ...]:
        return tuple(self._available_model_slots)

    def prompt_tokens(self, prompt: PromptInput) -> tuple[int, ...]:
        tokens = _prompt_ids(Path(self.generator.model_path), prompt)
        if not tokens:
            raise ValueError("Qwen3.5/PARO prompt tokenization produced no token IDs")
        return tokens

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        return max(1, int(request.max_tokens))

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int:
        self._ensure_open()
        if max_sequence_length is not None and int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be positive")
        if sampling_params is None and self._session_kv_policy is not None:
            kv_policy = self._session_kv_policy
        else:
            kv_policy = resolve_kv_policy(
                getattr(sampling_params, "kv_storage", "auto"),
                scale_dtype=getattr(sampling_params, "kv_scale_dtype", "fp16"),
                scale_granularity=getattr(
                    sampling_params,
                    "kv_scale_granularity",
                    "per_token_head",
                ),
            )
        auto_context_length = max_sequence_length is None
        requested = (
            int(getattr(self._runner.config, "max_position_embeddings", 0) or 0)
            if auto_context_length
            else int(max_sequence_length)
        )
        if requested <= 0:
            requested = _session_capacity_for(1)
        self._ensure_session(
            required_sequence_length=requested,
            kv_policy=kv_policy,
            auto_context_length=auto_context_length,
        )
        assert self._session is not None
        return int(self._session.max_sequence_length)

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        if int(max_batch_size) > self.capacity:
            raise ValueError("scratch probe max_batch_size exceeds resident owner capacity")
        if self._rows:
            raise RuntimeError("cannot probe PARO request scratch while requests are registered")
        kv_policy = resolve_kv_policy(
            getattr(sampling_params, "kv_storage", "auto"),
            scale_dtype=getattr(sampling_params, "kv_scale_dtype", "fp16"),
            scale_granularity=getattr(
                sampling_params,
                "kv_scale_granularity",
                "per_token_head",
            ),
        )
        required = max(1, int(max_prompt_tokens)) + max(0, int(max_new_tokens)) + 1
        self._ensure_session(required_sequence_length=required, kv_policy=kv_policy)
        assert self._session is not None
        prepare = getattr(self._session, "prepare_request_scratch", None)
        if not callable(prepare):
            return {
                "max_prompt_tokens": int(max_prompt_tokens),
                "max_new_tokens": int(max_new_tokens),
                "max_batch_size": int(max_batch_size),
                "release_after_probe": bool(release_after_probe),
                "skipped": True,
                "reason": "session_hook_unavailable",
            }
        return dict(
            prepare(
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                max_batch_size=max_batch_size,
                release_after_probe=release_after_probe,
            )
        )

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        self._ensure_open()
        ids = tuple(int(request_id) for request_id in request_ids)
        prompts = tuple(tuple(int(token) for token in row) for row in prompt_rows)
        if len(ids) != len(request.prompts) or len(prompts) != len(ids):
            raise ValueError("request_ids, prompts, and prompt_rows must have the same length")
        if request.row_seeds and len(request.row_seeds) != len(request.prompts):
            raise ValueError("row_seeds must have one entry per prompt")
        if any(not row for row in prompts):
            raise ValueError("Qwen3.5/PARO prompt tokenization produced no token IDs")
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        required = max(len(row) for row in prompts) + max(0, int(request.max_tokens)) + 1
        self._ensure_session(required_sequence_length=required, kv_policy=kv_policy)
        assert self._session is not None
        sampling_request = _request_with_tokenizer_eos(request, self._session.tokenizer)
        sampler_plan = plan_sampler(
            sampling_request,
            native_gpu_available=False,
            native_gpu_requested=False,
        )
        native_greedy = sampler_plan.mode is SamplingMode.GREEDY_FAST and int(request.max_tokens) > 0
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        now = time.perf_counter()
        for row_index, (request_id, prompt_ids) in enumerate(zip(ids, prompts, strict=True)):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            self._rows[request_id] = _ParoResidentLoopRow(
                request_id=request_id,
                batch_id=batch_id,
                row_index=row_index,
                request=request,
                sampling_request=sampling_request,
                prompt_ids=prompt_ids,
                sampler_plan=sampler_plan,
                native_greedy=native_greedy,
                submitted_at=now,
                sampling_state=(
                    None
                    if native_greedy or int(request.max_tokens) <= 0
                    else _row_sampling_state(sampling_request, prompt_ids, row_index=row_index)
                ),
            )

    def reserve_admission(self, request: RequestState) -> None:
        row = self._row(request.request_id)
        if row.model_slot is not None:
            raise RuntimeError(f"request_id {row.request_id} already owns model slot {row.model_slot}")
        if not self._available_model_slots:
            raise MemoryError("PARO resident model owner has no free model slots")
        slot = int(self._available_model_slots[0])
        self._reset_session_slots((slot,))
        self._available_model_slots.pop(0)
        row.model_slot = slot
        self._route_counts["admissions"] += 1

    def rollback_admission(self, request: RequestState) -> None:
        self._release_model_slot(self._row(request.request_id))

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("PARO resident prefill requires commit=True")
        assert self._session is not None
        for request_id, token_row in zip(work.request_ids, work.token_rows, strict=True):
            row = self._row(request_id)
            if row.model_slot is None:
                raise RuntimeError("PARO prefill row has no admitted model slot")
            start = int(row.prefill_tokens_seen)
            chunk = tuple(int(token) for token in token_row)
            expected = row.prompt_ids[start:start + len(chunk)]
            if not chunk or chunk != expected:
                raise RuntimeError(
                    f"PARO prefill chunk drift for request_id {request_id}: "
                    f"expected {expected!r}, got {chunk!r}"
                )
            row.prefill_tokens_seen += len(chunk)
            final_chunk = row.prefill_tokens_seen == len(row.prompt_ids)
            if row.prefill_tokens_seen > len(row.prompt_ids):
                raise RuntimeError("PARO prefill consumed beyond the registered prompt")
            if int(row.request.max_tokens) <= 0:
                continue
            raise_if_generation_deadline_expired(row.request)
            result = self._prefill_row_chunk(
                row,
                chunk,
                start_position=start,
                final_chunk=final_chunk,
            )
            raise_if_generation_deadline_expired(row.request)
            if final_chunk:
                if result is None:
                    raise RuntimeError("PARO final prefill chunk did not produce a token")
                self._record_step(row, result)

    def decode_batch(self, work: WorkItem, *, commit: bool) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("PARO resident decode requires commit=True")
        rows = [self._row(request_id) for request_id in work.request_ids]
        for row in rows:
            raise_if_generation_deadline_expired(row.request)
        greedy_step_rows = [
            row
            for row in rows
            if row.native_greedy and row.first_token_emitted and not self._row_finished(row)
        ]
        if greedy_step_rows:
            self._step_greedy_rows(greedy_step_rows)
        for row in rows:
            if (
                not row.native_greedy
                and row.first_token_emitted
                and int(row.request.max_tokens) > 0
                and not self._row_finished(row)
            ):
                self._step_sampled_row(row)
            raise_if_generation_deadline_expired(row.request)

        generated: list[GeneratedToken] = []
        for row in rows:
            if int(row.request.max_tokens) <= 0:
                generated.append(
                    GeneratedToken(
                        row.request_id,
                        0,
                        finished=True,
                        stream_chunk=GenerationStreamChunk(
                            text="",
                            finish_details=self._row_finish_details(row),
                            telemetry=self._row_telemetry(row),
                        ),
                    )
                )
                continue
            if not row.generated_steps:
                raise RuntimeError("PARO resident decode row has no prefill token")
            if not row.first_token_emitted:
                row.first_token_emitted = True
            result = row.generated_steps[-1]
            finished = self._row_finished(row)
            chunk = self._stream_chunk(row, result, finished=finished)
            row.scheduler_chunks.append(
                _scheduler_token_chunk_payload(
                    row.request_id,
                    len(row.generated_steps) - 1,
                    int(result.token_id),
                    chunk,
                )
            )
            generated.append(
                GeneratedToken(
                    row.request_id,
                    int(result.token_id),
                    finished=finished,
                    stream_chunk=chunk,
                )
            )
        return tuple(generated)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        # Model state is keyed by stable owner slots, so scheduler compaction is
        # deliberately metadata-only at this boundary.
        for move in moves:
            self._row(move.request_id)
        if any(move.old_slot != move.new_slot for move in moves):
            self._route_counts["scheduler_compactions"] += 1

    def reclaim(self, completed: CompletedRequest) -> None:
        request_id = int(completed.request_id)
        row = self._rows.get(request_id)
        if row is None:
            return
        output = self._output_for_row(row, completed)
        self._outputs[request_id] = output
        metadata = self._execution_metadata(row)
        self._completed_metadata[request_id] = metadata
        self._recent_completed_routes.append(
            {"request_id": request_id, **copy.deepcopy(metadata)}
        )
        self._release_model_slot(row)
        self._rows.pop(request_id, None)
        self._route_counts["reclaims"] += 1

    def has_outputs(self, request_ids: Sequence[int]) -> bool:
        return all(int(request_id) in self._outputs for request_id in request_ids)

    def missing_outputs(self, request_ids: Sequence[int]) -> list[int]:
        return [
            int(request_id)
            for request_id in request_ids
            if int(request_id) not in self._outputs
        ]

    def take_outputs(self, request_ids: Sequence[int]) -> list[GenerationOutput]:
        return [self._outputs.pop(int(request_id)) for request_id in request_ids]

    def discard(self, request_ids: Sequence[int]) -> None:
        for request_id in request_ids:
            rid = int(request_id)
            row = self._rows.pop(rid, None)
            if row is not None:
                self._release_model_slot(row)
            self._outputs.pop(rid, None)
            self._completed_metadata.pop(rid, None)

    def finalize_batch(
        self,
        request: GenerationRequest,
        request_ids: Sequence[int],
        outputs: Sequence[GenerationOutput],
    ) -> None:
        ids = tuple(int(request_id) for request_id in request_ids)
        output_tuple = tuple(outputs)
        metadata = [self._completed_metadata.pop(request_id, {}) for request_id in ids]
        native_steps = max((int(item.get("native_decode_steps", 0)) for item in metadata), default=0)
        serial_fallback = any(bool(item.get("serial_decode_fallback", False)) for item in metadata)
        native_prefill = bool(metadata) and all(bool(item.get("native_compact_prefill", False)) for item in metadata)
        scheduler_chunks = [
            copy.deepcopy(chunk)
            for item in metadata
            for chunk in item.get("scheduler_chunks", [])
        ]
        path = (
            "paro_resident_native_width_decode"
            if native_steps > 0 and not serial_fallback
            else "paro_resident_model_loop"
        )
        self.generator.last_generation_outputs = output_tuple
        self.generator.last_batch_generation = {
            "path": path,
            "batch_size": len(ids),
            "request_ids": list(ids),
            "prompt_lengths": [len(self.prompt_tokens(prompt)) for prompt in request.prompts],
            "native_decode_steps": native_steps,
            "serial_decode_fallback": serial_fallback,
            "native_compact_prefill": native_prefill,
            "native_caware_decode": native_steps > 0,
            "throughput_claim_eligible": native_steps > 0 and not serial_fallback,
            "scheduler_token_chunks": scheduler_chunks,
            "resident_model_owner": True,
            "last_width_plan": copy.deepcopy(self._last_width_plan),
        }

    def observability_snapshot(self) -> dict[str, Any]:
        session = self._session
        slot_by_request = {
            str(request_id): row.model_slot
            for request_id, row in self._rows.items()
            if row.model_slot is not None
        }
        owned_summary_provider = getattr(session, "owned_buffer_summary", None)
        owned_summary = (
            owned_summary_provider()
            if callable(owned_summary_provider)
            else {}
        )
        kv_total_bytes = int(owned_summary.get("full_attention_kv_total_bytes", 0))
        return {
            "model_runner": {
                "fixed_session": True,
                "capacity": int(self.capacity),
                "active_request_ids": list(self.active_request_ids),
                "active_requests": len(self._rows),
                "available_model_slots": list(self._available_model_slots),
                "stable_model_slot_by_request": slot_by_request,
                "session_max_sequence_length": (
                    None if session is None else int(session.max_sequence_length)
                ),
                "kv_storage": (
                    None
                    if self._session_kv_policy is None
                    else self._session_kv_policy.storage_dtype.value
                ),
            },
            "kv": {
                "ownership": "fixed_session_model_slots",
                "capacity_slots": int(self.capacity),
                "active_slots": len(slot_by_request),
                "available_slots": len(self._available_model_slots),
                "resident_total_bytes": kv_total_bytes,
                "resident_bytes_per_slot": (
                    kv_total_bytes // self.capacity if kv_total_bytes else 0
                ),
                "storage_dtype": owned_summary.get("kv_storage_dtype"),
                "storage_layout": owned_summary.get("kv_storage_layout"),
                "scale_dtype": owned_summary.get("kv_scale_dtype"),
                "scale_granularity": owned_summary.get("kv_scale_granularity"),
            },
            "routes": {
                "counts": {
                    "admissions": int(self._route_counts["admissions"]),
                    "prefill_chunks": int(self._route_counts["prefill_chunks"]),
                    "native_group_calls": int(self._route_counts["native_group_calls"]),
                    "native_rows": int(self._route_counts["native_rows"]),
                    "serial_row_calls": int(self._route_counts["serial_row_calls"]),
                    "scheduler_compactions": int(
                        self._route_counts["scheduler_compactions"]
                    ),
                    "slot_resets": int(self._route_counts["slot_resets"]),
                    "reclaims": int(self._route_counts["reclaims"]),
                },
                "fallback_reasons": {
                    str(key): int(value)
                    for key, value in sorted(self._fallback_reasons.items())
                },
                "last_width_plan": copy.deepcopy(self._last_width_plan),
                "last_execution_manifest": copy.deepcopy(
                    self._last_execution_manifest
                ),
                "recent_completed": list(copy.deepcopy(self._recent_completed_routes)),
            },
        }

    def close(self) -> None:
        if self._closed:
            return
        for row in tuple(self._rows.values()):
            self._release_model_slot(row)
        self._rows.clear()
        self._outputs.clear()
        self._completed_metadata.clear()
        if self._session is not None:
            self._clear_session_sampler()
            self._session.close()
            self._session = None
        self._closed = True
        if self.generator._resident_model_runner is self:
            self.generator._resident_model_runner = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PARO resident model owner is closed")

    @staticmethod
    def _kv_key(kv_policy: Any) -> tuple[str, str, str, int]:
        return (
            kv_policy.storage_dtype.value,
            kv_policy.scale_dtype.value,
            kv_policy.scale_granularity,
            int(kv_policy.block_size),
        )

    def _ensure_session(
        self,
        *,
        required_sequence_length: int,
        kv_policy: Any,
        auto_context_length: bool = False,
    ) -> None:
        required = int(required_sequence_length)
        if required <= 0:
            raise ValueError("required_sequence_length must be positive")
        key = self._kv_key(kv_policy)
        session = self._session
        capacity_ok = session is not None and int(session.max_sequence_length) >= required
        key_ok = session is not None and self._session_kv_key == key
        if key_ok and (capacity_ok or auto_context_length):
            return
        if self._rows:
            reason = "KV policy" if not key_ok else "sequence capacity"
            raise ValueError(
                f"PARO resident {reason} cannot change while requests are registered"
            )
        if session is not None:
            session.close()
            self._session = None
        # A direct pre-owner session would duplicate the full resident state and
        # KV allocation. Release it before constructing the fixed-capacity owner.
        if self.generator._session is not None:
            self.generator.close()
        requested_capacity = _session_capacity_for(required)
        kwargs: dict[str, Any] = {
            "max_sequence_length": requested_capacity,
            "max_batch_size": self.capacity,
            "kv_policy": kv_policy.create_policy(),
            "kv_scale_dtype": kv_policy.scale_dtype,
            "kv_scale_granularity": kv_policy.scale_granularity,
        }
        if auto_context_length:
            kwargs["auto_context_length"] = True
        self._session = Qwen35ParoResidentSession(self._runner, **kwargs)
        self._session_kv_policy = kv_policy
        self._session_kv_key = key
        self._available_model_slots = list(range(self.capacity))
        self._route_counts["session_builds"] += 1

    def _row(self, request_id: int) -> _ParoResidentLoopRow:
        rid = int(request_id)
        if rid not in self._rows:
            raise KeyError(f"request_id {rid} is not registered with the PARO resident owner")
        return self._rows[rid]

    def _prefill_row_chunk(
        self,
        row: _ParoResidentLoopRow,
        chunk: tuple[int, ...],
        *,
        start_position: int,
        final_chunk: bool,
    ) -> Qwen35ParoAutoregressiveStepResult | None:
        assert self._session is not None and row.model_slot is not None
        end_position = int(start_position) + len(chunk)
        block_count = max(
            1,
            (end_position + int(self._session.block_size) - 1)
            // int(self._session.block_size),
        )
        slab = CompactPromptSlab.from_token_rows(
            request_ids=(row.request_id,),
            token_rows=(chunk,),
            start_positions=(int(start_position),),
            block_count=block_count,
            block_size=int(self._session.block_size),
            slot_ids=(int(row.model_slot),),
        )
        if final_chunk and not row.native_greedy:
            self._configure_sampled_row(row)
        else:
            self._clear_session_sampler()
        try:
            try:
                results = self._session.prefill_native_packed(
                    slab,
                    sample=bool(final_chunk),
                )
                row.native_prefill = True
            except NotImplementedError:
                self._fallback_reasons["packed_prefill_unavailable"] += 1
                results = self._prefill_row_serial(
                    row,
                    chunk,
                    start_position=int(start_position),
                    sample_final=bool(final_chunk),
                )
            result_tuple = tuple(results)
            if len(result_tuple) != 1:
                raise RuntimeError(
                    f"PARO prefill returned {len(result_tuple)} result(s) for one row"
                )
            self._route_counts["prefill_chunks"] += 1
            return result_tuple[0]
        finally:
            self._clear_session_sampler()

    def _prefill_row_serial(
        self,
        row: _ParoResidentLoopRow,
        chunk: tuple[int, ...],
        *,
        start_position: int,
        sample_final: bool,
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        assert self._session is not None and row.model_slot is not None
        final_result: Qwen35ParoAutoregressiveStepResult | None = None
        for offset, token_id in enumerate(chunk):
            sample = bool(sample_final and offset == len(chunk) - 1)
            result = self._session.step_batch_serial(
                (int(token_id),),
                positions=(int(start_position) + offset,),
                slots=(int(row.model_slot),),
                sample=sample,
            )[0]
            if sample:
                final_result = result
        return (final_result,)

    def _step_greedy_rows(self, rows: Sequence[_ParoResidentLoopRow]) -> None:
        assert self._session is not None
        ordered = sorted(rows, key=lambda row: int(row.model_slot))
        positions = tuple(len(row.prompt_ids) + len(row.generated_steps) - 1 for row in ordered)
        profile_provider = getattr(self._session, "native_batch_width_profile", None)
        profile = profile_provider() if callable(profile_provider) else None
        if profile is not None and not isinstance(profile, NativeBatchWidthProfile):
            raise TypeError("native_batch_width_profile() must return NativeBatchWidthProfile or None")
        width_plan = plan_batch_width_partition(
            len(ordered),
            profile=profile,
            positions=positions,
        )
        self._last_width_plan = width_plan.to_json_dict()
        if width_plan.blockers:
            for blocker in width_plan.blockers:
                self._fallback_reasons[str(blocker)] += len(ordered)
        result_by_request: dict[int, Qwen35ParoAutoregressiveStepResult] = {}
        cursor = 0
        self._clear_session_sampler()
        for group in width_plan.groups:
            group_rows = ordered[cursor:cursor + int(group.width)]
            cursor += int(group.width)
            tokens = tuple(int(row.generated_steps[-1].token_id) for row in group_rows)
            group_positions = tuple(
                len(row.prompt_ids) + len(row.generated_steps) - 1
                for row in group_rows
            )
            slots = tuple(int(row.model_slot) for row in group_rows)
            results: tuple[Qwen35ParoAutoregressiveStepResult | None, ...]
            if group.mode == "native":
                try:
                    results = tuple(
                        self._session.step_batch_native(
                            tokens,
                            positions=group_positions,
                            slots=slots,
                            sample=True,
                        )
                    )
                    self._route_counts["native_group_calls"] += 1
                    self._route_counts["native_rows"] += len(group_rows)
                    for row in group_rows:
                        row.native_decode_steps += 1
                        row.last_execution_path = "paro_resident_native_width_decode"
                except NotImplementedError:
                    self._fallback_reasons["native_width_runtime_unavailable"] += len(group_rows)
                    results = self._serial_greedy_group(
                        group_rows,
                        tokens=tokens,
                        positions=group_positions,
                        slots=slots,
                    )
            else:
                results = self._serial_greedy_group(
                    group_rows,
                    tokens=tokens,
                    positions=group_positions,
                    slots=slots,
                )
            if len(results) != len(group_rows):
                raise RuntimeError(
                    f"PARO decode group returned {len(results)} result(s) for {len(group_rows)} rows"
                )
            for row, result in zip(group_rows, results, strict=True):
                if result is None:
                    raise RuntimeError("PARO decode group did not sample a token")
                result_by_request[row.request_id] = result
        if cursor != len(ordered):
            raise RuntimeError("PARO width plan did not consume every live row")
        manifest = getattr(self._session, "last_batch_decode_execution", None)
        if isinstance(manifest, Mapping):
            self._last_execution_manifest = copy.deepcopy(dict(manifest))
        for row in rows:
            self._record_step(row, result_by_request[row.request_id])

    def _serial_greedy_group(
        self,
        rows: Sequence[_ParoResidentLoopRow],
        *,
        tokens: tuple[int, ...],
        positions: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Qwen35ParoAutoregressiveStepResult | None, ...]:
        assert self._session is not None
        results = tuple(
            self._session.step_batch_serial(
                tokens,
                positions=positions,
                slots=slots,
                sample=True,
            )
        )
        self._route_counts["serial_row_calls"] += len(rows)
        for row in rows:
            row.serial_decode_steps += 1
            row.last_execution_path = "paro_resident_serial_decode"
        return results

    def _step_sampled_row(self, row: _ParoResidentLoopRow) -> None:
        assert self._session is not None and row.model_slot is not None
        self._configure_sampled_row(row)
        try:
            previous = int(row.generated_steps[-1].token_id)
            position = len(row.prompt_ids) + len(row.generated_steps) - 1
            result = self._session.step_batch_serial(
                (previous,),
                positions=(position,),
                slots=(int(row.model_slot),),
                sample=True,
            )[0]
        finally:
            self._clear_session_sampler()
        if result is None:
            raise RuntimeError("PARO sampled serial row did not produce a token")
        self._route_counts["serial_row_calls"] += 1
        self._fallback_reasons["sampled_request"] += 1
        row.serial_decode_steps += 1
        row.last_execution_path = "paro_resident_sampled_serial_decode"
        self._record_step(row, result)

    def _configure_sampled_row(self, row: _ParoResidentLoopRow) -> None:
        assert self._session is not None and row.model_slot is not None
        if row.sampling_state is None:
            raise RuntimeError("sampled PARO row has no sampling state")
        configure = getattr(self._session, "configure_host_sampler_rows", None)
        if not callable(configure):
            raise NotImplementedError(
                "PARO resident sampled fallback requires per-slot host sampler state"
            )
        configure(
            row.sampling_request,
            {int(row.model_slot): row.sampling_state},
        )

    def _clear_session_sampler(self) -> None:
        session = self._session
        if session is None:
            return
        configure = getattr(session, "configure_host_sampler_rows", None)
        if callable(configure):
            configure(None, None)

    def _record_step(
        self,
        row: _ParoResidentLoopRow,
        result: Qwen35ParoAutoregressiveStepResult,
    ) -> None:
        row.generated_steps.append(result)
        if row.sampling_state is not None:
            _queue_json_object_close_if_needed(
                row.sampling_state,
                self._session.tokenizer if self._session is not None else None,
                result.token_text,
                remaining_tokens=max(
                    0,
                    int(row.request.max_tokens) - len(row.generated_steps),
                ),
            )

    def _row_finished(self, row: _ParoResidentLoopRow) -> bool:
        if len(row.generated_steps) >= max(0, int(row.request.max_tokens)):
            return True
        return _is_finished(
            self._session.tokenizer if self._session is not None else None,
            tuple(step.token_id for step in row.generated_steps),
            ignore_eos=row.sampling_request.ignore_eos,
            stop_token_ids=row.sampling_request.stop_token_ids,
            stop_token_sequences=row.sampling_request.stop_token_sequences,
        )

    def _row_finish_details(self, row: _ParoResidentLoopRow) -> FinishDetails:
        return _finish_details_for_tokens(
            self._session.tokenizer if self._session is not None else None,
            tuple(step.token_id for step in row.generated_steps),
            ignore_eos=row.sampling_request.ignore_eos,
            stop_token_ids=row.sampling_request.stop_token_ids,
            stop_token_sequences=row.sampling_request.stop_token_sequences,
            max_tokens=row.request.max_tokens,
            sampler_mode=row.sampler_plan.mode.value,
            sampling_state=row.sampling_state,
        )

    def _stream_chunk(
        self,
        row: _ParoResidentLoopRow,
        result: Qwen35ParoAutoregressiveStepResult,
        *,
        finished: bool,
    ) -> GenerationStreamChunk:
        assert self._session is not None
        return GenerationStreamChunk(
            text=result.token_text,
            token_logprobs=_stream_token_logprobs_from_step(
                self._session.tokenizer,
                result,
                row.sampling_request,
            ),
            finish_details=self._row_finish_details(row) if finished else None,
            telemetry=self._row_telemetry(row, forced_sample=result),
        )

    def _row_telemetry(
        self,
        row: _ParoResidentLoopRow,
        *,
        forced_sample: Qwen35ParoAutoregressiveStepResult | None = None,
        generated_steps: Sequence[Qwen35ParoAutoregressiveStepResult] | None = None,
    ) -> GenerationTelemetry:
        visible_steps = row.generated_steps if generated_steps is None else generated_steps
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            row.sampler_plan,
            vocab_size=getattr(self._session, "vocab_size", None),
        )
        return _telemetry_for_tokens(
            row.prompt_ids,
            tuple(step.token_id for step in visible_steps),
            row_index=row.row_index,
            request_id=str(row.request_id),
            sampler_mode=row.sampler_plan.mode.value,
            stop_token_sequences=row.sampling_request.stop_token_sequences,
            active_processors=row.sampler_plan.active_processors,
            sampler_fast_path_blockers=row.sampler_plan.fast_path_blockers,
            sampler_fallback_reason=row.sampler_plan.fallback_reason,
            sampling_state=row.sampling_state,
            forced_sample=forced_sample,
            full_vocab_logits_d2h=full_vocab_logits_d2h,
            logits_d2h_bytes=logits_d2h_bytes,
            execution_path=row.last_execution_path,
            native_compact_prefill=row.native_prefill,
            native_caware_decode=row.native_decode_steps > 0,
            serial_decode_fallback=(
                not row.native_greedy or row.serial_decode_steps > 0
            ),
            native_sampler_rows=False,
            timing={"request_total_ms": (time.perf_counter() - row.submitted_at) * 1000.0},
            diagnostics={
                "stable_model_slot": row.model_slot,
                "last_width_plan": copy.deepcopy(self._last_width_plan),
            },
        )

    def _output_for_row(
        self,
        row: _ParoResidentLoopRow,
        completed: CompletedRequest,
    ) -> GenerationOutput:
        assert self._session is not None
        cancelled = completed.finish_reason in {"cancel", "disconnect", "timeout"}
        visible_steps = (
            row.generated_steps[:len(completed.generated_tokens)]
            if cancelled
            else row.generated_steps
        )
        ids = tuple(int(step.token_id) for step in visible_steps)
        finish_details = completed.finish_details if cancelled else self._row_finish_details(row)
        token_logprobs = (
            tuple(_token_logprob_from_step(self._session.tokenizer, step) for step in visible_steps)
            if row.sampling_request.logprobs or int(row.sampling_request.top_logprobs) > 0
            else ()
        )
        return GenerationOutput(
            text="".join(step.token_text for step in visible_steps),
            generated_token_ids=ids,
            token_logprobs=token_logprobs,
            finish_details=finish_details,
            telemetry=self._row_telemetry(
                row,
                forced_sample=visible_steps[-1] if visible_steps else None,
                generated_steps=visible_steps,
            ),
        )

    @staticmethod
    def _execution_metadata(row: _ParoResidentLoopRow) -> dict[str, Any]:
        return {
            "native_greedy": bool(row.native_greedy),
            "native_compact_prefill": bool(row.native_prefill),
            "native_decode_steps": int(row.native_decode_steps),
            "serial_decode_steps": int(row.serial_decode_steps),
            "serial_decode_fallback": (
                not row.native_greedy or row.serial_decode_steps > 0
            ),
            "stable_model_slot": row.model_slot,
            "scheduler_chunks": copy.deepcopy(row.scheduler_chunks),
        }

    def _release_model_slot(self, row: _ParoResidentLoopRow) -> None:
        slot = row.model_slot
        if slot is None:
            return
        self._reset_session_slots((int(slot),))
        if int(slot) in self._available_model_slots:
            raise RuntimeError(f"PARO model slot {slot} was released twice")
        self._available_model_slots.append(int(slot))
        self._available_model_slots.sort()
        row.model_slot = None

    def _reset_session_slots(self, slots: Sequence[int]) -> None:
        if self._session is None:
            return
        reset = getattr(self._session, "reset_slots", None)
        if not callable(reset):
            raise NotImplementedError(
                "PARO fixed-capacity model owner requires slot-local reset support"
            )
        slot_tuple = tuple(int(slot) for slot in slots)
        reset(slot_tuple)
        self._route_counts["slot_resets"] += len(slot_tuple)


def _per_row_sampling_params(request: GenerationRequest) -> PerRowSamplingParams:
    return PerRowSamplingParams(
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        logit_bias=request.logit_bias,
        suppress_tokens=request.suppress_token_ids,
        min_tokens=request.min_tokens,
        eos_token_id=request.eos_token_id,
        ignore_eos=request.ignore_eos,
        seed=request.seed,
        stop_tokens=request.stop_token_ids,
        stop_token_sequences=request.stop_token_sequences,
        forced_tokens_pending=request.forced_tokens_pending,
        forced_token_reason=request.forced_token_reason,
        post_thinking_forced_tokens_pending=request.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=request.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=request.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=request.force_sequence_completion_reason,
        json_object_close_forcing=request.json_object_close_forcing,
        thinking_close_token_ids=request.thinking_close_token_ids,
        thinking_hard_token_cap=request.thinking_hard_token_cap,
        thinking_soft_close_window=request.thinking_soft_close_window,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs,
    )


def _slot_sampler_state_clones(
    scheduler: ResidentBatchScheduler,
    request_ids: tuple[int, ...] | list[int],
    slots: tuple[int, ...] | list[int],
) -> dict[int, RowSamplingState]:
    return {
        int(slot): _clone_row_sampling_state(scheduler.sampler_state(int(request_id)))
        for request_id, slot in zip(request_ids, slots, strict=True)
    }


def _clone_row_sampling_state(state: RowSamplingState) -> RowSamplingState:
    thinking_budget = clone_thinking_budget_state(state.thinking_budget)
    if thinking_budget is not None:
        return RowSamplingState(
            prompt_tokens=state.prompt_tokens,
            seed=state.seed,
            request_id=state.request_id,
            row_index=state.row_index,
            generated_tokens=tuple(state.generated_tokens),
            step_index=state.step_index,
            stop_token_sequences=state.stop_token_sequences,
            post_thinking_forced_tokens_pending=state.post_thinking_forced_tokens_pending.pending_tokens,
            post_thinking_forced_token_reason=state.post_thinking_forced_token_reason,
            force_sequence_completion_token_sequences=state.force_sequence_completion_token_sequences,
            force_sequence_completion_reason=state.force_sequence_completion_reason,
            json_object_close_forcing=state.json_object_close_forcing,
            thinking_budget=thinking_budget,
        )
    return RowSamplingState(
        prompt_tokens=state.prompt_tokens,
        seed=state.seed,
        request_id=state.request_id,
        row_index=state.row_index,
        generated_tokens=tuple(state.generated_tokens),
        step_index=state.step_index,
        stop_token_sequences=state.stop_token_sequences,
        forced_tokens_pending=state.forced_tokens,
        forced_token_reason=state.forced_token_reason,
        post_thinking_forced_tokens_pending=state.post_thinking_forced_tokens_pending.pending_tokens,
        post_thinking_forced_token_reason=state.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=state.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=state.force_sequence_completion_reason,
        json_object_close_forcing=state.json_object_close_forcing,
    )


def _generation_output_from_steps(
    tokenizer: Any,
    steps: list[Qwen35ParoAutoregressiveStepResult] | tuple[Qwen35ParoAutoregressiveStepResult, ...],
    *,
    finish_details: FinishDetails,
    telemetry: GenerationTelemetry | None = None,
) -> GenerationOutput:
    tokens = tuple(_token_logprob_from_step(tokenizer, step) for step in steps)
    return GenerationOutput(
        text="".join(step.token_text for step in steps),
        token_logprobs=tokens,
        generated_token_ids=tuple(token.token_id for token in tokens),
        finish_details=finish_details,
        telemetry=telemetry,
    )


def _batch_scheduler_token_chunks(
    request_ids: tuple[int, ...],
    prompt_rows_by_request: dict[int, list[int]],
    generated_ids: dict[int, list[int]],
    generated_texts: dict[int, list[str]],
    *,
    tokenizer: Any,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
    max_tokens: int,
    sampler_mode: str,
    execution_path: str | None,
    native_compact_prefill: bool | None,
    native_caware_decode: bool | None,
    serial_decode_fallback: bool | None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for request_id in request_ids:
        ids = generated_ids[request_id]
        texts = generated_texts[request_id]
        prefix: list[int] = []
        for token_index, (token_id, token_text) in enumerate(zip(ids, texts, strict=True)):
            prefix.append(int(token_id))
            final = token_index == len(ids) - 1
            chunk = GenerationStreamChunk(
                text=token_text,
                finish_details=(
                    _finish_details_for_tokens(
                        tokenizer,
                        prefix,
                        ignore_eos=ignore_eos,
                        stop_token_ids=stop_token_ids,
                        stop_token_sequences=stop_token_sequences,
                        max_tokens=max_tokens,
                        sampler_mode=sampler_mode,
                    )
                    if final
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_rows_by_request[request_id],
                    prefix,
                    row_index=request_id,
                    request_id=str(request_id),
                    sampler_mode=sampler_mode,
                    stop_token_sequences=stop_token_sequences,
                    phase="answer",
                    execution_path=execution_path,
                    native_compact_prefill=native_compact_prefill,
                    native_caware_decode=native_caware_decode,
                    serial_decode_fallback=serial_decode_fallback,
                ),
            )
            chunks.append(_scheduler_token_chunk_payload(request_id, token_index, int(token_id), chunk))
    return chunks


def _sampled_batch_scheduler_token_chunks(
    request_ids: tuple[int, ...],
    prompt_rows_by_request: dict[int, list[int]],
    output_steps: dict[int, list[Qwen35ParoAutoregressiveStepResult]],
    sampling_state_step_snapshots: dict[int, list[RowSamplingState]],
    *,
    tokenizer: Any,
    vocab_size: Any | None,
    request: GenerationRequest,
    plans: dict[int, Any],
    execution_path: str | None,
    native_compact_prefill: bool | None,
    native_caware_decode: bool | None,
    serial_decode_fallback: bool | None,
    native_sampler_rows: bool | None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for request_id in request_ids:
        steps = output_steps[request_id]
        snapshots = sampling_state_step_snapshots[request_id]
        if len(snapshots) != len(steps):
            raise RuntimeError("sampled scheduler token snapshot count does not match generated steps")
        plan = plans[request_id]
        full_vocab_logits_d2h, logits_d2h_bytes = _sampler_logits_d2h_metadata(
            plan,
            vocab_size=vocab_size,
        )
        prefix: list[int] = []
        for token_index, (step, state) in enumerate(zip(steps, snapshots, strict=True)):
            prefix.append(int(step.token_id))
            final = token_index == len(steps) - 1
            phase = None if state.thinking_budget is not None else "answer"
            chunk = GenerationStreamChunk(
                text=step.token_text,
                token_logprobs=_stream_token_logprobs_from_step(tokenizer, step, request),
                finish_details=(
                    _finish_details_for_tokens(
                        tokenizer,
                        prefix,
                        ignore_eos=request.ignore_eos,
                        stop_token_ids=request.stop_token_ids,
                        stop_token_sequences=request.stop_token_sequences,
                        max_tokens=request.max_tokens,
                        sampler_mode=plan.mode.value,
                        sampling_state=state,
                    )
                    if final
                    else None
                ),
                telemetry=_telemetry_for_tokens(
                    prompt_rows_by_request[request_id],
                    prefix,
                    row_index=request_id,
                    request_id=str(request_id),
                    sampler_mode=plan.mode.value,
                    stop_token_sequences=request.stop_token_sequences,
                    phase=phase,
                    active_processors=plan.active_processors,
                    sampler_fast_path_blockers=plan.fast_path_blockers,
                    sampler_fallback_reason=plan.fallback_reason,
                    sampling_state=state,
                    forced_sample=step,
                    full_vocab_logits_d2h=full_vocab_logits_d2h,
                    logits_d2h_bytes=logits_d2h_bytes,
                    execution_path=execution_path,
                    native_compact_prefill=native_compact_prefill,
                    native_caware_decode=native_caware_decode,
                    serial_decode_fallback=serial_decode_fallback,
                    native_sampler_rows=native_sampler_rows,
                ),
            )
            chunks.append(_scheduler_token_chunk_payload(request_id, token_index, int(step.token_id), chunk))
    return chunks


def _scheduler_token_chunk_payload(
    request_id: int,
    token_index: int,
    token_id: int,
    chunk: GenerationStreamChunk,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": int(request_id),
        "token_index": int(token_index),
        "token_id": int(token_id),
        "finished": chunk.finish_details is not None,
        "chunk": {
            "text": chunk.text,
        },
    }
    if chunk.token_logprobs:
        payload["chunk"]["token_logprobs"] = [
            {
                "token_id": token.token_id,
                "token_text": token.token_text,
                "logprob": token.logprob,
                "top_logprobs": [
                    {"token_id": top_id, "token_text": top_text, "logprob": top_logprob}
                    for top_id, top_text, top_logprob in token.top_logprobs
                ],
            }
            for token in chunk.token_logprobs
        ]
    if chunk.finish_details is not None:
        payload["chunk"]["finish_details"] = chunk.finish_details.to_json_dict()
    if chunk.telemetry is not None:
        payload["chunk"]["telemetry"] = chunk.telemetry.to_json_dict()
    return payload


def _stream_token_logprobs_from_step(
    tokenizer: Any,
    step: Qwen35ParoAutoregressiveStepResult,
    request: GenerationRequest,
) -> tuple[TokenLogprob, ...]:
    if not request.logprobs and int(request.top_logprobs) <= 0:
        return ()
    return (_token_logprob_from_step(tokenizer, step),)


def _token_logprob_from_step(tokenizer: Any, step: Qwen35ParoAutoregressiveStepResult) -> TokenLogprob:
    return TokenLogprob(
        token_id=step.token_id,
        token_text=step.token_text,
        logprob=step.logprob,
        top_logprobs=tuple(
            (token_id, _decode_token_cached(tokenizer, token_id), logprob)
            for token_id, logprob in step.top_logprobs
        ),
    )


def _telemetry_for_tokens(
    prompt_ids: list[int] | tuple[int, ...],
    generated_token_ids: list[int] | tuple[int, ...],
    *,
    row_index: int,
    sampler_mode: str,
    stop_token_sequences: tuple[tuple[int, ...], ...],
    request_id: str | None = None,
    phase: str | None = None,
    active_processors: tuple[str, ...] = (),
    sampler_fast_path_blockers: tuple[str, ...] = (),
    sampler_fallback_reason: str | None = None,
    sampling_state: RowSamplingState | None = None,
    forced_sample: Qwen35ParoAutoregressiveStepResult | None = None,
    full_vocab_logits_d2h: bool | None = None,
    logits_d2h_bytes: int | None = None,
    execution_path: str | None = None,
    native_compact_prefill: bool | None = None,
    native_caware_decode: bool | None = None,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool | None = None,
    timing: dict[str, float] | None = None,
    timing_scope: str | None = None,
    batch_id: str | None = None,
    group_rows: int | None = None,
    timing_owner: bool | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> GenerationTelemetry:
    state_payload = _decode_state_from_sampling_state(sampling_state)
    forced_token_id, forced_token_reason, forced_tokens_remaining = _forced_token_metadata(forced_sample)
    if timing is not None and timing_scope is None:
        timing_scope = "choice"
        group_rows = 1 if group_rows is None else int(group_rows)
        timing_owner = True if timing_owner is None else bool(timing_owner)
    return GenerationTelemetry.from_decode_counts(
        request_id=request_id,
        row_index=row_index,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_token_ids),
        phase=phase or state_payload.get("phase", "done"),
        reasoning_tokens=int(state_payload.get("reasoning_tokens", 0)),
        answer_tokens=int(state_payload.get("answer_tokens", 0)),
        forced_tokens_pending=tuple(state_payload.get("forced_tokens_pending", ())),
        forced_token_id=forced_token_id,
        forced_token_reason=forced_token_reason,
        forced_tokens_remaining=forced_tokens_remaining,
        post_thinking_forced_tokens_pending=tuple(state_payload.get("post_thinking_forced_tokens_pending", ())),
        post_thinking_forced_token_reason=state_payload.get("post_thinking_forced_token_reason"),
        force_sequence_completion_token_sequences=tuple(
            tuple(sequence) for sequence in state_payload.get("force_sequence_completion_token_sequences", ())
        ),
        force_sequence_completion_reason=state_payload.get("force_sequence_completion_reason"),
        budget_pressure=state_payload.get("budget_pressure"),
        sampler_mode=sampler_mode,
        stop_suffix_state=_stop_suffix_state(generated_token_ids, stop_token_sequences),
        active_processors=active_processors,
        sampler_fast_path_blockers=sampler_fast_path_blockers,
        sampler_fallback_reason=sampler_fallback_reason,
        full_vocab_logits_d2h=full_vocab_logits_d2h,
        logits_d2h_bytes=logits_d2h_bytes,
        execution_path=execution_path,
        native_compact_prefill=native_compact_prefill,
        native_caware_decode=native_caware_decode,
        serial_decode_fallback=serial_decode_fallback,
        native_sampler_rows=native_sampler_rows,
        timing=timing,
        timing_scope=timing_scope,
        batch_id=batch_id,
        group_rows=group_rows,
        timing_owner=timing_owner,
        diagnostics=diagnostics,
    )


def _forced_token_metadata(
    sample: Qwen35ParoAutoregressiveStepResult | None,
) -> tuple[int | None, str | None, int | None]:
    if sample is None or not bool(getattr(sample, "forced", False)):
        return None, None, None
    return (
        int(sample.token_id),
        None if sample.forced_reason is None else str(sample.forced_reason),
        max(0, int(sample.forced_tokens_remaining)),
    )


def _sampler_logits_d2h_metadata(
    plan: Any,
    *,
    vocab_size: Any | None = None,
) -> tuple[bool | None, int | None]:
    mode = getattr(plan, "mode", None)
    if mode is SamplingMode.GPU_SAMPLE:
        return False, 0
    if mode in (SamplingMode.HOST_LOGITS_SAMPLE, SamplingMode.PROCESSED_ARGMAX):
        try:
            size = int(vocab_size)
        except (TypeError, ValueError):
            return None, None
        if size > 0:
            return True, size * 4
    return None, None


def _decode_state_from_sampling_state(state: RowSamplingState | None) -> dict[str, Any]:
    if state is None:
        return {}
    payload: dict[str, Any] = {}
    if state.forced_tokens:
        payload["forced_tokens_pending"] = state.forced_tokens
    if state.post_thinking_forced_tokens_pending.pending_tokens:
        payload["post_thinking_forced_tokens_pending"] = state.post_thinking_forced_tokens_pending.pending_tokens
    if state.post_thinking_forced_token_reason is not None:
        payload["post_thinking_forced_token_reason"] = state.post_thinking_forced_token_reason
    if state.force_sequence_completion_token_sequences:
        payload["force_sequence_completion_token_sequences"] = state.force_sequence_completion_token_sequences
    if state.force_sequence_completion_reason is not None:
        payload["force_sequence_completion_reason"] = state.force_sequence_completion_reason
    budget = state.thinking_budget
    if budget is None:
        return payload
    payload["phase"] = str(budget.phase)
    payload["reasoning_tokens"] = int(budget.reasoning_tokens)
    payload["answer_tokens"] = int(budget.answer_tokens)
    forced_reason = getattr(budget.forced_tokens, "reason", None)
    pressure = "hard_close" if forced_reason == "thinking_hard_close" else budget.budget_pressure
    if pressure is not None:
        payload["budget_pressure"] = str(pressure)
    return payload


def _stop_suffix_state(
    generated_token_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> dict[str, Any] | None:
    payload = token_sequence_state_for_tokens(generated_token_ids, stop_token_sequences).to_json_dict()
    return payload or None


def _row_sampling_state(
    request: GenerationRequest,
    prompt_ids: list[int] | tuple[int, ...],
    *,
    row_index: int,
) -> RowSamplingState:
    return RowSamplingState(
        prompt_tokens=tuple(int(token) for token in prompt_ids),
        seed=row_seed_for_index(request, row_index),
        row_index=row_index,
        stop_token_sequences=request.stop_token_sequences,
        forced_tokens_pending=request.forced_tokens_pending,
        forced_token_reason=request.forced_token_reason,
        post_thinking_forced_tokens_pending=request.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=request.post_thinking_forced_token_reason,
        force_sequence_completion_token_sequences=request.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=request.force_sequence_completion_reason,
        json_object_close_forcing=request.json_object_close_forcing,
        thinking_budget=thinking_budget_state_from_params(request),
    )


def _configure_sampled_session(
    session: Any,
    request: GenerationRequest | None,
    state: RowSamplingState | None,
    *,
    plan,
) -> None:
    if plan.mode is SamplingMode.GPU_SAMPLE:
        _configure_native_sampler(session, request, state)
    else:
        _configure_host_sampler(session, request, state)


def _configure_native_sampler(
    session: Any,
    request: GenerationRequest | None,
    state: RowSamplingState | None,
) -> None:
    configure = getattr(session, "configure_native_sampler", None)
    if not callable(configure):
        if request is None and state is None:
            return
        raise NotImplementedError(
            "Qwen3.5/PARO native GPU sampling requires resident sampler support"
        )
    configure(request, state)


def _configure_host_sampler(
    session: Any,
    request: GenerationRequest | None,
    state: RowSamplingState | None,
) -> None:
    configure = getattr(session, "configure_host_sampler", None)
    if not callable(configure):
        if request is None and state is None:
            return
        raise NotImplementedError(
            "Qwen3.5/PARO host-logits sampling requires resident sampler support"
        )
    configure(request, state)


def _native_gpu_sampler_route_available(*, prompt_count: int) -> bool:
    return int(prompt_count) == 1 and _native_gpu_sampler_requested()


def _native_gpu_sampler_requested() -> bool:
    return _env_flag("HIPENGINE_QWEN35_NATIVE_SAMPLER", default=True)


def _prefill_prompt(session: Any, token_ids: tuple[int, ...] | list[int], *, sample: bool) -> Any:
    """Prefill a prompt, using exact c1 steps below the native conv width."""

    tokens = tuple(int(token_id) for token_id in token_ids)
    if not tokens:
        raise ValueError("prompt produced no tokens")
    min_native_tokens = max(
        1,
        int(getattr(getattr(session, "config", None), "linear_conv_kernel_dim", 1) or 1),
    )
    if len(tokens) >= min_native_tokens:
        return session.prefill_native(tokens, sample=sample)

    result = None
    final_position = len(tokens) - 1
    for position, token_id in enumerate(tokens):
        result = session.step(
            token_id,
            position=position,
            sample=bool(sample and position == final_position),
        )
    if hasattr(session, "last_prefill_execution"):
        session.last_prefill_execution = {
            "path": "short_prompt_serial_c1",
            "tokens": len(tokens),
            "full_native": False,
            "native_min_tokens": min_native_tokens,
        }
    return result


def _session_capacity_for(required_sequence_length: int) -> int:
    """Return a reusable session capacity for a request.

    Chat prompts grow after every turn, so allocating exactly the current
    prompt+decode length forces resident weights/KV buffers to be torn down and
    rebuilt on each request.  Keep a modest floor and bucket growth to preserve
    the resident session across normal local chat turns while still allowing
    larger explicit contexts to expand on demand.
    """

    required = int(required_sequence_length)
    if required <= 0:
        raise ValueError("required_sequence_length must be positive")
    floor = max(1, _env_int("HIPENGINE_SESSION_MIN_TOKENS", 4096))
    bucket = max(1, _env_int("HIPENGINE_SESSION_BUCKET_TOKENS", 1024))
    capacity = max(required, floor)
    return ((capacity + bucket - 1) // bucket) * bucket


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _native_batch_width_profile_for_runner(
    runner: Qwen35ParoNextTokenRunner,
    kv_policy: Any,
) -> NativeBatchWidthProfile | None:
    retained_defaults = bool(
        backend_package_capability(
            runner.backend,
            "PARO_RETAINED_BATCH_DEFAULTS",
            False,
        )
    )
    native_decode_default = bool(
        backend_package_capability(
            runner.backend,
            "PARO_NATIVE_BATCH_DECODE_DEFAULT",
            False,
        )
    )
    if not _env_flag(
        "HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS",
        default=retained_defaults,
    ) or not _env_flag(
        "HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE",
        default=native_decode_default,
    ):
        return None
    artifact = os.environ.get(QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE_ENV)
    if artifact is None or not artifact.strip():
        artifact = DEFAULT_QWEN35_PARO_NATIVE_BATCH_WIDTH_PROFILE
    return load_qwen35_paro_native_batch_width_profile(
        artifact.strip(),
        backend=runner.backend,
        target_arch=runner.target_arch,
        model_path=runner.model,
        kv_dtype=kv_policy.storage_dtype.value,
    )


def _native_profile_prompt_position_blockers(
    profile: NativeBatchWidthProfile,
    *,
    model_path: Path,
    prompts: tuple[PromptInput, ...],
    max_tokens: int,
) -> tuple[str, ...]:
    starts: list[int] = []
    ends: list[int] = []
    for prompt in prompts:
        prompt_ids = _prompt_ids(model_path, prompt)
        starts.append(len(prompt_ids))
        ends.append(len(prompt_ids) + max(0, int(max_tokens) - 2))
    return tuple(
        dict.fromkeys(
            (
                *profile.position_blockers(tuple(starts)),
                *profile.position_blockers(tuple(ends)),
            )
        )
    )


def _relabel_isolated_group_output(
    output: GenerationOutput,
    *,
    row_index: int,
    group_index: int,
    group_width: int,
    parent_path: str,
    native_compact_prefill: bool | None = None,
    native_caware_decode: bool | None = None,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool | None = None,
) -> GenerationOutput:
    telemetry = _relabel_isolated_group_telemetry(
        output.telemetry,
        row_index=row_index,
        group_index=group_index,
        group_width=group_width,
        parent_path=parent_path,
        native_compact_prefill=native_compact_prefill,
        native_caware_decode=native_caware_decode,
        serial_decode_fallback=serial_decode_fallback,
        native_sampler_rows=native_sampler_rows,
    )
    if telemetry is None:
        return output
    return replace(output, telemetry=telemetry)


def _relabel_isolated_group_telemetry(
    telemetry: GenerationTelemetry | None,
    *,
    row_index: int,
    group_index: int,
    group_width: int,
    parent_path: str,
    native_compact_prefill: bool | None = None,
    native_caware_decode: bool | None = None,
    serial_decode_fallback: bool | None = None,
    native_sampler_rows: bool | None = None,
) -> GenerationTelemetry | None:
    if telemetry is None:
        return None
    decode_updates: dict[str, Any] = {
        "request_id": str(row_index),
        "row_index": row_index,
        "execution_path": parent_path,
    }
    for name, value in (
        ("native_compact_prefill", native_compact_prefill),
        ("native_caware_decode", native_caware_decode),
        ("serial_decode_fallback", serial_decode_fallback),
        ("native_sampler_rows", native_sampler_rows),
    ):
        if value is not None:
            decode_updates[name] = value
    decode_state = replace(telemetry.decode_state, **decode_updates)
    diagnostics = dict(telemetry.diagnostics or {})
    diagnostics["isolated_width_group"] = {
        "group_index": int(group_index),
        "group_width": int(group_width),
        "timing_scope": "group",
    }
    return replace(
        telemetry,
        decode_state=decode_state,
        diagnostics=diagnostics,
    )


def _relabel_isolated_scheduler_chunks(
    chunks: Any,
    *,
    request_offset: int,
    parent_path: str,
) -> list[dict[str, Any]]:
    relabeled: list[dict[str, Any]] = []
    for source in chunks:
        chunk = copy.deepcopy(source)
        local_request_id = int(chunk["request_id"])
        request_id = int(request_offset) + local_request_id
        chunk["request_id"] = request_id
        chunk_payload = chunk.get("chunk")
        if isinstance(chunk_payload, dict):
            telemetry = chunk_payload.get("telemetry")
            if isinstance(telemetry, dict):
                decode_state = telemetry.get("decode_state")
                if isinstance(decode_state, dict):
                    decode_state["request_id"] = str(request_id)
                    decode_state["row_index"] = request_id
                    decode_state["execution_path"] = parent_path
        relabeled.append(chunk)
    return relabeled


def _is_eos(tokenizer: Any | None, token_id: int) -> bool:
    return int(token_id) in _tokenizer_eos_ids(tokenizer)


def _tokenizer_eos_id(tokenizer: Any | None) -> int | None:
    """Primary EOS token id; first element of :func:`_tokenizer_eos_ids`."""
    ids = _tokenizer_eos_ids(tokenizer)
    return ids[0] if ids else None


# Qwen end-of-turn tokens, in priority order. The low-level ``tokenizers``
# object does not expose generation_config.json, so keep this model-specific
# fallback until special-token metadata is supplied by the model plugin.
# ``<|im_start|>`` is deliberately excluded because it starts a turn.
_EOS_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
)


def _normalize_eos_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError:
            values = (value,)
    normalized: list[int] = []
    for raw_value in values:
        try:
            token_id = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if token_id >= 0 and token_id not in normalized:
            normalized.append(token_id)
    return tuple(normalized)


def _tokenizer_eos_ids(tokenizer: Any | None) -> tuple[int, ...]:
    """Return explicit EOS ids followed by Qwen tokenizer fallbacks."""
    if tokenizer is None:
        return ()
    found = list(_normalize_eos_token_ids(getattr(tokenizer, "eos_token_id", None)))
    token_to_id = getattr(tokenizer, "token_to_id", None)
    for candidate in _EOS_TOKEN_CANDIDATES:
        value = _lookup_token_id(token_to_id, candidate)
        if value is not None and value not in found:
            found.append(value)
    return tuple(found)


def _request_with_tokenizer_eos(
    request: GenerationRequest,
    tokenizer: Any | None,
) -> GenerationRequest:
    if request.eos_token_id is not None:
        return request
    eos_token_id = _tokenizer_eos_id(tokenizer)
    if eos_token_id is None:
        return request
    return replace(request, eos_token_id=eos_token_id)


def _queue_json_object_close_if_needed(
    state: RowSamplingState,
    tokenizer: Any | None,
    token_text: str,
    *,
    remaining_tokens: int,
) -> None:
    state.observe_text_for_json_object_close(
        token_text,
        remaining_tokens=remaining_tokens,
        encode_text=lambda text: _tokenize_constraint_text(tokenizer, text),
    )


def _tokenize_constraint_text(tokenizer: Any | None, text: str) -> tuple[int, ...]:
    if tokenizer is None:
        return ()
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            token_ids = tuple(int(token) for token in encode(str(text)))
        except Exception:
            token_ids = ()
        if token_ids:
            return token_ids
    token_to_id = getattr(tokenizer, "token_to_id", None)
    whole = _lookup_token_id(token_to_id, str(text))
    if whole is not None:
        return (whole,)
    pieces: list[int] = []
    for char in str(text):
        token_id = _lookup_token_id(token_to_id, char)
        if token_id is None:
            return ()
        pieces.append(token_id)
    return tuple(pieces)


def _lookup_token_id(token_to_id: Any, token: str) -> int | None:
    try:
        value = token_to_id(token) if callable(token_to_id) else token_to_id.get(token)
    except Exception:
        return None
    return None if value is None else int(value)


def _is_finished(
    tokenizer: Any | None,
    generated_token_ids: list[int] | tuple[int, ...],
    *,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> bool:
    if not generated_token_ids:
        return False
    token_id = int(generated_token_ids[-1])
    if not ignore_eos and _is_eos(tokenizer, token_id):
        return True
    if token_id in {int(stop_id) for stop_id in stop_token_ids}:
        return True
    return _ends_with_stop_sequence(generated_token_ids, stop_token_sequences)


def _finish_details_for_tokens(
    tokenizer: Any | None,
    generated_token_ids: list[int] | tuple[int, ...],
    *,
    ignore_eos: bool,
    stop_token_ids: tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
    max_tokens: int,
    sampler_mode: str,
    sampling_state: RowSamplingState | None = None,
) -> FinishDetails:
    details: FinishDetails
    if generated_token_ids:
        token_id = int(generated_token_ids[-1])
        if not ignore_eos and _is_eos(tokenizer, token_id):
            details = FinishDetails(reason="eos", eos_token_id=token_id, sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, sampling_state)
        if token_id in {int(stop_id) for stop_id in stop_token_ids}:
            details = FinishDetails(reason="stop", stop_sequence=(token_id,), sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, sampling_state)
        sequence = _matched_stop_sequence(generated_token_ids, stop_token_sequences)
        if sequence:
            details = FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode=sampler_mode)
            return finish_details_with_sampling_state(details, sampling_state)
    if len(generated_token_ids) >= max(0, int(max_tokens)):
        details = FinishDetails(reason="length", length_limit=max_tokens, sampler_mode=sampler_mode)
        return finish_details_with_sampling_state(details, sampling_state)
    details = FinishDetails(reason="stop", sampler_mode=sampler_mode)
    return finish_details_with_sampling_state(details, sampling_state)


def _matched_stop_sequence(
    generated_token_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return token_sequence_state_for_tokens(generated_token_ids, stop_token_sequences).matched_sequence


def _ends_with_stop_sequence(
    generated_token_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> bool:
    return token_sequence_state_for_tokens(generated_token_ids, stop_token_sequences).matched


def make_qwen35_paro_one_token_generator(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
    )


def make_qwen35_paro_one_token_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1100",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator,
)
register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1151",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator_gfx1151,
)
