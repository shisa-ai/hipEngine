"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hipengine.generation.constraints import token_sequence_state_for_tokens
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.finish import finish_details_with_sampling_state
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationTelemetry,
    TokenLogprob,
    register_text_generator,
)
from hipengine.generation.sampling import (
    RowSamplingState,
    SamplingMode,
    plan_sampler,
    row_seed_for_index,
    select_token,
    thinking_budget_state_from_params,
)
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF greedy generator over a persistent resident session."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    tokenizer: Qwen35GGUFTokenizer = field(init=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(self, request: GenerationRequest) -> list[str]:
        outputs = self.generate_detailed(request)
        return [output.text for output in outputs]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        raise_if_generation_deadline_expired(request)
        plan = plan_sampler(request)
        if request.max_tokens == 0:
            self.last_generation_outputs = tuple(
                GenerationOutput(
                    text="",
                    finish_details=_gguf_finish_details((), self.tokenizer, request),
                    telemetry=_gguf_telemetry(
                        self.tokenizer.encode(prompt),
                        (),
                        request,
                        row_index=index,
                    ),
                )
                for index, prompt in enumerate(request.prompts)
            )
            return list(self.last_generation_outputs)
        outputs: list[GenerationOutput] = []
        with Qwen35GGUFResidentSession(self.model_path) as session:
            for row_index, prompt in enumerate(request.prompts):
                raise_if_generation_deadline_expired(request)
                prompt_ids = self.tokenizer.encode(prompt)
                raise_if_generation_deadline_expired(request)
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                if plan.mode is SamplingMode.GREEDY_FAST:
                    generated_ids = self._generate_greedy(session, prompt_ids, request)
                    finish_details = _gguf_finish_details(generated_ids, self.tokenizer, request)
                    outputs.append(
                        GenerationOutput(
                            text=self.tokenizer.decode(generated_ids),
                            finish_details=finish_details,
                            telemetry=_gguf_telemetry(
                                prompt_ids,
                                generated_ids,
                                request,
                                row_index=row_index,
                            ),
                        )
                    )
                else:
                    outputs.append(
                        self._generate_sampled(
                            session,
                            prompt_ids,
                            request,
                            row_index=row_index,
                        )
                    )
        self.last_generation_outputs = tuple(outputs)
        return outputs

    def _generate_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
    ) -> list[int]:
        generated_ids: list[int] = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=False)
        raise_if_generation_deadline_expired(request)
        generated_ids.append(int(result.token_id))
        if request.ignore_eos or int(result.token_id) != self.tokenizer.eos_token_id:
            remaining = request.max_tokens - 1
            if remaining > 0:
                if _session_uses_host_routed_decode(session):
                    for _ in range(remaining):
                        raise_if_generation_deadline_expired(request)
                        step = session.step(generated_ids[-1], return_logits=False)
                        raise_if_generation_deadline_expired(request)
                        generated_ids.append(int(step.token_id))
                        if (
                            not request.ignore_eos
                            and int(step.token_id) == self.tokenizer.eos_token_id
                        ):
                            break
                else:
                    with session.capture_decode_graph(
                        position=len(prompt_ids),
                        steps_per_replay=1,
                        max_replay_steps=remaining,
                        record_steps=remaining,
                    ) as graph:
                        raise_if_generation_deadline_expired(request)
                        graph.replay(remaining)
                        raise_if_generation_deadline_expired(request)
                        for token_id in graph.read_generated_token_ids(remaining):
                            raise_if_generation_deadline_expired(request)
                            generated_ids.append(int(token_id))
                            if (
                                not request.ignore_eos
                                and int(token_id) == self.tokenizer.eos_token_id
                            ):
                                break
        return generated_ids

    def _generate_sampled(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
        *,
        row_index: int,
    ) -> GenerationOutput:
        sampling_request = _request_with_tokenizer_eos(request, self.tokenizer)
        state = RowSamplingState(
            prompt_tokens=tuple(int(token) for token in prompt_ids),
            seed=row_seed_for_index(sampling_request, row_index),
            row_index=row_index,
            forced_tokens_pending=sampling_request.forced_tokens_pending,
            forced_token_reason=sampling_request.forced_token_reason,
            post_thinking_forced_tokens_pending=sampling_request.post_thinking_forced_tokens_pending,
            post_thinking_forced_token_reason=sampling_request.post_thinking_forced_token_reason,
            force_sequence_completion_token_sequences=sampling_request.force_sequence_completion_token_sequences,
            force_sequence_completion_reason=sampling_request.force_sequence_completion_reason,
            thinking_budget=thinking_budget_state_from_params(sampling_request),
        )
        samples = []
        raise_if_generation_deadline_expired(request)
        result = session.prefill(prompt_ids, return_logits=True)
        raise_if_generation_deadline_expired(request)
        sample = _select_from_gguf_logits(result, sampling_request, state)
        samples.append(sample)
        generated_ids = [int(sample.token_id)]
        if _gguf_finished(generated_ids, self.tokenizer, request):
            return _gguf_generation_output(
                self.tokenizer,
                samples,
                finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request, state),
                telemetry=_gguf_telemetry(
                    prompt_ids,
                    generated_ids,
                    request,
                    row_index=row_index,
                    sampling_state=state,
                ),
            )
        for _ in range(request.max_tokens - 1):
            raise_if_generation_deadline_expired(request)
            step = session.step(generated_ids[-1], return_logits=True)
            raise_if_generation_deadline_expired(request)
            sample = _select_from_gguf_logits(step, sampling_request, state)
            samples.append(sample)
            generated_ids.append(int(sample.token_id))
            if _gguf_finished(generated_ids, self.tokenizer, request):
                break
        return _gguf_generation_output(
            self.tokenizer,
            samples,
            finish_details=_gguf_finish_details(generated_ids, self.tokenizer, request, state),
            telemetry=_gguf_telemetry(
                prompt_ids,
                generated_ids,
                request,
                row_index=row_index,
                sampling_state=state,
            ),
        )


def _select_from_gguf_logits(
    result: Any,
    request: GenerationRequest,
    state: RowSamplingState,
):
    logits = getattr(result, "logits", None)
    if logits is None:
        raise RuntimeError("GGUF sampled generation requires logits from the resident session")
    return select_token(logits.reshape(-1), request, state)


def _request_with_tokenizer_eos(
    request: GenerationRequest,
    tokenizer: Qwen35GGUFTokenizer,
) -> GenerationRequest:
    if request.eos_token_id is not None:
        return request
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return request
    return replace(request, eos_token_id=int(eos_token_id))


def _gguf_generation_output(
    tokenizer: Qwen35GGUFTokenizer,
    samples,
    *,
    finish_details: FinishDetails,
    telemetry: GenerationTelemetry | None = None,
) -> GenerationOutput:
    token_logprobs = tuple(
        TokenLogprob(
            token_id=sample.token_id,
            token_text=tokenizer.decode([int(sample.token_id)]),
            logprob=sample.logprob,
            top_logprobs=tuple(
                (token_id, tokenizer.decode([int(token_id)]), logprob)
                for token_id, logprob in sample.top_logprobs
            ),
        )
        for sample in samples
    )
    return GenerationOutput(
        text="".join(token.token_text for token in token_logprobs),
        token_logprobs=token_logprobs,
        finish_details=finish_details,
        telemetry=telemetry,
    )


def _gguf_telemetry(
    prompt_ids: list[int] | tuple[int, ...],
    generated_ids: list[int] | tuple[int, ...],
    request: GenerationRequest,
    *,
    row_index: int,
    sampling_state: RowSamplingState | None = None,
) -> GenerationTelemetry:
    plan = plan_sampler(request)
    state_payload = _gguf_decode_state_from_sampling_state(sampling_state)
    return GenerationTelemetry.from_decode_counts(
        row_index=row_index,
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        phase=state_payload.get("phase", "done"),
        reasoning_tokens=int(state_payload.get("reasoning_tokens", 0)),
        answer_tokens=int(state_payload.get("answer_tokens", 0)),
        forced_tokens_pending=tuple(state_payload.get("forced_tokens_pending", ())),
        budget_pressure=state_payload.get("budget_pressure"),
        sampler_mode=plan.mode.value,
        stop_suffix_state=_gguf_stop_suffix_state(generated_ids, request.stop_token_sequences),
        active_processors=plan.active_processors,
        sampler_fast_path_blockers=plan.fast_path_blockers,
        sampler_fallback_reason=plan.fallback_reason,
    )


def _gguf_decode_state_from_sampling_state(state: RowSamplingState | None) -> dict[str, Any]:
    if state is None:
        return {}
    payload: dict[str, Any] = {}
    if state.forced_tokens:
        payload["forced_tokens_pending"] = state.forced_tokens
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


def _gguf_stop_suffix_state(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> dict[str, Any] | None:
    payload = token_sequence_state_for_tokens(generated_ids, stop_token_sequences).to_json_dict()
    return payload or None


def _gguf_finished(
    generated_ids: list[int] | tuple[int, ...],
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
) -> bool:
    if not generated_ids:
        return False
    token_id = int(generated_ids[-1])
    if not request.ignore_eos and int(token_id) == int(tokenizer.eos_token_id):
        return True
    if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
        return True
    for sequence in request.stop_token_sequences:
        if len(sequence) <= 0 or len(sequence) > len(generated_ids):
            continue
        if tuple(int(token) for token in generated_ids[-len(sequence) :]) == sequence:
            return True
    return False


def _gguf_finish_details(
    generated_ids: list[int] | tuple[int, ...],
    tokenizer: Qwen35GGUFTokenizer,
    request: GenerationRequest,
    state: RowSamplingState | None = None,
) -> FinishDetails:
    details: FinishDetails
    if generated_ids:
        token_id = int(generated_ids[-1])
        if not request.ignore_eos and int(token_id) == int(tokenizer.eos_token_id):
            details = FinishDetails(reason="eos", eos_token_id=token_id, sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
        if token_id in {int(stop_id) for stop_id in request.stop_token_ids}:
            details = FinishDetails(reason="stop", stop_sequence=(token_id,), sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
        sequence = _gguf_stop_sequence_match(generated_ids, request.stop_token_sequences)
        if sequence:
            details = FinishDetails(reason="stop", stop_sequence=sequence, sampler_mode=_sampler_mode_value(request))
            return finish_details_with_sampling_state(details, state)
    if len(generated_ids) >= max(0, int(request.max_tokens)):
        details = FinishDetails(reason="length", length_limit=request.max_tokens, sampler_mode=_sampler_mode_value(request))
        return finish_details_with_sampling_state(details, state)
    details = FinishDetails(reason="stop", sampler_mode=_sampler_mode_value(request))
    return finish_details_with_sampling_state(details, state)


def _gguf_stop_sequence_match(
    generated_ids: list[int] | tuple[int, ...],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    return token_sequence_state_for_tokens(generated_ids, stop_token_sequences).matched_sequence


def _sampler_mode_value(request: GenerationRequest) -> str:
    return plan_sampler(request).mode.value


def _session_uses_host_routed_decode(session: Qwen35GGUFResidentSession) -> bool:
    """Return True for GGUF paths whose decode step cannot be graph-captured yet."""

    _ = session
    return False


def make_qwen35_gguf_bringup_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


for _model in ("qwen3_5_gguf", "qwen3_5_moe_gguf"):
    for _quant in ("gguf_q4_k_m", "gguf_q8_0", "gguf_q4_1", "gguf_ud_q4_k_xl"):
        register_text_generator(
            model=_model,
            backend="hip_gfx1100",
            quant=_quant,
            factory=make_qwen35_gguf_bringup_generator,
        )


__all__ = [
    "Qwen35GGUFBringupGenerator",
    "make_qwen35_gguf_bringup_generator",
]
