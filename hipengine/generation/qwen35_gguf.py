"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.generation.sampling import (
    RowSamplingState,
    SamplingMode,
    plan_sampler,
    row_seed_for_index,
    select_token,
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

    def __post_init__(self) -> None:
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        plan = plan_sampler(request)
        if request.max_tokens == 0:
            return ["" for _ in request.prompts]
        outputs: list[str] = []
        with Qwen35GGUFResidentSession(self.model_path) as session:
            for row_index, prompt in enumerate(request.prompts):
                prompt_ids = self.tokenizer.encode(prompt)
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                if plan.mode is SamplingMode.GREEDY_FAST:
                    generated_ids = self._generate_greedy(session, prompt_ids, request)
                else:
                    generated_ids = self._generate_sampled(
                        session,
                        prompt_ids,
                        request,
                        row_index=row_index,
                    )
                outputs.append(self.tokenizer.decode(generated_ids))
        return outputs

    def _generate_greedy(
        self,
        session: Qwen35GGUFResidentSession,
        prompt_ids: list[int],
        request: GenerationRequest,
    ) -> list[int]:
        generated_ids: list[int] = []
        result = session.prefill(prompt_ids, return_logits=False)
        generated_ids.append(int(result.token_id))
        if request.ignore_eos or int(result.token_id) != self.tokenizer.eos_token_id:
            remaining = request.max_tokens - 1
            if remaining > 0:
                if _session_uses_host_routed_decode(session):
                    for _ in range(remaining):
                        step = session.step(generated_ids[-1], return_logits=False)
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
                        graph.replay(remaining)
                        for token_id in graph.read_generated_token_ids(remaining):
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
    ) -> list[int]:
        state = RowSamplingState(
            prompt_tokens=tuple(int(token) for token in prompt_ids),
            seed=row_seed_for_index(request, row_index),
            row_index=row_index,
        )
        generated_ids: list[int] = []
        result = session.prefill(prompt_ids, return_logits=True)
        token_id = _select_from_gguf_logits(result, request, state)
        generated_ids.append(token_id)
        if _gguf_finished(generated_ids, self.tokenizer, request):
            return generated_ids
        for _ in range(request.max_tokens - 1):
            step = session.step(generated_ids[-1], return_logits=True)
            token_id = _select_from_gguf_logits(step, request, state)
            generated_ids.append(token_id)
            if _gguf_finished(generated_ids, self.tokenizer, request):
                break
        return generated_ids


def _select_from_gguf_logits(
    result: Any,
    request: GenerationRequest,
    state: RowSamplingState,
) -> int:
    logits = getattr(result, "logits", None)
    if logits is None:
        raise RuntimeError("GGUF sampled generation requires logits from the resident session")
    sample = select_token(logits.reshape(-1), request, state)
    return int(sample.token_id)


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
