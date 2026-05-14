"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.loading import WeightIndex
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
    _select_token,
)


@dataclass
class Qwen35ParoOneTokenGenerator:
    """Greedy Qwen3.5/PARO generator backed by resident c=1 execution.

    The implementation is still serial across prompts, but each prompt runs real
    token-by-token prefill followed by multi-token autoregressive decode using
    the resident HIP layer chain.
    """

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    lm_head_chunk: int = 4096
    _runner: Qwen35ParoNextTokenRunner | None = field(default=None, init=False, repr=False)

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError("Qwen3.5/PARO generator currently supports greedy sampling only")
        if request.max_tokens == 0:
            return ["" for _ in request.prompts]
        runner = self._get_runner()
        return [self._generate_one(runner, prompt, request.max_tokens, ignore_eos=request.ignore_eos) for prompt in request.prompts]

    def _generate_one(self, runner: Qwen35ParoNextTokenRunner, prompt: str, max_tokens: int, *, ignore_eos: bool) -> str:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        max_sequence_length = len(prompt_ids) + max_tokens + 1
        generated = []
        with Qwen35ParoResidentSession(runner, max_sequence_length=max_sequence_length) as session:
            next_result = None
            for position, token_id in enumerate(prompt_ids):
                next_result = session.step(token_id, position=position, sample=(position == len(prompt_ids) - 1))
            if next_result is None:
                raise RuntimeError("prefill did not produce next-token logits")
            generated.append(next_result)
            if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
                return "".join(item.token_text for item in generated)
            current = next_result
            for offset in range(1, max_tokens):
                current = session.step(current.token_id, position=len(prompt_ids) + offset - 1)
                if current is None:
                    raise RuntimeError("decode step did not produce next-token logits")
                generated.append(current)
                if not ignore_eos and _is_eos(session.tokenizer, current.token_id):
                    break
        return "".join(item.token_text for item in generated)

    def _get_runner(self) -> Qwen35ParoNextTokenRunner:
        if self._runner is None:
            self._runner = Qwen35ParoNextTokenRunner(self.model_path, index=self.weight_index)
        return self._runner


def _is_eos(tokenizer: Any | None, token_id: int) -> bool:
    if tokenizer is None:
        return False
    try:
        eos_id = getattr(tokenizer, "token_to_id")("<|endoftext|>")
    except Exception:
        eos_id = None
    return eos_id is not None and int(token_id) == int(eos_id)


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
    )


register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1100",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator,
)
