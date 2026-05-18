"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.kvcache import resolve_kv_policy
from hipengine.loading import WeightIndex
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
    _decode_token_cached,
    _select_token,
)


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

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError("Qwen3.5/PARO generator currently supports greedy sampling only")
        if request.max_tokens == 0:
            return ["" for _ in request.prompts]
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        return [
            self._generate_one(
                runner,
                prompt,
                request.max_tokens,
                ignore_eos=request.ignore_eos,
                kv_policy=kv_policy,
            )
            for prompt in request.prompts
        ]

    def _generate_one(self, runner: Qwen35ParoNextTokenRunner, prompt: str, max_tokens: int, *, ignore_eos: bool, kv_policy) -> str:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        max_sequence_length = len(prompt_ids) + max_tokens + 1
        generated_text: list[str] = []
        with Qwen35ParoResidentSession(
            runner,
            max_sequence_length=max_sequence_length,
            kv_policy=kv_policy.create_policy(),
            kv_scale_dtype=kv_policy.scale_dtype,
            kv_scale_granularity=kv_policy.scale_granularity,
        ) as session:
            next_result = session.prefill_native(prompt_ids, sample=True)
            if next_result is None:
                raise RuntimeError("native prefill did not produce next-token logits")
            generated_text.append(next_result.token_text)
            if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
                return "".join(generated_text)

            remaining = max_tokens - 1
            if remaining:
                with session.capture_decode_graph(
                    position=len(prompt_ids),
                    steps_per_replay=1,
                    max_replay_steps=remaining,
                    record_steps=remaining,
                ) as graph:
                    graph.replay(remaining)
                    token_ids = graph.read_generated_token_ids(remaining)
                for token_id in token_ids:
                    generated_text.append(_decode_token_cached(session.tokenizer, token_id))
                    if not ignore_eos and _is_eos(session.tokenizer, token_id):
                        break
        return "".join(generated_text)

    def _get_runner(self) -> Qwen35ParoNextTokenRunner:
        if self._runner is None:
            self._runner = Qwen35ParoNextTokenRunner(
                self.model_path,
                index=self.weight_index,
                backend=self.backend,
            )
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
