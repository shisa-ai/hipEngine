"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import os
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
    _session: Qwen35ParoResidentSession | None = field(default=None, init=False, repr=False)
    _session_capacity: int = field(default=0, init=False, repr=False)
    _session_kv_key: tuple[str, str, str, int] | None = field(default=None, init=False, repr=False)

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError(
                "Qwen3.5/PARO generator currently supports greedy sampling only"
            )
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

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        if len(request.prompts) != 1:
            raise ValueError("streaming currently supports exactly one prompt")
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError(
                "Qwen3.5/PARO generator currently supports greedy sampling only"
            )
        if request.max_tokens == 0:
            return
        runner = self._get_runner()
        kv_policy = resolve_kv_policy(
            request.kv_storage,
            scale_dtype=request.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity,
        )
        yield from self._stream_one(
            runner,
            request.prompts[0],
            request.max_tokens,
            ignore_eos=request.ignore_eos,
            kv_policy=kv_policy,
        )

    def _generate_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
    ) -> str:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        generated_text: list[str] = []
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
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

    def _stream_one(
        self,
        runner: Qwen35ParoNextTokenRunner,
        prompt: str,
        max_tokens: int,
        *,
        ignore_eos: bool,
        kv_policy,
    ) -> Iterator[str]:
        _last_token_id, prompt_ids = _select_token(Path(self.model_path), prompt, None)
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        required_sequence_length = len(prompt_ids) + max_tokens + 1
        session_capacity = _session_capacity_for(required_sequence_length)
        session = self._get_session(
            runner,
            max_sequence_length=session_capacity,
            kv_policy=kv_policy,
        )
        next_result = session.prefill_native(prompt_ids, sample=True)
        if next_result is None:
            raise RuntimeError("native prefill did not produce next-token logits")
        yield next_result.token_text
        if not ignore_eos and _is_eos(session.tokenizer, next_result.token_id):
            return

        current_token_id = next_result.token_id
        for position in range(len(prompt_ids), len(prompt_ids) + max_tokens - 1):
            result = session.step(current_token_id, position=position, sample=True)
            if result is None:
                raise RuntimeError("decode step did not produce next-token logits")
            yield result.token_text
            current_token_id = result.token_id
            if not ignore_eos and _is_eos(session.tokenizer, result.token_id):
                return

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
    ) -> Qwen35ParoResidentSession:
        kv_key = (
            kv_policy.storage_dtype.value,
            kv_policy.scale_dtype.value,
            kv_policy.scale_granularity,
            int(kv_policy.block_size),
        )
        if (
            self._session is None
            or self._session_capacity < max_sequence_length
            or self._session_kv_key != kv_key
        ):
            self.close()
            self._session = Qwen35ParoResidentSession(
                runner,
                max_sequence_length=max_sequence_length,
                kv_policy=kv_policy.create_policy(),
                kv_scale_dtype=kv_policy.scale_dtype,
                kv_scale_granularity=kv_policy.scale_granularity,
            )
            self._session_capacity = max_sequence_length
            self._session_kv_key = kv_key
        else:
            self._session.reset()
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._session_capacity = 0
        self._session_kv_key = None


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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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
