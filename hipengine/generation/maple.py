"""Public greedy generation for deepgrove/maple-preview-2bit-mlx."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    register_text_generator,
)
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.maple import MapleCheckpoint, validate_maple_weight_index
from hipengine.loading.safetensors import WeightIndex
from hipengine.runtime.maple import MapleRunner
from hipengine.tokenization.maple import MapleTokenizer

_MAPLE_QUANT = "maple_ternary2"
_MAPLE_DEFAULT_CONTEXT = 4_096


@dataclass
class MapleGenerator:
    """One-model token-serial greedy generator with resident packed weights."""

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "hip_gfx1151"
    context_length: int = _MAPLE_DEFAULT_CONTEXT
    tokenizer: MapleTokenizer = field(init=False)
    checkpoint: MapleCheckpoint = field(init=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(
        default=(), init=False, repr=False
    )
    last_generation_seconds: float | None = field(default=None, init=False, repr=False)
    _runner: MapleRunner | None = field(default=None, init=False, repr=False)
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_resident_session_kv = False
    supports_stream_logprobs = False
    server_plain_ar_max_active_requests = 1
    chat_template_family = "qwen"
    reasoning_parser_name = "qwen_tags"
    tool_parser_name = "qwen_tags"

    def __post_init__(self) -> None:
        self.model_path = Path(self.weight_index.model_path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        self.context_length = int(self.context_length)
        self.checkpoint = MapleCheckpoint(
            index=self.weight_index,
            validation=validate_maple_weight_index(self.weight_index),
        )
        if self.context_length <= 0 or self.context_length > self.checkpoint.spec.max_position_embeddings:
            raise ValueError("Maple context_length is outside the checkpoint capacity")
        self.tokenizer = MapleTokenizer.from_model_path(
            self.model_path,
            model_vocab_size=self.checkpoint.spec.vocab_size,
            eos_token_id=self.checkpoint.spec.eos_token_id,
            bos_token_id=self.checkpoint.spec.bos_token_id,
        )

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize preformatted text; use ``tokenize_chat`` for a plain user message."""

        return self.tokenizer.encode(str(text))

    def tokenize_chat(self, user: str, *, system: str | None = None) -> tuple[int, ...]:
        return self.tokenizer.encode_chat(str(user), system=system)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special=skip_special)

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> tuple[GenerationOutput, ...]:
        self._validate_request(request)
        with self._lock:
            self._require_open()
            runner = self._ensure_runner()
            outputs: list[GenerationOutput] = []
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                prompt_ids = request.prompt_token_ids(row_index, self.tokenize)
                if not prompt_ids:
                    raise ValueError("Maple prompt produced no token IDs")
                if len(prompt_ids) + request.max_tokens > self.context_length:
                    raise ValueError("Maple prompt plus max_tokens exceeds context_length")
                started = time.perf_counter()
                runner.reset()
                if len(prompt_ids) <= self.checkpoint.spec.sliding_window:
                    next_step = runner.prefill_native(prompt_ids)
                else:
                    next_step = runner.prefill(prompt_ids)
                generated: list[int] = []
                finish_reason = "length"
                eos_id: int | None = None
                stop_ids = set(request.stop_token_ids)
                configured_eos = (
                    self.checkpoint.spec.eos_token_id
                    if request.eos_token_id is None
                    else int(request.eos_token_id)
                )
                for _ in range(request.max_tokens):
                    raise_if_generation_deadline_expired(request)
                    token_id = int(next_step.token_id)
                    generated.append(token_id)
                    if token_id in stop_ids:
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    if not request.ignore_eos and token_id == configured_eos:
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    next_step = runner.step(token_id)
                text = self.tokenizer.decode(generated, skip_special=False)
                outputs.append(
                    GenerationOutput(
                        text=text,
                        generated_token_ids=tuple(generated),
                        finish_details=FinishDetails(
                            reason=finish_reason,
                            eos_token_id=eos_id,
                            length_limit=(
                                request.max_tokens if finish_reason == "length" else None
                            ),
                            sampler_mode="greedy",
                            phase="decode",
                        ),
                    )
                )
                self.last_generation_seconds = time.perf_counter() - started
            self.last_generation_outputs = tuple(outputs)
            return self.last_generation_outputs

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._runner is not None:
                self._runner.close()
                self._runner = None

    def _ensure_runner(self) -> MapleRunner:
        if self._runner is not None:
            return self._runner
        started = time.perf_counter()
        self._runner = MapleRunner.load(
            self.checkpoint,
            backend=self.backend,
            max_context=self.context_length,
        )
        self._load_seconds = time.perf_counter() - started
        return self._runner

    def _validate_request(self, request: GenerationRequest) -> None:
        blockers: list[str] = []
        if request.temperature != 0.0:
            blockers.append("temperature must be 0")
        if request.top_p != 1.0 or request.top_k != 0 or request.min_p != 0.0:
            blockers.append("top-p/top-k/min-p sampling is not implemented")
        if (
            request.repetition_penalty != 1.0
            or request.presence_penalty != 0.0
            or request.frequency_penalty != 0.0
        ):
            blockers.append("logit penalties are not implemented")
        if request.logit_bias or request.suppress_token_ids:
            blockers.append("logit bias/suppression is not implemented")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences are not implemented")
        if request.forced_tokens_pending or request.post_thinking_forced_tokens_pending:
            blockers.append("forced-token queues are not implemented")
        if request.tool_call_constraint is not None or request.json_object_close_forcing:
            blockers.append("structured constraints are not implemented")
        if request.min_tokens or request.logprobs or request.top_logprobs:
            blockers.append("min_tokens/logprobs are not implemented")
        if request.kv_storage not in {"auto", "bf16"}:
            blockers.append("only BF16 KV storage is implemented")
        if blockers:
            raise NotImplementedError("Maple basic runner: " + "; ".join(blockers))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Maple generator is closed")


def make_maple_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> MapleGenerator:
    return MapleGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


def make_maple_generator_gfx1100(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> MapleGenerator:
    return MapleGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
    )


register_text_generator(
    model="maple",
    backend="hip_gfx1151",
    quant=_MAPLE_QUANT,
    factory=make_maple_generator_gfx1151,
)
register_text_generator(
    model="maple",
    backend="hip_gfx1100",
    quant=_MAPLE_QUANT,
    factory=make_maple_generator_gfx1100,
)


__all__ = [
    "MapleGenerator",
    "make_maple_generator_gfx1100",
    "make_maple_generator_gfx1151",
]
