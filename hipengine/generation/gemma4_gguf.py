"""Public greedy generation for Gemma 4 GGUF artifacts.

This is the adapter between ``LLM.generate()`` and
:class:`hipengine.runtime.gemma4.Gemma4Runner`. It owns the request protocol -
tokenization, the decode loop, stop conditions, and the output shape - and
delegates every forward pass to the runner.

Only the greedy path is implemented. ``_validate_request`` names each unimplemented
sampling feature rather than silently ignoring it, so a request that asks for
temperature or top-p fails loudly instead of quietly returning greedy output.
"""

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
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.safetensors import WeightIndex
from hipengine.runtime.gemma4 import (
    Gemma4DeviceWeights,
    Gemma4Runner,
    load_gemma4_device_weights,
)
from hipengine.tokenization.gguf import Gemma4GGUFTokenizer

_GEMMA4_QUANT = "gguf_q4_k_m"
_GEMMA4_DEFAULT_CONTEXT = 8_192


@dataclass
class Gemma4GGUFGenerator:
    """Greedy generator over a resident Gemma 4 GGUF artifact."""

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "hip_gfx1100"
    context_length: int = _GEMMA4_DEFAULT_CONTEXT
    last_generation_outputs: tuple[GenerationOutput, ...] = field(
        default=(), init=False, repr=False
    )
    last_generation_seconds: float | None = field(default=None, init=False, repr=False)
    _reader: GGUFReader | None = field(default=None, init=False, repr=False)
    _weights: Gemma4DeviceWeights | None = field(default=None, init=False, repr=False)
    _runner: Gemma4Runner | None = field(default=None, init=False, repr=False)
    _tokenizer: Gemma4GGUFTokenizer | None = field(default=None, init=False, repr=False)
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_resident_session_kv = False
    supports_stream_logprobs = False
    chat_template_family = "gemma"
    reasoning_parser_name = "gemma_tags"
    tool_parser_name = "gemma_tags"

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        self.context_length = int(self.context_length)
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

    # --- tokenizer ------------------------------------------------------

    @property
    def reader(self) -> GGUFReader:
        if self._reader is None:
            self._reader = GGUFReader(self.model_path)
        return self._reader

    @property
    def tokenizer(self) -> Gemma4GGUFTokenizer:
        if self._tokenizer is None:
            self._tokenizer = Gemma4GGUFTokenizer.from_gguf_info(self.reader.info)
        return self._tokenizer

    def tokenize(self, text: str) -> tuple[int, ...]:
        """Tokenize preformatted text; use ``tokenize_chat`` for a plain message."""

        return tuple(self.tokenizer.encode(str(text)))

    def tokenize_chat(self, user: str, *, system: str | None = None) -> tuple[int, ...]:
        template = self.tokenizer.chat_template
        if not template:
            raise ValueError(
                "the artifact carries no tokenizer.chat_template, so a chat prompt "
                "cannot be formatted; pass preformatted text instead"
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(user)})
        rendered = _render_chat_template(template, messages)
        return tuple(self.tokenizer.encode(rendered, add_special_tokens=True))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    # --- generation -----------------------------------------------------

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
                    raise ValueError("Gemma 4 prompt produced no token IDs")
                if len(prompt_ids) + request.max_tokens > self.context_length:
                    raise ValueError(
                        f"Gemma 4 prompt ({len(prompt_ids)} tokens) plus max_tokens "
                        f"({request.max_tokens}) exceeds context_length {self.context_length}"
                    )
                started = time.perf_counter()
                runner.reset()
                logits = runner.forward(list(prompt_ids))
                generated: list[int] = []
                finish_reason = "length"
                eos_id: int | None = None
                stop_ids = set(request.stop_token_ids)
                configured_eos = (
                    self.tokenizer.eos_token_id
                    if request.eos_token_id is None
                    else int(request.eos_token_id)
                )
                for _ in range(request.max_tokens):
                    raise_if_generation_deadline_expired(request)
                    token_id = runner.next_token(logits)
                    generated.append(token_id)
                    if token_id in stop_ids:
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    if (
                        not request.ignore_eos
                        and configured_eos is not None
                        and token_id == configured_eos
                    ):
                        finish_reason = "stop"
                        eos_id = token_id
                        break
                    logits = runner.forward([token_id])
                outputs.append(
                    GenerationOutput(
                        text=self.tokenizer.decode(generated, skip_special=False),
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
            if self._weights is not None:
                self._weights.free()
                self._weights = None

    def _ensure_runner(self) -> Gemma4Runner:
        if self._runner is not None:
            return self._runner
        started = time.perf_counter()
        self._weights = load_gemma4_device_weights(self.reader, backend=self.backend)
        self._runner = Gemma4Runner(
            weights=self._weights,
            capacity=self.context_length,
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
            raise NotImplementedError("Gemma 4 basic runner: " + "; ".join(blockers))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Gemma 4 generator is closed")


def _render_chat_template(template: str, messages: list[dict[str, str]]) -> str:
    """Render a Jinja chat template, refusing anything this adapter cannot run.

    The templates real Gemma artifacts carry use only ``messages``,
    ``add_generation_prompt``, ``bos_token`` and ``eos_token``. Rather than
    half-implement Jinja, this raises for a template that needs more, so a prompt
    is never silently formatted differently from what the model was trained on.
    """

    try:
        from jinja2 import Environment
    except ImportError as exc:  # pragma: no cover - jinja2 is a hard dependency here
        raise ImportError(
            "rendering a chat template requires jinja2; pass preformatted text instead"
        ) from exc

    environment = Environment(autoescape=False)  # noqa: S701 - not HTML
    try:
        compiled = environment.from_string(template)
        return compiled.render(
            messages=messages,
            add_generation_prompt=True,
            bos_token="<bos>",
            eos_token="<eos>",
        )
    except Exception as exc:
        raise ValueError(
            f"the artifact's chat template could not be rendered ({exc}); "
            "pass preformatted text instead"
        ) from exc


def make_gemma4_generator_gfx1100(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Gemma4GGUFGenerator:
    return Gemma4GGUFGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1100",
    )


register_text_generator(
    model="gemma4_gguf",
    backend="hip_gfx1100",
    quant=_GEMMA4_QUANT,
    factory=make_gemma4_generator_gfx1100,
)


__all__ = [
    "Gemma4GGUFGenerator",
    "make_gemma4_generator_gfx1100",
]
