"""Public generation for dense Llama-family safetensors checkpoints.

The path is prefill-then-decode on resident BF16 weights: one prompt pass fills
the KV cache, then each step appends a token and reads the context back. Token
selection is the shared :func:`hipengine.generation.sampling.select_token`, so
temperature, top-k/top-p/min-p, logit bias and the history penalties mean here
what they mean for every other generator.

Requests this path cannot serve are refused with a named reason rather than
quietly ignored, and the refusal names the capability instead of the model.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    register_text_generator,
)
from hipengine.generation.sampling import RowSamplingState, row_seed_for_index, select_token
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.llama_safetensors import (
    LlamaLoadedModel,
    load_llama_spec,
    materialize_llama_weights,
)
from hipengine.loading.safetensors import WeightIndex
from hipengine.models.llama import LlamaCapabilityMiss, LlamaSpec
from hipengine.runtime.llama_dense import (
    LlamaDenseRunner,
    default_dense_context_length,
    max_dense_prefill_tokens,
)
from hipengine.tokenization.llama_dense import LlamaDenseTokenizer

_LLAMA_DENSE_QUANT = "bf16"
_LLAMA_DENSE_RESIDENT_PATH = "llama_dense_prefill_then_decode"


@dataclass
class LlamaDenseGenerator:
    """Dense Llama decoder with a resident weight set and a per-request KV cache."""

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    backend: str = "hip_gfx1151"
    context_length: int | None = None
    #: An explicitly requested arithmetic profile. ``strict`` selects the exact
    #: block attention; anything else -- including the default -- takes the tiled
    #: kernel with the exact one as its fallback.
    execution_profile: str | None = None
    spec: LlamaSpec = field(init=False)
    tokenizer: LlamaDenseTokenizer = field(init=False)
    max_prompt_tokens: int = field(init=False)
    _loaded: LlamaLoadedModel | None = field(default=None, init=False, repr=False)
    _runner: LlamaDenseRunner | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    last_generation_seconds: float | None = field(default=None, init=False, repr=False)

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_resident_session_kv = False
    supports_stream_logprobs = False
    server_plain_ar_max_active_requests = 1
    chat_template_family = "chatml"
    execution_path = _LLAMA_DENSE_RESIDENT_PATH

    def __post_init__(self) -> None:
        self.model_path = Path(self.weight_index.model_path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        self.spec = load_llama_spec(self.weight_index)
        self.execution_profile = self._validate_execution_profile(self.execution_profile)
        self.context_length = (
            # The default is what this machine can serve: the checkpoint's
            # positions, capped by the cache that fits beside the resident
            # weights. Both attention passes tile keys, so no kernel bound is
            # tighter than that -- only memory is.
            default_dense_context_length(self.spec, self.weight_index)
            if self.context_length is None
            else self._validate_context(self.context_length)
        )
        self.tokenizer = self._build_tokenizer()
        self.max_prompt_tokens = min(
            max_dense_prefill_tokens(self.spec), self.context_length
        )

    # -- construction ------------------------------------------------------
    @staticmethod
    def _validate_execution_profile(value: str | None) -> str | None:
        if value is None:
            return None
        from hipengine.execution_profiles import resolve_requested_execution_profile

        # Unknown selectors raise here rather than being silently ignored.
        return resolve_requested_execution_profile(value).value

    @property
    def strict_arithmetic(self) -> bool:
        return self.execution_profile == "strict"

    def _validate_context(self, tokens: int) -> int:
        context = int(tokens)
        if context <= 0:
            raise ValueError("context_length must be positive")
        if context > self.spec.max_position_embeddings:
            raise ValueError(
                f"context_length {context} exceeds the checkpoint's "
                f"{self.spec.max_position_embeddings} positions"
            )
        return context

    def _build_tokenizer(self) -> LlamaDenseTokenizer:
        config = self.weight_index.config
        eos_token_id = config.get("eos_token_id")
        if eos_token_id is None:
            raise LlamaCapabilityMiss(
                "checkpoint declares no eos_token_id, so generation has no stop token"
            )
        bos_token_id = config.get("bos_token_id")
        try:
            return LlamaDenseTokenizer.from_model_path(
                self.model_path,
                model_vocab_size=self.spec.vocab_size,
                eos_token_id=int(eos_token_id),
                bos_token_id=None if bos_token_id is None else int(bos_token_id),
            )
        except (FileNotFoundError, ValueError, NotImplementedError) as exc:
            raise LlamaCapabilityMiss(f"tokenizer cannot serve this checkpoint: {exc}") from exc

    # -- serving hooks -----------------------------------------------------
    def declare_max_sequence_length(self, tokens: int) -> None:
        """Pin the context before weights load; the cache is sized from it."""

        requested = self._validate_context(tokens)
        if self._runner is not None:
            if requested != self.context_length:
                raise RuntimeError(
                    "the dense Llama context cannot change after the resident cache exists"
                )
            return
        self.context_length = requested

    def declare_serving_capacity(self, capacity: int | None) -> None:
        """This path serves one request at a time; a wider width is refused."""

        if capacity is not None and int(capacity) > 1:
            raise NotImplementedError(
                f"the dense Llama path serves 1 resident request, not {int(capacity)}; "
                "batched decode needs the paged attention path"
            )

    # -- text --------------------------------------------------------------
    def tokenize(self, text: str) -> tuple[int, ...]:
        return self.tokenizer.encode(str(text))

    def tokenize_chat(
        self,
        user: str,
        *,
        system: str | None = None,
        enable_thinking: bool = True,
    ) -> tuple[int, ...]:
        return self.tokenizer.encode_chat(
            str(user), system=system, enable_thinking=enable_thinking
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, token_ids, *, skip_special: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special=skip_special)

    # -- lifecycle ---------------------------------------------------------
    def _ensure_runner(self) -> LlamaDenseRunner:
        if self._runner is not None:
            return self._runner
        started = time.perf_counter()
        loaded = materialize_llama_weights(self.weight_index, self.spec)
        try:
            runner = LlamaDenseRunner(
                loaded, max_context=self.context_length, strict=self.strict_arithmetic
            )
        except Exception:
            loaded.free()
            raise
        self._loaded = loaded
        self._runner = runner
        self._load_seconds = time.perf_counter() - started
        return runner

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._runner is not None:
                self._runner.close()
                self._runner = None
            if self._loaded is not None:
                self._loaded.free()
                self._loaded = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("dense Llama generator is closed")

    # -- generation --------------------------------------------------------
    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> tuple[GenerationOutput, ...]:
        self._validate_request(request)
        with self._lock:
            self._require_open()
            # Tokenize and bound-check every row before the resident weights
            # load: an over-long prompt is cheaper to refuse than to load 66 GB
            # for.
            prompt_ids = tuple(
                self._prompt_ids(request, row_index)
                for row_index in range(len(request.prompts))
            )
            for ids in prompt_ids:
                self._validate_prompt_length(len(ids), request.max_tokens)
            runner = self._ensure_runner()
            outputs: list[GenerationOutput] = []
            for row_index in range(len(request.prompts)):
                raise_if_generation_deadline_expired(request)
                outputs.append(
                    self._generate_row(runner, request, row_index, prompt_ids[row_index])
                )
            return tuple(outputs)

    def _prompt_ids(self, request: GenerationRequest, row_index: int) -> Sequence[int]:
        prompt_ids = request.prompt_token_ids(row_index, self.tokenize)
        if not prompt_ids:
            raise ValueError("the prompt produced no token ids")
        return prompt_ids

    def _validate_prompt_length(self, prompt_tokens: int, max_tokens: int) -> None:
        if prompt_tokens > self.max_prompt_tokens:
            raise ValueError(
                f"prompt of {prompt_tokens} tokens exceeds the {self.max_prompt_tokens}-token "
                "prompt bound of the dense causal prefill attention, which stages one FP32 "
                "logit row per key in 64 KiB of LDS; a longer prompt needs the tiled or "
                "paged prefill path"
            )
        if prompt_tokens + max_tokens > self.context_length:
            raise ValueError(
                f"prompt of {prompt_tokens} tokens plus {max_tokens} generated "
                f"tokens exceeds the {self.context_length}-token context"
            )

    def _generate_row(
        self,
        runner: LlamaDenseRunner,
        request: GenerationRequest,
        row_index: int,
        prompt_ids: Sequence[int],
    ) -> GenerationOutput:
        started = time.perf_counter()

        runner.reset()
        logits = runner.prefill(prompt_ids)
        if logits is None:
            raise RuntimeError("the dense Llama prompt pass returned no logits")
        row_logits = np.asarray(logits[0], dtype=np.float32)

        state = self._sampling_state(request, prompt_ids, row_index)
        generated: list[int] = []
        finish_reason = "length"
        eos_token_id: int | None = None
        stop_ids = set(request.stop_token_ids)
        configured_eos = (
            self.tokenizer.eos_token_id if request.eos_token_id is None else int(request.eos_token_id)
        )
        sampler_mode = "greedy"

        for step in range(request.max_tokens):
            raise_if_generation_deadline_expired(request)
            selection = select_token(
                row_logits,
                request,
                state,
                token_text_for_id=self.tokenizer.decode_token,
            )
            sampler_mode = "greedy" if request.temperature <= 0.0 else str(selection.mode.value)
            token_id = int(selection.token_id)
            generated.append(token_id)
            if token_id in stop_ids:
                finish_reason = "stop"
                eos_token_id = token_id
                break
            if not request.ignore_eos and token_id == configured_eos:
                finish_reason = "stop"
                eos_token_id = token_id
                break
            if step + 1 >= request.max_tokens:
                break
            row_logits = runner.decode_step(token_id)

        text = self.tokenizer.decode(generated, skip_special=False)
        self.last_generation_seconds = time.perf_counter() - started
        return GenerationOutput(
            text=text,
            generated_token_ids=tuple(generated),
            prompt_tokens=len(prompt_ids),
            finish_details=FinishDetails(
                reason=finish_reason,
                eos_token_id=eos_token_id,
                length_limit=request.max_tokens if finish_reason == "length" else None,
                sampler_mode=sampler_mode,
                phase="decode",
            ),
        )

    def _sampling_state(
        self,
        request: GenerationRequest,
        prompt_ids: Sequence[int],
        row_index: int,
    ) -> RowSamplingState:
        return RowSamplingState(
            prompt_tokens=prompt_ids,
            seed=row_seed_for_index(request, row_index),
            row_index=row_index,
        )

    # -- refusals ----------------------------------------------------------
    def _validate_request(self, request: GenerationRequest) -> None:
        blockers: list[str] = []
        if request.grammar is not None:
            blockers.append("grammar-constrained decoding")
        if request.tool_call_constraint is not None:
            blockers.append("tool-call constraints")
        if request.json_object_close_forcing:
            blockers.append("json_object close forcing")
        if request.thinking_hard_token_cap is not None or request.thinking_close_token_ids:
            blockers.append("thinking budgets")
        if request.forced_tokens_pending or request.post_thinking_forced_tokens_pending:
            blockers.append("forced-token queues")
        if request.stop_token_sequences:
            blockers.append("multi-token stop sequences")
        if request.force_sequence_completion_token_sequences:
            blockers.append("forced sequence completion")
        if request.logprobs or request.top_logprobs:
            blockers.append("logprobs")
        if request.resident_session_key is not None:
            blockers.append("resident session KV")
        if request.speculative_mtp_static_eligibility is not None or request.speculative_mtp_automatic:
            blockers.append("speculative MTP")
        if request.kv_storage not in {"auto", "bf16"}:
            blockers.append(f"kv_storage={request.kv_storage!r} (this path stores BF16)")
        if blockers:
            raise NotImplementedError(
                "the dense Llama path does not implement: " + ", ".join(blockers)
            )


def _make_generator(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
    backend: str,
    max_sequence_length: int | None = None,
    execution_profile: str | None = None,
) -> LlamaDenseGenerator:
    generator = LlamaDenseGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend=backend,
        execution_profile=execution_profile,
    )
    if max_sequence_length is not None:
        generator.declare_max_sequence_length(int(max_sequence_length))
    return generator


def make_llama_dense_generator_gfx1100(**kwargs: Any) -> LlamaDenseGenerator:
    return _make_generator(backend="hip_gfx1100", **kwargs)


def make_llama_dense_generator_gfx1151(**kwargs: Any) -> LlamaDenseGenerator:
    return _make_generator(backend="hip_gfx1151", **kwargs)


register_text_generator(
    model="llama_dense",
    backend="hip_gfx1100",
    quant=_LLAMA_DENSE_QUANT,
    factory=make_llama_dense_generator_gfx1100,
)
register_text_generator(
    model="llama_dense",
    backend="hip_gfx1151",
    quant=_LLAMA_DENSE_QUANT,
    factory=make_llama_dense_generator_gfx1151,
)


__all__ = [
    "LlamaDenseGenerator",
    "make_llama_dense_generator_gfx1100",
    "make_llama_dense_generator_gfx1151",
]
