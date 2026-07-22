"""Public greedy generation for Poolside Laguna S 2.1 GGUF.

The initial route deliberately stays c=1 and token-serial.  It owns one shared
resident weight set, creates isolated eager KV/scratch state per request, and
fails closed for sampling or speculative features that the Laguna runner has
not correctness-gated yet.
"""

from __future__ import annotations

import codecs
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
    register_text_generator,
)
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.loading.laguna_gguf_materialize import (
    LagunaGGUFResidentWeights,
    materialize_laguna_gguf_weights,
)
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.tokenization.gguf import LagunaGGUFTokenizer

_LAGUNA_INITIAL_CONTEXT = 4_096
_LAGUNA_QUANT = "gguf_q4_k_m"
_LAGUNA_EXECUTION_PATH = "laguna_eager_c1"


@dataclass(frozen=True)
class _LagunaTokenStep:
    token_id: int
    generated_ids: tuple[int, ...]
    finish_details: FinishDetails | None
    telemetry: GenerationTelemetry


@dataclass
class LagunaGGUFGenerator:
    """One-model c=1 greedy generator with shared immutable resident weights."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    backend: str = "hip_gfx1151"
    context_length: int = _LAGUNA_INITIAL_CONTEXT
    server_plain_ar_max_active_requests: int = 1
    tokenizer: LagunaGGUFTokenizer = field(init=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(
        default=(), init=False, repr=False
    )
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _runtime: HipRuntime | None = field(default=None, init=False, repr=False)
    _weights: LagunaGGUFResidentWeights | None = field(default=None, init=False, repr=False)
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_stream_logprobs = False

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser().resolve()
        self.backend = resolve_backend(self.backend)
        if self.backend != "hip_gfx1151":
            raise ValueError(
                "Laguna public generation is currently registered only for hip_gfx1151"
            )
        self.context_length = int(self.context_length)
        if self.context_length <= 0 or self.context_length > _LAGUNA_INITIAL_CONTEXT:
            raise ValueError(
                f"Laguna public context_length must be within [1, {_LAGUNA_INITIAL_CONTEXT}]"
            )
        self.tokenizer = LagunaGGUFTokenizer.from_gguf_info(self.weight_index)

    @property
    def repacked_cache_path(self) -> Path | None:
        """Return the validated sibling cache candidate, if one exists."""

        candidate = self.model_path.with_suffix(".hipengine-repacked-v1")
        return candidate if candidate.is_dir() else None

    @property
    def resident_weights(self) -> LagunaGGUFResidentWeights | None:
        return self._weights

    def tokenize(self, text: str) -> tuple[int, ...]:
        # Public Laguna input is preformatted text. A rendered Poolside prompt
        # already begins with its explicit EOS/BOS marker, so never add another.
        return tuple(int(token) for token in self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def detokenize(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            tuple(int(token) for token in token_ids),
            skip_special=bool(skip_special),
        )

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: Any | None = None,
    ) -> int:
        del sampling_params
        requested = self.context_length if max_sequence_length is None else int(max_sequence_length)
        if requested <= 0 or requested > self.context_length:
            raise ValueError(
                f"Laguna public max_sequence_length must be within [1, {self.context_length}]"
            )
        with self._lock:
            self._prepare_locked()
        return requested

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        del sampling_params, release_after_probe
        if int(max_batch_size) != 1:
            raise NotImplementedError("Laguna public generation supports only batch size 1")
        required = int(max_prompt_tokens) + max(0, int(max_new_tokens) - 1)
        if required <= 0 or required > self.context_length:
            raise ValueError(
                f"Laguna request requires {required} positions; public limit is {self.context_length}"
            )
        with self._lock:
            self._prepare_locked()
            session = self._open_session_locked()
            try:
                resident_nbytes = int(session.resident_nbytes)
            finally:
                session.close()
        return {
            "schema": 1,
            "backend": self.backend,
            "execution_path": _LAGUNA_EXECUTION_PATH,
            "max_batch_size": 1,
            "max_sequence_length": required,
            "resident_session_nbytes": resident_nbytes,
            "released_after_probe": True,
        }

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        prompt_ids = self._prepare_request(request)
        if request.max_tokens == 0:
            output = self._empty_output(prompt_ids, request)
            self._record_outputs((output,), prompt_ids, request)
            return [output]

        started = time.perf_counter()
        generated_ids: list[int] = []
        finish: FinishDetails | None = None
        final_telemetry: GenerationTelemetry | None = None
        for step in self._token_steps(request, prompt_ids):
            generated_ids.append(step.token_id)
            final_telemetry = step.telemetry
            if step.finish_details is not None:
                finish = step.finish_details
        if finish is None or final_telemetry is None:
            raise RuntimeError("Laguna generation ended without terminal metadata")
        visible_ids = _visible_generated_ids(
            generated_ids,
            finish,
            tokenizer=self.tokenizer,
        )
        output = GenerationOutput(
            text=self.tokenizer.decode(visible_ids),
            generated_token_ids=tuple(generated_ids),
            finish_details=finish,
            telemetry=_with_total_timing(final_telemetry, started),
        )
        self._record_outputs((output,), prompt_ids, request)
        return [output]

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        for chunk in self.stream_detailed(request):
            yield chunk.text

    def stream_detailed(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        prompt_ids = self._prepare_request(request)
        if request.max_tokens == 0:
            output = self._empty_output(prompt_ids, request)
            self._record_outputs((output,), prompt_ids, request)
            yield GenerationStreamChunk(
                text="",
                finish_details=output.finish_details,
                telemetry=output.telemetry,
                generated_token_ids=(),
            )
            return

        started = time.perf_counter()
        pending: list[int] = []
        visible_parts: list[str] = []
        decoder = _IncrementalLagunaDecoder(self.tokenizer)
        longest_sequence = max(
            (len(sequence) for sequence in request.stop_token_sequences),
            default=1,
        )
        hold_tokens = max(0, longest_sequence - 1)
        terminal_output: GenerationOutput | None = None
        for step in self._token_steps(request, prompt_ids):
            pending.append(step.token_id)
            finish = step.finish_details
            if finish is None:
                safe_count = max(0, len(pending) - hold_tokens)
                safe_ids = pending[:safe_count]
                del pending[:safe_count]
                text = decoder.feed(_filter_output_specials(safe_ids, self.tokenizer))
                if text:
                    visible_parts.append(text)
                    yield GenerationStreamChunk(text=text, telemetry=step.telemetry)
                continue

            suppressed = _suppressed_suffix_length(finish)
            safe_ids = pending[:-suppressed] if suppressed else pending
            text = decoder.feed(
                _filter_output_specials(safe_ids, self.tokenizer),
                final=True,
            )
            if text:
                visible_parts.append(text)
            terminal_telemetry = _with_total_timing(step.telemetry, started)
            terminal_output = GenerationOutput(
                text="".join(visible_parts),
                generated_token_ids=step.generated_ids,
                finish_details=finish,
                telemetry=terminal_telemetry,
            )
            yield GenerationStreamChunk(
                text=text,
                finish_details=finish,
                telemetry=terminal_telemetry,
                generated_token_ids=step.generated_ids,
            )
        if terminal_output is None:
            raise RuntimeError("Laguna streaming ended without a terminal chunk")
        self._record_outputs((terminal_output,), prompt_ids, request)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            weights, self._weights = self._weights, None
            if weights is not None:
                weights.free(runtime=self._runtime)

    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        _validate_public_request(request)
        raise_if_generation_deadline_expired(request)
        prompt = request.prompts[0]
        prompt_ids = (
            self.tokenize(prompt)
            if isinstance(prompt, str)
            else tuple(int(token) for token in prompt)
        )
        if not prompt_ids:
            raise ValueError("Laguna prompt tokenization produced no token IDs")
        required = len(prompt_ids) + max(0, int(request.max_tokens) - 1)
        if required > self.context_length:
            raise ValueError(
                f"Laguna request requires {required} positions; public limit is {self.context_length}"
            )
        return prompt_ids

    def _token_steps(
        self,
        request: GenerationRequest,
        prompt_ids: tuple[int, ...],
    ) -> Iterator[_LagunaTokenStep]:
        with self._lock:
            self._prepare_locked()
            raise_if_generation_deadline_expired(request)
            session = self._open_session_locked()
            generated: list[int] = []
            prefill_started = time.perf_counter()
            prefill_seconds = 0.0
            decode_seconds = 0.0
            try:
                result = session.prefill(prompt_ids)
                prefill_seconds = time.perf_counter() - prefill_started
                raise_if_generation_deadline_expired(request)
                for step_index in range(int(request.max_tokens)):
                    token_id = int(result.next_token_id)
                    generated.append(token_id)
                    finish = _laguna_finish_details(
                        generated,
                        tokenizer=self.tokenizer,
                        request=request,
                    )
                    telemetry = _laguna_telemetry(
                        prompt_ids,
                        generated,
                        finish=finish,
                        prefill_seconds=prefill_seconds,
                        decode_seconds=decode_seconds,
                    )
                    yield _LagunaTokenStep(
                        token_id=token_id,
                        generated_ids=tuple(generated),
                        finish_details=finish,
                        telemetry=telemetry,
                    )
                    if finish is not None:
                        return
                    raise_if_generation_deadline_expired(request)
                    decode_started = time.perf_counter()
                    result = session.forward_token(token_id)
                    decode_seconds += time.perf_counter() - decode_started
                    raise_if_generation_deadline_expired(request)
                raise RuntimeError(f"Laguna token loop exhausted unexpectedly at step {step_index}")
            finally:
                session.close()

    def _prepare_locked(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna generator is closed")
        if self._weights is not None:
            return
        runtime = get_hip_runtime()
        started = time.perf_counter()
        weights = materialize_laguna_gguf_weights(
            self.model_path,
            context_length=self.context_length,
            runtime=runtime,
            backend=self.backend,
            repacked_cache=self.repacked_cache_path,
        )
        self._runtime = runtime
        self._weights = weights
        self._load_seconds = time.perf_counter() - started

    def _open_session_locked(self) -> LagunaGGUFResidentSession:
        if self._weights is None or self._runtime is None:
            raise RuntimeError("Laguna resident weights are not prepared")
        return LagunaGGUFResidentSession(
            resident_weights=self._weights,
            context_length=self.context_length,
            backend=self.backend,
            runtime=self._runtime,
        )

    def _empty_output(
        self,
        prompt_ids: tuple[int, ...],
        request: GenerationRequest,
    ) -> GenerationOutput:
        finish = FinishDetails(
            reason="length",
            length_limit=0,
            sampler_mode="greedy_fast",
        )
        return GenerationOutput(
            text="",
            generated_token_ids=(),
            finish_details=finish,
            telemetry=_laguna_telemetry(
                prompt_ids,
                (),
                finish=finish,
                prefill_seconds=0.0,
                decode_seconds=0.0,
            ),
        )

    def _record_outputs(
        self,
        outputs: tuple[GenerationOutput, ...],
        prompt_ids: tuple[int, ...],
        request: GenerationRequest,
    ) -> None:
        self.last_generation_outputs = outputs
        generated = outputs[0].generated_token_ids or ()
        self.last_batch_generation = {
            "path": _LAGUNA_EXECUTION_PATH,
            "backend": self.backend,
            "quant": _LAGUNA_QUANT,
            "batch_size": 1,
            "prompt_lengths": [len(prompt_ids)],
            "decode_steps": len(generated),
            "max_tokens": int(request.max_tokens),
            "resident_weights": self._weights is not None,
            "repacked_cache": (
                None if self.repacked_cache_path is None else str(self.repacked_cache_path)
            ),
            "model_load_seconds": self._load_seconds,
            "native_caware_decode": False,
            "serial_decode_fallback": False,
            "throughput_claim_eligible": False,
        }


class _IncrementalLagunaDecoder:
    """Decode byte-BPE pieces without emitting transient replacement glyphs."""

    def __init__(self, tokenizer: LagunaGGUFTokenizer) -> None:
        self.tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._native_byte_pieces = bool(getattr(tokenizer, "byte_decoder", None))

    def feed(self, token_ids: Sequence[int], *, final: bool = False) -> str:
        if not self._native_byte_pieces:
            text = self.tokenizer.decode(token_ids)
            return text
        output: list[str] = []
        byte_decoder = self.tokenizer.byte_decoder
        for token_id in token_ids:
            idx = int(token_id)
            token = self.tokenizer.tokens[idx]
            if all(char in byte_decoder for char in token):
                data = bytes(byte_decoder[char] for char in token)
                output.append(self._decoder.decode(data, final=False))
                continue
            output.append(self._decoder.decode(b"", final=True))
            self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            output.append(token)
        if final:
            output.append(self._decoder.decode(b"", final=True))
        return "".join(output)


def _validate_public_request(request: GenerationRequest) -> None:
    if len(request.prompts) != 1:
        raise ValueError("Laguna public generation requires exactly one prompt")
    if request.max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    unsupported: list[str] = []
    if request.temperature != 0.0:
        unsupported.append("temperature")
    if request.top_p != 1.0:
        unsupported.append("top_p")
    if request.top_k != 0:
        unsupported.append("top_k")
    if request.min_p != 0.0:
        unsupported.append("min_p")
    if request.repetition_penalty != 1.0:
        unsupported.append("repetition_penalty")
    if request.presence_penalty != 0.0:
        unsupported.append("presence_penalty")
    if request.frequency_penalty != 0.0:
        unsupported.append("frequency_penalty")
    for name in (
        "logit_bias",
        "suppress_token_ids",
        "forced_tokens_pending",
        "post_thinking_forced_tokens_pending",
        "force_sequence_completion_token_sequences",
        "thinking_close_token_ids",
    ):
        if getattr(request, name):
            unsupported.append(name)
    if request.json_object_close_forcing:
        unsupported.append("json_object_close_forcing")
    if request.thinking_hard_token_cap is not None or request.thinking_soft_close_window:
        unsupported.append("thinking_budget")
    if request.logprobs or request.top_logprobs:
        unsupported.append("logprobs")
    if request.kv_storage not in {"auto", "bf16"}:
        unsupported.append("kv_storage")
    if request.kv_scale_dtype != "fp16":
        unsupported.append("kv_scale_dtype")
    if request.kv_scale_granularity != "per_token_head":
        unsupported.append("kv_scale_granularity")
    if unsupported:
        raise NotImplementedError(
            "Laguna public generation currently supports greedy BF16 c=1 only; "
            f"unsupported request fields: {', '.join(unsupported)}"
        )


def _laguna_finish_details(
    generated_ids: Sequence[int],
    *,
    tokenizer: LagunaGGUFTokenizer,
    request: GenerationRequest,
) -> FinishDetails | None:
    token_id = int(generated_ids[-1])
    can_stop = len(generated_ids) >= int(request.min_tokens)
    if can_stop and not request.ignore_eos:
        eos_id = (
            tokenizer.eos_token_id if request.eos_token_id is None else int(request.eos_token_id)
        )
        if eos_id is not None and token_id == int(eos_id):
            return FinishDetails(
                reason="eos",
                eos_token_id=token_id,
                sampler_mode="greedy_fast",
            )
        if tokenizer.eot_token_id is not None and token_id == int(tokenizer.eot_token_id):
            return FinishDetails(
                reason="stop",
                stop_sequence=(token_id,),
                sampler_mode="greedy_fast",
            )
    if can_stop and token_id in set(request.stop_token_ids):
        return FinishDetails(
            reason="stop",
            stop_sequence=(token_id,),
            sampler_mode="greedy_fast",
        )
    if can_stop:
        for sequence in request.stop_token_sequences:
            width = len(sequence)
            if width and tuple(generated_ids[-width:]) == tuple(sequence):
                return FinishDetails(
                    reason="stop",
                    stop_sequence=tuple(sequence),
                    sampler_mode="greedy_fast",
                )
    if len(generated_ids) >= int(request.max_tokens):
        return FinishDetails(
            reason="length",
            length_limit=int(request.max_tokens),
            sampler_mode="greedy_fast",
        )
    return None


def _suppressed_suffix_length(finish: FinishDetails) -> int:
    if finish.stop_sequence:
        return len(finish.stop_sequence)
    if finish.reason == "eos" and finish.eos_token_id is not None:
        return 1
    return 0


def _filter_output_specials(
    token_ids: Sequence[int],
    tokenizer: LagunaGGUFTokenizer,
) -> tuple[int, ...]:
    suppressed = {
        int(token)
        for token in (tokenizer.eos_token_id, tokenizer.eot_token_id)
        if token is not None
    }
    return tuple(int(token) for token in token_ids if int(token) not in suppressed)


def _visible_generated_ids(
    generated_ids: Sequence[int],
    finish: FinishDetails,
    *,
    tokenizer: LagunaGGUFTokenizer,
) -> tuple[int, ...]:
    suppressed = _suppressed_suffix_length(finish)
    visible = generated_ids[:-suppressed] if suppressed else generated_ids
    return _filter_output_specials(visible, tokenizer)


def _laguna_telemetry(
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    *,
    finish: FinishDetails | None,
    prefill_seconds: float,
    decode_seconds: float,
) -> GenerationTelemetry:
    suppressed = 0 if finish is None else _suppressed_suffix_length(finish)
    answer_tokens = max(0, len(generated_ids) - suppressed)
    return GenerationTelemetry.from_decode_counts(
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        phase="done" if finish is not None else "answer",
        sampler_mode="greedy_fast",
        answer_tokens=answer_tokens,
        execution_path=_LAGUNA_EXECUTION_PATH,
        native_compact_prefill=False,
        native_caware_decode=False,
        serial_decode_fallback=False,
        native_sampler_rows=False,
        event="completed" if finish is not None else "token",
        timing={
            "prefill_ms": float(prefill_seconds) * 1_000.0,
            "decode_ms": float(decode_seconds) * 1_000.0,
        },
        timing_scope="request",
        group_rows=1,
        timing_owner=True,
        usage={
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(generated_ids),
            "total_tokens": len(prompt_ids) + len(generated_ids),
        },
        diagnostics={
            "backend": "hip_gfx1151",
            "model": "laguna_gguf",
            "quant": _LAGUNA_QUANT,
        },
    )


def _with_total_timing(
    telemetry: GenerationTelemetry,
    started: float,
) -> GenerationTelemetry:
    timing = dict(telemetry.timing or {})
    timing["request_total_ms"] = (time.perf_counter() - started) * 1_000.0
    return GenerationTelemetry(
        decode_state=telemetry.decode_state,
        event=telemetry.event,
        timing=timing,
        timing_scope=telemetry.timing_scope,
        batch_id=telemetry.batch_id,
        group_rows=telemetry.group_rows,
        timing_owner=telemetry.timing_owner,
        usage=telemetry.usage,
        diagnostics=telemetry.diagnostics,
    )


def make_laguna_gguf_generator_gfx1151(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> LagunaGGUFGenerator:
    return LagunaGGUFGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
        backend="hip_gfx1151",
    )


register_text_generator(
    model="laguna_gguf",
    backend="hip_gfx1151",
    quant=_LAGUNA_QUANT,
    factory=make_laguna_gguf_generator_gfx1151,
)


__all__ = ["LagunaGGUFGenerator", "make_laguna_gguf_generator_gfx1151"]
