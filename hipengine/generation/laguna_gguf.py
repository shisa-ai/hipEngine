"""Public greedy generation for Poolside Laguna S 2.1 GGUF.

The initial route deliberately stays c=1 and token-serial.  It owns one shared
resident weight set plus one resettable eager KV/scratch session under the
existing generator lock, and fails closed for sampling or speculative features
that the Laguna runner has not correctness-gated yet.
"""

from __future__ import annotations

import codecs
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from hipengine.chat.poolside_v1 import (
    PoolsideV1ReasoningParser,
    PoolsideV1ToolParser,
    render_poolside_v1_chat,
)
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.dispatch import RequestState, SlotMove, WorkItem
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken
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
_LAGUNA_RESIDENT_EXECUTION_PATH = "laguna_resident_scheduler_c1"
_LAGUNA_RESIDENT_SESSION_TTL_SECONDS = 900.0
_SAFE_RESIDENT_CACHE_ACTIONS = frozenset(("append_visible_only", "append_all"))


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
    server_plain_ar_max_active_requests: int = 2
    tokenizer: LagunaGGUFTokenizer = field(init=False)
    chat_reasoning_parser: PoolsideV1ReasoningParser = field(init=False)
    chat_tool_parser: PoolsideV1ToolParser = field(init=False)
    last_generation_outputs: tuple[GenerationOutput, ...] = field(
        default=(), init=False, repr=False
    )
    last_batch_generation: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _runtime: HipRuntime | None = field(default=None, init=False, repr=False)
    _weights: LagunaGGUFResidentWeights | None = field(default=None, init=False, repr=False)
    _session: LagunaGGUFResidentSession | None = field(default=None, init=False, repr=False)
    _session_recreate_pending: bool = field(default=False, init=False, repr=False)
    _retained_session_key: str | None = field(default=None, init=False, repr=False)
    _retained_token_ids: tuple[int, ...] = field(default=(), init=False, repr=False)
    _retained_at: float | None = field(default=None, init=False, repr=False)
    _resident_model_runner: "LagunaGGUFResidentModelRunner | None" = field(
        default=None,
        init=False,
        repr=False,
    )
    _load_seconds: float | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _speculative_provider: Any | None = field(default=None, init=False, repr=False)
    _repacked_cache_source_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )

    supports_speculative_mtp = False
    supports_stream_many = False
    supports_resident_session_kv = True
    supports_stream_logprobs = False
    chat_template_family = "poolside_v1"
    reasoning_parser_name = "poolside_v1"
    tool_parser_name = "poolside_v1"

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
        self.chat_reasoning_parser = PoolsideV1ReasoningParser(self.tokenizer)
        self.chat_tool_parser = PoolsideV1ToolParser()

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

    @property
    def supports_speculative(self) -> bool:
        return self._speculative_provider is not None

    def attach_speculative_provider(self, provider: Any) -> None:
        """Attach one registry-resolved provider before target materialization."""

        if provider is None:
            raise TypeError("speculative provider must not be None")
        for name in ("generate_detailed", "stream_detailed", "capabilities", "close"):
            if not callable(getattr(provider, name, None)):
                raise TypeError(f"speculative provider must implement {name}()")
        with self._lock:
            if self._closed:
                raise RuntimeError("Laguna generator is closed")
            if self._weights is not None:
                raise RuntimeError(
                    "speculative provider must attach before target materialization"
                )
            if self._speculative_provider is not None:
                raise RuntimeError("a speculative provider is already attached")
            self._speculative_provider = provider

    def bind_repacked_cache_source_sha256(self, sha256: str) -> None:
        """Require a source-bound sibling cache before later target loading."""

        digest = str(sha256).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("repacked-cache source SHA-256 must be 64 lowercase hex chars")
        with self._lock:
            if self._weights is not None:
                raise RuntimeError("target source identity must bind before materialization")
            existing = self._repacked_cache_source_sha256
            if existing is not None and existing != digest:
                raise RuntimeError("target source identity is already bound differently")
            self._repacked_cache_source_sha256 = digest

    def speculative_capabilities(self) -> dict[str, Any]:
        provider = self._speculative_provider
        return {} if provider is None else dict(provider.capabilities())

    def generate_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        provider = self._speculative_provider
        if provider is None:
            raise NotImplementedError("Laguna speculative provider is not configured")
        return list(provider.generate_detailed(request))

    def stream_speculative_detailed(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        provider = self._speculative_provider
        if provider is None:
            raise NotImplementedError("Laguna speculative provider is not configured")
        for chunk in provider.stream_detailed(request):
            yield GenerationStreamChunk.from_value(chunk)

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

    def render_chat_prompt(
        self,
        messages: Sequence[Any],
        *,
        tools: Sequence[Any] | None = None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return render_poolside_v1_chat(
            messages,
            tools=tools,
            enable_thinking=bool(enable_thinking),
            add_generation_prompt=bool(add_generation_prompt),
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
            if self._resident_model_runner is None:
                self._ensure_session_locked()
        return requested

    def create_resident_model_runner(
        self,
        *,
        capacity: int | None,
        config: Any | None = None,
    ) -> "LagunaGGUFResidentModelRunner":
        del config
        resolved_capacity = (
            int(self.server_plain_ar_max_active_requests)
            if capacity is None
            else int(capacity)
        )
        with self._lock:
            current = self._resident_model_runner
            if current is not None:
                if current.capacity != resolved_capacity:
                    raise RuntimeError("Laguna resident runner capacity cannot change while live")
                return current
            runner = LagunaGGUFResidentModelRunner(self, capacity=resolved_capacity)
            self._resident_model_runner = runner
            return runner

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: Any | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        resident_runner = self._resident_model_runner
        if resident_runner is not None:
            return resident_runner.prepare_request_scratch(
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                sampling_params=sampling_params,
                max_batch_size=max_batch_size,
                release_after_probe=release_after_probe,
            )
        del sampling_params, release_after_probe
        requested_batch = int(max_batch_size)
        if requested_batch <= 0 or requested_batch > int(self.server_plain_ar_max_active_requests):
            raise NotImplementedError(
                "Laguna resident serving supports at most "
                f"{self.server_plain_ar_max_active_requests} active c=1 rows"
            )
        required = int(max_prompt_tokens) + max(0, int(max_new_tokens) - 1)
        if required <= 0 or required > self.context_length:
            raise ValueError(
                f"Laguna request requires {required} positions; public limit is {self.context_length}"
            )
        with self._lock:
            self._prepare_locked()
            sessions: list[LagunaGGUFResidentSession] = []
            try:
                for _ in range(requested_batch):
                    sessions.append(self._open_session_locked())
                resident_nbytes_per_row = int(sessions[0].resident_nbytes)
            finally:
                for session in reversed(sessions):
                    session.close()
        return {
            "schema": 1,
            "backend": self.backend,
            "execution_path": _LAGUNA_EXECUTION_PATH,
            "max_batch_size": requested_batch,
            "max_sequence_length": required,
            "resident_session_nbytes": resident_nbytes_per_row * requested_batch,
            "released_after_probe": True,
        }

    def generate(self, request: GenerationRequest) -> list[str]:
        return [output.text for output in self.generate_detailed(request)]

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        prompt_ids, prompt_timing = self._prepare_request(request)
        if request.max_tokens == 0:
            output = self._empty_output(
                prompt_ids,
                request,
                prompt_timing=prompt_timing,
            )
            self._record_outputs((output,), prompt_ids, request)
            return [output]

        started = time.perf_counter()
        generated_ids: list[int] = []
        finish: FinishDetails | None = None
        final_telemetry: GenerationTelemetry | None = None
        for step in self._token_steps(
            request,
            prompt_ids,
            prompt_timing=prompt_timing,
        ):
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
        prompt_ids, prompt_timing = self._prepare_request(request)
        if request.max_tokens == 0:
            output = self._empty_output(
                prompt_ids,
                request,
                prompt_timing=prompt_timing,
            )
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
        stop_prefixes = _proper_stop_prefixes(request.stop_token_sequences)
        terminal_output: GenerationOutput | None = None
        for step in self._token_steps(
            request,
            prompt_ids,
            prompt_timing=prompt_timing,
        ):
            pending.append(step.token_id)
            finish = step.finish_details
            if finish is None:
                safe_count = _safe_pending_emit_count(pending, stop_prefixes)
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
            provider, self._speculative_provider = self._speculative_provider, None
            session, self._session = self._session, None
            weights, self._weights = self._weights, None
            self._session_recreate_pending = False
            self._clear_retained_session_locked()
            error: BaseException | None = None
            if provider is not None:
                try:
                    provider.close()
                except BaseException as exc:  # pragma: no cover - defensive cleanup
                    error = exc
            if session is not None:
                try:
                    session.close()
                except BaseException as exc:  # pragma: no cover - defensive cleanup
                    if error is None:
                        error = exc
            if weights is not None:
                try:
                    weights.free(runtime=self._runtime)
                except BaseException as exc:  # pragma: no cover - defensive cleanup
                    if error is None:
                        error = exc
            if error is not None:
                raise error

    def _prepare_request(
        self,
        request: GenerationRequest,
    ) -> tuple[tuple[int, ...], dict[str, float]]:
        _validate_public_request(request)
        raise_if_generation_deadline_expired(request)
        prompt = request.prompts[0]
        if isinstance(prompt, str):
            tokenize_started = time.perf_counter()
            prompt_ids = self.tokenize(prompt)
            tokenize_ms = (time.perf_counter() - tokenize_started) * 1_000.0
            prompt_timing = {
                "tokenize_ms": tokenize_ms,
                "prompt_encode_ms": tokenize_ms,
                "render_ms": 0.0,
                "admission_prepare_ms": 0.0,
            }
        else:
            prompt_ids = tuple(int(token) for token in prompt)
            tokenize_ms = max(0.0, float(getattr(prompt, "tokenize_ms", 0.0)))
            prompt_timing = {
                "tokenize_ms": tokenize_ms,
                "prompt_encode_ms": tokenize_ms,
                "render_ms": max(0.0, float(getattr(prompt, "render_ms", 0.0))),
                "admission_prepare_ms": max(
                    0.0,
                    float(getattr(prompt, "admission_prepare_ms", 0.0)),
                ),
            }
        if not prompt_ids:
            raise ValueError("Laguna prompt tokenization produced no token IDs")
        required = len(prompt_ids) + max(0, int(request.max_tokens) - 1)
        if required > self.context_length:
            raise ValueError(
                f"Laguna request requires {required} positions; public limit is {self.context_length}"
            )
        return prompt_ids, prompt_timing

    def _token_steps(
        self,
        request: GenerationRequest,
        prompt_ids: tuple[int, ...],
        *,
        prompt_timing: Mapping[str, float],
    ) -> Iterator[_LagunaTokenStep]:
        with self._lock:
            self._prepare_locked()
            raise_if_generation_deadline_expired(request)
            (
                session,
                prefill_ids,
                prefix_reused_tokens,
                session_prepare_seconds,
                session_prepare_mode,
            ) = self._acquire_session_locked(request, prompt_ids)
            generated: list[int] = []
            prefill_started = time.perf_counter()
            prefill_seconds = 0.0
            decode_seconds = 0.0
            try:
                result = session.prefill(prefill_ids)
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
                        prompt_timing=prompt_timing,
                        session_prepare_seconds=session_prepare_seconds,
                        session_prepare_mode=session_prepare_mode,
                        prefix_reused_tokens=prefix_reused_tokens,
                    )
                    if finish is not None:
                        self._retain_session_state_locked(
                            session,
                            request,
                            prompt_ids,
                            generated,
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
            except BaseException:
                self._retire_session_locked(session, suppress_errors=True)
                raise

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
            repacked_cache_source_sha256=self._repacked_cache_source_sha256,
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

    def _ensure_session_locked(self) -> LagunaGGUFResidentSession:
        session = self._session
        if session is not None:
            return session
        try:
            session = self._open_session_locked()
            assert self._runtime is not None
            self._runtime.device_synchronize()
        except BaseException:
            if session is not None:
                try:
                    session.close()
                except BaseException:  # pragma: no cover - preserve allocation failure
                    pass
            raise
        self._session = session
        self._session_recreate_pending = False
        return session

    def _acquire_session_locked(
        self,
        request: GenerationRequest,
        prompt_ids: tuple[int, ...],
    ) -> tuple[LagunaGGUFResidentSession, tuple[int, ...], int, float, str]:
        started = time.perf_counter()
        session = self._session
        if session is None:
            mode = "recreate_after_error" if self._session_recreate_pending else "create"
            try:
                session = self._ensure_session_locked()
            except BaseException:
                self._session_recreate_pending = True
                raise
            self._session_recreate_pending = False
            self._clear_retained_session_locked()
        else:
            reused_tokens = self._reusable_prefix_tokens_locked(
                session,
                request,
                prompt_ids,
            )
            if reused_tokens:
                assert self._runtime is not None
                self._runtime.device_synchronize()
                return (
                    session,
                    prompt_ids[reused_tokens:],
                    reused_tokens,
                    time.perf_counter() - started,
                    "reuse",
                )
            mode = "reset"
            try:
                session.reset_state()
                assert self._runtime is not None
                self._runtime.device_synchronize()
                self._clear_retained_session_locked()
            except BaseException:
                self._retire_session_locked(session, suppress_errors=True)
                raise
        return session, prompt_ids, 0, time.perf_counter() - started, mode

    def _reusable_prefix_tokens_locked(
        self,
        session: LagunaGGUFResidentSession,
        request: GenerationRequest,
        prompt_ids: tuple[int, ...],
    ) -> int:
        key = request.resident_session_key
        retained = self._retained_token_ids
        retained_at = self._retained_at
        if (
            key is None
            or key != self._retained_session_key
            or not retained
            or retained_at is None
            or time.monotonic() - retained_at > _LAGUNA_RESIDENT_SESSION_TTL_SECONDS
            or len(prompt_ids) <= len(retained)
            or prompt_ids[: len(retained)] != retained
            or int(getattr(session, "position", -1)) != len(retained) - 1
        ):
            return 0
        return len(retained)

    def _retain_session_state_locked(
        self,
        session: LagunaGGUFResidentSession,
        request: GenerationRequest,
        prompt_ids: tuple[int, ...],
        generated_ids: Sequence[int],
    ) -> None:
        key = request.resident_session_key
        if (
            key is None
            or request.resident_session_cache_action not in _SAFE_RESIDENT_CACHE_ACTIONS
            or not generated_ids
        ):
            self._clear_retained_session_locked()
            return
        retained = (*prompt_ids, *(int(token) for token in generated_ids[:-1]))
        if int(getattr(session, "position", -1)) != len(retained) - 1:
            self._clear_retained_session_locked()
            return
        self._retained_session_key = str(key)
        self._retained_token_ids = tuple(retained)
        self._retained_at = time.monotonic()

    def _clear_retained_session_locked(self) -> None:
        self._retained_session_key = None
        self._retained_token_ids = ()
        self._retained_at = None

    def _retire_session_locked(
        self,
        session: LagunaGGUFResidentSession,
        *,
        suppress_errors: bool,
    ) -> None:
        if self._session is session:
            self._session = None
        self._session_recreate_pending = True
        self._clear_retained_session_locked()
        try:
            session.close()
        except BaseException:
            if not suppress_errors:
                raise

    def _empty_output(
        self,
        prompt_ids: tuple[int, ...],
        request: GenerationRequest,
        *,
        prompt_timing: Mapping[str, float],
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
                prompt_timing=prompt_timing,
                session_prepare_seconds=0.0,
                session_prepare_mode="none",
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


@dataclass
class _LagunaResidentLoopRow:
    request_id: int
    row_index: int
    request: GenerationRequest
    prompt_ids: tuple[int, ...]
    submitted_at: float
    prompt_timing: dict[str, float] = field(default_factory=dict)
    session: LagunaGGUFResidentSession | None = None
    prefill_tokens_seen: int = 0
    next_result: Any | None = None
    generated_ids: list[int] = field(default_factory=list)
    pending_ids: list[int] = field(default_factory=list)
    decoder: _IncrementalLagunaDecoder | None = None
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    session_prepare_seconds: float = 0.0
    session_prepare_mode: str = "none"
    prefix_reused_tokens: int = 0


class LagunaGGUFResidentModelRunner:
    """Scheduler-owned exact c=1 Laguna rows backed by private resident sessions."""

    def __init__(self, generator: LagunaGGUFGenerator, *, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("Laguna resident runner capacity must be positive")
        self.generator = generator
        self.capacity = int(capacity)
        self._rows: dict[int, _LagunaResidentLoopRow] = {}
        self._outputs: dict[int, GenerationOutput] = {}
        self._available: list[LagunaGGUFResidentSession] = []
        self._all_sessions: list[LagunaGGUFResidentSession] = []
        self._retained_session: LagunaGGUFResidentSession | None = None
        self._retained_session_key: str | None = None
        self._retained_token_ids: tuple[int, ...] = ()
        self._retained_at: float | None = None
        self._prepared = False
        self._closed = False

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(self._rows)

    def observability_snapshot(self) -> dict[str, object]:
        return {
            "kind": "laguna_resident_model_runner",
            "capacity": self.capacity,
            "sessions": {
                "resident": len(self._all_sessions),
                "active": sum(row.session is not None for row in self._rows.values()),
                "available": len(self._available),
                "retained": int(self._retained_session is not None),
            },
            "active_request_ids": list(self.active_request_ids),
            "outputs_buffered": len(self._outputs),
            "closed": self._closed,
        }

    def prepare(self, *, max_sequence_length: int | None = None) -> None:
        if max_sequence_length is not None:
            requested = int(max_sequence_length)
            if requested <= 0 or requested > self.generator.context_length:
                raise ValueError(
                    "Laguna resident max_sequence_length exceeds the public context limit"
                )
        if self._prepared:
            return
        with self.generator._lock:
            if self._closed or self.generator._closed:
                raise RuntimeError("Laguna resident runner is closed")
            self.generator._prepare_locked()
            acquired: list[LagunaGGUFResidentSession] = []
            existing = self.generator._session
            self.generator._session = None
            if existing is not None:
                acquired.append(existing)
            try:
                while len(acquired) < self.capacity:
                    acquired.append(self.generator._open_session_locked())
            except BaseException:
                for session in reversed(acquired):
                    session.close()
                raise
            self._all_sessions = list(acquired)
            self._available = list(reversed(acquired))
            self.generator._clear_retained_session_locked()
            self._prepared = True

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
        requested_batch = int(max_batch_size)
        if requested_batch <= 0 or requested_batch > self.capacity:
            raise NotImplementedError(
                f"Laguna resident runner owns at most {self.capacity} active c=1 rows"
            )
        required = int(max_prompt_tokens) + max(0, int(max_new_tokens) - 1)
        if required <= 0 or required > self.generator.context_length:
            raise ValueError(
                f"Laguna request requires {required} positions; "
                f"public limit is {self.generator.context_length}"
            )
        self.prepare(max_sequence_length=required)
        resident_per_row = int(self._all_sessions[0].resident_nbytes)
        return {
            "schema": 1,
            "backend": self.generator.backend,
            "execution_path": _LAGUNA_RESIDENT_EXECUTION_PATH,
            "max_batch_size": requested_batch,
            "max_sequence_length": required,
            "resident_session_nbytes": resident_per_row * requested_batch,
            "released_after_probe": False,
        }

    def prompt_tokens(self, prompt: Any) -> tuple[int, ...]:
        if isinstance(prompt, str):
            tokens = self.generator.tokenize(prompt)
        else:
            tokens = tuple(int(token) for token in prompt)
        if not tokens:
            raise ValueError("Laguna prompt tokenization produced no token IDs")
        return tokens

    def record_prompt_tokenize_ms(
        self,
        request_ids: Sequence[int],
        tokenize_ms: Sequence[float],
    ) -> None:
        ids = tuple(int(request_id) for request_id in request_ids)
        values = tuple(max(0.0, float(value)) for value in tokenize_ms)
        if len(ids) != len(values):
            raise ValueError("request_ids and tokenize_ms must have the same length")
        for request_id, value in zip(ids, values, strict=True):
            row = self._row(request_id)
            row.prompt_timing["tokenize_ms"] = value
            row.prompt_timing.setdefault("prompt_encode_ms", value)

    def scheduler_max_new_tokens(self, request: GenerationRequest) -> int:
        return max(1, int(request.max_tokens))

    def register_batch(
        self,
        request_ids: Sequence[int],
        request: GenerationRequest,
        *,
        prompt_rows: Sequence[Sequence[int]],
    ) -> None:
        self.prepare()
        _validate_public_request(request, require_single_prompt=False)
        if len(request.prompts) > 1 and request.resident_session_key is not None:
            raise ValueError("Laguna stateful session keys require one prompt per submission")
        ids = tuple(int(request_id) for request_id in request_ids)
        prompts = tuple(tuple(int(token) for token in row) for row in prompt_rows)
        if len(ids) != len(request.prompts) or len(prompts) != len(ids):
            raise ValueError("request_ids, prompts, and prompt_rows must have the same length")
        now = time.perf_counter()
        for row_index, (request_id, prompt_ids) in enumerate(zip(ids, prompts, strict=True)):
            if request_id in self._rows or request_id in self._outputs:
                raise ValueError(f"request_id {request_id} is already registered")
            required = len(prompt_ids) + max(0, int(request.max_tokens) - 1)
            if required > self.generator.context_length:
                raise ValueError(
                    f"Laguna request requires {required} positions; "
                    f"public limit is {self.generator.context_length}"
                )
            prompt = request.prompts[row_index]
            tokenize_ms = max(0.0, float(getattr(prompt, "tokenize_ms", 0.0)))
            self._rows[request_id] = _LagunaResidentLoopRow(
                request_id=request_id,
                row_index=row_index,
                request=request,
                prompt_ids=prompt_ids,
                submitted_at=now,
                prompt_timing={
                    "tokenize_ms": tokenize_ms,
                    "prompt_encode_ms": tokenize_ms,
                    "render_ms": max(0.0, float(getattr(prompt, "render_ms", 0.0))),
                    "admission_prepare_ms": max(
                        0.0,
                        float(getattr(prompt, "admission_prepare_ms", 0.0)),
                    ),
                },
                decoder=_IncrementalLagunaDecoder(self.generator.tokenizer),
            )

    def reserve_admission(self, request: RequestState) -> bool:
        row = self._row(request.request_id)
        if int(row.request.max_tokens) == 0:
            return True
        if row.session is not None:
            return True
        started = time.perf_counter()
        reused = self._take_reusable_session(row)
        if reused is not None:
            row.session = reused
            row.session_prepare_seconds = time.perf_counter() - started
            row.session_prepare_mode = "reuse"
            return True
        if not self._available and self._retained_session is not None:
            self._release_retained_session()
        if not self._available and len(self._all_sessions) < self.capacity:
            with self.generator._lock:
                session = self.generator._open_session_locked()
            self._all_sessions.append(session)
            self._available.append(session)
        if not self._available:
            return False
        session = self._available.pop()
        try:
            session.reset_state()
            runtime = self.generator._runtime
            if runtime is not None:
                runtime.device_synchronize()
        except BaseException:
            try:
                session.close()
            finally:
                self._all_sessions.remove(session)
            raise
        row.session = session
        row.session_prepare_seconds = time.perf_counter() - started
        row.session_prepare_mode = "reset"
        return True

    def rollback_admission(self, request: RequestState) -> None:
        row = self._row(request.request_id)
        self._release_session(row)

    def prefill_batch(self, work: WorkItem, *, commit: bool) -> None:
        if not commit:
            raise ValueError("Laguna resident prefill requires commit=True")
        for request_id, token_row in zip(work.request_ids, work.token_rows, strict=True):
            row = self._row(request_id)
            raise_if_generation_deadline_expired(row.request)
            chunk = tuple(int(token) for token in token_row)
            start = int(row.prefill_tokens_seen)
            if chunk != row.prompt_ids[start : start + len(chunk)]:
                raise RuntimeError(
                    f"Laguna prefill chunk drift for request_id {request_id}"
                )
            row.prefill_tokens_seen += len(chunk)
            if int(row.request.max_tokens) == 0:
                continue
            session = row.session
            if session is None:
                raise RuntimeError("Laguna admitted row has no resident session")
            reused_in_chunk = max(
                0,
                min(len(chunk), int(row.prefix_reused_tokens) - start),
            )
            model_chunk = chunk[reused_in_chunk:]
            final_chunk = row.prefill_tokens_seen == len(row.prompt_ids)
            if not model_chunk:
                if final_chunk:
                    raise RuntimeError(
                        "Laguna resident KV reuse requires an unmatched prompt suffix"
                    )
                continue
            operation_started = time.perf_counter()
            try:
                row.next_result = session.prefill(model_chunk)
                row.prefill_seconds += time.perf_counter() - operation_started
                raise_if_generation_deadline_expired(row.request)
            except BaseException:
                self._retire_row_session(row)
                raise

    def decode_batch(self, work: WorkItem, *, commit: bool) -> tuple[GeneratedToken, ...]:
        if not commit:
            raise ValueError("Laguna resident decode requires commit=True")
        generated: list[GeneratedToken] = []
        for request_id in work.request_ids:
            row = self._row(request_id)
            raise_if_generation_deadline_expired(row.request)
            if int(row.request.max_tokens) == 0:
                generated.append(
                    GeneratedToken(
                        int(request_id),
                        0,
                        finished=True,
                        stream_chunk=GenerationStreamChunk(
                            text="",
                            finish_details=FinishDetails(
                                reason="length",
                                length_limit=0,
                                sampler_mode="greedy_fast",
                            ),
                            generated_token_ids=(),
                        ),
                    )
                )
                continue
            result = row.next_result
            if result is None:
                raise RuntimeError("Laguna resident row is not prefilled")
            token_id = int(result.next_token_id)
            row.generated_ids.append(token_id)
            row.pending_ids.append(token_id)
            finish = _laguna_finish_details(
                row.generated_ids,
                tokenizer=self.generator.tokenizer,
                request=row.request,
            )
            emitted = self._stream_text(row, finish=finish)
            telemetry = _laguna_telemetry(
                row.prompt_ids,
                row.generated_ids,
                finish=finish,
                prefill_seconds=row.prefill_seconds,
                decode_seconds=row.decode_seconds,
                prompt_timing=row.prompt_timing,
                session_prepare_seconds=row.session_prepare_seconds,
                session_prepare_mode=row.session_prepare_mode,
                prefix_reused_tokens=row.prefix_reused_tokens,
                execution_path=_LAGUNA_RESIDENT_EXECUTION_PATH,
            )
            generated.append(
                GeneratedToken(
                    int(request_id),
                    token_id,
                    finished=finish is not None,
                    stream_chunk=GenerationStreamChunk(
                        text=emitted,
                        finish_details=finish,
                        telemetry=telemetry,
                        generated_token_ids=(
                            tuple(row.generated_ids) if finish is not None else None
                        ),
                    ),
                )
            )
            if finish is None:
                operation_started = time.perf_counter()
                try:
                    row.next_result = (
                        row.session.forward_token(token_id) if row.session else None
                    )
                    row.decode_seconds += time.perf_counter() - operation_started
                    raise_if_generation_deadline_expired(row.request)
                except BaseException:
                    self._retire_row_session(row)
                    raise
        return tuple(generated)

    def compact_batch(self, moves: Sequence[SlotMove]) -> None:
        for move in moves:
            self._row(move.request_id)

    def reclaim(self, completed: CompletedRequest) -> None:
        row = self._rows.pop(int(completed.request_id), None)
        if row is None:
            return
        if int(row.request.max_tokens) == 0:
            output = self.generator._empty_output(
                row.prompt_ids,
                row.request,
                prompt_timing=row.prompt_timing,
            )
        else:
            generated_ids = tuple(row.generated_ids)
            finish = (
                completed.finish_details
                if completed.finish_reason in {"cancel", "disconnect", "timeout"}
                else _laguna_finish_details(
                    generated_ids,
                    tokenizer=self.generator.tokenizer,
                    request=row.request,
                )
            )
            if finish is None:
                finish = completed.finish_details
            visible_ids = _visible_generated_ids(
                generated_ids,
                finish,
                tokenizer=self.generator.tokenizer,
            )
            telemetry = _laguna_telemetry(
                row.prompt_ids,
                generated_ids,
                finish=finish,
                prefill_seconds=row.prefill_seconds,
                decode_seconds=row.decode_seconds,
                prompt_timing=row.prompt_timing,
                session_prepare_seconds=row.session_prepare_seconds,
                session_prepare_mode=row.session_prepare_mode,
                prefix_reused_tokens=row.prefix_reused_tokens,
                execution_path=_LAGUNA_RESIDENT_EXECUTION_PATH,
            )
            output = GenerationOutput(
                text=self.generator.tokenizer.decode(visible_ids),
                generated_token_ids=generated_ids,
                finish_details=finish,
                telemetry=_with_total_timing(telemetry, row.submitted_at),
            )
        self._outputs[row.request_id] = output
        if (
            completed.finished
            and completed.finish_reason not in {"cancel", "disconnect", "timeout"}
            and self._retain_session(row)
        ):
            return
        self._release_session(row)

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
                self._release_session(row)
            self._outputs.pop(rid, None)

    def finalize_batch(
        self,
        request: GenerationRequest,
        request_ids: Sequence[int],
        outputs: Sequence[GenerationOutput],
    ) -> None:
        del request_ids
        output_tuple = tuple(outputs)
        self.generator.last_generation_outputs = output_tuple
        self.generator.last_batch_generation = {
            "path": _LAGUNA_RESIDENT_EXECUTION_PATH,
            "backend": self.generator.backend,
            "quant": _LAGUNA_QUANT,
            "batch_size": len(output_tuple),
            "decode_steps": max(
                (len(output.generated_token_ids or ()) for output in output_tuple),
                default=0,
            ),
            "max_tokens": int(request.max_tokens),
            "resident_weights": self.generator._weights is not None,
            "model_load_seconds": self.generator._load_seconds,
            "native_caware_decode": False,
            "serial_decode_fallback": False,
            "throughput_claim_eligible": False,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for row in tuple(self._rows.values()):
            self._release_session(row)
        self._rows.clear()
        self._outputs.clear()
        error: BaseException | None = None
        for session in reversed(self._all_sessions):
            try:
                session.close()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                if error is None:
                    error = exc
        self._all_sessions.clear()
        self._available.clear()
        self._clear_retained_metadata()
        with self.generator._lock:
            self.generator._resident_model_runner = None
            self.generator._session = None
        try:
            self.generator.close()
        except BaseException as exc:  # pragma: no cover - defensive cleanup
            if error is None:
                error = exc
        if error is not None:
            raise error

    def _row(self, request_id: int) -> _LagunaResidentLoopRow:
        try:
            return self._rows[int(request_id)]
        except KeyError as exc:
            raise KeyError(f"unknown Laguna request_id {request_id}") from exc

    def _release_session(self, row: _LagunaResidentLoopRow) -> None:
        session, row.session = row.session, None
        if session is not None and session in self._all_sessions and session not in self._available:
            self._available.append(session)

    def _retire_row_session(self, row: _LagunaResidentLoopRow) -> None:
        session, row.session = row.session, None
        if session is None:
            return
        if self._retained_session is session:
            self._clear_retained_metadata()
        if session in self._available:
            self._available.remove(session)
        if session in self._all_sessions:
            self._all_sessions.remove(session)
        try:
            session.close()
        except BaseException:
            pass

    def _take_reusable_session(
        self,
        row: _LagunaResidentLoopRow,
    ) -> LagunaGGUFResidentSession | None:
        session = self._retained_session
        retained_at = self._retained_at
        retained = self._retained_token_ids
        key = row.request.resident_session_key
        if (
            session is None
            or key is None
            or str(key) != self._retained_session_key
            or not retained
            or retained_at is None
            or time.monotonic() - retained_at > _LAGUNA_RESIDENT_SESSION_TTL_SECONDS
            or len(row.prompt_ids) <= len(retained)
            or row.prompt_ids[: len(retained)] != retained
            or int(getattr(session, "position", -1)) != len(retained) - 1
        ):
            return None
        row.prefix_reused_tokens = len(retained)
        self._clear_retained_metadata()
        return session

    def _retain_session(self, row: _LagunaResidentLoopRow) -> bool:
        session = row.session
        key = row.request.resident_session_key
        if (
            session is None
            or key is None
            or row.request.resident_session_cache_action not in _SAFE_RESIDENT_CACHE_ACTIONS
            or not row.generated_ids
        ):
            return False
        retained = (*row.prompt_ids, *(int(token) for token in row.generated_ids[:-1]))
        if int(getattr(session, "position", -1)) != len(retained) - 1:
            return False
        if self._retained_session is not None:
            self._release_retained_session()
        row.session = None
        self._retained_session = session
        self._retained_session_key = str(key)
        self._retained_token_ids = tuple(retained)
        self._retained_at = time.monotonic()
        return True

    def _release_retained_session(self) -> None:
        session = self._retained_session
        self._clear_retained_metadata()
        if session is not None and session in self._all_sessions and session not in self._available:
            self._available.append(session)

    def _clear_retained_metadata(self) -> None:
        self._retained_session = None
        self._retained_session_key = None
        self._retained_token_ids = ()
        self._retained_at = None

    def _stream_text(
        self,
        row: _LagunaResidentLoopRow,
        *,
        finish: FinishDetails | None,
    ) -> str:
        decoder = row.decoder
        if decoder is None:
            return ""
        if finish is not None:
            suppressed = _suppressed_suffix_length(finish)
            count = max(0, len(row.pending_ids) - suppressed)
            emit_ids = tuple(row.pending_ids[:count])
            row.pending_ids.clear()
            return decoder.feed(emit_ids, final=True)
        count = _safe_pending_emit_count(
            row.pending_ids,
            _proper_stop_prefixes(row.request.stop_token_sequences),
        )
        if count <= 0:
            return ""
        emit_ids = tuple(row.pending_ids[:count])
        del row.pending_ids[:count]
        return decoder.feed(emit_ids)


def _proper_stop_prefixes(
    stop_sequences: Sequence[Sequence[int]],
) -> frozenset[tuple[int, ...]]:
    return frozenset(
        tuple(int(token) for token in sequence[:width])
        for sequence in stop_sequences
        for width in range(1, len(sequence))
    )


def _safe_pending_emit_count(
    pending: Sequence[int],
    stop_prefixes: frozenset[tuple[int, ...]],
) -> int:
    retained = max(
        (
            len(prefix)
            for prefix in stop_prefixes
            if len(prefix) <= len(pending)
            and tuple(pending[-len(prefix) :]) == prefix
        ),
        default=0,
    )
    return len(pending) - retained


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


def _validate_public_request(
    request: GenerationRequest,
    *,
    require_single_prompt: bool = True,
) -> None:
    if require_single_prompt and len(request.prompts) != 1:
        raise ValueError("Laguna public generation requires exactly one prompt")
    if not request.prompts:
        raise ValueError("Laguna public generation requires at least one prompt")
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
    prompt_timing: Mapping[str, float],
    session_prepare_seconds: float,
    session_prepare_mode: str,
    prefix_reused_tokens: int = 0,
    execution_path: str = _LAGUNA_EXECUTION_PATH,
) -> GenerationTelemetry:
    suppressed = 0 if finish is None else _suppressed_suffix_length(finish)
    answer_tokens = max(0, len(generated_ids) - suppressed)
    return GenerationTelemetry.from_decode_counts(
        prompt_tokens=len(prompt_ids),
        generated_tokens=len(generated_ids),
        phase="done" if finish is not None else "answer",
        sampler_mode="greedy_fast",
        answer_tokens=answer_tokens,
        execution_path=str(execution_path),
        native_compact_prefill=False,
        native_caware_decode=False,
        serial_decode_fallback=False,
        native_sampler_rows=False,
        event="completed" if finish is not None else "token",
        timing={
            "tokenize_ms": max(0.0, float(prompt_timing.get("tokenize_ms", 0.0))),
            "prompt_encode_ms": max(
                0.0,
                float(prompt_timing.get("prompt_encode_ms", 0.0)),
            ),
            "render_ms": max(0.0, float(prompt_timing.get("render_ms", 0.0))),
            "admission_prepare_ms": max(
                0.0,
                float(prompt_timing.get("admission_prepare_ms", 0.0)),
            ),
            "session_prepare_ms": max(0.0, float(session_prepare_seconds)) * 1_000.0,
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
            "session_prepare_mode": str(session_prepare_mode),
            "resident_kv_reused": bool(prefix_reused_tokens),
            "prefix_reused_tokens": max(0, int(prefix_reused_tokens)),
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
