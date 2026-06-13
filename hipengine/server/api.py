"""OpenAI-compatible FastAPI surface for hipEngine.

The server layer is optional and intentionally thin: it adapts OpenAI-style JSON
requests to the torch-free ``hipengine.LLM.generate()`` library API.  HTTP
requests are routed through the generation batcher; the remaining async lock is
limited to short model/session preparation mutations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

try:  # Pydantic v2; FastAPI's current default.
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 compatibility
    ConfigDict = None  # type: ignore[assignment]

from starlette.concurrency import run_in_threadpool

from hipengine import LLM, SamplingParams
from hipengine.generation import FinishDetails, GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS, GenerationOutput, TokenLogprob, derive_row_seed
from hipengine.kvcache import resolve_prefix_cache_mode


_LOGGER = logging.getLogger("uvicorn.error")
_GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET = frozenset(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the optional OpenAI-compatible server."""

    model: str
    backend: str = "auto"
    quant: str = "w4_paro"
    served_model_name: str | None = None
    api_key: str | None = None
    eager_load: bool = True
    eager_load_prompt: str = "one two three four"
    eager_load_max_tokens: int = 1
    max_context_tokens: int | None = None
    chat_default_max_tokens: int | None = 4096
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    generation_batch_window_ms: float = 0.0
    metrics: str = "off"
    prefix_cache: str = "off"
    debug: bool = False
    created: int = field(default_factory=lambda: int(time.time()))

    @property
    def model_id(self) -> str:
        """Public model identifier exposed through the OpenAI API."""

        if self.served_model_name:
            return self.served_model_name
        path = Path(self.model)
        if path.exists() and path.name:
            return path.name
        return self.model


class OpenAIHTTPError(Exception):
    """Exception converted to an OpenAI-style error JSON body."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        super().__init__(message)


if ConfigDict is not None:

    class _OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

else:  # pragma: no cover - Pydantic v1 compatibility

    class _OpenAIBaseModel(BaseModel):
        class Config:
            extra = "allow"


class CompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int | None = Field(default=16, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=0, ge=0)
    min_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=1.0, gt=0.0)
    presence_penalty: float | None = Field(default=0.0)
    frequency_penalty: float | None = Field(default=0.0)
    logit_bias: dict[str, float] | None = None
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    echo: bool = False
    logprobs: int | None = Field(default=None, ge=0, le=20)
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None


class ChatMessage(_OpenAIBaseModel):
    role: str
    content: str | list[Any] | None = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=0, ge=0)
    min_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=1.0, gt=0.0)
    presence_penalty: float | None = Field(default=0.0)
    frequency_penalty: float | None = Field(default=0.0)
    logit_bias: dict[str, float] | None = None
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    thinking: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None


@dataclass(frozen=True)
class _GeneratedBatch:
    outputs: list[str]
    usage: dict[str, int]
    details: list[GenerationOutput]


@dataclass
class _ServerMetrics:
    """Additive server counters rendered by the opt-in Prometheus endpoint."""

    request_total: int = 0
    request_completed_total: int = 0
    request_failed_total: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0

    def record_success(self, usage: Mapping[str, int]) -> None:
        self.request_total += 1
        self.request_completed_total += 1
        self.prompt_tokens_total += int(usage.get("prompt_tokens", 0))
        self.completion_tokens_total += int(usage.get("completion_tokens", 0))

    def record_failure(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1


_STREAM_DONE = object()


def _next_stream_item(iterator: Iterator[Any]) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_DONE


class _DebugPayloadMiddleware:
    """ASGI middleware that logs full HTTP payloads when explicitly enabled."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        query = scope.get("query_string", b"")
        if isinstance(query, bytes) and query:
            target = f"{path}?{query.decode('utf-8', 'replace')}"
        else:
            target = path
        request_chunks: list[bytes] = []
        response_chunks: list[bytes] = []
        request_logged = False
        response_status: int | None = None

        def log_request_once() -> None:
            nonlocal request_logged
            if request_logged:
                return
            request_logged = True
            _LOGGER.info(
                "DEBUG_PAYLOAD REQUEST %s %s body=%s",
                method,
                target,
                _debug_payload_text(request_chunks),
            )

        async def receive_wrapper() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if body:
                    request_chunks.append(bytes(body))
                if not message.get("more_body", False):
                    log_request_once()
            return message

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_status
            if message.get("type") == "http.response.start":
                response_status = int(message.get("status", 0))
            elif message.get("type") == "http.response.body":
                body = message.get("body", b"")
                if body:
                    response_chunks.append(bytes(body))
                await send(message)
                if not message.get("more_body", False):
                    log_request_once()
                    _LOGGER.info(
                        "DEBUG_PAYLOAD RESPONSE %s %s status=%s body=%s",
                        method,
                        target,
                        "unknown" if response_status is None else str(response_status),
                        _debug_payload_text(response_chunks),
                    )
                return
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception:
            log_request_once()
            _LOGGER.exception("DEBUG_PAYLOAD RESPONSE %s %s raised before completion", method, target)
            raise


def _debug_payload_text(chunks: Sequence[bytes]) -> str:
    data = b"".join(chunks)
    if not data:
        return "<empty>"
    text = data.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _request_target(request: Request) -> str:
    query = request.url.query
    return request.url.path if not query else f"{request.url.path}?{query}"


def _log_request_failure(
    request: Request,
    *,
    status_code: int,
    code: str | None,
    param: str | None,
    message: str,
) -> None:
    logger = _LOGGER.error if int(status_code) >= 500 else _LOGGER.warning
    logger(
        "REQUEST_FAILED: %s %s status=%d code=%s param=%s message=%s",
        request.method,
        _request_target(request),
        int(status_code),
        code,
        param,
        message,
    )


def _log_stream_failure(
    endpoint: str,
    *,
    status_code: int,
    code: str | None,
    param: str | None,
    message: str,
) -> None:
    logger = _LOGGER.error if int(status_code) >= 500 else _LOGGER.warning
    logger(
        "REQUEST_FAILED: %s status=%d code=%s param=%s message=%s",
        endpoint,
        int(status_code),
        code,
        param,
        message,
    )


@dataclass
class _QueuedGeneration:
    prompts: tuple[str, ...]
    sampling: SamplingParams
    future: asyncio.Future[list[Any]] | None = None
    stream_queue: asyncio.Queue[object] | None = None
    cancelled: bool = False


class _GenerationBatcher:
    """Coalesce compatible HTTP generations into prompt-list calls."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any],
        batch_window_seconds: float,
    ) -> None:
        self._engine_factory = engine_factory
        self._batch_window_seconds = max(0.0, float(batch_window_seconds))
        self._queue: deque[_QueuedGeneration] = deque()
        self._worker: asyncio.Task[None] | None = None

    async def submit(self, prompts: Sequence[str], sampling: SamplingParams) -> list[Any]:
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[Any]] = loop.create_future()
        self._queue.append(
            _QueuedGeneration(
                prompts=prompt_tuple,
                sampling=sampling,
                future=future,
            )
        )
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        return await future

    async def stream(self, prompts: Sequence[str], sampling: SamplingParams) -> AsyncIterator[str]:
        """Yield generated text through a per-request queue owned by the batcher."""

        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        item = _QueuedGeneration(
            prompts=prompt_tuple,
            sampling=sampling,
            stream_queue=queue,
        )
        self._queue.append(item)
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        try:
            while True:
                event = await queue.get()
                if event is _STREAM_DONE:
                    break
                if isinstance(event, BaseException):
                    raise event
                yield str(event)
        finally:
            item.cancelled = True

    async def _run(self) -> None:
        try:
            if self._batch_window_seconds > 0.0:
                await asyncio.sleep(self._batch_window_seconds)
            while self._queue:
                first = self._queue.popleft()
                if _queued_generation_cancelled(first):
                    continue
                key = _sampling_key(first.sampling)
                group = [first]
                deferred: deque[_QueuedGeneration] = deque()
                while self._queue:
                    item = self._queue.popleft()
                    if _queued_generation_cancelled(item):
                        continue
                    if _sampling_key(item.sampling) == key:
                        group.append(item)
                    else:
                        deferred.append(item)
                self._queue.extendleft(reversed(deferred))
                await self._run_group(group)
                if self._queue and self._batch_window_seconds > 0.0:
                    await asyncio.sleep(self._batch_window_seconds)
        finally:
            self._worker = None
            if self._queue:
                self._worker = asyncio.create_task(self._run())

    async def _run_group(self, group: Sequence[_QueuedGeneration]) -> None:
        if not group:
            return
        if len(group) == 1 and group[0].stream_queue is not None and len(group[0].prompts) == 1:
            await self._stream_single(group[0])
            return
        prompts: list[str] = []
        slices: list[tuple[_QueuedGeneration, int, int]] = []
        for item in group:
            start = len(prompts)
            prompts.extend(item.prompts)
            slices.append((item, start, len(prompts)))
        try:
            outputs = await self._generate_prompts(tuple(prompts), group[0].sampling)
        except Exception as exc:
            for item in group:
                _finish_queued_generation(item, exception=exc)
            return
        for item, start, end in slices:
            _finish_queued_generation(item, outputs=outputs[start:end])

    async def _generate_prompts(self, prompts: tuple[str, ...], sampling: SamplingParams) -> list[Any]:
        raw_outputs = await run_in_threadpool(
            self._engine_factory().generate,
            prompts,
            sampling,
        )
        outputs = list(raw_outputs)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        return outputs

    async def _stream_single(self, item: _QueuedGeneration) -> None:
        assert item.stream_queue is not None
        try:
            async for chunk in _stream_engine_text(self._engine_factory(), item.prompts[0], item.sampling):
                if _queued_generation_cancelled(item):
                    break
                item.stream_queue.put_nowait(str(chunk))
        except Exception as exc:
            _finish_queued_generation(item, exception=exc)
            return
        _finish_queued_generation(item, outputs=())


def _queued_generation_cancelled(item: _QueuedGeneration) -> bool:
    return item.cancelled or (item.future is not None and item.future.cancelled())


def _finish_queued_generation(
    item: _QueuedGeneration,
    *,
    outputs: Sequence[str] | None = None,
    exception: Exception | None = None,
) -> None:
    if item.future is not None and not item.future.done():
        if exception is not None:
            item.future.set_exception(exception)
        else:
            item.future.set_result(list(outputs or ()))
    if item.stream_queue is None:
        return
    if exception is not None:
        item.stream_queue.put_nowait(exception)
    else:
        for output in outputs or ():
            item.stream_queue.put_nowait(str(output))
    item.stream_queue.put_nowait(_STREAM_DONE)


async def _stream_engine_text(engine: Any, prompt: str, sampling: SamplingParams) -> AsyncIterator[str]:
    streamer = getattr(engine, "stream", None)
    if not callable(streamer):
        for output in await run_in_threadpool(engine.generate, (prompt,), sampling):
            yield str(output)
        return
    iterator = iter(streamer(prompt, sampling))
    done = False
    try:
        while True:
            item = await run_in_threadpool(_next_stream_item, iterator)
            if item is _STREAM_DONE:
                done = True
                break
            yield str(item)
    finally:
        if not done:
            closer = getattr(iterator, "close", None)
            if callable(closer):
                await run_in_threadpool(closer)


@dataclass(frozen=True)
class _ReasoningSplit:
    content: str
    reasoning_content: str


@dataclass(frozen=True)
class _ParsedToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class _ParsedChatOutput:
    text: str
    tool_calls: tuple[_ParsedToolCall, ...]


@dataclass(frozen=True)
class _ThinkingControl:
    enabled: bool | None = None
    effort: str | None = None


def create_app(config: ServerConfig, *, llm: Any | None = None) -> FastAPI:
    """Create a FastAPI app for OpenAI-compatible local inference.

    ``llm`` is injectable for tests and must expose ``generate(prompts,
    sampling_params)``.  Startup eagerly warms the configured model by default;
    disabling ``ServerConfig.eager_load`` keeps construction lazy until the first
    generation request.
    """

    app = FastAPI(title="hipEngine OpenAI-compatible API", version="0.2.1")
    metrics_mode = _metrics_mode(config.metrics)
    prefix_cache_mode = resolve_prefix_cache_mode(config.prefix_cache)
    app.state.hipengine_config = config
    app.state.hipengine_llm = llm
    app.state.hipengine_effective_max_context_tokens = config.max_context_tokens
    app.state.hipengine_prefix_cache_mode = prefix_cache_mode
    app.state.hipengine_server_metrics = _ServerMetrics()
    if config.debug:
        app.add_middleware(_DebugPayloadMiddleware)
    session_lock = asyncio.Lock()

    def get_llm() -> Any:
        if app.state.hipengine_llm is None:
            app.state.hipengine_llm = LLM(config.model, backend=config.backend, quant=config.quant)
        return app.state.hipengine_llm

    generation_batcher = _GenerationBatcher(
        engine_factory=get_llm,
        batch_window_seconds=float(config.generation_batch_window_ms) / 1000.0,
    )
    app.state.hipengine_generation_batcher = generation_batcher

    def configured_max_context_tokens() -> int | None:
        if config.max_context_tokens is None:
            return None
        return max(1, int(config.max_context_tokens))

    def effective_max_context_tokens(engine: Any) -> int | None:
        configured = configured_max_context_tokens()
        if configured is not None:
            return configured
        cached = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        if cached is not None:
            return max(1, int(cached))
        prepared = _prepared_context_tokens(engine)
        if prepared is not None:
            app.state.hipengine_effective_max_context_tokens = prepared
            return prepared
        return None

    def preparation_sampling(request: CompletionRequest | ChatCompletionRequest | None = None) -> SamplingParams:
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            kv_storage=(request.kv_storage if request is not None and request.kv_storage else config.kv_storage),
            kv_scale_dtype=(
                request.kv_scale_dtype if request is not None and request.kv_scale_dtype else config.kv_scale_dtype
            ),
            kv_scale_granularity=(
                request.kv_scale_granularity
                if request is not None and request.kv_scale_granularity
                else config.kv_scale_granularity
            ),
        )

    async def ensure_resident_context(
        engine: Any,
        sampling: SamplingParams,
        *,
        phase: str,
    ) -> int | None:
        requested_context = configured_max_context_tokens()
        prepared = _prepared_context_tokens(engine)
        if prepared is not None and (requested_context is None or prepared >= requested_context):
            app.state.hipengine_effective_max_context_tokens = prepared
            return effective_max_context_tokens(engine)
        preparer = getattr(engine, "prepare", None)
        if not callable(preparer):
            return effective_max_context_tokens(engine)
        prepare_started = time.perf_counter()
        try:
            prepared_result = await run_in_threadpool(
                lambda: preparer(
                    max_sequence_length=requested_context,
                    sampling_params=sampling,
                )
            )
        except MemoryError as exc:
            _LOGGER.error(
                "hipEngine %s failed to allocate resident KV cache: %s. "
                "Try a lower --max-context-tokens or --kv-storage int8_per_token_head.",
                phase,
                exc,
            )
            raise
        except Exception as exc:
            _LOGGER.error(
                "hipEngine %s failed to prepare resident session/KV cache: %s. "
                "Try a lower --max-context-tokens or --kv-storage int8_per_token_head.",
                phase,
                exc,
            )
            raise
        if prepared_result is not None:
            app.state.hipengine_effective_max_context_tokens = max(1, int(prepared_result))
        else:
            prepared = _prepared_context_tokens(engine)
            if prepared is not None:
                app.state.hipengine_effective_max_context_tokens = prepared
        effective = effective_max_context_tokens(engine)
        _LOGGER.info(
            "LOAD_TIMING: phase=%s resident_prepare_s=%.3f max_context_tokens=%s",
            phase,
            time.perf_counter() - prepare_started,
            "unknown" if effective is None else str(effective),
        )
        return effective

    async def eager_load_model() -> None:
        startup_started = time.perf_counter()
        max_tokens = max(1, int(config.eager_load_max_tokens))
        if not config.eager_load:
            max_context = configured_max_context_tokens()
            _LOGGER.info(
                "Config: model=%s served_model=%s max_context_tokens=%s "
                "chat_default_max_tokens=%s kv_storage=%s kv_scale_dtype=%s "
                "kv_scale_granularity=%s eager_load=False",
                config.model,
                config.model_id,
                "auto" if max_context is None else str(max_context),
                _chat_default_max_tokens_label(config),
                config.kv_storage,
                config.kv_scale_dtype,
                config.kv_scale_granularity,
            )
            _LOGGER.info("LOAD_TIMING: eager_load=False startup_total_s=%.3f", time.perf_counter() - startup_started)
            _LOGGER.info("hipEngine is ready (lazy load).")
            return
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            kv_storage=config.kv_storage,
            kv_scale_dtype=config.kv_scale_dtype,
            kv_scale_granularity=config.kv_scale_granularity,
        )
        async with session_lock:
            engine_started = time.perf_counter()
            engine = get_llm()
            engine_create_s = time.perf_counter() - engine_started
            prepare_started = time.perf_counter()
            max_context = await ensure_resident_context(engine, sampling, phase="startup")
            resident_prepare_s = time.perf_counter() - prepare_started
            _LOGGER.info(
                "Config: model=%s served_model=%s max_context_tokens=%s "
                "chat_default_max_tokens=%s kv_storage=%s kv_scale_dtype=%s "
                "kv_scale_granularity=%s eager_load=True",
                config.model,
                config.model_id,
                "unknown" if max_context is None else str(max_context),
                _chat_default_max_tokens_label(config),
                config.kv_storage,
                config.kv_scale_dtype,
                config.kv_scale_granularity,
            )
            _log_kv_capacity_summary(engine)
            _validate_context_budget(max_context, engine, (config.eager_load_prompt,), sampling)
        _LOGGER.info(
            "WARMUP: prompt_tokens<=%s max_tokens=%d",
            "unknown" if max_context is None else str(max_context),
            max_tokens,
        )
        warmup_started = time.perf_counter()
        await run_in_threadpool(engine.generate, (config.eager_load_prompt,), sampling)
        warmup_s = time.perf_counter() - warmup_started
        _LOGGER.info(
            "LOAD_TIMING: model=%s engine_create_s=%.3f resident_prepare_s=%.3f warmup_s=%.3f startup_total_s=%.3f",
            config.model_id,
            engine_create_s,
            resident_prepare_s,
            warmup_s,
            time.perf_counter() - startup_started,
        )
        _LOGGER.info("hipEngine is ready.")

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("startup", eager_load_model)
    else:  # FastAPI-lite compatibility in minimal test/runtime environments.
        app.router.on_startup.append(eager_load_model)

    async def require_auth(request: Request) -> None:
        if not config.api_key:
            return
        expected = f"Bearer {config.api_key}"
        if request.headers.get("authorization") != expected:
            raise OpenAIHTTPError(
                401,
                "missing or invalid bearer token",
                error_type="authentication_error",
                code="invalid_api_key",
            )

    @app.exception_handler(OpenAIHTTPError)
    async def openai_error_handler(request: Request, exc: OpenAIHTTPError) -> JSONResponse:
        _log_request_failure(
            request,
            status_code=exc.status_code,
            code=exc.code,
            param=exc.param,
            message=exc.message,
        )
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "param": exc.param,
                    "code": exc.code,
                }
            },
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        message = _format_validation_error(exc)
        _log_request_failure(
            request,
            status_code=422,
            code="validation_error",
            param=None,
            message=message,
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                }
            },
        )

    def sampling_params(
        request: CompletionRequest | ChatCompletionRequest,
        prompts: Sequence[str],
        engine: Any,
    ) -> SamplingParams:
        stop_token_ids, stop_token_sequences = _stop_tokens_from_stop(request.stop, engine)
        return SamplingParams(
            max_tokens=_request_max_tokens(
                request,
                prompts,
                engine,
                effective_max_context_tokens(engine),
                chat_default_max_tokens=config.chat_default_max_tokens,
            ),
            logprobs=_request_logprobs_enabled(request),
            top_logprobs=_request_top_logprobs(request),
            temperature=float(request.temperature if request.temperature is not None else 0.0),
            top_p=float(request.top_p if request.top_p is not None else 1.0),
            top_k=int(request.top_k if request.top_k is not None else 0),
            min_p=float(request.min_p if request.min_p is not None else 0.0),
            repetition_penalty=float(request.repetition_penalty if request.repetition_penalty is not None else 1.0),
            presence_penalty=float(request.presence_penalty if request.presence_penalty is not None else 0.0),
            frequency_penalty=float(request.frequency_penalty if request.frequency_penalty is not None else 0.0),
            logit_bias=request.logit_bias or (),
            stop_token_ids=stop_token_ids,
            stop_token_sequences=stop_token_sequences,
            ignore_eos=bool(request.ignore_eos),
            kv_storage=request.kv_storage or config.kv_storage,
            kv_scale_dtype=request.kv_scale_dtype or config.kv_scale_dtype,
            kv_scale_granularity=request.kv_scale_granularity or config.kv_scale_granularity,
            seed=request.seed,
        )

    async def generate(
        prompts: Sequence[str],
        request: CompletionRequest | ChatCompletionRequest,
    ) -> _GeneratedBatch:
        try:
            _validate_generation_request(config, request)
            async with session_lock:
                engine = get_llm()
                await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                sampling = sampling_params(request, prompts, engine)
                if _request_n(request) > 1:
                    sampling = replace(
                        sampling,
                        row_seeds=_row_seeds_for_request(request.seed, len(prompts)),
                    )
                _validate_context_budget(effective_max_context_tokens(engine), engine, prompts, sampling)
            if _request_logprobs_enabled(request):
                raw_outputs = await _generate_detailed(engine, tuple(prompts), sampling)
            else:
                raw_outputs = await generation_batcher.submit(tuple(prompts), sampling)
        except OpenAIHTTPError:
            app.state.hipengine_server_metrics.record_failure()
            raise
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(400, str(exc), code="unsupported_parameter") from exc
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(400, str(exc), code="invalid_request") from exc
        except Exception as exc:  # pragma: no cover - exercised by real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(
                500,
                f"generation failed: {exc}",
                error_type="server_error",
                code="generation_failed",
            ) from exc

        details = [_coerce_generation_output(item) for item in raw_outputs]
        outputs = [item.text for item in details]
        if len(outputs) != len(prompts):
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(
                500,
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts",
                error_type="server_error",
                code="bad_generator_output",
            )
        if _request_logprobs_enabled(request):
            _validate_logprob_details(details, outputs)
        batch = _GeneratedBatch(outputs=outputs, usage=_usage(engine, prompts, outputs), details=details)
        app.state.hipengine_server_metrics.record_success(batch.usage)
        return batch

    async def stream_completion_one(
        prompt: str,
        request: CompletionRequest,
    ) -> AsyncIterator[str]:
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream_started_at = time.perf_counter()
        include_hipengine = _stream_include_hipengine(request)
        full_text: list[str] = []
        try:
            _validate_generation_request(config, request)
            async with session_lock:
                engine = get_llm()
                await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                sampling = sampling_params(request, (prompt,), engine)
                _validate_context_budget(effective_max_context_tokens(engine), engine, (prompt,), sampling)
            async for token in generation_batcher.stream((prompt,), sampling):
                text = str(token)
                if not text:
                    continue
                full_text.append(text)
                yield _completion_stream_delta(
                    response_id,
                    created,
                    config.model_id,
                    text,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                )
        except OpenAIHTTPError as exc:
            app.state.hipengine_server_metrics.record_failure()
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            yield _completion_stream_error(response_id, created, config.model_id, exc.message)
            yield "data: [DONE]\n\n"
            return
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=400,
                code="unsupported_parameter",
                param=None,
                message=message,
            )
            yield _completion_stream_error(response_id, created, config.model_id, message)
            yield "data: [DONE]\n\n"
            return
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=400,
                code="invalid_request",
                param=None,
                message=message,
            )
            yield _completion_stream_error(response_id, created, config.model_id, message)
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=500,
                code="generation_failed",
                param=None,
                message=message,
            )
            yield _completion_stream_error(response_id, created, config.model_id, message)
            yield "data: [DONE]\n\n"
            return

        raw_text = "".join(full_text)
        text, finish_reason = _apply_stop(raw_text, request.stop)
        usage = _usage(engine, (prompt,), [text])
        app.state.hipengine_server_metrics.record_success(usage)
        yield _completion_stream_done(
            response_id,
            created,
            config.model_id,
            finish_reason,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
        if _stream_include_usage(request):
            yield _completion_stream_usage(
                response_id,
                created,
                config.model_id,
                usage,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
        yield "data: [DONE]\n\n"

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": config.model_id}

    if metrics_mode == "prometheus":

        @app.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics() -> PlainTextResponse:
            return PlainTextResponse(
                _render_prometheus_metrics(
                    app.state.hipengine_server_metrics,
                    engine=getattr(app.state, "hipengine_llm", None),
                ),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    @app.get("/v1/models")
    async def list_models(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "created": config.created,
                    "owned_by": "hipengine",
                }
            ],
        }

    @app.post("/v1/completions", response_model=None)
    async def completions(
        request: CompletionRequest,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model)
        _validate_generation_request(config, request)
        prompts = _normalize_prompts(request.prompt)
        n = _request_n(request)
        expanded_prompts = _expand_prompts_for_n(prompts, n)
        if request.stream and len(expanded_prompts) == 1 and not request.echo and not _request_logprobs_enabled(request):
            return StreamingResponse(
                stream_completion_one(expanded_prompts[0], request),
                media_type="text/event-stream",
            )
        batch = await generate(expanded_prompts, request)
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choices = []
        final_texts: list[str] = []
        for index, (prompt, output, detail) in enumerate(zip(expanded_prompts, batch.outputs, batch.details, strict=True)):
            generated_text, finish_reason = _apply_stop(output, request.stop)
            server_stop = generated_text != output
            finish_reason = _finish_reason_for_output(detail, finish_reason, server_stop=server_stop)
            text = prompt + generated_text if request.echo else generated_text
            final_texts.append(text)
            choice = {
                "text": text,
                "index": index,
                "logprobs": (
                    _completion_logprobs(detail, generated_text, echo_text=prompt if request.echo else "")
                    if request.logprobs is not None
                    else None
                ),
                "finish_reason": finish_reason,
                "finish_details": _finish_details_payload(
                    detail,
                    finish_reason,
                    reason_override="stop" if server_stop else None,
                ),
            }
            if n > 1:
                choice["request_id"] = _choice_request_id(response_id, index // n, index % n)
            choices.append(choice)
        response = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": config.model_id,
            "choices": choices,
            "usage": batch.usage,
        }
        if request.stream:
            stream_started_at = time.perf_counter()
            return StreamingResponse(
                _completion_stream(
                    response_id,
                    created,
                    config.model_id,
                    final_texts,
                    choices,
                    usage=batch.usage if _stream_include_usage(request) else None,
                    include_hipengine=_stream_include_hipengine(request),
                    stream_started_at=stream_started_at,
                ),
                media_type="text/event-stream",
            )
        return response

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model)
        _validate_generation_request(config, request)
        thinking = _thinking_control_from_request(request)
        prompt = render_chat_prompt(
            request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=thinking,
        )
        if request.stream:
            streamer = stream_chat_completion_many if _request_n(request) > 1 or request.logprobs else stream_chat_completion
            return StreamingResponse(
                streamer(prompt, request),
                media_type="text/event-stream",
            )
        n = _request_n(request)
        prompts = tuple(prompt for _ in range(n))
        batch = await generate(prompts, request)
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choices = []
        for index, (output, detail) in enumerate(zip(batch.outputs, batch.details, strict=True)):
            text, finish_reason = _apply_stop(output, request.stop)
            server_stop = text != output
            parsed = _parse_chat_tool_calls(text)
            message, parsed_finish_reason = _chat_message_from_parsed(parsed)
            finish_reason = _finish_reason_for_output(
                detail,
                parsed_finish_reason if parsed.tool_calls else finish_reason,
                server_stop=server_stop,
                tool_calls=bool(parsed.tool_calls),
            )
            choice = {
                "index": index,
                "message": message,
                "finish_reason": finish_reason,
                "finish_details": _finish_details_payload(
                    detail,
                    finish_reason,
                    reason_override="tool_calls" if parsed.tool_calls else "stop" if server_stop else None,
                ),
            }
            if request.logprobs:
                choice["logprobs"] = _chat_logprobs(detail, text)
            if n > 1:
                choice["request_id"] = _choice_request_id(response_id, 0, index)
            choices.append(choice)
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": config.model_id,
            "choices": choices,
            "usage": batch.usage,
        }
        return response

    async def stream_chat_completion_many(
        prompt: str,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream_started_at = time.perf_counter()
        include_hipengine = _stream_include_hipengine(request)
        try:
            n = _request_n(request)
            batch = await generate(tuple(prompt for _ in range(n)), request)
            for index, output in enumerate(batch.outputs):
                text, finish_reason = _apply_stop(output, request.stop)
                server_stop = text != output
                parsed = _parse_chat_tool_calls(text)
                detail = batch.details[index]
                finish_reason = _finish_reason_for_output(
                    detail,
                    finish_reason,
                    server_stop=server_stop,
                    tool_calls=bool(parsed.tool_calls),
                )
                logprobs = _chat_logprobs(batch.details[index], text) if request.logprobs else None
                yield _chat_stream_role(
                    response_id,
                    created,
                    config.model_id,
                    index=index,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                )
                for event in _chat_stream_parsed(
                    response_id,
                    created,
                    config.model_id,
                    parsed,
                    finish_reason,
                    index=index,
                    logprobs=logprobs,
                    finish_details=_finish_details_payload(
                        detail,
                        finish_reason,
                        reason_override="tool_calls" if parsed.tool_calls else "stop" if server_stop else None,
                    ),
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                ):
                    yield event
            if _stream_include_usage(request):
                yield _chat_stream_usage(
                    response_id,
                    created,
                    config.model_id,
                    batch.usage,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                )
        except OpenAIHTTPError as exc:
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, exc.message)
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=500,
                code="generation_failed",
                param=None,
                message=message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, message)
        yield "data: [DONE]\n\n"

    async def stream_chat_completion(
        prompt: str,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        try:
            _validate_generation_request(config, request)
        except OpenAIHTTPError as exc:
            app.state.hipengine_server_metrics.record_failure()
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, exc.message)
            yield "data: [DONE]\n\n"
            return
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        stream_started_at = time.perf_counter()
        include_hipengine = _stream_include_hipengine(request)
        full_text: list[str] = []
        splitter = _ReasoningSplitter()
        buffer_tool_output = bool(request.tools)

        try:
            async with session_lock:
                engine = get_llm()
                await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                sampling = sampling_params(request, (prompt,), engine)
                _validate_context_budget(effective_max_context_tokens(engine), engine, (prompt,), sampling)
            yield _chat_stream_role(
                response_id,
                created,
                config.model_id,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
            async for token in generation_batcher.stream((prompt,), sampling):
                text = str(token)
                if not text:
                    continue
                full_text.append(text)
                if buffer_tool_output:
                    continue
                for field, chunk in splitter.feed(text):
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        field,
                        chunk,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                    )
            if not buffer_tool_output:
                for field, chunk in splitter.finish():
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        field,
                        chunk,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                    )
        except OpenAIHTTPError as exc:
            app.state.hipengine_server_metrics.record_failure()
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, exc.message)
            yield "data: [DONE]\n\n"
            return
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=400,
                code="unsupported_parameter",
                param=None,
                message=message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, message)
            yield "data: [DONE]\n\n"
            return
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=400,
                code="invalid_request",
                param=None,
                message=message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, message)
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=500,
                code="generation_failed",
                param=None,
                message=message,
            )
            yield _chat_stream_error(response_id, created, config.model_id, message)
            yield "data: [DONE]\n\n"
            return

        raw_text = "".join(full_text)
        text, finish_reason = _apply_stop(raw_text, request.stop)
        if text != raw_text:
            # Stop strings can split across yielded chunks; current streaming keeps
            # transport simple and reports the stop after generation completes.
            finish_reason = "stop"
        usage = _usage(engine, (prompt,), [text])
        app.state.hipengine_server_metrics.record_success(usage)
        if buffer_tool_output:
            parsed = _parse_chat_tool_calls(text)
            for event in _chat_stream_parsed(
                response_id,
                created,
                config.model_id,
                parsed,
                "tool_calls" if parsed.tool_calls else finish_reason,
                finish_details=_finish_details_payload(None, "tool_calls" if parsed.tool_calls else finish_reason),
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            ):
                yield event
        else:
            yield _chat_stream_done(
                response_id,
                created,
                config.model_id,
                finish_reason,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
        if _stream_include_usage(request):
            yield _chat_stream_usage(
                response_id,
                created,
                config.model_id,
                usage,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
        yield "data: [DONE]\n\n"

    return app


def _metrics_mode(raw: str | None) -> str:
    value = "off" if raw is None or raw == "" else str(raw).strip().lower()
    if value not in {"off", "prometheus"}:
        raise ValueError("metrics must be one of: off, prometheus")
    return value


def _render_prometheus_metrics(metrics: _ServerMetrics, *, engine: Any | None) -> str:
    pool = _pool_metric_values(engine)
    graph = _graph_bucket_metric_values(engine)
    values = {
        "hipengine_requests_total": metrics.request_total,
        "hipengine_request_completed_total": metrics.request_completed_total,
        "hipengine_request_failed_total": metrics.request_failed_total,
        "hipengine_prompt_tokens_total": metrics.prompt_tokens_total,
        "hipengine_completion_tokens_total": metrics.completion_tokens_total,
        "hipengine_kv_pool_current_bytes": pool["current_bytes"],
        "hipengine_kv_pool_high_water_observed_bytes": pool["high_water_observed_bytes"],
        "hipengine_kv_pool_grow_events_total": pool["grow_events"],
        "hipengine_kv_pool_grow_failures_total": pool["grow_failures"],
        "hipengine_kv_pool_shrink_events_total": pool["shrink_events"],
        "hipengine_kv_pool_free_pages": pool["free_pages"],
        "hipengine_kv_pool_refcounted_pages": pool["refcounted_pages"],
        "hipengine_graph_bucket_entries": graph["entries"],
        "hipengine_graph_bucket_hits_total": graph["hits"],
        "hipengine_graph_bucket_misses_total": graph["misses"],
        "hipengine_graph_bucket_replay_hit_rate": graph["replay_hit_rate"],
    }
    help_text = {
        "hipengine_requests_total": "Total generation requests observed by the server.",
        "hipengine_request_completed_total": "Generation requests that completed successfully.",
        "hipengine_request_failed_total": "Generation requests that failed after reaching generation validation.",
        "hipengine_prompt_tokens_total": "Prompt tokens counted for successful requests.",
        "hipengine_completion_tokens_total": "Completion tokens counted for successful requests.",
        "hipengine_kv_pool_current_bytes": "Current dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_high_water_observed_bytes": "Peak observed dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_grow_events_total": "Dynamic KV pool grow events, or 0 when unavailable.",
        "hipengine_kv_pool_grow_failures_total": "Dynamic KV pool grow failures, or 0 when unavailable.",
        "hipengine_kv_pool_shrink_events_total": "Dynamic KV pool shrink events, or 0 when unavailable.",
        "hipengine_kv_pool_free_pages": "Current dynamic KV pool free pages, or 0 when unavailable.",
        "hipengine_kv_pool_refcounted_pages": "Current dynamic KV pool refcounted pages, or 0 when unavailable.",
        "hipengine_graph_bucket_entries": "Current graph bucket cache entries, or 0 when unavailable.",
        "hipengine_graph_bucket_hits_total": "Graph bucket cache hits, or 0 when unavailable.",
        "hipengine_graph_bucket_misses_total": "Graph bucket cache misses, or 0 when unavailable.",
        "hipengine_graph_bucket_replay_hit_rate": "Graph bucket replay hit rate, or 0 when unavailable.",
    }
    counter_names = {
        "hipengine_requests_total",
        "hipengine_request_completed_total",
        "hipengine_request_failed_total",
        "hipengine_prompt_tokens_total",
        "hipengine_completion_tokens_total",
        "hipengine_kv_pool_grow_events_total",
        "hipengine_kv_pool_grow_failures_total",
        "hipengine_kv_pool_shrink_events_total",
        "hipengine_graph_bucket_hits_total",
        "hipengine_graph_bucket_misses_total",
    }
    lines: list[str] = []
    for name, value in values.items():
        lines.append(f"# HELP {name} {help_text[name]}")
        lines.append(f"# TYPE {name} {'counter' if name in counter_names else 'gauge'}")
        lines.append(f"{name} {_format_metric_value(value)}")
    _append_labeled_counter_metrics(
        lines,
        "hipengine_graph_bucket_miss_reason_total",
        "Graph bucket cache misses by reason, or empty when unavailable.",
        "reason",
        graph["miss_reasons"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_graph_bucket_kernel_time_bucket_total",
        "Graph bucket kernel-time observations by duration bucket, or empty when unavailable.",
        "bucket",
        graph["kernel_time_histogram_ns"],
    )
    return "\n".join(lines) + "\n"


def _pool_metric_values(engine: Any | None) -> dict[str, float]:
    values = {
        "current_bytes": 0.0,
        "high_water_observed_bytes": 0.0,
        "grow_events": 0.0,
        "grow_failures": 0.0,
        "shrink_events": 0.0,
        "free_pages": 0.0,
        "refcounted_pages": 0.0,
    }
    stats = _first_stats_object(engine, ("kv_pool", "kv_cache_pool", "pool", "kv_pool_stats"))
    if stats is None:
        return values
    data = _stats_to_mapping(stats)
    for key in values:
        values[key] = _non_negative_metric_value(data.get(key))
    return values


def _graph_bucket_metric_values(engine: Any | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "entries": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "replay_hit_rate": 0.0,
        "miss_reasons": {},
        "kernel_time_histogram_ns": {},
    }
    stats = _first_stats_object(engine, ("graph_buckets", "graph_bucket_cache", "graph_bucket_stats"))
    if stats is None:
        return values
    data = _stats_to_mapping(stats)
    for key in ("entries", "hits", "misses"):
        values[key] = _non_negative_metric_value(data.get(key))
    lookups = values["hits"] + values["misses"]
    values["replay_hit_rate"] = values["hits"] / lookups if lookups > 0.0 else 0.0
    values["miss_reasons"] = _non_negative_metric_mapping(data.get("miss_reasons"))
    kernel_time_histogram = _non_negative_metric_mapping(data.get("kernel_time_histogram_ns"))
    known_kernel_time_histogram = {bucket: 0.0 for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS}
    for bucket, value in kernel_time_histogram.items():
        if bucket in _GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET:
            known_kernel_time_histogram[bucket] = value
    values["kernel_time_histogram_ns"] = known_kernel_time_histogram
    return values


def _first_stats_object(engine: Any | None, names: Sequence[str]) -> Any | None:
    if engine is None:
        return None
    session = _resident_session_for_engine(engine)
    for owner in (engine, session):
        if owner is None:
            continue
        for name in names:
            candidate = getattr(owner, name, None)
            if candidate is None:
                continue
            stats = getattr(candidate, "stats", candidate)
            return stats() if callable(stats) else stats
    return None


def _stats_to_mapping(stats: Any) -> Mapping[str, Any]:
    if isinstance(stats, Mapping):
        return stats
    to_json = getattr(stats, "to_json_dict", None)
    if callable(to_json):
        data = to_json()
        if isinstance(data, Mapping):
            return data
    keys = (
        "current_bytes",
        "high_water_observed_bytes",
        "grow_events",
        "grow_failures",
        "shrink_events",
        "free_pages",
        "refcounted_pages",
        "entries",
        "hits",
        "misses",
        "miss_reasons",
        "kernel_time_histogram_ns",
    )
    return {key: getattr(stats, key) for key in keys if hasattr(stats, key)}


def _non_negative_metric_value(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return default
    return numeric


def _non_negative_metric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        numeric = _non_negative_metric_value(raw, default=-1.0)
        if numeric < 0:
            continue
        metrics[str(key)] = numeric
    return metrics


def _append_labeled_counter_metrics(lines: list[str], name: str, help_text: str, label: str, values: Mapping[str, float]) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    for key, value in sorted(values.items()):
        lines.append(f'{name}{{{label}="{_escape_prometheus_label_value(key)}"}} {_format_metric_value(value)}')


def _escape_prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_metric_value(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return repr(numeric)


def _thinking_control_from_request(request: ChatCompletionRequest) -> _ThinkingControl:
    enabled: bool | None = None
    effort: str | None = None

    if isinstance(request.chat_template_kwargs, Mapping):
        enabled = _maybe_bool(request.chat_template_kwargs.get("enable_thinking"), enabled)
        effort = _maybe_effort(request.chat_template_kwargs.get("reasoning_effort"), effort)
        effort = _maybe_effort(request.chat_template_kwargs.get("thinking_budget"), effort)

    enabled = _maybe_bool(request.enable_thinking, enabled)
    effort = _maybe_effort(request.reasoning_effort, effort)
    if _effort_disables_thinking(effort):
        enabled = False

    if isinstance(request.thinking, Mapping):
        thinking_type = str(request.thinking.get("type", "")).strip().lower()
        if thinking_type in {"disabled", "disable", "off", "none"}:
            enabled = False
        elif thinking_type in {"enabled", "enable", "on"}:
            enabled = True
        enabled = _maybe_bool(request.thinking.get("enabled"), enabled)
        effort = _maybe_effort(request.thinking.get("effort"), effort)
        effort = _maybe_effort(request.thinking.get("budget_tokens"), effort)
    elif isinstance(request.thinking, str):
        effort = _maybe_effort(request.thinking, effort)
        if _effort_disables_thinking(effort):
            enabled = False

    if isinstance(request.reasoning, Mapping):
        enabled = _maybe_bool(request.reasoning.get("enabled"), enabled)
        reasoning_type = str(request.reasoning.get("type", "")).strip().lower()
        if reasoning_type in {"disabled", "disable", "off", "none"}:
            enabled = False
        elif reasoning_type in {"enabled", "enable", "on"}:
            enabled = True
        effort = _maybe_effort(request.reasoning.get("effort"), effort)

    if _effort_disables_thinking(effort):
        enabled = False
    return _ThinkingControl(enabled=enabled, effort=effort)


def _maybe_bool(value: Any, current: bool | None) -> bool | None:
    if value is None:
        return current
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled", "enable"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", "disable", "none"}:
            return False
    return current


def _maybe_effort(value: Any, current: str | None) -> str | None:
    if value is None:
        return current
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "low" if float(value) <= 1024 else "medium" if float(value) <= 4096 else "high"
    text = str(value).strip().lower()
    return text or current


def _effort_disables_thinking(effort: str | None) -> bool:
    return effort in {"0", "false", "none", "off", "disabled", "disable", "nothink", "no_think"}


def _render_thinking_prompt(thinking: _ThinkingControl | None) -> str:
    if thinking is None:
        return ""
    if thinking.enabled is False:
        return "Do not include hidden reasoning. Answer directly after the pre-closed <think></think> block."
    effort = thinking.effort
    if not effort or _effort_disables_thinking(effort):
        return ""
    if effort in {"minimal", "low"}:
        limit = "very brief"
    elif effort == "medium":
        limit = "concise"
    elif effort in {"high", "xhigh", "max"}:
        limit = "focused but complete"
    else:
        limit = "concise"
    return (
        f"If you use <think> reasoning, keep it {limit}; when ready, close </think> "
        "before emitting the final answer or any <tool_call> block."
    )


def _assistant_prefix_for_thinking(thinking: _ThinkingControl | None) -> str:
    prefix = "<|im_start|>assistant\n"
    if thinking is not None and thinking.enabled is False:
        return prefix + "<think>\n\n</think>\n\n"
    return prefix


def _render_tools_prompt(
    tools: Sequence[Mapping[str, Any]] | None,
    tool_choice: str | Mapping[str, Any] | None,
) -> str:
    if not tools or _tool_choice_name(tool_choice) == "none":
        if _tool_choice_name(tool_choice) == "none":
            return "Do not call tools for this response."
        return ""
    tool_lines = [json.dumps(dict(tool), ensure_ascii=False, separators=(",", ":")) for tool in tools]
    directive = _tool_choice_directive(tool_choice)
    return "\n".join(
        [
            "You may call one or more functions to assist with the user request.",
            "Available functions are provided in JSON schema form inside <tools></tools> tags:",
            "<tools>",
            *tool_lines,
            "</tools>",
            directive,
            "For each function call, respond with a JSON object inside <tool_call></tool_call> tags and no extra prose:",
            '<tool_call>{"name":"function_name","arguments":{"arg":"value"}}</tool_call>',
        ]
    )


def _tool_choice_name(tool_choice: str | Mapping[str, Any] | None) -> str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower()
    choice_type = str(tool_choice.get("type", "")).strip().lower()
    if choice_type == "function":
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            return None if name is None else str(name)
    return choice_type or None


def _tool_choice_directive(tool_choice: str | Mapping[str, Any] | None) -> str:
    name = _tool_choice_name(tool_choice)
    if name == "required":
        return "You must call at least one function."
    if name and name not in {"auto", "none"}:
        return f"You must call the function named {name!r}."
    return "Call a function only when it is useful; otherwise answer normally."


def _render_tool_call_for_prompt(tool_call: Mapping[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name", ""))
        raw_arguments = function.get("arguments", {})
    else:
        name = str(tool_call.get("name", ""))
        raw_arguments = tool_call.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except Exception:
            arguments = raw_arguments
    else:
        arguments = raw_arguments
    payload = {"name": name, "arguments": arguments if arguments is not None else {}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def render_chat_prompt(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = None,
    thinking: _ThinkingControl | None = None,
) -> str:
    """Render OpenAI chat messages to a Qwen-style text prompt.

    This is intentionally tokenizer-independent so the API can stay a thin
    adapter around ``LLM.generate()``.  Model-specific chat-template rendering
    can replace this helper once tokenizers are exposed by the runtime.
    """

    if not messages:
        raise OpenAIHTTPError(400, "messages must contain at least one item", param="messages")
    rendered: list[str] = []
    control_prompts = [item for item in (_render_thinking_prompt(thinking), _render_tools_prompt(tools, tool_choice)) if item]
    if control_prompts:
        rendered.append(f"<|im_start|>system\n{'\n\n'.join(control_prompts)}<|im_end|>")
    for index, message in enumerate(messages):
        if isinstance(message, Mapping):
            role_value = message.get("role", "")
            content_value = message.get("content", "")
            tool_calls = message.get("tool_calls")
        else:
            role_value = message.role
            content_value = message.content
            tool_calls = message.tool_calls
        role = str(role_value).strip()
        if not role:
            raise OpenAIHTTPError(
                400,
                "message role must be non-empty",
                param=f"messages[{index}].role",
            )
        content = _message_content_text(content_value, index)
        if role == "developer":
            role = "system"
        if role == "tool":
            rendered.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
            continue
        if role == "assistant" and tool_calls:
            tool_call_text = "\n".join(_render_tool_call_for_prompt(item) for item in tool_calls)
            content = "\n".join(part for part in (content, tool_call_text) if part)
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    rendered.append(_assistant_prefix_for_thinking(thinking))
    return "\n".join(rendered)


def _validate_model(config: ServerConfig, requested: str | None) -> None:
    if requested is not None and requested != config.model_id:
        raise OpenAIHTTPError(
            404,
            f"model {requested!r} is not served by this hipEngine instance",
            code="model_not_found",
            param="model",
        )


def _log_kv_capacity_summary(engine: Any) -> None:
    session = _resident_session_for_engine(engine)
    if session is None:
        return
    estimate = getattr(session, "kv_capacity_estimate", None)
    if estimate is not None:
        model_max = int(getattr(estimate, "model_max_context_tokens", 0) or 0)
        _LOGGER.info(
            "KVCache: storage=%s scale=%s max_context_tokens=%d model_max_context_tokens=%s "
            "allocatable_context_tokens=%d requested_kv=%s metadata=%s total=%s "
            "bytes_per_token=%d usable=%s reserve=%s",
            getattr(estimate, "kv_storage_dtype", "unknown"),
            getattr(estimate, "kv_scale_dtype", None) or "none",
            int(getattr(estimate, "requested_context_tokens", 0) or 0),
            "unknown" if model_max <= 0 else str(model_max),
            int(getattr(estimate, "allocatable_context_tokens", 0) or 0),
            _format_bytes(int(getattr(estimate, "requested_kv_bytes", 0) or 0)),
            _format_bytes(int(getattr(estimate, "requested_context_overhead_bytes", 0) or 0)),
            _format_bytes(int(getattr(estimate, "requested_total_bytes", 0) or 0)),
            int(getattr(estimate, "bytes_per_token", 0) or 0),
            _format_bytes(int(getattr(estimate, "usable_bytes", 0) or 0)),
            _format_bytes(int(getattr(estimate, "reserve_bytes", 0) or 0)),
        )
        if model_max > 0 and not bool(getattr(estimate, "fits_model_max", True)):
            _LOGGER.warning(
                "KVCache: selected policy can fit allocatable_context_tokens=%d, "
                "below model_max_context_tokens=%d",
                int(getattr(estimate, "allocatable_context_tokens", 0) or 0),
                model_max,
            )
    int8_estimate = getattr(session, "kv_capacity_int8_estimate", None)
    if int8_estimate is None:
        return
    int8_model_max = int(getattr(int8_estimate, "model_max_context_tokens", 0) or 0)
    if int8_model_max > 0 and not bool(getattr(int8_estimate, "fits_model_max", True)):
        _LOGGER.warning(
            "KVCache: int8_per_token_head can fit allocatable_context_tokens=%d, "
            "below model_max_context_tokens=%d",
            int(getattr(int8_estimate, "allocatable_context_tokens", 0) or 0),
            int8_model_max,
        )


def _resident_session_for_engine(engine: Any) -> Any | None:
    if hasattr(engine, "kv_capacity_estimate"):
        return engine
    generator = getattr(engine, "_text_generator", None)
    if generator is not None:
        session = getattr(generator, "_session", None)
        if session is not None:
            return session
    session = getattr(engine, "_session", None)
    if session is not None:
        return session
    return None


def _prepared_context_tokens(engine: Any) -> int | None:
    session = _resident_session_for_engine(engine)
    if session is None:
        return None
    value = getattr(session, "max_sequence_length", None)
    if value is None:
        return None
    return max(1, int(value))


def _format_bytes(value: int) -> str:
    return f"{int(value) / 1024**3:.2f} GiB"


def _request_max_tokens(
    request: CompletionRequest | ChatCompletionRequest,
    prompts: Sequence[str],
    engine: Any,
    max_context_tokens: int | None,
    *,
    chat_default_max_tokens: int | None = 4096,
) -> int:
    if request.max_tokens is not None:
        return max(0, int(request.max_tokens))
    if isinstance(request, ChatCompletionRequest):
        remaining = _remaining_context_tokens(prompts, engine, max_context_tokens)
        if chat_default_max_tokens is None:
            return 8192 if remaining is None else max(0, int(remaining))
        default_tokens = max(0, int(chat_default_max_tokens))
        if remaining is None:
            return default_tokens
        return max(0, min(default_tokens, int(remaining)))
    return 16


def _remaining_context_tokens(
    prompts: Sequence[str],
    engine: Any,
    max_context_tokens: int | None,
) -> int | None:
    if max_context_tokens is None:
        return None
    return min(
        int(max_context_tokens) - _count_tokens_for_admission(engine, str(prompt)) - 1
        for prompt in prompts
    )


def _chat_default_max_tokens_label(config: ServerConfig) -> str:
    return "auto" if config.chat_default_max_tokens is None else str(int(config.chat_default_max_tokens))


def _request_logprobs_enabled(request: CompletionRequest | ChatCompletionRequest) -> bool:
    if isinstance(request, CompletionRequest):
        return request.logprobs is not None
    return bool(request.logprobs)


def _request_top_logprobs(request: CompletionRequest | ChatCompletionRequest) -> int:
    if isinstance(request, CompletionRequest):
        return 0 if request.logprobs is None else int(request.logprobs)
    return 0 if request.top_logprobs is None else int(request.top_logprobs)


async def _generate_detailed(
    engine: Any,
    prompts: tuple[str, ...],
    sampling: SamplingParams,
) -> list[Any]:
    detailed = getattr(engine, "generate_detailed", None)
    if callable(detailed):
        return list(await run_in_threadpool(detailed, prompts, sampling))
    return [GenerationOutput(text=str(item)) for item in await run_in_threadpool(engine.generate, prompts, sampling)]


def _coerce_generation_output(value: Any) -> GenerationOutput:
    if isinstance(value, GenerationOutput):
        return value
    token_logprobs = getattr(value, "token_logprobs", None)
    finish_details = getattr(value, "finish_details", None)
    if token_logprobs is not None or finish_details is not None:
        return GenerationOutput(
            text=str(getattr(value, "text", value)),
            token_logprobs=tuple(token_logprobs or ()),
            finish_details=finish_details,
        )
    return GenerationOutput(text=str(value))


def _validate_logprob_details(details: Sequence[GenerationOutput], outputs: Sequence[str]) -> None:
    for output, text in zip(details, outputs, strict=True):
        if text and not output.token_logprobs:
            raise OpenAIHTTPError(
                500,
                "generator did not return token logprobs for a logprobs request",
                error_type="server_error",
                code="missing_logprobs",
                param="logprobs",
            )


def _validate_generation_request(config: ServerConfig, request: CompletionRequest | ChatCompletionRequest) -> None:
    _request_n(request)
    extra_keys = _request_extra_keys(request)
    if extra_keys:
        param = sorted(extra_keys)[0]
        raise OpenAIHTTPError(
            400,
            f"unsupported request parameter {param!r}",
            code="unsupported_parameter",
            param=param,
        )
    if isinstance(request, ChatCompletionRequest) and request.top_logprobs is not None and not request.logprobs:
        raise OpenAIHTTPError(
            400,
            "top_logprobs requires logprobs=true",
            code="invalid_request",
            param="top_logprobs",
        )
    try:
        from hipengine.kvcache import resolve_kv_policy

        server_policy = resolve_kv_policy(
            config.kv_storage,
            scale_dtype=config.kv_scale_dtype,
            scale_granularity=config.kv_scale_granularity,
        )
        request_policy = resolve_kv_policy(
            request.kv_storage or config.kv_storage,
            scale_dtype=request.kv_scale_dtype or config.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity or config.kv_scale_granularity,
        )
    except ValueError as exc:
        raise OpenAIHTTPError(400, str(exc), code="invalid_kv_policy", param="kv_storage") from exc
    if (
        request.kv_storage is not None
        or request.kv_scale_dtype is not None
        or request.kv_scale_granularity is not None
    ) and (
        request_policy.storage_dtype != server_policy.storage_dtype
        or request_policy.scale_dtype != server_policy.scale_dtype
        or request_policy.scale_granularity != server_policy.scale_granularity
    ):
        raise OpenAIHTTPError(
            400,
            "this server preallocates a fixed KV cache policy; restart with matching "
            "--kv-storage/--kv-scale-dtype to use different KV settings",
            code="unsupported_kv_policy",
            param="kv_storage",
        )


def _validate_context_budget(
    max_context_tokens: int | None,
    engine: Any,
    prompts: Sequence[str],
    sampling: SamplingParams,
) -> None:
    if max_context_tokens is None:
        return
    max_context = max(1, int(max_context_tokens))
    max_tokens = max(0, int(sampling.max_tokens))
    for index, prompt in enumerate(prompts):
        prompt_tokens = _count_tokens_for_admission(engine, str(prompt))
        required = prompt_tokens + max_tokens + 1
        if required > max_context:
            raise OpenAIHTTPError(
                400,
                f"request requires {required} context tokens (prompt {prompt_tokens} + "
                f"max_tokens {max_tokens} + 1), exceeding this server's "
                f"preallocated max_context_tokens={max_context}",
                code="context_length_exceeded",
                param=f"prompts[{index}].max_tokens" if len(prompts) > 1 else "max_tokens",
            )


def _count_tokens_for_admission(engine: Any, text: str) -> int:
    counter = getattr(engine, "count_tokens", None)
    if not callable(counter):
        return 0
    try:
        return max(0, int(counter(text)))
    except NotImplementedError:
        return 0


def _normalize_prompts(prompt: str | list[str]) -> tuple[str, ...]:
    if isinstance(prompt, str):
        return (prompt,)
    if not prompt:
        raise OpenAIHTTPError(400, "prompt must not be empty", param="prompt")
    return tuple(str(item) for item in prompt)


def _request_n(request: CompletionRequest | ChatCompletionRequest) -> int:
    n = 1 if request.n is None else int(request.n)
    if n < 1:
        raise OpenAIHTTPError(400, "n must be at least 1", code="invalid_request", param="n")
    return n


def _request_extra_keys(request: CompletionRequest | ChatCompletionRequest) -> set[str]:
    extra = getattr(request, "model_extra", None)
    if isinstance(extra, Mapping):
        return {str(key) for key in extra}
    extra = getattr(request, "__pydantic_extra__", None)
    if isinstance(extra, Mapping):
        return {str(key) for key in extra}
    fields = getattr(request, "__fields__", None)
    if isinstance(fields, Mapping):
        return {str(key) for key in vars(request) if str(key) not in fields}
    return set()


def _expand_prompts_for_n(prompts: Sequence[str], n: int) -> tuple[str, ...]:
    return tuple(prompt for prompt in prompts for _ in range(int(n)))


def _choice_request_id(response_id: str, prompt_index: int, choice_index: int) -> str:
    return f"{response_id}:prompt-{int(prompt_index)}:choice-{int(choice_index)}"


def _row_seeds_for_request(seed: int | None, row_count: int) -> tuple[int, ...]:
    return tuple(derive_row_seed(seed, index) for index in range(int(row_count)))


def _message_content_text(content: str | list[Any] | None, message_index: int) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise OpenAIHTTPError(
            400,
            "message content must be text or text content parts",
            param=f"messages[{message_index}].content",
        )
    text_parts: list[str] = []
    for part_index, part in enumerate(content):
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            raise OpenAIHTTPError(
                400,
                "message content parts must be text objects",
                param=f"messages[{message_index}].content[{part_index}]",
            )
        part_type = part.get("type", "text")
        if part_type != "text":
            raise OpenAIHTTPError(
                400,
                f"unsupported content part type {part_type!r}; text only for now",
                code="unsupported_content_type",
                param=f"messages[{message_index}].content[{part_index}]",
            )
        text = part.get("text", "")
        if not isinstance(text, str):
            raise OpenAIHTTPError(
                400,
                "text content part must contain a string text field",
                param=f"messages[{message_index}].content[{part_index}].text",
            )
        text_parts.append(text)
    return "".join(text_parts)


def _apply_stop(text: str, stop: str | list[str] | None) -> tuple[str, str]:
    stops = _stop_strings(stop)
    if not stops:
        return text, "stop"
    earliest: int | None = None
    for item in stops:
        if not item:
            continue
        index = text.find(item)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text, "stop"
    return text[:earliest], "stop"


def _finish_reason_for_output(
    detail: GenerationOutput | None,
    fallback: str,
    *,
    server_stop: bool = False,
    tool_calls: bool = False,
) -> str:
    if tool_calls:
        return "tool_calls"
    if server_stop:
        return "stop"
    finish = None if detail is None else detail.finish_details
    if finish is None:
        return str(fallback)
    reason = finish.reason.strip().lower()
    if reason in {"length", "max_length", "max_tokens", "token_budget_exhausted", "budget_exhausted"}:
        return "length"
    if reason in {"tool_call", "tool_calls"}:
        return "tool_calls"
    if reason == "content_filter":
        return "content_filter"
    return str(fallback)


def _finish_details_payload(
    detail: GenerationOutput | None,
    finish_reason: str,
    *,
    reason_override: str | None = None,
) -> dict[str, Any]:
    finish = None if detail is None else detail.finish_details
    if finish is None:
        finish = FinishDetails(reason=finish_reason)
    return finish.to_json_dict(reason=reason_override)


def _stop_strings(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(str(item) for item in stop)


def _stop_tokens_from_stop(
    stop: str | list[str] | None,
    engine: Any,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Lower tokenizable OpenAI stop strings to token stop metadata."""

    stops = tuple(item for item in _stop_strings(stop) if item)
    tokenizer = getattr(engine, "tokenize", None)
    if not stops or not callable(tokenizer):
        return (), ()
    token_ids: list[int] = []
    sequences: list[tuple[int, ...]] = []
    for item in stops:
        try:
            ids = tuple(int(token) for token in tokenizer(item))
        except (KeyError, NotImplementedError, TypeError, ValueError):
            continue
        if len(ids) == 1:
            if ids[0] not in token_ids:
                token_ids.append(ids[0])
        elif len(ids) > 1 and ids not in sequences:
            sequences.append(ids)
    return tuple(token_ids), tuple(sequences)


def _sampling_key(sampling: SamplingParams) -> tuple[Any, ...]:
    return (
        int(sampling.max_tokens),
        float(sampling.temperature),
        float(sampling.top_p),
        int(sampling.top_k),
        float(sampling.min_p),
        float(sampling.repetition_penalty),
        float(sampling.presence_penalty),
        float(sampling.frequency_penalty),
        tuple((int(token), float(bias)) for token, bias in sampling.logit_bias),
        tuple(int(token) for token in sampling.stop_token_ids),
        tuple(tuple(int(token) for token in row) for row in sampling.stop_token_sequences),
        bool(sampling.ignore_eos),
        str(sampling.kv_storage),
        str(sampling.kv_scale_dtype),
        str(sampling.kv_scale_granularity),
        None if sampling.seed is None else int(sampling.seed),
        tuple(int(seed) for seed in sampling.row_seeds),
        bool(sampling.logprobs),
        int(sampling.top_logprobs),
    )


def _completion_logprobs(detail: GenerationOutput, text: str, *, echo_text: str = "") -> dict[str, Any]:
    tokens = list(_trim_token_logprobs(detail.token_logprobs, text))
    response_tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float] | None] = []
    offsets: list[int] = []
    cursor = 0
    if echo_text:
        response_tokens.append(echo_text)
        token_logprobs.append(None)
        top_logprobs.append(None)
        offsets.append(0)
        cursor = len(echo_text)
    for token in tokens:
        response_tokens.append(token.token_text)
        token_logprobs.append(token.logprob)
        top_logprobs.append(_completion_top_logprobs(token))
        offsets.append(cursor)
        cursor += len(token.token_text)
    return {
        "tokens": response_tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    }


def _completion_top_logprobs(token: TokenLogprob) -> dict[str, float] | None:
    if not token.top_logprobs:
        return None
    return {text: float(logprob) for _token_id, text, logprob in token.top_logprobs}


def _chat_logprobs(detail: GenerationOutput, text: str) -> dict[str, Any]:
    tokens = _trim_token_logprobs(detail.token_logprobs, text)
    return {
        "content": [
            {
                "token": token.token_text,
                "logprob": token.logprob,
                "bytes": None,
                "top_logprobs": [
                    {"token": top_text, "logprob": float(top_logprob), "bytes": None}
                    for _top_id, top_text, top_logprob in token.top_logprobs
                ],
            }
            for token in tokens
        ],
        "refusal": None,
    }


def _trim_token_logprobs(tokens: Sequence[TokenLogprob], text: str) -> tuple[TokenLogprob, ...]:
    if not tokens or not text:
        return ()
    selected: list[TokenLogprob] = []
    cursor = 0
    for token in tokens:
        next_cursor = cursor + len(token.token_text)
        if next_cursor > len(text):
            break
        selected.append(token)
        cursor = next_cursor
        if cursor >= len(text):
            break
    return tuple(selected)


def _usage(engine: Any, prompts: Sequence[str], outputs: Sequence[str]) -> dict[str, int]:
    counter = getattr(engine, "count_tokens", None)
    if callable(counter):
        prompt_tokens = sum(_safe_count(counter, text) for text in prompts)
        completion_tokens = sum(_safe_count(counter, text) for text in outputs)
    else:
        # Compatibility placeholder until public tokenizer accounting lands.
        prompt_tokens = 0
        completion_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _safe_count(counter: Any, text: str) -> int:
    try:
        return max(0, int(counter(text)))
    except Exception:
        return 0


_REASONING_OPEN_TAG = "<think>"
_REASONING_CLOSE_TAG = "</think>"


class _ReasoningSplitter:
    """Incrementally split Qwen/DeepSeek-style thinking tags from answer text."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_reasoning = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        while self._buffer:
            tag = _REASONING_CLOSE_TAG if self._in_reasoning else _REASONING_OPEN_TAG
            index = self._buffer.find(tag)
            if index >= 0:
                self._append(outputs, self._buffer[:index])
                self._buffer = self._buffer[index + len(tag) :]
                self._in_reasoning = not self._in_reasoning
                continue
            if final:
                self._append(outputs, self._buffer)
                self._buffer = ""
                break
            keep = _tag_suffix_len(self._buffer, tag)
            emit_len = len(self._buffer) - keep
            if emit_len > 0:
                self._append(outputs, self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
            break
        return outputs

    def _append(self, outputs: list[tuple[str, str]], text: str) -> None:
        if text:
            field = "reasoning_content" if self._in_reasoning else "content"
            outputs.append((field, text))


def _tag_suffix_len(text: str, tag: str) -> int:
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


def _split_reasoning(text: str) -> _ReasoningSplit:
    splitter = _ReasoningSplitter()
    parts = splitter.feed(text) + splitter.finish()
    content = "".join(part for field, part in parts if field == "content")
    reasoning = "".join(part for field, part in parts if field == "reasoning_content")
    return _ReasoningSplit(content=content, reasoning_content=reasoning)


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _parse_chat_tool_calls(text: str) -> _ParsedChatOutput:
    calls: list[_ParsedToolCall] = []
    text_parts: list[str] = []
    last_end = 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        text_parts.append(text[last_end : match.start()])
        parsed = _parsed_tool_call_from_json(match.group(1).strip())
        if parsed is None:
            text_parts.append(match.group(0))
        else:
            calls.append(parsed)
        last_end = match.end()
    text_parts.append(text[last_end:])
    if not calls:
        parsed = _parsed_tool_call_from_json(text.strip())
        if parsed is not None:
            return _ParsedChatOutput(text="", tool_calls=(parsed,))
    return _ParsedChatOutput(text="".join(text_parts).strip(), tool_calls=tuple(calls))


def _parsed_tool_call_from_json(raw: str) -> _ParsedToolCall | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return _parsed_tool_call_from_mapping(payload)


def _parsed_tool_call_from_mapping(payload: Mapping[str, Any]) -> _ParsedToolCall | None:
    function = payload.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None
    return _ParsedToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=_tool_arguments_json(arguments),
    )


def _tool_arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if arguments is None:
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _chat_message_from_parsed(parsed: _ParsedChatOutput) -> tuple[dict[str, Any], str]:
    split = _split_reasoning(parsed.text)
    message: dict[str, Any] = {"role": "assistant", "content": split.content}
    if split.reasoning_content:
        message["reasoning_content"] = split.reasoning_content
    if parsed.tool_calls:
        message["tool_calls"] = [_openai_tool_call(call) for call in parsed.tool_calls]
        return message, "tool_calls"
    return message, "stop"


def _openai_tool_call(call: _ParsedToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        },
    }


def _stream_include_usage(request: CompletionRequest | ChatCompletionRequest) -> bool:
    options = request.stream_options
    return isinstance(options, Mapping) and bool(options.get("include_usage"))


def _stream_include_hipengine(request: CompletionRequest | ChatCompletionRequest) -> bool:
    options = request.stream_options
    return isinstance(options, Mapping) and bool(options.get("include_hipengine"))


def _stream_hipengine_payload(
    event: str,
    *,
    stream_started_at: float | None = None,
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"metadata_version": 1, "event": str(event)}
    if stream_started_at is not None:
        payload["timing"] = {"elapsed_ms": round(max(0.0, (time.perf_counter() - stream_started_at) * 1000.0), 3)}
    if usage is not None:
        payload["usage"] = dict(usage)
    return payload


def _choice_hipengine_payload(
    phase: str,
    *,
    finish_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"phase": str(phase)}
    if finish_details is not None:
        payload["finish_details"] = dict(finish_details)
    return payload


def _attach_stream_hipengine(
    payload: dict[str, Any],
    *,
    include_hipengine: bool,
    event: str,
    stream_started_at: float | None,
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if include_hipengine:
        payload["hipengine"] = _stream_hipengine_payload(event, stream_started_at=stream_started_at, usage=usage)
    return payload


def _completion_stream_delta(
    response_id: str,
    created: int,
    model: str,
    text: str,
    *,
    index: int = 0,
    logprobs: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    choice = {
        "text": text,
        "index": int(index),
        "logprobs": None if logprobs is None else dict(logprobs),
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("answer")
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="delta",
            stream_started_at=stream_started_at,
        )
    )


def _completion_stream_done(
    response_id: str,
    created: int,
    model: str,
    finish_reason: str,
    *,
    index: int = 0,
    finish_details: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    finish_payload = _finish_details_payload(None, finish_reason) if finish_details is None else dict(finish_details)
    choice = {
        "text": "",
        "index": int(index),
        "logprobs": None,
        "finish_reason": finish_reason,
        "finish_details": finish_payload,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("done", finish_details=finish_payload)
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="done",
            stream_started_at=stream_started_at,
        )
    )


def _completion_stream_usage(
    response_id: str,
    created: int,
    model: str,
    usage: Mapping[str, int],
    *,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": dict(usage),
            },
            include_hipengine=include_hipengine,
            event="usage",
            stream_started_at=stream_started_at,
            usage=usage,
        )
    )


def _completion_stream_error(response_id: str, created: int, model: str, message: str) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "error",
                }
            ],
            "error": {"message": message, "type": "server_error"},
        }
    )


def _completion_stream(
    response_id: str,
    created: int,
    model: str,
    texts: Sequence[str],
    choices: Sequence[dict[str, Any]],
    *,
    usage: Mapping[str, int] | None = None,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> Iterator[str]:
    choices_by_index = {int(choice["index"]): choice for choice in choices}
    for index, text in enumerate(texts):
        choice = choices_by_index.get(index, {})
        yield _completion_stream_delta(
            response_id,
            created,
            model,
            text,
            index=index,
            logprobs=choice.get("logprobs"),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    for choice in choices:
        yield _completion_stream_done(
            response_id,
            created,
            model,
            str(choice["finish_reason"]),
            index=choice["index"],
            finish_details=choice.get("finish_details"),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    if usage is not None:
        yield _completion_stream_usage(
            response_id,
            created,
            model,
            usage,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    yield "data: [DONE]\n\n"


def _chat_stream(
    response_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str,
) -> Iterator[str]:
    yield _chat_stream_role(response_id, created, model)
    split = _split_reasoning(text)
    if split.reasoning_content:
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "reasoning_content",
            split.reasoning_content,
        )
    if split.content:
        yield _chat_stream_delta(response_id, created, model, "content", split.content)
    yield _chat_stream_done(response_id, created, model, finish_reason)
    yield "data: [DONE]\n\n"


def _chat_stream_role(
    response_id: str,
    created: int,
    model: str,
    *,
    index: int = 0,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": int(index), "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            include_hipengine=include_hipengine,
            event="role",
            stream_started_at=stream_started_at,
        )
    )


def _chat_stream_delta(
    response_id: str,
    created: int,
    model: str,
    field: str,
    text: str,
    *,
    index: int = 0,
    logprobs: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    choice: dict[str, Any] = {"index": int(index), "delta": {field: text}, "finish_reason": None}
    if logprobs is not None:
        choice["logprobs"] = dict(logprobs)
    if include_hipengine:
        phase = "think" if field == "reasoning_content" else "answer"
        choice["hipengine"] = _choice_hipengine_payload(phase)
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="delta",
            stream_started_at=stream_started_at,
        )
    )


def _chat_stream_tool_call(
    response_id: str,
    created: int,
    model: str,
    call: _ParsedToolCall,
    *,
    index: int = 0,
    tool_index: int = 0,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    choice = {
        "index": int(index),
        "delta": {
            "tool_calls": [
                {
                    "index": int(tool_index),
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            ]
        },
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("tool_call")
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="tool_call",
            stream_started_at=stream_started_at,
        )
    )


def _chat_stream_parsed(
    response_id: str,
    created: int,
    model: str,
    parsed: _ParsedChatOutput,
    finish_reason: str,
    *,
    index: int = 0,
    logprobs: Mapping[str, Any] | None = None,
    finish_details: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> Iterator[str]:
    split = _split_reasoning(parsed.text)
    if split.reasoning_content:
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "reasoning_content",
            split.reasoning_content,
            index=index,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    if split.content:
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "content",
            split.content,
            index=index,
            logprobs=logprobs,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    for tool_index, call in enumerate(parsed.tool_calls):
        yield _chat_stream_tool_call(
            response_id,
            created,
            model,
            call,
            index=index,
            tool_index=tool_index,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    done_reason = "tool_calls" if parsed.tool_calls else finish_reason
    yield _chat_stream_done(
        response_id,
        created,
        model,
        done_reason,
        index=index,
        finish_details=finish_details,
        include_hipengine=include_hipengine,
        stream_started_at=stream_started_at,
    )


def _chat_stream_done(
    response_id: str,
    created: int,
    model: str,
    finish_reason: str,
    *,
    index: int = 0,
    finish_details: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    finish_payload = _finish_details_payload(None, finish_reason) if finish_details is None else dict(finish_details)
    choice = {
        "index": int(index),
        "delta": {},
        "finish_reason": finish_reason,
        "finish_details": finish_payload,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("done", finish_details=finish_payload)
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="done",
            stream_started_at=stream_started_at,
        )
    )


def _chat_stream_usage(
    response_id: str,
    created: int,
    model: str,
    usage: Mapping[str, int],
    *,
    include_hipengine: bool = False,
    stream_started_at: float | None = None,
) -> str:
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": dict(usage),
            },
            include_hipengine=include_hipengine,
            event="usage",
            stream_started_at=stream_started_at,
            usage=usage,
        )
    )


def _chat_stream_error(response_id: str, created: int, model: str, message: str) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "error",
                }
            ],
            "error": {"message": message, "type": "server_error"},
        }
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _format_validation_error(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid request"
    first = errors[0]
    loc = ".".join(str(item) for item in first.get("loc", ()) if item != "body")
    msg = str(first.get("msg", "invalid value"))
    return f"{loc}: {msg}" if loc else msg
