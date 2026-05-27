"""OpenAI-compatible FastAPI surface for hipEngine.

The server layer is optional and intentionally thin: it adapts OpenAI-style JSON
requests to the torch-free ``hipengine.LLM.generate()`` library API.  The current
runtime is still single-request/c=1, so requests are serialized behind a lock
until continuous batching lands in the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from hipengine.kvcache import resolve_prefix_cache_mode


_LOGGER = logging.getLogger("uvicorn.error")


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
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    generation_batch_window_ms: float = 0.0
    metrics: str = "off"
    prefix_cache: str = "off"
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
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    echo: bool = False
    logprobs: int | None = None
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None


class ChatMessage(_OpenAIBaseModel):
    role: str
    content: str | list[Any] | None = ""
    name: str | None = None


class ChatCompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None


@dataclass(frozen=True)
class _GeneratedBatch:
    outputs: list[str]
    usage: dict[str, int]


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


@dataclass
class _QueuedGeneration:
    prompts: tuple[str, ...]
    sampling: SamplingParams
    future: asyncio.Future[list[str]] | None = None
    stream_queue: asyncio.Queue[object] | None = None
    cancelled: bool = False


class _GenerationBatcher:
    """Coalesce compatible HTTP generations into prompt-list calls."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any],
        generation_lock: asyncio.Lock,
        batch_window_seconds: float,
    ) -> None:
        self._engine_factory = engine_factory
        self._generation_lock = generation_lock
        self._batch_window_seconds = max(0.0, float(batch_window_seconds))
        self._queue: deque[_QueuedGeneration] = deque()
        self._worker: asyncio.Task[None] | None = None

    async def submit(self, prompts: Sequence[str], sampling: SamplingParams) -> list[str]:
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        if self._batch_window_seconds <= 0.0:
            return await self._generate_prompts(prompt_tuple, sampling)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[str]] = loop.create_future()
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
        if self._batch_window_seconds <= 0.0:
            for output in await self._generate_prompts(prompt_tuple, sampling):
                yield output
            return
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

    async def _generate_prompts(self, prompts: tuple[str, ...], sampling: SamplingParams) -> list[str]:
        async with self._generation_lock:
            raw_outputs = await run_in_threadpool(
                self._engine_factory().generate,
                prompts,
                sampling,
            )
        outputs = [str(item) for item in raw_outputs]
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        return outputs


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


@dataclass(frozen=True)
class _ReasoningSplit:
    content: str
    reasoning_content: str


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
    generation_lock = asyncio.Lock()

    def get_llm() -> Any:
        if app.state.hipengine_llm is None:
            app.state.hipengine_llm = LLM(config.model, backend=config.backend, quant=config.quant)
        return app.state.hipengine_llm

    generation_batcher = _GenerationBatcher(
        engine_factory=get_llm,
        generation_lock=generation_lock,
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
        return effective_max_context_tokens(engine)

    async def eager_load_model() -> None:
        max_tokens = max(1, int(config.eager_load_max_tokens))
        if not config.eager_load:
            max_context = configured_max_context_tokens()
            _LOGGER.info(
                "Config: model=%s served_model=%s max_context_tokens=%s "
                "chat_default_max_tokens=auto kv_storage=%s kv_scale_dtype=%s "
                "kv_scale_granularity=%s eager_load=False",
                config.model,
                config.model_id,
                "auto" if max_context is None else str(max_context),
                config.kv_storage,
                config.kv_scale_dtype,
                config.kv_scale_granularity,
            )
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
        async with generation_lock:
            engine = get_llm()
            max_context = await ensure_resident_context(engine, sampling, phase="startup")
            _LOGGER.info(
                "Config: model=%s served_model=%s max_context_tokens=%s "
                "chat_default_max_tokens=auto kv_storage=%s kv_scale_dtype=%s "
                "kv_scale_granularity=%s eager_load=True",
                config.model,
                config.model_id,
                "unknown" if max_context is None else str(max_context),
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
            await run_in_threadpool(engine.generate, (config.eager_load_prompt,), sampling)
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
    async def openai_error_handler(_request: Request, exc: OpenAIHTTPError) -> JSONResponse:
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
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        message = _format_validation_error(exc)
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
        return SamplingParams(
            max_tokens=_request_max_tokens(request, prompts, engine, effective_max_context_tokens(engine)),
            temperature=float(request.temperature if request.temperature is not None else 0.0),
            top_p=float(request.top_p if request.top_p is not None else 1.0),
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
            engine = get_llm()
            async with generation_lock:
                await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                sampling = sampling_params(request, prompts, engine)
                if _request_n(request) > 1:
                    sampling = replace(
                        sampling,
                        row_seeds=_row_seeds_for_request(request.seed, len(prompts)),
                    )
                _validate_context_budget(effective_max_context_tokens(engine), engine, prompts, sampling)
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

        outputs = [str(item) for item in raw_outputs]
        if len(outputs) != len(prompts):
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(
                500,
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts",
                error_type="server_error",
                code="bad_generator_output",
            )
        batch = _GeneratedBatch(outputs=outputs, usage=_usage(engine, prompts, outputs))
        app.state.hipengine_server_metrics.record_success(batch.usage)
        return batch

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
        prompts = _normalize_prompts(request.prompt)
        n = _request_n(request)
        expanded_prompts = _expand_prompts_for_n(prompts, n)
        batch = await generate(expanded_prompts, request)
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choices = []
        final_texts: list[str] = []
        for index, (prompt, output) in enumerate(zip(expanded_prompts, batch.outputs)):
            text, finish_reason = _apply_stop(output, request.stop)
            if request.echo:
                text = prompt + text
            final_texts.append(text)
            choice = {
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": finish_reason,
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
            return StreamingResponse(
                _completion_stream(response_id, created, config.model_id, final_texts, choices),
                media_type="text/event-stream",
            )
        return response

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model)
        prompt = render_chat_prompt(request.messages)
        if request.stream:
            streamer = stream_chat_completion_many if _request_n(request) > 1 else stream_chat_completion
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
        for index, output in enumerate(batch.outputs):
            text, finish_reason = _apply_stop(output, request.stop)
            split = _split_reasoning(text)
            message: dict[str, Any] = {"role": "assistant", "content": split.content}
            if split.reasoning_content:
                message["reasoning_content"] = split.reasoning_content
            choice = {
                "index": index,
                "message": message,
                "finish_reason": finish_reason,
            }
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
        try:
            n = _request_n(request)
            batch = await generate(tuple(prompt for _ in range(n)), request)
            for index, output in enumerate(batch.outputs):
                text, finish_reason = _apply_stop(output, request.stop)
                split = _split_reasoning(text)
                yield _chat_stream_role(response_id, created, config.model_id, index=index)
                if split.reasoning_content:
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        "reasoning_content",
                        split.reasoning_content,
                        index=index,
                    )
                if split.content:
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        "content",
                        split.content,
                        index=index,
                    )
                yield _chat_stream_done(response_id, created, config.model_id, finish_reason, index=index)
        except OpenAIHTTPError as exc:
            yield _chat_stream_error(response_id, created, config.model_id, exc.message)
        except Exception as exc:  # pragma: no cover - real runtime failures
            yield _chat_stream_error(response_id, created, config.model_id, f"generation failed: {exc}")
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
            yield _chat_stream_error(response_id, created, config.model_id, exc.message)
            yield "data: [DONE]\n\n"
            return
        engine = get_llm()
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        full_text: list[str] = []
        splitter = _ReasoningSplitter()

        try:
            async with generation_lock:
                await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                sampling = sampling_params(request, (prompt,), engine)
                _validate_context_budget(effective_max_context_tokens(engine), engine, (prompt,), sampling)
            yield _chat_stream_role(response_id, created, config.model_id)
            async for token in generation_batcher.stream((prompt,), sampling):
                text = str(token)
                if not text:
                    continue
                full_text.append(text)
                for field, chunk in splitter.feed(text):
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        field,
                        chunk,
                    )
            for field, chunk in splitter.finish():
                yield _chat_stream_delta(
                    response_id,
                    created,
                    config.model_id,
                    field,
                    chunk,
                )
        except OpenAIHTTPError as exc:
            app.state.hipengine_server_metrics.record_failure()
            yield _chat_stream_error(response_id, created, config.model_id, exc.message)
            yield "data: [DONE]\n\n"
            return
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            yield _chat_stream_error(response_id, created, config.model_id, str(exc))
            return
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            yield _chat_stream_error(response_id, created, config.model_id, str(exc))
            return
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            yield _chat_stream_error(response_id, created, config.model_id, message)
            return

        text, finish_reason = _apply_stop("".join(full_text), request.stop)
        if text != "".join(full_text):
            # Stop strings can split across yielded chunks; current streaming keeps
            # transport simple and reports the stop after generation completes.
            finish_reason = "stop"
        app.state.hipengine_server_metrics.record_success(_usage(engine, (prompt,), [text]))
        yield _chat_stream_done(response_id, created, config.model_id, finish_reason)
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
        values[key] = float(data.get(key, 0) or 0)
    return values


def _graph_bucket_metric_values(engine: Any | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "entries": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "miss_reasons": {},
        "kernel_time_histogram_ns": {},
    }
    stats = _first_stats_object(engine, ("graph_buckets", "graph_bucket_cache", "graph_bucket_stats"))
    if stats is None:
        return values
    data = _stats_to_mapping(stats)
    for key in ("entries", "hits", "misses"):
        values[key] = float(data.get(key, 0) or 0)
    values["miss_reasons"] = _non_negative_metric_mapping(data.get("miss_reasons"))
    values["kernel_time_histogram_ns"] = _non_negative_metric_mapping(data.get("kernel_time_histogram_ns"))
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


def _non_negative_metric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
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


def render_chat_prompt(messages: Sequence[ChatMessage | Mapping[str, Any]]) -> str:
    """Render OpenAI chat messages to a Qwen-style text prompt.

    This is intentionally tokenizer-independent so the API can stay a thin
    adapter around ``LLM.generate()``.  Model-specific chat-template rendering
    can replace this helper once tokenizers are exposed by the runtime.
    """

    if not messages:
        raise OpenAIHTTPError(400, "messages must contain at least one item", param="messages")
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if isinstance(message, Mapping):
            role_value = message.get("role", "")
            content_value = message.get("content", "")
        else:
            role_value = message.role
            content_value = message.content
        role = str(role_value).strip()
        if not role:
            raise OpenAIHTTPError(
                400,
                "message role must be non-empty",
                param=f"messages[{index}].role",
            )
        content = _message_content_text(content_value, index)
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    rendered.append("<|im_start|>assistant\n")
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
) -> int:
    if request.max_tokens is not None:
        return max(0, int(request.max_tokens))
    if isinstance(request, ChatCompletionRequest) and max_context_tokens is not None:
        remaining = min(
            int(max_context_tokens) - _count_tokens_for_admission(engine, str(prompt)) - 1
            for prompt in prompts
        )
        return max(0, int(remaining))
    return 8192 if isinstance(request, ChatCompletionRequest) else 16


def _validate_generation_request(config: ServerConfig, request: CompletionRequest | ChatCompletionRequest) -> None:
    _request_n(request)
    if isinstance(request, CompletionRequest) and request.logprobs is not None:
        raise OpenAIHTTPError(
            400,
            "logprobs is not currently supported",
            code="unsupported_parameter",
            param="logprobs",
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


def _expand_prompts_for_n(prompts: Sequence[str], n: int) -> tuple[str, ...]:
    return tuple(prompt for prompt in prompts for _ in range(int(n)))


def _choice_request_id(response_id: str, prompt_index: int, choice_index: int) -> str:
    return f"{response_id}:prompt-{int(prompt_index)}:choice-{int(choice_index)}"


def _row_seeds_for_request(seed: int | None, row_count: int) -> tuple[int, ...]:
    base = 0 if seed is None else int(seed)
    mask = (1 << 63) - 1
    values = []
    for index in range(int(row_count)):
        value = (base + 0x9E3779B97F4A7C15 + index * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= (value >> 30)
        value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
        values.append(value & mask)
    return tuple(values)


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


def _stop_strings(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(str(item) for item in stop)


def _sampling_key(sampling: SamplingParams) -> tuple[Any, ...]:
    return (
        int(sampling.max_tokens),
        float(sampling.temperature),
        float(sampling.top_p),
        bool(sampling.ignore_eos),
        str(sampling.kv_storage),
        str(sampling.kv_scale_dtype),
        str(sampling.kv_scale_granularity),
        None if sampling.seed is None else int(sampling.seed),
        tuple(int(seed) for seed in sampling.row_seeds),
    )


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


def _completion_stream(
    response_id: str,
    created: int,
    model: str,
    texts: Sequence[str],
    choices: Sequence[dict[str, Any]],
) -> Iterator[str]:
    for index, text in enumerate(texts):
        yield _sse(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "text": text,
                        "index": index,
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        )
    for choice in choices:
        yield _sse(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "text": "",
                        "index": choice["index"],
                        "logprobs": None,
                        "finish_reason": choice["finish_reason"],
                    }
                ],
            }
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


def _chat_stream_role(response_id: str, created: int, model: str, *, index: int = 0) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": int(index), "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )


def _chat_stream_delta(
    response_id: str,
    created: int,
    model: str,
    field: str,
    text: str,
    *,
    index: int = 0,
) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": int(index), "delta": {field: text}, "finish_reason": None}],
        }
    )


def _chat_stream_done(response_id: str, created: int, model: str, finish_reason: str, *, index: int = 0) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": int(index), "delta": {}, "finish_reason": finish_reason}],
        }
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
