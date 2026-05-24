"""OpenAI-compatible FastAPI surface for hipEngine.

The server layer is optional and intentionally thin: it adapts OpenAI-style JSON
requests to the torch-free ``hipengine.LLM.generate()`` library API.  The current
runtime is still single-request/c=1, so requests are serialized behind a lock
until continuous batching lands in the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:  # Pydantic v2; FastAPI's current default.
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 compatibility
    ConfigDict = None  # type: ignore[assignment]

from starlette.concurrency import run_in_threadpool

from hipengine import LLM, SamplingParams


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
    max_tokens: int | None = Field(default=8192, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None


@dataclass(frozen=True)
class _GeneratedBatch:
    outputs: list[str]
    usage: dict[str, int]


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
    app.state.hipengine_config = config
    app.state.hipengine_llm = llm
    generation_lock = asyncio.Lock()

    def get_llm() -> Any:
        if app.state.hipengine_llm is None:
            app.state.hipengine_llm = LLM(config.model, backend=config.backend, quant=config.quant)
        return app.state.hipengine_llm

    async def eager_load_model() -> None:
        if not config.eager_load:
            return
        max_tokens = max(1, int(config.eager_load_max_tokens))
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        )
        async with generation_lock:
            await run_in_threadpool(get_llm().generate, (config.eager_load_prompt,), sampling)

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

    def sampling_params(request: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
        return SamplingParams(
            max_tokens=int(request.max_tokens if request.max_tokens is not None else 16),
            temperature=float(request.temperature if request.temperature is not None else 0.0),
            top_p=float(request.top_p if request.top_p is not None else 1.0),
            ignore_eos=bool(request.ignore_eos),
            kv_storage=request.kv_storage or "auto",
            kv_scale_dtype=request.kv_scale_dtype or "fp16",
            kv_scale_granularity=request.kv_scale_granularity or "per_token_head",
        )

    async def generate(
        prompts: Sequence[str],
        request: CompletionRequest | ChatCompletionRequest,
    ) -> _GeneratedBatch:
        _validate_generation_request(request)
        sampling = sampling_params(request)
        engine = get_llm()
        try:
            async with generation_lock:
                raw_outputs = await run_in_threadpool(engine.generate, tuple(prompts), sampling)
        except NotImplementedError as exc:
            raise OpenAIHTTPError(400, str(exc), code="unsupported_parameter") from exc
        except ValueError as exc:
            raise OpenAIHTTPError(400, str(exc), code="invalid_request") from exc
        except Exception as exc:  # pragma: no cover - exercised by real runtime failures
            raise OpenAIHTTPError(
                500,
                f"generation failed: {exc}",
                error_type="server_error",
                code="generation_failed",
            ) from exc

        outputs = [str(item) for item in raw_outputs]
        if len(outputs) != len(prompts):
            raise OpenAIHTTPError(
                500,
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts",
                error_type="server_error",
                code="bad_generator_output",
            )
        return _GeneratedBatch(outputs=outputs, usage=_usage(engine, prompts, outputs))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": config.model_id}

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
        batch = await generate(prompts, request)
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choices = []
        final_texts: list[str] = []
        for index, (prompt, output) in enumerate(zip(prompts, batch.outputs)):
            text, finish_reason = _apply_stop(output, request.stop)
            if request.echo:
                text = prompt + text
            final_texts.append(text)
            choices.append(
                {
                    "text": text,
                    "index": index,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            )
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
            return StreamingResponse(
                stream_chat_completion(prompt, request),
                media_type="text/event-stream",
            )
        batch = await generate((prompt,), request)
        text, finish_reason = _apply_stop(batch.outputs[0], request.stop)
        split = _split_reasoning(text)
        message: dict[str, Any] = {"role": "assistant", "content": split.content}
        if split.reasoning_content:
            message["reasoning_content"] = split.reasoning_content
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        response = {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": config.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": batch.usage,
        }
        return response

    async def stream_chat_completion(
        prompt: str,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        _validate_generation_request(request)
        sampling = sampling_params(request)
        engine = get_llm()
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        full_text: list[str] = []
        splitter = _ReasoningSplitter()
        sentinel = object()

        def next_or_done(iterator):
            try:
                return next(iterator)
            except StopIteration:
                return sentinel

        try:
            async with generation_lock:
                streamer = getattr(engine, "stream", None)
                if not callable(streamer):
                    text = (await run_in_threadpool(engine.generate, (prompt,), sampling))[0]
                    iterator = iter((text,))
                else:
                    iterator = await run_in_threadpool(streamer, prompt, sampling)
                yield _chat_stream_role(response_id, created, config.model_id)
                while True:
                    token = await run_in_threadpool(next_or_done, iterator)
                    if token is sentinel:
                        break
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
        except NotImplementedError as exc:
            yield _chat_stream_error(response_id, created, config.model_id, str(exc))
            return
        except ValueError as exc:
            yield _chat_stream_error(response_id, created, config.model_id, str(exc))
            return
        except Exception as exc:  # pragma: no cover - real runtime failures
            message = f"generation failed: {exc}"
            yield _chat_stream_error(response_id, created, config.model_id, message)
            return

        text, finish_reason = _apply_stop("".join(full_text), request.stop)
        if text != "".join(full_text):
            # Stop strings can split across yielded chunks; current streaming keeps
            # transport simple and reports the stop after generation completes.
            finish_reason = "stop"
        yield _chat_stream_done(response_id, created, config.model_id, finish_reason)
        yield "data: [DONE]\n\n"

    return app


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


def _validate_generation_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    if request.n not in (None, 1):
        raise OpenAIHTTPError(
            400,
            "only n=1 is currently supported",
            code="unsupported_parameter",
            param="n",
        )
    if isinstance(request, CompletionRequest) and request.logprobs is not None:
        raise OpenAIHTTPError(
            400,
            "logprobs is not currently supported",
            code="unsupported_parameter",
            param="logprobs",
        )
    try:
        from hipengine.kvcache import resolve_kv_policy

        resolve_kv_policy(
            request.kv_storage or "auto",
            scale_dtype=request.kv_scale_dtype or "fp16",
            scale_granularity=request.kv_scale_granularity or "per_token_head",
        )
    except ValueError as exc:
        raise OpenAIHTTPError(400, str(exc), code="invalid_kv_policy", param="kv_storage") from exc


def _normalize_prompts(prompt: str | list[str]) -> tuple[str, ...]:
    if isinstance(prompt, str):
        return (prompt,)
    if not prompt:
        raise OpenAIHTTPError(400, "prompt must not be empty", param="prompt")
    return tuple(str(item) for item in prompt)


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


def _chat_stream_role(response_id: str, created: int, model: str) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )


def _chat_stream_delta(response_id: str, created: int, model: str, field: str, text: str) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}],
        }
    )


def _chat_stream_done(response_id: str, created: int, model: str, finish_reason: str) -> str:
    return _sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
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
