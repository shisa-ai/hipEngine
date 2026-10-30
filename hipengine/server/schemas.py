"""OpenAI-compatible request schemas owned by the HTTP boundary.

This module deliberately depends only on Pydantic and the standard library so
that request models can be imported without pulling in the FastAPI application,
the generation engine, or any model/runtime module.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

try:  # Pydantic v2; FastAPI's current default.
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 compatibility
    ConfigDict = None  # type: ignore[assignment]


if ConfigDict is not None:

    class _OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

else:  # pragma: no cover - Pydantic v1 compatibility

    class _OpenAIBaseModel(BaseModel):
        class Config:
            extra = "allow"


class SongGenerationRequest(_OpenAIBaseModel):
    """One lyrics-to-audio request for a model that declares the song surface.

    The sampling settings are the model's own product defaults, not request
    fields: this pipeline's quality path is the pinned preset, and a caller that
    needs different arithmetic uses the Python API's phase overrides.
    """

    model: str | None = None
    style: str = Field(min_length=1)
    lyrics: str = Field(min_length=1)
    cot: str = "full"
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    abc: str | None = None
    cfg_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    steps: int | None = Field(default=None, ge=1)
    response_format: str = "wav"
    request_id: str | None = None


class CompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    prompt: str | list[str] | list[int] | list[list[int]] | None = None
    max_tokens: int | None = Field(default=16, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=0, ge=0)
    min_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=1.0, gt=0.0)
    presence_penalty: float | None = Field(default=0.0)
    frequency_penalty: float | None = Field(default=0.0)
    logit_bias: dict[str, float] | None = None
    suppress_token_ids: list[int] | None = None
    min_tokens: int | None = Field(default=0, ge=0)
    eos_token_id: int | None = Field(default=None, ge=0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    timeout_ms: float | None = Field(default=None, gt=0.0)
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    echo: bool = False
    logprobs: int | None = Field(default=None, ge=0, le=20)
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None
    response_format: Any | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None
    continuation_id: Any | None = None
    session: Any | None = None
    speculative_mtp: bool | dict[str, Any] | None = None
    speculative: bool | dict[str, Any] | None = None


class ChatMessage(_OpenAIBaseModel):
    role: str
    content: str | list[Any] | None = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=0, ge=0)
    min_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=1.0, gt=0.0)
    presence_penalty: float | None = Field(default=0.0)
    frequency_penalty: float | None = Field(default=0.0)
    logit_bias: dict[str, float] | None = None
    suppress_token_ids: list[int] | None = None
    min_tokens: int | None = Field(default=0, ge=0)
    eos_token_id: int | None = Field(default=None, ge=0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    timeout_ms: float | None = Field(default=None, gt=0.0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    invalid_tool_call_error_mode: str | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    max_think_tokens: int | None = Field(default=None, ge=0)
    min_answer_tokens: int | None = Field(default=None, ge=0)
    hard_think_cap: int | None = Field(default=None, ge=0)
    soft_close_window: int | None = Field(default=None, ge=0)
    hard_close_message: str | None = None
    hard_close_sequence: str | None = None
    thinking_token_budget: int | None = Field(default=None, ge=0)
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
    response_format: Any | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None
    continuation_id: Any | None = None
    session: Any | None = None
    speculative_mtp: bool | dict[str, Any] | None = None
    speculative: bool | dict[str, Any] | None = None


class SessionForkRequest(_OpenAIBaseModel):
    id: str


class SessionRollbackRequest(_OpenAIBaseModel):
    message_count: int = Field(..., ge=0)


class TokenizeRequest(_OpenAIBaseModel):
    text: str


class DetokenizeRequest(_OpenAIBaseModel):
    token_ids: list[int]
    skip_special: bool = False


class TokenDiagnosticRequest(_OpenAIBaseModel):
    text: str | None = None
    messages: list[ChatMessage] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    max_think_tokens: int | None = Field(default=None, ge=0)
    min_answer_tokens: int | None = Field(default=None, ge=0)
    hard_think_cap: int | None = Field(default=None, ge=0)
    soft_close_window: int | None = Field(default=None, ge=0)
    hard_close_message: str | None = None
    hard_close_sequence: str | None = None
    thinking_token_budget: int | None = Field(default=None, ge=0)
    chat_template_kwargs: dict[str, Any] | None = None
    thinking: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None
    session: Any | None = None


class FitContextRequest(TokenDiagnosticRequest):
    max_tokens: int | None = Field(default=None, ge=0)


__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "CompletionRequest",
    "DetokenizeRequest",
    "FitContextRequest",
    "SessionForkRequest",
    "SessionRollbackRequest",
    "SongGenerationRequest",
    "TokenDiagnosticRequest",
    "TokenizeRequest",
]
