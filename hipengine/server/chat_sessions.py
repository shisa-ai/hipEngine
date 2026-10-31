"""Application-local chat transcript and continuation state.

This owner holds the server's app-local chat session transcripts, pending
session-creation reservations and short-lived continuation records. It receives
explicit policy values and per-operation data. It does not import the HTTP
application, an engine, ``ServerConfig`` or ``app.state``; callers translate its
typed ``SessionStateError`` values at the HTTP boundary.

Records are immutable. Transcript commit, prefix reads, snapshot copies, fork
and rollback copies deep-copy message mappings so request code cannot mutate
stored transcript state through a borrowed reference. Continuation records
retain the tokenizer metadata mapping supplied at creation. ``snapshot_record``
exposes the stored record only for the duration of its lease; it does not copy.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContinuationRecord:
    id: str
    endpoint: str
    model_id: str
    auth_principal: str
    session_id: str | None
    tokenizer: dict[str, Any]
    prompts: tuple[str, ...]
    generated_texts: tuple[str, ...]
    created: float
    expires_at: float
    response_format: Any | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None

    def resume_prompts(self) -> tuple[str, ...]:
        return tuple(
            f"{prompt}{generated}"
            for prompt, generated in zip(self.prompts, self.generated_texts, strict=True)
        )


@dataclass(frozen=True)
class ChatSessionRecord:
    id: str
    messages: tuple[dict[str, Any], ...]
    created: float
    updated: float


@dataclass(frozen=True)
class SessionListing:
    records: tuple[ChatSessionRecord, ...]
    active_continuations: int
    pending_creations: int


@dataclass(frozen=True)
class RollbackResult:
    record: ChatSessionRecord
    previous_message_count: int
    rolled_back: bool


class SessionStateError(Exception):
    """Typed state failure translated to ``OpenAIHTTPError`` by the caller."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str,
        param: str | None = None,
        error_type: str = "invalid_request_error",
        details: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        requested_model: str | None = None,
        record_rejection: bool = False,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.code = code
        self.param = param
        self.error_type = error_type
        self.details = None if details is None else dict(details)
        self.headers = {} if headers is None else dict(headers)
        self.requested_model = requested_model
        self.record_rejection = bool(record_rejection)
        super().__init__(message)


def session_metadata(record: ChatSessionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "message_count": len(record.messages),
        "created": int(record.created),
        "updated": int(record.updated),
    }


def _chat_session_message_copy(message: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(message))


def _chat_message_to_session_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        payload = dict(message)
    elif hasattr(message, "model_dump"):
        payload = message.model_dump(exclude_none=True)
    else:
        payload = message.dict(exclude_none=True)
    return _chat_session_message_copy(
        {str(key): value for key, value in payload.items() if value is not None}
    )


def _assistant_visible_session_message(message: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    if "content" in message:
        payload["content"] = deepcopy(message.get("content"))
    else:
        payload["content"] = ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        payload["tool_calls"] = deepcopy(tool_calls)
    return payload


class ChatSessionStore:
    """Own app-local transcripts, reservations and continuation records."""

    def __init__(
        self,
        *,
        max_sessions: int | None,
        continuation_ttl_seconds: int,
        retry_after_seconds: int,
        clock: Callable[[], float] = time.time,
        continuation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.max_sessions = max_sessions
        self.continuation_ttl_seconds = int(continuation_ttl_seconds)
        self.retry_after_seconds = int(retry_after_seconds)
        self._clock = clock
        self._continuation_id_factory = (
            continuation_id_factory
            if continuation_id_factory is not None
            else lambda: f"gen_{uuid.uuid4().hex}"
        )
        self.sessions: dict[str, ChatSessionRecord] = {}
        self.pending: set[str] = set()
        self.continuations: dict[str, ContinuationRecord] = {}
        self._session_lock = asyncio.Lock()
        self._continuation_lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self.sessions)

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def summary(self) -> dict[str, Any]:
        return {
            "resident_context": True,
            "active": len(self.sessions),
            "pending_creations": len(self.pending),
            "max_active": None if self.max_sessions is None else int(self.max_sessions),
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "total_messages": sum(len(record.messages) for record in self.sessions.values()),
            "continuations": {
                "active": len(self.continuations),
                "ttl_seconds": self.continuation_ttl_seconds,
            },
        }

    async def listing(self) -> SessionListing:
        async with self._session_lock:
            records = tuple(
                sorted(
                    self.sessions.values(),
                    key=lambda item: (-item.updated, item.id),
                )
            )
            pending_creations = len(self.pending)
        # Acquire the continuation lock only after releasing the session lock.
        # pop_continuation takes continuation -> session; never the reverse.
        async with self._continuation_lock:
            self._cleanup_expired_locked()
            active_continuations = len(self.continuations)
        return SessionListing(
            records=records,
            active_continuations=active_continuations,
            pending_creations=pending_creations,
        )

    async def reserve(
        self,
        session_id: str | None,
        *,
        cache_action: str | None,
        continuation_id: str | None,
        requested_model: str | None,
    ) -> str | None:
        if session_id is None:
            return None
        if continuation_id is not None:
            return None
        if cache_action in (None, "append_none"):
            return None
        async with self._session_lock:
            if session_id in self.sessions:
                return None
            if session_id in self.pending:
                raise self._engine_busy_error(
                    "chat session is being created",
                    details={"overload_source": "chat_session_pending"},
                    requested_model=requested_model,
                )
            if self.max_sessions is not None and (
                len(self.sessions) + len(self.pending)
            ) >= int(self.max_sessions):
                raise self._engine_busy_error(
                    "chat session limit is full",
                    details={
                        "overload_source": "chat_session_cap",
                        "max_active_chat_sessions": int(self.max_sessions),
                    },
                    requested_model=requested_model,
                )
            self.pending.add(session_id)
            return session_id

    async def release(self, session_id: str | None) -> None:
        if session_id is None:
            return
        async with self._session_lock:
            self.pending.discard(session_id)

    async def prefix_messages(
        self,
        session_id: str | None,
    ) -> tuple[dict[str, Any], ...]:
        if session_id is None:
            return ()
        async with self._session_lock:
            record = self.sessions.get(session_id)
            if record is None:
                return ()
            return tuple(_chat_session_message_copy(message) for message in record.messages)

    async def commit(
        self,
        session_id: str | None,
        *,
        cache_action: str | None,
        request_messages: Sequence[Any],
        raw_output: str,
        visible_message: Mapping[str, Any],
        reset_session: bool = False,
        commit_base_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if session_id is None or cache_action in (None, "append_none"):
            return
        prompt_messages = tuple(
            _chat_message_to_session_dict(message) for message in request_messages
        )
        if cache_action == "append_prompt_only":
            appended = prompt_messages
        elif cache_action == "append_visible_only":
            appended = (*prompt_messages, _assistant_visible_session_message(visible_message))
        elif cache_action == "append_all":
            appended = (*prompt_messages, {"role": "assistant", "content": str(raw_output)})
        else:
            return
        now = self._clock()
        async with self._session_lock:
            previous = self.sessions.get(session_id)
            created = now if previous is None or reset_session else previous.created
            if commit_base_messages is not None:
                base = tuple(
                    _chat_session_message_copy(message)
                    for message in commit_base_messages
                )
            else:
                base = () if previous is None or reset_session else previous.messages
            self.sessions[session_id] = ChatSessionRecord(
                id=session_id,
                messages=(*base, *appended),
                created=created,
                updated=now,
            )

    async def get_copy(self, session_id: str) -> ChatSessionRecord | None:
        async with self._session_lock:
            record = self.sessions.get(session_id)
            if record is None:
                return None
            return ChatSessionRecord(
                id=record.id,
                messages=tuple(
                    _chat_session_message_copy(message) for message in record.messages
                ),
                created=record.created,
                updated=record.updated,
            )

    @asynccontextmanager
    async def snapshot_record(
        self,
        session_id: str,
    ) -> AsyncIterator[ChatSessionRecord | None]:
        """Lease the stored record while the caller builds its snapshot payload.

        The session lock stays held through the caller's body, matching the
        pre-move snapshot export. A missing session yields ``None`` inside the
        lease so the caller can raise its existing 404 without releasing early.
        """

        async with self._session_lock:
            yield self.sessions.get(session_id)

    async def put_if_capacity(
        self,
        record: ChatSessionRecord,
        *,
        requested_model: str | None = None,
    ) -> ChatSessionRecord:
        async with self._session_lock:
            is_new = record.id not in self.sessions
            if (
                is_new
                and self.max_sessions is not None
                and len(self.sessions) >= int(self.max_sessions)
            ):
                raise self._engine_busy_error(
                    "chat session limit is full",
                    details={
                        "overload_source": "chat_session_cap",
                        "max_active_chat_sessions": int(self.max_sessions),
                    },
                    requested_model=requested_model,
                )
            self.sessions[record.id] = record
            return record

    async def delete(self, session_id: str) -> bool:
        async with self._session_lock:
            return self.sessions.pop(session_id, None) is not None

    async def fork(self, source_id: str, target_id: str) -> ChatSessionRecord:
        async with self._session_lock:
            source = self.sessions.get(source_id)
            if source is None:
                raise SessionStateError(
                    404,
                    "chat session does not exist",
                    code="invalid_request",
                    param="session_id",
                )
            if target_id in self.sessions:
                raise SessionStateError(
                    400,
                    "target chat session already exists",
                    code="invalid_request",
                    param="id",
                )
            if (
                self.max_sessions is not None
                and len(self.sessions) >= int(self.max_sessions)
            ):
                raise self._engine_busy_error(
                    "chat session limit is full",
                    details={
                        "overload_source": "chat_session_cap",
                        "max_active_chat_sessions": int(self.max_sessions),
                    },
                    requested_model=None,
                )
            now = self._clock()
            forked = ChatSessionRecord(
                id=target_id,
                messages=tuple(
                    _chat_session_message_copy(message) for message in source.messages
                ),
                created=now,
                updated=now,
            )
            self.sessions[target_id] = forked
            return forked

    async def rollback(self, session_id: str, target_count: int) -> RollbackResult:
        async with self._session_lock:
            record = self.sessions.get(session_id)
            if record is None:
                raise SessionStateError(
                    404,
                    "chat session does not exist",
                    code="invalid_request",
                    param="session_id",
                )
            previous_count = len(record.messages)
            if target_count > previous_count:
                raise SessionStateError(
                    400,
                    "rollback message_count cannot exceed current session message_count",
                    code="invalid_request",
                    param="message_count",
                )
            rolled_back = target_count != previous_count
            if rolled_back:
                now = self._clock()
                record = ChatSessionRecord(
                    id=session_id,
                    messages=tuple(
                        _chat_session_message_copy(message)
                        for message in record.messages[:target_count]
                    ),
                    created=record.created,
                    updated=now,
                )
                self.sessions[session_id] = record
            return RollbackResult(
                record=record,
                previous_message_count=previous_count,
                rolled_back=rolled_back,
            )

    async def pop_continuation(
        self,
        raw_id: Any,
        *,
        endpoint: str,
        model_id: str,
        auth_principal: str,
        session_id: str | None,
        current_tokenizer_metadata: Callable[[], Mapping[str, Any] | None],
    ) -> ContinuationRecord | None:
        if raw_id is None:
            return None
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise SessionStateError(
                400,
                "continuation_id must be a non-empty string",
                code="invalid_request",
                param="continuation_id",
            )
        continuation_id = raw_id.strip()
        async with self._continuation_lock:
            record = self.continuations.get(continuation_id)
            if record is None:
                self._cleanup_expired_locked()
                raise SessionStateError(
                    400,
                    "continuation_id is invalid or has already been consumed",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.expires_at <= self._clock():
                self.continuations.pop(continuation_id, None)
                self._cleanup_expired_locked()
                raise SessionStateError(
                    410,
                    "continuation_id has expired",
                    code="continuation_expired",
                    param="continuation_id",
                )
            if record.endpoint != endpoint:
                raise SessionStateError(
                    400,
                    f"continuation_id was created for {record.endpoint}, not {endpoint}",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.model_id != model_id:
                raise SessionStateError(
                    400,
                    "continuation_id is scoped to a different served model",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.auth_principal != auth_principal:
                raise SessionStateError(
                    400,
                    "continuation_id is scoped to a different auth principal",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.session_id != session_id:
                raise SessionStateError(
                    400,
                    "continuation_id is scoped to a different session",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if endpoint == "chat" and record.session_id is not None:
                async with self._session_lock:
                    if record.session_id not in self.sessions:
                        raise SessionStateError(
                            400,
                            "continuation_id is scoped to a deleted session",
                            code="invalid_continuation",
                            param="continuation_id",
                        )
            current_tokenizer = current_tokenizer_metadata()
            if current_tokenizer is not None and record.tokenizer != dict(current_tokenizer):
                raise SessionStateError(
                    400,
                    "continuation_id is scoped to a different tokenizer",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            self._cleanup_expired_locked()
            return self.continuations.pop(continuation_id)

    async def store_continuation(
        self,
        *,
        endpoint: str,
        model_id: str,
        auth_principal: str,
        session_id: str | None,
        tokenizer_metadata: Mapping[str, Any],
        prompts: Sequence[str],
        generated_texts: Sequence[str],
        response_format: Any | None,
        guided_json: Any | None,
        guided_regex: Any | None,
        guided_choice: Any | None,
        guided_patch: Any | None,
        guided_diff: Any | None,
    ) -> ContinuationRecord:
        now = self._clock()
        record = ContinuationRecord(
            id=self._continuation_id_factory(),
            endpoint=endpoint,
            model_id=model_id,
            auth_principal=auth_principal,
            session_id=session_id,
            tokenizer=dict(tokenizer_metadata),
            prompts=tuple(str(prompt) for prompt in prompts),
            generated_texts=tuple(str(text) for text in generated_texts),
            created=now,
            expires_at=now + self.continuation_ttl_seconds,
            response_format=response_format,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_patch=guided_patch,
            guided_diff=guided_diff,
        )
        async with self._continuation_lock:
            self._cleanup_expired_locked(now)
            self.continuations[record.id] = record
        return record

    def _cleanup_expired_locked(self, now: float | None = None) -> None:
        current = self._clock() if now is None else float(now)
        expired = [
            continuation_id
            for continuation_id, record in self.continuations.items()
            if record.expires_at <= current
        ]
        for continuation_id in expired:
            self.continuations.pop(continuation_id, None)

    def _engine_busy_error(
        self,
        message: str,
        *,
        details: Mapping[str, Any],
        requested_model: str | None,
    ) -> SessionStateError:
        return SessionStateError(
            429,
            message,
            error_type="rate_limit_error",
            code="engine_busy",
            details=details,
            headers={"Retry-After": str(self.retry_after_seconds)},
            requested_model=requested_model,
            record_rejection=True,
        )
