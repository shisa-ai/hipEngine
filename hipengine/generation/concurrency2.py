"""Generation-2 child/parent request and isolated output host types."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from hipengine.generation.registry import GenerationOutput, GenerationStreamChunk


class ChildPhase(str, Enum):
    """Scheduler-owned lifecycle phase for one generated sequence."""

    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"
    TERMINAL = "terminal"


class OutputKind(str, Enum):
    TOKEN = "token"
    CHUNK = "chunk"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class ChildRequest:
    """One independently schedulable generated sequence.

    Parent identity is formatting metadata only.  Backend resources and resident
    state are intentionally absent from this immutable frontend value object.
    """

    request_id: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    parent_id: int | None = None
    choice_index: int = 0
    streaming: bool = False
    priority: int = 0
    deadline_at: float | None = None
    sampling_key: str = "greedy"

    def __post_init__(self) -> None:
        request_id = int(self.request_id)
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        prompt_tokens = tuple(int(token) for token in self.prompt_tokens)
        if any(token < 0 for token in prompt_tokens):
            raise ValueError("prompt token ids must be non-negative")
        max_new_tokens = int(self.max_new_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        parent_id = None if self.parent_id is None else int(self.parent_id)
        if parent_id is not None and parent_id < 0:
            raise ValueError("parent_id must be non-negative when set")
        choice_index = int(self.choice_index)
        if choice_index < 0:
            raise ValueError("choice_index must be non-negative")
        if parent_id is None and choice_index != 0:
            raise ValueError("standalone child choice_index must be zero")
        deadline = None if self.deadline_at is None else float(self.deadline_at)
        if deadline is not None and deadline <= 0.0:
            raise ValueError("deadline_at must be positive when set")
        sampling_key = str(self.sampling_key)
        if not sampling_key or sampling_key != sampling_key.strip():
            raise ValueError("sampling_key must be a non-empty trimmed string")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "prompt_tokens", prompt_tokens)
        object.__setattr__(self, "max_new_tokens", max_new_tokens)
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "choice_index", choice_index)
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "deadline_at", deadline)
        object.__setattr__(self, "sampling_key", sampling_key)


@dataclass(frozen=True, slots=True)
class ParentRequest:
    """Frontend-only aggregation metadata for independently owned children."""

    parent_id: int
    child_request_ids: tuple[int, ...]
    choice_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        parent_id = int(self.parent_id)
        if parent_id < 0:
            raise ValueError("parent_id must be non-negative")
        child_ids = tuple(int(request_id) for request_id in self.child_request_ids)
        choices = tuple(int(choice) for choice in self.choice_indices)
        if not child_ids:
            raise ValueError("parent must include at least one child")
        if len(child_ids) != len(choices):
            raise ValueError("choice_indices must align with child_request_ids")
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("parent child request ids must be unique")
        if any(request_id < 0 for request_id in child_ids):
            raise ValueError("child request ids must be non-negative")
        if any(choice < 0 for choice in choices) or len(choices) != len(set(choices)):
            raise ValueError("parent choice_index values must be unique and non-negative")
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "child_request_ids", child_ids)
        object.__setattr__(self, "choice_indices", choices)

    @classmethod
    def from_children(cls, parent_id: int, children: tuple[ChildRequest, ...]) -> "ParentRequest":
        normalized = tuple(children)
        if any(child.parent_id != int(parent_id) for child in normalized):
            raise ValueError("every child parent_id must match the parent")
        return cls(
            parent_id=int(parent_id),
            child_request_ids=tuple(child.request_id for child in normalized),
            choice_indices=tuple(child.choice_index for child in normalized),
        )


@dataclass(frozen=True, slots=True)
class EngineOutput:
    """One request-ID-keyed event published after a scheduler commit barrier."""

    kind: OutputKind
    request_id: int
    token_id: int | None = None
    token_index: int | None = None
    generated_token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None
    stream_chunk: GenerationStreamChunk | None = None
    generation_output: GenerationOutput | None = None
    error: BaseException | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", OutputKind(self.kind))
        request_id = int(self.request_id)
        if request_id < 0:
            raise ValueError("request_id must be non-negative")
        object.__setattr__(self, "request_id", request_id)
        generated = tuple(int(token) for token in self.generated_token_ids)
        if any(token < 0 for token in generated):
            raise ValueError("generated token ids must be non-negative")
        object.__setattr__(self, "generated_token_ids", generated)
        if self.kind is OutputKind.TOKEN:
            if self.token_id is None or int(self.token_id) < 0:
                raise ValueError("token output requires a non-negative token_id")
            if self.token_index is None or int(self.token_index) < 0:
                raise ValueError("token output requires a non-negative token_index")
            if (
                generated
                or self.finish_reason is not None
                or self.generation_output is not None
                or self.error is not None
            ):
                raise ValueError("token output cannot carry terminal fields")
            if self.stream_chunk is not None and not isinstance(
                self.stream_chunk, GenerationStreamChunk
            ):
                raise TypeError("stream_chunk must be GenerationStreamChunk when set")
            object.__setattr__(self, "token_id", int(self.token_id))
            object.__setattr__(self, "token_index", int(self.token_index))
        elif self.kind is OutputKind.CHUNK:
            if self.token_id is not None or self.token_index is not None:
                raise ValueError("chunk output cannot carry token identity")
            if self.stream_chunk is None or not isinstance(
                self.stream_chunk, GenerationStreamChunk
            ):
                raise TypeError("chunk output requires a GenerationStreamChunk")
            if (
                generated
                or self.finish_reason is not None
                or self.generation_output is not None
                or self.error is not None
            ):
                raise ValueError("chunk output cannot carry terminal fields")
        else:
            if self.token_id is not None or self.token_index is not None:
                raise ValueError("terminal output cannot carry token fields")
            if self.stream_chunk is not None:
                raise ValueError("terminal output cannot carry a stream_chunk")
            if self.generation_output is not None:
                if not isinstance(self.generation_output, GenerationOutput):
                    raise TypeError("generation_output must be GenerationOutput when set")
                output_ids = self.generation_output.generated_token_ids
                if output_ids is not None and output_ids != generated:
                    raise ValueError("generation_output token history must match terminal tokens")
            if self.error is not None and not isinstance(self.error, BaseException):
                raise TypeError("terminal error must be a BaseException when set")
            reason = "" if self.finish_reason is None else str(self.finish_reason)
            if not reason or reason != reason.strip():
                raise ValueError("terminal output requires a non-empty finish_reason")
            object.__setattr__(self, "finish_reason", reason)


@dataclass(frozen=True, slots=True)
class CollectedOutput:
    request_id: int
    generated_token_ids: tuple[int, ...]
    finish_reason: str
    generation_output: GenerationOutput | None = None
    error: BaseException | None = None


@runtime_checkable
class OutputCollector(Protocol):
    """Non-blocking engine-side output sink owned by one child request."""

    request_id: int | None

    def bind(self, request_id: int) -> None: ...

    def publish(self, output: EngineOutput) -> bool: ...

    def wait(self, timeout: float | None = None) -> CollectedOutput | None: ...

    @property
    def result(self) -> CollectedOutput | None: ...

    @property
    def generated_token_ids(self) -> tuple[int, ...]: ...


class _CollectorBase:
    def __init__(self, *, max_output_tokens: int) -> None:
        maximum = int(max_output_tokens)
        if maximum < 0:
            raise ValueError("max_output_tokens must be non-negative")
        self.max_output_tokens = maximum
        self.request_id: int | None = None
        self._tokens: list[int] = []
        self._result: CollectedOutput | None = None
        self._condition = threading.Condition()

    def bind(self, request_id: int) -> None:
        normalized = int(request_id)
        if normalized < 0:
            raise ValueError("request_id must be non-negative")
        with self._condition:
            if self.request_id is not None and self.request_id != normalized:
                raise ValueError("collector is already bound to another request")
            self.request_id = normalized

    def _validate_output_locked(self, output: EngineOutput) -> None:
        if self.request_id is None:
            raise RuntimeError("collector must be bound before publication")
        if output.request_id != self.request_id:
            raise ValueError("output request_id does not match collector")
        if self._result is not None:
            raise RuntimeError("cannot publish after terminal output")

    def _publish_token_locked(self, output: EngineOutput) -> bool:
        if len(self._tokens) >= self.max_output_tokens:
            return False
        if output.token_index != len(self._tokens):
            raise ValueError("token_index must be contiguous per child")
        assert output.token_id is not None
        self._tokens.append(output.token_id)
        return True

    def _publish_terminal_locked(self, output: EngineOutput) -> bool:
        if output.generated_token_ids != tuple(self._tokens):
            raise ValueError("terminal token history must match published child tokens")
        assert self.request_id is not None and output.finish_reason is not None
        self._result = CollectedOutput(
            request_id=self.request_id,
            generated_token_ids=tuple(self._tokens),
            finish_reason=output.finish_reason,
            generation_output=output.generation_output,
            error=output.error,
        )
        self._condition.notify_all()
        return True

    @property
    def result(self) -> CollectedOutput | None:
        with self._condition:
            return self._result

    @property
    def generated_token_ids(self) -> tuple[int, ...]:
        with self._condition:
            return tuple(self._tokens)

    def wait(self, timeout: float | None = None) -> CollectedOutput | None:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._result is None:
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._condition.wait(timeout=remaining)
            return self._result


class BlockingOutputCollector(_CollectorBase):
    """Bounded final buffer for a blocking/library child."""

    def publish(self, output: EngineOutput) -> bool:
        with self._condition:
            self._validate_output_locked(output)
            if output.kind is OutputKind.TOKEN:
                return self._publish_token_locked(output)
            if output.kind is OutputKind.CHUNK:
                return True
            return self._publish_terminal_locked(output)


class StreamingOutputCollector(_CollectorBase):
    """Bounded non-blocking token mailbox with an out-of-band terminal slot."""

    def __init__(
        self,
        *,
        max_output_tokens: int,
        max_chunks: int,
        enqueue_token_events: bool = True,
    ) -> None:
        super().__init__(max_output_tokens=max_output_tokens)
        maximum = int(max_chunks)
        if maximum <= 0:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = maximum
        self.enqueue_token_events = bool(enqueue_token_events)
        self._mailbox: deque[EngineOutput] = deque()
        self._terminal_event: EngineOutput | None = None
        self._terminal_delivered = False

    def publish(self, output: EngineOutput) -> bool:
        with self._condition:
            self._validate_output_locked(output)
            if output.kind is OutputKind.TOKEN:
                enqueue = self.enqueue_token_events or output.stream_chunk is not None
                if enqueue and len(self._mailbox) >= self.max_chunks:
                    return False
                accepted = self._publish_token_locked(output)
                if accepted and enqueue:
                    self._mailbox.append(output)
                    self._condition.notify_all()
                return accepted
            if output.kind is OutputKind.CHUNK:
                if len(self._mailbox) >= self.max_chunks:
                    return False
                self._mailbox.append(output)
                self._condition.notify_all()
                return True
            accepted = self._publish_terminal_locked(output)
            self._terminal_event = output
            return accepted

    def drain(self, *, max_chunks: int | None = None) -> tuple[EngineOutput, ...]:
        with self._condition:
            limit = (
                len(self._mailbox) + (1 if self._terminal_event is not None and not self._terminal_delivered else 0)
                if max_chunks is None
                else int(max_chunks)
            )
            if limit < 0:
                raise ValueError("max_chunks must be non-negative")
            drained: list[EngineOutput] = []
            while self._mailbox and len(drained) < limit:
                drained.append(self._mailbox.popleft())
            if (
                len(drained) < limit
                and self._terminal_event is not None
                and not self._terminal_delivered
            ):
                drained.append(self._terminal_event)
                self._terminal_delivered = True
            return tuple(drained)

    def wait_for_event(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while not self._mailbox and (
                self._terminal_event is None or self._terminal_delivered
            ):
                if self._terminal_event is not None and self._terminal_delivered:
                    return False
                if deadline is None:
                    self._condition.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
            return True


__all__ = [
    "BlockingOutputCollector",
    "ChildPhase",
    "ChildRequest",
    "CollectedOutput",
    "EngineOutput",
    "OutputCollector",
    "OutputKind",
    "ParentRequest",
    "StreamingOutputCollector",
]
