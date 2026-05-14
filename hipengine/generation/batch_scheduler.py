"""Batch-friendly generation scheduler shell.

The scheduler is model-agnostic: it owns request ids, pending/admitted queues,
physical batch slots, prefill/decode work items, and completion routing.  Model
runners consume the emitted ``WorkItem`` metadata and report generated tokens
back through ``record_generated``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from hipengine.dispatch import ActiveBatch, BatchShapeKey, RequestState, WorkItem, WorkKind


@dataclass(frozen=True, slots=True)
class BatchGenerateRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int

    @classmethod
    def from_tokens(cls, request_id: int, prompt_tokens: Iterable[int], *, max_new_tokens: int) -> "BatchGenerateRequest":
        return cls(request_id=int(request_id), prompt_tokens=tuple(int(token) for token in prompt_tokens), max_new_tokens=int(max_new_tokens))


@dataclass(frozen=True, slots=True)
class GeneratedToken:
    request_id: int
    token_id: int
    finished: bool = False


@dataclass(frozen=True, slots=True)
class CompletedRequest:
    request_id: int
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    finished: bool


class ResidentBatchScheduler:
    """Continuous-batching scheduler shell for resident decode runners."""

    def __init__(self, *, capacity: int, context_bucket_size: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if context_bucket_size <= 0:
            raise ValueError("context_bucket_size must be positive")
        self.capacity = int(capacity)
        self.context_bucket_size = int(context_bucket_size)
        self.active_batch = ActiveBatch(self.capacity)
        self._pending: deque[RequestState] = deque()
        self._completed: dict[int, CompletedRequest] = {}
        self._next_request_id = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_count(self) -> int:
        return self.active_batch.active_count

    @property
    def completed(self) -> Mapping[int, CompletedRequest]:
        return self._completed

    def submit(self, prompt_tokens: Iterable[int], *, max_new_tokens: int, request_id: int | None = None) -> int:
        rid = self._allocate_request_id() if request_id is None else int(request_id)
        if rid in self.active_batch.requests or any(req.request_id == rid for req in self._pending) or rid in self._completed:
            raise ValueError(f"request_id {rid} already exists")
        self._pending.append(RequestState.from_tokens(rid, prompt_tokens, max_new_tokens=max_new_tokens))
        return rid

    def admit_pending(self) -> tuple[int, ...]:
        """Fill free slots from the pending queue and return admitted request ids."""

        admitted: list[int] = []
        while self._pending and self.active_batch.active_count < self.capacity:
            request = self._pending.popleft()
            self.active_batch.admit(request)
            admitted.append(request.request_id)
        return tuple(admitted)

    def compact(self, order: Sequence[int] | None = None):
        return self.active_batch.compact(order=order)

    def next_prefill_work(self, *, chunk_size: int) -> WorkItem | None:
        """Emit one prefill chunk and advance the request's prompt cursor."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for request_id in self.active_batch.active_request_ids:
            request = self.active_batch.requests[request_id]
            if request.remaining_prefill <= 0:
                continue
            updated, chunk = request.take_prefill(chunk_size)
            self.active_batch.update_request(updated)
            return WorkItem(
                kind=WorkKind.PREFILL,
                request_ids=(request_id,),
                row_to_request=(request_id,),
                token_rows=(chunk,),
            )
        return None

    def next_decode_work(self) -> WorkItem | None:
        """Emit one decode step over active requests with completed prefill."""

        request_ids = tuple(
            request_id
            for request_id in self.active_batch.active_request_ids
            if self.active_batch.requests[request_id].remaining_prefill == 0
            and self.active_batch.requests[request_id].remaining_decode > 0
            and not self.active_batch.requests[request_id].finished
        )
        if not request_ids:
            return None
        return WorkItem(kind=WorkKind.DECODE, request_ids=request_ids, row_to_request=request_ids)

    def record_generated(self, tokens: Sequence[GeneratedToken | tuple[int, int] | tuple[int, int, bool]]) -> tuple[CompletedRequest, ...]:
        """Record generated tokens and reclaim newly completed requests."""

        completed: list[CompletedRequest] = []
        for item in tokens:
            token = _coerce_generated_token(item)
            request = self.active_batch.requests[token.request_id]
            updated = request.append_generated(token.token_id, finished=token.finished)
            self.active_batch.update_request(updated)
            if updated.finished:
                self.active_batch.finish(updated.request_id)
                reclaimed = self.active_batch.reclaim(updated.request_id)
                done = CompletedRequest(
                    request_id=reclaimed.request_id,
                    prompt_tokens=reclaimed.prompt_tokens,
                    generated_tokens=reclaimed.generated_tokens,
                    finished=reclaimed.finished,
                )
                self._completed[done.request_id] = done
                completed.append(done)
        return tuple(completed)

    def shape_key(self, *, mode: WorkKind | str, top_k: int = 0, experts_per_token: int = 0, replay_steps: int = 1) -> BatchShapeKey:
        return self.active_batch.shape_key(
            mode=mode,
            context_bucket_size=self.context_bucket_size,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
        )

    def _allocate_request_id(self) -> int:
        rid = self._next_request_id
        self._next_request_id += 1
        return rid


def _coerce_generated_token(item: GeneratedToken | tuple[int, int] | tuple[int, int, bool]) -> GeneratedToken:
    if isinstance(item, GeneratedToken):
        return item
    if len(item) == 2:
        request_id, token_id = item
        return GeneratedToken(int(request_id), int(token_id), False)
    request_id, token_id, finished = item
    return GeneratedToken(int(request_id), int(token_id), bool(finished))


__all__ = ["BatchGenerateRequest", "CompletedRequest", "GeneratedToken", "ResidentBatchScheduler"]
