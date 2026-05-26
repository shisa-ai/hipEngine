"""Long-lived scheduler-owned generation loop scaffolding.

This module is intentionally host-only and torch-free.  It wires the existing
``ResidentBatchScheduler`` to a small runner protocol so tests and early server
adapters can exercise a persistent ``submit``/``poll``/``cancel`` lifecycle
before native c>N sessions become correctness-green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken, ResidentBatchScheduler

PREFILL_DECODE_POLICIES = ("protect_decode", "protect_ttft", "fair")


@dataclass(frozen=True, slots=True)
class EngineLoopEvent:
    """One externally visible event produced by ``ResidentEngineLoop.poll``."""

    kind: str
    request_id: int | None = None
    request_ids: tuple[int, ...] = ()
    work_kind: WorkKind | None = None
    token_id: int | None = None
    completed: CompletedRequest | None = None


class EngineLoopRunner(Protocol):
    """Minimal serial-bridge/fake-runner hooks consumed by the engine loop."""

    def prefill(self, work: WorkItem) -> None:
        """Run or record one prefill work item."""

    def decode(self, work: WorkItem) -> Sequence[GeneratedToken]:
        """Return one generated token per decoded request row."""


class ResidentEngineLoop:
    """Persistent ``submit``/``poll``/``cancel`` driver for resident batches.

    The loop currently executes at most one scheduler work item per tick.  It is
    deliberately conservative: requests stay resident across polls, admission
    fills free slots, the prefill/decode choice is explicit, and completion
    reclaim is delegated to ``ResidentBatchScheduler``.
    """

    def __init__(
        self,
        runner: EngineLoopRunner,
        *,
        capacity: int,
        prefill_chunk_size: int = 256,
        context_bucket_size: int = 256,
        prefill_decode_policy: str = "protect_decode",
    ) -> None:
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if prefill_decode_policy not in PREFILL_DECODE_POLICIES:
            raise ValueError(f"prefill_decode_policy must be one of {PREFILL_DECODE_POLICIES!r}")
        self.runner = runner
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.prefill_decode_policy = prefill_decode_policy
        self._last_work_kind: WorkKind | None = None
        self.scheduler = ResidentBatchScheduler(capacity=capacity, context_bucket_size=context_bucket_size)

    @property
    def pending_count(self) -> int:
        return self.scheduler.pending_count

    @property
    def active_count(self) -> int:
        return self.scheduler.active_count

    @property
    def completed(self) -> dict[int, CompletedRequest]:
        return dict(self.scheduler.completed)

    def submit(self, prompt_tokens: Iterable[int], *, max_new_tokens: int, request_id: int | None = None) -> int:
        return self.scheduler.submit(prompt_tokens, max_new_tokens=max_new_tokens, request_id=request_id)

    def cancel(self, request_id: int) -> bool:
        """Cancel a pending or active request and reclaim active scheduler state."""

        return self.scheduler.cancel(request_id) is not None

    def poll(self, *, max_ticks: int = 1) -> tuple[EngineLoopEvent, ...]:
        """Advance the loop by up to ``max_ticks`` scheduler ticks."""

        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        events: list[EngineLoopEvent] = []
        for _ in range(int(max_ticks)):
            tick_events = self.tick()
            if not tick_events:
                break
            events.extend(tick_events)
        return tuple(events)

    def tick(self) -> tuple[EngineLoopEvent, ...]:
        """Run one admission/prefill/decode tick."""

        events: list[EngineLoopEvent] = []
        admitted = self.scheduler.admit_pending()
        events.extend(
            EngineLoopEvent(kind="admitted", request_id=request_id, request_ids=(request_id,))
            for request_id in admitted
        )

        decode = self.scheduler.next_decode_work()
        prefill_available = self.scheduler.has_prefill_work()
        if self._choose_decode_first(decode_available=decode is not None, prefill_available=prefill_available):
            assert decode is not None
            events.extend(self._run_decode(decode))
            return tuple(events)

        if prefill_available:
            prefill = self.scheduler.next_prefill_work(chunk_size=self.prefill_chunk_size)
            assert prefill is not None
            events.extend(self._run_prefill(prefill))
            return tuple(events)

        if decode is None:
            return tuple(events)
        events.extend(self._run_decode(decode))
        return tuple(events)

    def _choose_decode_first(self, *, decode_available: bool, prefill_available: bool) -> bool:
        if not decode_available:
            return False
        if not prefill_available:
            return True
        if self.prefill_decode_policy == "protect_decode":
            return True
        if self.prefill_decode_policy == "protect_ttft":
            return False
        return self._last_work_kind is WorkKind.PREFILL

    def _run_prefill(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        self.runner.prefill(work)
        self._last_work_kind = work.kind
        return (EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind),)

    def _run_decode(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        generated = tuple(self.runner.decode(work))
        completed = self.scheduler.record_generated(generated)
        self._last_work_kind = work.kind
        events = [EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind)]
        for token in generated:
            events.append(
                EngineLoopEvent(
                    kind="token",
                    request_id=token.request_id,
                    request_ids=(token.request_id,),
                    token_id=token.token_id,
                )
            )
        for item in completed:
            events.append(
                EngineLoopEvent(
                    kind="completed",
                    request_id=item.request_id,
                    request_ids=(item.request_id,),
                    completed=item,
                )
            )
        return tuple(events)


__all__ = ["EngineLoopEvent", "EngineLoopRunner", "PREFILL_DECODE_POLICIES", "ResidentEngineLoop"]
