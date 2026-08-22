"""Deterministic fixed/burst/Poisson load scenarios for resident engine loops."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

from hipengine.generation.engine_loop import GenerationAdmissionRejected, ResidentEngineLoop


@dataclass(frozen=True, slots=True)
class LoadArrival:
    arrival_id: str
    at_tick: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    disconnect_after_ticks: int | None = None

    def __post_init__(self) -> None:
        if not self.arrival_id:
            raise ValueError("arrival_id must be non-empty")
        if self.at_tick < 0 or self.max_new_tokens < 0:
            raise ValueError("arrival tick and max_new_tokens must be non-negative")
        if not self.prompt_tokens or any(int(token) < 0 for token in self.prompt_tokens):
            raise ValueError("load prompt tokens must be non-empty and non-negative")
        if self.disconnect_after_ticks is not None and self.disconnect_after_ticks <= 0:
            raise ValueError("disconnect_after_ticks must be positive when set")


@dataclass(frozen=True, slots=True)
class LoadScenarioResult:
    ticks: int
    offered: int
    submitted: int
    completed: int
    disconnected: int
    retryable_rejections: int
    max_active: int
    max_pending: int
    occupancy_history: tuple[int, ...]
    request_ids: tuple[tuple[str, int], ...]
    finish_reasons: tuple[tuple[str, str], ...]
    drained: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "offered": self.offered,
            "submitted": self.submitted,
            "completed": self.completed,
            "disconnected": self.disconnected,
            "retryable_rejections": self.retryable_rejections,
            "max_active": self.max_active,
            "max_pending": self.max_pending,
            "occupancy_history": list(self.occupancy_history),
            "request_ids": dict(self.request_ids),
            "finish_reasons": dict(self.finish_reasons),
            "drained": self.drained,
        }


def run_load_scenario(
    loop: ResidentEngineLoop,
    arrivals: Sequence[LoadArrival],
    *,
    max_ticks: int,
    retry_rejected: bool = True,
) -> LoadScenarioResult:
    """Drive arrivals/cancellation/ticks and require deterministic final drain."""

    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")
    ordered = sorted(arrivals, key=lambda item: (item.at_tick, item.arrival_id))
    if len({arrival.arrival_id for arrival in ordered}) != len(ordered):
        raise ValueError("arrival_id values must be unique")
    future = deque(ordered)
    retry_queue: deque[LoadArrival] = deque()
    request_ids: dict[str, int] = {}
    submitted_at: dict[str, int] = {}
    disconnected_ids: set[str] = set()
    retryable_rejections = 0
    occupancy: list[int] = []
    pending_history: list[int] = []
    tick = 0
    while tick < int(max_ticks):
        due: list[LoadArrival] = []
        while future and future[0].at_tick <= tick:
            due.append(future.popleft())
        while retry_queue:
            due.append(retry_queue.popleft())
        for arrival in due:
            if arrival.arrival_id in request_ids:
                continue
            try:
                request_id = loop.submit(
                    arrival.prompt_tokens,
                    max_new_tokens=arrival.max_new_tokens,
                )
            except GenerationAdmissionRejected:
                retryable_rejections += 1
                if retry_rejected:
                    retry_queue.append(arrival)
                continue
            request_ids[arrival.arrival_id] = request_id
            submitted_at[arrival.arrival_id] = tick

        for arrival in ordered:
            if (
                arrival.disconnect_after_ticks is None
                or arrival.arrival_id in disconnected_ids
                or arrival.arrival_id not in request_ids
            ):
                continue
            if tick - submitted_at[arrival.arrival_id] >= arrival.disconnect_after_ticks:
                if loop.disconnect(request_ids[arrival.arrival_id]):
                    disconnected_ids.add(arrival.arrival_id)

        loop.tick()
        occupancy.append(loop.active_count)
        pending_history.append(loop.pending_count)
        tick += 1
        if (
            not future
            and not retry_queue
            and loop.pending_count == 0
            and loop.active_count == 0
        ):
            break

    completed_by_request = loop.completed
    finish_reasons = tuple(
        sorted(
            (
                arrival_id,
                completed_by_request[request_id].finish_reason,
            )
            for arrival_id, request_id in request_ids.items()
            if request_id in completed_by_request
        )
    )
    drained = (
        not future
        and not retry_queue
        and loop.pending_count == 0
        and loop.active_count == 0
    )
    return LoadScenarioResult(
        ticks=tick,
        offered=len(ordered),
        submitted=len(request_ids),
        completed=len(finish_reasons),
        disconnected=len(disconnected_ids),
        retryable_rejections=retryable_rejections,
        max_active=max(occupancy, default=0),
        max_pending=max(pending_history, default=0),
        occupancy_history=tuple(occupancy),
        request_ids=tuple(sorted(request_ids.items())),
        finish_reasons=finish_reasons,
        drained=drained,
    )


def poisson_arrivals(
    *,
    count: int,
    rate_per_tick: float,
    seed: int,
    prompt_tokens: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[LoadArrival, ...]:
    if int(count) <= 0 or float(rate_per_tick) <= 0:
        raise ValueError("Poisson count and rate_per_tick must be positive")
    rng = random.Random(int(seed))
    current = 0.0
    rows: list[LoadArrival] = []
    for index in range(int(count)):
        uniform = max(rng.random(), 1e-12)
        current += -math.log(uniform) / float(rate_per_tick)
        rows.append(
            LoadArrival(
                arrival_id=f"poisson:{index}",
                at_tick=int(current),
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
        )
    return tuple(rows)


__all__ = [
    "LoadArrival",
    "LoadScenarioResult",
    "poisson_arrivals",
    "run_load_scenario",
]
