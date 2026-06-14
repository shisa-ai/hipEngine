"""Generation-layer cooperative deadline helpers."""

from __future__ import annotations

import time
from typing import Any

from hipengine.generation.registry import FinishDetails


class GenerationDeadlineExceeded(TimeoutError):
    """Raised when a backend observes an expired request deadline."""

    def __init__(self, *, deadline_at: float | None = None) -> None:
        super().__init__("request deadline exceeded")
        self.deadline_at = None if deadline_at is None else float(deadline_at)
        self.finish_details = FinishDetails(reason="deadline_exceeded", deadline_exceeded=True)


def generation_deadline_expired(deadline_at: float | None) -> bool:
    """Return whether an absolute monotonic deadline has expired."""

    return deadline_at is not None and time.perf_counter() >= float(deadline_at)


def raise_if_generation_deadline_expired(request_or_deadline: Any) -> None:
    """Raise ``GenerationDeadlineExceeded`` when the request deadline has expired."""

    deadline_at = getattr(request_or_deadline, "deadline_at", request_or_deadline)
    if generation_deadline_expired(deadline_at):
        raise GenerationDeadlineExceeded(deadline_at=deadline_at)
