"""Generation-layer cooperative deadline and cancellation helpers."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from hipengine.generation.registry import FinishDetails


class GenerationDeadlineExceeded(TimeoutError):
    """Raised when a backend observes an expired request deadline."""

    def __init__(self, *, deadline_at: float | None = None) -> None:
        super().__init__("request deadline exceeded")
        self.deadline_at = None if deadline_at is None else float(deadline_at)
        self.finish_details = FinishDetails(reason="deadline_exceeded", deadline_exceeded=True)


class GenerationCancelled(RuntimeError):
    """Raised when a backend observes a cancelled request token."""

    def __init__(self, finish_details: FinishDetails | None = None) -> None:
        details = finish_details or FinishDetails(reason="cancelled", cancelled=True)
        super().__init__("request cancelled")
        self.finish_details = FinishDetails.from_value(details)


class GenerationCancellationToken:
    """Thread-safe cancellation token shared between server and generation code."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._finish_details = FinishDetails(reason="cancelled", cancelled=True)
        self._cancel_dispatch: Callable[[FinishDetails], None] | None = None
        self._cancel_requested = False

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def finish_details(self) -> FinishDetails:
        with self._lock:
            return self._finish_details

    def cancel(self, finish_details: FinishDetails | None = None) -> None:
        details = FinishDetails.from_value(
            finish_details or FinishDetails(reason="cancelled", cancelled=True)
        )
        with self._lock:
            self._finish_details = details
            self._cancel_requested = True
            dispatch = self._cancel_dispatch
            if dispatch is None:
                self._event.set()
        if dispatch is not None:
            dispatch(details)

    def set_cancel_dispatch(self, dispatch: Callable[[FinishDetails], None]) -> None:
        """Route cancellation through a scheduler command before exposing it."""

        with self._lock:
            self._cancel_dispatch = dispatch

    def clear_cancel_dispatch(self, dispatch: Callable[[FinishDetails], None]) -> None:
        with self._lock:
            if self._cancel_dispatch is dispatch:
                self._cancel_dispatch = None
                if self._cancel_requested:
                    self._event.set()

    def acknowledge_cancel(self, finish_details: FinishDetails | None = None) -> None:
        """Publish cancellation only after scheduler-owned row reclamation."""

        with self._lock:
            if finish_details is not None:
                self._finish_details = FinishDetails.from_value(finish_details)
            self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(self.finish_details)


def generation_deadline_expired(deadline_at: float | None) -> bool:
    """Return whether an absolute monotonic deadline has expired."""

    return deadline_at is not None and time.perf_counter() >= float(deadline_at)


def raise_if_generation_cancelled(request_or_token: Any) -> None:
    """Raise ``GenerationCancelled`` when a request cancellation token is set."""

    token = getattr(request_or_token, "cancellation_token", request_or_token)
    if token is not None:
        token.raise_if_cancelled()


def raise_if_generation_deadline_expired(request_or_deadline: Any, *, cancellation_token: Any | None = None) -> None:
    """Raise when a request cancellation token is set or its deadline expired."""

    raise_if_generation_cancelled(cancellation_token or getattr(request_or_deadline, "cancellation_token", None))
    deadline_at = getattr(request_or_deadline, "deadline_at", request_or_deadline)
    if generation_deadline_expired(deadline_at):
        raise GenerationDeadlineExceeded(deadline_at=deadline_at)
