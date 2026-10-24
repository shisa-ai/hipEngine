"""Bounded lifecycle cache for provider-owned speculative stage graphs.

A provider builds a stage graph from its own kernels, buffers and stream (a PARO
target ``VERIFY|ACCEPT`` submission, a GGUF NextN proposal chain). This module
owns no HIP resource and submits no work. It owns lookup, capture identity,
eviction/invalidation, and close-all for one provider-chosen key space.

API contract
------------

``SpecGraphBinding``
    Immutable ``(bucket, addresses, configuration)`` identity. ``acquire``
    replays a cached value only when all three match the live binding.

``SpecGraphStageCache.acquire(key, live)``
    ``(value, None)`` on a matching hit, ``(None, None)`` on a miss,
    ``(None, GRAPH_STAGE_BINDING_MISS)`` after closing a drifted entry,
    ``(None, GRAPH_STAGE_CLOSED)`` once closed. A hit is promoted to
    most-recently-used; the least-recently-used entry is evicted past
    ``max_entries``.

``SpecGraphStageCache.capture(key, binding, capture, close)``
    Stores the value returned by ``capture()``. A raised ``capture`` stores
    nothing, records the failure, and returns ``GRAPH_STAGE_CAPTURE_FAILED`` for
    that attempt only; the key stays capturable. ``capture`` owns its own
    partial cleanup and must raise.

``invalidate`` / ``retire`` / ``close``
    Close and drop one entry, every entry, or every entry plus further use.
    ``invalidate`` and ``retire`` clear the recorded capture failure for the
    affected keys.

Close failures
    Every removal calls the provider's ``close`` exactly once and continues
    through the rest. A failed release keeps the value in ``pending_closes()``
    with its last error in ``pending_errors()``; ``close_failure_count`` is the
    true total and ``close_failures`` retains the first bounded set of details.
    ``retry_closes()`` re-attempts every retained release. A provider that
    zeroes each released handle skips it on retry, so nothing is destroyed twice.

Allocation after a failed release
    ``capture`` refuses with ``GRAPH_STAGE_RELEASE_BACKLOG`` once ``max_pending``
    releases are still outstanding, so a runtime that persistently refuses
    destroy cannot grow the cache without bound. The refusal drops no handle and
    is cleared by ``retry_closes()``.

Diagnostic histories (``last_reason``, ``last_failure``) are bounded windows;
counts are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

GRAPH_STAGE_CLOSED = "graph_stage_closed"
GRAPH_STAGE_BINDING_MISS = "graph_stage_binding_miss"
GRAPH_STAGE_CAPTURE_FAILED = "graph_stage_capture_failed"
GRAPH_STAGE_RELEASE_BACKLOG = "graph_stage_release_backlog"

#: Retained close-failure details. ``close_failure_count`` keeps the true total.
_CLOSE_FAILURE_DETAIL_LIMIT = 32
#: Retained ``last_reason`` / ``last_failure`` entries per diagnostic history.
_DIAGNOSTIC_HISTORY_LIMIT = 64


def describe_failure(error: BaseException) -> str:
    """Format one failure including its notes and every group member.

    ``str(error)`` omits ``__notes__``, and cleanup code attaches its context
    with notes. Dropping them here would hide the reason a release failed.
    """

    if isinstance(error, BaseExceptionGroup):
        inner = "; ".join(describe_failure(member) for member in error.exceptions)
        text = f"{type(error).__name__}: {error.message or ''} [{inner}]".strip()
    else:
        text = f"{type(error).__name__}: {error}"
    notes = tuple(getattr(error, "__notes__", ()) or ())
    if notes:
        text = f"{text} ({'; '.join(str(note) for note in notes)})"
    return text


def _named_pointers(
    addresses: Mapping[str, int] | Iterable[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = addresses.items() if isinstance(addresses, Mapping) else addresses
    named: dict[str, int] = {}
    for name, pointer in items:
        text = str(name)
        if not text:
            raise ValueError("graph binding pointer name must be non-empty")
        named[text] = int(pointer)
    return tuple(sorted(named.items()))


def _remember(history: dict[Any, str], key: Any, value: str) -> None:
    """Record one diagnostic value, keeping the most recent bounded window."""

    history.pop(key, None)
    history[key] = value
    while len(history) > _DIAGNOSTIC_HISTORY_LIMIT:
        history.pop(next(iter(history)))


@dataclass(frozen=True, slots=True)
class SpecGraphBinding:
    """Identity of one captured graph.

    ``bucket`` holds the provider's shape/slot axes, ``addresses`` every
    allocation the recorded kernels dereference, ``configuration`` the route
    identity that changes the recorded kernel sequence.
    """

    bucket: tuple[Any, ...] = ()
    addresses: tuple[tuple[str, int], ...] = ()
    configuration: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket", tuple(self.bucket))
        object.__setattr__(self, "configuration", tuple(self.configuration))
        object.__setattr__(self, "addresses", _named_pointers(self.addresses))

    @classmethod
    def build(
        cls,
        *,
        bucket: Iterable[Any] = (),
        addresses: Mapping[str, int] | Iterable[tuple[str, int]] = (),
        configuration: Iterable[Any] = (),
    ) -> "SpecGraphBinding":
        """Build a normalized binding from any address container."""

        return cls(
            bucket=tuple(bucket),
            addresses=_named_pointers(addresses),
            configuration=tuple(configuration),
        )

    def ineligibility_reason(self, live: "SpecGraphBinding") -> str | None:
        """Return why ``live`` may not replay this binding, or ``None``."""

        if not isinstance(live, SpecGraphBinding):
            raise TypeError("live binding must be SpecGraphBinding")
        if (
            self.bucket != live.bucket
            or self.configuration != live.configuration
            or self.addresses != live.addresses
        ):
            return GRAPH_STAGE_BINDING_MISS
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bucket": list(self.bucket),
            "addresses": {name: hex(pointer) for name, pointer in self.addresses},
            "configuration": list(self.configuration),
        }


@dataclass(slots=True)
class SpecGraphSlot:
    """One cached provider value with the binding it was captured under.

    ``close`` is the provider's own release callable. The cache never inspects
    ``value``; the provider owns every HIP handle.
    """

    binding: SpecGraphBinding
    value: Any
    close: Callable[[Any], None]


class SpecGraphStageCache:
    """Bounded, identity-checked cache of provider-owned stage graphs.

    Lookup order is least-recently-used first. Every removal path calls the
    provider's ``close`` once, continues through the remaining removals, and
    retains a failed release in :meth:`pending_closes` instead of dropping the
    only reference to it.
    """

    def __init__(self, *, max_entries: int = 8, max_pending: int = 8) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("graph stage cache max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("graph stage cache max_entries must be positive")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int):
            raise TypeError("graph stage cache max_pending must be an integer")
        if max_pending <= 0:
            raise ValueError("graph stage cache max_pending must be positive")
        self._max_entries = int(max_entries)
        self._max_pending = int(max_pending)
        self._slots: dict[Any, SpecGraphSlot] = {}
        self._failures: dict[Any, str] = {}
        self._reasons: dict[Any, str] = {}
        self._pending: dict[tuple[Any, int], SpecGraphSlot] = {}
        self._pending_errors: dict[tuple[Any, int], BaseException] = {}
        self._release_sequence = 0
        self._close_failure_details: list[tuple[Any, str]] = []
        self._close_failure_count = 0
        self._closed = False

    # --- inspection -------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_pending(self) -> int:
        return self._max_pending

    def __len__(self) -> int:
        """Return the number of reusable entries, excluding pending releases."""

        return len(self._slots)

    def __contains__(self, key: object) -> bool:
        return key in self._slots

    def keys(self) -> tuple[Any, ...]:
        """Return live keys in lookup order, least recently used first."""

        return tuple(self._slots)

    @property
    def close_failures(self) -> tuple[tuple[Any, str], ...]:
        """Retained close-failure details, oldest first, at most 32."""

        return tuple(self._close_failure_details)

    @property
    def close_failure_count(self) -> int:
        """Total close failures, including details dropped by the detail bound."""

        return int(self._close_failure_count)

    def pending_closes(self) -> tuple[tuple[Any, Any], ...]:
        """Return ``(key, value)`` for every release the provider has not accepted."""

        return tuple((token[0], slot.value) for token, slot in self._pending.items())

    def pending_close_keys(self) -> tuple[Any, ...]:
        """Return the keys of every retained pending release."""

        return tuple(token[0] for token in self._pending)

    def pending_errors(self) -> tuple[tuple[Any, BaseException], ...]:
        """Return the last release error for every retained pending release."""

        return tuple(
            (token[0], self._pending_errors[token])
            for token in self._pending
            if token in self._pending_errors
        )

    @property
    def pending_close_count(self) -> int:
        return len(self._pending)

    def failed_keys(self) -> tuple[Any, ...]:
        """Keys whose last capture failed and which currently hold no graph.

        This is a live view, not a latch: the next :meth:`capture` for such a
        key is attempted normally and clears the record on success.
        """

        return tuple(key for key in self._failures if key not in self._slots)

    def last_failure(self, key: Any) -> str | None:
        """Return the retained last capture failure for ``key``, if any."""

        return self._failures.get(key)

    def last_reason(self, key: Any) -> str | None:
        """Return the retained last lifecycle reason for ``key``, if any."""

        return self._reasons.get(key)

    # --- lifecycle --------------------------------------------------------

    def acquire(
        self,
        key: Any,
        live: SpecGraphBinding,
    ) -> tuple[Any | None, str | None]:
        """Return the value bound to ``key`` if ``live`` may replay it."""

        if not isinstance(live, SpecGraphBinding):
            raise TypeError("live binding must be SpecGraphBinding")
        if self._closed:
            return None, GRAPH_STAGE_CLOSED
        slot = self._slots.get(key)
        if slot is None:
            return None, None
        reason = slot.binding.ineligibility_reason(live)
        if reason is None:
            # Promote to most-recently-used so a hot bucket outlives a wide one.
            self._slots[key] = self._slots.pop(key)
            return slot.value, None
        self.invalidate(key, reason=reason)
        return None, reason

    def capture(
        self,
        key: Any,
        binding: SpecGraphBinding,
        capture: Callable[[], Any],
        close: Callable[[Any], None],
    ) -> tuple[Any | None, str | None]:
        """Record one graph under ``key``, replacing any existing entry."""

        if not isinstance(binding, SpecGraphBinding):
            raise TypeError("capture binding must be SpecGraphBinding")
        if not callable(capture):
            raise TypeError("capture must be callable")
        if not callable(close):
            raise TypeError("close must be callable")
        if self._closed:
            return None, GRAPH_STAGE_CLOSED
        if len(self._pending) >= self._max_pending:
            # A resource refusal, not an evidence gate: the cache will not
            # allocate another graph while this many releases are outstanding.
            # ``retry_closes()`` is the command that clears it.
            detail = (
                f"{GRAPH_STAGE_RELEASE_BACKLOG}: {len(self._pending)} provider "
                "releases still pending; retry_closes() must drain them"
            )
            _remember(self._failures, key, detail)
            _remember(self._reasons, key, GRAPH_STAGE_RELEASE_BACKLOG)
            return None, GRAPH_STAGE_RELEASE_BACKLOG
        self.invalidate(key, reason="replaced")
        try:
            value = capture()
        except Exception as error:
            _remember(self._failures, key, describe_failure(error))
            _remember(self._reasons, key, GRAPH_STAGE_CAPTURE_FAILED)
            return None, GRAPH_STAGE_CAPTURE_FAILED
        self._failures.pop(key, None)
        self._reasons.pop(key, None)
        self._slots[key] = SpecGraphSlot(binding=binding, value=value, close=close)
        self._evict_over_capacity()
        return value, None

    def invalidate(self, key: Any, *, reason: str) -> bool:
        """Close and drop one entry; return whether an entry was present.

        Clears the recorded capture failure for ``key``: the invalidation makes
        the key capturable again, so a stale failure record would misreport it.
        """

        _remember(self._reasons, key, str(reason))
        self._failures.pop(key, None)
        slot = self._slots.pop(key, None)
        if slot is None:
            return False
        self._close(key, slot)
        return True

    def retire(self, *, reason: str) -> tuple[Any, ...]:
        """Close and drop every entry, then clear recorded capture failures."""

        keys = tuple(self._slots)
        for key in keys:
            slot = self._slots.pop(key, None)
            if slot is not None:
                self._close(key, slot)
            _remember(self._reasons, key, str(reason))
        self._failures.clear()
        return keys

    def close(self, *, reason: str = GRAPH_STAGE_CLOSED) -> tuple[Any, ...]:
        """Retire every entry and refuse further capture or lookup."""

        keys = self.retire(reason=reason)
        self._closed = True
        return keys

    def retry_closes(self) -> tuple[Any, ...]:
        """Re-attempt every retained pending release; return the keys still pending.

        Safe after :meth:`close`: pending releases outlive the lookup table so a
        failed provider teardown is never silently dropped.
        """

        for token in tuple(self._pending):
            slot = self._pending[token]
            try:
                slot.close(slot.value)
            except BaseException as error:  # noqa: BLE001 - keep the release retained
                self._record_close_failure(token[0], error)
                self._pending_errors[token] = error
            else:
                del self._pending[token]
                self._pending_errors.pop(token, None)
        return self.pending_close_keys()

    # --- internals --------------------------------------------------------

    def _evict_over_capacity(self) -> None:
        while len(self._slots) > self._max_entries:
            key = next(iter(self._slots))
            slot = self._slots.pop(key)
            self._close(key, slot)
            _remember(self._reasons, key, "evicted")

    def _close(self, key: Any, slot: SpecGraphSlot) -> None:
        """Release one provider value; never abort the remaining removals."""

        try:
            slot.close(slot.value)
        except BaseException as error:  # noqa: BLE001 - retain, then keep removing
            self._record_close_failure(key, error)
            # Retain the value: dropping it here would strand provider handles
            # with no reference left to retry the release through.
            self._release_sequence += 1
            token = (key, self._release_sequence)
            self._pending[token] = slot
            self._pending_errors[token] = error

    def _record_close_failure(self, key: Any, error: BaseException) -> None:
        self._close_failure_count += 1
        if len(self._close_failure_details) < _CLOSE_FAILURE_DETAIL_LIMIT:
            self._close_failure_details.append((key, describe_failure(error)))


__all__ = [
    "GRAPH_STAGE_BINDING_MISS",
    "GRAPH_STAGE_CAPTURE_FAILED",
    "GRAPH_STAGE_CLOSED",
    "GRAPH_STAGE_RELEASE_BACKLOG",
    "SpecGraphBinding",
    "SpecGraphSlot",
    "SpecGraphStageCache",
    "describe_failure",
]
