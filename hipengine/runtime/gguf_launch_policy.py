"""Owner-local reuse of immutable, resource-free GGUF launch decisions."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Iterator, TypeVar

_Decision = TypeVar('_Decision')


class LaunchDecisionOwner:
    """Bounded decision interning after normal policy/resource resolution.

    This is not a dispatch cache: callers must resolve current policy and live
    resource availability before submitting a resource-free frozen decision.
    """

    def __init__(self, *, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError('launch decision capacity must be positive')
        self._capacity = int(capacity)
        self._generation: int | None = None
        self._decisions: OrderedDict[object, object] = OrderedDict()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._decisions)

    def intern(self, decision: _Decision) -> _Decision:
        generation = int(decision.registry_generation)
        with self._lock:
            if generation != self._generation:
                self._decisions.clear()
                self._generation = generation
            key = (type(decision), decision)
            existing = self._decisions.get(key)
            if existing is not None:
                self._decisions.move_to_end(key)
                return existing
            self._decisions[key] = decision
            if len(self._decisions) > self._capacity:
                self._decisions.popitem(last=False)
            return decision


_owner: ContextVar[LaunchDecisionOwner | None] = ContextVar(
    'gguf_launch_decision_owner', default=None,
)


@contextmanager
def launch_decision_session(owner: LaunchDecisionOwner | None) -> Iterator[None]:
    """Bind one profile owner's decisions for this execution context only."""
    token = _owner.set(owner)
    try:
        yield
    finally:
        _owner.reset(token)


def own_launch_decision(decision: _Decision) -> _Decision:
    """Reuse equal immutable decisions, never launch-local resident bindings."""
    owner = _owner.get()
    return decision if owner is None else owner.intern(decision)
