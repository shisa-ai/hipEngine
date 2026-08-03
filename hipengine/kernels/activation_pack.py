"""Owner-scoped exact activation-pack reuse.

The cache is deliberately one-entry and context-local.  Pointer equality is
safe only while an execution owner guarantees that the input is immutable, so
callers must open a fresh scope around one projection group.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator


ActivationPackKey = tuple[int, int, int, int, int, int]


@dataclass
class _ActivationPackReuseState:
    enabled: bool
    key: ActivationPackKey | None = None

    def invalidate(self) -> None:
        self.key = None


_activation_pack_reuse_state: ContextVar[_ActivationPackReuseState | None] = (
    ContextVar("activation_pack_reuse_state", default=None)
)


@contextlib.contextmanager
def activation_pack_reuse_scope(*, enabled: bool = True) -> Iterator[None]:
    """Bound exact pack reuse to one immutable projection group.

    Nested scopes invalidate their parent on both entry and exit because a
    child may write the same activation scratch through another key.
    """

    parent = _activation_pack_reuse_state.get()
    if parent is not None:
        parent.invalidate()
    state = _ActivationPackReuseState(enabled=bool(enabled))
    token = _activation_pack_reuse_state.set(state)
    try:
        yield
    finally:
        state.invalidate()
        _activation_pack_reuse_state.reset(token)
        if parent is not None:
            parent.invalidate()


def launch_scoped_activation_pack(
    producer: Callable[..., None],
    input_ptr: int,
    activation_ptr: int,
    rows: int,
    in_features: int,
    *,
    row_batch: int,
    stream: int = 0,
    **kwargs,
) -> bool:
    """Launch ``producer`` unless the active scope already published ``key``.

    Returns ``True`` when the retained producer executes and ``False`` on an
    exact scope-local hit.  A miss invalidates the old publication before the
    producer runs; the new key is published only after a successful return.
    """

    key = (
        int(input_ptr),
        int(activation_ptr),
        int(rows),
        int(in_features),
        int(row_batch),
        int(stream),
    )
    state = _activation_pack_reuse_state.get()
    if state is not None and state.enabled and state.key == key:
        return False
    if state is not None:
        state.invalidate()
    producer(
        input_ptr,
        activation_ptr,
        rows,
        in_features,
        stream=stream,
        **kwargs,
    )
    if state is not None and state.enabled:
        state.key = key
    return True


__all__ = [
    "ActivationPackKey",
    "activation_pack_reuse_scope",
    "launch_scoped_activation_pack",
]
