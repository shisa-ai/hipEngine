"""Resident GGUF state storage, independent of layer execution and scheduling.

Canonical bindings borrow pool-owned KV and own root scratch only. Slot bindings
borrow root scratch, but have separate cursors and leases. Verifier and rollback
stores own their buffers; resizing publishes only a complete replacement. These
are storage backends, not a second speculative transaction coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind


@dataclass(frozen=True)
class StateMemory:
    """The allocation operations selected by the runtime adapter."""

    runtime: object
    allocate: Callable
    release: Callable

    def free_all(self, buffers) -> None:
        for buffer in reversed(tuple(buffers)):
            if buffer is not None:
                self.release(buffer, runtime=self.runtime)

    def allocate_all(self, sizes) -> tuple:
        """Return a complete owner set, or release every acquired owner."""
        buffers = []
        try:
            for size in sizes:
                buffers.append(
                    None if size is None else self.allocate(int(size), runtime=self.runtime)
                )
        except BaseException:
            self.free_all(buffers)
            raise
        return tuple(buffers)


@dataclass(frozen=True)
class CanonicalLinearState:
    """Borrowed per-layer canonical state; never frees the source buffers."""

    conv: tuple
    recurrent: tuple

    @property
    def pairs(self) -> tuple:
        return tuple(zip(self.conv, self.recurrent, strict=True))

    @property
    def buffers(self) -> tuple:
        return tuple(buffer for pair in self.pairs for buffer in pair if buffer is not None)

    def paired_sizes(self, rows: int) -> tuple:
        sizes = []
        for conv, recurrent in self.pairs:
            if (conv is None) != (recurrent is None):
                raise RuntimeError("linear-attention state families are not paired")
            sizes.extend(
                (None, None) if conv is None
                else (int(rows) * int(conv.nbytes), int(rows) * int(recurrent.nbytes))
            )
        return tuple(sizes)


@dataclass
class CanonicalResidentState:
    """One logical cursor/KV binding over an owning root or borrowed slot."""

    scratch: object | None = None
    scratch_owner: object | None = None
    layout: object | None = None
    position: int = 0
    kv_pool: object | None = None
    kv_allocation: object | None = None
    owns_scratch: bool = True

    def linear(self) -> CanonicalLinearState:
        if self.scratch is None:
            raise RuntimeError("GGUF resident session is closed")
        return CanonicalLinearState(
            tuple(self.scratch.layer_conv_states),
            tuple(self.scratch.layer_recurrent_states),
        )

    def close_scratch(self, memory: StateMemory) -> None:
        if self.owns_scratch and self.scratch_owner is not None:
            memory.free_all(self.scratch_owner.buffers)
        self.scratch_owner = None
        self.scratch = None
        self.layout = None


@dataclass
class VerifyBlockStorage:
    hidden_seed: object | None = None
    hidden_a: object | None = None
    hidden_b: object | None = None
    conv_out: object | None = None
    token_ids: object | None = None
    token_counter: object | None = None
    rows: int = 0
    populated: int = 0

    @property
    def buffers(self) -> tuple:
        return (self.hidden_seed, self.hidden_a, self.hidden_b, self.conv_out,
                self.token_ids, self.token_counter)

    def _publish(self, buffers, rows: int) -> None:
        (self.hidden_seed, self.hidden_a, self.hidden_b, self.conv_out,
         self.token_ids, self.token_counter) = buffers
        self.rows = int(rows)
        self.populated = 0

    def ensure(self, *, rows: int, hidden_size: int, qkv_width: int,
               memory: StateMemory) -> None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("verify block rows must be positive")
        if rows <= self.rows:
            return
        hidden_bytes = rows * int(hidden_size) * DType.FP32.itemsize
        new = memory.allocate_all((
            hidden_bytes, hidden_bytes, hidden_bytes,
            rows * int(qkv_width) * DType.FP32.itemsize if qkv_width > 0 else None,
            rows * DType.INT64.itemsize, DType.INT64.itemsize,
        ))
        previous = self.buffers
        self._publish(new, rows)
        memory.free_all(previous)

    def close(self, memory: StateMemory) -> None:
        previous = self.buffers
        self._publish((None,) * 6, 0)
        memory.free_all(previous)


@dataclass
class LinearStateStorage:
    """Paired provisional rows or producer-owned initial-state snapshots."""

    conv: tuple = ()
    recurrent: tuple = ()
    rows: int = 0
    users: int = 0

    def ensure(self, source: CanonicalLinearState, *, rows: int,
               memory: StateMemory, exact: bool = False) -> None:
        rows = int(rows)
        if rows <= 0:
            raise ValueError("verify linear-state rows must be positive")
        if not exact and rows <= self.rows:
            return
        sizes = source.paired_sizes(rows)
        old = tuple(buffer for pair in zip(self.conv, self.recurrent, strict=True)
                    for buffer in pair)
        if exact and len(old) == len(sizes) and all(
            (buffer is None if size is None else
             buffer is not None and int(buffer.nbytes) == size)
            for buffer, size in zip(old, sizes, strict=True)
        ):
            return
        if self.users:
            raise RuntimeError("cannot replace an active initial-state snapshot binding")
        new = memory.allocate_all(sizes)
        self.conv, self.recurrent = new[::2], new[1::2]
        self.rows = rows
        memory.free_all(old)

    def close(self, memory: StateMemory) -> None:
        if self.users:
            raise RuntimeError("cannot free an active initial-state snapshot binding")
        old = (*self.conv, *self.recurrent)
        self.conv = self.recurrent = ()
        self.rows = 0
        memory.free_all(old)


@dataclass
class LinearRollbackStorage:
    """Reusable device backups; transaction commit/rollback stays with callers."""

    buffers: tuple = ()

    def capture(self, source: CanonicalLinearState, memory: StateMemory) -> list:
        states = source.buffers
        if len(states) != len(self.buffers) or any(
            int(state.nbytes) != int(backup.nbytes)
            for state, backup in zip(states, self.buffers, strict=False)
        ):
            new = memory.allocate_all(state.nbytes for state in states)
            old, self.buffers = self.buffers, new
            memory.free_all(old)
        snapshot = list(zip(states, self.buffers, strict=True))
        for state, backup in snapshot:
            memory.runtime.memcpy(backup.ptr, state.ptr, state.nbytes,
                                  HipMemcpyKind.DEVICE_TO_DEVICE)
        return snapshot

    @staticmethod
    def restore(snapshot, runtime) -> None:
        for state, backup in snapshot:
            runtime.memcpy(state.ptr, backup.ptr, state.nbytes,
                           HipMemcpyKind.DEVICE_TO_DEVICE)

    def close(self, memory: StateMemory) -> None:
        old, self.buffers = self.buffers, ()
        memory.free_all(old)


@dataclass
class ResidentStateStorage:
    """Slot-local resources; packed workspace ownership is shared separately."""

    canonical: CanonicalResidentState = field(default_factory=CanonicalResidentState)
    block: VerifyBlockStorage = field(default_factory=VerifyBlockStorage)
    linear_rows: LinearStateStorage = field(default_factory=LinearStateStorage)
    initial: LinearStateStorage = field(default_factory=LinearStateStorage)
    rollback: LinearRollbackStorage = field(default_factory=LinearRollbackStorage)


def resident_state_field(section: str, name: str) -> property:
    """Compatibility accessors for existing session consumers, not duplicate state.

    Only this adapter touches the session. Storage implementations above receive
    explicit geometry, borrowed buffers and memory operations, never a session.
    """
    def store(session):
        return getattr(session._state_storage(), section)

    def get(session):
        return getattr(store(session), name)

    def set_value(session, value):
        setattr(store(session), name, value)

    return property(get, set_value)
