"""Shared packed storage lifetime, independent of scheduling and session views."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def allocate_private_packed_backing(
    *, pages: int, page_bytes: int, pool: object | None, allocate: Callable,
) -> tuple[object, object | None]:
    """Acquire budget and storage atomically for a private packed KV backing.

    The returned token transfers to the packed storage owner, which releases it
    after freeing the returned backing's buffers. No-pool diagnostics own only
    the buffers. The allocator callback owns partial-buffer rollback.
    """
    token = None
    if pool is not None:
        token = pool.reserve_private_workspace(int(pages) * int(page_bytes))
    try:
        backing = allocate()
    except BaseException:
        if token is not None:
            pool.release_private_workspace(token)
        raise
    return backing, token


@dataclass
class StandalonePackedBackingLease:
    """Own standalone fallback storage with the same lifetime as arena leases."""

    backing: object
    pool: object | None
    token: object | None
    remaining_buffers: list

    @classmethod
    def acquire(cls, *, pages: int, page_bytes: int, pool: object | None, allocate: Callable):
        backing, token = allocate_private_packed_backing(
            pages=pages, page_bytes=page_bytes, pool=pool, allocate=allocate,
        )
        return cls(backing, pool, token, list(backing.buffers))

    def close(self, *, runtime: object, release: Callable) -> None:
        while self.remaining_buffers:
            release(self.remaining_buffers[-1], runtime=runtime)
            self.remaining_buffers.pop()
        if self.token is not None:
            self.pool.release_private_workspace(self.token)
            self.token = None


@dataclass
class PackedWorkspaceOwner:
    """One buffer/charge owner shared by the batch session and its slot views."""

    verify_state: object | None = None
    verify_scratch: object | None = None
    ar_attention_workspace: object | None = None
    verify_session_ids: tuple[int, ...] = ()
    verify_max_written_positions: tuple[int, ...] = ()
    decode_sessions: tuple = ()
    decode_last_layout: object | None = None
    decode_state_dirty: bool = False
    decode_session_ids: tuple[int, ...] = ()
    decode_positions: tuple[int, ...] = ()
    decode_direct_linear_state: bool = False

    def nbytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for workspace in (self.ar_attention_workspace, self.verify_scratch, self.verify_state):
            if workspace is None:
                continue
            for buffer in (
                *workspace.buffers,
                *(getattr(workspace, "full_attn_split_growth_buffers", ())
                  if workspace is self.verify_scratch else ()),
            ):
                if buffer is None or int(buffer.ptr) == 0 or int(buffer.ptr) in seen:
                    continue
                seen.add(int(buffer.ptr))
                total += int(buffer.nbytes)
        return total

    def close(self, *, runtime: object, release: Callable) -> None:
        """Quiesce device work, release buffers, then release backing pool charges.

        Graph teardown and canonical state scatter are caller preconditions.
        The session's guarded release API enforces them before entering here.
        """
        synchronize = getattr(runtime, "device_synchronize", None)
        if callable(synchronize):
            synchronize()
        if self.ar_attention_workspace is not None:
            for buffer in reversed(self.ar_attention_workspace.buffers):
                if buffer is not None:
                    release(buffer, runtime=runtime)
        self.ar_attention_workspace = None
        if self.verify_scratch is not None:
            for buffer in reversed(getattr(self.verify_scratch, "full_attn_split_growth_buffers", ())):
                if buffer is not None:
                    release(buffer, runtime=runtime)
            if hasattr(self.verify_scratch, "full_attn_split_growth_buffers"):
                object.__setattr__(self.verify_scratch, "full_attn_split_growth_buffers", ())
            for buffer in reversed(self.verify_scratch.buffers):
                if buffer is not None:
                    release(buffer, runtime=runtime)
        self.verify_scratch = None
        if self.verify_state is not None:
            lease = getattr(self.verify_state, "backing_lease", None)
            leased_buffers = (
                {id(buffer) for buffer in lease.backing.buffers}
                if lease is not None else set()
            )
            for buffer in reversed(self.verify_state.buffers):
                if buffer is not None and id(buffer) not in leased_buffers:
                    release(buffer, runtime=runtime)
            if lease is not None:
                lease.close(runtime=runtime, release=release)
            pool = getattr(self.verify_state, "private_workspace_pool", None)
            token = getattr(self.verify_state, "private_workspace_token", None)
            if token is not None:
                pool.release_private_workspace(token)
            charge_pool = getattr(self.verify_state, "resident_charge_pool", None)
            charge_token = getattr(self.verify_state, "resident_charge_token", None)
            if charge_token is not None:
                charge_pool.release_private_workspace(charge_token)
            growth_pool = getattr(self.verify_state, "growth_lease_pool", None)
            growth_key = getattr(self.verify_state, "growth_lease_key", None)
            if growth_key is not None:
                growth_pool.release_workspace(growth_key)
        self.verify_state = None
        self.verify_session_ids = ()
        self.verify_max_written_positions = ()
        self.decode_sessions = ()
        self.decode_last_layout = None
        self.decode_state_dirty = False
        self.decode_session_ids = ()
        self.decode_positions = ()
