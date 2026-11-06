"""Packed decode state ownership shared by resident and compatibility loops."""

from __future__ import annotations

from typing import Any, Sequence


def flush_packed_decode_owners(slots: Sequence[Any]) -> None:
    owners: list[Any] = []
    for slot in slots:
        owner = slot.packed_decode_owner
        if owner is not None and not any(existing is owner for existing in owners):
            owners.append(owner)
    for owner in owners:
        flush = getattr(owner, "flush_packed_decode_state", None)
        if callable(flush):
            flush()
    if owners:
        for slot in slots:
            if any(slot.packed_decode_owner is owner for owner in owners):
                slot.packed_decode_owner = None


def flush_packed_decode_owners_if_chunk_changed(chunk: Sequence[Any]) -> None:
    if not chunk:
        return
    owner = chunk[0].packed_decode_owner
    sessions = tuple(slot.session for slot in chunk)
    if owner is not None and all(slot.packed_decode_owner is owner for slot in chunk):
        owner_sessions = tuple(getattr(owner, "_packed_decode_sessions", ()))
        owner_dirty = bool(getattr(owner, "_packed_decode_state_dirty", False))
        if owner_dirty and owner_sessions == sessions:
            return
    flush_packed_decode_owners(chunk)
