"""Cancellation safety for a packed decode group.

``_flush_ar_packed_decode_owners_if_chunk_changed`` is the mechanism that keeps
a cancellation exact once a physical group has formed.  A slot that leaves the
group (cancelled, completed, or reclaimed) changes the chunk's session
composition; the group's accumulated packed state must then be written back
before the group re-forms, otherwise the newcomers that reuse the freed lanes
inherit a stale slot view.

The rule has a fast path and a flush path, and the distinction is what makes
the promoted direct-INT8 C>1 route safe rather than merely usually-correct:

* owner uniform across the chunk, owner dirty, and the *same* session objects
  in the *same* order -> keep the owner and keep decoding (no flush);
* anything else -> flush the owner once, then clear it from every slot it
  owned.

These are CPU-only invariants over that branch structure.  They deliberately do
not stub the method, because every existing call site does, which is why the
branch was previously untested.
"""

from __future__ import annotations

from types import SimpleNamespace

from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator


class _FakeOwner:
    """A packed-decode owner that records how many times it was flushed."""

    def __init__(self, *, sessions: tuple[object, ...] = (), dirty: bool = True) -> None:
        self._packed_decode_sessions = tuple(sessions)
        self._packed_decode_state_dirty = bool(dirty)
        self.flush_count = 0

    def flush_packed_decode_state(self) -> None:
        self.flush_count += 1


def _generator() -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator.__new__(Qwen35GGUFBringupGenerator)


def _slot(session: object, owner: object | None) -> SimpleNamespace:
    return SimpleNamespace(session=session, packed_decode_owner=owner)


def test_empty_chunk_is_a_noop() -> None:
    _generator()._flush_ar_packed_decode_owners_if_chunk_changed([])


def test_steady_group_keeps_its_owner_without_flushing() -> None:
    """The unchanged-composition fast path must not flush on every step."""

    sessions = (object(), object(), object())
    owner = _FakeOwner(sessions=sessions, dirty=True)
    chunk = [_slot(session, owner) for session in sessions]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 0
    assert all(slot.packed_decode_owner is owner for slot in chunk)


def test_clean_owner_is_flushed_even_when_the_composition_is_unchanged() -> None:
    """A non-dirty owner has nothing to reuse, so it must be released."""

    sessions = (object(), object())
    owner = _FakeOwner(sessions=sessions, dirty=False)
    chunk = [_slot(session, owner) for session in sessions]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 1
    assert all(slot.packed_decode_owner is None for slot in chunk)


def test_cancelled_slot_flushes_the_owner_before_the_group_re_forms() -> None:
    """A shrinking chunk is the cancellation case and must flush."""

    sessions = (object(), object(), object())
    owner = _FakeOwner(sessions=sessions, dirty=True)
    survivors = sessions[:2]
    chunk = [_slot(session, owner) for session in survivors]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 1
    assert all(slot.packed_decode_owner is None for slot in chunk)


def test_newcomer_reusing_a_freed_lane_flushes_the_owner() -> None:
    """Same width, different session objects: the newcomer must not inherit.

    This is the refill half of the cancellation contract.  The chunk has the
    same length as the owner's recorded composition, so a length comparison
    would wrongly take the fast path; identity comparison must catch it.
    """

    old_sessions = (object(), object())
    owner = _FakeOwner(sessions=old_sessions, dirty=True)
    new_sessions = (old_sessions[0], object())
    chunk = [_slot(session, owner) for session in new_sessions]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 1
    assert all(slot.packed_decode_owner is None for slot in chunk)


def test_reordered_sessions_flush_the_owner() -> None:
    """Row order is part of the packed layout, so a reorder is a change."""

    sessions = (object(), object())
    owner = _FakeOwner(sessions=sessions, dirty=True)
    chunk = [_slot(session, owner) for session in reversed(sessions)]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 1


def test_mixed_owners_are_each_flushed_exactly_once() -> None:
    sessions = (object(), object(), object(), object())
    first = _FakeOwner(sessions=sessions[:2], dirty=True)
    second = _FakeOwner(sessions=sessions[2:], dirty=True)
    chunk = [
        _slot(sessions[0], first),
        _slot(sessions[1], first),
        _slot(sessions[2], second),
        _slot(sessions[3], second),
    ]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert first.flush_count == 1
    assert second.flush_count == 1
    assert all(slot.packed_decode_owner is None for slot in chunk)


def test_ownerless_slots_are_left_alone_and_do_not_block_the_flush() -> None:
    sessions = (object(), object())
    owner = _FakeOwner(sessions=sessions, dirty=True)
    chunk = [_slot(sessions[0], owner), _slot(sessions[1], None)]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 1
    assert chunk[0].packed_decode_owner is None
    assert chunk[1].packed_decode_owner is None


def test_owner_without_a_flush_hook_is_still_detached() -> None:
    """A partial owner object must not strand the slot reference."""

    sessions = (object(),)
    owner = SimpleNamespace(_packed_decode_sessions=sessions, _packed_decode_state_dirty=False)
    chunk = [_slot(sessions[0], owner)]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert chunk[0].packed_decode_owner is None


def test_owner_missing_the_dirty_flag_is_not_treated_as_steady_state() -> None:
    """Absent bookkeeping defaults to flush, never to silent reuse."""

    sessions = (object(),)
    owner = _FakeOwner(sessions=sessions)
    del owner._packed_decode_state_dirty
    chunk = [_slot(sessions[0], owner)]

    _generator()._flush_ar_packed_decode_owners_if_chunk_changed(chunk)

    assert owner.flush_count == 1
    assert chunk[0].packed_decode_owner is None
