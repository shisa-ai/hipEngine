"""Packed-decode reuse must not survive a slot's occupant changing.

Contracts under test (CPU, fake device):

- the import-reuse guard in ``_sync_packed_decode_initial_state`` permits
  reuse only when the recorded session tuple, the recorded session ids, and
  the recorded per-slot positions all still describe the sessions being
  bound -- so a cancelled or reassigned private slot is re-imported instead
  of attending over its previous occupant's KV;
- ``discard_packed_decode_state`` (the cancel path) clears the reuse tokens,
  so the round after a cancellation re-imports;
- reuse is batch-scoped rather than per-slot: a multi-slot round reuses only
  when its whole recorded membership still matches, so one changed member
  forces every slot to re-import.

These pin guards that already exist in ``qwen35_gguf_runner``. The sibling
``test_unit_gguf_packed_kv_import_skip.py`` exercises the same entry point but
never populates the recorded tuple, so every one of its cases takes the
``can_reuse = False`` branch and the reuse decision itself is unexercised
there.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

from tests.test_unit_gguf_packed_kv_import_skip import _sync_fixture
from tests.test_unit_gguf_packed_verify_layout import _rebind_test_state


def _record_reuse(owner, session, *, position: int = 256) -> None:
    """Make ``owner``'s recorded reuse tokens describe ``session`` exactly."""

    owner._packed_decode_sessions = (session,)
    owner._packed_decode_session_ids = (id(session),)
    owner._packed_decode_positions = (position,)
    owner._packed_decode_state_dirty = True


def test_packed_decode_reuse_skips_import_for_the_same_occupant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: a genuine continuation still reuses and imports nothing.

    Without this case the rejection tests below would pass even if reuse were
    impossible, so they would prove nothing about the guard.
    """

    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    _record_reuse(owner, session)

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == ()
    assert kv_copies == []
    assert fused_calls == []


def test_packed_decode_reuse_rejects_a_reassigned_private_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different occupant at the same position must not inherit the slot.

    The recorded ids still match the incoming session, so only the recorded
    session tuple can reject this. Reusing here would attend over the previous
    occupant's KV, which is the hazard the guard names.
    """

    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    previous_occupant = object.__new__(Qwen35GGUFResidentSession)
    owner._packed_decode_sessions = (previous_occupant,)
    owner._packed_decode_session_ids = (id(session),)
    owner._packed_decode_positions = (256,)

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == (0,)
    assert kv_copies == [(0, 0, 0, 256, False)]
    assert len(fused_calls) == 1


def test_packed_decode_reuse_rejects_a_mismatched_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded id that names another session must not be reused."""

    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    other = object.__new__(Qwen35GGUFResidentSession)
    owner._packed_decode_sessions = (session,)
    owner._packed_decode_session_ids = (id(other),)
    owner._packed_decode_positions = (256,)

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == (0,)
    assert kv_copies == [(0, 0, 0, 256, False)]


def test_packed_decode_reuse_rejects_a_moved_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same occupant, stale recorded position: the slot is re-imported."""

    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    _record_reuse(owner, session, position=128)

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == (0,)
    assert kv_copies == [(0, 0, 0, 256, False)]


def test_packed_decode_cancel_clears_reuse_for_the_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cancel path must force a re-import on the next round.

    ``discard_packed_decode_state`` is what a cancellation runs once every
    bound session is terminal. If it left the reuse tokens behind, the next
    round would bind a fresh occupant and skip its KV import.
    """

    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    _record_reuse(owner, session)

    assert owner.discard_packed_decode_state() is True
    assert owner._packed_decode_sessions == ()
    assert owner._packed_decode_session_ids == ()
    assert owner._packed_decode_positions == ()
    assert owner._packed_decode_state_dirty is False

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == (0,)
    assert kv_copies == [(0, 0, 0, 256, False)]


def test_packed_decode_reuse_is_batch_scoped_not_per_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: a multi-slot round reuses only when its whole membership matches.

    Two slots both continuing their own sessions re-import nothing. One
    changed member then forces a full re-import of every slot, because the
    recorded membership is compared as a whole before any per-slot position
    check runs. That conservatism is the point: an unrelated request joining
    the round cannot leave a slot reusing state it no longer owns.
    """

    owner, session, _layout, _packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    other = object.__new__(Qwen35GGUFResidentSession)
    other.__dict__.update(
        _position=1024,
        scratch=SimpleNamespace(
            layer_conv_states=(None, DeviceBuffer(ptr=0x800000, nbytes=64)),
            layer_recurrent_states=(None, DeviceBuffer(ptr=0x900000, nbytes=64)),
        ),
    )
    layout = gguf_runner._build_gguf_packed_verify_layout(
        (
            gguf_runner._GGUFPackedVerifySlotBlock(
                input_token_ids=tuple(range(8)), start_position=256
            ),
            gguf_runner._GGUFPackedVerifySlotBlock(
                input_token_ids=(7,), start_position=1024
            ),
        ),
        slot_capacity=2048,
    )
    packed_state = replace(
        _rebind_test_state(slot_count=2, blocks_per_slot=4),
        layer_conv_states=(None, DeviceBuffer(ptr=0xA00000, nbytes=64)),
        layer_recurrent_states=(None, DeviceBuffer(ptr=0xB00000, nbytes=64)),
    )
    owner._packed_decode_sessions = (session, other)
    owner._packed_decode_session_ids = (id(session), id(other))
    owner._packed_decode_positions = (256, 1024)

    imported = owner._sync_packed_decode_initial_state(
        (session, other), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == ()
    assert kv_copies == []
    assert fused_calls == []

    # Slot 1 is now occupied by a different session at the same position.
    replacement = object.__new__(Qwen35GGUFResidentSession)
    replacement.__dict__.update(
        _position=1024,
        scratch=SimpleNamespace(
            layer_conv_states=(None, DeviceBuffer(ptr=0xC00000, nbytes=64)),
            layer_recurrent_states=(None, DeviceBuffer(ptr=0xD00000, nbytes=64)),
        ),
    )

    imported = owner._sync_packed_decode_initial_state(
        (session, replacement), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == (0, 1)
    assert [entry[0] for entry in kv_copies] == [0, 1]
