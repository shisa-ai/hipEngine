"""Slot-local packed prefill must not import whole-history KV per slab.

2026-09-10 follow-up to the packed slot-local INT8 prefill work: each slab's
``_sync_packed_decode_initial_state`` call imports every session's whole
prior KV history into packed storage, but slot-local attention reads
request-owned KV and the end-of-slab scatter already skips packed KV on that
route (``copy_kv=not slot_local_full_prefill`` at the scatter). The import is
quadratic dead work at fixed chunk size.

Contracts under test (CPU, fake device):

- ``_sync_packed_decode_initial_state`` with ``copy_kv=False`` performs no
  full-attention KV segment copies while still importing the Conv/GDN
  linear state (the slot-local slab consumes the packed linear state);
- the default (``copy_kv=True``) keeps the whole-history import for the
  non-slot-local fallback and the packed decode rounds;
- ``copy_linear_state=False`` keeps skipping only the linear side.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFResidentSession,
    _GGUFPackedVerifySlotBlock,
    _build_gguf_packed_verify_layout,
)

from tests.test_unit_gguf_packed_verify_layout import _rebind_test_state

_LAYER_TYPES = (FULL_ATTENTION, LINEAR_ATTENTION)


def _sync_fixture(monkeypatch: pytest.MonkeyPatch):
    kv_copies: list[tuple[int, int, int, int, bool]] = []
    fused_calls: list[list[tuple[int, int, int, int, int, int]]] = []

    def fake_kv_copy(
        self, session, packed_state, slot_index, layer_id, *,
        start_position, rows, packed_to_session, runtime, stream,
    ):
        del self, session, packed_state, runtime, stream
        kv_copies.append((slot_index, layer_id, start_position, rows, packed_to_session))

    def fake_fused(self, copies, *, runtime, stream):
        del self, runtime, stream
        fused_calls.append(list(copies))
        return True

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_copy_session_packed_kv_segments",
        fake_kv_copy,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_fused_linear_state_pair_copy",
        fake_fused,
    )

    owner = object.__new__(Qwen35GGUFResidentSession)
    owner.__dict__.update(
        runner=SimpleNamespace(
            weights=SimpleNamespace(
                config=SimpleNamespace(layer_types=_LAYER_TYPES)
            )
        ),
    )
    session = object.__new__(Qwen35GGUFResidentSession)
    session.__dict__.update(
        _position=256,
        scratch=SimpleNamespace(
            layer_conv_states=(None, DeviceBuffer(ptr=0x400000, nbytes=64)),
            layer_recurrent_states=(None, DeviceBuffer(ptr=0x500000, nbytes=64)),
        ),
    )
    layout = _build_gguf_packed_verify_layout(
        (_GGUFPackedVerifySlotBlock(
            input_token_ids=tuple(range(8)), start_position=256
        ),),
        slot_capacity=512,
    )
    packed_state = replace(
        _rebind_test_state(slot_count=1, blocks_per_slot=4),
        layer_conv_states=(None, DeviceBuffer(ptr=0x600000, nbytes=64)),
        layer_recurrent_states=(None, DeviceBuffer(ptr=0x700000, nbytes=64)),
    )
    return owner, session, layout, packed_state, kv_copies, fused_calls


def test_sync_with_copy_kv_false_skips_kv_and_keeps_linear_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
        copy_kv=False,
    )

    assert imported == (0,)
    assert kv_copies == []
    # The Conv/GDN import survives: the packed slab's linear-attention layers
    # consume the packed per-slot state.
    assert len(fused_calls) == 1
    conv_src, conv_dst, recurrent_src, recurrent_dst, conv_nbytes, recurrent_nbytes = (
        fused_calls[0][0]
    )
    assert conv_src == 0x400000 and conv_dst == 0x600000 and conv_nbytes == 64
    assert recurrent_src == 0x500000 and recurrent_dst == 0x700000
    assert recurrent_nbytes == 64


def test_sync_default_keeps_the_whole_history_kv_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )

    imported = owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
    )

    assert imported == (0,)
    # Non-slot-local fallback and packed decode rounds still import the
    # session's whole prior history (rows 0..256) for the full-attention
    # layer only.
    assert kv_copies == [(0, 0, 0, 256, False)]
    assert len(fused_calls) == 1


def test_sync_copy_linear_state_false_skips_only_the_linear_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )

    owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
        copy_linear_state=False,
    )

    assert kv_copies == [(0, 0, 0, 256, False)]
    assert fused_calls == []


def test_sync_at_position_zero_imports_nothing_either_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, session, layout, packed_state, kv_copies, fused_calls = _sync_fixture(
        monkeypatch
    )
    session._position = 0
    layout = _build_gguf_packed_verify_layout(
        (_GGUFPackedVerifySlotBlock(input_token_ids=tuple(range(8)), start_position=0),),
        slot_capacity=512,
    )

    owner._sync_packed_decode_initial_state(
        (session,), layout, packed_state,
        runtime=SimpleNamespace(), stream=0,
        copy_kv=False,
    )

    assert kv_copies == []
    assert len(fused_calls) == 1
