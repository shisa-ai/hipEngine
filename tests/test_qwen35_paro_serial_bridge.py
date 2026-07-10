from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.runtime import qwen35_paro_runner
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


class _FakeSession:
    closed = False

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def _check_slot(self, slot: int) -> None:
        self.calls.append(("check_slot", slot))

    def _check_position(self, position: int) -> None:
        self.calls.append(("check_position", position))

    def _set_batch_token_embeddings(self, token_ids, *, stream: int = 0):
        self.calls.append(("embedding", tuple(token_ids), stream))

    def _set_batch_positions(self, positions, *, slots, stream: int = 0) -> None:
        self.calls.append(("position", tuple(positions), tuple(slots), stream))

    def _run_layers_batch_decode(self, *, rows: int, positions, slots, stream: int = 0):
        hidden = f"hidden-for-slot-{slots[0]}"
        self.calls.append(("decode", rows, tuple(positions), tuple(slots), stream))
        return hidden

    def _sample_from_hidden_for_slot(self, hidden, slot: int):
        self.calls.append(("sample", hidden, slot))
        return SimpleNamespace(token_id=1000 + slot)


def test_serial_bridge_uses_exact_row_aware_decode_per_physical_slot() -> None:
    session = _FakeSession()

    results = Qwen35ParoResidentSession.step_batch_serial(
        session,
        [11, 22],
        positions=[512, 513],
        slots=[2, 5],
        sample=True,
    )

    assert [result.token_id for result in results] == [1002, 1005]
    assert session.calls == [
        ("check_slot", 2),
        ("check_position", 512),
        ("embedding", (11,), 0),
        ("position", (512,), (2,), 0),
        ("decode", 1, (512,), (2,), 0),
        ("sample", "hidden-for-slot-2", 2),
        ("check_slot", 5),
        ("check_position", 513),
        ("embedding", (22,), 0),
        ("position", (513,), (5,), 0),
        ("decode", 1, (513,), (5,), 0),
        ("sample", "hidden-for-slot-5", 5),
    ]


def test_batch_position_upload_addresses_sparse_physical_slots(monkeypatch) -> None:
    calls: list[tuple[int, int, int, int]] = []
    session = SimpleNamespace(
        max_batch_size=6,
        position_arr=np.zeros((6,), dtype=np.int64),
        context_arr=np.zeros((6,), dtype=np.int64),
        position_buf=SimpleNamespace(ptr=1000),
        context_buf=SimpleNamespace(ptr=2000),
        libraries={"runtime_state": 7},
        runtime=object(),
        _check_position=lambda position: None,
        _check_slot=lambda slot: None,
    )
    monkeypatch.setattr(
        qwen35_paro_runner,
        "set_decode_position_i64",
        lambda position_ptr, context_ptr, position, *, stream, library, runtime: calls.append(
            (position_ptr, context_ptr, position, stream)
        ),
    )

    Qwen35ParoResidentSession._set_batch_positions(
        session,
        [512, 513],
        slots=[2, 5],
        stream=3,
    )

    assert session.position_arr.tolist() == [0, 0, 512, 0, 0, 513]
    assert session.context_arr.tolist() == [0, 0, 513, 0, 0, 514]
    assert calls == [
        (1000 + 2 * 8, 2000 + 2 * 8, 512, 3),
        (1000 + 5 * 8, 2000 + 5 * 8, 513, 3),
    ]
