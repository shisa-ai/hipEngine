from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import qwen35_paro_runner
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from scripts.qwen35_batch_shrinking_correctness import (
    _cancellation_order,
    _decode_counts_for_order,
    _inactive_snapshot_immutability,
    _parse_prompt_lengths,
    build_parser as build_shrinking_parser,
)
from scripts.qwen35_batch_sparse_slot_correctness import build_parser as build_sparse_parser


class _FakeSession:
    closed = False

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.hidden = "saved-hidden"
        self.next_hidden = "saved-next-hidden"
        self.runtime = SimpleNamespace(
            device_synchronize=lambda: self.calls.append(("synchronize",))
        )

    def _check_slot(self, slot: int) -> None:
        self.calls.append(("check_slot", slot))

    def _check_position(self, position: int) -> None:
        self.calls.append(("check_position", position))

    def _set_slot_token_embedding(self, token_id: int, *, slot: int) -> None:
        self.calls.append(("embedding", token_id, slot))

    def _set_slot_position(self, position: int, *, slot: int) -> None:
        self.calls.append(("position", position, slot))

    def _run_layers(self, *, position: int, slot: int, persist_aliases: bool, stream: int = 0):
        hidden = f"hidden-for-slot-{slot}"
        self.calls.append(("decode", position, slot, persist_aliases, stream))
        return hidden

    def _sample_from_hidden_for_slot(self, hidden, slot: int):
        self.calls.append(("sample", hidden, slot))
        return SimpleNamespace(token_id=1000 + slot)


def test_serial_bridge_uses_true_c1_decode_per_physical_slot() -> None:
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
        ("embedding", 11, 2),
        ("position", 512, 2),
        ("decode", 512, 2, False, 0),
        ("sample", "hidden-for-slot-2", 2),
        ("check_slot", 5),
        ("check_position", 513),
        ("embedding", 22, 5),
        ("position", 513, 5),
        ("decode", 513, 5, False, 0),
        ("sample", "hidden-for-slot-5", 5),
    ]
    assert session.hidden == "saved-hidden"
    assert session.next_hidden == "saved-next-hidden"


def test_serial_bridge_synchronizes_when_sampling_is_disabled() -> None:
    session = _FakeSession()

    results = Qwen35ParoResidentSession.step_batch_serial(
        session,
        [11],
        positions=[512],
        slots=[2],
        sample=False,
    )

    assert results == (None,)
    assert session.calls[-1] == ("synchronize",)


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


def test_sparse_slot_correctness_defaults_to_true_c1_fallback() -> None:
    assert build_sparse_parser().parse_args([]).decode_execution == "serial"


def test_shrinking_correctness_defaults_cover_c8_to_c1_with_holes() -> None:
    args = build_shrinking_parser().parse_args([])

    assert args.batch_size == 8
    assert args.decode_execution == "serial"
    assert args.prompt_lengths is None
    assert args.survivor_slot == 0
    assert args.eos_slot is None
    assert sorted(_cancellation_order(args.batch_size)) == list(range(1, 8))
    assert _cancellation_order(args.batch_size)[0] not in {0, 7}


def test_shrinking_correctness_can_leave_a_middle_slot_and_remove_both_edges() -> None:
    order = _cancellation_order(8, survivor_slot=4)

    assert sorted(order) == [0, 1, 2, 3, 5, 6, 7]
    assert 0 in order
    assert 7 in order
    assert _decode_counts_for_order(8, order, steps_per_width=1) == [
        order.index(slot) + 1 if slot != 4 else 8
        for slot in range(8)
    ]


def test_shrinking_correctness_parses_exact_ragged_prompt_vector() -> None:
    assert _parse_prompt_lengths("449,458,467", batch_size=3, fallback_length=512) == (
        449,
        458,
        467,
    )
    assert _parse_prompt_lengths(None, batch_size=3, fallback_length=512) == (512, 512, 512)

    with pytest.raises(ValueError, match="exactly batch_size"):
        _parse_prompt_lengths("449,458", batch_size=3, fallback_length=512)
    with pytest.raises(ValueError, match="positive"):
        _parse_prompt_lengths("449,0,467", batch_size=3, fallback_length=512)


def test_shrinking_correctness_reports_inactive_state_and_kv_mutation() -> None:
    retirement = {
        "live_count": 514,
        "linear": {"0": {"conv_sha256": "conv", "recurrent_sha256": "state"}},
        "full_kv": {"3": {"key_prefix_sha256": "key", "value_prefix_sha256": "value"}},
        "aggregate_sha256": "retirement",
    }
    unchanged = dict(retirement)
    changed = {
        **retirement,
        "linear": {"0": {"conv_sha256": "mutated", "recurrent_sha256": "state"}},
        "aggregate_sha256": "post-lifecycle",
    }

    result = _inactive_snapshot_immutability(
        {1: retirement, 2: retirement},
        {1: unchanged, 2: changed},
        retired_slots=(1, 2),
    )

    assert result == {
        "passed": False,
        "retired_slots": [1, 2],
        "mismatch_components_by_slot": {
            "1": [],
            "2": ["linear.0.conv_sha256"],
        },
        "retirement_aggregate_sha256_by_slot": {
            "1": "retirement",
            "2": "retirement",
        },
        "post_lifecycle_aggregate_sha256_by_slot": {
            "1": "retirement",
            "2": "post-lifecycle",
        },
    }
