"""A compact-DMS row is journaled by the store, not by the resident span planes.

Task #11's contract: commit applies the journal only for accepted candidates,
and rollback restores the full pre-cycle state. A DMS row's KV lives in per-head
extents with per-slot payload and scale planes, so ``_DMSStoreJournal`` drives
``DMSCompactBackend.begin_transaction``/``commit``/``rollback`` while the
resident journal keeps owning every non-KV tap.

The ordering these tests pin is the load-bearing part: a row is captured *after*
that row's append, so restoring the selected row's snapshot keeps exactly the
accepted prefix and discards the rejected rows' writes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hipengine.kvcache.dms import (
    DMSCompactBackend,
    DMSCodecQualification,
    DMSRetrofitConfig,
)
from hipengine.runtime import qwen35_gguf_mtp as mtp_module
from hipengine.runtime.qwen35_gguf_mtp import _DMSStoreJournal


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_from_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def _backend(*, codec: str = "int8_per_token_head") -> DMSCompactBackend:
    retrofit = DMSRetrofitConfig(
        artifact_fingerprint="fixture:dms-verifier-journal",
        model_family="qwen35",
        num_layers=1,
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=16,
        window_size=4,
        target_compression_ratio=2,
        alpha_scale=100.0,
        alpha_offset=5.0,
        borrowed_query_channel=15,
        corrected_mask=True,
        trained_checkpoint=True,
        evidence_source="unit fixture",
        source_path="tests/fixtures/dms_verifier_journal",
    )
    qualification = None
    if codec == "int8_per_token_head":
        qualification = DMSCodecQualification(
            codec=codec,
            artifact_fingerprint=retrofit.artifact_fingerprint,
            kl_divergence=0,
            top1_agreement=1,
            no_dense_shadow=True,
            evidence_source="unit fixture; not model qualification",
        )
    return DMSCompactBackend(
        retrofit=retrofit,
        codec=codec,
        slots_per_layer=128,
        max_request_rows=2,
        max_pack_rows=64,
        device_payloads=False,
        codec_qualification=qualification,
    )


def _admit(backend: DMSCompactBackend, request_id: int, tokens: int) -> None:
    request = SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tuple(range(tokens)),
        max_new_tokens=0,
    )
    claims = backend.estimate(
        request, None, {"kind": "admission", "tokens": tokens, "max_new_tokens": 0}
    )
    backend.reserve(claims)


def _rows(rng: np.random.Generator, rows: int, heads: int = 2, dim: int = 16) -> np.ndarray:
    return _bf16_from_bits(
        _bf16_bits(rng.normal(size=(rows, heads, dim)).astype(np.float32))
    )


def _pack_rows(rng: np.random.Generator, tokens: int, heads: int = 2, dim: int = 16) -> np.ndarray:
    return _bf16_from_bits(
        _bf16_bits(rng.normal(size=(tokens, 1, heads, dim)).astype(np.float32))
    )


def _store_state(backend: DMSCompactBackend, request_id: int) -> dict[str, Any]:
    state = backend.state_for_request(request_id)
    return {
        "live_counts": state.live_counts.copy(),
        "token_positions": state.token_positions.copy(),
        "evict_mask": state.evict_mask.copy(),
        "range_capacity": state.range_capacity.copy(),
        "extents": tuple(state.extents),
        "k_payload": {key: value.copy() for key, value in state.k_payload.items()},
        "v_payload": {key: value.copy() for key, value in state.v_payload.items()},
        "k_scales": {key: value.copy() for key, value in state.k_scales.items()},
        "v_scales": {key: value.copy() for key, value in state.v_scales.items()},
        "extent_pool": backend.extents.state_snapshot(),
        "ledger": backend.ledger.state_snapshot(),
    }


def _assert_store_state_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    np.testing.assert_array_equal(actual["live_counts"], expected["live_counts"])
    np.testing.assert_array_equal(actual["token_positions"], expected["token_positions"])
    np.testing.assert_array_equal(actual["evict_mask"], expected["evict_mask"])
    np.testing.assert_array_equal(actual["range_capacity"], expected["range_capacity"])
    assert actual["extents"] == expected["extents"]
    for name in ("k_payload", "v_payload", "k_scales", "v_scales"):
        assert set(actual[name]) == set(expected[name]), name
        for key in expected[name]:
            np.testing.assert_array_equal(actual[name][key], expected[name][key], err_msg=name)
    assert actual["extent_pool"]["owners"] == expected["extent_pool"]["owners"]
    assert actual["extent_pool"]["free"] == expected["extent_pool"]["free"]
    assert actual["ledger"]["owners"] == expected["ledger"]["owners"]
    assert actual["ledger"]["used"] == expected["ledger"]["used"]


class _RecordingResident:
    """Stand-in for _StateJournal: records delegation, owns no device buffers."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.producer_capture_initial_state = False
        self.initial_state_only = False
        self.row_hidden = SimpleNamespace(ptr=0x1234)
        self.closed = False

    def hidden_nbytes(self) -> int:
        return 64

    def state_row_capacity(self) -> int:
        return 4

    def hidden_rows_tensor(self, rows: int) -> Any:
        self.calls.append(("hidden_rows_tensor", rows))
        return SimpleNamespace(rows=rows)

    def capture_initial(self, *, stream: int = 0, force_consumer_state: bool = False) -> None:
        self.calls.append(("capture_initial", (stream, force_consumer_state)))

    def mark_initial_state_captured(self) -> None:
        self.calls.append(("mark_initial_state_captured", None))

    def capture_hidden_rows(self, hidden_rows: np.ndarray, *, stream: int = 0) -> None:
        self.calls.append(("capture_hidden_rows", stream))

    def capture_row(self, row: int, *, stream: int = 0) -> None:
        self.calls.append(("capture_row", int(row)))

    def restore_initial(self, *, stream: int = 0) -> None:
        self.calls.append(("restore_initial", stream))

    def restore_row(self, row: int, *, stream: int = 0) -> None:
        self.calls.append(("restore_row", int(row)))

    def restore_native_row(self, row: int, *, position: int, stream: int = 0) -> None:
        self.calls.append(("restore_native_row", (int(row), int(position))))

    def _copy_d2d(self, dst: int, src: int, nbytes: int, *, stream: int) -> None:
        self.calls.append(("_copy_d2d", (dst, src, nbytes)))

    def close(self) -> None:
        self.closed = True


def _journal(backend: DMSCompactBackend, *, request_id: int = 0) -> tuple[_DMSStoreJournal, _RecordingResident]:
    resident = _RecordingResident()
    journal = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend),
        resident=resident,
        max_rows=4,
    )
    journal.bind_request(request_id)
    return journal, resident


def _run_cycle(
    backend: DMSCompactBackend,
    journal: _DMSStoreJournal,
    *,
    request_id: int,
    rows: int,
    start_position: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Append `rows` candidate rows, capturing each row after its append."""

    after_each: list[dict[str, Any]] = []
    for row in range(rows):
        backend.append_decode(
            request_id,
            _rows(rng, 1),
            _rows(rng, 1),
            np.zeros((1, 2), dtype=bool),
            position=start_position + row,
        )
        journal.capture_row(row)
        after_each.append(_store_state(backend, request_id))
    return after_each


def _prepared_backend(*, packed: int = 9) -> DMSCompactBackend:
    """Admit five slots and pack them so every cycle row can evict one token.

    Every packed position is an eviction candidate, which is how a real DMS
    pack behaves: the window decides what survives. The five tokens inside the
    window survive the pack and stay marked, so as the cycle pushes them out
    each appended row retires exactly one and the extent never overflows.
    """

    backend = _backend()
    _admit(backend, 0, 5)
    rng = np.random.default_rng(4242)
    candidates = np.ones((packed, 1, 2), dtype=bool)
    backend.streaming_pack(
        0,
        _pack_rows(rng, packed),
        _pack_rows(rng, packed),
        candidates,
    )
    state = backend.state_for_request(0)
    assert int(state.live_counts[0, 0]) == 5, "the pack must fill the extent exactly"
    assert bool(state.evict_mask[0, 0, :5].all()), "the packed rows stay evictable"
    return backend


def test_full_rejection_restores_the_pre_cycle_store() -> None:
    backend = _prepared_backend()
    journal, resident = _journal(backend)
    before = _store_state(backend, 0)

    journal.capture_initial(stream=0)
    rng = np.random.default_rng(1)
    _run_cycle(backend, journal, request_id=0, rows=3, start_position=9, rng=rng)

    journal.restore_initial(stream=0)

    _assert_store_state_equal(_store_state(backend, 0), before)
    assert ("restore_initial", 0) in resident.calls, "the resident journal must be restored too"


def test_partial_acceptance_keeps_exactly_the_accepted_prefix() -> None:
    backend = _prepared_backend()
    journal, resident = _journal(backend)
    journal.capture_initial(stream=0)
    rng = np.random.default_rng(2)
    after_each = _run_cycle(backend, journal, request_id=0, rows=4, start_position=9, rng=rng)

    accepted = 2
    journal.restore_row(accepted, stream=0)

    _assert_store_state_equal(_store_state(backend, 0), after_each[accepted])
    assert ("restore_row", accepted) in resident.calls
    # The rejected rows really were written before the commit discarded them.
    assert not np.array_equal(
        after_each[-1]["token_positions"], after_each[accepted]["token_positions"]
    )


def test_a_rejected_cycle_leaves_the_row_usable_for_the_next_one() -> None:
    backend = _prepared_backend()
    journal, _resident = _journal(backend)
    before = _store_state(backend, 0)

    journal.capture_initial(stream=0)
    rng = np.random.default_rng(3)
    _run_cycle(backend, journal, request_id=0, rows=2, start_position=9, rng=rng)
    journal.restore_initial(stream=0)
    _assert_store_state_equal(_store_state(backend, 0), before)

    # The next cycle on the same row starts from the restored state and commits.
    journal.capture_initial(stream=0)
    after_each = _run_cycle(backend, journal, request_id=0, rows=2, start_position=9, rng=rng)
    journal.restore_row(1, stream=0)
    _assert_store_state_equal(_store_state(backend, 0), after_each[1])


def test_commit_without_a_row_snapshot_fails_loudly() -> None:
    backend = _prepared_backend()
    journal, _resident = _journal(backend)
    journal.capture_initial(stream=0)
    rng = np.random.default_rng(4)
    _run_cycle(backend, journal, request_id=0, rows=1, start_position=9, rng=rng)

    with pytest.raises(RuntimeError, match="no snapshot for row 3"):
        journal.restore_row(3)


def test_the_journal_is_bound_to_one_request() -> None:
    backend = _prepared_backend()
    journal, _resident = _journal(backend, request_id=0)

    with pytest.raises(RuntimeError, match="bound to one request"):
        journal.bind_request(1)

    unbound = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend),
        resident=_RecordingResident(),
        max_rows=4,
    )
    with pytest.raises(RuntimeError, match="no bound request"):
        unbound.capture_initial(stream=0)


def test_the_journal_requires_a_dms_backend() -> None:
    with pytest.raises(ValueError, match="DMS backend"):
        _DMSStoreJournal(
            target=SimpleNamespace(),
            resident=_RecordingResident(),
            max_rows=4,
        )


def test_the_journal_is_always_serial_capable_and_delegates_hidden_state() -> None:
    backend = _prepared_backend()
    resident = _RecordingResident()
    resident.producer_capture_initial_state = True
    journal = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend),
        resident=resident,
        max_rows=4,
    )

    # A DMS row cannot use the native target graph, so it always needs rows.
    assert journal.initial_state_only is False
    assert journal.producer_capture_initial_state is True
    assert journal.hidden_nbytes() == 64
    assert journal.state_row_capacity() == 4
    journal.hidden_rows_tensor(3)
    journal.mark_initial_state_captured()
    journal.capture_hidden_rows(np.zeros((2, 4), dtype=np.float32), stream=0)
    journal._copy_d2d(1, 2, 3, stream=0)
    assert journal.row_hidden.ptr == 0x1234
    assert ("hidden_rows_tensor", 3) in resident.calls
    assert ("_copy_d2d", (1, 2, 3)) in resident.calls

    journal.close()
    assert resident.closed is True


def test_cancellation_mid_cycle_restores_the_pre_cycle_store() -> None:
    """A cycle abandoned after some rows leaves nothing behind."""

    backend = _prepared_backend()
    journal, resident = _journal(backend)
    before = _store_state(backend, 0)

    journal.capture_initial(stream=0)
    rng = np.random.default_rng(5)
    # Two rows were verified when the request was cancelled.
    _run_cycle(backend, journal, request_id=0, rows=2, start_position=9, rng=rng)

    journal.restore_initial(stream=0)

    _assert_store_state_equal(_store_state(backend, 0), before)
    assert journal._initial is None
    assert journal._rows == {}
    assert ("restore_initial", 0) in resident.calls


def test_deadline_mid_cycle_is_exact_and_the_unwind_is_idempotent() -> None:
    """A deadline can fire with only part of the chain captured."""

    backend = _prepared_backend()
    journal, _resident = _journal(backend)
    before = _store_state(backend, 0)

    journal.capture_initial(stream=0)
    rng = np.random.default_rng(7)
    # The deadline fired after the first row was captured.
    _run_cycle(backend, journal, request_id=0, rows=1, start_position=9, rng=rng)
    journal.restore_initial(stream=0)
    _assert_store_state_equal(_store_state(backend, 0), before)

    # A second unwind (a deadline handler plus a shutdown, say) must not move it.
    journal.restore_initial(stream=0)
    _assert_store_state_equal(_store_state(backend, 0), before)


def test_a_cancelled_cycle_leaves_the_next_one_exact() -> None:
    """Cancellation, then a committed cycle on the same row."""

    backend = _prepared_backend()
    journal, _resident = _journal(backend)
    rng = np.random.default_rng(6)

    journal.capture_initial(stream=0)
    _run_cycle(backend, journal, request_id=0, rows=2, start_position=9, rng=rng)
    journal.restore_initial(stream=0)

    journal.capture_initial(stream=0)
    after_each = _run_cycle(backend, journal, request_id=0, rows=3, start_position=9, rng=rng)
    journal.restore_row(2, stream=0)
    _assert_store_state_equal(_store_state(backend, 0), after_each[2])


def test_shutdown_rolls_back_an_open_cycle_instead_of_dropping_it() -> None:
    """A disconnect or shutdown reaching close() must not strand the cycle."""

    backend = _prepared_backend()
    resident = _RecordingResident()
    journal = _DMSStoreJournal(
        target=SimpleNamespace(_dms_backend=backend),
        resident=resident,
        max_rows=4,
    )
    journal.bind_request(0)
    before = _store_state(backend, 0)

    journal.capture_initial(stream=0)
    rng = np.random.default_rng(8)
    _run_cycle(backend, journal, request_id=0, rows=2, start_position=9, rng=rng)

    journal.close()

    _assert_store_state_equal(_store_state(backend, 0), before)
    assert resident.closed is True
    journal.close()  # idempotent: a second teardown changes nothing
    _assert_store_state_equal(_store_state(backend, 0), before)


def test_the_adapter_owns_no_hip_allocations(monkeypatch) -> None:
    """Teardown cannot leak adapter-owned HIP allocations: it makes none.

    Everything the adapter holds is host-side store state, released by rollback
    or commit, and the resident journal keeps owning every device buffer. A full
    lifecycle therefore has to complete without touching the allocator at all,
    which is what makes "zero outstanding HIP allocations after teardown" hold
    for the adapter by construction rather than by measurement.
    """

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the compact DMS journal must not allocate device memory")

    monkeypatch.setattr(mtp_module, "malloc", _boom)
    monkeypatch.setattr(mtp_module, "free", _boom)

    backend = _prepared_backend()
    journal, _resident = _journal(backend)
    journal.capture_initial(stream=0)
    rng = np.random.default_rng(9)
    after_each = _run_cycle(backend, journal, request_id=0, rows=2, start_position=9, rng=rng)
    journal.restore_row(1, stream=0)
    _assert_store_state_equal(_store_state(backend, 0), after_each[1])
    journal.close()
    assert journal._initial is None
    assert journal._rows == {}
