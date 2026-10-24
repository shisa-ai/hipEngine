"""PARO MTP bulk historical-prompt K/V integration on the packed public path.

These tests pin the host-visible contracts only: which prompt token, hidden row,
and position each historical append consumes, how the target's hidden chunks are
split and published, when the single full final prediction runs, and when the
bounded workspace is released. Device arithmetic is unchanged by design and is
covered by the serial-vs-bulk device oracle, not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro as paro_generation_module
import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
import hipengine.runtime.qwen35_paro_runner as paro_runner_module
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.generation.batch_scheduler import CompactPromptSlab
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import (
    PROMPT_PRIME_CHUNK_ROWS,
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
    _ParoPromptPrimeSink,
)
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from hipengine.speculative.prompt_prime import PromptPrimeSink

_HIDDEN = 8
_BF16_ROW_BYTES = _HIDDEN * DType.BF16.itemsize
_FP16_ROW_BYTES = _HIDDEN * DType.FP16.itemsize


class _FakeAllocator:
    """Deterministic device allocator double for the prompt-priming workspace."""

    def __init__(self, base_ptr: int = 0x500000) -> None:
        self.base_ptr = int(base_ptr)
        self.next_ptr = self.base_ptr
        self.allocations: list[DeviceBuffer] = []
        self.freed: list[DeviceBuffer] = []

    def malloc(self, nbytes, *, runtime=None):
        buffer = DeviceBuffer(ptr=self.next_ptr, nbytes=int(nbytes))
        self.next_ptr += max(int(nbytes), 0x100)
        self.allocations.append(buffer)
        return buffer

    def free(self, buffer, *, runtime=None):
        self.freed.append(buffer)


class _BulkProposer:
    """Proposer double that records the bulk and full-advance seams."""

    hidden = _HIDDEN
    closed = False
    cache_len = 0
    position = -1

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.bulk_calls: list[tuple[tuple[int, int, int], ...]] = []
        self.bulk_chunk_rows: list[int | None] = []
        self.advances: list[dict[str, object]] = []
        self.fail_on_call = fail_on_call

    def append_prompt_kv_bulk(self, rows, *, chunk_rows=None, stream=0):
        if self.fail_on_call is not None and len(self.bulk_calls) == self.fail_on_call:
            raise RuntimeError("injected bulk prompt append failure")
        pending = tuple(rows)
        self.bulk_calls.append(pending)
        self.bulk_chunk_rows.append(chunk_rows)
        self.cache_len += len(pending)
        self.position = int(pending[-1][2])
        return len(pending)

    def advance_with_target_hidden(self, **kwargs):
        self.advances.append(kwargs)
        return SimpleNamespace(token=0)

    def reset(self) -> None:
        self.cache_len = 0
        self.position = -1

    def close(self) -> None:
        self.closed = True


class _SessionDouble:
    """Minimal target session surface the prompt sink touches."""

    def __init__(self, *, hidden: int = _HIDDEN) -> None:
        self.config = SimpleNamespace(hidden_size=hidden, rms_norm_eps=1e-6)
        self.runtime = None
        self.device = Device("hip", 0)
        self.normalize_calls: list[dict[str, object]] = []

    def _write_final_normalized_hidden_bf16_rows(
        self,
        hidden,
        *,
        rows,
        destination_ptr,
        scratch_ptr,
        stream=0,
    ):
        self.normalize_calls.append(
            {
                "hidden_ptr": int(hidden.ptr),
                "shape": tuple(hidden.shape),
                "rows": int(rows),
                "destination_ptr": int(destination_ptr),
                "scratch_ptr": int(scratch_ptr),
                "stream": int(stream),
            }
        )


def _row(prompt_ids, **overrides) -> SimpleNamespace:
    fields = {
        "request_id": 7,
        "prompt_ids": tuple(int(token) for token in prompt_ids),
        "mtp2_prompt_prime_ms": 0.0,
        "mtp2_prompt_hidden_buffer": None,
        "mtp2_bulk_prompt_chunks": 0,
        "mtp2_bulk_prompt_rows": 0,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _adapter(*, prompt_ids, proposer=None, session=None):
    row = _row(prompt_ids)
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100"),
        _row=lambda request_id: row,
        _session=session if session is not None else _SessionDouble(),
    )
    adapter = Qwen35ParoMTP2Adapter(owner)
    state = _ParoMTP2RequestState(7, proposer if proposer is not None else _BulkProposer())
    adapter._states[7] = state
    return adapter, state, row


@pytest.fixture()
def allocator(monkeypatch):
    fake = _FakeAllocator()
    monkeypatch.setattr(paro_mtp2_module, "malloc", fake.malloc)
    monkeypatch.setattr(paro_mtp2_module, "free", fake.free)
    return fake


def _sink(adapter, state, row, *, chunk_rows=PROMPT_PRIME_CHUNK_ROWS):
    sink = _ParoPromptPrimeSink(
        adapter=adapter,
        state=state,
        request_id=7,
        prompt_ids=row.prompt_ids,
        session=adapter.owner._session,
        chunk_rows=chunk_rows,
    )
    state.prompt_sink = sink
    return sink


def test_paro_prompt_sink_satisfies_the_shared_prompt_prime_contract(allocator) -> None:
    adapter, state, row = _adapter(prompt_ids=range(5))
    sink = _sink(adapter, state, row)

    assert isinstance(sink, PromptPrimeSink)
    assert sink.request_id == 7
    assert sink.hidden_size == _HIDDEN
    assert sink.total_rows == 5
    assert sink.published_rows == 0


def test_paro_bulk_prompt_chunk_aligns_tokens_hidden_rows_and_positions(allocator) -> None:
    prompt_ids = tuple(range(100, 140))
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=prompt_ids, proposer=proposer)
    sink = _sink(adapter, state, row)
    slab_ptr = allocator.allocations[0].ptr

    sink.consume(
        request_id=7,
        chunk_start=0,
        hidden_ptr=0x900000,
        rows=40,
        stream=0,
    )

    # Historical rows only: the last prompt row waits for the target root.
    assert [len(call) for call in proposer.bulk_calls] == [
        PROMPT_PRIME_CHUNK_ROWS,
        PROMPT_PRIME_CHUNK_ROWS,
        40 - 2 * PROMPT_PRIME_CHUNK_ROWS - 1,
    ]
    rows = tuple(row_spec for call in proposer.bulk_calls for row_spec in call)
    assert [token for token, _ptr, _position in rows] == list(prompt_ids[1:40])
    assert [position for _token, _ptr, position in rows] == list(range(1, 40))
    # Each slice refills the same bounded slab from row zero, so the historical
    # run of one slice is contiguous from the slab base.
    expected_ptrs = [
        slab_ptr + offset * _BF16_ROW_BYTES
        for count in (PROMPT_PRIME_CHUNK_ROWS, PROMPT_PRIME_CHUNK_ROWS, 40 - 2 * PROMPT_PRIME_CHUNK_ROWS - 1)
        for offset in range(count)
    ]
    assert [ptr for _token, ptr, _position in rows] == expected_ptrs
    assert proposer.bulk_chunk_rows == [None, None, None]
    assert not proposer.advances
    assert sink.published_rows == 39
    assert state.prompt_rows_consumed == 39
    assert (row.mtp2_bulk_prompt_chunks, row.mtp2_bulk_prompt_rows) == (3, 39)
    # One row-capable normalize/cast launch set per slice, over the source slab.
    assert [call["rows"] for call in adapter.owner._session.normalize_calls] == [
        PROMPT_PRIME_CHUNK_ROWS,
        PROMPT_PRIME_CHUNK_ROWS,
        40 - 2 * PROMPT_PRIME_CHUNK_ROWS,
    ]
    assert [call["hidden_ptr"] for call in adapter.owner._session.normalize_calls] == [
        0x900000,
        0x900000 + PROMPT_PRIME_CHUNK_ROWS * _FP16_ROW_BYTES,
        0x900000 + 2 * PROMPT_PRIME_CHUNK_ROWS * _FP16_ROW_BYTES,
    ]
    assert all(call["destination_ptr"] == slab_ptr for call in adapter.owner._session.normalize_calls)

    sink.finish(request_id=7, total_rows=40, stream=0)
    sink.complete(seed_token=99)

    assert len(proposer.advances) == 1
    advance = proposer.advances[0]
    assert advance["input_token"] == 99
    assert advance["position"] == 40
    assert advance["need_result"] is True
    assert advance["read_token_id"] is False
    assert advance["read_expert_topk"] is False
    assert advance["read_lm_head_value"] is False
    assert advance["target_hidden_ptr"] == slab_ptr + 7 * _BF16_ROW_BYTES
    assert sink.published_rows == 40
    assert state.prompt_rows_consumed == 40
    assert sink.ledger.complete
    assert sink.ledger.finished


def test_paro_bulk_prompt_chunks_split_across_scheduler_boundaries(allocator) -> None:
    prompt_ids = tuple(range(200, 220))
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=prompt_ids, proposer=proposer)
    sink = _sink(adapter, state, row)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=10, stream=0)
    assert sink.published_rows == 10
    assert not proposer.advances

    sink.consume(request_id=7, chunk_start=10, hidden_ptr=0x900000, rows=10, stream=0)
    rows = tuple(row_spec for call in proposer.bulk_calls for row_spec in call)
    # The row that ended the first scheduler chunk consumed token 110, not 111.
    assert [token for token, _ptr, _position in rows] == list(prompt_ids[1:20])
    assert [position for _token, _ptr, position in rows] == list(range(1, 20))
    assert sink.published_rows == 19
    assert not proposer.advances

    sink.finish(request_id=7, total_rows=20, stream=0)
    sink.complete(seed_token=7)
    assert len(proposer.advances) == 1
    assert proposer.advances[0]["position"] == 20
    assert sink.ledger.complete


def test_paro_bulk_prompt_chunk_must_stay_contiguous_across_chunks(allocator) -> None:
    adapter, state, row = _adapter(prompt_ids=range(20))
    sink = _sink(adapter, state, row)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=10, stream=0)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        sink.consume(request_id=7, chunk_start=12, hidden_ptr=0x900000, rows=8, stream=0)
    assert sink.published_rows == 10
    assert not state.proposer.advances


def test_paro_bulk_prompt_sink_refuses_a_peer_request(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(20), proposer=proposer)
    sink = _sink(adapter, state, row)

    with pytest.raises(RuntimeError, match="owner mismatch"):
        sink.consume(request_id=8, chunk_start=0, hidden_ptr=0x900000, rows=20, stream=0)
    assert sink.published_rows == 0
    assert not proposer.bulk_calls
    with pytest.raises(RuntimeError, match="owner mismatch"):
        sink.finish(request_id=8, total_rows=20, stream=0)


def test_paro_bulk_prompt_one_token_prompt_runs_one_full_prediction(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=(42,), proposer=proposer)
    sink = _sink(adapter, state, row)
    slab_ptr = allocator.allocations[0].ptr

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=1, stream=0)
    assert not proposer.bulk_calls
    assert sink.published_rows == 0
    assert [call["rows"] for call in adapter.owner._session.normalize_calls] == [1]

    sink.finish(request_id=7, total_rows=1, stream=0)
    sink.complete(seed_token=42)
    sink.complete(seed_token=42)

    assert len(proposer.advances) == 1
    assert proposer.advances[0]["input_token"] == 42
    assert proposer.advances[0]["position"] == 1
    assert proposer.advances[0]["target_hidden_ptr"] == slab_ptr
    assert sink.published_rows == 1


def test_paro_bulk_prompt_publishes_rows_only_after_the_device_work_succeeds(allocator) -> None:
    proposer = _BulkProposer(fail_on_call=1)
    adapter, state, row = _adapter(prompt_ids=range(40), proposer=proposer)
    sink = _sink(adapter, state, row)

    with pytest.raises(RuntimeError, match="injected bulk prompt append failure"):
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=40, stream=0)

    # The slice that landed is published; the failed slice is not.
    assert len(proposer.bulk_calls) == 1
    assert sink.published_rows == PROMPT_PRIME_CHUNK_ROWS
    assert state.prompt_rows_consumed == PROMPT_PRIME_CHUNK_ROWS
    assert sink.ledger.finished is False
    assert not proposer.advances
    with pytest.raises(RuntimeError, match="failed"):
        sink.consume(request_id=7, chunk_start=PROMPT_PRIME_CHUNK_ROWS, hidden_ptr=0x900000, rows=24, stream=0)
    sink.complete(seed_token=1)
    assert not proposer.advances


def test_paro_bulk_prompt_final_row_requires_the_target_root(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(4), proposer=proposer)
    sink = _sink(adapter, state, row)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=4, stream=0)
    sink.finish(request_id=7, total_rows=4, stream=0)
    with pytest.raises(RuntimeError, match="requires the target root token"):
        sink.complete(seed_token=None)
    assert not proposer.advances
    assert sink.published_rows == 3

    sink.complete(seed_token=11)
    assert len(proposer.advances) == 1
    assert sink.published_rows == 4


def test_paro_bulk_prompt_completion_requires_the_target_fanout_to_finish(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(4), proposer=proposer)
    sink = _sink(adapter, state, row)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=4, stream=0)
    with pytest.raises(RuntimeError, match="before the target finished the prompt"):
        sink.complete(seed_token=5)
    assert not proposer.advances
    assert sink.published_rows == 3

    sink.finish(request_id=7, total_rows=4, stream=0)
    sink.complete(seed_token=5)
    assert len(proposer.advances) == 1


def test_paro_bulk_prompt_completion_is_a_noop_before_the_prompt_ends(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(30), proposer=proposer)
    sink = _sink(adapter, state, row)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=16, stream=0)
    adapter.finish_prompt_capture(7, seed_token=None)
    adapter.finish_prompt_capture(7, seed_token=5)
    assert not proposer.advances
    assert sink.published_rows == 16


def test_paro_bulk_prompt_workspace_is_bounded_and_released(allocator) -> None:
    short_adapter, short_state, short_row = _adapter(prompt_ids=range(3))
    long_adapter, long_state, long_row = _adapter(prompt_ids=range(5000))
    short_sink = _sink(short_adapter, short_state, short_row)
    long_sink = _sink(long_adapter, long_state, long_row)

    assert short_sink.chunk_rows == 3
    assert long_sink.chunk_rows == PROMPT_PRIME_CHUNK_ROWS
    short_slab, short_scratch = allocator.allocations[0:2]
    long_slab, long_scratch = allocator.allocations[2:4]
    assert short_slab.nbytes == 3 * _BF16_ROW_BYTES
    assert short_scratch.nbytes == 3 * _FP16_ROW_BYTES
    # The workspace tracks the bounded chunk, never the prompt length.
    assert long_slab.nbytes == PROMPT_PRIME_CHUNK_ROWS * _BF16_ROW_BYTES
    assert long_scratch.nbytes == PROMPT_PRIME_CHUNK_ROWS * _FP16_ROW_BYTES

    long_adapter.release_prompt_capture(7)
    assert long_sink.closed
    assert long_state.prompt_sink is None
    assert allocator.freed == [long_scratch, long_slab]

    # A request reclaimed before priming finished still frees its workspace.
    short_adapter._drop_request(7, disable=False)
    assert short_sink.closed
    assert allocator.freed[-2:] == [short_scratch, short_slab]
    assert 7 not in short_adapter._states

    # Cancellation takes the same reclaim route.
    cancelled_adapter, cancelled_state, cancelled_row = _adapter(prompt_ids=range(40))
    cancelled_sink = _sink(cancelled_adapter, cancelled_state, cancelled_row)
    cancelled_slab, cancelled_scratch = allocator.allocations[4:6]
    cancelled_adapter.close_requests((7,))
    assert cancelled_sink.closed
    assert allocator.freed[-2:] == [cancelled_scratch, cancelled_slab]


def test_paro_prefill_observation_releases_the_prompt_workspace(allocator) -> None:
    proposer = _BulkProposer()
    released: list[object] = []
    adapter, state, row = _adapter(prompt_ids=range(4), proposer=proposer)
    row.request = SimpleNamespace(max_tokens=4)
    adapter.owner._session.prepare_specdec2_verify_scratch = (
        lambda *, rows, chain_attn_mode, max_context_tokens: None
    )
    adapter.owner._release_mtp2_prompt_capture = released.append
    sink = _sink(adapter, state, row)
    slab, scratch = allocator.allocations[0:2]

    with pytest.raises(RuntimeError, match="priming is incomplete"):
        adapter.observe_prefill_result(7)
    assert not sink.closed

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=4, stream=0)
    sink.finish(request_id=7, total_rows=4, stream=0)
    sink.complete(seed_token=9)

    adapter.observe_prefill_result(7)

    assert sink.closed
    assert state.prompt_sink is None
    assert allocator.freed == [scratch, slab]
    assert released == [row]


def test_paro_bulk_prompt_route_counters_report_device_work(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(40), proposer=proposer)
    sink = _sink(adapter, state, row)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x900000, rows=40, stream=0)
    sink.finish(request_id=7, total_rows=40, stream=0)
    sink.complete(seed_token=3)

    snapshot = adapter.observability_snapshot()
    assert snapshot["bulk_prompt_chunks"] == 3
    assert snapshot["bulk_prompt_rows"] == 39
    assert (row.mtp2_bulk_prompt_chunks, row.mtp2_bulk_prompt_rows) == (3, 39)


def test_paro_prompt_capture_sink_refuses_an_unstaged_request(allocator) -> None:
    row = _row(range(4))
    owner = SimpleNamespace(
        generator=SimpleNamespace(backend="hip_gfx1100"),
        _row=lambda request_id: row,
        _session=_SessionDouble(),
    )
    adapter = Qwen35ParoMTP2Adapter(owner)

    with pytest.raises(RuntimeError, match="no staged request state"):
        adapter.prompt_capture_sink(7)
    assert allocator.allocations == []


def test_paro_prompt_capture_sink_is_reused_across_scheduler_chunks(allocator) -> None:
    adapter, state, row = _adapter(prompt_ids=range(40))

    first = adapter.prompt_capture_sink(7)
    second = adapter.prompt_capture_sink(7)

    assert first is second
    assert state.prompt_sink is first
    assert len(allocator.allocations) == 2


def test_paro_bulk_prompt_sink_resumes_from_rows_a_row_seam_published(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(20), proposer=proposer)
    state.prompt_rows_consumed = 6

    sink = _sink(adapter, state, row)

    assert sink.published_rows == 6
    sink.consume(request_id=7, chunk_start=6, hidden_ptr=0x900000, rows=14, stream=0)
    rows = tuple(row_spec for call in proposer.bulk_calls for row_spec in call)
    assert [token for token, _ptr, _position in rows] == list(row.prompt_ids[7:20])
    assert sink.published_rows == 19


def test_paro_row_seam_keeps_one_ledger_when_a_chunk_sink_exists(allocator) -> None:
    proposer = _BulkProposer()
    adapter, state, row = _adapter(prompt_ids=range(4), proposer=proposer)
    proposer.append_prompt_kv = lambda **kwargs: proposer.bulk_calls.append(tuple())
    sink = _sink(adapter, state, row)
    state.prompt_sink = sink

    adapter.consume_prompt_row(7, prompt_index=0, target_hidden_ptr=0x910000, seed_token=None)
    adapter.consume_prompt_row(7, prompt_index=1, target_hidden_ptr=0x910000, seed_token=None)

    assert sink.published_rows == 2
    assert state.prompt_rows_consumed == 2
    assert sink.ledger.finished is False

    adapter.consume_prompt_row(7, prompt_index=2, target_hidden_ptr=0x910000, seed_token=None)
    adapter.consume_prompt_row(7, prompt_index=3, target_hidden_ptr=0x910000, seed_token=88)
    assert sink.published_rows == 4
    assert sink.ledger.complete and sink.ledger.finished
    assert len(proposer.advances) == 1
    assert proposer.advances[0]["input_token"] == 88


# -- public packed path -------------------------------------------------------


class _ChunkCaptureSession:
    """Target session double for the packed prefill seam."""

    block_size = 256
    hidden_nbytes = _HIDDEN * DType.FP16.itemsize
    config = SimpleNamespace(hidden_size=_HIDDEN)
    runtime = object()

    def __init__(self, calls, *, raise_not_implemented: bool = False) -> None:
        self.calls = calls
        self.raise_not_implemented = raise_not_implemented

    def prefill_native_packed(
        self,
        slab,
        *,
        sample=True,
        final_hidden_row_sink=None,
        final_hidden_chunk_sink=None,
    ):
        self.calls.append(
            (
                "packed",
                slab.token_rows,
                sample,
                final_hidden_row_sink,
                final_hidden_chunk_sink,
            )
        )
        if self.raise_not_implemented:
            raise NotImplementedError("packed prefill unavailable")
        assert final_hidden_chunk_sink is not None
        assert final_hidden_row_sink is None
        final_hidden_chunk_sink.consume(
            request_id=7,
            chunk_start=0,
            hidden_ptr=0x900000,
            rows=2,
            stream=0,
        )
        final_hidden_chunk_sink.finish(request_id=7, total_rows=2, stream=0)
        return (SimpleNamespace(token_id=99),)

    def step_with_hidden_taps(self, token_id, **kwargs):
        self.calls.append(("serial", int(token_id), kwargs["position"]))
        return SimpleNamespace(token_id=1234 + int(token_id))


def _runner(session, adapter, *, row):
    runner = object.__new__(Qwen35ParoResidentModelRunner)
    runner._session = session
    runner._resolved_mtp2_adapter = lambda: adapter
    runner._clear_session_sampler = lambda: None
    runner._configure_sampled_row = lambda selected: None
    runner._fallback_reasons = {"packed_prefill_unavailable": 0}
    runner._route_counts = {"prefill_chunks": 0}
    return runner


class _AdapterDouble:
    def __init__(self, calls, sink=None) -> None:
        self.calls = calls
        self.sink = sink if sink is not None else _FanOutSink()
        self.begin_prompt = lambda request_id: calls.append(("begin", request_id))
        self.consume_prompt_row = lambda request_id, **kwargs: calls.append(
            ("consume_row", request_id, kwargs)
        )

    def prompt_capture_sink(self, request_id):
        self.calls.append(("sink", request_id))
        return self.sink

    def finish_prompt_capture(self, request_id, *, seed_token):
        self.calls.append(("finish_capture", request_id, seed_token))


def test_paro_packed_prefill_ships_the_chunk_sink_by_default(monkeypatch) -> None:
    calls: list[tuple] = []
    adapter = _AdapterDouble(calls)
    session = _ChunkCaptureSession(calls)
    row = _row(range(2), model_slot=0, native_greedy=True, mtp2_candidate_budget=1, native_prefill=False)
    runner = _runner(session, adapter, row=row)
    monkeypatch.setattr(
        paro_generation_module,
        "malloc",
        lambda nbytes, runtime=None: SimpleNamespace(ptr=0x8000, nbytes=nbytes),
    )

    result = runner._prefill_row_chunk(row, (0, 1), start_position=0, final_chunk=True)

    assert result.token_id == 99
    assert row.native_prefill is True
    assert calls[0] == ("begin", 7)
    assert calls[1] == ("sink", 7)
    packed = calls[2]
    assert packed[0] == "packed" and packed[1] == ((0, 1),) and packed[2] is True
    assert packed[3] is None
    assert packed[4] is adapter.sink
    assert calls[3] == ("finish_capture", 7, 99)


def test_paro_packed_prefill_without_mtp_leaves_the_sink_seam_alone(monkeypatch) -> None:
    calls: list[tuple] = []
    adapter = _AdapterDouble(calls)

    class _PlainSession(_ChunkCaptureSession):
        def prefill_native_packed(self, slab, *, sample=True, final_hidden_row_sink=None, final_hidden_chunk_sink=None):
            calls.append(("packed_plain", final_hidden_row_sink, final_hidden_chunk_sink))
            return (SimpleNamespace(token_id=5),)

    session = _PlainSession(calls)
    row = _row(range(2), model_slot=0, native_greedy=True, mtp2_candidate_budget=0, native_prefill=False)
    runner = _runner(session, adapter, row=row)

    result = runner._prefill_row_chunk(row, (0, 1), start_position=0, final_chunk=True)

    assert result.token_id == 5
    assert calls == [("packed_plain", None, None)]


def test_paro_serial_prefill_fallback_keeps_the_row_seam(monkeypatch) -> None:
    calls: list[tuple] = []
    adapter = _AdapterDouble(calls)
    session = _ChunkCaptureSession(calls, raise_not_implemented=True)
    row = _row(range(3), model_slot=0, native_greedy=True, mtp2_candidate_budget=1, native_prefill=False)
    runner = _runner(session, adapter, row=row)
    monkeypatch.setattr(
        paro_generation_module,
        "malloc",
        lambda nbytes, runtime=None: SimpleNamespace(ptr=0x8000, nbytes=nbytes),
    )

    result = runner._prefill_row_chunk(row, (0, 1, 2), start_position=0, final_chunk=True)

    assert result.token_id == 1236
    assert row.native_prefill is False
    assert runner._fallback_reasons["packed_prefill_unavailable"] == 1
    assert calls[0] == ("begin", 7)
    assert calls[1][0] == "sink"
    assert calls[2][0] == "packed"
    consumed = [call for call in calls if call[0] == "consume_row"]
    assert [call[2]["prompt_index"] for call in consumed] == [0, 1, 2]
    assert [call[2]["target_hidden_ptr"] for call in consumed] == [0x8000, 0x8000, 0x8000]
    assert [call[2]["seed_token"] for call in consumed] == [None, None, 1236]
    assert [call for call in calls if call[0] == "finish_capture"] == []


# -- runner chunk fan-out -----------------------------------------------------


class _FanOutSink:
    request_id = 7
    hidden_size = _HIDDEN
    total_rows = 9

    def __init__(self) -> None:
        self.consumes: list[dict[str, int]] = []
        self.finishes: list[dict[str, int]] = []

    def consume(self, *, request_id, chunk_start, hidden_ptr, rows, stream):
        self.consumes.append(
            {
                "request_id": int(request_id),
                "chunk_start": int(chunk_start),
                "hidden_ptr": int(hidden_ptr),
                "rows": int(rows),
                "stream": int(stream),
            }
        )

    def finish(self, *, request_id, total_rows, stream):
        self.finishes.append(
            {"request_id": int(request_id), "total_rows": int(total_rows), "stream": int(stream)}
        )


def _slab(token_rows, start_positions):
    return CompactPromptSlab.from_token_rows(
        request_ids=tuple(range(7, 7 + len(token_rows))),
        token_rows=token_rows,
        start_positions=start_positions,
        block_count=1,
        slot_ids=tuple(range(len(token_rows))),
    )


def test_paro_packed_chunk_fanout_hands_over_contiguous_request_spans() -> None:
    session = SimpleNamespace(config=SimpleNamespace(hidden_size=_HIDDEN), hidden_nbytes=_BF16_ROW_BYTES)
    slab = _slab(((1, 2, 3, 4),), (0,))
    sink = _FanOutSink()

    Qwen35ParoResidentSession._consume_packed_prefill_hidden_chunks(
        session,
        slab,
        SimpleNamespace(ptr=0x9000),
        (SimpleNamespace(token_id=1),),
        sink=sink,
        stream=3,
    )

    assert sink.consumes == [
        {"request_id": 7, "chunk_start": 0, "hidden_ptr": 0x9000, "rows": 4, "stream": 3},
    ]
    assert sink.finishes == []


def test_paro_packed_chunk_fanout_carries_a_later_scheduler_chunk() -> None:
    session = SimpleNamespace(config=SimpleNamespace(hidden_size=_HIDDEN), hidden_nbytes=_BF16_ROW_BYTES)
    slab = _slab(((1, 2),), (6,))
    sink = _FanOutSink()
    sink.total_rows = 20

    Qwen35ParoResidentSession._consume_packed_prefill_hidden_chunks(
        session,
        slab,
        SimpleNamespace(ptr=0x9000),
        (SimpleNamespace(token_id=1),),
        sink=sink,
    )

    assert sink.consumes == [
        {"request_id": 7, "chunk_start": 6, "hidden_ptr": 0x9000, "rows": 2, "stream": 0},
    ]
    assert sink.finishes == []


def test_paro_packed_chunk_fanout_finishes_only_the_span_that_ends_the_prompt() -> None:
    session = SimpleNamespace(config=SimpleNamespace(hidden_size=_HIDDEN), hidden_nbytes=_BF16_ROW_BYTES)
    slab = _slab(((1, 2),), (6,))
    sink = _FanOutSink()
    sink.total_rows = 8

    Qwen35ParoResidentSession._consume_packed_prefill_hidden_chunks(
        session,
        slab,
        SimpleNamespace(ptr=0x9000),
        (SimpleNamespace(token_id=1),),
        sink=sink,
    )

    assert [entry["chunk_start"] for entry in sink.consumes] == [6]
    assert sink.finishes == [{"request_id": 7, "total_rows": 8, "stream": 0}]


def test_paro_packed_chunk_fanout_refuses_a_peer_request_sink() -> None:
    session = SimpleNamespace(config=SimpleNamespace(hidden_size=_HIDDEN), hidden_nbytes=_BF16_ROW_BYTES)
    slab = _slab(((1, 2),), (0,))
    sink = _FanOutSink()
    sink.request_id = 8

    with pytest.raises(RuntimeError, match="peer request"):
        Qwen35ParoResidentSession._consume_packed_prefill_hidden_chunks(
            session,
            slab,
            SimpleNamespace(ptr=0x9000),
            (SimpleNamespace(token_id=1),),
            sink=sink,
        )
    assert sink.consumes == []


def test_paro_packed_chunk_fanout_requires_a_sampled_root_for_the_final_span() -> None:
    session = SimpleNamespace(config=SimpleNamespace(hidden_size=_HIDDEN), hidden_nbytes=_BF16_ROW_BYTES)
    slab = _slab(((1, 2),), (0,))
    sink = _FanOutSink()
    sink.total_rows = 2

    with pytest.raises(RuntimeError, match="produced no target root"):
        Qwen35ParoResidentSession._consume_packed_prefill_hidden_chunks(
            session,
            slab,
            SimpleNamespace(ptr=0x9000),
            (None,),
            sink=sink,
        )


def test_paro_packed_prefill_refuses_two_hidden_sinks() -> None:
    session = object.__new__(Qwen35ParoResidentSession)
    slab = _slab(((1, 2),), (0,))
    with pytest.raises(ValueError, match="not both"):
        Qwen35ParoResidentSession.prefill_native_packed(
            session,
            slab,
            final_hidden_row_sink=lambda *args: None,
            final_hidden_chunk_sink=_FanOutSink(),
        )


# -- row-capable normalization ------------------------------------------------


def test_paro_row_capable_normalize_keeps_single_row_arithmetic(monkeypatch) -> None:
    launches: list[tuple] = []
    session = object.__new__(Qwen35ParoResidentSession)
    session.config = SimpleNamespace(hidden_size=_HIDDEN, rms_norm_eps=1e-6)
    session.norm_weight = SimpleNamespace(tensor=SimpleNamespace(ptr=0x100))
    session.norm_out = SimpleNamespace(ptr=0x200)
    session.libraries = {"norm": object(), "cast": object()}
    session.runtime = object()
    monkeypatch.setattr(
        paro_runner_module,
        "paro_rmsnorm_out_fp16",
        lambda *args, **kwargs: launches.append(("norm", args[2], args[3])),
    )
    monkeypatch.setattr(
        paro_runner_module,
        "fp16_to_bf16",
        lambda *args, **kwargs: launches.append(("cast", args[0], args[1], args[2])),
    )

    Qwen35ParoResidentSession._write_final_normalized_hidden_bf16(
        session,
        Tensor.from_handle(0x300, (1, _HIDDEN), DType.FP16, Device("hip", 0)),
        destination_ptr=0x400,
    )
    Qwen35ParoResidentSession._write_final_normalized_hidden_bf16_rows(
        session,
        Tensor.from_handle(0x500, (4, _HIDDEN), DType.FP16, Device("hip", 0)),
        rows=4,
        destination_ptr=0x600,
        scratch_ptr=0x700,
    )

    assert launches == [
        ("norm", 0x200, 1),
        ("cast", 0x200, 0x400, _HIDDEN),
        ("norm", 0x700, 4),
        ("cast", 0x700, 0x600, 4 * _HIDDEN),
    ]
