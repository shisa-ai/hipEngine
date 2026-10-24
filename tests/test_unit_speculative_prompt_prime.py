"""Shared prompt-priming chunk contract, plan, and ledger.

Covers the host-side extraction only: chunk validation, split boundaries, and
row accounting are shared, while every device call stays inside the model sink.
The dense and MoE GGUF sinks are exercised at the end because they are the two
real consumers of the shared contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.speculative.prompt_prime import (
    PromptPrimeChunk,
    PromptPrimeLedger,
    PromptPrimePlan,
    PromptPrimeSink,
)


# ---------------------------------------------------------------------------
# Ledger: sequential (append) addressing
# ---------------------------------------------------------------------------


def test_prompt_prime_ledger_sequential_accepts_contiguous_chunks() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=5)

    assert ledger.request_id == 7
    assert ledger.total_rows == 5
    assert ledger.contiguous is True
    assert ledger.published_rows == 0
    assert ledger.complete is False
    assert ledger.finished is False

    assert ledger.check(chunk_start=0, rows=2) is None
    ledger.publish(chunk_start=0, rows=2)
    assert ledger.published_rows == 2
    assert ledger.complete is False

    assert ledger.check(chunk_start=2, rows=3) is None
    ledger.publish(chunk_start=2, rows=3)
    assert ledger.published_rows == 5
    assert ledger.complete is True


def test_prompt_prime_ledger_sequential_refuses_a_gap_or_replay() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=6)
    ledger.publish(chunk_start=0, rows=2)

    assert ledger.check(chunk_start=3, rows=1) == "not_contiguous"
    assert ledger.check(chunk_start=1, rows=1) == "not_contiguous"
    # A refusal never moves the cursor, so the retry still lands on the gap.
    assert ledger.published_rows == 2
    assert ledger.check(chunk_start=2, rows=1) is None


def test_prompt_prime_ledger_refuses_zero_and_overlong_chunks() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=4)

    assert ledger.check(chunk_start=0, rows=0) == "empty_chunk"
    assert ledger.check(chunk_start=0, rows=-1) == "empty_chunk"
    assert ledger.check(chunk_start=0, rows=5) == "outside_prompt"
    # A singleton chunk on the last row is admissible.
    assert ledger.check(chunk_start=0, rows=4) is None
    ledger.publish(chunk_start=0, rows=4)
    assert ledger.complete is True
    # With the cursor at the end, one more row runs past the prompt.
    assert ledger.check(chunk_start=4, rows=1) == "outside_prompt"
    assert ledger.check(chunk_start=4, rows=0) == "empty_chunk"


def test_prompt_prime_ledger_refuses_a_chunk_after_finish() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=1)
    ledger.publish(chunk_start=0, rows=1)
    ledger.mark_finished()

    assert ledger.finished is True
    assert ledger.check(chunk_start=1, rows=0) == "finished"


def test_prompt_prime_ledger_cannot_publish_a_refused_chunk() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=3)

    with pytest.raises(ValueError, match="cannot publish"):
        ledger.publish(chunk_start=0, rows=0)
    with pytest.raises(ValueError, match="cannot publish"):
        ledger.publish(chunk_start=1, rows=1)
    assert ledger.published_rows == 0


def test_prompt_prime_ledger_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="request_id"):
        PromptPrimeLedger(request_id=-1, total_rows=1)
    with pytest.raises(ValueError, match="total_rows"):
        PromptPrimeLedger(request_id=0, total_rows=-1)


# ---------------------------------------------------------------------------
# Ledger: position-addressed mode
# ---------------------------------------------------------------------------


def test_prompt_prime_ledger_position_mode_accepts_any_order() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=5, contiguous=False)

    assert ledger.contiguous is False
    # Out-of-order and gapped chunks are legal for a position-addressed sink.
    assert ledger.check(chunk_start=3, rows=2) is None
    ledger.publish(chunk_start=3, rows=2)
    assert ledger.published_rows == 2
    assert ledger.complete is False

    assert ledger.check(chunk_start=0, rows=3) is None
    ledger.publish(chunk_start=0, rows=3)
    assert ledger.published_rows == 5
    assert ledger.complete is True


def test_prompt_prime_ledger_position_mode_counts_rows_once() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=6, contiguous=False)
    ledger.publish(chunk_start=1, rows=3)

    # A partially overlapping replay must not double-count covered rows.
    ledger.publish(chunk_start=2, rows=2)
    assert ledger.published_rows == 3
    ledger.publish(chunk_start=0, rows=1)
    assert ledger.published_rows == 4
    assert ledger.complete is False


def test_prompt_prime_ledger_position_mode_still_bounds_checks() -> None:
    ledger = PromptPrimeLedger(request_id=7, total_rows=4, contiguous=False)

    assert ledger.check(chunk_start=0, rows=0) == "empty_chunk"
    assert ledger.check(chunk_start=-1, rows=1) == "outside_prompt"
    assert ledger.check(chunk_start=2, rows=3) == "outside_prompt"
    assert ledger.check(chunk_start=0, rows=4) is None


# ---------------------------------------------------------------------------
# Plan: split boundaries
# ---------------------------------------------------------------------------


def test_prompt_prime_plan_returns_one_slice_without_cuts() -> None:
    plan = PromptPrimePlan()

    assert plan.split(chunk_start=3, rows=4) == (PromptPrimeChunk(chunk_start=3, rows=4, row_offset=0),)
    assert plan.split(chunk_start=0, rows=0) == ()


def test_prompt_prime_plan_splits_on_the_chunk_row_limit_in_order() -> None:
    plan = PromptPrimePlan(max_chunk_rows=4)

    assert plan.split(chunk_start=0, rows=11) == (
        PromptPrimeChunk(chunk_start=0, rows=4, row_offset=0),
        PromptPrimeChunk(chunk_start=4, rows=4, row_offset=4),
        PromptPrimeChunk(chunk_start=8, rows=3, row_offset=8),
    )
    # The limit is applied from the incoming chunk's first row, not from zero.
    assert plan.split(chunk_start=6, rows=5) == (
        PromptPrimeChunk(chunk_start=6, rows=4, row_offset=0),
        PromptPrimeChunk(chunk_start=10, rows=1, row_offset=4),
    )


def test_prompt_prime_plan_ignores_a_limit_at_or_above_the_chunk() -> None:
    plan = PromptPrimePlan(max_chunk_rows=4)

    assert plan.split(chunk_start=0, rows=4) == (PromptPrimeChunk(chunk_start=0, rows=4, row_offset=0),)
    assert plan.split(chunk_start=0, rows=1) == (PromptPrimeChunk(chunk_start=0, rows=1, row_offset=0),)


def test_prompt_prime_plan_splits_on_interior_capture_boundaries() -> None:
    plan = PromptPrimePlan()

    assert plan.split(chunk_start=0, rows=5, capture_boundaries=(2, 4, 5)) == (
        PromptPrimeChunk(chunk_start=0, rows=2, row_offset=0),
        PromptPrimeChunk(chunk_start=2, rows=2, row_offset=2),
        PromptPrimeChunk(chunk_start=4, rows=1, row_offset=4),
    )


def test_prompt_prime_plan_offsets_boundaries_by_the_start_position() -> None:
    plan = PromptPrimePlan(start_position=10)

    assert plan.split(chunk_start=0, rows=7, capture_boundaries=(12, 14, 17)) == (
        PromptPrimeChunk(chunk_start=0, rows=2, row_offset=0),
        PromptPrimeChunk(chunk_start=2, rows=2, row_offset=2),
        PromptPrimeChunk(chunk_start=4, rows=3, row_offset=4),
    )
    # A boundary outside this chunk's absolute span never cuts it.
    assert plan.split(chunk_start=4, rows=3, capture_boundaries=(12, 14)) == (
        PromptPrimeChunk(chunk_start=4, rows=3, row_offset=0),
    )


def test_prompt_prime_plan_ignores_a_boundary_on_either_edge() -> None:
    plan = PromptPrimePlan(start_position=10)

    # 10 is the first row and 13 is one past the last: neither may split, and
    # neither may emit a zero-row slice.
    assert plan.split(chunk_start=0, rows=3, capture_boundaries=(10, 13)) == (
        PromptPrimeChunk(chunk_start=0, rows=3, row_offset=0),
    )


def test_prompt_prime_plan_merges_the_row_limit_with_capture_boundaries() -> None:
    plan = PromptPrimePlan(max_chunk_rows=4)

    # The row limit and a boundary coincide at 4; the slice is emitted once.
    assert plan.split(chunk_start=0, rows=7, capture_boundaries=(2, 4)) == (
        PromptPrimeChunk(chunk_start=0, rows=2, row_offset=0),
        PromptPrimeChunk(chunk_start=2, rows=2, row_offset=2),
        PromptPrimeChunk(chunk_start=4, rows=3, row_offset=4),
    )


def test_prompt_prime_plan_rejects_negative_configuration() -> None:
    with pytest.raises(ValueError, match="start_position"):
        PromptPrimePlan(start_position=-1)
    with pytest.raises(ValueError, match="max_chunk_rows"):
        PromptPrimePlan(max_chunk_rows=-1)


# ---------------------------------------------------------------------------
# Dense GGUF sink: real consumer
# ---------------------------------------------------------------------------


class _DenseRuntime:
    def __init__(self) -> None:
        self.memory = {0x5000 + i * 8: i for i in range(16)}
        self.memory[0x2000] = 999
        self.queue: list = []

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        return None

    def memcpy(self, dst: int, src: int, nbytes: int, kind) -> None:
        self.memory[dst] = self.memory[src]

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind, stream: int) -> None:
        assert stream == 3
        self.queue.append(lambda: self.memory.__setitem__(dst, self.memory[src]))


class _DenseExecutor:
    def __init__(self, runtime: _DenseRuntime) -> None:
        self.runtime = runtime
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def enqueue_prompt_rows(self, request_id, token_ids, **kwargs) -> None:
        assert kwargs["stream"] == 3
        tokens = tuple(int(token) for token in token_ids)
        ptr = int(kwargs["target_hidden_base_ptr"])
        position = int(kwargs["position_start"])
        self.calls.append((tokens, position, ptr))
        self.runtime.queue.append(lambda: None)


def _dense_sink(monkeypatch, *, prompt_tokens, **kwargs):
    from hipengine.runtime import qwen35_gguf_mtp as mtp_module

    monkeypatch.setattr(
        mtp_module, "malloc", lambda nbytes, **kw: DeviceBuffer(0x1000, nbytes)
    )
    monkeypatch.setattr(mtp_module, "free", lambda *args, **kw: None)
    runtime = _DenseRuntime()
    executor = _DenseExecutor(runtime)
    sink = mtp_module._StreamingNextNPromptSink(
        request_id=7,
        prompt_tokens=prompt_tokens,
        hidden_size=4,
        executor=executor,
        runtime=runtime,
        checkpoint=None,
        **kwargs,
    )
    return sink, runtime, executor


def test_dense_sink_is_a_prompt_prime_sink(monkeypatch) -> None:
    sink, _runtime, _executor = _dense_sink(monkeypatch, prompt_tokens=(10, 20))

    assert isinstance(sink, PromptPrimeSink)


def test_dense_sink_accounts_through_the_shared_ledger(monkeypatch) -> None:
    sink, _runtime, _executor = _dense_sink(monkeypatch, prompt_tokens=(10, 20, 30))

    ledger = sink.prompt_prime_ledger
    assert isinstance(ledger, PromptPrimeLedger)
    assert ledger.contiguous is True
    assert ledger.request_id == sink.request_id
    assert ledger.total_rows == sink.total_rows

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=2, stream=3)
    assert ledger.published_rows == sink.consumed_rows == 2
    assert ledger.complete is False

    sink.consume(request_id=7, chunk_start=2, hidden_ptr=0x5010, rows=1, stream=3)
    assert ledger.complete is True
    assert ledger.finished is False
    sink.finish(request_id=7, total_rows=3, stream=3)
    assert ledger.finished is True


def test_dense_sink_split_order_matches_the_row_limit_and_boundaries(monkeypatch) -> None:
    sink, runtime, executor = _dense_sink(
        monkeypatch,
        prompt_tokens=tuple(range(11, 18)),
        start_position=10,
        initial_hidden=Tensor.from_handle(0x2000, (1, 4), DType.BF16, Device("hip", 0)),
        transform_hidden_rows=lambda src, count, stream: src,
        max_chunk_rows=4,
    )
    sink.capture_boundaries = (12, 14, 17)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=7, stream=3)

    # Slices are (0,2), (2,2), (4,3): one carried-row enqueue plus one chunk
    # enqueue per slice, so positions stay in order and only the last slice
    # carries two rows in its second enqueue.
    assert [position for _, position, _ in executor.calls] == [
        10, 11, 12, 13, 14, 15,
    ]
    assert [len(tokens) for tokens, _, _ in executor.calls] == [1, 1, 1, 1, 1, 2]
    assert sink.consumed_rows == 7
    sink.finish(request_id=7, total_rows=7, stream=3)


def test_dense_sink_slices_keep_the_borrowed_row_pointer_offsets(monkeypatch) -> None:
    sink, runtime, executor = _dense_sink(
        monkeypatch,
        prompt_tokens=(10, 20, 30, 40, 50),
        transform_hidden_rows=lambda src, count, stream: src,
    )
    sink.capture_boundaries = (2, 4, 5)

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=5, stream=3)

    # Each slice's first enqueue targets the carried row buffer; its second
    # enqueue must address that slice's own first borrowed row, not the chunk
    # base. Slices are (0,2), (2,2), (4,1), so the last one has no second call.
    assert [(position, ptr) for _, position, ptr in executor.calls] == [
        (0, 0x1000),
        (1, 0x5000),
        (2, 0x1000),
        (3, 0x5010),
        (4, 0x1000),
    ]
    assert sink.consumed_rows == 5


def test_dense_sink_refuses_a_non_contiguous_chunk(monkeypatch) -> None:
    sink, runtime, executor = _dense_sink(monkeypatch, prompt_tokens=(10, 20, 30))

    with pytest.raises(RuntimeError, match="must be contiguous: expected 0, got 1"):
        sink.consume(request_id=7, chunk_start=1, hidden_ptr=0x5000, rows=1, stream=3)
    assert sink.consumed_rows == 0
    assert executor.calls == []


def test_dense_sink_refuses_a_zero_row_chunk(monkeypatch) -> None:
    sink, runtime, executor = _dense_sink(monkeypatch, prompt_tokens=(10, 20, 30))

    with pytest.raises(ValueError, match="outside the prompt"):
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=0, stream=3)
    with pytest.raises(ValueError, match="outside the prompt"):
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=4, stream=3)
    assert sink.consumed_rows == 0


def test_dense_sink_accepts_a_singleton_chunk(monkeypatch) -> None:
    sink, runtime, executor = _dense_sink(monkeypatch, prompt_tokens=(10, 20))

    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=1, stream=3)
    sink.consume(request_id=7, chunk_start=1, hidden_ptr=0x5008, rows=1, stream=3)
    sink.finish(request_id=7, total_rows=2, stream=3)

    assert sink.consumed_rows == 2
    assert [position for _, position, _ in executor.calls] == [0, 1]


def test_dense_sink_publishes_rows_only_after_the_enqueues_land(monkeypatch) -> None:
    def refuse_second_slice(src: int, count: int, stream: int) -> int:
        if src != 0x5000:
            raise RuntimeError("transform refused")
        return src

    sink, runtime, executor = _dense_sink(
        monkeypatch,
        prompt_tokens=(10, 20, 30, 40),
        transform_hidden_rows=refuse_second_slice,
    )
    sink.capture_boundaries = (2,)

    with pytest.raises(RuntimeError, match="transform refused"):
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=4, stream=3)

    # The first slice landed and is published; the refused one is not.
    assert sink.consumed_rows == 2
    assert [position for _, position, _ in executor.calls] == [0, 1]
    with pytest.raises(RuntimeError, match="is failed"):
        sink.consume(request_id=7, chunk_start=2, hidden_ptr=0x5010, rows=2, stream=3)


def test_dense_sink_finish_requires_every_row(monkeypatch) -> None:
    sink, runtime, executor = _dense_sink(monkeypatch, prompt_tokens=(10, 20, 30))
    sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=2, stream=3)

    with pytest.raises(RuntimeError, match="finished before every row was consumed"):
        sink.finish(request_id=7, total_rows=3, stream=3)
    with pytest.raises(RuntimeError, match="finished before every row was consumed"):
        sink.finish(request_id=7, total_rows=2, stream=3)

    sink.consume(request_id=7, chunk_start=2, hidden_ptr=0x5010, rows=1, stream=3)
    sink.finish(request_id=7, total_rows=3, stream=3)
    with pytest.raises(RuntimeError, match="already finished"):
        sink.consume(request_id=7, chunk_start=3, hidden_ptr=0x5018, rows=0, stream=3)


# ---------------------------------------------------------------------------
# MoE GGUF sink: real consumer
# ---------------------------------------------------------------------------


class _MoeRuntime:
    def __init__(self) -> None:
        self.copies: list[tuple[int, int, int, int]] = []

    def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), int(stream)))


def _moe_sink(*, total_rows: int, runtime: _MoeRuntime | None = None):
    from hipengine.generation.qwen35_gguf_moe_mtp2 import _MoeTargetHiddenSink

    runtime = runtime or _MoeRuntime()
    target = SimpleNamespace(
        runtime=runtime,
        scratch=SimpleNamespace(hidden_seed_fp32=DeviceBuffer(0x7000, 16)),
        _hidden_seed_fp32_populated=False,
        _last_target_hidden_ptr=0,
    )
    norm_calls: list[int] = []

    def run_norm(src, dst, *, stream, capture_hidden_seed_fp32):
        norm_calls.append(int(src))

    target._run_output_norm_hidden = run_norm
    sink = _MoeTargetHiddenSink(
        request_id=7,
        hidden_size=4,
        total_rows=total_rows,
        target=target,
        buffer=DeviceBuffer(0x1000, total_rows * 4 * DType.BF16.itemsize),
        normalized=DeviceBuffer(0x2000, total_rows * 4 * DType.FP32.itemsize),
        normalized_bf16=DeviceBuffer(0x3000, 4 * DType.BF16.itemsize),
    )
    return sink, runtime, target, norm_calls


def test_moe_sink_is_a_prompt_prime_sink() -> None:
    sink, _runtime, _target, _calls = _moe_sink(total_rows=3)

    assert isinstance(sink, PromptPrimeSink)
    assert sink.request_id == 7
    assert sink.hidden_size == 4
    assert sink.total_rows == 3


def test_moe_sink_tracks_published_rows_in_position_order() -> None:
    sink, runtime, target, norm_calls = _moe_sink(total_rows=3)

    sink.consume(request_id=7, chunk_start=1, hidden_ptr=0x5000, rows=2, stream=9)

    assert runtime.copies == [(0x1008, 0x5000, 16, 9)]
    assert sink.prompt_prime_ledger.published_rows == 2
    assert sink.prompt_prime_ledger.complete is False
    assert target._hidden_seed_fp32_populated is False
    assert norm_calls == []

    sink.finish(request_id=7, total_rows=3, stream=9)
    assert sink.finished is True
    assert [call for call in norm_calls] == [0x1000, 0x1008, 0x1010]


def test_moe_sink_refuses_an_invalid_chunk_before_copying() -> None:
    sink, runtime, _target, _calls = _moe_sink(total_rows=3)

    with pytest.raises(ValueError, match="chunk is invalid"):
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=0, stream=9)
    with pytest.raises(ValueError, match="chunk is invalid"):
        sink.consume(request_id=7, chunk_start=2, hidden_ptr=0x5000, rows=2, stream=9)
    with pytest.raises(RuntimeError, match="owner changed"):
        sink.consume(request_id=8, chunk_start=0, hidden_ptr=0x5000, rows=1, stream=9)

    assert runtime.copies == []
    assert sink.prompt_prime_ledger.published_rows == 0


def test_moe_sink_does_not_publish_a_failed_copy() -> None:
    class RefusingRuntime(_MoeRuntime):
        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            raise RuntimeError("copy refused")

    sink, runtime, _target, _calls = _moe_sink(total_rows=3, runtime=RefusingRuntime())

    with pytest.raises(RuntimeError, match="copy refused"):
        sink.consume(request_id=7, chunk_start=0, hidden_ptr=0x5000, rows=2, stream=9)
    assert sink.prompt_prime_ledger.published_rows == 0


def test_moe_sink_keeps_its_finish_identity_checks() -> None:
    sink, _runtime, _target, _calls = _moe_sink(total_rows=3)

    with pytest.raises(RuntimeError, match="completion identity changed"):
        sink.finish(request_id=8, total_rows=3, stream=9)
    with pytest.raises(RuntimeError, match="completion identity changed"):
        sink.finish(request_id=7, total_rows=2, stream=9)
