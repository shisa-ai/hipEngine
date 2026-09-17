"""The streaming mailbox must be bounded by the child's own output budget.

A consumer that is slower than the producer has to slow the child down, not lose
the request: the mailbox is the buffer between the engine and the HTTP writer, so
a fixed 64-entry bound cancelled any stream that outran its client (measured on
gfx1151: an MTP stream was cancelled after ~96 tokens with
``budget_pressure=client_backpressure`` and no usage block). These tests pin the
resolved bound and the collector behavior at it.
"""

from __future__ import annotations

import pytest

from hipengine.generation import (
    EngineOutput,
    GenerationStreamChunk,
    OutputKind,
    StreamingOutputCollector,
)
from hipengine.generation.concurrency2 import (
    STREAM_MAILBOX_BUDGET_SLACK,
    STREAM_MAILBOX_HARD_CAP_CHUNKS,
    stream_mailbox_bound,
)


def test_stream_mailbox_bound_scales_with_the_output_budget() -> None:
    assert stream_mailbox_bound(64, 128) == 128 + STREAM_MAILBOX_BUDGET_SLACK
    assert stream_mailbox_bound(64, 200) == 200 + STREAM_MAILBOX_BUDGET_SLACK


def test_stream_mailbox_bound_keeps_a_floor_for_small_budgets() -> None:
    assert stream_mailbox_bound(64, 2) == 64
    assert stream_mailbox_bound(64, 0) == 64
    assert stream_mailbox_bound(1, 20) == 20 + STREAM_MAILBOX_BUDGET_SLACK


def test_stream_mailbox_bound_caps_a_dead_client_backlog() -> None:
    assert stream_mailbox_bound(64, 1_000_000) == STREAM_MAILBOX_HARD_CAP_CHUNKS
    # An explicitly larger configured bound is honored rather than lowered.
    assert stream_mailbox_bound(STREAM_MAILBOX_HARD_CAP_CHUNKS + 10, 4) == (
        STREAM_MAILBOX_HARD_CAP_CHUNKS + 10
    )


def test_stream_mailbox_bound_tolerates_a_missing_budget() -> None:
    assert stream_mailbox_bound(64, None) == 64


def test_stream_mailbox_bound_rejects_a_nonpositive_floor() -> None:
    with pytest.raises(ValueError):
        stream_mailbox_bound(0, 128)


def _chunk(request_id: int) -> EngineOutput:
    return EngineOutput(
        kind=OutputKind.CHUNK,
        request_id=request_id,
        stream_chunk=GenerationStreamChunk(text="x"),
    )


def test_collector_holds_a_whole_output_budget_without_rejecting() -> None:
    budget = 128
    collector = StreamingOutputCollector(
        max_output_tokens=budget,
        max_chunks=stream_mailbox_bound(64, budget),
        enqueue_token_events=False,
    )
    collector.bind(1)
    for _ in range(budget):
        assert collector.publish(_chunk(1)) is True
    # The whole budget is buffered and the consumer can still drain it.
    assert len(collector.drain()) == budget


def test_collector_rejects_a_backlog_past_the_hard_cap() -> None:
    collector = StreamingOutputCollector(
        max_output_tokens=1_000_000,
        max_chunks=stream_mailbox_bound(1, 1_000_000),
        enqueue_token_events=False,
    )
    collector.bind(1)
    for _ in range(STREAM_MAILBOX_HARD_CAP_CHUNKS):
        assert collector.publish(_chunk(1)) is True
    assert collector.publish(_chunk(1)) is False


def test_resident_stream_queue_bound_reads_the_submission_budget() -> None:
    from hipengine.generation.engine_loop import (
        DEFAULT_RESIDENT_STREAM_QUEUE_MAX_CHUNKS,
        GenerationSubmission,
        _ResidentStreamState,
        _resident_stream_queue_bound,
    )
    from hipengine.generation.registry import GenerationRequest

    def state(max_tokens: int) -> _ResidentStreamState:
        request = GenerationRequest(
            prompts=("prompt",),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=False,
        )
        return _ResidentStreamState(
            submission=GenerationSubmission(
                request_ids=(1,),
                request=request,
                max_ticks=max_tokens,
            )
        )

    assert _resident_stream_queue_bound(state(128)) == (
        128 + STREAM_MAILBOX_BUDGET_SLACK
    )
    assert _resident_stream_queue_bound(state(4)) == (
        DEFAULT_RESIDENT_STREAM_QUEUE_MAX_CHUNKS
    )
    assert _resident_stream_queue_bound(state(1_000_000)) == (
        STREAM_MAILBOX_HARD_CAP_CHUNKS
    )
