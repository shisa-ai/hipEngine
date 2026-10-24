"""Attribute accepted AR emissions before stream snapshots and terminal reclaim."""
from dataclasses import replace

import pytest

from hipengine.generation.engine_loop import ResidentEngineLoop
from hipengine.generation.registry import GenerationStreamChunk, GenerationTelemetry
from tests.test_unit_specdec2_engine_loop import _ArCommitRunner


class _PublicationRunner(_ArCommitRunner):
    def __init__(self, *, cancel_on_decode=False):
        super().__init__()
        self.counts = {}
        self.reclaimed_counts = {}
        self.cancel_on_decode = cancel_on_decode
        self.cancelled = set()

    def decode_batch(self, work, *, commit):
        result = super().decode_batch(work, commit=commit)
        rows = []
        for token in result:
            self.counts.setdefault(token.request_id, 0)
            if self.cancel_on_decode:
                self.cancelled.add(token.request_id)
            telemetry = GenerationTelemetry.from_decode_counts(
                prompt_tokens=1, generated_tokens=self._decode_counts[token.request_id]
            )
            rows.append(replace(token, stream_chunk=GenerationStreamChunk(
                text="x", telemetry=replace(telemetry, diagnostics={
                    "ar_commits": self.counts[token.request_id]
                })
            )))
        return tuple(rows)

    def note_speculative_ar_commit(self, request_id, reason=None):
        assert request_id not in self.reclaimed_counts
        super().note_speculative_ar_commit(request_id, reason)
        self.counts[request_id] += 1

    def decorate_autoregressive_tokens(self, tokens):
        return tuple(replace(token, stream_chunk=replace(
            token.stream_chunk, telemetry=replace(
                token.stream_chunk.telemetry,
                diagnostics={"ar_commits": self.counts[token.request_id]},
            )
        )) for token in tokens)

    def reclaim(self, completed):
        self.reclaimed_counts[completed.request_id] = self.counts.get(completed.request_id, 0)

    def request_cancel_reason(self, request_id):
        return "cancel" if request_id in self.cancelled else None


@pytest.mark.parametrize("limit", [1, 3])
def test_ar_publication_includes_the_terminal_attribution(limit):
    runner = _PublicationRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    rid = loop.submit_speculative([10], max_new_tokens=limit, desired_candidate_count=1)

    events = loop.poll(max_ticks=8)

    tokens = [event for event in events if event.kind == "token"]
    assert len(tokens) == limit
    assert [event.stream_chunk.telemetry.diagnostics["ar_commits"] for event in tokens] == list(range(1, limit + 1))
    assert runner.reclaimed_counts[rid] == limit
    assert len(runner.ar_commits) == limit
    assert loop._pending_ar_reasons == {}


def test_cancelled_ar_decode_is_neither_attributed_nor_published():
    runner = _PublicationRunner(cancel_on_decode=True)
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    rid = loop.submit_speculative([10], max_new_tokens=1, desired_candidate_count=1)

    events = loop.poll(max_ticks=8)

    assert not [event for event in events if event.kind == "token"]
    assert runner.ar_commits == []
    assert runner.reclaimed_counts[rid] == 0
    assert loop._pending_ar_reasons == {}
