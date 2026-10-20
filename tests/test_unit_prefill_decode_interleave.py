"""Resumable prefill returns between device segments, before publishing tokens."""
from dataclasses import dataclass

from hipengine.generation import EngineLoopConfig, GeneratedToken, ResidentEngineLoop


@dataclass
class _Runner:
    capacity: int = 3
    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = False

    def __post_init__(self):
        self.calls = []
        self.segments = 0

    def prefill_batch(self, work, *, commit):
        self.calls.append(('prefill', work.request_ids))

    def prefill_batch_resumable(self, work, *, commit):
        self.calls.append(('segment', work.request_ids))
        self.segments += 1
        return work.request_ids if self.segments % 3 else ()

    def decode_batch(self, work, *, commit):
        self.calls.append(('decode', work.request_ids))
        return tuple(GeneratedToken(request_id, 100) for request_id in work.request_ids)

    def reclaim(self, completed):
        pass


def _loop(runner):
    return ResidentEngineLoop(runner, config=EngineLoopConfig(
        prefill_decode_policy='token_budget', max_active_requests=3,
        max_prefill_chunk_tokens=8, round_prefill_token_budget=8,
        round_decode_row_budget=3,
    ))


def test_resumable_prompt_stays_unready_and_decode_returns_each_tick():
    runner = _Runner()
    loop = _loop(runner)
    peer = loop.submit((1,), max_new_tokens=5)
    loop.tick()
    runner.calls.clear()
    arrival = loop.submit((2, 3, 4), max_new_tokens=2)
    for _ in range(2):
        events = loop.tick()
        assert loop.scheduler.active_batch.requests[arrival].remaining_prefill == 3
        assert sum(event.kind == 'token' for event in events) == 1
    loop.tick()
    assert runner.calls == [
        ('segment', (arrival,)), ('decode', (peer,)),
        ('segment', (arrival,)), ('decode', (peer,)),
        ('segment', (arrival,)), ('decode', (peer, arrival)),
    ]
    assert loop.scheduler.active_batch.requests[arrival].remaining_decode == 1


def test_legacy_prefill_runner_keeps_one_decode_per_round():
    runner = _Runner()
    runner.prefill_batch_resumable = None
    loop = _loop(runner)
    peer = loop.submit((1,), max_new_tokens=10)
    loop.tick()
    arrival = loop.submit((2, 3, 4), max_new_tokens=2)
    runner.calls.clear()
    loop.tick()
    assert runner.calls == [('prefill', (arrival,)), ('decode', (peer, arrival))]
