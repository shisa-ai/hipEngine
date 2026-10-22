"""Decode-aware token-budget scheduling uses bounded work, not timer promises."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from hipengine.generation import EngineLoopConfig, GeneratedToken, ResidentEngineLoop
from hipengine.llm import _engine_loop_config_with_generator_defaults


@dataclass
class _Runner:
    capacity: int = 3
    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = True

    def __post_init__(self):
        self.chunks = []
        self.clock = 0.0

    def prefill_batch(self, work, *, commit):
        assert commit
        self.chunks.append(tuple(len(row) for row in work.token_rows))
        self.clock += sum(len(row) for row in work.token_rows) * 0.01

    def decode_batch(self, work, *, commit):
        assert commit
        self.clock += 0.001
        return tuple(GeneratedToken(request_id, 100) for request_id in work.request_ids)

    def reclaim(self, completed):
        pass


def _loop(monkeypatch, *, multiple=True):
    runner = _Runner()
    runner.supports_multiple_prefill_quanta_per_round = multiple
    monkeypatch.setattr("hipengine.generation.engine_loop.time.perf_counter", lambda: runner.clock)
    loop = ResidentEngineLoop(runner, config=EngineLoopConfig(
        prefill_decode_policy="token_budget", max_active_requests=3,
        max_prefill_chunk_tokens=8, round_prefill_token_budget=8,
        decode_prefill_token_budget=3, round_decode_row_budget=3,
    ))
    return loop, runner


def test_idle_active_and_drained_decode_restore_large_prefill(monkeypatch):
    loop, runner = _loop(monkeypatch)
    loop.submit((1,), max_new_tokens=3)
    loop.tick()
    long_id = loop.submit(tuple(range(23)), max_new_tokens=1)
    loop.tick()
    assert runner.chunks[-1] == (3,)
    policy = loop.observability_snapshot()["scheduler_policy"]
    recent = policy["recent_prefill_slices"][-1]
    assert recent["decode_rows"] == 1
    assert recent["token_budget"] == 3
    assert recent["tokens"] == 3
    assert recent["context_lengths"] == [0]
    assert recent["duration_ms"] == pytest.approx(30)
    loop.tick()
    assert runner.chunks[-1] == (3,)
    loop.tick()
    assert runner.chunks[-1] == (8,)
    for _ in range(10):
        if not loop.active_count and not loop.pending_count:
            break
        loop.tick()
    assert long_id in loop.completed
    assert sum(sum(chunk) for chunk in runner.chunks) == 24


def test_active_budget_bounds_the_whole_round_not_only_each_chunk(monkeypatch):
    loop, runner = _loop(monkeypatch)
    loop.submit((1,), max_new_tokens=10)
    loop.tick()
    loop.submit(tuple(range(19)), max_new_tokens=1)
    loop.submit(tuple(range(11)), max_new_tokens=1)
    before = len(runner.chunks)
    loop.tick()
    assert sum(sum(chunk) for chunk in runner.chunks[before:]) == 3
    loop.tick()
    snapshots = loop.observability_snapshot()["scheduler_policy"]["recent_prefill_slices"]
    assert snapshots[-1]["request_ids"] != snapshots[-2]["request_ids"]


def test_cancelled_peer_no_longer_reduces_idle_prefill(monkeypatch):
    loop, runner = _loop(monkeypatch)
    peer = loop.submit((1,), max_new_tokens=10)
    loop.tick()
    loop.cancel(peer)
    loop.submit(tuple(range(19)), max_new_tokens=1)
    loop.tick()
    assert runner.chunks[-1] == (8,)


@pytest.mark.parametrize("bad", [0, -1])
def test_nonpositive_decode_prefill_budget_is_rejected(bad):
    with pytest.raises(ValueError, match="decode_prefill_token_budget"):
        EngineLoopConfig(decode_prefill_token_budget=bad)


def test_operator_round_budget_disables_implicit_decode_slice():
    class Generator:
        engine_loop_config_defaults = {
            "prefill_decode_policy": "token_budget",
            "round_prefill_token_budget": 1024,
            "decode_prefill_token_budget": 64,
        }

    defaults = _engine_loop_config_with_generator_defaults(EngineLoopConfig(), Generator(), environ={})
    assert defaults.decode_prefill_token_budget == 64
    pinned = _engine_loop_config_with_generator_defaults(
        EngineLoopConfig(round_prefill_token_budget=512), Generator(),
        environ={"HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET": "512"},
    )
    assert pinned.round_prefill_token_budget == 512
    assert pinned.decode_prefill_token_budget is None


@pytest.mark.parametrize("chunk", [512, 1024, 1537, 4096])
def test_qwen4exp_admitted_chunk_bounds_idle_and_decode_active_work(chunk):
    from types import SimpleNamespace
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator

    defaults = Qwen4ExpGGUFTextGenerator.engine_loop_config_defaults.fget(
        SimpleNamespace(runner=SimpleNamespace(prefill_chunk_size=chunk),
                        _configured_resident_capacity=3)
    )
    generator = SimpleNamespace(engine_loop_config_defaults=defaults)
    config = _engine_loop_config_with_generator_defaults(
        EngineLoopConfig(max_active_requests=3), generator, environ={}
    )
    assert config.max_prefill_chunk_tokens == chunk
    assert config.round_prefill_token_budget == chunk
    assert config.decode_prefill_token_budget == min(1024, chunk)
    runner = _Runner()
    loop = ResidentEngineLoop(runner, config=config)
    long = loop.submit(tuple(range(2 * chunk + 13)), max_new_tokens=1)
    loop.tick()
    assert runner.chunks == [(chunk,)]
    loop.cancel(long)
    peer = loop.submit((1,), max_new_tokens=10)
    loop.tick()
    loop.submit(tuple(range(2 * chunk + 13)), max_new_tokens=1)
    before = len(runner.chunks)
    loop.tick()
    assert sum(sum(row) for row in runner.chunks[before:]) == min(1024, chunk)
    loop.cancel(peer)
    loop.tick()
    assert runner.chunks[-1] == (chunk,)

    pinned = _engine_loop_config_with_generator_defaults(
        EngineLoopConfig(max_prefill_chunk_tokens=137, round_prefill_token_budget=251,
                         decode_prefill_token_budget=113), generator,
        environ={"HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": "137",
                 "HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET": "251",
                 "HIPENGINE_DECODE_PREFILL_TOKEN_BUDGET": "113"},
    )
    assert pinned.max_prefill_chunk_tokens == 137
    assert pinned.round_prefill_token_budget == 251
    assert pinned.decode_prefill_token_budget == 113
