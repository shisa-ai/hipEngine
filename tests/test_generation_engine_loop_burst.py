from __future__ import annotations

import threading
import time

import hipengine.generation.engine_loop as engine_loop_module
from hipengine.generation.batch_scheduler import GeneratedToken
from hipengine.generation.engine_loop import EngineLoopConfig, ResidentEngineLoop


class _RampRunner:
    def __init__(self) -> None:
        self.work_sequence: list[tuple[str, tuple[int, ...]]] = []
        self.decode_widths: list[int] = []
        self._counts: dict[int, int] = {}

    def prefill_batch(self, work, *, commit: bool) -> None:
        assert commit is True
        self.work_sequence.append(("prefill", tuple(work.request_ids)))

    def decode_batch(self, work, *, commit: bool):
        assert commit is True
        request_ids = tuple(work.request_ids)
        self.work_sequence.append(("decode", request_ids))
        self.decode_widths.append(len(request_ids))
        generated = []
        for request_id in request_ids:
            count = self._counts.get(request_id, 0) + 1
            self._counts[request_id] = count
            generated.append(GeneratedToken(request_id, 1000 + count))
        return tuple(generated)


def _prefill_ramp(*, burst_chunks: int, prompt_length: int = 512) -> _RampRunner:
    runner = _RampRunner()
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="fair",
            fair_prefill_burst_chunks=burst_chunks,
            max_active_requests=8,
            max_prefill_chunk_tokens=256,
        ),
    )
    for _ in range(8):
        loop.submit([9707] * prompt_length, max_new_tokens=128)

    for _ in range(64):
        loop.tick()
        if loop.pending_count == 0 and not loop.scheduler.has_prefill_work():
            loop.tick()
            break
    else:
        raise AssertionError("prefill ramp did not complete")
    return runner


def test_fair_two_chunk_bound_drains_short_cold_cohort_to_full_width() -> None:
    runner = _prefill_ramp(burst_chunks=2)

    assert runner.decode_widths == [8]
    kinds = [kind for kind, _ in runner.work_sequence]
    first_decode = kinds.index("decode")
    assert kinds[:first_decode] == ["prefill"] * 16


def test_fair_two_chunk_bound_does_not_drain_cold_rows_above_bound() -> None:
    runner = _prefill_ramp(burst_chunks=2, prompt_length=768)

    assert runner.decode_widths[0] == 1
    kinds = [kind for kind, _ in runner.work_sequence]
    assert kinds[: kinds.index("decode")] == ["prefill"] * 3


def test_fair_default_keeps_one_chunk_alternation() -> None:
    runner = _prefill_ramp(burst_chunks=1)

    assert runner.decode_widths == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]


def test_cold_drain_absorbs_a_bounded_predecode_arrival() -> None:
    runner = _RampRunner()
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="fair",
            fair_prefill_burst_chunks=2,
            max_active_requests=2,
            max_prefill_chunk_tokens=256,
        ),
    )
    loop.submit([9707] * 512, max_new_tokens=8)
    loop.tick()
    loop.submit([9708] * 512, max_new_tokens=8)

    for _ in range(4):
        loop.tick()

    assert [kind for kind, _ in runner.work_sequence] == ["prefill"] * 4 + ["decode"]
    assert runner.decode_widths == [2]


def test_submission_priority_hands_the_loop_to_waiting_admission_before_poll() -> None:
    gate = engine_loop_module._SubmissionPriority()
    loop_lock = threading.Lock()
    loop_lock.acquire()
    order: list[str] = []

    def submit() -> None:
        with gate.submission(loop_lock):
            order.append("submit")

    def poll() -> None:
        gate.wait_for_submissions()
        with loop_lock:
            order.append("poll")

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    deadline = time.perf_counter() + 1.0
    while gate.waiting_count != 1 and time.perf_counter() < deadline:
        time.sleep(0.001)
    assert gate.waiting_count == 1

    poll_thread = threading.Thread(target=poll)
    poll_thread.start()
    time.sleep(0.01)
    assert order == []

    loop_lock.release()
    submit_thread.join(timeout=1.0)
    poll_thread.join(timeout=1.0)

    assert not submit_thread.is_alive()
    assert not poll_thread.is_alive()
    assert order == ["submit", "poll"]


def test_fair_two_chunk_bound_keeps_lone_staggered_prefill_alternating() -> None:
    runner = _RampRunner()
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="fair",
            fair_prefill_burst_chunks=2,
            max_active_requests=2,
            max_prefill_chunk_tokens=256,
        ),
    )
    loop.submit([9707] * 256, max_new_tokens=8)
    loop.tick()
    loop.tick()
    loop.submit([9708] * 512, max_new_tokens=8)
    runner.work_sequence.clear()

    loop.tick()
    loop.tick()
    loop.tick()

    assert [kind for kind, _ in runner.work_sequence] == ["prefill", "decode", "prefill"]
