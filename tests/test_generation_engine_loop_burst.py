from __future__ import annotations

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


def _prefill_ramp(*, burst_chunks: int) -> _RampRunner:
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
        loop.submit([9707] * 512, max_new_tokens=128)

    for _ in range(64):
        loop.tick()
        if loop.pending_count == 0 and not loop.scheduler.has_prefill_work():
            break
    else:
        raise AssertionError("prefill ramp did not complete")
    return runner


def test_fair_two_chunk_burst_nearly_halves_static_c8_partial_width_ramp() -> None:
    runner = _prefill_ramp(burst_chunks=2)

    # The final lone prefill row alternates to preserve fair ITL; every earlier
    # row completes in one two-chunk burst and avoids a duplicate partial tick.
    assert runner.decode_widths == [1, 2, 3, 4, 5, 6, 7, 7]
    kinds = [kind for kind, _ in runner.work_sequence]
    first_decode = kinds.index("decode")
    assert kinds[:first_decode] == ["prefill", "prefill"]
    assert max(
        len(run)
        for run in "".join("P" if kind == "prefill" else "D" for kind in kinds).split("D")
    ) == 2


def test_fair_default_keeps_one_chunk_alternation() -> None:
    runner = _prefill_ramp(burst_chunks=1)

    assert runner.decode_widths == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]


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
