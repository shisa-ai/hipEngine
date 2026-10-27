"""CPU lifecycle contracts for the resumable Flash-Next prefill path.

Every device call here is a behavioral fake, and every bookkeeping decision is
the real implementation: the serving runner's ``prefill_batch_resumable`` and
``_prefill_row``, the scheduler's ``defer_prefill_work`` and round-robin
prefill rotation, the token-budget round, the AR batch owner, and the resident
model runner's suspended-segment context. The fakes keep the state the real
code reads (prompt cursor, ``_pending_prefill_append``, staging-runner
ownership, batch import) and do no arithmetic.

Each test pins one lifecycle edge:

* an incomplete segment cannot enter decode and cannot advance prompt
  accounting;
* a completing segment commits its chunk and publishes the first token once;
* a decode drain mid-chunk resumes the identical pending chunk before the
  wider round's tail, and does not spend the leftover budget on a second
  chunk;
* a cancellation closes the suspended segment before its staging runner is
  reused, and a peer survives;
* a failed segment cannot publish prompt progress for work that never ran;
* each prefill slice and the round's prefill token counter report committed
  prompt tokens, not the planned tokens a suspension repeats;
* deferring a segment tolerates a work row the runner already retired;
* the operator's prefill budget and the round-robin prefill rotation survive
  suspensions;
* the AR batch imports a resumed prefill exactly once.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation import GenerationExecutionFailed
from hipengine.generation.batch_scheduler import ResidentBatchScheduler
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
AR_BATCH_ENV = "HIPENGINE_QWEN4_EXP_AR_BATCH"
from hipengine.generation.qwen4_exp_gguf import (
    Qwen4ExpGGUFTextGenerator,
    Qwen4ExpResidentServingRunner,
)
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from tests._qwen4exp_ar_batch_fixtures import (
    FakeBatchRunner,
    FakeC1Runner,
    FakeTokenizer,
    make_generator,
)


@pytest.fixture(autouse=True)
def _request_owned_c1_path(monkeypatch):
    """Drive c1 fixtures; expose the real path for the explicit batch test."""
    original = Qwen4ExpResidentServingRunner._prepare_ar_batch
    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)
    return original


class _SegmentingC1Runner(FakeC1Runner):
    """A c1 runner whose prefill chunk takes ``segments_per_chunk`` segments.

    Mirrors the two pieces of the resident runner the serving runner reads: a
    ``_pending_prefill_append`` triple and a ``prefill_append_resumable`` that
    returns ``None`` until the chunk's bounded layer work is done. Prompt
    accounting is committed by the completing segment only, which is where the
    real runner returns from ``StopIteration``.
    """

    def __init__(self, *args, segments_per_chunk=1, fail_on_resume=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments_per_chunk = int(segments_per_chunk)
        self.fail_on_resume = fail_on_resume
        self._pending_prefill_append = None
        self._pending_remaining = 0
        self.resume_calls = []
        self.plain_calls = []
        self.pending_closed = 0

    def reset(self):
        if self._pending_prefill_append is not None:
            self.pending_closed += 1
            self._pending_prefill_append = None
        super().reset()

    def prefill_append(self, tokens, **kwargs):
        self.plain_calls.append(tuple(int(token) for token in tokens))
        return super().prefill_append(tokens, **kwargs)

    def prefill_append_resumable(self, tokens, *, time_budget_seconds=0.1, **kwargs):
        del time_budget_seconds
        tokens = tuple(int(token) for token in tokens)
        if self._pending_prefill_append is None:
            self._pending_prefill_append = (SimpleNamespace(), SimpleNamespace(), tokens)
            self._pending_remaining = self.segments_per_chunk
        expected = self._pending_prefill_append[2]
        if tokens != expected:
            raise ValueError("resumed prefill must name the same prompt chunk")
        self.resume_calls.append(tokens)
        if self.fail_on_resume is not None and len(self.resume_calls) > self.fail_on_resume:
            # The resident runner closes its suspended generator and clears the
            # pending tuple before re-raising; a fake that kept it would hide a
            # real defect behind its own leak.
            self._pending_prefill_append = None
            raise RuntimeError("simulated device fault inside the layer stack")
        self._pending_remaining -= 1
        if self._pending_remaining > 0:
            return None
        self._pending_prefill_append = None
        return super().prefill_append(tokens, **kwargs)


def _generator(monkeypatch, *, segments=1, capacity=2, fail_on_resume=None):
    """A generator whose whole staging pool segments prefill work."""

    created = []

    def factory(*args, **kwargs):
        del args, kwargs
        runner = _SegmentingC1Runner(
            segments_per_chunk=segments, fail_on_resume=fail_on_resume,
        )
        created.append(runner)
        return runner

    primary = _SegmentingC1Runner(
        segments_per_chunk=segments, fail_on_resume=fail_on_resume,
    )
    created.append(primary)
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=FakeTokenizer(), runner=primary,
        resident_capacity=capacity,
    )
    generator._resident = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner", factory
    )
    return generator, created


def _config(capacity, *, chunk, budget, decode_budget=None):
    return EngineLoopConfig(
        prefill_decode_policy="token_budget",
        max_active_requests=capacity,
        max_prefill_chunk_tokens=chunk,
        round_prefill_token_budget=budget,
        decode_prefill_token_budget=decode_budget,
        round_decode_row_budget=capacity,
    )


def _request(prompt, *, max_tokens):
    return GenerationRequest(
        prompts=(prompt,), max_tokens=max_tokens, temperature=0.0, top_p=1.0,
        ignore_eos=True,
    )


def _state(driver, request_id):
    return driver._loop.scheduler.active_batch.requests[request_id]


def _tokens(events, request_id):
    return [event.token_id for event in events if event.kind == "token"
            and event.request_id == request_id]


def _steps(runner):
    return [call for call in runner.calls if call[0] == "step"]


def _prefill_slices(native):
    return list(native.decode_telemetry()["recent_prefill_slices"])


def test_incomplete_segment_holds_prompt_accounting_and_the_decode_turn(monkeypatch):
    """A suspended chunk commits nothing and keeps the row out of decode."""

    generator, _ = _generator(monkeypatch, segments=2)
    driver = SubmitPollTextGenerator(
        generator, config=_config(2, chunk=4, budget=4),
    )
    native = generator._resident_model_runner
    assert isinstance(native, Qwen4ExpResidentServingRunner)
    try:
        peer = driver.submit_detailed(_request((1, 2), max_tokens=4))
        prompt = (11, 12, 13, 14, 15, 16)
        arrival = driver.submit_detailed(_request(prompt, max_tokens=2))
        peer_id = peer.request_ids[0]
        arrival_id = arrival.request_ids[0]

        # Round 1 completes the peer's one-chunk prompt and decodes it, so the
        # next round has a due decode row and takes the resumable path.
        events = list(driver._loop.tick())
        row = native._rows[arrival_id]
        runner = row.runner
        assert runner is not None

        events.extend(driver._loop.tick())
        assert runner._pending_prefill_append is not None, (
            "the resumable path did not suspend the arrival's first chunk"
        )
        state = _state(driver, arrival_id)
        assert state.next_prompt_index == 0, (
            "an incomplete segment advanced the scheduler prompt cursor"
        )
        assert state.remaining_prefill == len(prompt)
        assert row.prefill_tokens_seen == 0, (
            "an incomplete segment advanced the runner's prompt accounting"
        )
        assert row.next_result is None, (
            "an incomplete segment published a prefilled result"
        )
        assert _steps(runner) == [], (
            "the row decoded while its prefill chunk was still suspended"
        )
        assert _tokens(events, arrival_id) == []
        assert _tokens(events, peer_id) == [3, 4]

        # The next round resumes the identical chunk and completes it without
        # publishing a prompt token early or a second time.
        events = driver._loop.tick()
        assert runner.resume_calls == [(11, 12, 13, 14)] * 2
        assert row.prefill_tokens_seen == 4
        assert _state(driver, arrival_id).next_prompt_index == 4
        assert row.next_result is None
        assert _tokens(events, arrival_id) == []
    finally:
        driver.close()


def test_completing_segment_commits_its_chunk_and_publishes_once(monkeypatch):
    """Completion commits the prompt once and yields exactly one first token."""

    generator, _ = _generator(monkeypatch, segments=2)
    driver = SubmitPollTextGenerator(
        generator, config=_config(2, chunk=4, budget=4),
    )
    native = generator._resident_model_runner
    try:
        peer = driver.submit_detailed(_request((1, 2), max_tokens=6))
        prompt = (11, 12, 13, 14)
        arrival = driver.submit_detailed(_request(prompt, max_tokens=2))
        arrival_id = arrival.request_ids[0]

        driver._loop.tick()
        row = native._rows[arrival_id]
        runner = row.runner
        published = []
        for _ in range(6):
            published.extend(driver._loop.tick())
            if driver.generation_complete(arrival):
                break
        assert driver.generation_complete(arrival)
        # Two segments per chunk and no replay: the suspended chunk is resumed,
        # never restarted as a third device call.
        assert runner.resume_calls == [prompt, prompt]
        assert runner.plain_calls == []
        assert runner.prompt_ids == list(prompt), (
            "the prompt chunk was committed more than once"
        )
        assert _tokens(published, arrival_id) == [sum(prompt) % 50, sum(prompt) % 50 + 1], (
            "the completing segment published its first token zero or twice"
        )
        output = driver.take_result(arrival)[0]
        assert output.generated_token_ids == (sum(prompt) % 50, sum(prompt) % 50 + 1)
    finally:
        driver.close()


def test_decode_drain_resumes_the_suspended_chunk_before_the_wider_tail(monkeypatch):
    """After the decode row drains, the wider round resumes then finishes."""

    generator, _ = _generator(monkeypatch, segments=3)
    driver = SubmitPollTextGenerator(
        generator, config=_config(2, chunk=8, budget=8, decode_budget=4),
    )
    native = generator._resident_model_runner
    try:
        peer = driver.submit_detailed(_request((1, 2), max_tokens=2))
        prompt = tuple(range(11, 21))
        arrival = driver.submit_detailed(_request(prompt, max_tokens=2))
        arrival_id = arrival.request_ids[0]

        driver._loop.tick()
        row = native._rows[arrival_id]
        runner = row.runner

        # While the peer is still due, the operator's decode-time prefill share
        # caps the chunk at 4 tokens and the segment suspends inside it.
        driver._loop.tick()
        suspended = _prefill_slices(native)[-1]
        assert runner._pending_prefill_append is not None
        assert runner.resume_calls == [prompt[:4]]
        assert suspended["decode_rows"] == 1
        assert suspended["token_budget"] == 4
        assert suspended["tokens"] == 4
        assert suspended["committed_tokens"] == 0, (
            "a suspended segment reported committed prompt tokens"
        )
        assert _state(driver, arrival_id).next_prompt_index == 0
        # The round counter follows committed prompt tokens, so the repeated
        # planned 4-token segment does not move it.
        policy = driver._loop.observability_snapshot()["scheduler_policy"]
        assert policy["round_prefill_tokens"] == 2, (
            "the prefill token counter counted a planned but uncommitted chunk"
        )
        assert policy["round_prefill_tokens"] == sum(
            sample["committed_tokens"] for sample in _prefill_slices(native)
        )

        # The peer drains, so this round's budget widens to 8. The pending
        # 4-token chunk is resumed as the identical work, then the tail of the
        # wider chunk runs; the leftover budget starts no second chunk.
        before = len(_prefill_slices(native))
        driver._loop.tick()
        slices = _prefill_slices(native)
        assert len(slices) - before == 1, (
            "the round spent its leftover prefill budget on another chunk"
        )
        drained = slices[-1]
        assert runner.resume_calls == [prompt[:4]] * 3, (
            "the drained round did not resume the identical suspended chunk"
        )
        assert runner.plain_calls == [prompt[4:8]], (
            "the wider round's tail did not run after the resumed chunk"
        )
        assert drained["decode_rows"] == 0
        assert drained["token_budget"] == 8
        assert drained["tokens"] == 8
        assert drained["committed_tokens"] == 8
        assert row.prefill_tokens_seen == 8
        assert _state(driver, arrival_id).next_prompt_index == 8
        policy = driver._loop.observability_snapshot()["scheduler_policy"]
        assert policy["round_prefill_tokens"] == 10, (
            "the drained round did not count exactly its committed prompt tokens"
        )
        assert policy["round_prefill_tokens"] == sum(
            sample["committed_tokens"] for sample in slices
        )
    finally:
        driver.close()


def test_cancelled_suspended_prefill_closes_its_context_before_runner_reuse(monkeypatch):
    """Cancellation closes the suspended segment, and a peer keeps running."""

    generator, _ = _generator(monkeypatch, segments=2)
    driver = SubmitPollTextGenerator(
        generator, config=_config(2, chunk=4, budget=4),
    )
    native = generator._resident_model_runner
    try:
        peer = driver.submit_detailed(_request((1, 2), max_tokens=4))
        victim_prompt = (31, 32, 33, 34, 35, 36)
        victim = driver.submit_detailed(_request(victim_prompt, max_tokens=2))
        victim_id = victim.request_ids[0]
        peer_id = peer.request_ids[0]

        events = []
        events.extend(driver._loop.tick())
        events.extend(driver._loop.tick())
        row = native._rows[victim_id]
        runner = row.runner
        assert runner._pending_prefill_append is not None
        assert runner not in native._available, (
            "a suspended prefill released its staging runner"
        )

        driver.cancel_submission(victim)
        assert runner._pending_prefill_append is None, (
            "the cancelled row left its suspended prefill context open"
        )
        assert runner.pending_closed == 1
        assert runner in native._available
        assert victim_id not in native._rows

        # The freed staging runner serves the next arrival with its own prompt
        # and no residue from the cancelled chunk.
        late_prompt = (41, 42, 43)
        late = driver.submit_detailed(_request(late_prompt, max_tokens=2))
        late_id = late.request_ids[0]
        late_runner = None
        for _ in range(10):
            events.extend(driver._loop.tick())
            late_row = native._rows.get(late_id)
            if late_row is not None and late_row.runner is not None:
                late_runner = late_row.runner
            if driver.generation_complete(peer) and driver.generation_complete(late):
                break
        assert driver.generation_complete(peer), "the peer did not survive the cancel"
        assert _tokens(events, peer_id) == [3, 4, 5, 6]
        cancelled = driver.take_result(victim)[0]
        assert cancelled.generated_token_ids == ()
        assert cancelled.finish_details.cancelled
        assert driver.generation_complete(late)
        assert late_runner is runner, (
            "the freed staging runner was not reused"
        )
        assert runner.prompt_ids == list(late_prompt)
        assert runner.resume_calls[-2:] == [late_prompt, late_prompt]
        assert driver.take_result(late)[0].generated_token_ids == (
            sum(late_prompt) % 50, sum(late_prompt) % 50 + 1,
        )
    finally:
        driver.close()


def test_failed_segment_cannot_publish_prompt_progress(monkeypatch):
    """A segment that raises must leave its chunk uncommitted in the scheduler."""

    generator, _ = _generator(monkeypatch, segments=3, fail_on_resume=2)
    driver = SubmitPollTextGenerator(
        generator, config=_config(2, chunk=4, budget=4),
    )
    native = generator._resident_model_runner
    try:
        # A decoding peer keeps the round resumable; it is not under test here.
        driver.submit_detailed(_request((1, 2), max_tokens=4))
        prompt = (11, 12, 13, 14, 15, 16)
        arrival = driver.submit_detailed(_request(prompt, max_tokens=2))
        arrival_id = arrival.request_ids[0]

        driver._loop.tick()
        driver._loop.tick()
        row = native._rows[arrival_id]
        runner = row.runner
        assert runner._pending_prefill_append is not None
        driver._loop.tick()
        assert runner._pending_prefill_append is not None, (
            "the simulated failure fired before the chunk was still suspended"
        )
        with pytest.raises(GenerationExecutionFailed) as failed:
            driver._loop.tick()
        assert failed.value.phase == "prefill"
        assert failed.value.request_ids == (arrival_id,)
        # The runner closed the failed segment and committed no prompt token.
        assert runner._pending_prefill_append is None
        assert row.prefill_tokens_seen == 0
        assert row.next_result is None
        assert row.generated_ids == []
        assert native._outputs == {}
        # So the scheduler must not claim the failed chunk either.
        state = _state(driver, arrival_id)
        assert state.next_prompt_index == 0, (
            "a failed prefill segment left its chunk committed in the scheduler "
            f"prompt cursor (cursor={state.next_prompt_index}, "
            f"remaining_prefill={state.remaining_prefill}, "
            f"runner_seen={row.prefill_tokens_seen}); the next round then replays "
            "the wrong chunk and a later decode step refuses the row"
        )
    finally:
        driver.close()


def test_defer_prefill_work_skips_a_row_the_runner_retired():
    """A retired member of a prefill item is skipped; its peers still rewind."""

    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    scheduler.submit((1, 2, 3, 4, 5, 6), max_new_tokens=2, request_id=1)
    scheduler.submit((7, 8, 9, 10, 11, 12), max_new_tokens=2, request_id=2)
    assert scheduler.admit_pending() == (1, 2)
    work = scheduler.next_prefill_batch_work(chunk_size=4, max_rows=2)
    assert work is not None
    assert work.request_ids == (1, 2)
    assert scheduler.active_batch.requests[1].next_prompt_index == 4
    assert scheduler.active_batch.requests[2].next_prompt_index == 4

    # The runner retired the cancelled row before the engine defers the
    # segment, so the rewind must skip it instead of raising KeyError.
    assert scheduler.cancel(2, reason="cancel") is not None
    scheduler.defer_prefill_work(work, (1, 2))
    assert scheduler.active_batch.requests[1].next_prompt_index == 0
    assert 2 not in scheduler.active_batch.requests
    # An item whose only incomplete row was retired is a no-op.
    scheduler.defer_prefill_work(work, (2,))
    assert scheduler.active_batch.requests[1].next_prompt_index == 0


def test_prefill_budget_and_round_robin_rotation_survive_suspensions(monkeypatch):
    """Suspension does not break the operator budget or the prefill rotation."""

    generator, created = _generator(monkeypatch, segments=2, capacity=3)
    # This fixture specifically measures three concurrent c1 owners, rather
    # than the production two-runner staging pool used by shared decode.
    generator._staging_capacity = 3
    driver = SubmitPollTextGenerator(
        generator, config=_config(3, chunk=8, budget=4, decode_budget=4),
    )
    native = generator._resident_model_runner
    try:
        prompts = ((1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12), (13, 14, 15, 16, 17, 18))
        submissions = [driver.submit_detailed(_request(prompt, max_tokens=2))
                       for prompt in prompts]
        ids = [submission.request_ids[0] for submission in submissions]
        owners = {request_id: None for request_id in ids}
        for _ in range(40):
            driver._loop.tick()
            for request_id in ids:
                row = native._rows.get(request_id)
                if row is None:
                    continue
                assert row.runner is not None
                if owners[request_id] is None:
                    owners[request_id] = row.runner
                else:
                    assert owners[request_id] is row.runner, (
                        "a suspended row changed staging-runner owner mid-prompt"
                    )
            if all(driver.generation_complete(item) for item in submissions):
                break
        assert all(driver.generation_complete(item) for item in submissions)

        slices = _prefill_slices(native)
        assert slices
        for sample in slices:
            assert sample["tokens"] <= sample["token_budget"]
            assert sample["committed_tokens"] <= sample["tokens"]
            assert sample["decode_rows"] <= 3
        # One prefill work item per round, rotating over the three owners: no
        # row takes two turns in a row, and no row waits longer than one full
        # rotation for its next chunk.
        rotation = [sample["request_ids"][0] for sample in slices]
        assert set(rotation) == set(ids)
        assert all(first != second for first, second in zip(rotation, rotation[1:])), (
            f"prefill rotation repeated an owner: {rotation}"
        )
        for request_id in ids:
            turns = [index for index, picked in enumerate(rotation) if picked == request_id]
            assert len(turns) >= 2, f"request {request_id} got only one prefill turn"
            assert all(later - earlier <= len(ids)
                       for earlier, later in zip(turns, turns[1:])), (
                f"request {request_id} waited past a full rotation: {rotation}"
            )
        # Each staging runner only ever resumed its own row's chunk.
        resumable = 0
        for request_id, prompt in zip(ids, prompts):
            runner = owners[request_id]
            if runner.resume_calls:
                resumable += 1
            for chunk in runner.resume_calls:
                assert any(
                    prompt[start:start + len(chunk)] == chunk
                    for start in range(len(prompt))
                ), (
                    "a staging runner resumed a chunk that is not its own row's prompt"
                )
            for other_id in ids:
                if other_id != request_id:
                    assert owners[other_id] is not runner
        assert resumable >= 1, "no row took the resumable path"
        assert len(created) == 3
    finally:
        driver.close()


def test_ar_batch_imports_a_resumed_prefill_exactly_once(monkeypatch, _request_owned_c1_path):
    """The default shared-batch path imports the completed prefill once."""
    monkeypatch.setattr(Qwen4ExpResidentServingRunner, "_prepare_ar_batch", _request_owned_c1_path)

    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_batch_runner.Qwen4ExpGGUFBatchRunner",
        FakeBatchRunner,
    )
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: _SegmentingC1Runner(segments_per_chunk=2),
    )
    generator = make_generator()
    generator.runner.segments_per_chunk = 2
    native = generator.create_resident_model_runner(capacity=2)
    assert isinstance(native, Qwen4ExpResidentServingRunner)
    native.prepare()
    assert native.decode_telemetry()["decode_path"] == "qwen4exp_full_model_ar_batch"
    prompt = (1, 2, 3, 4)
    work = WorkItem(
        kind=WorkKind.PREFILL, request_ids=(1,), row_to_request=(1,),
        token_rows=(prompt,),
    )
    try:
        native.register_batch((1,), _request(prompt, max_tokens=2), prompt_rows=(prompt,))
        native.reserve_admission(SimpleNamespace(request_id=1))
        row = native._row(1)
        staging = row.runner
        batch = native._ar_batch.runner

        assert native.prefill_batch_resumable(work, commit=True) == (1,)
        assert staging._pending_prefill_append is not None
        assert staging not in native._available, (
            "a suspended prefill handed its staging runner to the batch pool"
        )
        assert batch.import_calls == []
        assert native._ar_batch.owns(1)
        assert row.next_result is None

        assert native.prefill_batch_resumable(work, commit=True) == ()
        assert batch.import_calls == [(0, 1, 4)], (
            "the completed prefill was imported zero or more than once"
        )
        assert row.next_result is not None
        assert row.runner is None
        assert native._available.count(staging) == 1
        assert staging._pending_prefill_append is None

        decode = WorkItem(
            kind=WorkKind.DECODE, request_ids=(1,), row_to_request=(1,),
            token_rows=((0,),),
        )
        first = native.decode_batch(decode, commit=True)
        second = native.decode_batch(decode, commit=True)
        assert [token.token_id for token in first] == [sum(prompt) % 50]
        assert [token.token_id for token in second] == [
            sum(prompt) % 50 + FakeBatchRunner.token_bias
        ]
        assert second[0].finished
        assert len(batch.step_calls) == 1, (
            "the completed prefill took more than one shared batch step"
        )
    finally:
        native.close()


def test_resident_runner_closes_a_failed_segment_and_reissues_cleanly(monkeypatch):
    """The real suspended segment closes on failure and retries from scratch."""

    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.position = 0
    attempts = []
    closed = []

    def steps(tokens, **kwargs):
        del kwargs
        attempts.append(tuple(tokens))
        try:
            yield None
            if len(attempts) == 1:
                raise RuntimeError("simulated device fault inside the layer stack")
            return SimpleNamespace(token_id=7)
        finally:
            closed.append(tuple(tokens))

    runner._prefill_chunked_steps = steps
    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_runner.time.perf_counter", lambda: 0.0
    )

    with pytest.raises(RuntimeError, match="simulated device fault"):
        runner.prefill_append_resumable((1, 2))
    assert closed == [(1, 2)], "the failed segment's generator was left open"
    assert runner._pending_prefill_append is None
    # The chunk can be issued again from a clean generator.
    result = runner.prefill_append_resumable((1, 2))
    assert result.token_id == 7
    assert attempts == [(1, 2), (1, 2)]
    assert runner._pending_prefill_append is None


def test_reset_closes_a_suspended_prefill_before_the_runner_is_reused(monkeypatch):
    """The resident runner's own reset() closes a suspended segment."""

    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.state = SimpleNamespace(zero=lambda: None)
    runner.attention_states = []
    runner.index_states = []
    runner.runtime = SimpleNamespace()
    runner._ple_hash_states = {}
    runner.position = 0
    runner.calls = []
    closed = []

    def steps(tokens, **kwargs):
        del kwargs
        try:
            for layer in range(3):
                runner.calls.append((tuple(tokens), layer))
                yield None
        finally:
            closed.append(tuple(tokens))
        return SimpleNamespace(token_id=7)

    runner._prefill_chunked_steps = steps
    clock = iter(range(0, 64))
    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_runner.time.perf_counter", lambda: next(clock)
    )

    assert runner.prefill_append_resumable((1, 2)) is None
    assert runner.calls == [((1, 2), 0)]
    assert runner._pending_prefill_append is not None

    runner.reset()
    assert closed == [(1, 2)], "reset() left the suspended prefill context open"
    assert runner._pending_prefill_append is None

    # A reused runner starts a new chunk instead of resuming the old prompt.
    assert runner.prefill_append_resumable((5, 6)) is None
    assert runner.calls[-1] == ((5, 6), 0)
    assert runner.prefill_append_resumable((5, 6)) is None
    assert runner.calls[-1] == ((5, 6), 1)
    assert runner.prefill_append_resumable((5, 6)) is None
    assert runner.calls[-1] == ((5, 6), 2)
    result = runner.prefill_append_resumable((5, 6))
    assert result.token_id == 7
    assert runner._pending_prefill_append is None
