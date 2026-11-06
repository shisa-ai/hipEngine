"""Pin the existing legacy MTP chunk policy without opening a device session."""

from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf


def test_verifier_width_does_not_inherit_prefill_or_warmup_limit(monkeypatch):
    monkeypatch.setattr(qwen35_gguf, "_MTP_SERVING_PREFILL_MAX_SLOTS", 2)
    monkeypatch.setattr(qwen35_gguf, "_MTP_WARMUP_BATCH_MAX_SLOTS", 2)
    generator = _generator()
    calls = []
    monkeypatch.setattr(generator, "_try_verify_mtp_serving_cycle_chunks_streams", lambda chunks: None)

    def verify(chunk, **kwargs):
        calls.append(len(chunk))
        return list(chunk)

    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)
    assert generator._try_verify_mtp_serving_cycles_batch(list(range(7))) == list(range(7))
    assert calls == [4, 3]


@pytest.mark.parametrize("request_owned", [False, True])
@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_stream_draft_failure_frees_successful_sibling_snapshots(monkeypatch, failure_index, request_owned):
    generator = _generator()
    slots = [SimpleNamespace(
        request_id=index, draft_stream=0, generated_ids=[1], timing={},
        resident_context=SimpleNamespace(pending_seed=object()),
        session=SimpleNamespace(runtime=SimpleNamespace(
            stream_create=lambda **kwargs: 700,
            stream_synchronize=lambda stream: None,
            stream_destroy=lambda stream: None,
        )),
    ) for index in range(3)]
    completed = []
    freed = []

    def draft(slot, *args, **kwargs):
        completed.append(slot.request_id)
        if slot.request_id == failure_index:
            raise RuntimeError("draft failed")
        return SimpleNamespace(slot=slot, snapshot=object())

    monkeypatch.setattr(generator, "_draft_mtp_serving_slot", draft)
    monkeypatch.setattr(generator, "_free_mtp_serving_cycle_snapshot", lambda result: freed.append(result.slot.request_id))
    if request_owned:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=3)
        for slot in slots:
            slot.worker_pool = pool
    try:
        with pytest.raises(RuntimeError, match="draft failed"):
            generator._try_draft_mtp_serving_slots_streams(
                slots, object(), SimpleNamespace(max_tokens=4, deadline_at=None), base_env={},
            )
    finally:
        if request_owned:
            pool.shutdown(wait=True)
    assert sorted(completed) == [0, 1, 2]
    assert sorted(freed) == [index for index in range(3) if index != failure_index]


@pytest.mark.parametrize("submission_failure", [False, True])
def test_stream_draft_cleanup_continues_and_preserves_original_failure(monkeypatch, submission_failure):
    generator = _generator()
    slots = [SimpleNamespace(
        request_id=index, draft_stream=700, generated_ids=[1], timing={},
        resident_context=SimpleNamespace(pending_seed=object()),
        session=SimpleNamespace(runtime=SimpleNamespace(
            stream_create=lambda **kwargs: 700,
            stream_synchronize=lambda stream: None,
            stream_destroy=lambda stream: None,
        )),
    ) for index in range(3)]
    freed = []
    real_pool = qwen35_gguf.concurrent.futures.ThreadPoolExecutor

    class Pool(real_pool):
        def submit(self, fn, slot):
            if submission_failure and slot.request_id == 2:
                raise RuntimeError("submit failed")
            return super().submit(fn, slot)

    def draft(slot, *args, **kwargs):
        if not submission_failure and slot.request_id == 2:
            raise RuntimeError("worker failed")
        return SimpleNamespace(slot=slot)

    def free(result):
        freed.append(result.slot.request_id)
        if result.slot.request_id == 0:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(qwen35_gguf.concurrent.futures, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(generator, "_draft_mtp_serving_slot", draft)
    monkeypatch.setattr(generator, "_free_mtp_serving_cycle_snapshot", free)
    with pytest.raises(RuntimeError, match="submit failed" if submission_failure else "worker failed") as error:
        generator._try_draft_mtp_serving_slots_streams(
            slots, object(), SimpleNamespace(max_tokens=4, deadline_at=None), base_env={},
        )
    assert freed == [0, 1]
    assert error.value.__notes__ == ["draft snapshot cleanup failed: cleanup failed"]


@pytest.mark.parametrize("failure_index", [1, 2])
def test_verifier_snapshot_acquisition_failure_releases_earlier_owners(monkeypatch, failure_index):
    generator = _generator()
    freed = []
    cycles = []
    for index in range(3):
        def snapshot(index=index):
            if index == failure_index:
                raise RuntimeError("snapshot failed")
            return index
        cycles.append(SimpleNamespace(
            slot=SimpleNamespace(session=SimpleNamespace(_linear_state_snapshot=snapshot), timing={}),
            direct_commit_exact=False, snapshot=None,
        ))
    monkeypatch.setattr(generator, "_free_mtp_serving_cycle_snapshot", lambda cycle: freed.append(cycle.snapshot) if cycle.snapshot is not None else None)
    with pytest.raises(RuntimeError, match="snapshot failed"):
        generator._verify_mtp_serving_cycles(cycles, object())
    assert freed == list(range(failure_index))


def test_verifier_failure_releases_all_snapshots_despite_cleanup_error(monkeypatch):
    generator = _generator()
    freed = []
    cycles = [SimpleNamespace(
        slot=SimpleNamespace(session=SimpleNamespace(_linear_state_snapshot=lambda: object()), timing={}),
        direct_commit_exact=False, snapshot=None,
    ) for _ in range(3)]

    def fail(*args, **kwargs):
        raise RuntimeError("verify failed")

    def free(cycle):
        freed.append(cycle)
        if cycle is cycles[0]:
            raise RuntimeError("release failed")

    monkeypatch.setattr(generator, "_try_verify_mtp_serving_cycles_batch", fail)
    monkeypatch.setattr(generator, "_free_mtp_serving_cycle_snapshot", free)
    with pytest.raises(RuntimeError, match="verify failed") as error:
        generator._verify_mtp_serving_cycles(cycles, object())
    assert len(freed) == len(cycles)
    assert error.value.__notes__ == ["verifier snapshot cleanup failed: release failed"]


def _generator():
    return object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 8])
@pytest.mark.parametrize("precondition", ["ready", "tail", "seed", "runtime"])
def test_stream_draft_keeps_singleton_tail_seed_and_runtime_fallback(
    monkeypatch, count, precondition,
):
    generator = _generator()
    calls = []
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT", "0")
    slots = []
    for index in range(count):
        runtime = SimpleNamespace(
            stream_create=lambda **kwargs: 700,
            stream_synchronize=lambda stream: None,
            stream_destroy=lambda stream: None,
        )
        if precondition == "runtime" and index == count - 1:
            runtime.stream_synchronize = None
        slots.append(SimpleNamespace(
            request_id=index, draft_stream=0, generated_ids=([1, 2] if precondition == "tail" else [1]),
            resident_context=SimpleNamespace(pending_seed=None if precondition == "seed" else object()),
            session=SimpleNamespace(runtime=runtime), timing={},
        ))

    def draft(slot, assets, request, **kwargs):
        calls.append((slot.request_id, kwargs["draft_stream"]))
        return slot.request_id

    generator._draft_mtp_serving_slot = draft
    request = SimpleNamespace(max_tokens=3, deadline_at=None)
    result = generator._try_draft_mtp_serving_slots_streams(slots, object(), request, base_env={})

    ready = count > 1 and precondition == "ready"
    if ready:
        assert result == list(range(count))
        assert sorted(calls) == [(index, 700) for index in range(count)]
    else:
        assert result is None
        assert calls == []
        assert all(slot.draft_stream == 0 for slot in slots)


@pytest.mark.parametrize(
    "count,widths",
    [
        (0, ()),
        (1, ()),
        (2, (2,)),
        (3, (3,)),
        (4, (4,)),
        (5, (3, 2)),
        (6, (4, 2)),
        (7, (4, 3)),
        (8, (4, 4)),
        (9, (4, 3, 2)),
        (13, (4, 4, 3, 2)),
    ],
)
def test_verifier_chunks_preserve_order_without_singleton_tail(monkeypatch, count, widths):
    generator = _generator()
    calls = []
    owner = object()
    monkeypatch.setattr(generator, "_try_verify_mtp_serving_cycle_chunks_streams", lambda chunks: None)

    def verify(chunk, *, verify_owner_session):
        assert verify_owner_session is owner
        calls.append(tuple(chunk))
        return list(chunk)

    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)
    cycles = list(range(count))
    result = generator._try_verify_mtp_serving_cycles_batch(cycles, verify_owner_session=owner)

    assert tuple(map(len, calls)) == widths
    assert [row for chunk in calls for row in chunk] == (cycles if count > 1 else [])
    assert result == (cycles if count > 1 else None)


@pytest.mark.parametrize("stream_available", [False, True])
def test_verifier_uses_stream_chunks_or_serial_capability_fallback(monkeypatch, stream_available):
    generator = _generator()
    streamed = []
    serial = []
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY", "0")

    def streams(chunks):
        streamed.append([list(chunk) for chunk in chunks])
        return list(range(9)) if stream_available else None

    def verify(chunk, *, verify_owner_session):
        serial.append(list(chunk))
        return list(chunk)

    monkeypatch.setattr(generator, "_try_verify_mtp_serving_cycle_chunks_streams", streams)
    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)

    assert generator._try_verify_mtp_serving_cycles_batch(list(range(9))) == list(range(9))
    expected = [list(range(4)), list(range(4, 7)), list(range(7, 9))]
    assert streamed == [expected]
    assert serial == ([] if stream_available else expected)


@pytest.mark.parametrize("failure_index", [0, 1])
def test_verifier_declines_only_before_any_chunk_advances(monkeypatch, failure_index):
    generator = _generator()
    calls = []
    monkeypatch.setattr(generator, "_try_verify_mtp_serving_cycle_chunks_streams", lambda chunks: None)

    def verify(chunk, *, verify_owner_session):
        index = len(calls)
        calls.append(list(chunk))
        return None if index == failure_index else list(chunk)

    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)
    if failure_index:
        with pytest.raises(RuntimeError, match="prior chunk advanced state"):
            generator._try_verify_mtp_serving_cycles_batch(list(range(8)))
    else:
        assert generator._try_verify_mtp_serving_cycles_batch(list(range(8))) is None
    assert len(calls) == failure_index + 1


@pytest.mark.parametrize("count", [5, 7, 8, 9])
@pytest.mark.parametrize("capable", [False, True])
def test_stream_verifier_uses_capability_fallback_before_any_submission(monkeypatch, count, capable):
    generator = _generator()
    calls = []
    streams = []
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY", "0")
    cycles = []
    for index in range(count):
        runtime = SimpleNamespace(
            stream_create=lambda **kwargs: 900,
            stream_synchronize=lambda stream: None,
            stream_destroy=lambda stream: None,
        )
        if not capable:
            runtime.stream_synchronize = None
        cycles.append(SimpleNamespace(slot=SimpleNamespace(
            request_id=index, verify_stream=0, timing={},
            session=SimpleNamespace(runtime=runtime, verify_target_blocks_batch=lambda jobs: None),
        )))

    def verify(chunk, *, stream=0, verify_owner_session=None):
        ids = [cycle.slot.request_id for cycle in chunk]
        calls.extend(ids)
        streams.append(stream)
        return ids

    generator._verify_mtp_serving_cycle_chunk = verify
    result = generator._try_verify_mtp_serving_cycles_batch(cycles)

    assert result == list(range(count))
    assert sorted(calls) == list(range(count))
    assert all(stream != 0 for stream in streams) is capable
    assert all(stream == 0 for stream in streams) is (not capable)


@pytest.mark.parametrize("failure", ["none", "exception"])
def test_stream_verifier_drains_launched_chunks_without_serial_resubmission(monkeypatch, failure):
    generator = _generator()
    calls = []
    cycles = []
    for index in range(8):
        runtime = SimpleNamespace(stream_create=lambda **kwargs: 900,
                                  stream_synchronize=lambda stream: None,
                                  stream_destroy=lambda stream: None)
        cycles.append(SimpleNamespace(slot=SimpleNamespace(
            request_id=index, verify_stream=0, timing={},
            session=SimpleNamespace(runtime=runtime, verify_target_blocks_batch=lambda jobs: None),
        )))

    def verify(chunk, *, stream=0, verify_owner_session=None):
        assert stream != 0, "launched chunks must not replay serially"
        ids = [cycle.slot.request_id for cycle in chunk]
        calls.extend(ids)
        if ids[0] == 0:
            if failure == "exception":
                raise RuntimeError("injected chunk result failure")
            return None
        return ids

    generator._verify_mtp_serving_cycle_chunk = verify
    with pytest.raises(RuntimeError, match="after launch|injected chunk"):
        generator._try_verify_mtp_serving_cycles_batch(cycles)
    assert sorted(calls) == list(range(8)), "every launched future must drain once"


@pytest.mark.parametrize("request_owned", [False, True])
@pytest.mark.parametrize("submission_failure", [False, True])
def test_stream_verifier_collects_every_submitted_future(monkeypatch, submission_failure, request_owned):
    generator = _generator()
    collected = []
    runtime = SimpleNamespace(stream_create=lambda **kwargs: 900,
                              stream_synchronize=lambda stream: None,
                              stream_destroy=lambda stream: None)
    chunks = [[SimpleNamespace(slot=SimpleNamespace(
        verify_stream=0, timing={}, session=SimpleNamespace(
            runtime=runtime, verify_target_blocks_batch=lambda jobs: None
        ),
    ))] for _ in range(3)]

    class Future:
        def __init__(self, index):
            self.index = index

        def result(self):
            collected.append(self.index)
            raise RuntimeError(f"worker {self.index}")

    class Pool:
        def __init__(self, **kwargs):
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args):
            index = self.count
            self.count += 1
            if submission_failure and index == 2:
                raise RuntimeError("submit failed")
            return Future(index)

    monkeypatch.setattr(qwen35_gguf.concurrent.futures, "ThreadPoolExecutor", Pool)
    if request_owned:
        pool = Pool()
        for chunk in chunks:
            chunk[0].slot.worker_pool = pool
    with pytest.raises(RuntimeError, match="submit failed" if submission_failure else "worker 0"):
        generator._try_verify_mtp_serving_cycle_chunks_streams(chunks)
    assert collected == ([0, 1] if submission_failure else [0, 1, 2])


@pytest.mark.parametrize("count", [5, 7, 8, 13])
def test_prefill_above_cap_declines_before_acquiring_slots(monkeypatch, count):
    generator = _generator()
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL", "0")

    def unexpected(*args, **kwargs):
        pytest.fail("the over-cap packed route must decline before acquiring sessions")

    monkeypatch.setattr(generator, "_acquire_shared_session", unexpected)
    assert generator._try_open_mtp_serving_slots_batch_prefill(
        None,
        None,
        {},
        SimpleNamespace(prompts=tuple(range(count))),
        pool_sessions=True,
    ) is None


@pytest.mark.parametrize("count", [2, 3, 4])
def test_prefill_capability_refusal_closes_every_acquired_owner(monkeypatch, count):
    generator = _generator()
    sessions = []
    drafts = []
    closed = []
    released = []
    calls = []
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL", "0")

    class Session:
        runtime = object()

        def close(self):
            closed.append(self)

        def prefill_batch_native(self, prompts, **kwargs):
            calls.append((prompts, kwargs))
            raise NotImplementedError("unsupported slab")

    def acquire_session(*args, **kwargs):
        session = Session()
        sessions.append(session)
        return session, None, False

    def acquire_draft(*args, **kwargs):
        draft = object()
        drafts.append(draft)
        return draft, None, False

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setattr(generator, "_acquire_shared_session", acquire_session)
    monkeypatch.setattr(generator, "_acquire_mtp_draft_runner", acquire_draft)
    monkeypatch.setattr(
        generator, "_release_mtp_draft_runner",
        lambda key, draft: released.append(draft),
    )
    prompts = {index: list(range(index + 2)) for index in range(count)}
    assert generator._try_open_mtp_serving_slots_batch_prefill(
        None, None, prompts,
        SimpleNamespace(prompts=tuple(prompts), deadline_at=None),
        pool_sessions=True,
    ) is None
    assert len(sessions) == count
    assert closed == sessions[::-1]
    assert released == drafts[::-1]
    assert len(calls) == 1
    assert calls[0][0] == [tuple(prompts[index]) for index in range(count)]
    assert calls[0][1]["return_hidden_seeds"] is True


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_prefill_singleton_or_missing_hook_declines_before_acquisition(monkeypatch, count):
    generator = _generator()
    monkeypatch.setenv("HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL", "false")
    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", SimpleNamespace())

    def unexpected(*args, **kwargs):
        pytest.fail("an unavailable packed route must not acquire sessions")

    monkeypatch.setattr(generator, "_acquire_shared_session", unexpected)
    assert generator._try_open_mtp_serving_slots_batch_prefill(
        None, None, {}, SimpleNamespace(prompts=tuple(range(count))),
        pool_sessions=True,
    ) is None
