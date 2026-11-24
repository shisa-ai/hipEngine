"""DMS prefill observes cancellation intent without acknowledging reclamation."""
from types import SimpleNamespace

import pytest

from hipengine.generation.deadline import (
    GenerationCancellationToken, GenerationCancelled, GenerationDeadlineExceeded,
    raise_if_generation_stop_requested,
)
from hipengine.generation.registry import FinishDetails
from hipengine.generation import qwen35_gguf_dms
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession


@pytest.mark.parametrize('native', ['greedy', 'sampled'])
@pytest.mark.parametrize('reason', ['cancelled', 'deadline_exceeded'])
def test_dms_adapter_binds_requested_cancellation_to_its_session(native, reason):
    token = GenerationCancellationToken()
    dispatched = []
    token.set_cancel_dispatch(dispatched.append)
    session = SimpleNamespace(prefill_control=None)
    request = SimpleNamespace(cancellation_token=token, deadline_at=None)
    row = SimpleNamespace(request=request, prompt_ids=(1, 2), prefill_tokens_seen=0,
                          native_greedy=native == 'greedy', native_sampled=native == 'sampled',
                          lease=None)
    runner = object.__new__(qwen35_gguf_dms.Qwen35GGUFDMSModelRunner)
    runner.generator = SimpleNamespace(target_arch=None, dms_metadata_path='dms.json')
    runner._row = lambda request_id: row
    runner._acquire_lease = lambda: SimpleNamespace(session=session)
    runner._execution_stepped_request_ids = set()
    runner._execution_committed_request_ids = set()
    runner._execution_completed_prefill_request_ids = set()
    runner._begin_execution_step = lambda: None
    details = FinishDetails(reason=reason, cancelled=reason == 'cancelled',
                            deadline_exceeded=reason == 'deadline_exceeded')

    def prefill(actual_row):
        assert actual_row is row
        token.cancel(details)
        assert dispatched and token.cancel_requested and not token.cancelled
        session.prefill_control()
        pytest.fail('cancel intent must interrupt the in-flight prefill')

    runner._prefill_dms_row_controlled = prefill
    work = SimpleNamespace(request_ids=(7,), token_rows=((1, 2),))
    with pytest.raises(GenerationCancelled) as exc:
        runner.prefill_batch(work, commit=True)
    assert exc.value.finish_details.reason == reason
    assert not token.cancelled  # Scheduler, not the checkpoint, owns acknowledgement.
    assert session.prefill_control is None


def test_controlled_prefill_boundary_quiesces_before_observing_intent():
    events = []
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = SimpleNamespace(stream_synchronize=lambda stream: events.append(('sync', stream)))

    def check():
        events.append(('check',))
        raise GenerationCancelled()

    session.prefill_control = check
    with pytest.raises(GenerationCancelled):
        session._check_prefill_control(stream=3)
    assert events == [('sync', 3), ('check',)]


def test_uncontrolled_prefill_does_not_add_synchronization():
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = SimpleNamespace(stream_synchronize=lambda stream: pytest.fail('unexpected drain'))
    session.prefill_control = None
    session._check_prefill_control(stream=0)


def test_prefill_control_observes_an_absolute_deadline():
    with pytest.raises(GenerationDeadlineExceeded):
        raise_if_generation_stop_requested(SimpleNamespace(deadline_at=0, cancellation_token=None))


@pytest.mark.parametrize('chunk_outer', [False, True])
@pytest.mark.parametrize('cancel_after', [1, 3, 5])
@pytest.mark.parametrize('stop_kind', ['cancel', 'deadline', 'sync_failure'])
def test_real_bulk_loop_stops_between_existing_chunks_and_drains_before_cleanup(
    monkeypatch, chunk_outer, cancel_after, stop_kind,
):
    from hipengine.runtime import qwen35_gguf_runner as runtime_module

    from hipengine.generation import deadline as deadline_module

    calls = []
    drained = []
    token = GenerationCancellationToken()
    request = SimpleNamespace(deadline_at=2 if stop_kind == 'deadline' else None,
                              cancellation_token=token)
    monkeypatch.setattr(deadline_module, 'time', SimpleNamespace(
        perf_counter=lambda: 1 if len(calls) < cancel_after else 3,
    ))
    session = object.__new__(Qwen35GGUFResidentSession)

    def synchronize(stream):
        drained.append(len(calls))
        if stop_kind == 'sync_failure' and len(calls) >= cancel_after:
            raise RuntimeError('device did not quiesce')

    session.runtime = SimpleNamespace(stream_synchronize=synchronize)

    def compute(layer_id, src, dst, scratch, **kwargs):
        calls.append((layer_id, kwargs.get('rows', scratch.rows)))
        if len(calls) == cancel_after and stop_kind == 'cancel':
            token.cancel()

    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=SimpleNamespace(
            layer_types=(runtime_module.LINEAR_ATTENTION, runtime_module.FULL_ATTENTION),
            ssm_conv_kernel=2, is_moe=False,
        )),
        hidden_size=8, vocab_size=100,
        _run_linear_attention_prefill_layer_rows=compute,
        _run_full_attention_prefill_layer_aotriton=compute,
    )
    session.scratch = SimpleNamespace(max_positions=12, cos_table_buf=SimpleNamespace(ptr=0),
                                      sin_table_buf=SimpleNamespace(ptr=0))
    session._prefill_token_buf = SimpleNamespace(ptr=0x1000)
    session._prefill_hidden_a = SimpleNamespace(ptr=0x2000, nbytes=(4 if chunk_outer else 12) * 8 * 2)
    session._prefill_hidden_b = SimpleNamespace(ptr=0x3000)
    session._bulk_prefill_scratch = SimpleNamespace(
        rows=4, for_chunk=lambda start, rows, **kwargs: SimpleNamespace(
            rows=rows, norm=SimpleNamespace(ptr=0x4000),
        ),
    )
    session._runtime_state_library = None
    # Storage is sized before routing; this fixture fixes the owner's capacity.
    session._ensure_bulk_prefill_hidden_capacity = (
        lambda rows, **kwargs: 4 if chunk_outer else 12)
    session._prefill_scratch_rows = lambda rows: 4
    session._linear_prefill_layer_chunk_size = lambda rows: 4
    session._full_attention_prefill_layer_chunk_size = lambda rows: 4
    session._full_attention_prefill_scratch_for_layer = lambda scratch, layer: scratch
    session._prefill_aotriton_bridge_for_rows = lambda rows: None
    session._copy_token_embeddings_to_device = lambda *args, **kwargs: None
    session.reset = lambda **kwargs: None
    session._release_int8_prefill_oracle_buffers = lambda: drained.append(('release', len(calls)))
    session.prefill_control = lambda: raise_if_generation_stop_requested(request)
    expected = {'cancel': GenerationCancelled, 'deadline': GenerationDeadlineExceeded,
                'sync_failure': RuntimeError}[stop_kind]
    with pytest.raises(expected):
        session._run_bulk_prefill_and_sample([1] * 12)
    assert len(calls) == cancel_after
    if stop_kind == 'sync_failure':
        assert ('release', cancel_after) not in drained
        assert session._deferred_prefill_cleanup
        # Failed drain must retain owners. A later successful drain permits
        # cleanup exactly once, without pretending the original request worked.
        session.runtime.stream_synchronize = lambda stream: drained.append(('recovered', stream))
        session._drain_deferred_prefill_cleanup()
        assert drained[-1] == ('release', cancel_after)
        assert not session._deferred_prefill_cleanup
    else:
        assert drained[-2:] == [cancel_after, ('release', cancel_after)]


@pytest.mark.parametrize('sampled', [False, True])
def test_legacy_dms_route_checks_a_deadline_that_expires_during_prefill(monkeypatch, sampled):
    from hipengine.generation import deadline as deadline_module
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    now = [1]
    monkeypatch.setattr(deadline_module, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = SimpleNamespace(stream_synchronize=lambda stream: None)

    def prefill(*args, **kwargs):
        now[0] = 3
        session._check_prefill_control()
        pytest.fail('legacy DMS route lost its deadline')

    session.prefill = prefill
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._prefix_cache_mode = 'off'
    runner._acquire_lease = lambda: SimpleNamespace(session=session)
    runner._begin_mtp2_prompt_streaming = lambda rows: (None,)
    runner._finish_mtp2_prompt_streaming = lambda *args, **kwargs: None
    runner._prepare_sampled_prefill = lambda row: None
    request = SimpleNamespace(deadline_at=2, max_tokens=8, cancellation_token=None)
    row = SimpleNamespace(request=request, lease=None, prompt_ids=(1, 2), request_id=7,
                          mtp2_candidate_budget=0, native_sampled=sampled)
    with pytest.raises(GenerationDeadlineExceeded):
        runner._prefill_dms_row(row)
    assert session.prefill_control is None


@pytest.mark.parametrize('cooperative', [False, True])
def test_resident_admission_retains_deadlines_only_for_cooperative_runners(monkeypatch, cooperative):
    from dataclasses import replace
    from hipengine.generation import SubmitPollTextGenerator
    from hipengine.generation import deadline as deadline_module
    from tests.test_unit_generation_execution_failure_containment import (
        _ContainingInner, _service_request,
    )

    now = [1]
    monkeypatch.setattr(deadline_module, 'time', SimpleNamespace(perf_counter=lambda: now[0]))
    inner = _ContainingInner()
    runner = inner.runner
    runner.supports_cooperative_prefill_deadlines = cooperative
    registered = []
    original_register = runner.register_batch

    def register(ids, request, **kwargs):
        registered.append(request)
        original_register(ids, request, **kwargs)

    runner.register_batch = register

    def prefill(work, **kwargs):
        now[0] = 3
        raise_if_generation_stop_requested(registered[-1])

    runner.prefill_batch = prefill
    driver = SubmitPollTextGenerator(inner, capacity=3, prefill_chunk_size=4)
    try:
        driver.submit_detailed(replace(_service_request(1), deadline_at=2))
        assert registered[-1].deadline_at == (2 if cooperative else None)
        events = driver.poll(max_ticks=1)
        failures = [event for event in events if event.kind == 'failed']
        if cooperative:
            assert failures and isinstance(failures[0].error, GenerationDeadlineExceeded)
        else:
            assert not failures  # Dense shared work does not raise one row's deadline.
    finally:
        driver.close()


def test_failed_drain_retains_nested_capture_owners_until_session_teardown_can_quiesce():
    session = object.__new__(Qwen35GGUFResidentSession)
    closed = []

    def fail(stream):
        raise RuntimeError('device did not quiesce')

    session.runtime = SimpleNamespace(stream_synchronize=fail)
    session.prefill_control = lambda: None
    capture = SimpleNamespace(close=lambda: closed.append('capture'))
    backend = SimpleNamespace(close=lambda: closed.append('backend'))
    with pytest.raises(RuntimeError, match='did not quiesce'):
        with session._closing_prefill_resource(capture):
            try:
                session._check_prefill_control(stream=7)
            finally:
                session._cleanup_prefill_resource(backend.close, stream=7)
    assert not closed
    assert len(session._deferred_prefill_cleanup) == 2
    with pytest.raises(RuntimeError, match='did not quiesce'):
        session.close()
    assert not closed  # Teardown cannot free other session state either.
    session.runtime.stream_synchronize = lambda stream: closed.append(('sync', stream))
    session._drain_deferred_prefill_cleanup()
    assert closed == [('sync', 7), 'backend', 'capture']
    session._drain_deferred_prefill_cleanup()
    assert closed == [('sync', 7), 'backend', 'capture']