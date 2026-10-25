"""Compact DMS leases use shared registration and request-owned retirement."""
from types import SimpleNamespace

import pytest

from hipengine import DMSConfig
from hipengine.generation import GenerationRequest
from hipengine.generation.engine_loop import EngineLoopConfig
from hipengine.generation.qwen35_gguf_dms import Qwen35GGUFDMSModelRunner


@pytest.fixture
def runner(monkeypatch):
    sessions = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.scratch = SimpleNamespace(max_positions=kwargs['max_sequence_length'])
            self.closed = 0
            self.reset_calls = 0
            sessions.append(self)

        def close(self):
            self.closed += 1

        def reset(self):
            self.reset_calls += 1

    def acquire(shared, **kwargs):
        return Session(**kwargs), ('compact-test',), False

    generator = SimpleNamespace(
        backend='hip_gfx1151', target_arch='gfx1151', model_path='unused.gguf',
        _prepared_max_sequence_length=512, _defer_resident_session_policy_resolution=True,
        dms_config=DMSConfig('sidecar.json'), dms_metadata_path='sidecar.json',
        dms_output_token_cap=lambda prompt_length: None,
        _acquire_shared_session=acquire,
        _get_shared_runner=lambda: SimpleNamespace(runtime=object(), weights=SimpleNamespace(
                config=SimpleNamespace(ssm_conv_kernel=4))),
        _prepared_session_kv_kwargs=lambda: {}, close=lambda: None,
    )
    owner = Qwen35GGUFDMSModelRunner(generator, capacity=1)
    owner.configure_engine_loop(EngineLoopConfig(
        max_active_requests=1, prefix_cache='off', speculative_mtp_serving='off',
    ))
    yield owner, sessions
    owner.close()


def register(owner, ids=(1,), lengths=(17,), max_tokens=4):
    prompts = tuple((42,) * length for length in lengths)
    request = GenerationRequest(prompts=prompts, max_tokens=max_tokens, temperature=0,
                                top_p=1.0, ignore_eos=True)
    owner.register_batch(ids, request, prompt_rows=prompts)


@pytest.mark.parametrize('length', [17, 509, 510, 900])
def test_context_refusal_precedes_dms_row_publication(runner, length):
    owner, sessions = runner
    if length + 3 <= 512:
        register(owner, lengths=(length,))
        assert tuple(owner._rows) == (1,)
    else:
        before = owner._next_batch_id
        with pytest.raises(ValueError, match='resident capacity is 512'):
            register(owner, lengths=(length,))
        assert not owner._rows
        assert owner._next_batch_id == before
        register(owner, ids=(2,))
        assert tuple(owner._rows) == (2,)


def test_invalid_dms_cohort_publishes_no_partial_rows(runner):
    owner, _ = runner
    with pytest.raises(ValueError, match='resident capacity is 512'):
        register(owner, ids=(1, 2), lengths=(17, 900))
    assert not owner._rows
    assert owner._next_batch_id == 0


def test_admission_assigns_the_compact_session_to_its_request(runner):
    owner, sessions = runner
    register(owner)
    owner.reserve_admission(SimpleNamespace(request_id=1))
    assert owner._rows[1].lease.session is sessions[0]
    assert not owner._available
    with pytest.raises(RuntimeError, match='already has admission resources'):
        owner.reserve_admission(SimpleNamespace(request_id=1))


def test_queued_cancellation_does_not_close_an_unowned_session(runner):
    owner, sessions = runner
    register(owner, ids=(1, 2), lengths=(17, 21))
    lease = owner._available[0]
    owner.discard((2,))
    assert sessions[0].closed == 0
    assert owner._available == [lease]
    owner.reserve_admission(SimpleNamespace(request_id=1))
    assert owner._rows[1].lease is lease


def test_cancelled_admitted_session_is_closed_not_reset_or_reused(runner):
    owner, sessions = runner
    register(owner)
    owner.reserve_admission(SimpleNamespace(request_id=1))
    row = owner._rows[1]
    owner.discard((1,))
    assert row.lease is None
    assert sessions[0].closed == 1
    assert sessions[0].reset_calls == 0
    assert not owner._available
    register(owner, ids=(2,))
    owner.reserve_admission(SimpleNamespace(request_id=2))
    assert owner._rows[2].lease.session is sessions[1]
    assert sessions[0].closed == 1


def test_queued_cancel_preserves_the_admitted_neighbor(runner):
    owner, sessions = runner
    register(owner, ids=(1, 2), lengths=(17, 21))
    owner.reserve_admission(SimpleNamespace(request_id=1))
    lease = owner._rows[1].lease
    owner.discard((2,))
    assert owner._rows[1].lease is lease and lease is not None
    assert sessions[0].closed == 0


@pytest.mark.parametrize('length', [1, 3])
def test_dms_short_prompt_capability_refusal_precedes_publication(runner, length):
    owner, _ = runner
    with pytest.raises(ValueError, match='DMS bulk prefill requires at least 4 tokens'):
        register(owner, lengths=(length,))
    assert not owner._rows


def test_zero_output_never_claims_or_retires_a_compact_lease(runner):
    owner, sessions = runner
    register(owner, max_tokens=0, lengths=(900,))
    owner.reserve_admission(SimpleNamespace(request_id=1))
    assert owner._rows[1].lease is None
    owner.discard((1,))
    assert sessions[0].closed == 0


@pytest.mark.parametrize('entry', ['llm', 'server'])
def test_public_dms_selectors_are_mutually_exclusive(entry):
    from hipengine import LLM
    from hipengine.server.api import ServerConfig

    kwargs = dict(dms=DMSConfig('typed.json'), dms_metadata_path='metadata.json',
                  prefix_cache='off', speculative_mtp_serving='off')
    with pytest.raises(ValueError, match='dms and dms_metadata_path are mutually exclusive'):
        if entry == 'llm':
            LLM('unused.gguf', **kwargs)
        else:
            ServerConfig(model='unused.gguf', **kwargs)


def test_typed_dms_has_one_sidecar_and_session_options_owner():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    gen = object.__new__(Qwen35GGUFBringupGenerator)
    gen.configure_dms(DMSConfig('sidecar.json', prefill_mode='dense_pool', decision_mode='no_evict'))
    assert gen.dms_metadata_path == 'sidecar.json'
    kwargs = gen._prepared_session_kv_kwargs()
    assert kwargs['dms_metadata_path'] == 'sidecar.json'
    assert kwargs['dms_prefill_mode'] == 'dense_pool'
    assert kwargs['dms_decision_mode'] == 'no_evict'
    assert kwargs['dms_backend_factory'].__name__ == 'create_dms_bf16_backend'


def test_typed_dms_uses_shared_session_configuration_and_auto_context(runner):
    owner, sessions = runner
    calls = []
    session = SimpleNamespace(close=lambda: None)

    def acquire(shared, **kwargs):
        calls.append(kwargs)
        return session, ('shared-configuration',), False

    owner.generator._acquire_shared_session = acquire
    owner._max_sequence_length = None
    owner._reserve_sessions()
    assert len(calls) == 1
    assert calls[0]['max_sequence_length'] is None  # shared owner resolves auto
    assert calls[0]['max_batch_size'] == 1
    assert not calls[0]['defer_kv_allocation']
    assert owner._available[0].session is session


def test_compact_session_creation_honors_explicit_context_argument(runner):
    owner, sessions = runner
    owner._reserve_sessions(max_sequence_length=768)
    assert sessions[0].kwargs['max_sequence_length'] == 768


def test_typed_acquisition_reaches_shared_auto_sizing_and_configuration():
    from tests.test_unit_qwen35_gguf_capacity import _auto_context_generator

    events = []
    gen = _auto_context_generator(_prepared_kv_signature=None)
    gen.configure_dms(DMSConfig('sidecar.json', prefill_mode='dense_pool', decision_mode='no_evict'))
    gen._prepare_kv_policy = lambda request: setattr(
        gen, '_prepared_kv_signature', ('bf16', 'uniform', 'fp16', 'per_token_head'))
    gen._resolve_auto_context = lambda shared, **kwargs: events.append('auto') or 768
    session = SimpleNamespace(max_sequence_length=768, close=lambda: events.append('close'))

    def construct(shared, **kwargs):
        assert kwargs['max_sequence_length'] == 768
        assert gen._prepared_session_kv_kwargs()['dms_prefill_mode'] == 'dense_pool'
        events.append('construct')
        return session

    gen._construct_shared_session = construct
    gen._configure_session = lambda selected: events.append('configure')
    owner = object.__new__(Qwen35GGUFDMSModelRunner)
    owner.generator, owner._shared_runner = gen, object()
    owner._max_sequence_length, owner._available = None, []
    owner._reserve_sessions()
    assert events == ['auto', 'construct', 'configure']
    assert owner._available[0].session is session
    owner._release_available_sessions()
    assert events[-1] == 'close'


def test_typed_and_metadata_dms_apply_the_same_compact_output_bound():
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator

    limits = []
    for config in ('sidecar.json', DMSConfig('sidecar.json')):
        gen = object.__new__(Qwen35GGUFBringupGenerator)
        gen.configure_dms(config)
        gen.dms_output_token_cap = lambda prompt_length: 24
        owner = object.__new__(Qwen35GGUFDMSModelRunner)
        owner.generator = gen
        owner.scheduler_max_new_tokens = lambda request: request.max_tokens
        limits.append(owner.scheduler_max_new_tokens_for_prompt(
            SimpleNamespace(max_tokens=2048), (42,)*4000))
    assert limits == [24, 24]


def test_dms_capacity_sizing_releases_head_major_probe_allocations(monkeypatch):
    from hipengine.core import memory
    from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator
    from hipengine.runtime import qwen35_gguf_runner as runtime
    from tests.test_unit_qwen35_gguf_capacity_route_terms import _dense_27b_runner

    tracker = memory._MemoryStatsTracker()
    monkeypatch.setattr(memory, '_MEMORY_STATS', tracker)
    model = _dense_27b_runner()
    model.backend = 'hip_gfx1151'
    generator = object.__new__(Qwen35GGUFBringupGenerator)
    generator._dms_exact_budget_retention = lambda: None
    monkeypatch.setattr(runtime, '_gguf_aotriton_head_major_kv_enabled', lambda backend: True)
    monkeypatch.setattr(runtime, '_gguf_prefill_scratch_nbytes', lambda *a, **k: 0)
    terms = generator._dms_layer_outer_capacity_terms(model, {}, mtp_sized=False)
    for context in (512, 1024, 8448):
        assert terms['prefill_workspace_bytes_resolver'](context) > 0
        assert tracker.snapshot()['active_allocations'] == 0
        assert tracker.snapshot()['current_allocated_bytes'] == 0


def test_delayed_compact_admission_reserves_under_the_target_arch(runner, monkeypatch):
    import os

    owner, _ = runner
    register(owner, ids=(1, 2), lengths=(17, 21))
    owner.reserve_admission(SimpleNamespace(request_id=1))
    owner.discard((1,))
    acquire = owner.generator._acquire_shared_session
    seen = []

    def observe(*args, **kwargs):
        seen.append(os.environ.get('HIPENGINE_HIP_ARCH'))
        return acquire(*args, **kwargs)

    monkeypatch.delenv('HIPENGINE_HIP_ARCH', raising=False)
    owner.generator._acquire_shared_session = observe
    owner.reserve_admission(SimpleNamespace(request_id=2))
    assert seen == ['gfx1151']
    assert 'HIPENGINE_HIP_ARCH' not in os.environ


def test_dms_observability_uses_the_resolved_prefix_mode(runner):
    owner, _ = runner
    owner.generator._dms_serving_policy = {'metadata_path': 'sidecar.json', 'prefix_cache': 'off'}
    owner._prefix_cache_mode = 'radix'
    assert owner.observability_snapshot()['model_runner']['dms']['prefix_cache'] == 'radix'
