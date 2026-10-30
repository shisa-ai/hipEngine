"""Provider-scoped sampling admission must preserve unsupported-control guards."""
from types import SimpleNamespace

import pytest

from hipengine import SamplingParams
from hipengine.server import ServerConfig
from hipengine.server.api import (
    OpenAIHTTPError, _speculative_mtp_capability,
    _speculative_mtp_route_for_request, _speculative_provider_capability,
    _speculative_provider_route_for_request, _resolve_realized_generation_route,
)
from hipengine.server.schemas import CompletionRequest
from tests.test_unit_qwen4exp_mtp_resident_sampling import driver


@pytest.fixture(autouse=True)
def _request_owned_c1_path(monkeypatch):
    """These cases drive the request-owned c1 path with a stand-in resident.

    The shared Flash-Next decode batch is the default; its own suite is
    ``tests/test_unit_qwen4exp_ar_batch_decode.py``.
    """

    from tests._qwen4_exp_c1_control import select_request_owned_c1
    select_request_owned_c1(monkeypatch)


@pytest.fixture
def engine(monkeypatch):
    api, _, _ = driver(monkeypatch)
    # Exercise the real LLM -> submission adapter -> model -> provider forwarding.
    from hipengine.llm import LLM
    llm = SimpleNamespace(speculative_provider='qwen4_exp_mtp', _get_text_generator=lambda: api)
    result = SimpleNamespace(
        speculative_capabilities=LLM.speculative_capabilities.fget(llm),
        generate_speculative_detailed=lambda *a: None,
        generate_speculative_mtp_detailed=lambda *a: None,
        stream_speculative_detailed=lambda *a: None,
    )
    yield result
    api.close()


def config(**kw):
    return ServerConfig(model='target', speculative_provider='qwen4_exp_mtp',
                        draft_model='unused', speculative_candidate_budget=3,
                        speculative_mtp_serving=kw.pop('speculative_mtp_serving', 'opt_in'),
                        speculative_mtp_thinking='hard', **kw)


def route(engine, sampling, spelling='speculative'):
    req = CompletionRequest(model='target', prompt='one', **{spelling: True})
    if spelling == 'speculative':
        return _speculative_provider_route_for_request(config(), req, engine=engine,
                                                       sampling=sampling, prompt_count=1)
    return _speculative_mtp_route_for_request(config(), req, engine=engine, sampling=sampling)


@pytest.mark.parametrize('spelling', ['speculative', 'speculative_mtp'])
@pytest.mark.parametrize('controls', [
    dict(temperature=.8), dict(top_k=2, top_p=.8, min_p=.1),
    dict(logprobs=True, top_logprobs=3), dict(logit_bias={0: 1}),
    dict(repetition_penalty=1.2, presence_penalty=.3, frequency_penalty=.2),
    dict(suppress_token_ids=(2,)), dict(ignore_eos=True),
    dict(stop_token_ids=(2,)), dict(stop_token_sequences=((1, 2),)),
    dict(eos_token_id=8), dict(min_tokens=1, eos_token_id=9),
    dict(forced_tokens_pending=(1,)), dict(post_thinking_forced_tokens_pending=(1,), thinking_close_token_ids=(2,), thinking_hard_token_cap=2),
    dict(thinking_close_token_ids=(2,), thinking_hard_token_cap=2),
    dict(grammar={'format': 'json_schema', 'value': {'type': 'object'}}),
    dict(force_sequence_completion_token_sequences=((1, 2),)),
    dict(json_object_close_forcing=True),
])
def test_admit_only_provider_declared_sampling(engine, spelling, controls):
    result = route(engine, SamplingParams(**controls), spelling)
    assert result == ('speculative' if spelling == 'speculative' else 'speculative_mtp')


@pytest.mark.parametrize('controls,blocker', [
    (dict(resident_session_key='session'), 'resident_session_key'),
])
def test_unsupported_controls_remain_early_errors_or_named_fallback(engine, controls, blocker):
    sampling = SamplingParams(temperature=.8, logprobs=True, **controls)
    with pytest.raises(OpenAIHTTPError) as caught:
        route(engine, sampling)
    detail = caught.value.extra['hipengine']['speculative']
    assert caught.value.status_code == 400
    assert blocker in detail['blockers']
    assert 'temperature' not in detail['blockers']
    assert 'logprobs' not in detail['blockers']
    assert detail['compatibility_guard'] == 'target_distribution_bf16_c1'
    selected = route(engine, sampling, 'speculative_mtp')
    assert selected == 'speculative_mtp_k0'
    _, decision = _resolve_realized_generation_route(selected, group_rows=1,
                                                      sampling=sampling, engine=engine)
    assert blocker in decision['sampling_blockers']
    assert 'temperature' not in decision['sampling_blockers']


@pytest.mark.parametrize('controls,field', [
    (dict(grammar={'format': 'json_schema', 'value': {'type': 'object'}}), 'grammar'),
    (dict(thinking_close_token_ids=(2,), thinking_hard_token_cap=2), 'thinking_budget'),
    (dict(stop_token_sequences=((1, 2),)), 'stop_token_sequences'),
])
def test_withdrawing_one_control_restores_only_its_guard(engine, controls, field):
    caps = engine.speculative_capabilities
    caps['supported_sampling_fields'] = [item for item in caps['supported_sampling_fields'] if item != field]
    with pytest.raises(OpenAIHTTPError) as caught:
        route(engine, SamplingParams(temperature=.8, **controls))
    assert caught.value.extra['hipengine']['speculative']['blockers'] == [field]
    assert route(engine, SamplingParams(temperature=.8, **controls), 'speculative_mtp') == 'speculative_mtp_k0'


def test_sampling_capabilities_match_admission_and_do_not_promote_automatic(engine):
    generic = _speculative_provider_capability(config(), engine=engine)
    explicit = _speculative_mtp_capability(config(), engine=engine)
    for payload in (generic, explicit):
        assert payload['processed_target_verification'] is True
        assert payload['compatibility_guard'] == 'target_distribution_bf16_c1'
        assert 'temperature' in payload['supported_sampling_fields']
        assert payload['sampling_algorithm'] == 'target_draw_match_deterministic_draft'
        assert not payload['default_enabled']
    assert 'temperature' not in explicit['incompatible_fields']
    assert 'temperature' not in explicit['incompatible_conditions']
    assert 'thinking_budget' not in explicit['incompatible_fields']
    assert 'grammar' in explicit['supported_sampling_fields']
    assert 'sampled' in explicit['allowed_execution_modes']
    req = CompletionRequest(model='target', prompt='one')
    assert _speculative_mtp_route_for_request(config(speculative_mtp_serving='enabled'),
        req, engine=engine, sampling=SamplingParams(temperature=.8)) == 'speculative_mtp_k0'


def test_boolean_processed_capability_alone_does_not_disable_guards(engine):
    engine.speculative_capabilities = {'processed_target_verification': True}
    with pytest.raises(OpenAIHTTPError):
        route(engine, SamplingParams(temperature=.8, logprobs=True))


def test_legacy_provider_guards_stay_raw_greedy(engine):
    engine.speculative_capabilities = {'processed_target_verification': False}
    with pytest.raises(OpenAIHTTPError) as caught:
        route(engine, SamplingParams(temperature=.8, logprobs=True, top_k=2))
    detail = caught.value.extra['hipengine']['speculative']
    assert set(detail['blockers']) == {'temperature', 'logprobs', 'top_k'}
    assert detail['compatibility_guard'] == 'raw_greedy_bf16_c1'
    # Existing generic providers own stop-token handling; do not change that contract.
    assert route(engine, SamplingParams(stop_token_ids=(2,))) == 'speculative'
