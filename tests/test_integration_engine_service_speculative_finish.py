"""Semantic finishes at the output bound must survive service normalization."""
from dataclasses import replace

import pytest

from hipengine.generation import (
    EngineService,
    FinishDetails,
    GenerationOutput,
    GenerationSubmission,
    TokenLogprob,
)
from tests.test_integration_engine_service_speculative_ownership import _GenericProviderDriver, _request


class EOSDriver(_GenericProviderDriver):
    def __init__(self, count):
        super().__init__()
        self.output = GenerationOutput(
            text='A' * (count - 1), generated_token_ids=(10,) * (count - 1) + (9,),
            token_logprobs=tuple(TokenLogprob(token_id=10, token_text='A', logprob=0.) for _ in range(count - 1)),
            finish_details=FinishDetails(reason='eos', eos_token_id=9,
                                         sampler_mode='sampled_speculative_mtp'))

    def generate_speculative_mtp_detailed(self, request):
        return [self.output]


@pytest.mark.parametrize('count,budget', [(1, 1), (2, 2), (4, 4), (2, 4)])
@pytest.mark.parametrize('explicit_eos', [None, 9])
def test_speculative_default_eos_at_limit_preserves_finish_and_logprobs(count, budget, explicit_eos):
    driver = EOSDriver(count)
    service = EngineService(driver)
    request = replace(_request('eos', max_tokens=budget), eos_token_id=explicit_eos)
    try:
        output = service.generate_speculative_detailed(request)[0]
        chunk, = service.stream_speculative_detailed(request)
        assert output == driver.output
        assert chunk.finish_details == driver.output.finish_details
        assert chunk.generated_token_ids == driver.output.generated_token_ids
        assert chunk.token_logprobs == driver.output.token_logprobs
    finally:
        service.close()


@pytest.mark.parametrize('controls', [dict(ignore_eos=True), dict(eos_token_id=8)])
def test_normalization_keeps_request_eos_policy_and_sampler_metadata(controls):
    driver = EOSDriver(2)
    service = EngineService(driver)
    try:
        output = service.generate_speculative_detailed(replace(_request('eos', max_tokens=2), **controls))[0]
        assert output.finish_details.reason == 'length'
        assert output.finish_details.length_limit == 2
        assert output.finish_details.eos_token_id is None
        assert output.finish_details.sampler_mode == 'sampled_speculative_mtp'
    finally:
        service.close()


@pytest.mark.parametrize('kind', ['grammar', 'thinking'])
@pytest.mark.parametrize('budget', [2, 4])
def test_exact_bound_preserves_active_control_finish(kind, budget):
    driver = EOSDriver(2)
    details = (FinishDetails(reason='stop', sampler_mode='sampled_speculative_mtp') if kind == 'grammar' else
               FinishDetails(reason='thinking_budget_exhausted', length_limit=budget, forced_close=True,
                             reasoning_tokens=2, answer_tokens=0, phase='closing_think',
                             budget_pressure='hard_close', sampler_mode='sampled_speculative_mtp'))
    driver.output = GenerationOutput(text='{}', generated_token_ids=(123, 125),
        token_logprobs=(TokenLogprob(token_id=123, token_text='{', logprob=0.),
                       TokenLogprob(token_id=125, token_text='}', logprob=0.)), finish_details=details)
    controls = ({'grammar': {'format': 'json_schema', 'value': {'type': 'object'}}} if kind == 'grammar' else
                {'thinking_hard_token_cap': 0, 'thinking_close_token_ids': (123, 125)})
    service = EngineService(driver)
    try:
        req = replace(_request('control', max_tokens=budget), **controls)
        output, = service.generate_speculative_detailed(req)
        chunk, = service.stream_speculative_detailed(req)
        assert output == driver.output
        assert chunk.finish_details == details
        assert chunk.token_logprobs == driver.output.token_logprobs
    finally:
        service.close()


@pytest.mark.parametrize('controls', [{}, {'grammar': {'format': 'json_schema', 'value': {'type': 'object'}}}])
@pytest.mark.parametrize('overrun', [False, True])
def test_semantic_finish_does_not_bypass_hard_bounds_or_require_no_control(controls, overrun):
    driver = EOSDriver(2)
    driver.output = GenerationOutput(text='AB' + ('C' if overrun else ''),
        generated_token_ids=(10, 11, 12) if overrun else (10, 11),
        finish_details=FinishDetails(reason='stop', sampler_mode='sampled_speculative_mtp'))
    service = EngineService(driver)
    try:
        output, = service.generate_speculative_detailed(replace(_request('control', max_tokens=2), **controls))
        assert output.generated_token_ids == (10, 11)
        assert output.finish_details.reason == ('stop' if controls and not overrun else 'length')
        assert output.finish_details.sampler_mode == 'sampled_speculative_mtp'
    finally:
        service.close()


class EarlyEOSDriver(_GenericProviderDriver):
    """A provider that names a model end-of-generation token as its cause.

    Gemma 4 reports an early model EOS as ``reason="stop"`` with the token
    that ended the run in ``eos_token_id``, on both its plain and MTP routes.
    The shared trim classifier would call the same token ``"eos"``; the
    service must not make the MTP route disagree with the plain one.
    """

    def __init__(self):
        super().__init__()
        common = dict(
            text="<1><turn|>",
            generated_token_ids=(1, 9),
            token_logprobs=(
                TokenLogprob(token_id=1, token_text="<1>", logprob=0.0),
                TokenLogprob(token_id=9, token_text="<turn|>", logprob=0.0),
            ),
        )
        self.plain_output = GenerationOutput(
            **common,
            finish_details=FinishDetails(
                reason="stop",
                eos_token_id=9,
                sampler_mode="greedy",
            ),
        )
        self.spec_output = GenerationOutput(
            **common,
            finish_details=FinishDetails(
                reason="stop",
                eos_token_id=9,
                sampler_mode="greedy_speculative_mtp",
            ),
        )

    def generate_speculative_mtp_detailed(self, request):
        return [self.spec_output]

    def submit_detailed(self, request):
        request_id = self._next_request_id
        self._next_request_id += 1
        self._active[request_id] = request
        self._outputs[request_id] = self.plain_output
        return GenerationSubmission(
            request_ids=(request_id,),
            request=request,
            max_ticks=1,
        )


def test_an_early_model_eos_keeps_the_provider_cause_on_both_routes():
    """An unshortened speculative run must not be relabelled against plain.

    The provider published the whole run and named the boundary itself, so the
    service's trim adds no information. Rewriting its ``stop`` as ``eos`` made
    the MTP route report a different finish cause than the plain route for the
    same tokens, and on the resident token-event stream the ``eos`` label also
    suppresses the terminal token while the blocking reply keeps it.
    """

    driver = EarlyEOSDriver()
    service = EngineService(driver)
    request = _request("early-eos", max_tokens=8)
    try:
        plain = service.generate_detailed(request)[0]
        spec = service.generate_speculative_detailed(request)[0]
        chunks = tuple(service.stream_speculative_detailed(request))
    finally:
        service.close()

    assert plain.generated_token_ids == spec.generated_token_ids == (1, 9)
    assert plain.text == spec.text == "<1><turn|>"
    assert plain.finish_details.reason == spec.finish_details.reason == "stop"
    assert plain.finish_details.eos_token_id == spec.finish_details.eos_token_id == 9
    assert plain.finish_details.sampler_mode == "greedy"
    assert spec.finish_details.sampler_mode == "greedy_speculative_mtp"
    assert chunks[-1].finish_details == spec.finish_details
    assert "".join(chunk.text for chunk in chunks) == spec.text
