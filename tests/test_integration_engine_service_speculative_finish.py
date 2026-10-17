"""Model-default EOS at the output bound must survive service normalization."""
from dataclasses import replace

import pytest

from hipengine.generation import EngineService, FinishDetails, GenerationOutput, TokenLogprob
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
