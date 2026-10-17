"""The raw-top1 MTP provider must reject processed requests before GPU work."""
from dataclasses import replace

import pytest

from tests.test_unit_qwen4exp_mtp_steps import _provider, _request


@pytest.mark.parametrize('method', ['blocking', 'iterator', 'stream'])
@pytest.mark.parametrize('controls', [
    {'temperature': .7}, {'logprobs': True}, {'top_logprobs': 3},
    {'frequency_penalty': 1.0}, {'logit_bias': {0: 5}},
    {'suppress_token_ids': (10,)}, {'thinking_hard_token_cap': 0, 'thinking_close_token_ids': (42,)},
])
def test_raw_mtp_rejects_sampling_and_metadata_before_prefill(method, controls):
    provider, target, draft = _provider()
    request = replace(_request(), **controls)
    try:
        with pytest.raises(NotImplementedError):
            if method == 'blocking':
                provider.generate_detailed(request)
            elif method == 'stream':
                next(provider.stream_detailed(request))
            else:
                next(provider.iter_generate((1, 2), request))
        assert target.runner.prefill_calls == draft.prime_calls == draft.propose_calls == 0
    finally:
        provider.close()
