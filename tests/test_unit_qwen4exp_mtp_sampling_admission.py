"""Unsupported MTP controls must reject before GPU work, not be ignored."""
from dataclasses import replace

import pytest

from tests.test_unit_qwen4exp_mtp_steps import _provider, _request


@pytest.mark.parametrize('method', ['blocking', 'iterator', 'stream'])
@pytest.mark.parametrize('controls', [{'resident_session_key': 'reuse-not-supported'}])
def test_mtp_rejects_unimplemented_controls_before_prefill(method, controls):
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
