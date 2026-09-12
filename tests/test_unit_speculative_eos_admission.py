"""Only runner-owned EOS handling may relax the shared EOS blocker."""
from types import SimpleNamespace as NS
import pytest
from hipengine.generation.registry import GenerationRequest


@pytest.mark.parametrize('supported,fields,expected', [
    (False, {}, 'greedy'), (False, {'eos_token_id': 7}, 'processed'),
    (True, {'eos_token_id': 7}, 'greedy'),
    (True, {'eos_token_id': 7, 'min_tokens': 1}, 'processed'),
    (True, {'eos_token_id': 7, 'temperature': 1.0}, 'processed'),
    (True, {'eos_token_id': 7, 'ignore_eos': True}, 'processed'),
])
def test_runner_eos_permission_does_not_relax_other_sampling(supported, fields, expected):
    from hipengine.generation.engine_loop import _speculative_sampling_mode
    from hipengine.generation.sampling import speculative_mtp_sampling_blockers
    request = GenerationRequest(**(dict(prompts=('p',),max_tokens=24,temperature=0.0,
                                        top_p=1.0,ignore_eos=False) | fields))
    runner = NS(speculative_eos_supported=lambda rid: supported and rid == 41)
    assert _speculative_sampling_mode(runner, 41, request) == expected
    if fields.get('eos_token_id') is not None:
        assert 'eos_token_id' in speculative_mtp_sampling_blockers(request)


@pytest.mark.parametrize('enabled,physical,expected', [(False,True,False),(True,False,False),(True,True,True)])
def test_gguf_runner_requires_enabled_physical_owner(enabled,physical,expected):
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
    runner = NS(_resolved_mtp2_adapter=lambda: NS(enabled=enabled,
        _physical_c1_request=lambda rid: rid == 41 and physical))
    assert Qwen35GGUFResidentModelRunner.speculative_eos_supported(runner,41) is expected
    assert not Qwen35GGUFResidentModelRunner.speculative_eos_supported(runner,42)


def test_missing_runner_eos_contract_stays_closed():
    from hipengine.generation.engine_loop import _speculative_sampling_mode
    request=GenerationRequest(prompts=('p',),max_tokens=24,temperature=0.0,
                              top_p=1.0,ignore_eos=False,eos_token_id=7)
    assert _speculative_sampling_mode(object(),41,request) == 'processed'
