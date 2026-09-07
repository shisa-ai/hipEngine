"""Real-output EOS diagnostics must bind request and collector identities."""
from types import SimpleNamespace as NS
import pytest


def test_real_pair_submission_configures_only_survivor():
    from scripts.qwen38_packed_c1_lifecycle import submit_engine_pair
    oracle = list(range(24)); evidence = dict(oracle_ids=oracle, index=12)
    from hipengine.speculative import SpeculativeMTPStaticEligibility
    intent = SpeculativeMTPStaticEligibility(state='speculative_capable', reason='test',
        max_candidate_count=3, max_realized_group_rows=2, automatic_eligible=False,
        strict_fallback_key='gguf_target_ar', packed_c1_target=True)
    def submit(requests):
        assert requests[0].eos_token_id is None
        assert requests[1].eos_token_id == 12 and requests[1].min_tokens == 0
        assert all(r.speculative_mtp_static_eligibility is intent for r in requests)
        handles = []
        for i, ids in enumerate((tuple(range(8)), tuple(range(13)))):
            reason = 'length' if i == 0 else 'eos'
            output = NS(generated_token_ids=ids, finish_details=NS(reason=reason))
            terminal = NS(generated_token_ids=ids, request_id=10+i, error=None, finish_reason=reason)
            handles.append(NS(request_id=10+i, backend_request_id=40+i,
                _state=NS(collector=NS(result=terminal)), result=lambda out=output: out))
        return handles
    result = submit_engine_pair(NS(submit_speculative_children=submit), 'p', (intent,intent), eos=evidence)
    assert result[1]['generated_ids'] == oracle[:13]
    assert evidence['backend_request_id'] == 41 and evidence['collector_exact']


@pytest.mark.parametrize('budget', range(1,8))
def test_eos_marker_leaves_room_after_peer_retirement(budget):
    from scripts.qwen38_packed_c1_eos import select_survivor_eos_index
    ids = list(range(24))
    assert select_survivor_eos_index(ids, budget=budget) == max(12,8+budget)
    ids[max(12,8+budget)] = ids[0]
    assert select_survivor_eos_index(ids, budget=budget) == max(12,8+budget)+1


@pytest.mark.parametrize('budget,ids', [(0,list(range(24))), (8,list(range(24))),
                                        (7,[1]*24), (7,list(range(16)))])
def test_eos_marker_selection_fails_without_valid_room(budget,ids):
    from scripts.qwen38_packed_c1_eos import select_survivor_eos_index
    with pytest.raises(ValueError):
        select_survivor_eos_index(ids,budget=budget)


def test_eos_request_keeps_execution_ownership():
    from hipengine.generation.registry import GenerationRequest
    from scripts.qwen38_packed_c1_eos import configure_eos_request
    request = GenerationRequest(prompts=('p',), max_tokens=24, temperature=0.0,
                                top_p=1.0, ignore_eos=False)
    result = configure_eos_request(request, list(range(24)), index=12)
    assert result.eos_token_id == 12 and result.min_tokens == 0
    assert result.max_tokens == request.max_tokens and result.prompts == request.prompts
    assert request.eos_token_id is None


@pytest.mark.parametrize('fault', ['repeated_marker', 'minimum_suppression', 'out_of_range'])
def test_eos_fixture_rejects_ambiguous_boundary(fault):
    from hipengine.generation.registry import GenerationRequest
    from scripts.qwen38_packed_c1_eos import configure_eos_request
    request = GenerationRequest(prompts=('p',), max_tokens=24, temperature=0.0,
        top_p=1.0, ignore_eos=False, eos_token_id=99,
        min_tokens=1 if fault == 'minimum_suppression' else 0)
    ids = list(range(24))
    if fault == 'repeated_marker': ids[0] = ids[12]
    with pytest.raises(ValueError):
        configure_eos_request(request, ids, index=24 if fault == 'out_of_range' else 12)


@pytest.mark.parametrize('fault', [None, 'ids', 'reason', 'collector_ids', 'collector_request', 'collector_error'])
def test_eos_collector_gate(fault):
    from scripts.qwen38_packed_c1_eos import validate_eos_terminal
    expected = list(range(24)); ids = tuple(expected[:13])
    output = NS(generated_token_ids=ids, finish_details=NS(reason='eos'))
    terminal = NS(generated_token_ids=ids, request_id=5, error=None, finish_reason='eos')
    handle = NS(request_id=5, backend_request_id=41, _state=NS(collector=NS(result=terminal)))
    if fault == 'ids': output.generated_token_ids = tuple(expected)
    if fault == 'reason': output.finish_details.reason = 'length'
    if fault == 'collector_ids': terminal.generated_token_ids = tuple(expected)
    if fault == 'collector_request': terminal.request_id = 6
    if fault == 'collector_error': terminal.error = RuntimeError('bad')
    if fault:
        with pytest.raises(ValueError): validate_eos_terminal(handle, output, expected, index=12)
    else:
        evidence = validate_eos_terminal(handle, output, expected, index=12)
        assert evidence['backend_request_id'] == 41
        assert evidence['visible_tokens'] == 13
