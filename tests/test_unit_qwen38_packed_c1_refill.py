"""Real-ID trace requirements for engine survivor refill."""
import pytest
from types import SimpleNamespace


@pytest.mark.parametrize('ready', [True, False])
def test_controller_waits_before_refill_submission(ready):
    from scripts.qwen38_packed_c1_refill import collect_refilled_pair
    calls = []
    def handle(rid):
        return SimpleNamespace(backend_request_id=rid,
            result=lambda timeout: calls.append(('result', rid)) or rid)
    request = object()
    def submit(requests):
        assert requests == (request,)
        calls.append('refill')
        return [handle(12)]
    event = SimpleNamespace(wait=lambda timeout: calls.append('singleton') or ready)
    service = SimpleNamespace(submit_speculative_children=submit)
    if ready:
        outputs, evidence = collect_refilled_pair(service, [handle(10), handle(11)], request, event)
        assert outputs == [10, 11, 12]
        assert evidence == dict(initial_request_ids=[10, 11], refill_request_id=12)
        assert calls == ['singleton', 'refill', ('result', 10), ('result', 11), ('result', 12)]
    else:
        with pytest.raises(TimeoutError):
            collect_refilled_pair(service, [handle(10), handle(11)], request, event)
        assert calls == ['singleton']


def test_engine_submission_refill_branch_preserves_request_intent():
    from hipengine.speculative import SpeculativeMTPStaticEligibility
    from scripts.qwen38_packed_c1_lifecycle import submit_engine_pair
    intent = SpeculativeMTPStaticEligibility(
        state='speculative_capable', reason='test', max_candidate_count=3,
        max_realized_group_rows=2, automatic_eligible=False,
        strict_fallback_key='gguf_target_ar', packed_c1_target=True)
    calls = []
    class Service:
        def submit_speculative_children(self, requests):
            calls.append(tuple(r.max_tokens for r in requests))
            assert all(r.speculative_mtp_static_eligibility is intent for r in requests)
            ids = (10, 11) if len(calls) == 1 else (12,)
            return [SimpleNamespace(backend_request_id=rid,
                result=lambda timeout, rid=rid: SimpleNamespace(generated_token_ids=(rid,)))
                for rid in ids]
    evidence = {}
    ready = SimpleNamespace(wait=lambda timeout: len(calls) == 1)
    outputs = submit_engine_pair(Service(), 'prompt', (intent, intent),
                                 singleton_ready=ready, refill=evidence)
    assert calls == [(8, 24), (8,)]
    assert evidence == dict(initial_request_ids=[10, 11], refill_request_id=12)
    assert [r['generated_ids'] for r in outputs] == [[10], [11], [12]]


def row(ids, slots):
    return dict(request_ids=ids, active_request_ids=ids, packed_request_ids=ids,
                packed_group_sizes=[len(ids)], resident_slots=slots,
                native_c1=len(ids) == 1, error=None)


@pytest.mark.parametrize('fault', [None, 'no_singleton', 'no_refill', 'wrong_peer',
                                 'moved', 'fake_packed', 'legacy', 'error', 'duplicate_id', 'shared_slot'])
def test_refill_requires_ordered_real_survivor(fault):
    from scripts.qwen38_packed_c1_refill import validate_refill_transition
    rows = [row([10, 11], [7, 6]), row([11], [6]), row([11, 12], [6, 7])]
    if fault == 'no_singleton': del rows[1]
    elif fault == 'no_refill': rows.pop()
    elif fault == 'wrong_peer': rows[-1] = row([11, 13], [6, 7])
    elif fault == 'moved': rows[-1]['resident_slots'] = [7, 6]
    elif fault == 'fake_packed': rows[-1]['packed_group_sizes'] = [1, 1]
    elif fault == 'legacy': rows[1]['native_c1'] = False
    elif fault == 'error': rows[-1]['error'] = 'failure'
    elif fault == 'duplicate_id': rows[-1] = row([11, 12, 12], [6, 7, 5])
    elif fault == 'shared_slot': rows[-1] = row([11, 12], [6, 6])
    evidence = dict(initial_request_ids=[10, 11], refill_request_id=12)
    if fault is None:
        assert validate_refill_transition(rows, evidence)['survivor_request_id'] == 11
    else:
        with pytest.raises(ValueError):
            validate_refill_transition(rows, evidence)
