"""CPU contract for controller-thread cancellation of a real engine peer."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('fault', [None, 'request_id', 'error_identity', 'unexpected_error'])
def test_cancelled_exception_uses_real_terminal_collector(fault):
    from hipengine.generation.concurrency2 import BlockingOutputCollector, EngineOutput
    from hipengine.generation.deadline import GenerationCancelled
    from scripts.qwen38_packed_c1_cancel import collect_cancelled_pair
    error = RuntimeError('reclaim failed') if fault == 'unexpected_error' else GenerationCancelled()
    collector = BlockingOutputCollector(max_output_tokens=24)
    collector.bind(100)
    collector.publish(EngineOutput(kind='token', request_id=100, token_id=4, token_index=0))
    collector.publish(EngineOutput(kind='terminal', request_id=100, generated_token_ids=(4,),
        finish_reason='cancelled', error=GenerationCancelled() if fault == 'error_identity' else error))

    def result(timeout):
        raise error

    cancelled = SimpleNamespace(backend_request_id=10, request_id=999 if fault == 'request_id' else 100,
        _state=SimpleNamespace(collector=collector), cancel=lambda reason: True, result=result)
    peer = SimpleNamespace(backend_request_id=11, result=lambda timeout: SimpleNamespace(
        generated_token_ids=(4, 5), finish_details=SimpleNamespace(cancelled=False)))
    ready = SimpleNamespace(wait=lambda timeout: True)
    if fault:
        expected_error = RuntimeError if fault == 'unexpected_error' else ValueError
        with pytest.raises(expected_error):
            collect_cancelled_pair([cancelled, peer], ready)
    else:
        outputs, evidence = collect_cancelled_pair([cancelled, peer], ready)
        assert outputs[0].generated_token_ids == (4,)
        assert outputs[0].finish_details.cancelled
        assert evidence['cancelled_output_source'] == 'terminal_collector'


def test_real_submission_helper_carries_cancel_identity_and_equal_horizons():
    from hipengine.speculative import SpeculativeMTPStaticEligibility
    from scripts.qwen38_packed_c1_lifecycle import submit_engine_pair
    intent = SpeculativeMTPStaticEligibility(
        state='speculative_capable', reason='test', max_candidate_count=3,
        max_realized_group_rows=2, automatic_eligible=False,
        strict_fallback_key='gguf_target_ar', packed_c1_target=True)
    calls = []

    class Service:
        def submit_speculative_children(self, requests):
            assert tuple(r.max_tokens for r in requests) == (24, 24)
            assert all(r.speculative_mtp_static_eligibility is intent for r in requests)
            calls.append('submit_pair')
            return [SimpleNamespace(backend_request_id=i,
                cancel=lambda reason: calls.append('cancel') or True,
                result=lambda timeout, i=i: SimpleNamespace(generated_token_ids=(1, 2),
                    finish_details=SimpleNamespace(cancelled=i == 10))) for i in (10, 11)]

    evidence = {}
    ready = SimpleNamespace(wait=lambda timeout: calls.append('paired') or True)
    outputs = submit_engine_pair(Service(), 'prompt', (intent, intent),
                                 horizons=(24, 24), paired_ready=ready, cancellation=evidence)
    assert calls == ['submit_pair', 'paired', 'cancel']
    assert evidence == dict(cancelled_request_id=10, survivor_request_id=11, acknowledged=True)
    assert [o['generated_ids'] for o in outputs] == [[1, 2], [1, 2]]


def test_cancel_waits_for_pair_and_only_cancels_first_child():
    from scripts.qwen38_packed_c1_cancel import collect_cancelled_pair
    calls = []

    class Ready:
        def wait(self, timeout):
            calls.append('paired')
            return True

    class Handle:
        def __init__(self, rid):
            self.backend_request_id = rid

        def cancel(self, reason):
            calls.append(('cancel', self.backend_request_id))
            return True

        def result(self, timeout):
            calls.append(('result', self.backend_request_id))
            return SimpleNamespace(generated_token_ids=(1, 2),
                                   finish_details=SimpleNamespace(cancelled=self.backend_request_id == 10))

    outputs, evidence = collect_cancelled_pair([Handle(10), Handle(11)], Ready())
    assert len(outputs) == 2
    assert calls == ['paired', ('cancel', 10), ('result', 10), ('result', 11)]
    assert evidence == dict(cancelled_request_id=10, survivor_request_id=11, acknowledged=True)


@pytest.mark.parametrize('fault', [None, 'prefix', 'full_cancelled', 'survivor', 'identity'])
def test_cancel_output_and_survivor_identity(fault):
    from scripts.qwen38_packed_c1_cancel import validate_cancel_outputs
    expected = [{'generated_ids': [1, 2, 3]}, {'generated_ids': [1, 2, 3]}]
    actual = [{'generated_ids': [1]}, {'generated_ids': [1, 2, 3]}]
    evidence = dict(cancelled_request_id=10, survivor_request_id=11, acknowledged=True)
    transition = dict(survivor_request_id=11)
    if fault == 'prefix': actual[0]['generated_ids'] = [9]
    elif fault == 'full_cancelled': actual[0]['generated_ids'] = [1, 2, 3]
    elif fault == 'survivor': actual[1]['generated_ids'] = [1, 2]
    elif fault == 'identity': transition['survivor_request_id'] = 10
    if fault is None:
        validate_cancel_outputs(expected, actual, evidence, transition)
    else:
        with pytest.raises(ValueError):
            validate_cancel_outputs(expected, actual, evidence, transition)


@pytest.mark.parametrize('fault', ['no_pair', 'not_acknowledged', 'not_cancelled', 'peer_cancelled'])
def test_cancel_rejects_missing_lifecycle_evidence(fault):
    from scripts.qwen38_packed_c1_cancel import collect_cancelled_pair
    handles = [SimpleNamespace(backend_request_id=i,
        cancel=lambda reason: fault != 'not_acknowledged',
        result=lambda timeout, i=i: SimpleNamespace(finish_details=SimpleNamespace(
            cancelled=(i == 0 and fault != 'not_cancelled') or (i == 1 and fault == 'peer_cancelled'))))
        for i in (0, 1)]
    ready = SimpleNamespace(wait=lambda timeout: fault != 'no_pair')
    with pytest.raises((ValueError, TimeoutError)):
        collect_cancelled_pair(handles, ready)
