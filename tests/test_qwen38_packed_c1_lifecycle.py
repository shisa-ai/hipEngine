"""CPU contract for a real concurrent C2-to-C1 route diagnostic."""
import copy

import pytest

from scripts.qwen38_packed_c1_lifecycle import validate_transition


@pytest.mark.parametrize('close_fails', [False, True])
def test_teardown_failure_cannot_publish_pass(tmp_path, close_fails):
    import json
    from scripts.qwen38_packed_c1_lifecycle import close_and_report

    class Runtime:
        closed = False

        def close(self):
            self.closed = True
            if close_fails:
                raise RuntimeError('drain failed')

    runtime = Runtime()
    output = tmp_path / 'nested' / 'result.json'
    result = dict(passed=True, cells=[{'exact': True}])
    if close_fails:
        with pytest.raises(RuntimeError, match='drain failed'):
            close_and_report(runtime, output, result)
    else:
        close_and_report(runtime, output, result)
    saved = json.loads(output.read_text())
    assert runtime.closed
    assert saved['passed'] is (not close_fails)
    assert saved['cells'] == result['cells']
    assert saved['teardown_error'] == ('RuntimeError: drain failed' if close_fails else None)


@pytest.mark.parametrize('capacity,widths', [(2, (1, 2)), (8, (1, 2)), (8, tuple(range(1, 9)))])
def test_lifecycle_scope_extension_is_diagnostic_only(monkeypatch, capacity, widths):
    from types import SimpleNamespace
    from hipengine.models import qwen35
    from scripts.qwen38_packed_c1_lifecycle import install_lifecycle_evidence
    original = qwen35.QWEN35_GGUF.speculative_mtp_serving_evidence
    plugin = SimpleNamespace(speculative_mtp_serving_evidence=original)
    monkeypatch.setattr(qwen35, 'QWEN35_GGUF', plugin)
    install_lifecycle_evidence(capacity, widths)
    rows = plugin.speculative_mtp_serving_evidence
    assert rows[:-len(widths)] == original
    for width, row in zip(widths, rows[-len(widths):], strict=True):
        assert row.min_output_horizon_tokens == 8
        assert row.max_output_horizon_tokens == 24
        assert row.resident_capacity == capacity
        assert row.realized_group_rows == width
        assert row.packed_c1_target is (width == 1)
        assert not row.automatic_eligible
        assert 'unqualified' in row.reason
        assert row.evidence_artifacts == ('scripts/qwen38_packed_c1_lifecycle.py',)


def engine_intent():
    from hipengine.speculative import SpeculativeMTPStaticEligibility
    return SpeculativeMTPStaticEligibility(
        state='speculative_capable', reason='test', max_candidate_count=3,
        max_realized_group_rows=2, automatic_eligible=False,
        strict_fallback_key='gguf_target_ar', packed_c1_target=True)


def test_engine_intent_requires_separate_c1_permission():
    from dataclasses import replace
    from scripts.qwen38_packed_c1_lifecycle import combine_engine_intent
    wide = replace(engine_intent(), packed_c1_target=False)
    single = replace(engine_intent(), max_realized_group_rows=1)
    combined = combine_engine_intent(single, wide)
    assert combined.packed_c1_target and combined.max_realized_group_rows == 2
    assert not combined.automatic_eligible
    assert combined.fingerprint not in (single.fingerprint, wide.fingerprint)
    with pytest.raises(ValueError):
        combine_engine_intent(replace(single, packed_c1_target=False), wide)


def test_engine_pair_is_one_atomic_submission():
    from types import SimpleNamespace
    from scripts.qwen38_packed_c1_lifecycle import submit_engine_pair
    calls = []

    class Service:
        def submit_speculative_children(self, requests):
            calls.append(tuple(r.max_tokens for r in requests))
            assert all(r.prompts == ('prompt',) for r in requests)
            return [SimpleNamespace(result=lambda: SimpleNamespace(generated_token_ids=(1, 2)))
                    for _ in requests]

    results = submit_engine_pair(Service(), 'prompt', (engine_intent(), engine_intent()))
    assert calls == [(8, 24)]
    assert all(r['generated_ids'] == [1, 2] and r['usage'] is None for r in results)


@pytest.mark.parametrize('fault', ['missing_handle', 'missing_ids'])
def test_engine_pair_rejects_incomplete_outputs(fault):
    from types import SimpleNamespace
    from scripts.qwen38_packed_c1_lifecycle import submit_engine_pair

    class Service:
        def submit_speculative_children(self, requests):
            count = 1 if fault == 'missing_handle' else 2
            return [SimpleNamespace(result=lambda: SimpleNamespace(generated_token_ids=None))
                    for _ in range(count)]

    with pytest.raises(ValueError):
        submit_engine_pair(Service(), 'prompt', (engine_intent(), engine_intent()))


def evidence():
    return [
        dict(request_ids=[10, 11], active_request_ids=[10, 11], scheduler_slots=[0, 1], resident_slots=[7, 6],
             packed_request_ids=[10, 11], packed_group_sizes=[2], native_c1=False, error=None),
        dict(request_ids=[11], active_request_ids=[11], scheduler_slots=[1], resident_slots=[6],
             packed_request_ids=[11], packed_group_sizes=[1], native_c1=True, error=None),
    ]


def response():
    return dict(generated_ids=[4, 5], usage=dict(prompt_tokens=3, completion_tokens=2, total_tokens=5))


def test_response_usage_matches_independent_ar():
    from scripts.qwen38_packed_c1_lifecycle import validate_response
    validate_response(response(), response())


@pytest.mark.parametrize('fault', ['tokens', 'completion', 'prompt', 'total', 'missing'])
def test_reject_output_or_usage_drift(fault):
    from scripts.qwen38_packed_c1_lifecycle import validate_response
    actual = response()
    if fault == 'tokens': actual['generated_ids'] = [4, 6]
    elif fault == 'missing': actual['usage'] = None
    else: actual['usage'][fault + '_tokens'] += 1
    with pytest.raises(ValueError):
        validate_response(response(), actual)


def test_real_survivor_keeps_storage_not_scheduler_identity():
    result = validate_transition(evidence())
    assert result['survivor_request_id'] == 11
    assert result['resident_slot'] == 6


@pytest.mark.parametrize('fault', ['only_c1', 'only_c2', 'new_request', 'no_packed',
                                  'legacy', 'storage_changed', 'error', 'reverse', 'singleton_loop', 'ar_peer'])
def test_reject_incomplete_or_false_transition(fault):
    rows = copy.deepcopy(evidence())
    if fault == 'only_c1': rows = rows[1:]
    elif fault == 'only_c2': rows = rows[:1]
    elif fault == 'new_request': rows[1]['request_ids'] = [12]
    elif fault == 'no_packed': rows[1]['packed_request_ids'] = []
    elif fault == 'legacy': rows[1]['native_c1'] = False
    elif fault == 'storage_changed': rows[1]['resident_slots'] = [5]
    elif fault == 'error': rows[0]['error'] = 'failed target'
    elif fault == 'reverse': rows.reverse()
    elif fault == 'singleton_loop': rows[0]['packed_group_sizes'] = [1, 1]
    elif fault == 'ar_peer': rows[1]['active_request_ids'] = [10, 11]
    with pytest.raises(ValueError):
        validate_transition(rows)
