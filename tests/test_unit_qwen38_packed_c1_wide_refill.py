"""Require actual width-eight shrink/refill, not capacity-eight pairs."""
import pytest


@pytest.mark.parametrize('missing', [None, 1, 4, 8])
@pytest.mark.parametrize('shared_bound', [False, True])
def test_wide_intent_requires_every_intermediate_width(missing, shared_bound):
    from types import SimpleNamespace
    from hipengine.speculative import SpeculativeMTPStaticEligibility
    from scripts.qwen38_packed_c1_wide_refill import resolve_wide_intents
    calls = []
    def resolve(**kwargs):
        width = kwargs['realized_group_rows']
        calls.append((width, kwargs['output_horizon_tokens']))
        return SimpleNamespace(admitted=width != missing,
            static_eligibility=SpeculativeMTPStaticEligibility(
                state='speculative_capable', reason='test', max_candidate_count=3,
                max_realized_group_rows=(8 if shared_bound and width > 1 else width), automatic_eligible=False,
                strict_fallback_key='gguf_target_ar', packed_c1_target=width == 1),
            as_dict=lambda: {'width': width})
    llm = SimpleNamespace(count_tokens=lambda prompt: 42,
                          resolve_speculative_mtp_serving_plan=resolve)
    if missing:
        with pytest.raises(ValueError): resolve_wide_intents(llm, 'prompt')
    else:
        intents, sources = resolve_wide_intents(llm, 'prompt')
        assert len(intents) == len(sources) == 2
        assert all(i.packed_c1_target and i.max_realized_group_rows == 8
                   and not i.automatic_eligible for i in intents)
        assert calls == [(w, h) for h in (8, 24) for w in range(1, 9)]


def test_wide_submission_uses_atomic_groups_and_fifteen_identities():
    from types import SimpleNamespace
    from scripts.qwen38_packed_c1_wide_refill import submit_wide_refill
    from hipengine.speculative import SpeculativeMTPStaticEligibility
    intent = SpeculativeMTPStaticEligibility(state='speculative_capable', reason='test',
        max_candidate_count=3, max_realized_group_rows=8, packed_c1_target=True,
        strict_fallback_key='gguf_target_ar', automatic_eligible=False)
    calls = []
    def submit(requests):
        start = 0 if not calls else 8
        calls.append([r.max_tokens for r in requests])
        assert all(r.speculative_mtp_static_eligibility is intent for r in requests)
        return [SimpleNamespace(backend_request_id=i,
            result=lambda timeout, i=i: SimpleNamespace(generated_token_ids=(i,)))
            for i in range(start, start + len(requests))]
    service = SimpleNamespace(submit_speculative_children=submit)
    ready = SimpleNamespace(wait=lambda timeout: calls == [[8] * 7 + [24]])
    outputs, evidence = submit_wide_refill(service, 'prompt', (intent, intent), ready)
    assert calls == [[8] * 7 + [24], [8] * 7]
    assert evidence == dict(initial_request_ids=list(range(8)), refill_request_ids=list(range(8, 15)))
    assert [r['generated_ids'] for r in outputs] == [[i] for i in range(15)]


def row(ids, slots):
    return dict(request_ids=ids, active_request_ids=ids, packed_request_ids=ids,
                packed_group_sizes=[len(ids)], resident_slots=slots,
                native_c1=len(ids) == 1, error=None)


@pytest.mark.parametrize('fault', [None, 'initial_pair', 'refill_pair', 'no_singleton',
    'old_peer', 'moved', 'shared_slot', 'duplicate_id', 'legacy', 'ar_peer'])
def test_full_width_survivor_transition(fault):
    from scripts.qwen38_packed_c1_wide_refill import validate_wide_refill
    initial = list(range(8))
    added = list(range(8, 15))
    rows = [row(initial, initial), row([7], [7]), row([7] + added, [7] + list(range(7)))]
    if fault == 'initial_pair': rows[0] = row([6, 7], [6, 7])
    elif fault == 'refill_pair': rows[-1] = row([7, 8], [7, 0])
    elif fault == 'no_singleton': del rows[1]
    elif fault == 'old_peer': rows[-1] = row([7, 0] + added[1:], [7] + list(range(7)))
    elif fault == 'moved': rows[1]['resident_slots'] = [0]
    elif fault == 'shared_slot': rows[-1]['resident_slots'][1] = 7
    elif fault == 'duplicate_id': rows[-1]['request_ids'][-1] = 8
    elif fault == 'legacy': rows[1]['native_c1'] = False
    elif fault == 'ar_peer': rows[1]['active_request_ids'] = [6, 7]
    if fault:
        with pytest.raises(ValueError):
            validate_wide_refill(rows, initial, added)
    else:
        assert validate_wide_refill(rows, initial, added) == dict(
            survivor_request_id=7, resident_slot=7, initial_width=8, refill_width=8)
