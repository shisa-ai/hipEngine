"""Validate full-prefix/cursor-only target backup samples and summarize timings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('full_prefix', 'cursor_only')


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def validate(data, expected_cases=21):
    assert data['passed'], data.get('error')
    assert data['memory_after_close']['active_allocations'] == 0
    assert data['memory_after_close']['current_allocated_bytes'] == 0
    assert data['provenance']['resolved_backend'] == 'hip_gfx1151'
    return validate_completed_cases(data, expected_cases)


def validate_completed_cases(data, expected_cases):
    """Validate complete case records; does not certify process teardown."""
    assert len(data['cases']) == expected_cases
    assert len({c['id'] for c in data['cases']}) == expected_cases
    protocol = data['protocol']
    pairs = protocol['warmups_per_case_per_arm'] + protocol['samples_per_case_per_arm']
    total = 0
    for case in data['cases']:
        assert case['full_prefix_ids'] == case['cursor_only_ids']
        assert case['full_prefix_cursor_only_ids_equal']
        assert len(case['full_prefix_ids']) == 128
        assert len(case['samples']) == 2 * pairs
        reference_accepts = case['samples'][0]['route']['specdec2_mtp2_accepted_counts']
        for pair in range(pairs):
            samples = [s for s in case['samples'] if s['pair'] == pair]
            assert [s['arm'] for s in samples] == list(ARMS if pair % 2 == 0 else reversed(ARMS))
        for sample in case['samples']:
            total += 1
            assert sample['warmup'] == (sample['pair'] < protocol['warmups_per_case_per_arm'])
            assert sample['generated_ids_sha256'] == digest(case['full_prefix_ids'])
            assert sample['generated_tokens'] == 128 and sample['decode_tokens'] == 127
            events = sample['token_events']
            assert events[0]['tokens'] == 1 and events[-1]['tokens'] == 128
            assert all(a['tokens'] < b['tokens'] and a['seconds'] <= b['seconds'] for a,b in zip(events,events[1:]))
            assert math.isclose(sample['first_token_seconds'], events[0]['seconds'])
            assert math.isclose(sample['decode_seconds'], events[-1]['seconds'] - events[0]['seconds'])
            assert sample['whole_call_seconds'] >= events[-1]['seconds']
            assert math.isclose(sample['decode_tokens_per_second'], 127 / sample['decode_seconds'])
            route = sample['route']
            count = route['specdec2_mtp2_cycles']
            accepts = route['specdec2_mtp2_accepted_counts']
            assert route['specdec2_mtp2_used'] and count > 0
            observations = sample['copy_observations']
            assert len(observations) == count
            assert all((item['rows'] == 0) == (sample['arm'] == 'cursor_only') for item in observations)
            assert not data['attribution'], 'Synchronized diagnostics are not throughput measurements'
            assert len(accepts) == count and accepts == reference_accepts
            assert route['specdec2_mtp2_candidate_counts'] == [1] * count
            assert route['specdec2_mtp2_execution_routes'] == ['eager'] * count
            assert 1 + count + sum(accepts) + route['native_decode_steps'] + route['serial_decode_steps'] == 128
            assert not any(k.startswith('specdec2_mtp2_recovered') for k in sample['fallback_reasons'])
    return total


def summarize(data):
    result = {'natural': {}, 'controls': {}}
    for arm in ARMS:
        natural = [s for c in data['cases'] if c['split'] != 'control' for s in c['samples'] if s['arm'] == arm and not s['warmup']]
        result['natural'][arm] = {
            'samples': len(natural),
            'decode_tokens_per_second': sum(s['decode_tokens'] for s in natural) / sum(s['decode_seconds'] for s in natural),
            'whole_call_tokens_per_second': sum(s['generated_tokens'] for s in natural) / sum(s['whole_call_seconds'] for s in natural),
            'median_first_token_seconds': statistics.median(s['first_token_seconds'] for s in natural),
            'accepted_candidates': sum(sum(s['route']['specdec2_mtp2_accepted_counts']) for s in natural),
            'proposed_candidates': sum(s['route']['specdec2_mtp2_cycles'] for s in natural),
        }
    for case in data['cases']:
        if case['split'] == 'control':
            result['controls'][case['id']] = case['medians']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',type=Path,required=True)
    p.add_argument('--confirmation',type=Path)
    p.add_argument('--confirmation-partial',type=Path)
    p.add_argument('--confirmation-tail',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    first=json.loads(a.first.read_text())
    result={'passed':True, 'first':summarize(first), 'checked_calls':validate(first)}
    if a.confirmation_partial or a.confirmation_tail:
        assert a.confirmation_partial and a.confirmation_tail and not a.confirmation
        partial=json.loads(a.confirmation_partial.read_text())
        tail=json.loads(a.confirmation_tail.read_text())
        assert not partial['passed'] and 'error' not in partial
        result['checked_calls'] += validate_completed_cases(partial,19)
        result['checked_calls'] += validate(tail,2)
        assert {c['id'] for c in tail['cases']} == {'p1024','p4096'}
        assert not {c['id'] for c in partial['cases']} & {c['id'] for c in tail['cases']}
        assert partial['protocol'] == tail['protocol']
        assert partial['driver_sha256'] == first['driver_sha256']
        second={**tail,'cases':partial['cases']+tail['cases']}
        result['confirmation_completion']={
            'partial_cases':19,'tail_cases':2,
            'partial_process_teardown':'not observed: process time limit terminated it',
            'tail_process_teardown':'zero tracked allocations and bytes after close',
            'driver_change':'resume copy adds only explicit case selection; original driver hash is retained in the partial artifact',
        }
    elif a.confirmation:
        second=json.loads(a.confirmation.read_text())
        result['checked_calls'] += validate(second)
        assert first['driver_sha256']==second['driver_sha256']
    else:
        second=None
    if second is not None:
        for x,y in zip(first['cases'],second['cases'],strict=True):
            assert x['id']==y['id'] and x['prompt_ids_sha256']==y['prompt_ids_sha256']
            assert x['full_prefix_ids']==y['full_prefix_ids'] and x['cursor_only_ids']==y['cursor_only_ids']
            assert x['samples'][0]['route']['specdec2_mtp2_accepted_counts']==y['samples'][0]['route']['specdec2_mtp2_accepted_counts']
        for key in ('hipengine_commit','model_fingerprint','device_name','host_name'):
            assert first['provenance'][key]==second['provenance'][key],key
        result['confirmation']=summarize(second)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
