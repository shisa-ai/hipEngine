"""Validate public AR/eager/graph measurements independently of their driver."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('ar', 'eager', 'graph')


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def validate(data, expected_cases=27):
    assert data['passed'], data.get('error')
    assert not data['attribution'], 'Synchronized attribution is not throughput evidence'
    assert data['memory_after_close']['active_allocations'] == 0
    assert data['memory_after_close']['current_allocated_bytes'] == 0
    assert data['provenance']['resolved_backend'] == 'hip_gfx1151'
    assert len(data['cases']) == expected_cases
    assert len({c['id'] for c in data['cases']}) == expected_cases
    protocol = data['protocol']
    pairs = protocol['warmups_per_case_per_arm'] + protocol['samples_per_case_per_arm']
    total = 0
    for case in data['cases']:
        assert case['eager_ids'] == case['graph_ids']
        assert all(len(case[arm + '_ids']) == 128 for arm in ARMS)
        assert len(case['samples']) == 3 * pairs
        reference = next(s['route']['specdec2_mtp2_accepted_counts'] for s in case['samples'] if s['arm'] == 'eager')
        for pair in range(pairs):
            samples = [s for s in case['samples'] if s['pair'] == pair]
            assert [s['arm'] for s in samples] == list(ARMS if pair % 2 == 0 else reversed(ARMS))
        for sample in case['samples']:
            total += 1
            arm = sample['arm']
            assert sample['warmup'] == (sample['pair'] < protocol['warmups_per_case_per_arm'])
            assert sample['generated_ids_sha256'] == digest(case[arm + '_ids'])
            assert sample['generated_tokens'] == 128 and sample['decode_tokens'] == 127
            events = sample['token_events']
            assert events[0]['tokens'] == 1 and events[-1]['tokens'] == 128
            assert all(a['tokens'] < b['tokens'] and a['seconds'] <= b['seconds'] for a, b in zip(events, events[1:]))
            assert math.isclose(sample['first_token_seconds'], events[0]['seconds'])
            assert math.isclose(sample['decode_seconds'], events[-1]['seconds'] - events[0]['seconds'])
            assert sample['whole_call_seconds'] >= events[-1]['seconds']
            assert math.isclose(sample['decode_tokens_per_second'], 127 / sample['decode_seconds'])
            assert math.isclose(sample['whole_call_tokens_per_second'], 128 / sample['whole_call_seconds'])
            route = sample['route']
            count = route['specdec2_mtp2_cycles']
            assert bool(route['specdec2_mtp2_used']) == (arm != 'ar')
            verifications = sample['verification_events']
            assert all(v['seconds'] is None for v in verifications)
            if arm == 'ar':
                assert count == 0 and not verifications
            else:
                assert count > 0 and len(verifications) == count
                accepts = route['specdec2_mtp2_accepted_counts']
                assert len(accepts) == count and accepts == reference
                assert route['specdec2_mtp2_candidate_counts'] == [1] * count
                assert 1 + count + sum(accepts) + route['native_decode_steps'] + route['serial_decode_steps'] == 128
                statuses = [v['graph']['status'] for v in verifications]
                if arm == 'eager':
                    assert route['specdec2_mtp2_execution_routes'] == ['eager'] * count
                    assert set(statuses) == {'disabled'}
                else:
                    assert 'graph' in route['specdec2_mtp2_execution_routes']
                    assert 'replayed' in statuses
                    assert not any(s.endswith('fallback') for s in statuses)
                    for event, executed in zip(verifications, route['specdec2_mtp2_execution_routes'], strict=True):
                        assert (executed == 'graph') == bool(event['graph']['replayed'])
            assert not any(k.startswith('specdec2_mtp2_recovered') for k in sample['fallback_reasons'])
        for arm in ARMS:
            for key in ('decode_tokens_per_second', 'first_token_seconds', 'whole_call_tokens_per_second'):
                assert math.isclose(case['medians'][arm][key], statistics.median(s[key] for s in case['samples'] if s['arm'] == arm and not s['warmup']))
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
        if case['split'] == 'control': result['controls'][case['id']] = case['medians']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--confirmation', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-cases', type=int, default=27)
    args = p.parse_args()
    first = json.loads(args.first.read_text())
    result = {'passed': True, 'checked_calls': validate(first, args.expected_cases), 'first': summarize(first)}
    if args.confirmation:
        second = json.loads(args.confirmation.read_text())
        result['checked_calls'] += validate(second, args.expected_cases)
        assert first['driver_sha256'] == second['driver_sha256']
        for x, y in zip(first['cases'], second['cases'], strict=True):
            assert x['id'] == y['id'] and x['prompt_ids_sha256'] == y['prompt_ids_sha256']
            for arm in ARMS: assert x[arm + '_ids'] == y[arm + '_ids']
        for key in ('hipengine_commit', 'model_fingerprint', 'device_name', 'host_name'):
            assert first['provenance'][key] == second['provenance'][key], key
        result['confirmation'] = summarize(second)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
