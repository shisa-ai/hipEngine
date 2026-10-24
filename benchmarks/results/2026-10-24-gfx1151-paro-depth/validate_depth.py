"""Independently validate the PARO candidate-depth measurement records."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('ar', 'b1', 'b2', 'b3')

def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()

def validate(data, expected_cases=27):
    assert data['passed'], data.get('error')
    assert data['memory_after_close']['active_allocations'] == 0
    assert data['memory_after_close']['current_allocated_bytes'] == 0
    assert data['provenance']['resolved_backend'] == 'hip_gfx1151'
    assert len(data['cases']) == expected_cases
    assert len({c['id'] for c in data['cases']}) == expected_cases
    protocol = data['protocol']
    assert tuple(protocol['arms']) == ARMS and protocol['warmups_per_arm'] == 1
    assert protocol['output_tokens'] == 128
    pairs = 1 + protocol['repetitions']
    checked = 0
    for case in data['cases']:
        assert all(len(case[arm + '_ids']) == 128 for arm in ARMS)
        assert len(case['samples']) == 4 * pairs
        for pair in range(pairs):
            samples = [s for s in case['samples'] if s['pair'] == pair]
            assert [s['arm'] for s in samples] == list(ARMS[pair % 4:] + ARMS[:pair % 4])
        references = {}
        for s in case['samples']:
            checked += 1
            arm = s['arm']
            assert s['warmup'] == (s['pair'] == 0)
            assert s['ids_sha256'] == digest(case[arm + '_ids'])
            assert s['generated_tokens'] == 128 and s['decode_tokens'] == 127
            events = s['events']
            assert events[0]['tokens'] == 1 and events[-1]['tokens'] == 128
            assert all(a['tokens'] < b['tokens'] and a['seconds'] <= b['seconds'] for a, b in zip(events, events[1:]))
            assert math.isclose(s['first_token_seconds'], events[0]['seconds'])
            assert math.isclose(s['decode_seconds'], events[-1]['seconds'] - events[0]['seconds'])
            assert s['whole_call_seconds'] >= events[-1]['seconds']
            assert math.isclose(s['decode_tokens_per_second'], 127 / s['decode_seconds'])
            assert math.isclose(s['whole_call_tokens_per_second'], 128 / s['whole_call_seconds'])
            route = s['route']
            count = route['specdec2_mtp2_cycles']
            assert bool(route['specdec2_mtp2_used']) == (arm != 'ar')
            if arm == 'ar':
                assert count == 0
            else:
                candidates = route['specdec2_mtp2_candidate_counts']
                accepts = route['specdec2_mtp2_accepted_counts']
                assert len(candidates) == len(accepts) == count and count > 0
                depth = int(arm[1:])
                assert max(candidates) == depth and all(1 <= n <= depth for n in candidates)
                assert all(0 <= a <= n for a, n in zip(accepts, candidates, strict=True))
                assert 1 + count + sum(accepts) + route['native_decode_steps'] + route['serial_decode_steps'] == 128
                assert route['specdec2_mtp2_execution_routes'] == ['eager'] * count
                if arm not in references:
                    references[arm] = (candidates, accepts)
                assert references[arm] == (candidates, accepts)
            assert not any(k.startswith('specdec2_mtp2_recovered') for k in s['fallback_reasons'])
        for arm in ARMS:
            assert case['cross_depth_ids_equal'][arm] == (case[arm + '_ids'] == case['b1_ids'])
            for key in ('decode_tokens_per_second', 'whole_call_tokens_per_second', 'first_token_seconds'):
                assert math.isclose(case['medians'][arm][key], statistics.median(s[key] for s in case['samples'] if s['arm'] == arm and not s['warmup']))
    return checked

def summarize(data):
    natural = {}
    for arm in ARMS:
        samples = [s for c in data['cases'] if c['split'] != 'control' for s in c['samples'] if s['arm'] == arm and not s['warmup']]
        natural[arm] = {'samples': len(samples), 'decode_tokens_per_second': sum(s['decode_tokens'] for s in samples) / sum(s['decode_seconds'] for s in samples),
                        'whole_call_tokens_per_second': sum(s['generated_tokens'] for s in samples) / sum(s['whole_call_seconds'] for s in samples),
                        'median_first_token_seconds': statistics.median(s['first_token_seconds'] for s in samples),
                        'accepted_candidates': sum(sum(s['route']['specdec2_mtp2_accepted_counts']) for s in samples),
                        'proposed_candidates': sum(sum(s['route']['specdec2_mtp2_candidate_counts']) for s in samples)}
    return {'natural': natural, 'controls': {c['id']: c['medians'] for c in data['cases'] if c['split'] == 'control'}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first', type=Path, required=True)
    p.add_argument('--confirmation', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, default=27)
    args = p.parse_args()
    first = json.loads(args.first.read_text())
    checked = validate(first, args.cases)
    result = {'passed': True, 'checked_calls': checked, 'first': summarize(first)}
    if args.confirmation:
        confirmation = json.loads(args.confirmation.read_text())
        result['checked_calls'] += validate(confirmation, args.cases)
        assert first['driver_sha256'] == confirmation['driver_sha256']
        assert first['protocol']['sources'] == confirmation['protocol']['sources']
        for a, b in zip(first['cases'], confirmation['cases'], strict=True):
            assert a['id'] == b['id'] and a['prompt_ids_sha256'] == b['prompt_ids_sha256']
            for arm in ARMS:
                assert a[arm + '_ids'] == b[arm + '_ids']
                ar = next(s['route'] for s in a['samples'] if s['arm'] == arm)
                br = next(s['route'] for s in b['samples'] if s['arm'] == arm)
                for key in ('specdec2_mtp2_candidate_counts', 'specdec2_mtp2_accepted_counts'):
                    assert ar[key] == br[key]
        result['confirmation'] = summarize(confirmation)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
