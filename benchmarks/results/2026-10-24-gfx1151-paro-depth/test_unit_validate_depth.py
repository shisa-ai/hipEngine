"""Synthetic positive and corrupted-record checks for the depth validator."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('depth_validator', Path(__file__).with_name('validate_depth.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def fixture():
    ids = list(range(128))
    c = {'id': 'natural', 'split': 'category', 'samples': [],
         'cross_depth_ids_equal': {arm: True for arm in v.ARMS}, 'medians': {}}
    for arm in v.ARMS:
        c[arm + '_ids'] = ids
        c['medians'][arm] = {'decode_tokens_per_second': 127.0, 'whole_call_tokens_per_second': 64.0, 'first_token_seconds': 0.5}
    for pair in range(2):
        for arm in v.ARMS[pair:] + v.ARMS[:pair]:
            depth = 0 if arm == 'ar' else int(arm[1:])
            count = 0 if not depth else (127 + depth) // (depth + 1)
            candidates = [depth] * count
            accepted = [depth] * count
            if count:
                accepted[-1] = 127 - count - depth * (count - 1)
                candidates[-1] = max(1, accepted[-1])
            route = {'specdec2_mtp2_used': bool(depth), 'specdec2_mtp2_cycles': count,
                     'specdec2_mtp2_candidate_counts': candidates, 'specdec2_mtp2_accepted_counts': accepted,
                     'specdec2_mtp2_execution_routes': ['eager'] * count,
                     'native_decode_steps': 0, 'serial_decode_steps': 0}
            c['samples'].append({'pair': pair, 'warmup': pair == 0, 'arm': arm,
                'ids_sha256': v.digest(ids), 'generated_tokens': 128, 'decode_tokens': 127,
                'first_token_seconds': 0.5, 'decode_seconds': 1.0, 'decode_tokens_per_second': 127.0,
                'whole_call_seconds': 2.0, 'whole_call_tokens_per_second': 64.0,
                'events': [{'tokens': 1, 'seconds': 0.5}, {'tokens': 128, 'seconds': 1.5}],
                'route': route, 'fallback_reasons': {}})
    return {'passed': True, 'cases': [c], 'memory_after_close': {'active_allocations': 0, 'current_allocated_bytes': 0},
            'provenance': {'resolved_backend': 'hip_gfx1151'},
            'protocol': {'arms': list(v.ARMS), 'warmups_per_arm': 1, 'repetitions': 1, 'output_tokens': 128}}


def test_valid_record_and_summary():
    data = fixture()
    assert v.validate(data, 1) == 8
    assert v.summarize(data)['natural']['b3']['decode_tokens_per_second'] == 127


def test_corruptions_are_rejected():
    changes = [
        lambda d: d.update(passed=False),
        lambda d: d['memory_after_close'].update(active_allocations=1),
        lambda d: d['memory_after_close'].update(current_allocated_bytes=1),
        lambda d: d['provenance'].update(resolved_backend='other'),
        lambda d: d['protocol'].update(output_tokens=129),
        lambda d: d['cases'][0]['samples'].pop(),
        lambda d: d['cases'][0]['samples'][1].update(ids_sha256='bad'),
        lambda d: d['cases'][0]['samples'][1].update(warmup=False),
        lambda d: d['cases'][0]['samples'][1].update(decode_tokens=128),
        lambda d: d['cases'][0]['samples'][1].update(decode_seconds=2),
        lambda d: d['cases'][0]['samples'][1].update(whole_call_seconds=0.5),
        lambda d: d['cases'][0]['samples'][1].update(decode_tokens_per_second=999),
        lambda d: d['cases'][0]['samples'][1].update(whole_call_tokens_per_second=999),
        lambda d: d['cases'][0]['samples'][1]['events'][0].update(tokens=0),
        lambda d: d['cases'][0]['samples'][1]['events'][-1].update(tokens=127),
        lambda d: d['cases'][0]['samples'][1]['route'].update(specdec2_mtp2_used=False),
        lambda d: d['cases'][0]['samples'][1]['route'].update(specdec2_mtp2_cycles=2),
        lambda d: d['cases'][0]['samples'][1]['route']['specdec2_mtp2_candidate_counts'].__setitem__(0, 4),
        lambda d: d['cases'][0]['samples'][1]['route']['specdec2_mtp2_accepted_counts'].__setitem__(0, 2),
        lambda d: d['cases'][0]['samples'][1]['route'].update(native_decode_steps=1),
        lambda d: d['cases'][0]['samples'][1]['route']['specdec2_mtp2_execution_routes'].__setitem__(0, 'graph'),
        lambda d: d['cases'][0]['samples'][1]['fallback_reasons'].update(specdec2_mtp2_recovered_error=1),
        lambda d: d['cases'][0]['cross_depth_ids_equal'].update(b2=False),
        lambda d: d['cases'][0]['medians']['b2'].update(decode_tokens_per_second=999),
    ]
    for index, change in enumerate(changes):
        data = deepcopy(fixture())
        change(data)
        with pytest.raises(AssertionError, match='.*'):
            v.validate(data, 1)
