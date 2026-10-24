#!/usr/bin/env python3
"""Reconcile output counts and compare the prior adapter-check prefixes."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--first', type=Path, required=True)
parser.add_argument('--confirmation', type=Path, required=True)
parser.add_argument('--prior', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
prior = {case['id']: case for case in json.loads(args.prior.read_text())['cases']}
artifacts = {name: json.loads(path.read_text()) for name, path in [('first', args.first), ('confirmation', args.confirmation)]}
result = {'passed': False, 'inputs': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in [('first', args.first), ('confirmation', args.confirmation), ('prior', args.prior)]}, 'sweeps': {}}
for name, artifact in artifacts.items():
    assert artifact['passed']
    counts = {'samples_reconciled': 0, 'mtp_serial_tail_tokens_measured': 0, 'prior_prefix_cases': [], 'prior_prefix_arm_checks': 0}
    for case in artifact['cases']:
        for sample in case['samples']:
            route = sample['route']
            accounted = route['serial_decode_steps'] + route['native_decode_steps'] + route['specdec2_mtp2_cycles'] + sum(route['specdec2_mtp2_accepted_counts'])
            assert accounted == sample['decode_tokens'], (name, case['id'], sample['pair'], sample['arm'])
            counts['samples_reconciled'] += 1
            if sample['arm'] == 'mtp' and not sample['warmup']:
                counts['mtp_serial_tail_tokens_measured'] += route['serial_decode_steps']
        if case['id'] in prior:
            old = prior[case['id']]
            assert case['prompt_tokens'] == old['prompt_tokens']
            for run in old['runs']:
                ids = case['mtp_ids' if run['mtp'] else 'ar_ids']
                assert ids[:len(run['ids'])] == run['ids'], (name, case['id'], run['mtp'])
                counts['prior_prefix_arm_checks'] += 1
            counts['prior_prefix_cases'].append(case['id'])
    result['sweeps'][name] = counts
first = {case['id']: case for case in artifacts['first']['cases']}
for case in artifacts['confirmation']['cases']:
    for key in ('prompt_ids_sha256', 'ar_ids', 'mtp_ids'):
        assert case[key] == first[case['id']][key], (case['id'], key)
result['cross_sweep_case_arm_id_checks'] = 2 * len(first)
result['explanation'] = 'Cycles plus accepted candidates exclude ordinary serial tail steps when only one output slot remains. Adding serial/native steps reconciles every post-first-token output. Acceptance uses candidates only; throughput uses output IDs, not inferred cycle counts.'
result['passed'] = True
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['sweeps'], indent=2))
