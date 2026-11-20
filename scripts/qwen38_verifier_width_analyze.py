#!/usr/bin/env python3
"""Reduce fixed-prefix verifier ROCTx windows to kernel/API/copy attribution.

Uses the existing overlap-aware attribution reducer. HIP API duration overlaps
GPU execution and is not an additive host-cost estimate. Raw traces stay local;
the compact output retains full kernel names and a hash of each input artifact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_c1_decode_attribution import (
    summarize as build_attribution, read_kernels, read_hip_api, read_memory_copies,
)


def measured_windows(rows):
    groups = defaultdict(list)
    ordered = []
    for row in rows:
        label = row['Function']
        if not label.startswith(('verify:', 'ar_step:')):
            continue
        begin, end = int(row['Start_Timestamp']), int(row['End_Timestamp'])
        if end <= begin:
            raise ValueError(f'invalid marker interval: {label}')
        ordered.append((begin, end))
        groups[label].append(dict(start_ns=begin, end_ns=end))
    if not ordered:
        raise ValueError('no measured verifier or AR marker windows')
    ordered.sort()
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError('measured ranges overlap; attribution would double count')
    return dict(groups)


def validate_coverage(run, windows):
    protocol = run['protocol']
    expected = {f'verify:c{c}:r{w}': protocol['repetitions']
                for c in protocol['contexts'] for w in protocol['widths']}
    expected.update({f'ar_step:c{c}': protocol['repetitions'] for c in protocol['contexts']})
    actual = {label: len(ranges) for label, ranges in windows.items()}
    if actual != expected:
        raise ValueError(f'marker coverage differs from protocol: {actual} != {expected}')
    if (not run['exact_all'] or len(run['rows']) != sum(n for k, n in expected.items() if k.startswith('verify:'))
            or not all(row['exact'] for row in run['rows'])):
        raise ValueError('profile run has incomplete or non-exact verifier results')


def one_trace(root, suffix):
    paths = list(root.glob(f'*_{suffix}.csv'))
    if len(paths) != 1:
        raise ValueError(f'expected one {suffix} CSV in {root}, found {len(paths)}')
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_root', type=Path)
    parser.add_argument('--run-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {key: one_trace(args.trace_root, suffix) for key, suffix in (
        ('kernels', 'kernel_trace'), ('markers', 'marker_api_trace'),
        ('hip_api', 'hip_api_trace'), ('copies', 'memory_copy_trace'))}
    with paths['markers'].open(newline='') as handle:
        windows = measured_windows(csv.DictReader(handle))
    kernels = read_kernels(paths['kernels'])
    for row in kernels:
        row['family'] = row['kernel']  # Keep template shape/variant, not coarse quant families.
    apis = read_hip_api(paths['hip_api'])
    copies = read_memory_copies(paths['copies'])
    run = json.loads(args.run_json.read_text())
    validate_coverage(run, windows)
    summaries = {}
    for label, ranges in sorted(windows.items()):
        summary = build_attribution(windows=ranges, kernels=kernels, hip_api=apis,
                                    copies=copies, step_walls_ms=[])
        summaries[label] = summary
    artifact = dict(schema=1, kind='qwen38_verifier_width_attribution', status='diagnostic',
                    performance_claim=False, provenance=run['provenance'], profile=run['profile'],
                    session=run['session'], protocol=run['protocol'], exact_all=run['exact_all'],
                    summaries=summaries, inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in [args.run_json, *paths.values()]},
                    command=[sys.executable, *sys.argv])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
