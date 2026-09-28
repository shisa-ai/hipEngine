#!/usr/bin/env python3
"""Audit separately marked requests: role time, interval union and observed scratch.

HIP API durations are host intervals, not device-copy time or exclusive overhead.
Scratch_Size is profiler evidence, not a replacement for code-object metadata or
an ordered trigger/victim experiment. Missing resource fields remain unknown.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen4exp_role_analyze import analyze


def interval_union_ns(intervals):
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("reversed timestamp interval")
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def resource_census(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['Kernel_Name']].append(row)
    result = []
    for name, launches in groups.items():
        result.append({
            'kernel': name,
            'launches': len(launches),
            'kernel_sum_ms': sum(int(r['End_Timestamp']) - int(r['Start_Timestamp'])
                                 for r in launches) / 1e6,
            'scratch_bytes_observed': sorted({int(r['Scratch_Size']) for r in launches
                                              if r.get('Scratch_Size', '') != ''}),
            'scratch_unknown_launches': sum(r.get('Scratch_Size', '') == '' for r in launches),
            'vgpr_observed': sorted({int(r['VGPR_Count']) for r in launches
                                     if r.get('VGPR_Count', '') != ''}),
            'lds_bytes_observed': sorted({int(r['LDS_Block_Size']) for r in launches
                                          if r.get('LDS_Block_Size', '') != ''}),
        })
    return sorted(result, key=lambda r: (-r['kernel_sum_ms'], r['kernel']))


def _read(root, pattern):
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f'expected one {pattern}, found {paths}')
    with paths[0].open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit(root, prefix):
    markers = _read(root, '*_marker_api_trace.csv')
    kernels = _read(root, '*_kernel_trace.csv')
    apis = _read(root, '*_hip_api_trace.csv')
    copies = (_read(root, '*_memory_copy_trace.csv')
              if list(root.glob('*_memory_copy_trace.csv')) else None)
    scratch = (_read(root, '*_scratch_memory_trace.csv')
               if list(root.glob('*_scratch_memory_trace.csv')) else None)
    windows = [r for r in markers if r['Function'].startswith(prefix)]
    if not windows:
        raise ValueError(f'no measured windows for {prefix}')
    names = [r['Function'] for r in windows]
    if len(set(names)) != len(names):
        raise ValueError('measured marker names must be unique')
    reports = []
    for window in windows:
        start, end = int(window['Start_Timestamp']), int(window['End_Timestamp'])
        def inside(row):
            return start <= int(row['Start_Timestamp']) <= int(row['End_Timestamp']) <= end
        selected = [r for r in kernels if inside(r)]
        selected_apis = [r for r in apis if inside(r)]
        selected_copies = None if copies is None else [r for r in copies if inside(r)]
        selected_scratch = None if scratch is None else [r for r in scratch if inside(r)]
        device_intervals = [(int(r['Start_Timestamp']), int(r['End_Timestamp']))
                            for r in selected + (selected_copies or [])]
        device_union = interval_union_ns(device_intervals)
        # Analyze each repeat independently; never divide a warmup-inclusive trace.
        roles = analyze(root, window['Function'])
        if len(selected) != roles['kernel_rows']:
            raise ValueError('kernel window disagrees with role analyzer')
        kernel_union = interval_union_ns([(int(r['Start_Timestamp']), int(r['End_Timestamp']))
                                          for r in selected])
        api_groups = defaultdict(list)
        for row in selected_apis:
            api_groups[row['Function']].append((int(row['Start_Timestamp']), int(row['End_Timestamp'])))
        reports.append({
            'marker': window['Function'],
            'window_ms': (end - start) / 1e6,
            'kernel_union_ms': kernel_union / 1e6,
            'outside_kernel_union_ms': (end - start - kernel_union) / 1e6,
            'device_copy_rows': selected_copies,
            'scratch_operation_rows': selected_scratch,
            'kernel_and_copy_union_ms': None if copies is None else device_union / 1e6,
            'outside_kernel_and_copy_union_ms': (None if copies is None else
                                                 (end - start - device_union) / 1e6),
            'roles': roles,
            'resources': resource_census(selected),
            'hip_api_host_intervals': {
                name: {'calls': len(intervals),
                       'sum_ms': sum(b-a for a, b in intervals) / 1e6,
                       'union_ms': interval_union_ns(intervals) / 1e6}
                for name, intervals in sorted(api_groups.items())
            },
        })
    return {'schema': 1, 'kind': 'qwen4exp_marked_trace_audit',
            'trace_dir': str(root), 'measure_prefix': prefix,
            'performance_claim': False,
            'whole_capture_scratch_operations': scratch,
            'notes': ['HIP API intervals overlap device work and are not additive overhead.',
                      'Outside-kernel time is not necessarily host overhead: copies may be absent from kernel trace.',
                      'Scratch_Size is observed profiler data; private-segment metadata and throttle causality need separate verification.'],
            'windows': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--measure-prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace_dir, args.measure_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'windows': len(report['windows'])}))


if __name__ == '__main__':
    main()
