#!/usr/bin/env python3
"""Measure staged Gemma attention on deterministic prefill block geometries.

Synthetic kernel measurements, not model throughput. An explicit frozen shared
library permits baseline/candidate measurements on identical device buffers.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Query blocks match the runner's chunked prefill, including a partial tail.
CASES = ((512, 513), (512, 2053), (512, 8191), (127, 17000))


def inputs(tokens, keys, heads, kv_heads, dim, window, seed=20261003):
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(shape, dtype=np.float32) * 0.7 for shape in
              ((tokens, heads, dim), (keys, kv_heads, dim), (keys, kv_heads, dim))]
    packed = []
    for array in arrays:
        bits = array.view(np.uint32)
        packed.append(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16))
    offset = keys - tokens
    j = np.arange(keys)[None, :]
    own = offset + np.arange(tokens)[:, None]
    mask = j <= own
    if window:
        mask &= j > own - window
    return (*packed, mask.astype(np.uint8), offset)


def capture_bf16(launch, poison, readback):
    """Capture an arm independently, rejecting missing stores outside timing."""
    poison()
    launch()
    output = readback()
    values = (output.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError('non-finite or unwritten BF16 output')
    return output


def require_bitwise_equal(baseline, candidate):
    """A mandatory comparison, including when Python assertions are disabled."""
    equal = bool(np.array_equal(baseline, candidate))
    if not equal:
        raise RuntimeError('paired bitwise strict-parity failure')
    return equal


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path)
    p.add_argument('--baseline-library', type=Path,
                   help='Frozen baseline .so; alternate paired launches on shared buffers')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--case', type=int, choices=range(len(CASES)))
    args = p.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0:
        p.error('repeats must be positive and warmup nonnegative')
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import Gemma4AttentionScratch
    lib = ctypes.CDLL(str(args.library.resolve())) if args.library else staged.build_gemma4_attention_staged(load=True)
    baseline = ctypes.CDLL(str(args.baseline_library.resolve())) if args.baseline_library else None
    rt = get_hip_runtime()
    report = {'kind': 'gemma4_staged_prefill_kernel_measurement', 'model_throughput_claim': False,
              'workload': 'deterministic synthetic BF16 Q/K/V; causal prefill block',
              'provenance': collect_artifact_provenance(repo_root=ROOT, quant='synthetic_bf16', kv_dtype='bf16'),
              'library': lib._name, 'library_sha256': hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(staged._SOURCE.read_bytes()).hexdigest() if not args.library else None,
              'timing': 'HIP events on default stream; reusable workspace; no profiler; median of repetitions',
              'correctness': 'independent per-arm capture after output and FP32 workspace NaN poisoning; finite BF16 output and mandatory raw-bit parity',
              'warmup': args.warmup, 'repetitions': args.repeats, 'cases': []}
    if baseline is not None:
        report['baseline_library'] = baseline._name
        report['baseline_library_sha256'] = hashlib.sha256(Path(baseline._name).read_bytes()).hexdigest()
        report['timing'] = ('HIP events on default stream; shared device buffers and reusable workspace; '
                            'alternating AB/BA pairs; per-arm median; no profiler')
    for index, (tokens, keys) in enumerate(CASES):
        if args.case is not None and index != args.case:
            continue
        for dim, kv_heads, window in ((256, 8, 1024), (512, 2, 0)):
            q, k, v, mask, offset = inputs(tokens, keys, 16, kv_heads, dim, window)
            out = np.empty_like(q)
            buffers = []
            scratch = Gemma4AttentionScratch()
            start = stop = None
            try:
                for array in (q, k, v, mask, out):
                    buffer = malloc(array.nbytes); buffers.append(buffer)
                    copy_host_array_to_device(buffer, array)
                def launch(library=lib):
                    return staged.gemma4_attention_staged_bf16(
                        *(b.ptr for b in buffers), tokens=tokens, keys=keys, num_heads=16,
                        num_kv_heads=kv_heads, head_dim=dim, scale=1.0, window=window,
                        row_offset=offset, scratch=scratch, library=library, runtime=rt)
                plan = launch()
                arms = {'candidate': lib} if baseline is None else {'baseline': baseline, 'candidate': lib}
                captures = {}
                output_poison = np.full_like(out, 0x7fc1)
                workspace = scratch.buffer(plan.workspace_bytes, stream=0, runtime=rt)
                workspace_poison = np.full(plan.workspace.total_floats, np.nan, dtype=np.float32)
                def poison():
                    copy_host_array_to_device(buffers[-1], output_poison)
                    copy_host_array_to_device(workspace, workspace_poison)
                def readback():
                    rt.device_synchronize()
                    copy_device_to_host(out.ctypes.data, buffers[-1])
                    return out.copy()
                for name, library in arms.items():
                    for _ in range(args.warmup + 1):
                        launch(library)
                    rt.device_synchronize()
                    captures[name] = capture_bf16(lambda: launch(library), poison, readback)
                equal = (require_bitwise_equal(captures['baseline'], captures['candidate'])
                         if baseline is not None else None)
                start, stop = rt.event_create(), rt.event_create()
                samples = {name: [] for name in arms}
                for repeat in range(args.repeats):
                    names = list(arms) if repeat % 2 == 0 else list(reversed(arms))
                    for name in names:
                        rt.event_record(start); launch(arms[name]); rt.event_record(stop)
                        rt.event_synchronize(stop)
                        samples[name].append(rt.event_elapsed_time_ms(start, stop))
                times = samples['candidate']
                out = captures['candidate']
                row = {'tokens': tokens, 'keys': keys, 'heads': 16, 'kv_heads': kv_heads,
                       'head_dim': dim, 'window': window, 'row_offset': offset,
                       'plan': plan.describe(), 'times_ms': times, 'median_ms': statistics.median(times),
                       'output_sha256': hashlib.sha256(out.tobytes()).hexdigest()}
                if baseline is not None:
                    row['baseline_times_ms'] = samples['baseline']
                    row['baseline_median_ms'] = statistics.median(samples['baseline'])
                    row['candidate_over_baseline'] = row['median_ms'] / row['baseline_median_ms']
                    row['outputs_bitwise_equal'] = equal
                    row['baseline_output_sha256'] = hashlib.sha256(captures['baseline'].tobytes()).hexdigest()
                report['cases'].append(row)
                print(json.dumps(row), flush=True)
            finally:
                scratch.close()
                for event in (start, stop):
                    if event is not None: rt.event_destroy(event)
                for buffer in buffers: free(buffer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
