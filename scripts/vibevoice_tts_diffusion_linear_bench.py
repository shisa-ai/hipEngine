"""Compare diffusion reduction schedules on identical recorded conditions/noise.

Timings cover a complete 20-step solve with final readback, not TTS end-to-end.
Arithmetic differences against the torch fixture and 256-thread path are diagnostic;
the unchanged numerical tests and generated-audio suite are the acceptance gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import shlex
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--all-calls', action='store_true')
    parser.add_argument('--session', action='store_true', help='also time four interleaved full-session arms')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    import numpy as np
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_tts import load_vibevoice_tts_diffusion_head
    from hipengine.runtime.vibevoice_tts_diffusion import VibevoiceTTSDiffusionHeadGPU
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_out_bf16

    spec, weights, _, _ = load_vibevoice_tts_diffusion_head(resolve_model_path('microsoft/VibeVoice-1.5B'))
    heads = {}
    results = {}
    def make_linear(threads):
        def linear(self, key, x, out, rows):
            weight, n, k = self._linears[key]
            dense_gemv_out_bf16(x, weight.ptr, out, rows, k, n, threads=threads)
        return linear
    try:
        for name, threads in [('old64', 64), ('candidate128', 128), ('reference256', 256)]:
            h = VibevoiceTTSDiffusionHeadGPU(spec, weights)
            heads[name] = h
            if threads is not None:
                h._gemv = MethodType(make_linear(threads), h)
        for fixture in ('single', 'two'):
            with np.load(ROOT / f'tests/fixtures/vibevoice_tts/{fixture}_diffusion.npz') as archive:
                data = dict(archive)
            def solve(h, call, collect=None):
                return h.sample_speech_tokens(data[f'call{call}_condition'], data[f'call{call}_neg_condition'],
                    float(data[f'call{call}_cfg_scale']), data[f'call{call}_initial_noise'], collect=collect)
            differences = {name: 0 for name in heads}
            fixture_drift = {name: {'eps': 0.0, 'speech': 0.0} for name in heads}
            steps = 0
            for call in range(int(data['num_calls_recorded']) if args.all_calls else 1):
                collected = {}
                for name, head in heads.items():
                    snapshots = []
                    solve(head, call, snapshots)
                    collected[name] = np.stack([np.stack([s['eps'], s['speech']]) for s in snapshots])
                ref = collected['reference256']
                steps += len(ref)
                for name in heads:
                    differences[name] += int(np.count_nonzero(collected[name] != ref))
                    for index, field in enumerate(('eps', 'speech')):
                        oracle = data[f'call{call}_{field}'].astype(np.float64)
                        actual = collected[name][:, index].astype(np.float64)
                        drift = float(np.abs(actual - oracle).max() / max(np.abs(oracle).max(), 1e-12))
                        fixture_drift[name][field] = max(fixture_drift[name][field], drift)
            timings = {name: [] for name in heads}
            names = list(heads)
            for rep in range(args.repeats):
                for name in names if rep % 2 == 0 else names[::-1]:
                    h = heads[name]
                    h.runtime.device_synchronize()
                    start = time.perf_counter()
                    solve(h, 0)
                    h.runtime.device_synchronize()
                    timings[name].append(time.perf_counter() - start)
            results[fixture] = {'solver_steps_compared': steps,
                'eps_and_latent_elements_differing_from_reference256': differences,
                'torch_fixture_max_call_normalized_drift': fixture_drift,
                'solve_seconds': timings,
                'median_solve_seconds': {name: statistics.median(ts) for name, ts in timings.items()}}
            print(fixture, results[fixture], flush=True)
    finally:
        for h in heads.values():
            h.close()
    sessions = []
    if args.session:
        from scripts.vibevoice_tts_session_bench import bench
        original = VibevoiceTTSDiffusionHeadGPU._gemv
        try:
            for arm in ('candidate128', 'old64', 'candidate128', 'old64'):
                VibevoiceTTSDiffusionHeadGPU._gemv = make_linear(128 if arm == 'candidate128' else 64)
                r = bench('microsoft/VibeVoice-1.5B', ROOT / 'tests/fixtures/vibevoice_tts', 3)
                sessions.append({'arm': arm, 'warm_seconds': r['warm_synthesis_seconds'],
                    'pooled_rtf': r['pooled_rtf'], 'stages': r['stages'],
                    'chain_exact': r['chain_exact'], 'negative_conditions_match': r['negative_conditions_match']})
                assert r['chain_exact'] and r['negative_conditions_match'], sessions[-1]
                print('session', sessions[-1], flush=True)
        finally:
            VibevoiceTTSDiffusionHeadGPU._gemv = original
    from scripts.vibevoice_tts_quality_suite import _gpu_name
    result = {'status': 'diagnostic-arithmetic-comparison-not-task-acceptance', 'model': 'microsoft/VibeVoice-1.5B', 'quant': 'bf16', 'host': socket.gethostname(),
        'gpu': _gpu_name(), 'backend': h.kernels.backend,
        'environment': {'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS')},
        'workload': 'two CFG rows; 20 solver steps; recorded single/two-speaker conditions and noise',
        'command': 'uv run python scripts/vibevoice_tts_diffusion_linear_bench.py ' + shlex.join(sys.argv[1:]),
        'correctness': 'arithmetic differences reported; numerical/task gates recorded separately; optional session checks are recorded in sessions',
        'results': results, 'sessions': sessions}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
