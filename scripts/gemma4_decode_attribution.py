#!/usr/bin/env python3
"""Time fixed-context Gemma decode and optionally mark decode-step windows.

Declare --instrumentation clean for unprofiled latency evidence. Instrumented
wall times are diagnostics; marker-contained kernels provide attribution only.
Rewind preserves the teacher prefix; every replay writes its own current KV row.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def instrumentation_state(declaration, *, local=False, env=None):
    """A clean declaration is required; known profiler injection overrides it.

    This is not a proof against arbitrary external instrumentation. Record the
    declaration and detected environment keys, never infer clean from silence.
    """
    env = os.environ if env is None else env
    signals = [key for key, value in env.items() if value and
               (key.startswith(('ROCPROFILER_', 'ROCPROF_TOOL_', 'ROCTRACER_')) or
                key in ('HSA_TOOLS_LIB', 'HSA_TOOLS_LIBRARIES') or
                (key == 'LD_PRELOAD' and any(name in value.lower() for name in
                 ('rocprof', 'roctracer', 'omnitrace'))))]
    status = ('instrumented' if local or signals or declaration == 'profiled' else
              'declared_clean' if declaration == 'clean' else 'unknown')
    return {'status': status, 'declaration': declaration, 'local_instrumentation': local,
            'detected_environment_keys': sorted(signals)}


def finite_logit_bits(logits):
    values = np.asarray(logits)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError('non-finite or non-FP32 decode logits')
    return values.view(np.uint32).copy()


def summarize_samples(samples):
    if not samples:
        raise ValueError('decode samples must not be empty')
    for row in samples:
        for key in ('forward_s', 'sample_s', 'synchronize_s', 'wall_s'):
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'invalid decode timing: {key}')
        if row['wall_s'] <= 0:
            raise ValueError('decode wall must be positive')
        accounted = row['forward_s'] + row['sample_s'] + row['synchronize_s']
        if not np.isclose(accounted, row['wall_s'], rtol=1e-9, atol=1e-9):
            raise ValueError('decode wall has inconsistent phase accounting')
    return {key: statistics.median(row[key] for row in samples)
            for key in ('forward_s', 'sample_s', 'synchronize_s', 'wall_s')}


def kernel_family(name):
    if 'gemma4_attention' in name:
        return 'attention'
    if 'selected' in name:
        return 'expert_projection'
    if 'q8_0_t16_gemv' in name:
        return 'dense_projection'
    if 'gguf_k_pack8_prefill_out' in name:
        return 'output_head'
    return 'other'


def summarize_trace_windows(windows, kernels):
    """Attribute fully contained launches; never infer unprofiled host overhead."""
    seen = set()
    for rows in (windows, kernels):
        for row in rows:
            if (not np.isfinite(row['start']) or not np.isfinite(row['end']) or
                    row['end'] <= row['start']):
                raise ValueError('invalid trace timestamp or duration')
    for window in windows:
        if window['message'] in seen:
            raise ValueError('duplicate decode marker')
        seen.add(window['message'])
    ordered = sorted(windows, key=lambda row: row['start'])
    if any(a['end'] > b['start'] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('overlapping decode markers')
    result = []
    for window in windows:
        families = {}
        launches = []
        for kernel in kernels:
            overlaps = kernel['start'] < window['end'] and kernel['end'] > window['start']
            contained = kernel['start'] >= window['start'] and kernel['end'] <= window['end']
            if overlaps and not contained:
                raise ValueError('kernel crosses decode marker boundary')
            if not contained:
                continue
            duration = kernel['end'] - kernel['start']
            launches.append(kernel)
            family = families.setdefault(kernel_family(kernel['name']), {'ns': 0, 'launches': 0})
            family['ns'] += duration
            family['launches'] += 1
        if not launches:
            raise ValueError('decode marker has no kernels')
        result.append({'marker': window['message'], 'launches': len(launches), 'families': families,
                       'kernel_sum_ns': sum(x['ns'] for x in families.values()),
                       'profiled_marker_ns': window['end'] - window['start'],
                       'scope': 'profiled kernel-family attribution, not unprofiled wall/host overhead'})
    if not result:
        raise ValueError('no decode markers')
    return result


def decode_orders(repeats, *, paired):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    return [('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')
            for i in range(repeats)] if paired else [('baseline',)] * repeats


@contextmanager
def observe_build_keys(enabled):
    """Time CPU-only cache-key construction, not ctypes launches or stream waits.

    Timer-call and wrapper overhead perturb this diagnostic. Do not subtract it
    from clean decode wall or describe it as total host/driver overhead.
    """
    stats = {'calls': 0, 'wall_ns': 0}
    if not enabled:
        yield stats
        return
    from hipengine.core import build
    original = build._build_fast_key
    def observed(**kwargs):
        start = time.perf_counter_ns()
        try:
            return original(**kwargs)
        finally:
            stats['wall_ns'] += time.perf_counter_ns() - start
            stats['calls'] += 1
    build._build_fast_key = observed
    try:
        yield stats
    finally:
        build._build_fast_key = original


@contextmanager
def observe_host_launches(enabled):
    """Time signed C launch exports without GPU sync or profiler injection.

    This observes host call wall, which can include driver/queue waits. It is
    neither device duration nor total Python overhead. Wrappers add diagnostic
    overhead, so the enclosing decode wall is not clean latency evidence.
    """
    stats = {'calls': 0, 'wall_ns': 0}
    if not enabled:
        yield stats
        return
    from hipengine.core.ctypes_cache import signed_kernel_fn
    modules = [module for name, module in list(sys.modules.items())
               if name.startswith('hipengine.kernels.') and
               getattr(module, 'signed_kernel_fn', None) is signed_kernel_fn]
    def observed(library, symbol, argtypes, restype):
        fn = signed_kernel_fn(library, symbol, argtypes, restype)
        if restype is not ctypes.c_int:
            return fn  # size/geometry exports are not launch calls
        def timed(*args):
            start = time.perf_counter_ns()
            try:
                return fn(*args)
            finally:
                stats['wall_ns'] += time.perf_counter_ns() - start
                stats['calls'] += 1
        return timed
    for module in modules:
        module.signed_kernel_fn = observed
    try:
        yield stats
    finally:
        for module in modules:
            module.signed_kernel_fn = signed_kernel_fn


@contextmanager
def decode_instrumentation(marker, profiler, name):
    pushed = enabled = False
    try:
        if marker is not None:
            marker.roctxRangePushA(name.encode())
            pushed = True
        if profiler is not None:
            profiler.enable()
            enabled = True
        yield
    finally:
        try:
            if enabled:
                profiler.disable()
        finally:
            if pushed:
                marker.roctxRangePop()


def _run_decode(args, llm, runner, loading, marker):
    from scripts.gemma4_campaign_bench import exact_prompt_ids
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.hip import get_hip_runtime
    runtime = get_hip_runtime()
    generator = llm._get_text_generator()
    python_profiler = None
    if args.python_profile:
        import cProfile
        python_profiler = cProfile.Profile()
    attention_original = None
    libraries = {}
    selected = {}
    try:
        instrumentation = instrumentation_state(args.instrumentation,
                                                local=marker is not None or python_profiler is not None or args.host_build_timing or args.host_launch_timing)
        report = {'kind': 'gemma4_fixed_context_decode_attribution',
                  'profiled': instrumentation['status'] == 'instrumented',
                  'performance_claim': instrumentation['status'] == 'declared_clean',
                  'instrumentation': instrumentation,
                  'protocol': 'fixed teacher token and KV prefix, rewind each replay, one untimed decode warmup, median repeated forward/argmax/sync phase wall',
                  'loading': loading, 'cases': [],
                  'provenance': collect_artifact_provenance(repo_root=ROOT, model_path=args.artifact,
                      quant='UD-Q4_K_XL', kv_dtype='bf16', warmups=1, repetitions=args.repeats,
                      profiler=instrumentation)}
        if args.attention_library:
            from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged
            for arm, path in (('baseline', args.attention_library), ('candidate', args.candidate_library)):
                if path is not None:
                    libraries[arm] = ctypes.CDLL(str(path.resolve()))
                    report[f'{arm}_attention_library_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            selected['current'] = libraries['baseline']
            attention_original = gemma4_attention_staged.build_gemma4_attention_staged
            gemma4_attention_staged.build_gemma4_attention_staged = lambda **kwargs: selected['current']
        orders = decode_orders(args.repeats, paired=args.candidate_library is not None)
        for context in args.contexts:
            if libraries:
                selected['current'] = libraries['baseline']
            ids = exact_prompt_ids(generator.tokenize, context)
            runner.reset()
            logits = runner.forward(ids)
            runtime.device_synchronize()
            finite_logit_bits(logits)
            teacher = runner.next_token(logits)
            reference = runner.forward([teacher])
            runtime.device_synchronize()
            reference_bits = finite_logit_bits(reference)
            if args.candidate_library:
                selected['current'] = libraries['candidate']
                runner.rewind(context)
                warm = runner.forward([teacher])
                runtime.device_synchronize()
                if not np.array_equal(reference_bits, finite_logit_bits(warm)):
                    raise RuntimeError('candidate warmup changed baseline logits')
            rows = []
            for repeat, order in enumerate(orders):
                for arm in order:
                    if libraries:
                        selected['current'] = libraries[arm]
                    runner.rewind(context)
                    runtime.device_synchronize()
                    name = f'gemma4_decode_c{context}_r{repeat}'
                    if args.candidate_library:
                        name += f'_{arm}'
                    with observe_build_keys(args.host_build_timing) as build_keys, observe_host_launches(args.host_launch_timing) as host_launches, decode_instrumentation(marker, python_profiler, name):
                        start = time.perf_counter()
                        actual = runner.forward([teacher])
                        forward_end = time.perf_counter()
                        sampled = runner.next_token(actual)
                        sample_end = time.perf_counter()
                        runtime.device_synchronize()
                        end = time.perf_counter()
                    if not np.array_equal(reference_bits, finite_logit_bits(actual)):
                        raise RuntimeError('fixed-context decode replay changed baseline logits')
                    row = {'repeat': repeat, 'arm': arm, 'order': list(order), 'marker': name,
                           'forward_s': forward_end-start, 'sample_s': sample_end-forward_end,
                           'synchronize_s': end-sample_end, 'wall_s': end-start,
                           'sampled_token': sampled,
                           'logits_sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
                    if args.host_build_timing:
                        row['host_build_key_diagnostic'] = dict(build_keys)
                    if args.host_launch_timing:
                        row['host_c_launch_diagnostic'] = dict(host_launches)
                    rows.append(row)
                    print(json.dumps({'context': context, **row}, allow_nan=False), flush=True)
            case = {'context_tokens': context, 'keys': context+1, 'teacher_token': teacher,
                    'prompt_ids_sha256': hashlib.sha256(np.array(ids, dtype=np.int32).tobytes()).hexdigest(),
                    'samples': rows, 'replay_logits_bitwise_equal': True}
            if args.candidate_library:
                medians = {arm: summarize_samples([r for r in rows if r['arm'] == arm]) for arm in libraries}
                case['median_by_arm'] = medians
                case['candidate_over_baseline_wall'] = medians['candidate']['wall_s'] / medians['baseline']['wall_s']
            else:
                case['median'] = summarize_samples(rows)
            report['cases'].append(case)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    finally:
        if attention_original is not None:
            gemma4_attention_staged.build_gemma4_attention_staged = attention_original
        if python_profiler is not None:
            args.python_profile.parent.mkdir(parents=True, exist_ok=True)
            python_profiler.dump_stats(str(args.python_profile))
    return 0


def main(argv=None):
    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, _resolve_generator
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument('--contexts', type=int, nargs='+', default=[513, 2053, 8191])
    p.add_argument('--repeats', type=int, default=9)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--attention-library', type=Path, help='Frozen staged-attention diagnostic baseline')
    p.add_argument('--candidate-library', type=Path, help='Pair against baseline, alternating AB/BA in one residency')
    p.add_argument('--roctx-library', type=Path)
    p.add_argument('--instrumentation', choices=('clean', 'profiled'), help='Declare external instrumentation; known injection overrides clean')
    p.add_argument('--host-launch-timing', action='store_true', help='Time signed native launch exports on host; includes driver/queue waits, not device duration')
    p.add_argument('--host-build-timing', action='store_true', help='Time CPU cache-key construction only; instrumented diagnostic, not total host/driver overhead')
    p.add_argument('--python-profile', type=Path, help='cProfile decode only; diagnostic overhead, not latency evidence')
    args = p.parse_args(argv)
    if args.repeats < 1 or not args.contexts or any(n < 1 for n in args.contexts):
        p.error('repeats and contexts must be positive')
    if len(set(args.contexts)) != len(args.contexts):
        p.error('contexts must be unique')
    if args.candidate_library and not args.attention_library:
        p.error('--candidate-library requires --attention-library as the baseline')
    marker = ctypes.CDLL(str(args.roctx_library.resolve())) if args.roctx_library else None
    if marker is not None:
        marker.roctxRangePushA.argtypes = [ctypes.c_char_p]
        marker.roctxRangePushA.restype = ctypes.c_int
        marker.roctxRangePop.argtypes = []
        marker.roctxRangePop.restype = ctypes.c_int
    llm, runner, loading = _resolve_generator(args.artifact, max(args.contexts) + 16)
    try:
        return _run_decode(args, llm, runner, loading, marker)
    finally:
        llm.close()


if __name__ == '__main__':
    raise SystemExit(main())
