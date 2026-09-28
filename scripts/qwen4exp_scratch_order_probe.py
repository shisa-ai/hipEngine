"""Ordered GDN scratch-trigger pilot, not a whole-model throttle verdict.

Run sham and scratch arms in separate fresh processes, alternating process order.
The default victim is strict scratch-free GDN. Select wave-norm to test a
scratch-enabled victim and dense-wide to grow the private segment from 36 to
92 bytes in the pinned gfx1151 build. Never warm the selected trigger before
the before-window. This tests only the specified kernels on the default queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc, free, copy_device_to_host
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from tests.test_gpu_gguf_q8_0_dense_wide import make_q8_0_weight, _dequantize_weight, _f16_reference
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from tests.test_gpu_qwen4exp_gdn_register import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('sham', 'scratch'), required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--victim', choices=('strict', 'wave-norm'), default='strict')
    parser.add_argument('--trigger', choices=('gdn', 'dense-wide'), default='gdn')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 3:
        parser.error('positive tokens and at least three repeats required')
    f = Fixture(args.tokens)
    rt = f.runtime
    start, stop = rt.event_create(), rt.event_create()
    victim = (gdn.qwen4_exp_gdn_prefill_f32 if args.victim == 'strict'
              else gdn.qwen4_exp_gdn_wave_norm_prefill_f32)
    trigger = gdn.qwen4_exp_gdn_wave_norm_prefill_f32 if args.arm == 'scratch' else victim
    dense_buffers = []
    dense_library = None
    dense_check = None
    telemetry = []

    def snapshot(stage):
        values = {}
        for device in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
            for name in ('gpu_busy_percent', 'pp_dpm_sclk', 'pp_dpm_mclk'):
                path = device / name
                try:
                    values[str(path)] = path.read_text().strip()
                except OSError:
                    pass
        telemetry.append({'stage': stage, 'monotonic_ns': time.monotonic_ns(), 'values': values})

    if args.trigger == 'dense-wide':
        # Allocate/load but do NOT execute the trigger before the before-window.
        rows, k, n = 257, 256, 256
        x = np.random.default_rng(713).normal(0, .1, (rows, k)).astype(np.float32)
        weights = make_q8_0_weight(n, k)
        dense_out = np.empty((rows, n), dtype=np.float32)
        reference = _f16_reference(x, _dequantize_weight(weights, k))
        dense_library = dw.build_gguf_q8_0_dense_wide(load=True)
        for array in (x, weights, dense_out):
            buffer = malloc(array.nbytes, runtime=rt)
            dense_buffers.append(buffer)
            copy_host_to_device(buffer, host_array_ptr(array), runtime=rt)

    def dense_launch():
        rt.device_synchronize()
        rt.event_record(start)
        dw.gguf_q8_0_dense_wide256_f32_f32_out(
            *[x.ptr for x in dense_buffers], rows, k, n, library=dense_library, runtime=rt)
        rt.event_record(stop)
        rt.event_synchronize(stop)
        elapsed = rt.event_elapsed_time_ms(start, stop)
        copy_device_to_host(host_array_ptr(dense_out), dense_buffers[2], runtime=rt)
        error = float(np.max(np.abs(dense_out - reference)))
        scale = float(np.max(np.abs(reference)))
        assert error <= 1e-3 * scale
        return elapsed, {'max_abs_error': error, 'reference_scale': scale,
                         'f16_reference_pass': True}

    def launch(fn):
        copy_host_to_device(f.states[0], host_array_ptr(f.state), runtime=rt)
        rt.device_synchronize()
        rt.event_record(start)
        fn(*[x.ptr for x in f.inputs], f.states[0].ptr, f.outputs[0].ptr,
           args.tokens, 16, 48, 128, 128, library=f.library, runtime=rt)
        rt.event_record(stop)
        rt.event_synchronize(stop)
        return rt.event_elapsed_time_ms(start, stop)

    try:
        for _ in range(3):
            launch(victim)
        expected = f.result(False)
        snapshot('before')
        before = [launch(victim) for _ in range(args.repeats)]
        snapshot('pre-trigger')
        if args.arm == 'scratch' and args.trigger == 'dense-wide':
            trigger_ms, dense_check = dense_launch()
        else:
            trigger_ms = launch(trigger)
            for actual, reference_value in zip(f.result(False), expected):
                np.testing.assert_array_equal(actual.view(np.uint32), reference_value.view(np.uint32))
        snapshot('post-trigger')
        after = [launch(victim) for _ in range(args.repeats)]
        snapshot('after')
        actual = f.result(False)
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value.view(np.uint32), reference.view(np.uint32))
        library_path = Path(f.library._name).resolve()
        report = {
            'schema': 1, 'kind': 'qwen4exp_gdn_ordered_scratch_pilot',
            'command': sys.argv, 'arm': args.arm,
            'shape': {'tokens': args.tokens, 'Hk': 16, 'Hv': 48, 'Dk': 128, 'Dv': 128},
            'victim': victim.__name__,
            'trigger': ('dense_wide256_f32_f32_out' if args.arm == 'scratch' and args.trigger == 'dense-wide'
                        else trigger.__name__),
            'dense_shape': [rows, k, n] if dense_library else None,
            'dense_reference_check': dense_check,
            'clock_snapshots': telemetry,
            'dense_library': ({'path': dense_library._name,
                               'sha256': hashlib.sha256(Path(dense_library._name).read_bytes()).hexdigest()}
                              if dense_library else None),
            'before_ms': before, 'after_ms': after, 'trigger_ms': trigger_ms,
            'after_over_before': statistics.median(after) / statistics.median(before),
            'output_and_state_exact': True,
            'hashes': [hashlib.sha256(x.tobytes()).hexdigest() for x in actual],
            'build': {'path': str(library_path),
                      'sha256': hashlib.sha256(library_path.read_bytes()).hexdigest()},
            'host': {'machine_id': Path('/etc/machine-id').read_text().strip(),
                     'uname': list(os.uname())},
            'performance_claim': False,
            'limits': ['specified GDN victim only', 'not a serving or model benchmark',
                       'boundary clock snapshots only; allocation needs separate trace'],
        }
    finally:
        rt.event_destroy(start)
        rt.event_destroy(stop)
        for buffer in reversed(dense_buffers):
            free(buffer, runtime=rt)
        f.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'ratio': report['after_over_before'], 'exact': True}))


if __name__ == '__main__':
    main()
