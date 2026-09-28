"""Ordered GDN scratch-trigger pilot, not a whole-model throttle verdict.

Run sham and scratch arms in separate fresh processes, alternating process order.
The victim is the strict scratch-free GDN kernel; the trigger is wave-norm GDN
(36-byte private segment in the pinned gfx1151 build). Never warm the trigger
before the before-window. This tests effects on this victim only, not every
scratch-enabled dispatch or other command queues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hipengine.core.memory import copy_host_to_device, host_array_ptr
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from tests.test_gpu_qwen4exp_gdn_register import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('sham', 'scratch'), required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=21)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 3:
        parser.error('positive tokens and at least three repeats required')
    f = Fixture(args.tokens)
    rt = f.runtime
    start, stop = rt.event_create(), rt.event_create()
    victim = gdn.qwen4_exp_gdn_prefill_f32
    trigger = gdn.qwen4_exp_gdn_wave_norm_prefill_f32 if args.arm == 'scratch' else victim

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
        before = [launch(victim) for _ in range(args.repeats)]
        trigger_ms = launch(trigger)
        # The trigger is an existing exact variant; still verify this input.
        for actual, reference in zip(f.result(False), expected):
            np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
        after = [launch(victim) for _ in range(args.repeats)]
        actual = f.result(False)
        for value, reference in zip(actual, expected):
            np.testing.assert_array_equal(value.view(np.uint32), reference.view(np.uint32))
        library_path = Path(f.library._name).resolve()
        report = {
            'schema': 1, 'kind': 'qwen4exp_gdn_ordered_scratch_pilot',
            'command': sys.argv, 'arm': args.arm,
            'shape': {'tokens': args.tokens, 'Hk': 16, 'Hv': 48, 'Dk': 128, 'Dv': 128},
            'victim': victim.__name__, 'trigger': trigger.__name__,
            'before_ms': before, 'after_ms': after, 'trigger_ms': trigger_ms,
            'after_over_before': statistics.median(after) / statistics.median(before),
            'output_and_state_exact': True,
            'hashes': [hashlib.sha256(x.tobytes()).hexdigest() for x in actual],
            'build': {'path': str(library_path),
                      'sha256': hashlib.sha256(library_path.read_bytes()).hexdigest()},
            'host': {'machine_id': Path('/etc/machine-id').read_text().strip(),
                     'uname': list(os.uname())},
            'performance_claim': False,
            'limits': ['scratch-free GDN victim only', 'not a serving or model benchmark',
                       'clocks and scratch allocation require separate telemetry/trace'],
        }
    finally:
        rt.event_destroy(start)
        rt.event_destroy(stop)
        f.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'ratio': report['after_over_before'], 'exact': True}))


if __name__ == '__main__':
    main()
