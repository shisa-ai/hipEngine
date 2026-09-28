"""First-scratch ordered screen on the real-weight Q5_1 decode down owner.

Synthetic activations/routes, real model weights; not a serving benchmark.
Run sham and scratch arms in separate serial fresh processes.
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

from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
    build_qwen4_exp_q5_1,
    qwen4_exp_q5_1_selected_weighted_sum_logical256_t64_bf16_bf16_out as down,
)
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from hipengine.core.memory import free
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_moe_gemv_decode_screen import _bf16_bits
from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from tests.test_gpu_gguf_q8_0_dense_wide import make_q8_0_weight, _dequantize_weight, _f16_reference
from tests.test_gpu_qwen4exp_gdn_register import Fixture


def snapshot():
    result = {}
    for path in Path('/sys/class/drm').glob('card[0-9]*/device/pp_dpm_*clk'):
        try:
            result[str(path)] = path.read_text().strip()
        except OSError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--arm', choices=('sham', 'scratch'), required=True)
    parser.add_argument('--trigger', choices=('gdn', 'dense-wide'), default='gdn')
    parser.add_argument('--repeats', type=int, default=31)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('at least three repeats required')
    # Fixture constructs buffers/libraries only. No GDN executes before timing.
    f = Fixture(64)
    rt = f.runtime
    library = build_qwen4_exp_q5_1(load=True)
    readers = [GGUFReader(p) for p in discover_gguf_files(args.model_root)]
    name = 'blk.0.ffn_down_exps.weight'
    reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
    raw = reader.tensor_data(name)
    info = reader.tensor_info(name)
    if tuple(info.shape) != (512, 2560, 640) or info.ggml_type_name != 'Q5_1':
        raise ValueError(f'unexpected down tensor: {info.shape}, {info.ggml_type_name}')
    allocations = []
    start, stop = rt.event_create(), rt.event_create()
    clocks = {}
    try:
        rng = np.random.default_rng(2211)
        weight = _upload(raw, rt, allocations)
        x = _upload(_bf16_bits(rng.normal(size=(10, 640))), rt, allocations)
        selected = _upload(np.sort(rng.choice(512, 10, replace=False)).astype(np.int64), rt, allocations)
        routing = rng.uniform(size=10).astype(np.float32)
        routing /= routing.sum()
        rates = _upload(routing, rt, allocations)
        output = _alloc(2560, np.uint16, rt, allocations)
        dense_library = None
        dense_check = None
        if args.trigger == 'dense-wide':
            # Prepare but never launch scratch before the baseline window.
            dense_library = dw.build_gguf_q8_0_dense_wide(load=True)
            dense_x = rng.normal(0, .1, (257, 256)).astype(np.float32)
            dense_w = make_q8_0_weight(256, 256)
            dense_ref = _f16_reference(dense_x, _dequantize_weight(dense_w, 256))
            dense_dx = _upload(dense_x, rt, allocations)
            dense_dw = _upload(dense_w, rt, allocations)
            dense_out = _alloc(257 * 256, np.float32, rt, allocations)

        def launch_victim():
            rt.event_record(start)
            down(x.ptr, selected.ptr, weight.ptr, rates.ptr, output.ptr,
                 10, 512, 640, 2560, library=library, runtime=rt)
            rt.event_record(stop)
            rt.event_synchronize(stop)
            return rt.event_elapsed_time_ms(start, stop)

        for _ in range(3):
            launch_victim()
        expected = _download(output, (2560,), np.uint16, rt)
        clocks['before'] = snapshot()
        before = [launch_victim() for _ in range(args.repeats)]
        clocks['pre-trigger'] = snapshot()
        if args.arm == 'scratch' and args.trigger == 'dense-wide':
            dw.gguf_q8_0_dense_wide256_f32_f32_out(
                dense_dx.ptr, dense_dw.ptr, dense_out.ptr,
                257, 256, 256, library=dense_library, runtime=rt)
            rt.device_synchronize()
        elif args.arm == 'scratch':
            gdn.qwen4_exp_gdn_wave_norm_prefill_f32(
                *[p.ptr for p in f.inputs], f.states[1].ptr, f.outputs[1].ptr,
                64, 16, 48, 128, 128, library=f.library, runtime=rt)
            rt.device_synchronize()
        else:
            launch_victim()
        clocks['post-trigger'] = snapshot()
        after = [launch_victim() for _ in range(args.repeats)]
        clocks['after'] = snapshot()
        actual = _download(output, (2560,), np.uint16, rt)
        np.testing.assert_array_equal(actual, expected)
        trigger_exact = None
        if args.arm == 'scratch' and args.trigger == 'dense-wide':
            dense_actual = _download(dense_out, (257, 256), np.float32, rt)
            error = float(np.max(np.abs(dense_actual - dense_ref)))
            scale = float(np.max(np.abs(dense_ref)))
            assert np.isfinite(error) and error <= 1e-3 * scale
            dense_check = {'f16_reference_pass': True, 'max_abs_error': error,
                           'reference_scale': scale, 'shape': [257, 256, 256]}
        elif args.arm == 'scratch':
            # Reference launch only AFTER both windows: never poisons baseline.
            f.run(False)
            for got, ref in zip(f.result(True), f.result(False)):
                np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
            trigger_exact = True
        report = {
            'schema': 1, 'kind': 'qwen4exp_decode_first_scratch_screen',
            'command': sys.argv, 'arm': args.arm, 'model_root': str(args.model_root),
            'tensor': name, 'quant': info.ggml_type_name,
            'tensor_sha256': hashlib.sha256(raw).hexdigest(),
            'geometry': {'experts': 512, 'top_k': 10, 'in': 640, 'out': 2560},
            'victim': down.__name__,
            'trigger': args.trigger if args.arm == 'scratch' else 'victim',
            'requested_trigger': args.trigger, 'dense_reference_check': dense_check,
            'before_ms': before, 'after_ms': after,
            'after_over_before': statistics.median(after) / statistics.median(before),
            'victim_repeat_exact': True, 'trigger_parent_exact': trigger_exact,
            'output_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
            'clocks': clocks,
            'libraries': {lib._name: hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()
                          for lib in (library, f.library, dense_library) if lib is not None},
            'host': {'machine_id': Path('/etc/machine-id').read_text().strip(), 'uname': list(os.uname())},
            'performance_claim': False,
            'limits': ['single real-weight Q5_1 layer, synthetic activations/routes',
                       'boundary clocks only', 'does not admit a numerical variant'],
        }
    finally:
        rt.event_destroy(start)
        rt.event_destroy(stop)
        for p in reversed(allocations):
            free(p, runtime=rt)
        f.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'arm': args.arm, 'ratio': report['after_over_before'], 'exact': True}))


if __name__ == '__main__':
    main()
