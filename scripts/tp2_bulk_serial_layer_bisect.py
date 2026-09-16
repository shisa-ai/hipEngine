#!/usr/bin/env python3
"""Bounded diagnostic: first divergent per-layer hidden / Conv / GDN state
between bulk and token-serial prefill for the saved TP2 failing prompt.

This is a diagnostic-only, non-promoting probe. It uses the *same* loaded
resident TP1 session and the *same* prompt / positions for both schedules, so
the only changed axis is the prefill schedule. It does not repair arithmetic,
does not substitute a denominator, and emits no performance claim.

Output: one compact JSON artifact plus one NPZ of the captured boundary rows.
The process is intended to run under an external ``timeout`` watchdog; it
flushes and persists after every stage so a hang leaves a usable artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tp2_resident_control import bind_resident_profile, create_native_adapter
from scripts.tp2_xtx_tp1_eager_stage_probe import StageRecorder, validate_result

# Report a layer as divergent once the per-layer final-row RMS crosses this.
# A bf16 value of magnitude 1 has an ulp near 8e-3, so 1e-3 RMS is well below
# any single rounding event and cannot be produced by one bf16 store.
HIDDEN_RMS_THRESHOLD = 1e-3
STATE_RMS_THRESHOLD = 1e-3


def _read_buffer(runtime, buffer, dtype):
    out = np.empty(buffer.nbytes // np.dtype(dtype).itemsize, dtype=dtype)
    runtime.device_synchronize()
    runtime.memcpy(out.ctypes.data, int(buffer.ptr), out.nbytes, 2)
    return out


def _summary(left: np.ndarray, right: np.ndarray) -> dict:
    diff = left.astype(np.float64) - right.astype(np.float64)
    return {
        'n': int(diff.size),
        'max_abs': float(np.max(np.abs(diff))) if diff.size else 0.0,
        'rms': float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
        'left_absmax': float(np.max(np.abs(left))) if left.size else 0.0,
        'right_absmax': float(np.max(np.abs(right))) if right.size else 0.0,
    }


def _first_above(curve: list[dict], key: str, threshold: float) -> int | None:
    for row in curve:
        if row[key] > threshold:
            return int(row['layer'])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/models/gguf/Qwen3.8-27B-Q4_K_M.gguf')
    parser.add_argument('--suite-json', required=True,
                        help='quality-tp1-d0.json holding the shared teacher tokens')
    parser.add_argument('--prompt-id', default='mixed_ja_en_translate')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--npz', type=Path, required=True)
    parser.add_argument('--profile', default='production')
    args = parser.parse_args()

    os.environ['HIPENGINE_GGUF_DECODE_REPACK'] = '1'
    suite = json.loads(Path(args.suite_json).read_text())
    index = suite['suite']['ids'].index(args.prompt_id)
    tokens = [int(t) for t in suite['suite']['tokens'][index]]
    record = StageRecorder(args.json, {
        'kind': 'tp2_bulk_serial_layer_bisect',
        'performance_claim': False,
        'production_qualified': False,
        'host': platform.node(),
        'argv': sys.argv,
        'prompt_id': args.prompt_id,
        'tokens': tokens,
        'profile': bind_resident_profile(args.profile),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True, cwd=ROOT).strip(),
        'interpretation': (
            'Compares two prefill schedules of the same resident model/weights on '
            'the same prompt and positions. A smooth, broadly distributed growth '
            'is T2 association drift; a step change localized to one layer/state '
            'is an ownership/control candidate. Neither is automatically a defect.'
        ),
    })
    state: dict = {}

    def build():
        adapter = create_native_adapter(args.model, 'tp1-d0', capacity=200)
        state['adapter'] = adapter
        session = adapter.session
        state['session'] = session
        config = session.runner.weights.config
        state['layers'] = list(range(len(config.layer_types)))
        state['layer_types'] = list(config.layer_types)
        device = adapter.runtime.device_info(adapter.runtime.get_device())
        return {
            'device': device.name,
            'uuid': device.uuid,
            'pci_bus_id': device.pci_bus_id,
            'layers': len(state['layers']),
            'hidden_size': int(session.runner.hidden_size),
            'fp16_recurrent_state': bool(session.runner.fp16_recurrent_state),
            'kv_storage_dtype': str(session.kv_storage_dtype),
        }

    def snapshot(tag: str) -> dict:
        session = state['session']
        runtime = state['adapter'].runtime
        hidden = {
            int(layer): np.asarray(rows, dtype=np.float32).reshape(-1).copy()
            for layer, rows in session._last_layer_output_hidden.items()
        }
        conv: dict[int, np.ndarray] = {}
        recurrent: dict[int, np.ndarray] = {}
        scratch = session.scratch
        rec_dtype = np.float16 if session.runner.fp16_recurrent_state else np.float32
        for layer in state['layers']:
            cbuffer = scratch.layer_conv_states[layer]
            rbuffer = scratch.layer_recurrent_states[layer]
            if cbuffer is not None:
                conv[layer] = _read_buffer(runtime, cbuffer, np.float32)
            if rbuffer is not None:
                recurrent[layer] = _read_buffer(runtime, rbuffer, rec_dtype)
        state[tag] = {'hidden': hidden, 'conv': conv, 'recurrent': recurrent}
        missing = [layer for layer in state['layers'] if layer not in hidden]
        if missing:
            raise ValueError(f'{tag}: missing hidden rows for layers {missing[:5]}')
        return {'layers': len(hidden), 'conv_states': len(conv),
                'recurrent_states': len(recurrent)}

    def run_bulk():
        session = state['session']
        session.reset()
        result = session.prefill(tokens, use_bulk=None, bulk_attention_mode='bulk',
                                 return_logits=True,
                                 capture_layer_output_hidden=state['layers'])
        state['adapter'].runtime.device_synchronize()
        payload = validate_result(result, session.runner.vocab_size)
        state['bulk_logits'] = np.asarray(result.logits, dtype=np.float32).reshape(-1).copy()
        state['bulk_token'] = int(result.token_id)
        payload.update(snapshot('bulk'))
        return payload

    def run_serial():
        session = state['session']
        session.reset()
        result = session.prefill(tokens, use_bulk=False, return_logits=True,
                                 capture_layer_output_hidden=state['layers'])
        state['adapter'].runtime.device_synchronize()
        payload = validate_result(result, session.runner.vocab_size)
        state['serial_logits'] = np.asarray(result.logits, dtype=np.float32).reshape(-1).copy()
        state['serial_token'] = int(result.token_id)
        payload.update(snapshot('serial'))
        return payload

    def compare():
        bulk, serial = state['bulk'], state['serial']
        layers = state['layers']
        hidden_curve = [
            {'layer': layer, **_summary(bulk['hidden'][layer], serial['hidden'][layer])}
            for layer in layers
        ]
        conv_curve = [
            {'layer': layer, **_summary(bulk['conv'][layer], serial['conv'][layer])}
            for layer in layers if layer in bulk['conv'] and layer in serial['conv']
        ]
        recurrent_curve = [
            {'layer': layer, **_summary(bulk['recurrent'][layer], serial['recurrent'][layer])}
            for layer in layers if layer in bulk['recurrent'] and layer in serial['recurrent']
        ]
        kl = float('nan')
        try:
            from scripts.tp2_teacher_coverage_broad import _kl_rows
            kl_value, top1 = _kl_rows(state['bulk_logits'][None, :],
                                      state['serial_logits'][None, :])
            kl = float(kl_value[0])
            top1 = bool(top1[0])
        except Exception:
            top1 = None
        np.savez(args.npz,
                 bulk_hidden=np.stack([bulk['hidden'][l] for l in layers]),
                 serial_hidden=np.stack([serial['hidden'][l] for l in layers]),
                 bulk_conv=np.stack([bulk['conv'][l] for l in layers if l in bulk['conv']]),
                 serial_conv=np.stack([serial['conv'][l] for l in layers if l in serial['conv']]),
                 bulk_recurrent=np.stack([bulk['recurrent'][l] for l in layers
                                          if l in bulk['recurrent']]),
                 serial_recurrent=np.stack([serial['recurrent'][l] for l in layers
                                            if l in serial['recurrent']]),
                 bulk_logits=state['bulk_logits'], serial_logits=state['serial_logits'])
        payload = {
            'hidden_rms_threshold': HIDDEN_RMS_THRESHOLD,
            'state_rms_threshold': STATE_RMS_THRESHOLD,
            'first_hidden_divergent_layer': _first_above(
                hidden_curve, 'rms', HIDDEN_RMS_THRESHOLD),
            'first_conv_divergent_layer': _first_above(
                conv_curve, 'rms', STATE_RMS_THRESHOLD),
            'first_recurrent_divergent_layer': _first_above(
                recurrent_curve, 'rms', STATE_RMS_THRESHOLD),
            'hidden_curve': hidden_curve,
            'conv_curve': conv_curve,
            'recurrent_curve': recurrent_curve,
            'bulk_token': state['bulk_token'],
            'serial_token': state['serial_token'],
            'logits_kl_bulk_serial': kl,
            'logits_top1_match': top1,
            'fixture': str(args.npz),
            'fixture_sha256': hashlib.sha256(args.npz.read_bytes()).hexdigest(),
            'layer_types': state['layer_types'],
            'diagnostic_only': True,
        }
        return payload

    try:
        if record.guard('build', build):
            if record.guard('bulk-prefill', run_bulk):
                if record.guard('serial-prefill', run_serial):
                    record.guard('compare', compare)
    finally:
        if 'adapter' in state:
            try:
                state['adapter'].close()
            except Exception:
                pass
    record.finish()
    sys.stdout.flush()
    sys.stderr.flush()
    return 1 if record.exit_code else 0


if __name__ == '__main__':
    raise SystemExit(main())
