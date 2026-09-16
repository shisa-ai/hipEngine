#!/usr/bin/env python3
"""Diagnostic-only synchronized resident boundaries; use a 240s process watchdog.

No dispatch/profile switches. Check exact H2D payloads before their consumers,
then embedding, per-layer norms/attention/MLP/hidden/state finiteness. On failure,
flush a small fixture and exit without further GPU calls. Optional token-eager
reference and bulk repeat reuse the same weights/session, not a weaker baseline.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tp2_xtx_tp1_eager_stage_probe import DEFAULT_MODEL, PROMPT, StageRecorder, validate_result


class BoundaryFailure(RuntimeError):
    pass


def array_summary(array):
    values = np.asarray(array)
    return {'shape': list(values.shape), 'dtype': str(values.dtype),
            'finite': int(np.isfinite(values).sum()), 'elements': int(values.size),
            'sha256': hashlib.sha256(values.tobytes()).hexdigest()}


class Observer:
    def __init__(self, session, recorder, directory):
        self.session, self.recorder, self.directory = session, recorder, directory
        self.runtime = session.runtime
        self.region = 'prefill'
        self.saved = []
        self.layer_outputs = {}

    def check(self, name, fn):
        if not self.recorder.guard(name, fn):
            raise BoundaryFailure(name)

    def read(self, ptr, shape, dtype='bf16'):
        storage = np.uint16 if dtype == 'bf16' else np.dtype(dtype)
        out = np.empty(shape, dtype=storage)
        self.runtime.device_synchronize()
        self.runtime.memcpy(out.ctypes.data, int(ptr), out.nbytes, 2)
        return (out.astype(np.uint32) << 16).view(np.float32) if dtype == 'bf16' else out

    def verify(self, name, actual, expected=None):
        summary = array_summary(actual)
        bad = summary['finite'] != summary['elements'] or not actual.size
        if expected is not None:
            summary['exact_reference'] = bool(np.array_equal(actual, expected))
            bad |= not summary['exact_reference']
        if bad:
            path = self.directory / 'first-failure.npz'
            np.savez(path, actual=actual, **({} if expected is None else {'expected': expected}))
            self.recorder.artifact['failure_fixture'] = {
                'boundary': name, 'path': str(path), 'summary': summary,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            raise ValueError(f'{name}: {summary}')
        return summary

    def tensor(self, name, ptr, shape, dtype='bf16'):
        def checked():
            data = self.read(ptr, shape, dtype)
            return self.verify(name, data)
        self.check(name, checked)

    def patch(self, owner, name, fn):
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, fn)

    def install(self):
        import hipengine.runtime.qwen35_gguf_runner as mod
        runner = self.session.runner
        original_copy = mod.copy_host_to_device
        def upload(buffer, host_ptr, nbytes=None, *, runtime=None):
            count = int(buffer.nbytes if nbytes is None else nbytes)
            if count < 0 or count > buffer.nbytes:
                raise ValueError('invalid H2D shape')
            expected = np.frombuffer(ctypes.string_at(host_ptr, count), dtype=np.uint8).copy()
            labels = []
            for owner_name, owner in [('session', self.session), ('decode_scratch', self.session.scratch),
                                      ('bulk_scratch', self.session._bulk_prefill_scratch)]:
                if owner is not None:
                    labels.extend(f'{owner_name}.{key}' for key, value in vars(owner).items()
                                  if getattr(value, 'ptr', None) == buffer.ptr)
            name = f'{self.region}/H2D/' + ('+'.join(labels) or 'derived-view')
            def copy_and_verify():
                rt = runtime or self.runtime
                # Poison only embedding destinations (fresh output/token rows),
                # never state, aliased layer scratch, or live weight storage.
                if self.region == 'embedding':
                    poison = np.full(count, 0xA5, dtype=np.uint8)
                    rt.memcpy(buffer.ptr, poison.ctypes.data, count, 1)
                original_copy(buffer, host_ptr, nbytes, runtime=rt)
                actual = self.read(buffer.ptr, (count,), np.uint8)
                detail = {'attributed': buffer.device is not None,
                          'current_device': rt.get_device(), 'bytes': count,
                          'sentinel_remaining_bytes': int((actual == 0xA5).sum()) if self.region == 'embedding' else None}
                self.recorder.artifact['last_upload'] = {'boundary': name, **detail}
                result = self.verify(name, actual, expected)
                return {**result, **detail}
            self.check(name, copy_and_verify)
        self.patch(mod, 'copy_host_to_device', upload)
        original_embed = self.session._copy_token_embeddings_to_device
        def embedding(tokens, out_ptr, *, rows, **kwargs):
            self.region = 'embedding'
            self.recorder.artifact['inflight_region'] = self.region
            self.recorder.persist()
            original_embed(tokens, out_ptr, rows=rows, **kwargs)
            self.tensor('embedding/output', out_ptr, (rows, runner.hidden_size))
            self.region = 'prefill-metadata'
        self.patch(self.session, '_copy_token_embeddings_to_device', embedding)
        for method in ('_run_linear_attention_prefill_layer_rows', '_run_full_attention_prefill_layer_aotriton'):
            original = getattr(runner, method)
            def layer(layer_id, src, dst, scratch, _original=original, **kwargs):
                rows = int(kwargs.get('rows', scratch.rows))
                self.region = f'layer-{layer_id}'
                self.recorder.artifact['inflight_region'] = self.region
                self.recorder.persist()
                self.tensor(f'{self.region}/input', src, (rows, runner.hidden_size))
                self.states(layer_id, 'before')
                timing = kwargs.get('gpu_stage_recorder')
                mark = timing.mark if timing is not None else None
                if timing is not None:
                    def observed_mark(name, *aliases):
                        self.mark(name, scratch, rows, layer_id)
                        return mark(name, *aliases)
                    timing.mark = observed_mark
                try:
                    _original(layer_id, src, dst, scratch, **kwargs)
                finally:
                    if timing is not None:
                        timing.mark = mark
                data = self.read(dst, (rows, runner.hidden_size))
                self.check(f'{self.region}/output', lambda: self.verify(self.region, data))
                self.layer_outputs[layer_id] = data[-1:].copy()
                self.states(layer_id, 'after')
            self.patch(runner, method, layer)

    def states(self, layer, phase):
        scratch = self.session.scratch
        for label, buffers, dtype in (
            ('conv', scratch.layer_conv_states, np.float32),
            ('recurrent', scratch.layer_recurrent_states,
             np.float16 if self.session.runner.fp16_recurrent_state else np.float32)):
            buffer = buffers[layer]
            if buffer is not None:
                self.tensor(f'layer-{layer}/{phase}-{label}', buffer.ptr,
                            (buffer.nbytes // np.dtype(dtype).itemsize,), dtype)

    def mark(self, name, scratch, rows, layer):
        hidden = self.session.runner.hidden_size
        # Each read covers only the active typed region, not padded/unused scratch.
        fields = []
        if name.endswith('_attn_norm'): fields = [('norm', hidden)]
        elif '_attn_qkv_gate' in name:
            fields = [('linear_qkv', self.session.runner.linear_qkv_width),
                      ('linear_z', self.session.runner.weights.config.ssm_inner_size)]
        elif name.endswith(('_ssm_out', '_output')): fields = [('attn_out', hidden)]
        elif name.endswith('_post_norm_residual'): fields = [('post_norm', hidden), ('residual', hidden)]
        elif name.endswith('_silu'): fields = [('ffn_intermediate', self.session.runner.ffn_size)]
        for field, width in fields:
            buffer = getattr(scratch, field)
            if rows * width * 2 > buffer.nbytes:
                raise ValueError(f'{name}/{field}: requested shape exceeds buffer')
            self.tensor(f'layer-{layer}/{name}/{field}', buffer.ptr, (rows, width))

    def restore(self):
        for owner, name, original in reversed(self.saved):
            setattr(owner, name, original)
        self.saved.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    os.environ['HIPENGINE_GGUF_DECODE_REPACK'] = '1'
    record = StageRecorder(args.json, {'kind': 'tp2_resident_prefill_boundaries',
        'host': platform.node(), 'argv': sys.argv, 'performance_claim': False,
        'env': {k: v for k, v in os.environ.items() if k.startswith(('HIP', 'ROCR', 'HSA'))},
        'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'memory_sha256': hashlib.sha256((ROOT / 'hipengine/core/memory.py').read_bytes()).hexdigest()})
    state = {}
    def build():
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
        from hipengine.runtime.prefill import PrefillConfig
        from hipengine.loading.gguf import scan_gguf
        from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
        from scripts.gguf_mtp_bench import build_chat_prompt
        from scripts.tp2_teacher_coverage_broad import _model_sha256
        record.artifact['model'] = {'path': str(args.model), 'sha256': _model_sha256(args.model)}
        tokens = build_chat_prompt(Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model)), args.prompt)
        record.artifact['tokens'] = tokens
        record.persist()
        session = Qwen35GGUFResidentSession(args.model, max_sequence_length=len(tokens)+2,
            use_wmma_prefill=True, use_gemv_decode=True,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
            kv_policy=None, kv_scale_dtype='fp16', kv_scale_granularity=None)
        state.update(session=session, tokens=tokens)
        session.runtime.device_synchronize()
        device = session.runtime.device_info(session.runtime.get_device())
        return {'device': device.name, 'uuid': device.uuid, 'pci': device.pci_bus_id,
                'host_embedding': session.host_token_embedding_enabled,
                'mapped_embedding': session.runner.host_token_embedding_mapped_weight is not None,
                'fp16_state': session.runner.fp16_recurrent_state,
                'kv': str(session.kv_storage_dtype), 'profile': 'direct-session default; no promotion certificate'}
    if record.guard('build', build):
        session, tokens = state['session'], state['tokens']
        observer = Observer(session, record, args.json.parent)
        observer.install()
        try:
            result = session.prefill(tokens, use_bulk=None, bulk_attention_mode='bulk', return_logits=True)
            observer.check('bulk-logits', lambda: validate_result(result, session.runner.vocab_size))
        except Exception as error:
            if not record.exit_code:
                def fail(): raise error
                record.guard(f'{observer.region}/unhandled', fail)
        finally:
            observer.restore()  # Python-only restoration; no GPU cleanup on failure.
        if not record.exit_code and args.reference:
            bulk = result.logits.copy()
            def reference():
                from hipengine.benchmark.correctness import evaluate_logits
                session.reset()
                teacher = session.prefill(tokens, use_bulk=False, return_logits=True,
                                          capture_layer_output_hidden=list(range(len(session.runner.weights.config.layer_types))))
                session.runtime.device_synchronize()
                validate_result(teacher, session.runner.vocab_size)
                vocab = session.runner.vocab_size
                gate = evaluate_logits(teacher.logits.reshape(-1, vocab), bulk.reshape(-1, vocab),
                                       kl_threshold=0.05, top1_threshold=0.90)
                record.artifact['layer_reference'] = {
                    str(layer): {'max_abs': float(np.max(np.abs(value - session._last_layer_output_hidden[layer])))}
                    for layer, value in observer.layer_outputs.items()}
                payload = {'kl_mean': float(gate.kl_mean), 'kl_max': float(gate.kl_max),
                           'top1': float(gate.top1_agreement), 'outer_smoke_pass': bool(gate.passed),
                           'production_qualified': False}
                record.artifact['reference_gate'] = payload
                if not gate.passed: raise ValueError(f'token-eager comparison failed: {payload}')
                return payload
            if record.guard('token-eager-reference', reference):
                def repeat_and_check():
                    again = session.prefill(tokens, use_bulk=None, bulk_attention_mode='bulk', return_logits=True)
                    session.runtime.device_synchronize()
                    payload = validate_result(again, session.runner.vocab_size)
                    if not np.array_equal(again.logits, bulk):
                        raise ValueError('bulk repeat changed logits after token-eager reference')
                    state['next_token'] = int(again.token_id)
                    return {**payload, 'repeat_exact': True}
                if record.guard('bulk-repeat', repeat_and_check):
                    def step_and_check():
                        step = session.step(state['next_token'], return_logits=True)
                        session.runtime.device_synchronize()
                        return validate_result(step, session.runner.vocab_size)
                    record.guard('eager-step', step_and_check)
        if not record.exit_code:
            record.guard('teardown', session.close)
    record.finish()
    sys.stdout.flush()
    sys.stderr.flush()
    if record.exit_code: os._exit(1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
