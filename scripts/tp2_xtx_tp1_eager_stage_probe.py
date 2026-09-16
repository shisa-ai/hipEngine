#!/usr/bin/env python3
"""Bounded single-prompt resident TP1 diagnostic, not a numerical certificate.

Every stage is persisted BEFORE work and after synchronization/validation. A
failure stops all subsequent GPU calls, including explicit cleanup. CLI failure
uses os._exit(1) after flushing evidence to avoid GPU-owning destructors; process
exit is not teardown qualification. Run under an external timeout (e.g. 240s).
Nonfinite output logits identify an observation boundary, not a faulty kernel or
layer. No sentinel is initialized here, so no unwritten-buffer claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL = Path('/models/gguf/Qwen3.8-27B-Q4_K_M.gguf')
PROMPT = 'Write one line of Python that prints hello.'


class StageRecorder:
    """Fail-closed stage journal; an unfinished START is never a success."""

    def __init__(self, path: Path, metadata: dict):
        self.path = path
        self.artifact = {**metadata, 'status': 'running', 'stages': [],
                         'first_bad_stage': None, 'active_stage': None}
        self.persist()

    @property
    def exit_code(self) -> int:
        return int(self.artifact['first_bad_stage'] is not None)

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        with temporary.open('w') as handle:
            json.dump(self.artifact, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def guard(self, stage, fn, *, sync=None) -> bool:
        if self.exit_code:
            return False
        entry = {'stage': stage, 'ok': None}
        self.artifact['stages'].append(entry)
        self.artifact['active_stage'] = stage
        self.persist()
        print(f'STAGE {stage} START', flush=True)
        start = time.perf_counter()
        try:
            payload = fn()
            if sync is not None:
                sync()
            entry.update(payload or {})
            entry['ok'] = True
        except Exception as error:
            entry.update(ok=False, error=f'{type(error).__name__}: {error}')
            self.artifact['first_bad_stage'] = stage
            self.artifact['status'] = 'failed'
        entry['ms'] = 1000 * (time.perf_counter() - start)
        self.artifact['active_stage'] = None
        self.persist()
        print(f"STAGE {stage} {'OK' if entry['ok'] else 'FAIL'} {entry}", flush=True)
        return bool(entry['ok'])

    def finish(self):
        self.artifact['status'] = 'failed' if self.exit_code else 'complete'
        self.persist()


def validate_result(result, vocab_size: int) -> dict:
    token = result.token_id
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
        raise ValueError(f'non-integer token id: {token!r}')
    if not 0 <= token < vocab_size:
        raise ValueError(f'token id {token} outside [0, {vocab_size})')
    if result.logits is None:
        raise ValueError('missing logits')
    logits = np.asarray(result.logits)
    if vocab_size <= 0 or logits.shape not in {(vocab_size,), (1, vocab_size)}:
        raise ValueError(f'logits shape {logits.shape} is not one vocabulary row ({vocab_size})')
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite logits (upstream source unresolved)')
    if not np.any(logits != 0):
        raise ValueError('all-zero logits (not evidence of unwritten memory)')
    if int(token) != int(np.argmax(logits)):
        raise ValueError('greedy token does not match logit argmax')
    return {'token_id': int(token), 'token_in_range': True, 'logits': {
        'shape': list(logits.shape), 'finite': True, 'all_zero': False,
        'argmax': int(np.argmax(logits)), 'min': float(logits.min()),
        'max': float(logits.max()), 'sha256': hashlib.sha256(logits.tobytes()).hexdigest()}}


def graph_device_state(session) -> dict:
    """Read the graph's prepared-next-transition metadata, not host mirrors."""
    runtime = session.runtime
    runtime.device_synchronize()
    result = {}
    for name, buffer in (('position', session.scratch.position_buf),
                         ('context', session.scratch.context_buf),
                         ('sampled_token', session._lm_out_index)):
        host = np.empty(1, dtype=np.int64)
        runtime.memcpy(host.ctypes.data, buffer.ptr, host.nbytes, 2)
        result[name] = int(host[0])
    return result


def compare_graph_eager(eager, graph, vocab_size: int) -> dict:
    """Full-vocabulary comparison; finiteness alone cannot certify replay."""
    from scripts.tp2_teacher_coverage_broad import _kl_rows, _aggregate, _envelope_gate
    validate_result(eager, vocab_size)
    validate_result(graph, vocab_size)
    left = np.asarray(eager.logits).reshape(1, vocab_size)
    right = np.asarray(graph.logits).reshape(1, vocab_size)
    kl, top1 = _kl_rows(left, right)
    metrics = _aggregate(kl, top1)
    gate = _envelope_gate(metrics, top1_bar=0.99)
    if not gate['passed']:
        raise ValueError(f'graph/eager numerical comparison failed: {metrics}')
    return {**metrics, 'max_abs': float(np.max(np.abs(left - right))),
            'byte_identical': bool(np.array_equal(left, right)), 'single_row_smoke_only': True}


def run_stages(session, tokens, recorder, *, use_bulk=None, with_serial=False, with_graph=False):
    runtime = session.runtime
    vocab_size = int(session.runner.vocab_size)
    results = {}
    samples = {}

    def sample(name, fn):
        def checked():
            result = fn()
            runtime.device_synchronize()
            payload = validate_result(result, vocab_size)
            results[name] = int(result.token_id)
            samples[name] = result
            return payload
        return recorder.guard(name, checked)

    if not recorder.guard('reset', lambda: session.reset(), sync=runtime.device_synchronize):
        return
    if not sample('prefill-auto', lambda: session.prefill(
            tokens, use_bulk=use_bulk, bulk_attention_mode='bulk', return_logits=True)):
        return
    if not sample('eager-step', lambda: session.step(results['prefill-auto'], return_logits=True)):
        return
    if with_serial:
        if not recorder.guard('serial-reset', lambda: session.reset(), sync=runtime.device_synchronize):
            return
        if not sample('prefill-serial', lambda: session.prefill(tokens, use_bulk=False, return_logits=True)):
            return
    if with_graph:
        graphs = []
        start_position = int(session.position)
        seed_name = 'prefill-serial' if with_serial else 'eager-step'
        seed_token = results[seed_name]
        def capture():
            graphs.append(session.capture_decode_graph(position=start_position,
                steps_per_replay=1, max_replay_steps=2,
                attention_max_context_len=start_position + 2))
            if int(session.position) != start_position:
                raise ValueError('graph capture advanced the session position')
            device_state = graph_device_state(session)
            if device_state != {'position': start_position, 'context': start_position + 1, 'sampled_token': seed_token}:
                raise ValueError(f'graph capture device state mismatch: {device_state}')
            return {'input_position': start_position, 'input_token': seed_token, 'device_state': device_state}
        if not recorder.guard('graph-capture', capture, sync=runtime.device_synchronize):
            return
        graph = graphs[0]
        for step in range(2):
            suffix = '' if step == 0 else '-2'
            def replay():
                graph.replay(1)
                expected = start_position + step + 1
                if int(session.position) != expected:
                    raise ValueError('graph replay did not advance exactly one position')
                device_state = graph_device_state(session)
                if device_state['position'] != expected or device_state['context'] != expected + 1:
                    raise ValueError(f'graph replay device cursor mismatch: {device_state}')
                return {'output_position': expected, 'device_state': device_state,
                        'transport': graph.transport_provenance()}
            if not recorder.guard('graph-replay' + suffix, replay, sync=runtime.device_synchronize):
                return
            if not sample('graph-read' + suffix, lambda: graph.read_sample(return_logits=True)):
                return
        if not recorder.guard('graph-close', lambda: graph.close(), sync=runtime.device_synchronize):
            return
        if not recorder.guard('graph-eager-reset', lambda: session.reset(), sync=runtime.device_synchronize):
            return
        if not sample('graph-eager-prefill', lambda: session.prefill(
                tokens, use_bulk=False if with_serial else use_bulk,
                bulk_attention_mode='bulk', return_logits=True)):
            return
        reconstructed = 'graph-eager-prefill'
        if not with_serial:
            if not sample('graph-eager-reconstruct', lambda: session.step(
                    results['graph-eager-prefill'], return_logits=True)):
                return
            reconstructed = 'graph-eager-reconstruct'
        def check_start():
            if int(session.position) != start_position or results[reconstructed] != seed_token:
                raise ValueError('graph/eager input token or position mismatch')
            if not np.array_equal(samples[reconstructed].logits, samples[seed_name].logits):
                raise ValueError('same-schedule reconstruction changed seed logits')
            return {'input_token': seed_token, 'input_position': start_position}
        if not recorder.guard('graph-eager-input', check_start):
            return
        for step in range(2):
            suffix = '' if step == 0 else '-2'
            input_token = seed_token if step == 0 else results['graph-read']
            if not sample('graph-eager-reference' + suffix, lambda: session.step(input_token, return_logits=True)):
                return
            def compare():
                if int(session.position) != start_position + step + 1:
                    raise ValueError('eager reference position mismatch')
                return {**compare_graph_eager(samples['graph-eager-reference' + suffix], samples['graph-read' + suffix], vocab_size),
                        'input_token': input_token, 'input_position': start_position + step,
                        'output_position': int(session.position)}
            if not recorder.guard('graph-eager-compare' + suffix, compare):
                return
    recorder.guard('teardown', lambda: session.close(), sync=runtime.device_synchronize)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--prompt', default=PROMPT)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--use-gemv-decode', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--decode-repack', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--with-graph', action='store_true')
    parser.add_argument('--with-serial', action='store_true')
    parser.add_argument('--os-exit', action='store_true', help='skip successful-process destructors too')
    parser.add_argument('--compiler-version-file', type=Path)
    parser.add_argument('--require-cached-build', action='store_true')
    parser.add_argument('--execution-profile', choices=('strict', 'production'), default=None)
    parser.add_argument('--max-sequence-length', type=int, default=None)
    args = parser.parse_args(argv)
    # Exactly the true-AR CLI assignment, not setdefault: inherited 0 must not
    # silently override the declared True. No causal claim about repack/NaNs.
    inherited_repack = os.environ.get('HIPENGINE_GGUF_DECODE_REPACK')
    os.environ['HIPENGINE_GGUF_DECODE_REPACK'] = '1' if args.decode_repack else '0'
    recorder = StageRecorder(args.json, {
        'kind': 'tp2_xtx_tp1_eager_stage_probe',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'host': platform.node(), 'command': shlex.join([sys.executable, __file__, *(argv if argv is not None else sys.argv[1:])]),
        'inherited_decode_repack': inherited_repack,
        'env': {k: v for k, v in os.environ.items() if k.startswith(('HIP', 'ROCR', 'HSA', 'ROCM'))},
        'config': {'use_wmma_prefill': True, 'use_gemv_decode': args.use_gemv_decode,
                   'use_bulk': None, 'bulk_attention_mode': 'bulk', 'attn_aotriton_min_tokens': 512,
                   'kv_policy': 'session default (BF16)', 'kv_scale_dtype': 'fp16',
                   'profile': args.execution_profile or 'session/environment default',
                   'decode_transitions': 3 if args.with_graph else 1,
                   'require_cached_build': args.require_cached_build},
        'unwritten_check': 'not performed: no initialized sentinel',
    })
    state = {}
    def build():
        from hipengine.loading.gguf import scan_gguf
        from hipengine.runtime.prefill import PrefillConfig
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
        from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
        from scripts.gguf_mtp_bench import build_chat_prompt
        from scripts.tp2_teacher_coverage_broad import _model_sha256
        from hipengine.core.build import _resolve_cache_root
        cache_root = _resolve_cache_root(None).resolve()
        recorder.artifact['cache_root'] = str(cache_root)
        manifests = sorted(cache_root.glob('*/manifest.txt'))
        recorder.artifact['cache_manifest_sha256'] = hashlib.sha256(
            b''.join(str(p.relative_to(cache_root)).encode() + b'\0' + p.read_bytes()
                     for p in manifests)).hexdigest()
        recorder.artifact['cache_manifest_count'] = len(manifests)
        recorder.artifact['source_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        recorder.artifact['source_status'] = subprocess.check_output(['git', 'status', '--short'], text=True)
        recorder.artifact['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        recorder.artifact['model'] = {'path': str(args.model.resolve()), 'size': args.model.stat().st_size,
                                      'sha256': _model_sha256(args.model)}
        version = args.compiler_version_file.read_text() if args.compiler_version_file else None
        recorder.artifact['compiler_version'] = version
        tokens = build_chat_prompt(Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model)), args.prompt)
        recorder.artifact['prompt_token_ids'] = tokens
        # True-AR formula at warmup=0/decode=1. Optional graph diagnostics
        # additionally reserve their two-position attention capture horizon.
        if args.execution_profile is not None:
            from scripts.tp2_resident_control import bind_resident_profile
            recorder.artifact['execution_profile'] = bind_resident_profile(args.execution_profile)
            recorder.artifact['env'] = {k: v for k, v in os.environ.items()
                                       if k.startswith(('HIP', 'ROCR', 'HSA', 'ROCM'))}
        capacity = args.max_sequence_length or (len(tokens) + (4 if args.with_graph else 2))
        if capacity < len(tokens) + (4 if args.with_graph else 2):
            raise ValueError('declared capacity does not cover diagnostic horizon')
        recorder.artifact['config']['max_sequence_length'] = capacity
        recorder.persist()
        session = Qwen35GGUFResidentSession(args.model, max_sequence_length=capacity,
            use_wmma_prefill=True, use_gemv_decode=args.use_gemv_decode,
            compiler_version=version, require_cached_build=args.require_cached_build,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
            kv_policy=None, kv_scale_dtype='fp16', kv_scale_granularity=None)
        state.update(session=session, tokens=tokens)
        runtime = session.runtime
        runtime.device_synchronize()
        current = runtime.get_device()
        device = runtime.device_info(current)
        recorder.artifact['loaded_cache_libraries'] = sorted({
            line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
            if str(cache_root) in line and '.so' in line})
        return {'current_device': current, 'device': {'name': device.name, 'uuid': device.uuid,
                'pci_bus_id': device.pci_bus_id}, 'vocab_size': int(session.runner.vocab_size),
                'effective_kv_storage_dtype': str(session.kv_storage_dtype),
                'fastpath_safety': None if session.fastpath_safety is None else session.fastpath_safety.as_dict()}
    if recorder.guard('build', build):
        run_stages(state['session'], state['tokens'], recorder,
                   with_serial=args.with_serial, with_graph=args.with_graph)
    recorder.finish()
    code = recorder.exit_code
    print(f"first_bad_stage={recorder.artifact['first_bad_stage']} exit_code={code}", flush=True)
    if code or args.os_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
