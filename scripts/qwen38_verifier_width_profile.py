#!/usr/bin/env python3
"""Profile fixed-prefix Qwen GGUF verifier widths using a public LLM owner.

This is a diagnostic, not a throughput benchmark. Each call starts from identical
Conv/GDN state and the same live KV prefix; continuation rows are greedy teacher
IDs. Width one uses the eager verifier (the native graph requires 2..8 rows).
Widths alternate ascending/descending each repetition. Snapshot restore, warmup,
prefill and teacher generation are outside the measured windows.

Prewarm this exact command without rocprofv3. Then repeat with cached builds and
ROCTX_LIB pointing to librocprofiler-sdk-roctx.so under kernel/marker tracing.
The marker labels are verify:c<context>:r<rows>. Public output checks are outside
those windows. Never compare profiled wall times with unprofiled wall times.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def width_schedule(widths, repetitions):
    widths = tuple(widths)
    if not widths or len(set(widths)) != len(widths) or any(w not in range(1, 9) for w in widths):
        raise ValueError('widths must be distinct values in 1..8')
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    return [(rep, w) for rep in range(repetitions)
            for w in (widths if rep % 2 == 0 else tuple(reversed(widths)))]


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['context'], row['width']].append(row)
    return [dict(context=context, width=width, calls=len(group),
                 wall_ms_median=statistics.median(r['wall_ms'] for r in group),
                 wall_ms_min=min(r['wall_ms'] for r in group),
                 wall_ms_max=max(r['wall_ms'] for r in group),
                 graph_calls=sum(r['graph'] for r in group),
                 capture_ms_total=sum(r['capture_ms'] for r in group),
                 submit_ms_median=statistics.median(r['submit_ms'] for r in group),
                 readback_ms_median=statistics.median(r['readback_ms'] for r in group),
                 exact_all=all(r['exact'] for r in group))
            for (context, width), group in sorted(groups.items())]


@contextmanager
def saved_linear_state(session):
    # Own the copies: verify may overwrite the session's reusable rollback buffers.
    from hipengine.core.memory import malloc, free
    from hipengine.core.hip import HipMemcpyKind
    runtime = session.runtime
    pairs, allocated = [], []
    try:
        for state, backup in session._linear_state_snapshot():
            copy = malloc(backup.nbytes, runtime=runtime)
            allocated.append(copy)
            runtime.memcpy(copy.ptr, backup.ptr, backup.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
            pairs.append((state, copy))
        yield pairs
    finally:
        runtime.device_synchronize()
        for buffer in allocated:
            free(buffer, runtime=runtime)


class Markers:
    def __init__(self):
        path = os.environ.get('ROCTX_LIB')
        self.library = ctypes.CDLL(path) if path else None
        if self.library is not None:
            self.library.roctxRangePushA.argtypes = [ctypes.c_char_p]
            self.library.roctxRangePushA.restype = ctypes.c_int
            self.library.roctxRangePop.argtypes = []
            self.library.roctxRangePop.restype = ctypes.c_int

    @contextmanager
    def range(self, label):
        if self.library is not None:
            self.library.roctxRangePushA(label.encode())
        try:
            yield
        finally:
            if self.library is not None:
                self.library.roctxRangePop()


def run(args):
    from hipengine import LLM, SamplingParams
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.runtime.gguf_native_spec_cycle import verify_qwen35_gguf_native_b2_target
    from hipengine.speculative import native_cycle_graph  # noqa: F401

    widths = tuple(int(x) for x in args.widths.split(','))
    schedule = width_schedule(widths, args.repetitions)
    contexts = tuple(int(x) for x in args.contexts.split(','))
    if not contexts or min(contexts) < 1 or args.warmups < 1:
        raise ValueError('contexts and warmups must be positive')
    layers = tuple(int(x) for x in args.capture_layers.split(',') if x)
    maximum = max(contexts) + max(widths) + 32
    markers = Markers()
    rows, checks, ar_rows = [], [], []
    completed = False
    artifact = dict(schema=1, kind='qwen38_verifier_width_profile', status='diagnostic',
                    performance_claim=False, rows=rows, public_checks=checks, ar_step_rows=ar_rows)
    compiler_path = os.environ.get('HIPENGINE_COMPILER_VERSION_FILE')
    artifact['provenance'] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend=args.backend, model_path=args.model,
        quant='Q4_K_M', kv_dtype='bf16', command=[sys.executable, *sys.argv],
        environment={k: v for k, v in os.environ.items() if k.startswith(('HIPENGINE_', 'ROCTX_'))},
        hipcc_version=Path(compiler_path).read_text() if compiler_path else None,
        timing_protocol='fixed-prefix sync-bounded verify; excludes restore/warmup/prefill',
        warmups=args.warmups, repetitions=args.repetitions,
        profiler={'enabled': markers.library is not None},
    )
    artifact['protocol'] = dict(contexts=contexts, widths=widths, repetitions=args.repetitions,
                                capture_layers=layers, order='alternating forward/reverse widths',
                                one_row_route='eager verifier; native graph only supports 2..8 rows')
    llm = LLM(args.model, backend=args.backend, max_sequence_length=maximum,
              max_active_requests=1, execution_profile=args.execution_profile)
    try:
        llm.prepare(max_sequence_length=maximum)
        owner = llm._get_text_generator()
        seed = tuple(owner.tokenize('Write a Python function to merge overlapping intervals. '
                                    'Explain its correctness and complexity. 日本語でも説明してください。\n'))
        artifact['profile'] = dict(resolved=str(owner.execution_profile),
                                   manifest_sha256=llm.execution_profile_manifest_sha256,
                                   manifest=llm.execution_profile_manifest)
        for context in contexts:
            prompt = (seed * ((context + len(seed) - 1) // len(seed)))[:context]
            params = SamplingParams(temperature=0, max_tokens=max(widths) + 1, ignore_eos=True)
            public = llm.generate_detailed([prompt], params)[0]
            with hip_target_arch_environment(owner.target_arch), owner._resident_session_scope(
                shared_runner=owner._get_shared_runner(), pool_name='verifier_width_profile',
                max_sequence_length=maximum,
            ) as (session, reused):
                rt = session.runtime
                session.reset()
                root = int(session.prefill(prompt, use_bulk=True, return_logits=False).token_id)
                artifact['session'] = dict(
                    target_verifier_production_q4_rowtile=session.target_verifier_production_q4_rowtile,
                    fp16_recurrent_state=session.runner.fp16_recurrent_state,
                    use_wmma_prefill=session.use_wmma_prefill, use_gemv_decode=session.use_gemv_decode,
                    kv_storage_dtype=str(session.kv_storage_dtype),
                )
                with saved_linear_state(session) as snapshot:
                    teacher = [root]
                    for _ in range(max(widths)):
                        teacher.append(int(session.step(teacher[-1], return_logits=False).token_id))
                    exact = tuple(teacher) == public.generated_token_ids
                    checks.append(dict(context=context, exact=exact, public_ids=list(public.generated_token_ids),
                                       teacher_ids=teacher, prompt_ids=list(prompt),
                                       prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()))
                    if not exact:
                        raise RuntimeError(f'public AR and verifier-owner teacher differ at context {context}')

                    def restore():
                        session._restore_linear_state_snapshot(snapshot, position=context)
                        rt.device_synchronize()

                    def call(width, label):
                        restore()
                        with markers.range(label):
                            start = time.perf_counter()
                            result = verify_qwen35_gguf_native_b2_target(
                                session, teacher[:width], bulk_attention_mode='native',
                                capture_layer_output_hidden=layers, capture_linear_state_rows=True,
                                defer_linear_state_commit=True, fallback=True,
                            )
                            rt.device_synchronize()
                            elapsed = (time.perf_counter() - start) * 1000
                        return dict(context=context, width=width, wall_ms=elapsed,
                                    graph=bool(session.last_native_spec_target_submitted),
                                    fallback_reason=session.last_native_spec_target_fallback_reason,
                                    capture_ms=session.last_native_spec_target_capture_ms,
                                    submit_ms=session.last_native_spec_target_submit_ms,
                                    readback_ms=session.last_native_spec_target_readback_ms,
                                    exact=list(result.token_ids) == teacher[1:width + 1],
                                    output_ids=list(result.token_ids))

                    for _, width in width_schedule(widths, args.warmups):
                        warm = call(width, f'warm:c{context}:r{width}')
                        if not warm['exact']:
                            raise RuntimeError(f'warm verifier mismatch: {warm}')
                    for rep, width in schedule:
                        row = call(width, f'verify:c{context}:r{width}')
                        row['repetition'] = rep
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                        if not row['exact']:
                            raise RuntimeError(f'verifier mismatch: {row}')
                        if width == widths[0]:
                            restore()
                            with markers.range(f'ar_step:c{context}'):
                                start = time.perf_counter()
                                result = session.step(root, return_logits=False)
                                rt.device_synchronize()
                                elapsed = (time.perf_counter() - start) * 1000
                            ar_rows.append(dict(context=context, repetition=rep, wall_ms=elapsed,
                                                exact=int(result.token_id) == teacher[1]))
        completed = True
    finally:
        llm.close()
        artifact['completed'] = completed
        artifact['summary'] = summarize_rows(rows)
        artifact['exact_all'] = completed and bool(rows) and all(r['exact'] for r in rows + checks + ar_rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    return 0 if artifact['exact_all'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/models/gguf/Qwen3.8-27B-Q4_K_M.gguf')
    parser.add_argument('--backend', default='hip_gfx1151')
    parser.add_argument('--execution-profile', choices=('strict', 'production'), default='production')
    parser.add_argument('--contexts', default='64,2048')
    parser.add_argument('--widths', default='1,2,3,4,6,8')
    parser.add_argument('--capture-layers', default='3,31,59', help='EAGLE aux layers by default; empty disables taps')
    parser.add_argument('--repetitions', type=int, default=6)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
