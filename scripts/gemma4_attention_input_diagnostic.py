#!/usr/bin/env python3
"""Compare shipped and strict attention on identical actual model inputs.

This diagnostic leaves the public request's selected output untouched. It runs
strict attention into a separate buffer, captures raw BF16 Q/K/V and keep masks
outside the repository, and compares sampled rows with a float64 CPU oracle.
The oracle is diagnostic: it is not the strict arithmetic contract or a full
production numerical gate. Model loading and HIP are deferred until main().
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def bf16_to_f64(bits):
    words = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return words.view(np.float32).astype(np.float64)


def f64_to_bf16(values):
    """Round finite float64 values to the nearest BF16 lattice, ties to even."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite reference values')
    magnitude = np.abs(values)
    if np.any(magnitude > bf16_to_f64(np.array([0x7f7f], np.uint16))[0]):
        raise ValueError('reference value exceeds finite BF16 range')
    base = (np.asarray(magnitude, dtype=np.float32).view(np.uint32) >> 16).astype(np.int32)
    candidates = np.stack([np.clip(base + d, 0, 0x7f7f) for d in (-1, 0, 1)])
    distances = np.abs(bf16_to_f64(candidates) - magnitude)
    # Prefer even lattice coordinates on exact ties; sort that preference first.
    order = np.argsort(candidates & 1, axis=0, kind='stable')
    candidates = np.take_along_axis(candidates, order, axis=0)
    distances = np.take_along_axis(distances, order, axis=0)
    choice = np.argmin(distances, axis=0)
    result = np.take_along_axis(candidates, np.expand_dims(choice, 0), axis=0)[0]
    return (result.astype(np.uint16) | (np.signbit(values).astype(np.uint16) << 15))


def output_comparison(strict, wmma):
    strict, wmma = np.asarray(strict), np.asarray(wmma)
    if strict.shape != wmma.shape:
        raise ValueError('output shapes differ')
    b, c = bf16_to_f64(strict), bf16_to_f64(wmma)
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError('nonfinite attention outputs')
    diff = c - b
    return {'elements': int(b.size), 'differing_elements': int(np.count_nonzero(strict != wmma)),
            'bitwise_equal': bool(np.array_equal(strict, wmma)),
            'max_abs_diff': float(np.max(np.abs(diff))),
            'rms_diff': float(np.sqrt(np.mean(diff * diff))),
            'relative_l2_diff': float(np.linalg.norm(diff.ravel()) /
                                      max(np.linalg.norm(b.ravel()), np.finfo(np.float64).tiny))}


def oracle_samples(query, key, value, mask, strict, wmma, *, scale, rows, heads):
    q, k, v = map(bf16_to_f64, (query, key, value))
    b, c = map(bf16_to_f64, (strict, wmma))
    ratio = q.shape[1] // k.shape[1]
    result = []
    for row in rows:
        for head in heads:
            keep = np.asarray(mask[row], dtype=bool)
            if not keep.any():
                raise ValueError('fully masked reference row')
            kv_head = head // ratio
            scores = (k[keep, kv_head] @ q[row, head]) * scale
            weights = np.exp(scores - np.max(scores))
            weights /= weights.sum()
            reference = weights @ v[keep, kv_head]
            rounded = f64_to_bf16(reference)
            result.append({'row': int(row), 'head': int(head), 'live_keys': int(keep.sum()),
                           'score_min': float(scores.min()), 'score_max': float(scores.max()),
                           'strict_bf16_reference_mismatches': int(np.count_nonzero(strict[row, head] != rounded)),
                           'wmma_bf16_reference_mismatches': int(np.count_nonzero(wmma[row, head] != rounded)),
                           'strict_float64_reference_l2': float(np.linalg.norm(b[row, head] - reference)),
                           'wmma_float64_reference_l2': float(np.linalg.norm(c[row, head] - reference))})
    return result


def main(argv=None):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import malloc, free
    from hipengine.core.runtime import MemcpyKind
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        Gemma4AttentionScratch, gemma4_attention_prefill_bf16,
    )
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from scripts.gemma4_campaign_bench import DEFAULT_ARTIFACT, _resolve_generator
    from scripts.gemma4_production_quality import padded_chat
    from hipengine.llm import SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument('--packet-directory', type=Path, required=True)
    parser.add_argument('--case', default='general_en_plan')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--per-phase-limit', type=int, default=4)
    args = parser.parse_args(argv)
    if args.per_phase_limit < 1:
        parser.error('per-phase-limit must be positive')
    packet = json.loads((args.packet_directory / 'report.json').read_text())
    index, case = next((i, c) for i, c in enumerate(packet['cases']) if c['id'] == args.case)
    fixtures = []
    for name in ('mtpbench-code-general-ja.jsonl', 'gdn-prefill-category-heldouts.jsonl'):
        fixtures.extend(json.loads(l) for l in (_ROOT / 'benchmarks/prompts' / name).read_text().splitlines())
    llm, runner, loading = _resolve_generator(args.artifact, case['prompt_tokens'] + 64)
    generator = llm._get_text_generator()
    rt = get_hip_runtime()
    scratch = Gemma4AttentionScratch()
    original = layer.select_prefill_attention
    captures, seen = [], set()
    counts = {'prefill': 0, 'decode': 0}
    args.snapshots.mkdir(parents=True, exist_ok=True)

    def read_pointer(ptr, shape, dtype):
        host = np.empty(shape, dtype=dtype)
        rt.memcpy(int(host.ctypes.data), int(ptr), host.nbytes, MemcpyKind.DEVICE_TO_HOST)
        return host

    def observed(**geometry):
        selection = original(**geometry)
        launch = selection.launcher

        def invoked(*ptrs, **kw):
            result = launch(*ptrs, **kw)
            phase = 'decode' if kw['tokens'] == 1 else 'prefill'
            signature = tuple(kw[n] for n in ('tokens', 'keys', 'head_dim', 'row_offset',
                                               'num_heads', 'num_kv_heads', 'scale', 'window'))
            if counts[phase] >= args.per_phase_limit or signature in seen:
                return result
            seen.add(signature)
            counts[phase] += 1
            rt.device_synchronize()
            tokens, keys, heads, kv_heads, dim = (kw[n] for n in
                                                ('tokens', 'keys', 'num_heads', 'num_kv_heads', 'head_dim'))
            shape = (tokens, heads, dim)
            wmma = read_pointer(ptrs[4], shape, np.uint16)
            strict_out = malloc(wmma.nbytes)
            try:
                strict_kw = {k: v for k, v in kw.items() if k not in ('scratch', 'library')}
                gemma4_attention_prefill_bf16(*ptrs[:4], strict_out.ptr,
                                             scratch=scratch, **strict_kw)
                rt.device_synchronize()
                strict = read_pointer(strict_out.ptr, shape, np.uint16)
                if not np.array_equal(wmma, read_pointer(ptrs[4], shape, np.uint16)):
                    raise AssertionError('strict diagnostic modified the selected output buffer')
            finally:
                free(strict_out)
            q = read_pointer(ptrs[0], shape, np.uint16)
            k = read_pointer(ptrs[1], (keys, kv_heads, dim), np.uint16)
            v = read_pointer(ptrs[2], (keys, kv_heads, dim), np.uint16)
            mask = read_pointer(ptrs[3], (tokens, keys), np.uint8)
            target = args.snapshots / f'{len(captures):02d}-{phase}-{dim}-{keys}.npz'
            np.savez(target, query=q, key=k, value=v, keep_mask=mask, strict=strict,
                     wmma=wmma, scale=np.array(kw['scale']))
            sampled_rows = sorted(set((0, tokens // 2, tokens - 1)))
            sampled_heads = sorted(set((0, max(0, heads // 2 - 1), heads // 2, heads - 1)))
            comparison = output_comparison(strict, wmma)
            samples = oracle_samples(q, k, v, mask, strict, wmma,
                                     scale=kw['scale'], rows=sampled_rows, heads=sampled_heads)
            captures.append({'phase': phase, 'variant': selection.variant,
                             'selected_output_unchanged': True,
                             'selection_reason': selection.reason,
                             'geometry': {n: kw[n] for n in ('tokens', 'keys', 'num_heads', 'num_kv_heads',
                                                            'head_dim', 'scale', 'window', 'row_offset')},
                             'snapshot': str(target), 'snapshot_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                             'comparison': comparison, 'oracle_samples': samples})
            print(json.dumps({'phase': phase, 'keys': keys, 'head_dim': dim,
                              'variant': selection.variant, **comparison}), flush=True)
            return result
        return replace(selection, launcher=invoked)

    try:
        message = '\n'.join(m['content'] for m in fixtures[index]['messages'])
        prompt = padded_chat(generator, message, case['prompt_tokens'], 20261003 + index)
        prompt_hash = hashlib.sha256(np.array(prompt, dtype=np.int32).tobytes()).hexdigest()
        if prompt_hash != case['prompt_ids_sha256']:
            raise ValueError('saved prompt hash mismatch')
        layer.select_prefill_attention = observed
        outputs = llm.generate(prompt, SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True))
        rt.device_synchronize()
        report = {'kind': 'gemma4_identical_attention_inputs', 'performance_claim': False,
                  'surface': 'hipengine.LLM.generate', 'outputs': outputs,
                  'scope': 'Diagnostic only: saved prompt, public greedy request, identical inputs; sampled float64 oracle, not a teacher-forced production gate.',
                  'case': args.case, 'prompt_ids_sha256': prompt_hash, 'loading': loading,
                  'sources': {str(p.relative_to(_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (_ROOT / 'hipengine/kernels/hip_gfx1100/gemma4').glob('gemma4_attention*.hip')},
                  'provenance': collect_artifact_provenance(repo_root=_ROOT, model_path=args.artifact,
                                                          quant='UD-Q4_K_XL', kv_dtype='bf16'),
                  'captures': captures}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        return 0
    finally:
        layer.select_prefill_attention = original
        scratch.close()
        llm.close()


if __name__ == '__main__':
    raise SystemExit(main())
