#!/usr/bin/env python3
"""Replay Gufo's prepared homogeneous HTTP cohorts from a pinned source tree.

Use one fresh server per workload and width. The pinned source supplies prompt
calibration, text, request controls, stream parsing and barrier-synchronized
cohorts. Its prepared protocol replaces the configured warmup with preparation
before every measured round; it does not run an additional decode warmup.
One repetition preserves the publication sample count. Repeat extensions are
separate runs. No output hash substitutes for native IDs or logit comparison.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_gufo_http_replay import host_state, quality_columns, sha256, validate_paths

WIDTHS = (1, 2, 4, 6, 8)


def checked_round(cohort, *, width, speculative):
    if len(cohort.samples) != width:
        raise ValueError('cohort sample count differs from the requested width')
    if sorted(sample.request_index for sample in cohort.samples) != list(range(width)):
        raise ValueError('cohort request index ownership is not exact')
    if not math.isfinite(cohort.span_ms) or cohort.span_ms <= 0:
        raise ValueError('cohort wall must be finite and positive')
    samples = []
    for sample in cohort.samples:
        counts = [getattr(sample, name) for name in ('prompt_tokens', 'cached_prompt_tokens',
            'prefill_tokens', 'completion_tokens', 'draft_tokens', 'draft_accepted_tokens')]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError('native token counts must be non-negative integers')
        if (abs(sample.prompt_tokens - 2048) > 32 or sample.completion_tokens != 128
                or sample.cached_prompt_tokens + sample.prefill_tokens != sample.prompt_tokens
                or sample.prefill_tokens > 4 or sample.cached_prompt_tokens < sample.prompt_tokens - 4
                or sample.draft_accepted_tokens > sample.draft_tokens):
            raise ValueError('native counts disagree with the prepared pp2048/tg128 protocol')
        if bool(sample.draft_tokens) != speculative:
            raise ValueError('native draft work differs from the requested AR/MTP route')
        if (type(sample.decode_ms) not in (int, float)
                or not math.isfinite(sample.decode_ms) or sample.decode_ms <= 0):
            raise ValueError('native decode time must be finite and positive')
        samples.append(sample.public())
    generated = sum(sample.completion_tokens for sample in cohort.samples)
    return {'samples': samples, 'generated_tokens': generated, 'span_ms': cohort.span_ms,
            'sum_request_decode_output_tok_s': sum(
                sample.completion_tokens * 1000 / sample.decode_ms for sample in cohort.samples),
            'cohort_output_tok_s': generated * 1000 / cohort.span_ms,
            'output_identity': 'visible_text_sha256_not_native_ids'}


def prepared_rounds(serving, url, model, prompt, *, task, width, speculative,
                    repetitions, timeout, record, save=None):
    """Mirror run_corpus_benchmark's homogeneous prefill_first branch."""
    case = serving.PromptCase(f'synthetic_{task}_pp2048', task, prompt)
    common = dict(base_url=url, model=model, cases=[case] * width, temperature=0.0,
                  timeout_seconds=timeout, concurrency=width, group_index=0,
                  endpoint_profile='gufo', cache_prompt=True, pin_slots=False)
    rows = []
    for repetition in range(repetitions):
        prepared = serving._run_corpus_round(**common, max_tokens=1, repetition=-(repetition + 1))
        record('preparation', repetition, [sample.public() for sample in prepared.samples])
        if (len(prepared.samples) != width
                or any(sample.completion_tokens != 1 for sample in prepared.samples)):
            raise ValueError('prompt preparation did not complete for every session')
        measured = serving._run_corpus_round(**common, max_tokens=128, repetition=repetition)
        record('measurement', repetition, [sample.public() for sample in measured.samples])
        rows.append({'repetition': repetition,
                     **checked_round(measured, width=width, speculative=speculative)})
        if save is not None:
            save(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='First target GGUF shard')
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--widths', default=','.join(map(str, WIDTHS)))
    parser.add_argument('--arms', default='ar,mtp-prose,mtp-repetition')
    parser.add_argument('--repetitions', type=int, default=1)
    args = parser.parse_args(argv)
    validate_paths(args.source_root, args.output_dir)
    widths, arms = [int(c) for c in args.widths.split(',')], args.arms.split(',')
    if (not widths or widths != sorted(set(widths)) or any(c not in WIDTHS for c in widths)
            or not arms or len(arms) != len(set(arms))
            or set(arms) - {'ar', 'ar-repetition', 'mtp-prose', 'mtp-repetition'} or args.repetitions < 1):
        parser.error('require unique ordered widths from 1,2,4,6,8, unique arms and positive repetitions')
    source = args.source_root.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / 'tools'))
    from gufo.model_bench import llm as upstream
    from gufo.model_bench.config import load_config
    from gufo import serving_bench as serving
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata

    config_path = source / 'docs/models/qwen3.8-flash-next/artifacts/bench.json'
    config = load_config(source, 'qwen3.8-flash-next', config_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_override = args.output_dir
    config.files = {'gguf': {'default': args.model.resolve()}, 'mtp': {'default': args.sidecar.resolve()}}
    identity = _git_metadata(source)
    journal = args.output_dir / 'requests.jsonl'

    class Session(upstream.Session):
        scope = None
        def request(self, url, prompt, tokens, **kwargs):
            try:
                result = super().request(url, prompt, tokens, **kwargs)
            except BaseException as exc:
                record('request_failure', None, {'error': str(exc), 'prompt': prompt, 'tokens': tokens})
                raise
            record('request', None, {'prompt': prompt, 'tokens': tokens, 'options': kwargs,
                                     'response': result.public()})
            return result

    session = Session(config, 'gufo', gufo_binary=args.binary.resolve(), reference_binary='unused',
        source={'revision': identity['head'], 'dirty': not identity['tracked_clean']}, fingerprint=None,
        log_dir=args.output_dir, document='BENCHMARKS', todo_only=False)
    def record(kind, repetition, result):
        with journal.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'scope': session.scope, 'repetition': repetition,
                                     'result': result}, ensure_ascii=False) + '\n')

    report = {'schema': 1, 'status': 'running', 'performance_claim': False,
        'protocol': 'published_prepared_cohorts' if args.repetitions == 1 else 'prepared_repeat_extension',
        'full_published_width_order': tuple(widths) == WIDTHS,
        'extra_arms': [arm for arm in arms if arm == 'ar-repetition'],
        'warmup_policy': 'prepare every session; no additional decode warmup',
        'repetitions': args.repetitions, 'source_before': identity,
        'binary_sha256': sha256(args.binary), 'harness_sha256': sha256(Path(__file__)),
        'recipe_hashes': {p: sha256(source / p) for p in ('tools/gufo/model_bench/llm.py',
            'tools/gufo/serving_bench.py', 'docs/models/qwen3.8-flash-next/artifacts/bench.json')},
        'command': [str(Path(__file__).relative_to(ROOT)), *(sys.argv[1:] if argv is None else argv)],
        'quality': quality_columns(), 'arms': {}}
    def save():
        (args.output_dir / 'replay.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        for arm in arms:
            mode = 'ar' if arm.startswith('ar') else 'mtp'
            task = 'repetition' if arm.endswith('repetition') else 'prose'
            table = config.table('multi-ar' if mode == 'ar' else 'multi-mtp')
            block = {'capacity_probe': {}, 'widths': {}}
            report['arms'][arm] = block
            # Allocate the largest requested session pool and exercise a short
            # concurrent cohort before paying for full-prompt measurements.
            probe = session.server(table, mode=mode, context=4096, sessions=max(widths), tag=f'{arm}-probe')
            block['capacity_probe']['server_command'] = probe.command
            try:
                with probe:
                    session.scope = {'arm': arm, 'phase': 'capacity_probe'}
                    case = serving.PromptCase('capacity', 'probe', upstream.synthetic_text(101, 128))
                    result = serving._run_corpus_round(base_url=probe.base_url, model=session.request_model,
                        cases=[case]*max(widths), max_tokens=4, temperature=0.0, timeout_seconds=1800,
                        concurrency=max(widths), repetition=0, group_index=0, endpoint_profile='gufo')
                    block['capacity_probe']['samples'] = [sample.public() for sample in result.samples]
                    record('capacity_probe', 0, block['capacity_probe']['samples'])
                    if any(sample.completion_tokens != 4 for sample in result.samples):
                        raise ValueError('short capacity cohort did not complete')
            finally:
                probe.stop()
            save()
            for width in widths:
                server = session.server(table, mode=mode, context=4096, sessions=width, tag=f'{arm}-c{width}')
                cell = {'server_command': server.command, 'host_before': host_state(), 'rows': []}
                block['widths'][str(width)] = cell
                try:
                    with server:
                        session.scope = {'arm': arm, 'width': width, 'phase': 'calibration'}
                        tokenizer = upstream.Tokenizer(session, server.base_url, table.variant)
                        prompt = upstream.turn_prompt(2048-tokenizer.overhead, tokenizer.ratio, task=task)
                        cell['prompt'] = prompt
                        cell['calibration'] = {'overhead': tokenizer.overhead, 'ratio': tokenizer.ratio}
                        session.scope = {'arm': arm, 'width': width, 'phase': 'prepared'}
                        def progress(rows):
                            cell['rows'] = rows
                            save()
                        prepared_rounds(serving, server.base_url, session.request_model, prompt,
                            task=task, width=width, speculative=mode == 'mtp', repetitions=args.repetitions,
                            timeout=1800, record=record, save=progress)
                finally:
                    server.stop()
                    cell['host_after'] = host_state()
                    save()
                print(f'{arm} C{width}: prepared counts and route checked', flush=True)
        report['status'] = 'measured_diagnostic'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report.update(source_after=_git_metadata(source),
                      request_journal_sha256=sha256(journal) if journal.exists() else None)
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
