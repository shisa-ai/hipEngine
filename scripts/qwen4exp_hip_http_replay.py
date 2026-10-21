#!/usr/bin/env python3
"""Adapt Gufo's ordered depth recipe to hipEngine's public prompt checkpoints.

Use a fresh hipEngine server for each arm. The pinned Gufo source supplies text,
calibration, prefix replies and retry order. Before each measured continuation,
prepare the shared token prefix of its rendered conversation and prior history.
This extra untimed checkpoint replaces Gufo's live reply-state reuse; it is not
an exact replay of Gufo's cache implementation. Native counts must confirm reuse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_gufo_http_replay import DEPTHS, host_state, ordered_depths, quality_columns, sha256, validate_paths
from scripts.qwen4exp_http_mtp_economics import _one, _token_ids, route_matches_request
from scripts.qwen4exp_mtp_inprocess_bench import source_snapshot


def checkpoint_prefix(full, history):
    count = 0
    for left, right in zip(full, history):
        if left != right:
            break
        count += 1
    if not 0 < count < len(full):
        raise ValueError("history must share a nonempty proper prefix with the measured prompt")
    return full[:count]


class DepthSession:
    def __init__(self, client, model, speculative, journal):
        self.client, self.model, self.speculative, self.journal = client, model, speculative, journal
        self.tokenizer_calibration = {}
        self.scope = None

    def record(self, kind, inputs, output):
        with self.journal.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'scope': self.scope, 'inputs': inputs,
                                     'output': output}, ensure_ascii=False) + '\n')

    def render(self, url, messages):
        response = self.client.post(f'{url}/v1/hipengine/count_tokens', json={
            'messages': messages, 'enable_thinking': False})
        response.raise_for_status()
        rendered = response.json()
        response = self.client.post(f'{url}/v1/hipengine/tokenize', json={'text': rendered['text']})
        response.raise_for_status()
        tokens = response.json()
        ids = _token_ids(tokens['token_ids'])
        if len(ids) != tokens['token_count'] or len(ids) != rendered['token_count']:
            raise ValueError('rendered prompt and native tokenizer counts disagree')
        return ids

    def send(self, url, ids, max_tokens, kind, *, require_max_tokens=True, messages=None):
        row = {'id': kind, 'prompt_token_ids': ids}
        inputs = {**row, 'messages': messages, 'max_tokens': max_tokens,
                  'speculative_mtp': self.speculative, 'ignore_eos': False,
                  'require_max_tokens': require_max_tokens}
        try:
            case = _one(self.client, url, self.model, row, self.speculative, max_tokens,
                        ignore_eos=False, require_max_tokens=require_max_tokens)
            if not route_matches_request(case):
                raise ValueError(f'requested AR/MTP route did not run: {case}')
            if case['phase_metrics'] is None:
                raise ValueError('response has no native phase accounting')
        except BaseException as exc:
            self.record(kind + '_failure', inputs, {'error': str(exc)})
            raise
        self.record(kind, inputs, case)
        return case

    def request(self, base_url, prompt, max_tokens, *, index=0, messages=None,
                cache_prompt='default', extra_body=None):
        if extra_body or cache_prompt != 'default':
            raise ValueError('this depth adapter implements thinking-off default cache requests only')
        messages = messages if messages is not None else [{'role': 'user', 'content': prompt}]
        ids = self.render(base_url, messages)
        checkpoint = None
        if len(messages) > 1:
            checkpoint = checkpoint_prefix(ids, self.render(base_url, messages[:-1]))
            self.send(base_url, checkpoint, 1, 'checkpoint')
        case = self.send(base_url, ids, max_tokens, 'request', messages=messages)
        counts = case['phase_metrics']['counts']
        if checkpoint is not None and counts['cached_prompt_tokens'] != len(checkpoint):
            raise ValueError('measured continuation did not reuse the declared prompt checkpoint')
        case['checkpoint_tokens'] = 0 if checkpoint is None else len(checkpoint)
        times = case['phase_metrics']['times_ms']
        diagnostics = case['diagnostics']
        return SimpleNamespace(
            prompt_tokens=counts['prompt_tokens'], cached_prompt_tokens=counts['cached_prompt_tokens'],
            prefill_tokens=counts['new_prompt_tokens'], completion_tokens=counts['generated_tokens'],
            prefill_ms=times['prefill_wall_ms'], decode_ms=times['decode_wall_ms'],
            draft_tokens=diagnostics.get('proposed_draft_tokens', 0),
            draft_accepted_tokens=diagnostics.get('accepted_draft_tokens', 0), public=lambda: case)

    def chat_text(self, base_url, messages, max_tokens, extra_body=None):
        if extra_body:
            raise ValueError('prefix replies use thinking-off controls')
        case = self.send(base_url, self.render(base_url, messages), max_tokens, 'prefix_reply',
                         require_max_tokens=False, messages=messages)
        return case['text'] or ''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True, help='Read-only pinned Gufo recipe checkout')
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True, help='Served hipEngine model alias')
    parser.add_argument('--server-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--arm', choices=('ar','mtp-prose','mtp-repetition'), required=True)
    parser.add_argument('--depths', default=','.join(map(str, DEPTHS)))
    parser.add_argument('--context', type=int, default=133760)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    validate_paths(args.source_root, args.output_dir)
    depths = [int(value) for value in args.depths.split(',')]
    if (not depths or depths != sorted(set(depths)) or min(depths) < 0
            or args.context < max(depths) + 2176 or args.repetitions < 1 or args.timeout <= 0):
        parser.error('require ordered unique depths, sufficient context, positive repeats and timeout')
    manifest = json.loads(args.server_manifest.read_text())
    command = manifest['command']
    if (int(command[command.index('--max-context-tokens') + 1]) != args.context
            or command[command.index('--served-model-name') + 1] != args.model):
        parser.error('server manifest context or model alias differs from the replay')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root.resolve() / 'tools'))
    from gufo.model_bench import llm as upstream
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata
    import httpx

    args.output_dir.mkdir(parents=True, exist_ok=True)
    journal = args.output_dir / 'requests.jsonl'
    report_path = args.output_dir / 'replay.json'
    quality = quality_columns()
    quality['G1'] = {'status': 'pending', 'reason': 'native output IDs recorded; bind identical-input comparison'}
    quality['G2'] = {'status': 'pending', 'reason': 'bind full-vocabulary strict-relative numerical packet'}
    report = {'schema': 1, 'status': 'running', 'performance_claim': False,
        'protocol': 'hipengine_ordered_prompt_checkpoint_adaptation',
        'arm': args.arm, 'depths': depths, 'repetitions': args.repetitions,
        'full_published_depth_order': tuple(depths) == DEPTHS,
        'source_before': source_snapshot(), 'host_before': host_state(),
        'recipe_source': _git_metadata(args.source_root),
        'recipe_hashes': {p: sha256(args.source_root / p) for p in (
            'tools/gufo/model_bench/llm.py', 'tools/gufo/serving_bench.py',
            'docs/models/qwen3.8-flash-next/artifacts/bench.json')},
        'harness_sha256': sha256(Path(__file__)), 'server_manifest': manifest,
        'helper_sha256': {p: sha256(ROOT / p) for p in (
            'scripts/qwen4exp_gufo_http_replay.py', 'scripts/qwen4exp_http_mtp_economics.py')},
        'server_manifest_sha256': sha256(args.server_manifest),
        'command': [str(Path(__file__).relative_to(ROOT)), *(sys.argv[1:] if argv is None else argv)],
        'timing_scope': 'hipEngine backend wall including scheduler waits, not GPU intervals',
        'cache_protocol': 'untimed one-output raw-token checkpoint of the common rendered-history prefix; exact native reuse required',
        'quality': quality, 'rows': []}
    def save(rows=None):
        if rows is not None:
            report['rows'] = [{**r, 'output_identity': 'response_owned_native_token_ids'} for r in rows]
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        with httpx.Client(timeout=args.timeout) as client:
            session = DepthSession(client, args.model, args.arm != 'ar', journal)
            session.scope = {'phase': 'calibration'}
            tokenizer = upstream.Tokenizer(session, args.url)
            report['calibration'] = {'overhead': tokenizer.overhead, 'initial_ratio': tokenizer.ratio}
            ordered_depths(upstream, session, args.url, tokenizer, depths=depths,
                           repetitions=args.repetitions, task='repetition' if args.arm == 'mtp-repetition' else 'prose',
                           speculative=args.arm != 'ar', save=save)
            report['calibration']['final_ratio'] = tokenizer.ratio
        report['status'] = 'measured_diagnostic'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report.update(source_after=source_snapshot(), host_after=host_state(),
                      recipe_source_after=_git_metadata(args.source_root),
                      request_journal_sha256=sha256(journal) if journal.exists() else None)
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
