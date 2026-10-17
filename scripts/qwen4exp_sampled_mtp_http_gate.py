#!/usr/bin/env python3
"""Sampled-MTP HTTP functional gate, not a task-quality or performance gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qwen4exp_sampling_http_gate import parse_response, unpack, validate
PROMPTS = (ROOT / 'benchmarks/prompts/mtpbench-code-general-ja.jsonl',
           ROOT / 'benchmarks/prompts/gdn-prefill-category-heldouts.jsonl')
CATEGORIES = {'code', 'general_en', 'general_ja', 'mixed_ja_en'}


def extract(body, *, chat: bool) -> dict:
    events = body if isinstance(body, list) else [body]
    assert events and not any('error' in event for event in events), events
    text, tokens, ids, finish, reason, usage, metadata = '', [], None, None, None, None, {}
    algorithm, execution_path = None, None
    for event in events:
        metadata.update(event.get('hipengine') or {})
        usage = event.get('usage') or usage
        for choice in event.get('choices', []):
            piece, logs = unpack(choice, chat=chat)
            text += piece
            tokens.extend(logs)
            extra = choice.get('hipengine') or {}
            # Buffered/live SSE attach execution summaries to the terminal
            # choice, while blocking responses attach them to the root.
            for key in ('speculative_mtp', 'generation_shape'):
                if key in extra:
                    metadata[key] = extra[key]
            ids = extra.get('generated_token_ids') or ids
            finish = extra.get('finish_details') or finish
            algorithm = (extra.get('diagnostics') or {}).get('sampling_algorithm') or algorithm
            execution_path = (extra.get('decode_state') or {}).get('execution_path') or execution_path
            reason = choice.get('finish_reason') or reason
    assert ids and reason and usage, ('missing terminal accounting', ids, reason, usage)
    return dict(text=text, tokens=tokens, ids=ids, finish=finish, finish_reason=reason,
                usage=usage, mtp=metadata.get('speculative_mtp'),
                generation_shape=metadata.get('generation_shape'),
                sampling_algorithm=algorithm, execution_path=execution_path)


def validate_route(result: dict, *, mtp: bool, logs: bool = True, full_mass: bool = True):
    summary = result['mtp']
    assert summary and summary['used'] is mtp, summary
    if mtp:
        assert summary['draft_tokens'] > 0 and summary['draft_cycles'] > 0, summary
        assert summary['accepted_draft_tokens'] + summary['rejected_draft_tokens'] == summary['draft_tokens'], summary
        assert result['sampling_algorithm'] == 'target_draw_match_deterministic_draft', result
        assert result['execution_path'] == 'qwen4exp_mtp_batched_state_rows_verify', result
        assert result['finish']['sampler_mode'] in {'sampled_speculative_mtp', 'processed_speculative_mtp'}, result['finish']
    assert result['usage']['completion_tokens'] == len(result['ids']), result['usage']
    if logs:
        validate(result['text'], result['tokens'], full_mass=full_mass)
    else:
        assert result['text'] and not result['tokens']


def fingerprint(result: dict):
    return {key: result[key] for key in ('text', 'tokens', 'ids', 'finish', 'finish_reason', 'usage')}


def load_prompts():
    rows = []
    for path in PROMPTS:
        group = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert {row['category'] for row in group} == CATEGORIES, path
        rows.extend(dict(row, suite=path.name) for row in group)
    assert len({row['id'] for row in rows}) == len(rows)
    return rows


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(kind='functional_sampled_mtp_only', source_commit=args.source_commit,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  prompts=[dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in PROMPTS],
                  rows=[], comparisons=[])
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        deadline = time.monotonic() + 600
        while True:
            try:
                ready = await client.get('/ready')
                if ready.status_code == 200 and ready.json().get('ready'):
                    report['ready_before'] = ready.json()
                    break
            except httpx.ConnectError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('server not ready')
            await asyncio.sleep(1)
        report['capabilities'] = (await client.get('/v1/hipengine/capabilities')).json()
        for spelling in ('speculative', 'speculative_mtp'):
            caps = report['capabilities']['sampling'][spelling]
            assert caps['processed_target_verification'] and not caps['default_enabled'], caps
            assert 'temperature' in caps['supported_sampling_fields'], caps

        async def execute(name, body, *, chat=True, expected_mtp=True, full_mass=True):
            endpoint = '/v1/chat/completions' if chat else '/v1/completions'
            if body.get('stream'):
                body = body | dict(stream_options=(body.get('stream_options') or {}) | dict(
                    include_usage=True, include_hipengine=True))
            row = dict(name=name, endpoint=endpoint, request=body)
            report['rows'].append(row)
            started = time.monotonic()
            try:
                response = await client.post(endpoint, json=body)
                row.update(status=response.status_code, body=parse_response(response, stream=body.get('stream', False)))
                assert response.status_code == 200, row
                if body.get('stream'):
                    assert isinstance(row['body'], list) and 'data: [DONE]' in response.text
                result = extract(row['body'], chat=chat)
                row['result'] = result
                validate_route(result, mtp=expected_mtp, logs=bool(body.get('logprobs')), full_mass=full_mass)
                row['passed'] = True
            except Exception as error:
                row.update(passed=False, error=repr(error))
            row['elapsed_s_diagnostic'] = time.monotonic() - started
            save()
            print(name, 'PASS' if row['passed'] else 'FAIL ' + row['error'][:300], flush=True)
            return row

        def compare(name, left, right, *, different=False, include_logs=True):
            check = dict(name=name, left=left['name'], right=right['name'])
            try:
                assert left['passed'] and right['passed'], 'row failed'
                a, b = fingerprint(left['result']), fingerprint(right['result'])
                if not include_logs:
                    a.pop('tokens'); b.pop('tokens')
                assert (a['ids'] != b['ids'] if different else a == b), 'repeat/seed/stream mismatch'
                check['passed'] = True
            except Exception as error:
                check.update(passed=False, error=repr(error))
            report['comparisons'].append(check)
            save()

        def finish_report():
            report['passed'] = (bool(report['rows']) and all(row['passed'] for row in report['rows'])
                                and all(c['passed'] for c in report['comparisons'])
                                and report['ready_after'].get('ready') is True)
            save()
            print('SAMPLED_MTP_HTTP_GATE_COMPLETE', report['passed'], flush=True)
            return report

        if args.replay_report:
            baseline = json.loads(args.replay_report.read_text())
            report['baseline'] = dict(path=str(args.replay_report),
                sha256=hashlib.sha256(args.replay_report.read_bytes()).hexdigest())
            originals = {row['name']: row for row in baseline['rows']}
            names = set(args.case or [r['name'] for r in baseline['rows'] if not r['passed']])
            assert names and names <= originals.keys(), 'unknown or empty replay selection'
            repaired = {}
            for name, old in originals.items():
                if name not in names:
                    continue
                row = await execute(name, old['request'], chat=old['endpoint'].endswith('chat/completions'),
                                    expected_mtp=not name.endswith('-ar'))
                repaired[name] = row
                for key in ('category', 'suite'):
                    if key in old:
                        row[key] = old[key]
                if row.get('result') and old.get('result'):
                    # Metadata repair may change the finish reason, never IDs/logs.
                    row['baseline_content_equal'] = all(row['result'][key] == old['result'][key]
                                                        for key in ('text', 'tokens', 'ids', 'usage'))
                    if not row['baseline_content_equal']:
                        row.update(passed=False, error='baseline content changed')
            merged = originals | repaired
            for check in baseline['comparisons']:
                if check['left'] in names or check['right'] in names:
                    compare(check['name'], merged[check['left']], merged[check['right']],
                            different=check['name'].endswith('-different-seed'),
                            include_logs=not check['name'].endswith('-no-log-drift'))
            report['ready_after'] = (await client.get('/ready')).json()
            return finish_report()

        refs = []
        for index, prompt in enumerate(load_prompts()):
            body = dict(model=args.model, messages=prompt['messages'], enable_thinking=False,
                        max_tokens=args.max_tokens, temperature=.8, top_k=2, seed=104,
                        logprobs=True, top_logprobs=5)
            if index % 2:
                body.update(top_k=5, top_p=.9, min_p=.05, repetition_penalty=1.1,
                            presence_penalty=.15, frequency_penalty=.1)
            name = prompt['id']
            ar = await execute(name + '-ar', body | dict(speculative_mtp=False), expected_mtp=False)
            generic = await execute(name + '-generic', body | dict(speculative=True))
            explicit = await execute(name + '-explicit', body | dict(speculative_mtp=True))
            sse = await execute(name + '-sse', body | dict(speculative=True, stream=True,
                                                        stream_options=dict(include_usage=True)))
            for row in (ar, generic, explicit, sse):
                row.update(category=prompt['category'], suite=prompt['suite'])
            compare(name + '-spellings', generic, explicit)
            compare(name + '-blocking-sse', generic, sse)
            # Cross-width AR/MTP ID equality is diagnostic, not a production gate.
            if ar.get('result') and generic.get('result'):
                generic['ar_ids_equal_diagnostic'] = ar['result']['ids'] == generic['result']['ids']
            refs.append((name, body, generic, prompt))
        for name, body, generic, prompt in reversed(refs):
            repeated = await execute(name + '-after-neighbors', body | dict(speculative_mtp=True))
            repeated.update(category=prompt['category'], suite=prompt['suite'])
            compare(name + '-neighbor-isolation', generic, repeated)

        # Completion endpoint and no-logprob route also have to execute MTP.
        prompt = 'Write a list of invented names for a tiny moon robot.'
        body = dict(model=args.model, prompt=prompt, max_tokens=48, temperature=.8,
                    top_k=2, seed=104, logprobs=5)
        for spelling in ('speculative', 'speculative_mtp'):
            base = await execute('completion-' + spelling, body | {spelling: True}, chat=False)
            sse = await execute('completion-' + spelling + '-sse', body | {spelling: True, 'stream': True,
                                 'stream_options': dict(include_usage=True)}, chat=False)
            compare('completion-' + spelling + '-blocking-sse', base, sse)
            other = await execute('completion-' + spelling + '-seed105', body | {spelling: True, 'seed': 105}, chat=False)
            compare('completion-' + spelling + '-different-seed', base, other, different=True)
            no_logs_body = {k: v for k, v in body.items() if k != 'logprobs'}
            no_logs = await execute('completion-' + spelling + '-no-logs', no_logs_body | {spelling: True}, chat=False)
            compare('completion-' + spelling + '-no-log-drift', base, no_logs, include_logs=False)
        report['ready_after'] = (await client.get('/ready')).json()
        return finish_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18389')
    parser.add_argument('--model', default='qwen38fn-sampling')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--max-tokens', type=int, default=48)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-report', type=Path, help='Rerun only failed cases from a saved report')
    parser.add_argument('--case', action='append', help='Exact replay row name (repeatable); requires --replay-report')
    args = parser.parse_args()
    if args.case and not args.replay_report:
        parser.error('--case requires --replay-report')
    return 0 if asyncio.run(run(args))['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
