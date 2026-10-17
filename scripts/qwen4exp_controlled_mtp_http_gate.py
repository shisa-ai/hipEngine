#!/usr/bin/env python3
"""Functional structured-MTP HTTP probe; not numerical, task-quality or speed evidence."""
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
from scripts.qwen4exp_sampled_mtp_http_gate import extract
from scripts.qwen4exp_sampling_http_gate import parse_response, validate

EXPECTED = {'answer': 'alpha beta gamma', 'number': 42}
SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'result', 'strict': True,
          'schema': {'const': EXPECTED}}}
TOOL = {'type': 'function', 'function': {'name': 'weather', 'description': 'Get weather.',
        'parameters': {'type': 'object', 'properties': {'city': {'const': 'Tokyo'}},
                       'required': ['city'], 'additionalProperties': False}}}


def structured(body):
    events = body if isinstance(body, list) else [body]
    reasoning, calls = [], {}
    for event in events:
        for choice in event.get('choices', []):
            msg = choice.get('message') or choice.get('delta') or {}
            reasoning.append(msg.get('reasoning_content') or '')
            for position, call in enumerate(msg.get('tool_calls') or []):
                index = call.get('index', position)
                entry = calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                entry['id'] += call.get('id') or ''
                entry['name'] += call.get('function', {}).get('name') or ''
                entry['arguments'] += call.get('function', {}).get('arguments') or ''
    return dict(reasoning=''.join(reasoning), calls=[calls[i] for i in sorted(calls)])


def validate_control(result, *, kind, logs):
    summary = result['mtp']
    assert summary and summary['used'] is True, summary
    assert summary['draft_tokens'] > 0 and summary['draft_cycles'] > 0, summary
    assert summary['accepted_draft_tokens'] + summary['rejected_draft_tokens'] == summary['draft_tokens'], summary
    assert result['execution_path'] == 'qwen4exp_mtp_batched_state_rows_verify', result
    assert result['sampling_algorithm'] == 'target_draw_match_deterministic_draft', result
    assert result['usage']['completion_tokens'] == len(result['ids']), result
    if kind in ('required_tool', 'auto_tool'):
        assert result['finish_reason'] == 'tool_calls', result
        assert len(result['calls']) == 1, result
        call = result['calls'][0]
        assert call['id'] and call['name'] == 'weather', call
        assert json.loads(call['arguments']) == {'city': 'Tokyo'}, call
    else:
        assert result['finish_reason'] == 'stop', result
        assert json.loads(result['text']) == EXPECTED, result
        assert not result['calls'], result
        if logs:
            validate(result['text'], result['tokens'], full_mass=False)
    if kind == 'hard_thinking':
        assert result['reasoning'], result
    else:
        assert not result['reasoning'], result


def cases(model):
    for kind in ('schema', 'required_tool', 'auto_tool', 'auto_answer', 'hard_thinking'):
        base = dict(model=model, max_tokens=128, seed=104, enable_thinking=False,
                    messages=[dict(role='user', content='Return the requested JSON object.')])
        if kind in ('schema', 'auto_answer', 'hard_thinking'):
            base['response_format'] = SCHEMA
        if kind in ('required_tool', 'auto_tool', 'auto_answer'):
            base.update(tools=[TOOL], tool_choice='required' if kind == 'required_tool' else 'auto',
                        parallel_tool_calls=False)
            base['messages'] = [dict(role='user', content=(
                'Call the weather tool for Tokyo now.' if kind != 'auto_answer' else
                'Do not call any tools. Return the requested JSON answer directly.'))]
        if kind == 'hard_thinking':
            base.update(enable_thinking=True, hard_think_cap=4, min_answer_tokens=48)
        for temperature in (0., .8):
            for spelling in ('speculative', 'speculative_mtp'):
                for logs in (False, True):
                    for stream in (False, True):
                        body = base | dict(temperature=temperature, stream=stream, **{spelling: True})
                        if logs:
                            body.update(logprobs=True, top_logprobs=5)
                        if stream:
                            body['stream_options'] = dict(include_usage=True, include_hipengine=True)
                        yield dict(name=f'{kind}-t{temperature}-{spelling}-logs{int(logs)}-sse{int(stream)}',
                                   kind=kind, request=body)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(kind='functional_structured_mtp_only', source_commit=args.source_commit,
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[], comparisons=[])
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    selected = list(cases(args.model))
    if args.replay_report:
        previous = json.loads(args.replay_report.read_text())
        selected = [dict(name=r['name'], kind=r['kind'], request=r['request'])
                    for r in previous['rows'] if not r['passed'] or (args.case and r['name'] in args.case)]
        report['baseline'] = dict(path=str(args.replay_report), sha256=hashlib.sha256(args.replay_report.read_bytes()).hexdigest())
    if args.case:
        selected = [r for r in selected if r['name'] in args.case]
        assert {r['name'] for r in selected} == set(args.case), 'unknown case'
    assert selected, 'empty probe'
    async with httpx.AsyncClient(base_url=args.base_url, timeout=240) as client:
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
        for row in selected:
            report['rows'].append(row)
            started = time.monotonic()
            try:
                response = await client.post('/v1/chat/completions', json=row['request'])
                stream = row['request']['stream']
                row.update(status=response.status_code, body=parse_response(response, stream=stream))
                assert response.status_code == 200, row['body']
                if stream:
                    assert isinstance(row['body'], list) and 'data: [DONE]' in response.text
                result = extract(row['body'], chat=True) | structured(row['body'])
                row['result'] = result
                validate_control(result, kind=row['kind'], logs=bool(row['request'].get('logprobs')))
                row['passed'] = True
            except Exception as error:
                row.update(passed=False, error=repr(error))
            row['elapsed_s_diagnostic'] = time.monotonic() - started
            save()
            print(row['name'], 'PASS' if row['passed'] else 'FAIL ' + row['error'][:500], flush=True)
        # Compare buffered SSE with the matching blocking request, ignoring random call IDs.
        by_name = {r['name']: r for r in report['rows']}
        for row in report['rows']:
            if not row['request']['stream']:
                continue
            other = by_name.get(row['name'].removesuffix('1') + '0')
            if other is None:
                continue
            def fingerprint(r):
                result = r['result']
                return {k: result[k] for k in ('text', 'reasoning', 'ids', 'tokens', 'finish_reason', 'usage')} | dict(
                    calls=[{k: c[k] for k in ('name', 'arguments')} for c in result['calls']])
            passed = row['passed'] and other['passed'] and fingerprint(row) == fingerprint(other)
            report['comparisons'].append(dict(left=other['name'], right=row['name'], passed=passed))
        report['ready_after'] = (await client.get('/ready')).json()
        report['passed'] = (all(r['passed'] for r in report['rows']) and
                            all(r['passed'] for r in report['comparisons']) and report['ready_after'].get('ready') is True)
        save()
        print('STRUCTURED_MTP_GATE_COMPLETE', report['passed'], flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18389')
    parser.add_argument('--model', default='qwen38fn-sampling')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-report', type=Path)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    return 0 if asyncio.run(run(args))['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
