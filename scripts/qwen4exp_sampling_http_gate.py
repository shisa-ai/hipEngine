#!/usr/bin/env python3
"""Functional Flash-Next sampling/logprob HTTP gate; not a speed or quality gate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time

import httpx


def unpack(choice: dict, *, chat: bool) -> tuple[str, list[dict]]:
    text = ((choice.get('message') or choice.get('delta') or {}).get('content') or '') if chat else choice.get('text', '')
    metadata = choice.get('logprobs')
    if not metadata:
        return text, []
    if chat:
        return text, metadata['content']
    extra = metadata['hipengine']
    return text, [dict(token=token, logprob=lp, bytes=raw, top_logprobs=top)
                  for token, lp, raw, top in zip(metadata['tokens'], metadata['token_logprobs'],
                                                extra['token_bytes'], extra['top_logprob_tokens'], strict=True)]


def validate(text: str, tokens: list[dict], *, full_mass: bool) -> None:
    assert text and tokens, ('empty visible output', text)
    assert all(t['bytes'] is not None for t in tokens), 'missing selected bytes'
    reconstructed = bytes(b for t in tokens for b in t['bytes']).decode('utf-8', errors='replace')
    assert reconstructed == text, ('byte alignment', text, reconstructed)
    for t in tokens:
        lp, top = t['logprob'], t['top_logprobs']
        assert lp is not None and math.isfinite(lp) and lp <= 1e-6, t
        assert top and all(p['bytes'] is not None and math.isfinite(p['logprob']) and p['logprob'] <= 1e-6 for p in top), t
        mass = sum(math.exp(p['logprob']) for p in top)
        assert mass <= 1 + 1e-6, mass
        if full_mass:
            assert abs(mass - 1) <= 1e-6, mass
            assert any(p['bytes'] == t['bytes'] and abs(p['logprob'] - lp) <= 1e-6 for p in top), t


def parse_response(response: httpx.Response, *, stream: bool):
    if stream and response.headers.get('content-type', '').startswith('text/event-stream'):
        return [json.loads(line[6:]) for line in response.text.splitlines()
                if line.startswith('data: ') and line != 'data: [DONE]']
    return response.json()


def validate_rejection(body, status: int) -> None:
    events = body if isinstance(body, list) else [body]
    assert status in (200, 400), status
    errors = [event['error'] for event in events if 'error' in event]
    assert len(errors) == 1, events
    error = errors[0]
    assert error['code'] == 'unsupported_parameter' and error['param'] == 'speculative', error
    assert error['hipengine']['speculative']['compatibility_guard'] == 'raw_greedy_bf16_c1', error
    assert 'temperature' in error['hipengine']['speculative']['blockers'], error
    assert not any((c.get('delta', {}).get('content') or c.get('text')) for event in events for c in event.get('choices', [])), events


async def run(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = {}
    if args.baseline_report:
        baseline = {row['name']: row for row in json.loads(args.baseline_report.read_text())['rows']}
    report = dict(source_commit=args.source_commit, rows=[], kind='functional_sampling_only',
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        deadline = time.monotonic() + 600
        while True:
            try:
                response = await client.get('/ready')
                if response.status_code == 200 and response.json().get('ready'):
                    report['ready_before'] = response.json()
                    break
            except httpx.ConnectError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('server not ready')
            await asyncio.sleep(1)
        common = dict(model=args.model, max_tokens=32, temperature=.8, top_p=1, top_k=2, seed=104,
                      speculative_mtp=False)
        prompts = dict(en='Write a list of invented names for a tiny moon robot.',
                       ja='日本語で、猫と月について短い物語を一文だけ書いてください。')
        cases = [
            ('greedy', 'en', dict(temperature=0, top_k=0)),
            ('sampled', 'en', {}), ('repeat1', 'en', {}), ('other_seed', 'en', dict(seed=105)),
            ('ja', 'ja', {}), ('mtp_requested', 'en', dict(speculative_mtp=True)),
            ('generic_requested', 'en', dict(speculative_mtp=None, speculative=True)),
        ]
        pairs = {}
        for chat in (False, True):
            selected = cases + ([('ja_stop', 'ja', dict(stop=['。'])),
                                 ('hard_thinking', 'en', dict(enable_thinking=True, hard_think_cap=4,
                                                            min_answer_tokens=8))] if chat else [])
            if args.case:
                requested = set(args.case)
                if requested.intersection({'repeat1', 'other_seed', 'mtp_requested'}):
                    requested.add('sampled')
                selected = [case for case in selected if case[0] in requested]
            for name, language, fields in selected:
                body = common | fields
                if body.get('speculative_mtp') is None:
                    body.pop('speculative_mtp')
                if chat:
                    body.update(messages=[dict(role='user', content=prompts[language])],
                                enable_thinking=bool(fields.get('enable_thinking')), logprobs=True, top_logprobs=5)
                else:
                    body.update(prompt=prompts[language], logprobs=5)
                endpoint = '/v1/chat/completions' if chat else '/v1/completions'
                for stream in (False, True):
                    request = body | dict(stream=stream)
                    if stream:
                        request['stream_options'] = dict(include_usage=True)
                    label = f'{"chat" if chat else "completion"}-{name}-sse{int(stream)}'
                    row = dict(name=label, endpoint=endpoint, request=request,
                               check_kind='early_rejection_only' if name == 'generic_requested' else 'sampling_logprobs')
                    report['rows'].append(row)
                    try:
                        response = await client.post(endpoint, json=request)
                        row.update(status=response.status_code, body=parse_response(response, stream=stream))
                        if name == 'generic_requested':
                            validate_rejection(row['body'], response.status_code)
                        else:
                            assert response.status_code == 200, row
                            if stream:
                                assert isinstance(row['body'], list) and 'data: [DONE]' in response.text
                                assert not any('error' in event for event in row['body']), row
                                chunks = [choice for event in row['body'] for choice in event.get('choices', [])]
                                pieces = [unpack(choice, chat=chat) for choice in chunks]
                                text, tokens = ''.join(t for t, _ in pieces), [token for _, ts in pieces for token in ts]
                                assert any(choice.get('finish_reason') for choice in chunks)
                                assert any(event.get('usage') for event in row['body'])
                                assert (text, tokens) == pairs[(chat, name)], 'blocking/SSE mismatch'
                            else:
                                text, tokens = unpack(row['body']['choices'][0], chat=chat)
                                pairs[(chat, name)] = text, tokens
                                if name in ('repeat1', 'mtp_requested'):
                                    assert (text, tokens) == pairs[(chat, 'sampled')], 'seed/fallback drift'
                                if name == 'other_seed':
                                    assert text != pairs[(chat, 'sampled')][0], 'seed did not change this sequence'
                                if name == 'mtp_requested':
                                    assert row['body']['hipengine']['speculative_mtp']['used'] is False
                                old = baseline.get(label)
                                if old and old['status'] == 200:
                                    before, after = old['body']['choices'][0], row['body']['choices'][0]
                                    assert before['hipengine']['generated_token_ids'] == after['hipengine']['generated_token_ids'], 'generated IDs changed'
                                    row['baseline_ids_equal'] = True
                            validate(text, tokens, full_mass=body['temperature'] > 0 and name != 'hard_thinking')
                            row.update(visible_tokens=len(tokens), visible_text=text)
                        row['passed'] = True
                    except Exception as error:
                        row.update(passed=False, error=repr(error))
                    save()
                    print(label, 'PASS' if row['passed'] else 'FAIL ' + row['error'][:250], flush=True)
        report['ready_after'] = (await client.get('/ready')).json()
        report['passed'] = bool(report['rows']) and all(row['passed'] for row in report['rows'])
        save()
        print('SAMPLING_HTTP_GATE_COMPLETE', report['passed'], flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18389')
    parser.add_argument('--model', default='qwen38fn-sampling')
    parser.add_argument('--source-commit', required=True, help='Commit loaded by the already-started server')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--case', action='append', choices=('greedy', 'sampled', 'repeat1', 'other_seed',
                        'ja', 'mtp_requested', 'generic_requested', 'ja_stop', 'hard_thinking'),
                        help='Run only this case (repeatable); include required seed reference automatically')
    args = parser.parse_args()
    return 0 if asyncio.run(run(args))['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
