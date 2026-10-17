"""Functional hard-thinking HTTP gate; run against a hard-policy server."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines()
            if line.startswith('data: ') and line != 'data: [DONE]']


async def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    base = dict(model=args.model, messages=[dict(role='user', content='Think briefly about 2 + 2, then answer with only the number 4.')],
                enable_thinking=True, temperature=0, top_p=1, max_tokens=32)
    cases = [
        ('cap4', dict(hard_think_cap=4, min_answer_tokens=8, soft_close_window=0), 4),
        ('cap0', dict(hard_think_cap=0), 0),
        ('cap1', dict(hard_think_cap=1), 1),
        ('max_only', dict(max_think_tokens=2), 2),
        ('reserve_only', dict(min_answer_tokens=8, max_tokens=16), 7),
        ('custom_close', dict(hard_think_cap=3, hard_close_sequence='\n</think>\n\n'), 3),
        ('json', dict(hard_think_cap=3, max_tokens=64,
                     messages=[dict(role='user', content='Calculate 2+2. Return a JSON object with answer equal to the result.')],
                     response_format={'type':'json_schema', 'json_schema':{'name':'answer','schema':{'type':'object','properties':{'answer':{'type':'integer'}},'required':['answer'],'additionalProperties':False}}}), 3),
    ]
    report = {'source_commit':args.source, 'kind':'functional_controls_only', 'rows':[]}
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2))
    async with httpx.AsyncClient(base_url=args.base_url, timeout=120) as client:
        end = time.monotonic() + 600
        while True:
            try:
                ready = await client.get('/ready')
                if ready.status_code == 200 and ready.json().get('ready'):
                    report['ready_before'] = ready.json(); break
                if ready.json().get('startup', {}).get('error'):
                    raise RuntimeError('startup failed: ' + ready.text)
            except httpx.ConnectError:
                pass
            if time.monotonic() >= end:
                raise TimeoutError('server not ready')
            await asyncio.sleep(1)
        for name, fields, cap in cases:
            close = fields.get('hard_close_sequence', '</think>')
            lowered = await client.post('/v1/hipengine/tokenize', json={'text':close})
            close_ids = lowered.json()['token_ids']
            for mtp in (False, True):
                for stream in (False, True):
                    body = base | fields | {'speculative_mtp':mtp, 'stream':stream}
                    if stream:
                        body['stream_options'] = {'include_usage':True}
                    label = f'{name}-mtp{int(mtp)}-sse{int(stream)}'
                    response = await client.post('/v1/chat/completions', json=body)
                    row = dict(name=label, request=body, status=response.status_code, close_ids=close_ids,
                               body=events(response.text) if stream else response.json())
                    report['rows'].append(row); save()
                    assert response.status_code == 200, row
                    if stream:
                        assert '[DONE]' in response.text
                        assert not any('error' in e for e in row['body']), row
                        choices = [c for e in row['body'] for c in e.get('choices', [])]
                        text = ''.join(c.get('delta', {}).get('content', '') or '' for c in choices)
                        terminal = next(c for c in reversed(choices) if c.get('finish_reason'))
                        details = terminal.get('finish_details', {})
                    else:
                        choice = row['body']['choices'][0]
                        text = choice['message'].get('content', '')
                        details = choice.get('finish_details', {})
                        ids = choice.get('hipengine', {}).get('generated_token_ids', [])
                        assert any(ids[i:i+len(close_ids)] == close_ids for i in range(len(ids))), row
                        assert row['body']['hipengine']['speculative_mtp']['used'] is False, row
                    assert text.strip(), row
                    assert details.get('reasoning_tokens', -1) >= 0, row
                    assert details['reasoning_tokens'] <= cap + len(close_ids), row
                    assert details.get('phase') == 'answer', row
                    if name == 'json':
                        assert json.loads(text) == {'answer':4}, row
                    row['passed'] = True; save()
                    print(label, 'PASS', json.dumps(details), flush=True)
        # Explicit hint is intentionally not a hard cap: prove it still executes MTP.
        body = base | {'hard_think_cap':1, 'speculative_mtp':{'enabled':True,'thinking':'hint'}}
        response = await client.post('/v1/chat/completions', json=body)
        row = dict(name='intentional-hint', request=body, status=response.status_code, body=response.json())
        report['rows'].append(row); save()
        assert response.status_code == 200, row
        mtp = row['body']['hipengine']['speculative_mtp']
        assert mtp['used'] and mtp['draft_tokens'] > 0 and mtp['thinking_controls'] == 'prompt_hint_only', row
        row['passed'] = True
        ready = await client.get('/ready'); ready.raise_for_status()
        report['ready_after'] = ready.json(); report['passed'] = True; save()
        print('THINKING_GATE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18389')
    parser.add_argument('--model', default='qwen38fn-hardthink')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    asyncio.run(run(parser.parse_args()))
