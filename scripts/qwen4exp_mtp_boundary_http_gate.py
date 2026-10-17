#!/usr/bin/env python3
"""Live MTP stop/EOS boundary checks derived from captured baseline output."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen4exp_sampled_mtp_http_gate import extract, validate_route
from scripts.qwen4exp_sampling_http_gate import parse_response


def choose_boundary(result):
    text, ids = result['text'], result['ids']
    assert len(text) >= 30 and len(ids) >= 16, 'baseline too short to test interior boundaries'
    stop = text[10:30]
    assert text.find(stop) == 10, 'ambiguous stop fixture'
    index = next(i for i in range(6, 14) if ids[i] not in ids[:i])
    return dict(stop=stop, prefix=text[:10], eos_id=ids[index], eos_index=index)


def validate_boundary(result, kind, boundary, limit=48):
    if kind == 'stop':
        assert result['text'] == boundary['prefix'], result
        assert result['finish_reason'] == 'stop', result
    elif kind == 'alternate_eos':
        assert result['finish_reason'] == 'stop', result
        assert result['ids'][-1] == boundary['eos_id'], result
        assert len(result['ids']) == boundary['eos_index'] + 1, result
    elif kind == 'min_tokens':
        assert len(result['ids']) >= 16, result
        assert boundary['eos_id'] not in result['ids'][:15], result
    elif kind == 'ignore_eos':
        assert len(result['ids']) == limit and result['finish_reason'] == 'length', result
        assert boundary['eos_id'] in result['ids'], 'did not exercise ignored EOS'
    else:
        raise AssertionError(kind)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(source_commit=args.source_commit, kind='functional_mtp_boundaries_only',
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=[], comparisons=[])
    def save():
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180) as client:
        report['ready_before'] = (await client.get('/ready')).json()
        assert report['ready_before']['ready']
        async def execute(name, body, kind=None, boundary=None):
            row = dict(name=name, request=body, control=kind, boundary=boundary)
            report['rows'].append(row)
            try:
                response = await client.post('/v1/completions', json=body)
                row.update(status=response.status_code, body=parse_response(response, stream=body.get('stream',False)))
                assert response.status_code == 200, row['body']
                if body.get('stream'):
                    assert 'data: [DONE]' in response.text
                row['result'] = extract(row['body'], chat=False)
                validate_route(row['result'], mtp=True, logs=bool(body.get('logprobs')), full_mass=False)
                if kind:
                    validate_boundary(row['result'], kind, boundary)
                row['passed'] = True
            except Exception as error:
                row.update(passed=False,error=repr(error))
            save()
            print(name, 'PASS' if row['passed'] else 'FAIL '+row['error'][:300],flush=True)
            return row
        prompts = {'en':'Write a long list of invented names for tiny moon robots.\n1.',
                   'ja':'猫と月について、日本語で長い物語を書いてください。\nある夜、'}
        for language, prompt in prompts.items():
            for temperature in (0., .8):
                common = dict(model=args.model,prompt=prompt,max_tokens=48,temperature=temperature,
                              top_k=2,seed=104,logprobs=5)
                label=f'{language}-t{temperature}'
                baseline=await execute(label+'-baseline',common|dict(speculative=True))
                assert baseline['passed'], 'baseline failed; inspect report'
                boundary=choose_boundary(baseline['result'])
                tokenized=await client.post('/v1/hipengine/tokenize',json={'text':boundary['stop']})
                tokenized.raise_for_status()
                boundary['stop_token_ids']=tokenized.json()['token_ids']
                assert len(boundary['stop_token_ids'])>1, 'not a multi-token stop'
                for kind, controls in (
                    ('stop',dict(stop=boundary['stop'])),
                    ('alternate_eos',dict(eos_token_id=boundary['eos_id'])),
                    ('min_tokens',dict(eos_token_id=boundary['eos_id'],min_tokens=16)),
                    ('ignore_eos',dict(eos_token_id=boundary['eos_id'],ignore_eos=True)),
                ):
                    for spelling in ('speculative','speculative_mtp'):
                        for logs in ((True,False) if kind=='stop' else (True,)):
                            pair=[]
                            for stream in (False,True):
                                body=common|controls|{spelling:True,'stream':stream}
                                if not logs:body.pop('logprobs')
                                if stream:body['stream_options']=dict(include_usage=True,include_hipengine=True)
                                pair.append(await execute(f'{label}-{kind}-{spelling}-logs{int(logs)}-sse{int(stream)}',body,kind,boundary))
                            fields=('text','ids','tokens','finish','finish_reason','usage')
                            equal=all(r['passed'] for r in pair) and all(pair[0]['result'][k]==pair[1]['result'][k] for k in fields)
                            report['comparisons'].append(dict(left=pair[0]['name'],right=pair[1]['name'],passed=equal))
        report['ready_after']=(await client.get('/ready')).json()
        report['passed']=all(r['passed'] for r in report['rows']) and all(c['passed'] for c in report['comparisons']) and report['ready_after']['ready']
        save()
        print('BOUNDARY_GATE_COMPLETE',report['passed'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:18389')
    parser.add_argument('--model',default='qwen38fn-sampling')
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',required=True,type=Path)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args()))['passed'] else 1)
