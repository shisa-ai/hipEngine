"""Historical Gemma MTP HTTP batching checks (2026-10-18).

Requires httpx and an already-running hipengine server at URL, with MODEL,
capacity 4, context 1024, BF16 KV and its discoverable MTP assistant. Edit URL
and MODEL for a different deployment. Writes raw responses under /tmp; do not
commit those responses. Pass --sse-only to check streaming only.
The original run is recorded in the gemma-mtp-batching worklog entry.
"""
import concurrent.futures as cf
import json
import threading
import sys
import httpx

URL = 'http://127.0.0.1:18999/v1/completions'
MODEL = 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'
records = []
def call(index, *, stream=False, explicit=None, sampled=False, barrier=None):
    body = dict(model=MODEL, prompt=['Count to five:', 'def add(a, b):', 'Hello world', 'The capital of France is'][index % 4], max_tokens=128, temperature=0.7 if sampled else 0, seed=123)
    if sampled:
        body.update(top_p=0.9, repetition_penalty=1.1)
    if explicit is not None:
        body['speculative_mtp'] = explicit
    with httpx.Client(timeout=120) as client:
        if barrier:
            barrier.wait()
        if not stream:
            response = client.post(URL, json=body)
            response.raise_for_status()
            value = response.json()
            chunks = [value]
            text = value['choices'][0]['text']
        else:
            body['stream'] = True
            body['stream_options'] = {'include_hipengine': True}
            chunks = []
            with client.stream('POST', URL, json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        chunks.append(json.loads(line[6:]))
            text = ''.join(c.get('text', '') for chunk in chunks for c in chunk.get('choices', []))
    diagnostics = [c['hipengine']['diagnostics'] for chunk in chunks for c in chunk.get('choices', []) if c.get('hipengine', {}).get('diagnostics')]
    record = dict(index=index, body=body, text=text, diagnostics=diagnostics, chunks=chunks)
    records.append(record)
    return record
try:
    for stream in ((True,) if '--sse-only' in sys.argv else (False, True)):
        for explicit in (None, True):
            for width in (1, 2, 3, 4):
                barrier = threading.Barrier(width)
                with cf.ThreadPoolExecutor(width) as pool:
                    rows = list(pool.map(lambda i: call(i, stream=stream, explicit=explicit, barrier=barrier), range(width)))
                widths = [v for row in rows for d in row['diagnostics'] for v in d['physical_target_request_widths']]
                # Asynchronous arrivals and variable acceptance need not reach
                # the full live-request width in the same proposal quantum.
                assert widths and max(widths) <= width, (stream, explicit, width, widths)
                assert width == 1 or max(widths) > 1, (stream, explicit, width, widths)
                print('PASS concurrency', stream, explicit, width, 'physical peak', max(widths), flush=True)
    alone = call(0, sampled=True)
    barrier = threading.Barrier(4)
    with cf.ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(lambda i: call(i, sampled=True, barrier=barrier), range(4)))
    assert rows[0]['text'] == alone['text']
    print('PASS sampled isolation', flush=True)
    ar = call(0, explicit=False)
    mtp = call(0, explicit=True)
    assert ar['text'] == mtp['text'], (ar['text'], mtp['text'])
    print('PASS greedy AR equality', flush=True)
finally:
    with open('/tmp/gemma-mtp-http-batching.json', 'w') as f:
        json.dump(records, f)
