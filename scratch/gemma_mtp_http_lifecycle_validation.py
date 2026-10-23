"""Historical HTTP cancellation/refill and prompt-list checks (2026-10-18).

Requires httpx and an already-running hipengine server at URL, with MODEL,
capacity 4, context 1024, BF16 KV and its discoverable MTP assistant. Edit URL
and MODEL for a different deployment. Writes raw responses under /tmp; do not
commit those responses. The original run is recorded in the
gemma-mtp-batching worklog entry.
"""
import concurrent.futures as cf
import json
import threading
import httpx

URL = 'http://127.0.0.1:18999/v1/completions'
MODEL = 'gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'
records = []

def complete(prompt, tokens=64, barrier=None):
    with httpx.Client(timeout=180) as client:
        if barrier:
            barrier.wait()
        r = client.post(URL, json=dict(model=MODEL, prompt=prompt, max_tokens=tokens, temperature=0, speculative_mtp=True))
        r.raise_for_status()
        result = r.json()
        records.append(result)
        return result

try:
    for width in (1, 2, 3, 4):
        value = complete(['Count to five:'] * width)
        widths = [v for c in value['choices'] for v in c['hipengine']['diagnostics']['physical_target_request_widths']]
        assert max(widths) == width, (width, widths)
        print('PASS HTTP prompt-list physical width', width, flush=True)

    baseline = complete('The capital of France is', 128)
    barrier = threading.Barrier(2)
    cancelled = threading.Event()
    def cancel_stream():
        with httpx.Client(timeout=180) as client:
            barrier.wait()
            with client.stream('POST', URL, json=dict(model=MODEL, prompt='Count to five:', max_tokens=512, temperature=0, stream=True, speculative_mtp=True)) as r:
                r.raise_for_status()
                seen = 0
                for line in r.iter_lines():
                    if not line.startswith('data: ') or line == 'data: [DONE]':
                        continue
                    chunk = json.loads(line[6:])
                    if any(c.get('text') for c in chunk.get('choices', [])):
                        seen += 1
                    if seen >= 3:
                        break
                assert seen >= 3
            cancelled.set()
    with cf.ThreadPoolExecutor(3) as pool:
        aborted = pool.submit(cancel_stream)
        survivor = pool.submit(complete, 'The capital of France is', 128, barrier)
        assert cancelled.wait(120)
        refill = pool.submit(complete, 'Hello world', 32)
        aborted.result()
        surviving = survivor.result()
        replacement = refill.result()
    assert surviving['choices'][0]['text'] == baseline['choices'][0]['text']
    assert replacement['choices'][0]['text'] == complete('Hello world', 32)['choices'][0]['text']
    print('PASS SSE disconnect, survivor equality, refill equality', flush=True)
finally:
    with open('/tmp/gemma-mtp-http-lifecycle.json', 'w') as f:
        json.dump(records, f)
