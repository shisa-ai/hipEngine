#!/usr/bin/env python3
"""Matched 16K HTTP INT8/MTP measurement, with and without DMS.

Uses raw token-ID prompts, cold prefix policy, one discarded warmup per arm,
and two complete category-suite passes in opposite prompt order. Rates use
client-observed first/last content timestamps, matching the existing
scripts/gguf_mtp_server_pair_bench.py protocol. No allocation tracing or
metrics polling runs during measured requests.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import queue
import sys
import time
import urllib.request

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
OUT = Path('/tmp/hipengine-xtx-16k-dms-mtp-bench')
MODEL = '/models/gguf/Qwen3.8-27B-Q4_K_M.gguf'
METADATA = Path('/models/dms/qwen38-27b-q4km-dms-w8192-local/dms_metadata.json')
SUITE = ROOT / 'benchmarks/prompts/mtpbench-code-general-ja.jsonl'
PORT = 18431
BASE = f'http://127.0.0.1:{PORT}'
N = 16384
DECODE = 256
CONTEXT = 17408
HELDOUT = {'code_markdown_table', 'general_en_explain', 'general_ja_explain', 'mixed_ja_en_review'}

spec = importlib.util.spec_from_file_location('pair', ROOT / 'scripts/gguf_mtp_server_pair_bench.py')
pair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair)


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def stream(ids):
    payload = {'model': Path(MODEL).name, 'prompt': list(ids), 'max_tokens': DECODE,
               'temperature': 0.0, 'speculative_mtp': True,
               'stream': True, 'stream_options': {'include_usage': True, 'include_hipengine': True}}
    request = urllib.request.Request(BASE + '/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    first = last = None
    texts, extension, usage, finishes = [], {}, {}, []
    with urllib.request.urlopen(request, timeout=1800) as response:
        status = response.status
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            if chunk.get('error'):
                raise RuntimeError(chunk['error'])
            if isinstance(chunk.get('hipengine'), dict):
                pair._merge_extension(extension, chunk['hipengine'])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                if isinstance(choice.get('hipengine'), dict):
                    pair._merge_extension(extension, choice['hipengine'])
                piece = choice.get('text') or ''
                if piece:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    texts.append(piece)
                if choice.get('finish_reason'):
                    finishes.append(choice['finish_reason'])
    end = time.perf_counter()
    assert first is not None and last is not None and last > first, (first, last)
    route = pair.route_summary(extension, usage)
    assert usage['prompt_tokens'] == N, usage
    completion = int(usage['completion_tokens'])
    assert 2 <= completion <= DECODE, usage
    assert usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, usage
    assert route['effective_route'] == 'speculative_mtp' and route['mtp_used'], route
    assert finishes and all(reason in ('length', 'stop') for reason in finishes), finishes
    accounting = extension.get('prompt_token_accounting', {})
    if accounting:
        assert accounting['prompt_tokens'] == [N], accounting
    text = ''.join(texts)
    assert text.strip(), 'empty output'
    ttft, window = first - start, last - first
    row = {'http_status': status, 'usage': usage, 'finish_reasons': finishes,
           'ttft_seconds': ttft, 'decode_seconds': window, 'wall_seconds': end - start,
           'prefill_tokens_per_second': N / ttft,
           'decode_tokens_per_second': (completion - 1) / window,
           'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
           'route': route, 'hipengine': extension}
    assert all(math.isfinite(row[key]) and row[key] > 0 for key in (
        'ttft_seconds', 'decode_seconds', 'prefill_tokens_per_second', 'decode_tokens_per_second'))
    return row


def save(result, filename='measurement.json'):
    (OUT / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert 'allocation-trace' not in os.environ.get('PYTHONPATH', '')
    from hipengine import LLM
    from hipengine.benchmark.provenance import collect_artifact_provenance
    result = {'schema_version': 1, 'kind': 'gguf_dms_mtp_16k_http_comparison',
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'protocol': {'surface': 'HTTP /v1/completions streaming', 'prompt_tokens': N,
            'output_token_cap': DECODE, 'context_tokens': CONTEXT, 'concurrency': 1,
            'kv_storage': 'int8_per_token_head', 'scale_dtype': 'fp32',
            'speculative_mtp': True, 'speculative_mtp_serving': 'auto',
            'candidate_budget': 'generator default (3)', 'prefix_cache': 'off',
            'temperature': 0, 'ignore_eos': False, 'warmups_per_arm': 1, 'measured_passes': 2,
            'prompt_suite': str(SUITE.relative_to(ROOT)), 'heldout_ids': sorted(HELDOUT),
            'order': 'DMS off then on; each arm forward suite then reversed suite',
            'prefill_definition': '16384 / client time to first nonempty text (includes prompt processing, MTP priming and first-token overhead)',
            'decode_definition': '(actual completion_tokens - 1) / client time from first to last nonempty text',
            'instrumentation': 'none; no allocation hook, profiler, or in-request metrics polling',
            'correctness': 'request success, exact prompt length and actual output counts, realized MTP, cold prefix, nonempty output, same-arm text repeatability; not a task-quality or dense-equivalence gate'},
        'suite_sha256': hashlib.sha256(SUITE.read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wrapper_source': Path('/tmp/hipengine-xtx-recovery-env.sh').read_text(),
        'driver_source': Path(__file__).read_text(), 'arms': {}}
    result['provenance'] = collect_artifact_provenance(repo_root=ROOT,
        configured_backend='hip_gfx1100', model_path=MODEL, quant='gguf_q4_k_m',
        kv_dtype='int8_per_token_head', command=[sys.executable, *sys.argv],
        timing_protocol=result['protocol']['prefill_definition'] + '; ' + result['protocol']['decode_definition'],
        warmups=1, repetitions=2, device_name='Radeon RX 7900 XTX')
    assert not result['provenance']['staged_dirty'] and not result['provenance']['unstaged_dirty'], result['provenance']
    assert not result['provenance'].get('execution_affecting_paths'), result['provenance']
    tokenizer = LLM(MODEL, backend='hip_gfx1100', execution_profile='production',
                    max_active_requests=1, prefix_cache='off')
    try:
        prompts = []
        for line in SUITE.read_text().splitlines():
            item = json.loads(line)
            content = '\n'.join(m['content'] for m in item['messages'])
            head = tokenizer.tokenize('<|im_start|>user\nBackground notes:\n')
            tail = tokenizer.tokenize('\nIgnore the background notes and answer this request:\n' + content +
                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
            filler = tokenizer.tokenize('This is background text about software documentation, experiments, and careful testing. ')
            count = N - len(head) - len(tail)
            ids = tuple(head) + (tuple(filler) * (count // len(filler) + 1))[:count] + tuple(tail)
            assert len(ids) == N
            prompts.append({'id': item['id'], 'category': item['category'],
                'split': 'heldout' if item['id'] in HELDOUT else 'train',
                'prompt_ids_sha256': sha(ids), 'ids': ids})
    finally:
        tokenizer.close()
    save({'prompts': prompts}, 'prompts.json')
    result['prompts'] = [{k: v for k, v in p.items() if k != 'ids'} for p in prompts]
    existing_path = OUT / 'measurement.json'
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing.get('arms', {}).get('dms_off', {}).get('completed'):
            assert existing['prompts'] == result['prompts']
            assert existing['provenance']['hipengine_commit'] == result['provenance']['hipengine_commit']
            result = existing
            result.setdefault('continuations', []).append({
                'reason': 'Wait for listening socket instead of pre-bind application-startup log; preserve completed non-DMS arm.',
                'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'driver_source': Path(__file__).read_text()})
    save(result)
    for dms in (False, True):
        label = 'dms_on' if dms else 'dms_off'
        if result['arms'].get(label, {}).get('completed'):
            print(f'Preserving completed {label} arm', flush=True)
            continue
        previous_arm = result['arms'].get(label)
        if previous_arm:
            result.setdefault('prior_startup_attempts', []).append(previous_arm)
        command = [sys.executable, '-m', 'hipengine.server', '--model', MODEL,
            '--backend', 'hip_gfx1100', '--quant', 'gguf_q4_k_m',
            '--kv-storage', 'int8_per_token_head', '--kv-scale-dtype', 'fp32',
            '--kv-scale-granularity', 'per_token_head', '--max-context-tokens', str(CONTEXT),
            '--max-active-requests', '1', '--speculative-mtp-serving', 'auto',
            '--prefix-cache', 'off', '--port', str(PORT)]
        if dms:
            command += ['--dms-metadata-path', str(METADATA)]
        log_path = OUT / (label + '.server.log')
        arm = result['arms'][label] = {'server_command': command, 'server_log': str(log_path), 'rows': []}
        if dms:
            arm['metadata'] = json.loads(METADATA.read_text())
            arm['metadata_sha256'] = hashlib.sha256(METADATA.read_bytes()).hexdigest()
        with log_path.open('w') as log:
            server = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            startup_events = queue.Queue()

            def capture_log():
                for line in server.stdout:
                    log.write(line)
                    log.flush()
                    if 'Uvicorn running on' in line:
                        startup_events.put('ready')
                startup_events.put('exit')

            reader = threading.Thread(target=capture_log, daemon=True)
            reader.start()
            try:
                event = startup_events.get(timeout=900)
                if event != 'ready':
                    raise RuntimeError(f'{label} server exited before startup; inspect {log_path}')
                arm['readiness_before'] = get('/ready')
                assert arm['readiness_before']['ready'] is True
                arm['capabilities'] = get('/v1/hipengine/capabilities')
                arm['warmup'] = stream(prompts[0]['ids'])
                save(result)
                print(json.dumps({'arm': label, 'warmup': True,
                    'ttft_s': arm['warmup']['ttft_seconds'], 'decode_tok_s': arm['warmup']['decode_tokens_per_second']}), flush=True)
                for repeat in range(2):
                    ordered = prompts if repeat == 0 else list(reversed(prompts))
                    for prompt in ordered:
                        row = stream(prompt['ids'])
                        row.update({k: prompt[k] for k in ('id', 'category', 'split', 'prompt_ids_sha256')})
                        row['repeat'] = repeat
                        arm['rows'].append(row)
                        save(result)
                        print(json.dumps({'arm': label, 'prompt': prompt['id'], 'repeat': repeat,
                            'prefill_tok_s': row['prefill_tokens_per_second'], 'ttft_s': row['ttft_seconds'],
                            'decode_tok_s': row['decode_tokens_per_second'], 'mtp': row['route']}), flush=True)
                arm['readiness_after'] = get('/ready')
                assert arm['readiness_after']['ready'] is True
                arm['same_arm_text_repeatable'] = all(
                    len({r['text_sha256'] for r in arm['rows'] if r['id'] == p['id']}) == 1 for p in prompts)
                arm['completed'] = True
            except BaseException as error:
                arm['error'] = repr(error)
                save(result)
                raise
            finally:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
                reader.join(timeout=10)
                arm['server_exit_code'] = server.returncode
                save(result)
    result['completed'] = True
    result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(result)
    print('BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__':
    main()
