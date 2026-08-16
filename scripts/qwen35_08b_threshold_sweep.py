from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path('/home/lhl/hipEngine')
OUT_DIR = Path('/tmp/d08-review/threshold-fresh')
OUT = Path('/tmp/d08-review/prompt-threshold-sweep-raw.json')
LENGTHS = (16, 32, 64, 128, 256, 511, 512, 513, 768, 1024, 4096)


def workload_label(length: int) -> str:
    return f'{length // 1024}K/1' if length >= 1024 and length % 1024 == 0 else f'{length}/1'
MODELS = {
    'q4': ('/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf', 'gguf_q4_k_m'),
    'q8': ('/models/gguf/Qwen3.5-0.8B-Q8_0.gguf', 'gguf_q8_0'),
}
ROLE_ENVS: dict[str, dict[str, dict[str, str | None]]] = {
    'q4': {
        'current': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'pre_x2': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'strict_x2': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': 'exact',
        },
    },
    'q8': {
        'current': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'strict_x2': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': 'exact',
        },
    },
}
BLOCKS = {'q4': 3, 'q8': 4}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def run_one(
    quant: str, role: str, block: int, order_index: int, length: int
) -> dict[str, Any]:
    model, quant_label = MODELS[quant]
    artifact = OUT_DIR / f'{quant}-b{block:02d}-{order_index}-{role}-p{length}.json'
    env = os.environ.copy()
    env['HIPENGINE_HIP_ARCH'] = 'gfx1151'
    env['HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING'] = '0'
    for name, value in ROLE_ENVS[quant][role].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    command = [
        sys.executable, str(ROOT / 'scripts/qwen35_readme_sweep.py'),
        '--engine', 'gguf', '--model', model, '--quant', quant_label,
        '--workloads', f'{length}/1',
        '--token-id', '9707', '--warmup-runs', '1', '--measured-runs', '1',
        '--warmup-decode-tokens', '1', '--backend', 'hip_gfx1151',
        '--compiler-version-file', '/tmp/d08-c0/hipcc-version.txt', '--require-cached-build',
        '--attn-aotriton-min-tokens', '512', '--force-bulk-prefill',
        '--use-wmma-prefill', '--use-gemv-decode', '--no-graph-replay-decode',
        '--json', str(artifact),
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    artifact.with_suffix('.stdout.log').write_text(proc.stdout, encoding='utf-8')
    artifact.with_suffix('.stderr.log').write_text(proc.stderr, encoding='utf-8')
    if proc.returncode:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f'failed {quant=} {role=} {block=} {length=}')
    payload = json.loads(artifact.read_text(encoding='utf-8'))
    label = workload_label(length)
    summary = payload['summary_by_workload'][label]
    measured = [row for row in payload['runs_by_workload'][label] if row['measured']]
    assert len(measured) == 1
    run = measured[0]
    return {
        'artifact': str(artifact),
        'command': command,
        'elapsed_s': time.perf_counter() - started,
        'source_commit': payload['provenance']['hipengine_commit'],
        'clean_tree': not bool(payload['provenance']['dirty']),
        'row': {
            'prefill_tok_s': float(summary['prefill_tok_s']['median']),
            'prefill_ms': float(run['timings']['prefill_seconds']) * 1000.0,
            'finite': bool(run['correctness_sanity']['finite_final_logits']),
            'final_token_id': int(run['correctness_sanity']['final_token_id']),
            'prefill_chunk_sizes': run['prefill_chunk_sizes'],
            'scratch_max_positions': int(run['memory_snapshots']['after_prefill']['scratch_max_positions']),
        },
    }


def run_role(quant: str, role: str, block: int, order_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    executions = {
        str(length): run_one(quant, role, block, order_index, length) for length in LENGTHS
    }
    return {
        'role': role,
        'order_index': order_index,
        'artifacts': [executions[str(length)]['artifact'] for length in LENGTHS],
        'env': ROLE_ENVS[quant][role],
        'commands': [executions[str(length)]['command'] for length in LENGTHS],
        'elapsed_s': time.perf_counter() - started,
        'source_commits': sorted({executions[str(length)]['source_commit'] for length in LENGTHS}),
        'all_clean_tree': all(executions[str(length)]['clean_tree'] for length in LENGTHS),
        'rows': {str(length): executions[str(length)]['row'] for length in LENGTHS},
    }


def summarize(blocks: list[dict[str, Any]], roles: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {'roles': {}, 'comparisons_vs_current': {}}
    by_role = {role: [next(x for x in block['execution'] if x['role'] == role) for block in blocks] for role in roles}
    for role, executions in by_role.items():
        role_rows = {}
        for length in LENGTHS:
            samples = [float(x['rows'][str(length)]['prefill_tok_s']) for x in executions]
            milliseconds = [float(x['rows'][str(length)]['prefill_ms']) for x in executions]
            role_rows[str(length)] = {
                'prefill_tok_s_samples': samples,
                'prefill_tok_s_median': float(statistics.median(samples)),
                'prefill_ms_samples': milliseconds,
                'prefill_ms_median': float(statistics.median(milliseconds)),
                'all_finite': all(bool(x['rows'][str(length)]['finite']) for x in executions),
                'final_token_ids': sorted({int(x['rows'][str(length)]['final_token_id']) for x in executions}),
                'prefill_chunk_sizes': executions[0]['rows'][str(length)]['prefill_chunk_sizes'],
                'scratch_max_positions': sorted({int(x['rows'][str(length)]['scratch_max_positions']) for x in executions}),
            }
        out['roles'][role] = role_rows
    current = by_role['current']
    for control_role in roles:
        if control_role == 'current':
            continue
        controls = by_role[control_role]
        comparison = {}
        for length in LENGTHS:
            cand = [float(x['rows'][str(length)]['prefill_tok_s']) for x in current]
            ctrl = [float(x['rows'][str(length)]['prefill_tok_s']) for x in controls]
            cmed, bmed = float(statistics.median(cand)), float(statistics.median(ctrl))
            comparison[str(length)] = {
                'current_median': cmed,
                'control_median': bmed,
                'ratio': cmed / bmed,
                'paired_wins': sum(a > b for a, b in zip(cand, ctrl, strict=True)),
                'blocks': len(cand),
            }
        out['comparisons_vs_current'][control_role] = comparison
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for quant in ('q4', 'q8'):
        roles = list(ROLE_ENVS[quant])
        blocks = []
        for block in range(BLOCKS[quant]):
            shift = block % len(roles)
            order = roles[shift:] + roles[:shift]
            execution = []
            for order_index, role in enumerate(order):
                row = run_role(quant, role, block, order_index)
                execution.append(row)
                values = row['rows']
                print(quant, block, role, 'p256', round(values['256']['prefill_tok_s'], 1), 'p512', round(values['512']['prefill_tok_s'], 1), 'p4k', round(values['4096']['prefill_tok_s'], 1), flush=True)
            blocks.append({'block': block, 'order': order, 'execution': execution})
        result[quant] = {
            'model': MODELS[quant][0],
            'model_sha256': sha256(Path(MODELS[quant][0])),
            'blocks': blocks,
            'summary': summarize(blocks, roles),
        }
    payload = {
        'schema': 1,
        'task': 'D08 post-review p16-p4096 current/default threshold sweep',
        'repo': {'head': git('rev-parse', 'HEAD'), 'status_porcelain': git('status', '--porcelain=v1')},
        'harness': str(Path(__file__).resolve()), 'harness_sha256': sha256(Path(__file__).resolve()),
        'driver': 'scripts/qwen35_readme_sweep.py', 'driver_sha256': sha256(ROOT / 'scripts/qwen35_readme_sweep.py'),
        'lengths': list(LENGTHS), 'decode_tokens': 1, 'blocks': BLOCKS,
        'process_isolation': 'one fresh resident-session process per prompt length and block; avoids cross-shape AOTriton lifecycle contamination while preserving current production routes',
        'role_env': ROLE_ENVS,
        'quant': result,
    }
    OUT.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    for quant, row in result.items():
        print('SUMMARY', quant)
        for role, values in row['summary']['comparisons_vs_current'].items():
            print(role, {length: round(values[str(length)]['ratio'], 4) for length in LENGTHS})
    print(OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
