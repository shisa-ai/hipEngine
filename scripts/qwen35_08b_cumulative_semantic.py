#!/usr/bin/env python3
"""Run the final Qwen3.5-0.8B current-vs-X2-control semantic packet."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.dtype import DType
from hipengine.loading.gguf import scan_gguf
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from scripts.gguf_eager_teacher_forced_oracle import _copy_device_fingerprint
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _compare_teacher_forced, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.gguf_mtp_category_bench import prompt_sha256

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == 'scripts' else Path('/home/lhl/hipEngine')
OUT = Path('/tmp/d08-review/cumulative-semantic.json')
COMPILER_VERSION_FILE = Path('/tmp/d08-c0/hipcc-version.txt')
MODELS = {
    'q4': Path('/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf'),
    'q8': Path('/models/gguf/Qwen3.5-0.8B-Q8_0.gguf'),
}
ROLE_ENVS: dict[str, dict[str, dict[str, str | None]]] = {
    'q4': {
        'strict_x2': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': '0',
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': '0',
            'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': 'exact',
        },
        'pre_x2': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': '0',
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'current_x3_rollback': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'current_x6_rollback': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'current_x8_rollback': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL': None,
            'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'current': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL': None,
            'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL': None,
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
    },
    'q8': {
        'strict_x2': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': '0',
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': '0',
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': '0',
            'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': 'exact',
        },
        'current_x8_rollback': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL': None,
            'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL': '0',
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
        'current': {
            'HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK': None,
            'HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL': None,
            'HIPENGINE_GGUF_DENSE_WMMA_BULK': None,
            'HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL': None,
            'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL': None,
            'HIPENGINE_GGUF_GDN_PREFILL_MODE': None,
        },
    },
}
DECODE_STEPS = 24
KL_THRESHOLD = 0.05
TOP1_THRESHOLD = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


@contextlib.contextmanager
def role_environment(quant: str, role: str) -> Iterator[None]:
    updates = {
        **ROLE_ENVS[quant][role],
        'HIPENGINE_GGUF_DECODE_REPACK': '1',
        'HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING': '0',
    }
    previous = {name: os.environ.get(name) for name in updates}
    for name, value in updates.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def create_session(quant: str, role: str) -> Qwen35GGUFResidentSession:
    with role_environment(quant, role):
        return Qwen35GGUFResidentSession(
            MODELS[quant],
            backend='hip_gfx1151',
            compiler_version=COMPILER_VERSION_FILE.read_text(encoding='utf-8'),
            require_cached_build=True,
            max_sequence_length=512 + DECODE_STEPS + 8,
            token_embedding_placement='device',
            use_wmma_prefill=True,
            use_gemv_decode=True,
            prefill_config=PrefillConfig(attn_aotriton_min_tokens=512),
        )


def expand_to_512(tokens: Sequence[int]) -> list[int]:
    values = [int(token) for token in tokens]
    if not values:
        raise ValueError('cannot expand an empty prompt')
    repeats = (512 + len(values) - 1) // len(values)
    return (values * repeats)[:512]


def trajectory_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(int(row['token_id']).to_bytes(8, 'little', signed=True))
        digest.update(np.ascontiguousarray(row['logits'], dtype='<f4').tobytes())
    return digest.hexdigest()


def compact_comparison(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = _compare_teacher_forced(
        baseline,
        candidate,
        kl_threshold=KL_THRESHOLD,
        top1_threshold=TOP1_THRESHOLD,
    )
    return {key: value for key, value in result.items() if key != 'transitions'}


def capture_prefill_state(session: Qwen35GGUFResidentSession) -> dict[str, Any]:
    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError('session is closed')
    session.runtime.device_synchronize()
    scratch = session.scratch
    components: list[tuple[str, str, bool]] = []
    for layer, (conv_state, recurrent_state) in enumerate(
        zip(scratch.layer_conv_states, scratch.layer_recurrent_states, strict=True)
    ):
        if conv_state is None or recurrent_state is None:
            continue
        for part, buffer in (('conv', conv_state), ('recurrent', recurrent_state)):
            fp = _copy_device_fingerprint(
                session, ptr=int(buffer.ptr), nbytes=int(buffer.nbytes), dtype='fp32'
            )
            components.append((f'linear.{layer}.{part}', str(fp['blake2b_128']), bool(fp['finite'])))
    cfg = session.runner.weights.config
    live_positions = int(session.position)
    row_nbytes = int(cfg.head_count_kv) * int(cfg.key_length) * DType.BF16.itemsize
    live_nbytes = live_positions * row_nbytes
    for layer, (key_cache, value_cache) in enumerate(
        zip(scratch.full_key_caches, scratch.full_value_caches, strict=True)
    ):
        if key_cache is None or value_cache is None:
            continue
        checked = min(live_nbytes, int(key_cache.nbytes), int(value_cache.nbytes))
        for part, buffer in (('key', key_cache), ('value', value_cache)):
            fp = _copy_device_fingerprint(
                session, ptr=int(buffer.ptr), nbytes=checked, dtype='bf16'
            )
            components.append((f'kv.{layer}.{part}', str(fp['blake2b_128']), bool(fp['finite'])))
    digest = hashlib.sha256(
        json.dumps([(name, value) for name, value, _ in components], separators=(',', ':')).encode()
    ).hexdigest()
    return {
        'position': live_positions,
        'component_count': len(components),
        'digest': digest,
        'finite': all(finite for _, _, finite in components),
    }


def run_free(
    session: Qwen35GGUFResidentSession, *, prompt_ids: Sequence[int]
) -> dict[str, Any]:
    session.reset()
    first = session.prefill(
        [int(token) for token in prompt_ids],
        use_bulk=True,
        bulk_attention_mode='bulk',
        return_logits=True,
        capture_hidden_seed_fp32=False,
    )
    state = capture_prefill_state(session)
    rows = [{'token_id': int(first.token_id), 'logits': np.ascontiguousarray(first.logits, dtype=np.float32)}]
    current = int(first.token_id)
    for _ in range(DECODE_STEPS):
        result = session.step(current, return_logits=True)
        current = int(result.token_id)
        rows.append({'token_id': current, 'logits': np.ascontiguousarray(result.logits, dtype=np.float32)})
    return {'trajectory': rows, 'digest': trajectory_digest(rows), 'state': state}


def run_teacher_forced(
    session: Qwen35GGUFResidentSession,
    *,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
) -> dict[str, Any]:
    session.reset()
    first = session.prefill(
        [int(token) for token in prompt_ids],
        use_bulk=True,
        bulk_attention_mode='bulk',
        return_logits=True,
        capture_hidden_seed_fp32=False,
    )
    state = capture_prefill_state(session)
    rows = [{'token_id': int(first.token_id), 'logits': np.ascontiguousarray(first.logits, dtype=np.float32)}]
    for token_id in forced_input_ids:
        result = session.step(int(token_id), return_logits=True)
        rows.append({'token_id': int(result.token_id), 'logits': np.ascontiguousarray(result.logits, dtype=np.float32)})
    return {'trajectory': rows, 'digest': trajectory_digest(rows), 'state': state}


def run_recorded_graph(
    session: Qwen35GGUFResidentSession, *, prompt_ids: Sequence[int]
) -> dict[str, Any]:
    session.reset()
    first = session.prefill(
        [int(token) for token in prompt_ids],
        use_bulk=True,
        bulk_attention_mode='bulk',
        return_logits=True,
        capture_hidden_seed_fp32=False,
    )
    graph = session.capture_decode_graph(
        position=int(session.position),
        steps_per_replay=1,
        max_replay_steps=DECODE_STEPS,
        record_steps=DECODE_STEPS,
        input_token_id=int(first.token_id),
    )
    try:
        nodes = len(session.runtime.graph_nodes(graph.graph))
        graph.replay(DECODE_STEPS)
        tokens = [int(first.token_id), *graph.read_generated_token_ids(DECODE_STEPS)]
        final = graph.read_sample(return_logits=True)
        transport = graph.transport_provenance()
    finally:
        graph.close()
    return {
        'token_ids': tokens,
        'final_logits': np.ascontiguousarray(final.logits, dtype=np.float32),
        'nodes': nodes,
        'transport': {
            key: value
            for key, value in transport.items()
            if key not in {'graph_handle', 'graph_exec', 'closed'}
        },
    }


def role_summary(
    *,
    strict: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    free: Mapping[str, Any] | None,
    graph: Mapping[str, Any] | None,
) -> dict[str, Any]:
    comparison = compact_comparison(strict['trajectory'], first['trajectory'])
    result: dict[str, Any] = {
        'correctness': comparison,
        'teacher_forced_digest': first['digest'],
        'teacher_forced_deterministic': first['digest'] == second['digest'],
        'state_digest': first['state']['digest'],
        'state_deterministic': first['state']['digest'] == second['state']['digest'],
        'state_finite': bool(first['state']['finite'] and second['state']['finite']),
        'state_exact_vs_strict_diagnostic': first['state']['digest'] == strict['state']['digest'],
        'state_component_count': int(first['state']['component_count']),
        'free_running_exact_vs_strict_diagnostic': None,
        'graph': None,
    }
    if free is not None:
        result['free_running_exact_vs_strict_diagnostic'] = [
            int(row['token_id']) for row in free['trajectory']
        ] == [int(row['token_id']) for row in strict['trajectory']]
    if graph is not None and free is not None:
        eager_ids = [int(row['token_id']) for row in free['trajectory']]
        quality = evaluate_logits(
            np.asarray(free['trajectory'][-1]['logits'], dtype=np.float32),
            graph['final_logits'],
            kl_threshold=KL_THRESHOLD,
            top1_threshold=TOP1_THRESHOLD,
        )
        result['graph'] = {
            'trajectory_exact_vs_eager': graph['token_ids'] == eager_ids,
            'final_kl_max_vs_eager': float(quality.kl_max),
            'final_top1_vs_eager': float(quality.top1_agreement),
            'nodes': int(graph['nodes']),
            'transport': graph['transport'],
        }
    return result


def aggregate(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    selected = [row['roles'][role] for row in rows]
    correctness = [row['correctness'] for row in selected]
    total = sum(int(row['transitions_total']) for row in correctness)
    hits = sum(int(row['top1_matches']) for row in correctness)
    graph_rows = [row['graph'] for row in selected if row['graph'] is not None]
    return {
        'prompts': len(selected),
        'transitions': total,
        'top1_matches': hits,
        'top1_agreement': hits / total,
        'kl_max': max(float(row['kl_max']) for row in correctness),
        'passed': all(bool(row['passed']) for row in correctness),
        'teacher_forced_deterministic': all(bool(row['teacher_forced_deterministic']) for row in selected),
        'state_deterministic': all(bool(row['state_deterministic']) for row in selected),
        'state_finite': all(bool(row['state_finite']) for row in selected),
        'state_exact_vs_strict_prompts': sum(bool(row['state_exact_vs_strict_diagnostic']) for row in selected),
        'free_running_exact_vs_strict_prompts': sum(row['free_running_exact_vs_strict_diagnostic'] is True for row in selected),
        'graph_prompts': len(graph_rows),
        'graph_trajectory_exact': all(bool(row['trajectory_exact_vs_eager']) for row in graph_rows),
        'graph_kl_max': max((float(row['final_kl_max_vs_eager']) for row in graph_rows), default=0.0),
        'graph_top1_min': min((float(row['final_top1_vs_eager']) for row in graph_rows), default=1.0),
        'graph_node_sets': sorted({int(row['nodes']) for row in graph_rows}),
    }


def run_quant(quant: str, prompt_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(MODELS[quant]))
    base_tokens = {
        str(row['id']): build_chat_prompt(tokenizer, str(row['prompt'])) for row in prompt_rows
    }
    profiles = {
        'natural': base_tokens,
        'category_p512': {prompt_id: expand_to_512(tokens) for prompt_id, tokens in base_tokens.items()},
    }
    sessions = {role: create_session(quant, role) for role in ROLE_ENVS[quant]}
    records: dict[str, list[dict[str, Any]]] = {profile: [] for profile in profiles}
    try:
        for profile, prompts in profiles.items():
            for index, row in enumerate(prompt_rows):
                prompt_id = str(row['id'])
                tokens = prompts[prompt_id]
                with role_environment(quant, 'strict_x2'):
                    strict_first = run_free(sessions['strict_x2'], prompt_ids=tokens)
                    strict_second = run_free(sessions['strict_x2'], prompt_ids=tokens)
                    strict_graph = (
                        run_recorded_graph(sessions['strict_x2'], prompt_ids=tokens)
                        if profile == 'category_p512'
                        else None
                    )
                forced = [int(step['token_id']) for step in strict_first['trajectory'][:-1]]
                roles: dict[str, Any] = {}
                roles['strict_x2'] = role_summary(
                    strict=strict_first,
                    first=strict_first,
                    second=strict_second,
                    free=strict_first,
                    graph=strict_graph,
                )
                for role in ROLE_ENVS[quant]:
                    if role == 'strict_x2':
                        continue
                    with role_environment(quant, role):
                        first = run_teacher_forced(
                            sessions[role], prompt_ids=tokens, forced_input_ids=forced
                        )
                        second = run_teacher_forced(
                            sessions[role], prompt_ids=tokens, forced_input_ids=forced
                        )
                        free = (
                            run_free(sessions[role], prompt_ids=tokens)
                            if role == 'current'
                            else None
                        )
                        graph = (
                            run_recorded_graph(sessions[role], prompt_ids=tokens)
                            if role == 'current' and profile == 'category_p512'
                            else None
                        )
                    roles[role] = role_summary(
                        strict=strict_first,
                        first=first,
                        second=second,
                        free=free,
                        graph=graph,
                    )
                records[profile].append(
                    {
                        'id': prompt_id,
                        'category': str(row['category']),
                        'suite': str(row['suite']),
                        'prompt_sha256': prompt_sha256(str(row['prompt'])),
                        'natural_tokens': len(base_tokens[prompt_id]),
                        'profile_tokens': len(tokens),
                        'profile_token_sha256_i64': hashlib.sha256(
                            np.asarray(tokens, dtype='<i8').tobytes()
                        ).hexdigest(),
                        'roles': roles,
                    }
                )
                print(quant, profile, f'{index + 1}/{len(prompt_rows)}', prompt_id, {
                    role: (round(value['correctness']['top1_agreement'], 4), round(value['correctness']['kl_max'], 6))
                    for role, value in roles.items()
                }, flush=True)
    finally:
        for session in sessions.values():
            session.close()
    summaries = {
        profile: {role: aggregate(rows, role) for role in ROLE_ENVS[quant]}
        for profile, rows in records.items()
    }
    categories: dict[str, Any] = {}
    for profile, rows in records.items():
        categories[profile] = {}
        for category in sorted({str(row['category']) for row in rows}):
            category_rows = [row for row in rows if row['category'] == category]
            categories[profile][category] = {
                role: aggregate(category_rows, role) for role in ROLE_ENVS[quant]
            }
    return {
        'model': str(MODELS[quant]),
        'model_sha256': sha256(MODELS[quant]),
        'role_env': ROLE_ENVS[quant],
        'summary': summaries,
        'categories': categories,
        'prompts': records,
    }


def main() -> int:
    prompt_rows = _load_suites(DEFAULT_PROMPTS)
    results = {quant: run_quant(quant, prompt_rows) for quant in ('q4', 'q8')}
    all_summaries = [
        summary
        for quant in results.values()
        for profile in quant['summary'].values()
        for summary in profile.values()
    ]
    gate_passed = all(
        bool(row['passed'])
        and bool(row['teacher_forced_deterministic'])
        and bool(row['state_deterministic'])
        and bool(row['state_finite'])
        and bool(row['graph_trajectory_exact'])
        and float(row['graph_kl_max']) <= KL_THRESHOLD
        and float(row['graph_top1_min']) >= TOP1_THRESHOLD
        for row in all_summaries
    )
    payload = {
        'schema': 1,
        'date': '2026-08-15',
        'status': 'passed' if gate_passed else 'failed',
        'task': 'D08 post-X6 cumulative natural and category-derived p512 semantic packet',
        'gate_passed': gate_passed,
        'repo': {'head': git('rev-parse', 'HEAD'), 'status_porcelain': git('status', '--porcelain=v1')},
        'hardware': {'host': os.uname().nodename, 'gpu': 'AMD Radeon 8060S Graphics', 'arch': 'gfx1151'},
        'protocol': {
            'command': 'HIPENGINE_HIP_ARCH=gfx1151 uv run python scripts/qwen35_08b_cumulative_semantic.py',
            'script': str(Path(__file__).resolve()),
            'script_sha256': sha256(Path(__file__).resolve()),
            'compiler_version_file': str(COMPILER_VERSION_FILE),
            'compiler_version_sha256': sha256(COMPILER_VERSION_FILE),
            'prompt_suites': [str(path.resolve()) for path in DEFAULT_PROMPTS],
            'prompt_suite_sha256': {str(path.resolve()): sha256(path) for path in DEFAULT_PROMPTS},
            'prompt_count': len(prompt_rows),
            'profiles': {
                'natural': 'unaltered build_chat_prompt token IDs',
                'category_p512': 'repeat each complete category-derived chat-token sequence and truncate to exactly 512 IDs',
            },
            'decode_steps': DECODE_STEPS,
            'same_context_rule': 'candidate teacher-forced decode consumes strict-X2 free-running token prefixes',
            'kl_threshold': KL_THRESHOLD,
            'top1_threshold': TOP1_THRESHOLD,
            'state_contract': 'prefill Conv/GDN and live full-attention KV fingerprints; exactness versus strict is diagnostic, within-role repeat determinism and finiteness are binding',
            'graph_contract': 'recorded public graph free-running trajectory and final logits must match same-role eager on all category_p512 prompts',
        },
        'results': results,
        'decision': 'Cumulative semantic packet passes.' if gate_passed else 'Cumulative semantic packet blocks further optimization.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({q: row['summary'] for q, row in results.items()}, indent=2))
    print(OUT)
    return 0 if gate_passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
