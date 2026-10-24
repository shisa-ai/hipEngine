#!/usr/bin/env python3
"""Pair full historical prompt priming with cache-only priming; same MTP route."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from contextlib import closing
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from tokenizers import Tokenizer
from hipengine.speculative.mtp_native import NativeMtpChainProposer

CACHE_ONLY_APPEND = NativeMtpChainProposer.append_prompt_kv


def full_append(self, *, input_token, target_hidden_ptr, position, stream=0):
    self.advance_with_target_hidden(
        input_token=input_token, target_hidden_ptr=target_hidden_ptr, position=position,
        need_result=False, read_token_id=False, read_expert_topk=False,
        read_lm_head_value=False, stream=stream,
    )


def find_runner(value):
    if isinstance(value, dict):
        if 'routes' in value and 'recent_completed' in value['routes']:
            return value
        for child in value.values():
            result = find_runner(child)
            if result is not None:
                return result
    return None


def token_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--cases', choices=('all', 'smoke'), default='all')
    args = parser.parse_args()
    root = Path.cwd()
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    cases = []
    sources = []
    for split, name in [('category', 'mtpbench-code-general-ja.jsonl'), ('heldout', 'gdn-prefill-category-heldouts.jsonl')]:
        path = root / 'benchmarks/prompts' / name
        sources.append({'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            item = json.loads(line)
            prompt = '<|im_start|>user\n' + item['messages'][0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'
            cases.append({'id': item['id'], 'category': item['category'], 'split': split, 'tokens': tokenizer.encode(prompt).ids})
    for length in (512, 1024, 4096):
        cases.append({'id': f'p{length}', 'category': 'length_control', 'split': 'control', 'tokens': [9707] * length})
    if args.cases == 'smoke':
        cases = [cases[0], cases[-3]]
    result = {'schema': 1, 'kind': 'public_paro_cache_only_priming_comparison', 'protocol': {
        'backend': 'hip_gfx1151', 'quant': 'w4_paro', 'kv_storage': 'bf16', 'max_active_requests': 1,
        'max_sequence_length': 4352, 'prefix_cache': 'off', 'temperature': 0, 'ignore_eos': True,
        'output_tokens': 128, 'warmups_per_case_per_arm': args.warmups, 'samples_per_case_per_arm': args.repetitions,
        'order': 'Full then cache-only on even pairs; cache-only then full on odd pairs',
        'surface': 'LLM.stream_speculative_mtp_detailed for both arms',
        'baseline': 'Current repaired target, historical full draft step for each non-final prompt row. Test-only method substitution, not a shipping flag.',
        'candidate': 'Default append_prompt_kv cache-only non-final rows; same final prediction and MTP cycle.',
        'decode_timing': 'Observed first-token to last-token wall; denominator is generated tokens minus first observed cumulative count. Terminal duplicate chunks excluded.',
        'first_token_timing': 'Public call start to first token, including prefill and MTP prompt priming.',
        'whole_call_timing': 'Public call start through iterator exhaustion; model construction and snapshot extraction excluded after warmup.',
        'profile': 'Shipping backend defaults; no gfx1151 PARO execution-profile plan resolves. No arithmetic or variant override.',
        'cross_arm_id_equality': 'Required: full and cache-only prompt priming must preserve all output IDs and acceptance decisions.',
        'sources': sources,
    }, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': [], 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    completed_ids = []
    original_reclaim = Qwen35ParoResidentModelRunner.reclaim

    def observe_reclaim(owner, completed):
        original_reclaim(owner, completed)
        completed_ids.append(tuple(int(token) for token in completed.generated_tokens))

    # The AR stream need not expose final IDs. Observe scheduler-owned IDs at
    # reclaim, as the existing PARO live-server harness does; do not replace
    # generation, sampling, verification, or commitment.
    Qwen35ParoResidentModelRunner.reclaim = observe_reclaim
    try:
        with closing(LLM(str(args.model), backend='hip_gfx1151', quant='w4_paro', max_active_requests=1,
                         max_sequence_length=4352, prefix_cache='off')) as llm:
            for case in cases:
                record = {k: v for k, v in case.items() if k != 'tokens'}
                record.update(prompt_tokens=len(case['tokens']), prompt_ids_sha256=token_hash(case['tokens']), samples=[])
                result['cases'].append(record)
                references = {}
                for pair in range(args.warmups + args.repetitions):
                    for cache_only in ((False, True) if pair % 2 == 0 else (True, False)):
                        arm = 'cache_only' if cache_only else 'full'
                        NativeMtpChainProposer.append_prompt_kv = CACHE_ONLY_APPEND if cache_only else full_append
                        params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, speculative_mtp=True, kv_storage='bf16')
                        events = []
                        ids = None
                        text = []
                        previous_count = 0
                        completed_ids.clear()
                        started = time.perf_counter()
                        stream = llm.stream_speculative_mtp_detailed([case['tokens']], params)
                        for chunk in stream:
                            observed = time.perf_counter()
                            text.append(chunk.text)
                            if chunk.generated_token_ids is not None:
                                ids = tuple(chunk.generated_token_ids)
                            if chunk.telemetry is not None:
                                count = int(chunk.telemetry.decode_state.generated_tokens)
                                assert count >= previous_count, (case['id'], count, previous_count)
                                if count > previous_count:
                                    events.append({'tokens': count, 'seconds': observed - started})
                                    previous_count = count
                        elapsed = time.perf_counter() - started
                        assert len(completed_ids) == 1, (case['id'], arm, completed_ids)
                        if ids is not None:
                            assert ids == completed_ids[0]
                        ids = completed_ids[0]
                        assert len(ids) == 128, (case['id'], arm, ids)
                        assert events[0]['tokens'] == 1 and events[-1]['tokens'] == 128, events
                        assert all(0 <= token < tokenizer.get_vocab_size(with_added_tokens=True) for token in ids)
                        runner = find_runner(llm.live_loop_snapshot())
                        assert runner is not None
                        route = dict(runner['routes']['recent_completed'][-1])
                        route.pop('scheduler_chunks', None)
                        assert route['specdec2_mtp2_used'], route
                        assert runner['model_runner']['active_requests'] == 0
                        assert route['specdec2_mtp2_cycles'] > 0
                        assert set(route['specdec2_mtp2_candidate_counts']) == {1}
                        assert set(route['specdec2_mtp2_execution_routes']) == {'eager'}
                        fallback = runner['routes']['fallback_reasons']
                        assert not any(k.startswith('specdec2_mtp2_recovered') for k in fallback), fallback
                        if arm not in references:
                            references[arm] = ids
                            record[arm + '_ids'] = list(ids)
                            record[arm + '_text'] = ''.join(text)
                        assert ids == references[arm], (case['id'], arm, 'repeat mismatch')
                        if len(references) == 2:
                            assert references['full'] == references['cache_only'], (case['id'], 'cross-arm mismatch')
                        acceptance = route['specdec2_mtp2_accepted_counts']
                        if record['samples']:
                            assert acceptance == record['samples'][0]['route']['specdec2_mtp2_accepted_counts'], (case['id'], 'accept mismatch')
                        decode_seconds = events[-1]['seconds'] - events[0]['seconds']
                        sample = {'pair': pair, 'warmup': pair < args.warmups, 'arm': arm,
                                  'generated_tokens': len(ids), 'generated_ids_sha256': token_hash(ids),
                                  'first_token_seconds': events[0]['seconds'], 'decode_seconds': decode_seconds,
                                  'decode_tokens': 127, 'decode_tokens_per_second': 127 / decode_seconds,
                                  'whole_call_seconds': elapsed, 'whole_call_tokens_per_second': len(ids) / elapsed,
                                  'token_events': events, 'route': route, 'fallback_reasons': fallback}
                        record['samples'].append(sample)
                    save()
                record['full_cache_only_ids_equal'] = references['full'] == references['cache_only']
                record['medians'] = {arm: {key: statistics.median(s[key] for s in record['samples'] if s['arm'] == arm and not s['warmup'])
                                          for key in ('decode_tokens_per_second', 'first_token_seconds', 'whole_call_tokens_per_second')}
                                     for arm in ('full', 'cache_only')}
                print(json.dumps({'case': case['id'], 'medians': record['medians'], 'full_cache_only_ids_equal': record['full_cache_only_ids_equal']}), flush=True)
                save()
        result['memory_after_close'] = memory_stats()
        assert result['memory_after_close']['active_allocations'] == 0
        result['passed'] = True
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    finally:
        Qwen35ParoResidentModelRunner.reclaim = original_reclaim
        NativeMtpChainProposer.append_prompt_kv = CACHE_ONLY_APPEND
        result['provenance'] = collect_artifact_provenance(
            repo_root=root, configured_backend='hip_gfx1151', resolved_backend='hip_gfx1151', target_arch='gfx1151',
            model_path=args.model, quant='w4_paro', kv_dtype='bf16', command=[sys.executable, *sys.argv],
            build_profile='shipping public PARO staged adapter', timing_protocol=result['protocol']['decode_timing'],
            warmups=args.warmups, repetitions=args.repetitions, profiler={'enabled': False, 'reason': 'paired public latency measurement'},
            hipcc_version=Path('/tmp/hipengine-paro-refresh-20261023/hipcc-version.txt').read_text())
        save()
    print('PASS', args.output, flush=True)

if __name__ == '__main__':
    main()
