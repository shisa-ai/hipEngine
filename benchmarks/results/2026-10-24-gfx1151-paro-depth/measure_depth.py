#!/usr/bin/env python3
"""Measure public PARO candidate depths against a true autoregressive baseline."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.memory import memory_stats
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from tokenizers import Tokenizer

ARMS = ('ar', 'b1', 'b2', 'b3')

def digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()

def runner_in(value):
    if isinstance(value, dict):
        if 'routes' in value and 'recent_completed' in value['routes']:
            return value
        for child in value.values():
            result = runner_in(child)
            if result is not None:
                return result
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', choices=('all', 'smoke'), default='all')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--parent', type=Path, help='Archived depth-one reference, not timing samples')
    args = parser.parse_args()
    assert args.repetitions >= 1
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    cases, sources = [], []
    for split, name in [('category', 'mtpbench-code-general-ja.jsonl'), ('heldout', 'gdn-prefill-category-heldouts.jsonl')]:
        path = Path('benchmarks/prompts') / name
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            item = json.loads(line)
            prompt = '<|im_start|>user\n' + item['messages'][0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'
            cases.append({'id': item['id'], 'category': item['category'], 'split': split, 'tokens': tokenizer.encode(prompt).ids})
    for length in (37, 255, 257, 511, 512, 513, 1024, 1025, 4096):
        cases.append({'id': f'p{length}', 'category': 'length_control', 'split': 'control', 'tokens': [9707] * length})
    if args.cases == 'smoke':
        cases = [cases[0], *[c for c in cases if c['id'] in ('p513', 'p4096')]]
    parent = {} if args.parent is None else {c['id']: c for c in json.loads(args.parent.read_text())['cases']}
    result = {'schema': 1, 'kind': 'public_paro_candidate_depth_comparison', 'passed': False, 'cases': [],
              'protocol': {'arms': ARMS, 'warmups_per_arm': 1, 'repetitions': args.repetitions,
                           'output_tokens': 128, 'max_sequence_length': 4352, 'prefix_cache': 'off',
                           'kv_storage': 'bf16', 'temperature': 0, 'ignore_eos': True,
                           'order': 'Rotate the four arms by pair index; one complete warmup round.',
                           'selection': 'Only desired candidate count is controlled; shared planner applies capacity and output tails.',
                           'timing': 'Public first-token to last-token wall for 127 post-first tokens; whole-call includes prefill and iterator exhaustion.',
                           'cross_depth_parity': 'Diagnostic; repeatability and acceptance are required within each depth.',
                           'sources': sources},
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    original_budget = Qwen35ParoResidentModelRunner.speculative_desired_candidate_count
    original_reclaim = Qwen35ParoResidentModelRunner.reclaim
    completed_ids, active_arm = [], 'ar'
    def budget(owner, request):
        return 0 if request.max_tokens <= 1 or active_arm == 'ar' else int(active_arm[1:])
    def reclaim(owner, completed):
        original_reclaim(owner, completed)
        completed_ids.append(tuple(int(x) for x in completed.generated_tokens))
    Qwen35ParoResidentModelRunner.speculative_desired_candidate_count = budget
    Qwen35ParoResidentModelRunner.reclaim = reclaim
    try:
        with closing(LLM(str(args.model), backend='hip_gfx1151', quant='w4_paro', max_active_requests=1,
                         max_sequence_length=4352, prefix_cache='off')) as llm:
            for case in cases:
                record = {k: v for k, v in case.items() if k != 'tokens'}
                record.update(prompt_tokens=len(case['tokens']), prompt_ids_sha256=digest(case['tokens']), samples=[])
                result['cases'].append(record)
                references, acceptance = {}, {}
                for pair in range(1 + args.repetitions):
                    order = ARMS[pair % 4:] + ARMS[:pair % 4]
                    for active_arm in order:
                        mtp = active_arm != 'ar'
                        params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, speculative_mtp=mtp, kv_storage='bf16')
                        completed_ids.clear()
                        events, previous_count = [], 0
                        started = time.perf_counter()
                        stream = (llm.stream_speculative_mtp_detailed([case['tokens']], params) if mtp else llm.stream_detailed(case['tokens'], params))
                        for chunk in stream:
                            observed = time.perf_counter()
                            if chunk.telemetry is not None:
                                count = int(chunk.telemetry.decode_state.generated_tokens)
                                assert count >= previous_count
                                if count > previous_count:
                                    events.append({'tokens': count, 'seconds': observed - started})
                                    previous_count = count
                        elapsed = time.perf_counter() - started
                        assert len(completed_ids) == 1
                        ids = completed_ids[0]
                        assert len(ids) == 128 and events[0]['tokens'] == 1 and events[-1]['tokens'] == 128
                        assert all(0 <= x < tokenizer.get_vocab_size(with_added_tokens=True) for x in ids)
                        runner = runner_in(llm.live_loop_snapshot())
                        assert runner is not None and runner['model_runner']['active_requests'] == 0
                        route = dict(runner['routes']['recent_completed'][-1])
                        route.pop('scheduler_chunks', None)
                        assert bool(route['specdec2_mtp2_used']) == mtp
                        fallback = runner['routes']['fallback_reasons']
                        assert not any(k.startswith('specdec2_mtp2_recovered') for k in fallback), fallback
                        if mtp:
                            counts = route['specdec2_mtp2_candidate_counts']
                            accepted = route['specdec2_mtp2_accepted_counts']
                            depth = int(active_arm[1:])
                            assert counts and max(counts) == depth and all(1 <= n <= depth for n in counts), counts
                            assert all(0 <= a <= n for a, n in zip(accepted, counts, strict=True))
                            assert set(route['specdec2_mtp2_execution_routes']) == {'eager'}
                            if active_arm not in acceptance:
                                acceptance[active_arm] = (counts, accepted)
                            assert (counts, accepted) == acceptance[active_arm], (case['id'], active_arm, 'acceptance repeat')
                        if active_arm not in references:
                            references[active_arm] = ids
                            record[active_arm + '_ids'] = list(ids)
                            record[active_arm + '_text'] = tokenizer.decode(list(ids), skip_special_tokens=False)
                        assert ids == references[active_arm], (case['id'], active_arm, 'output repeat')
                        if active_arm == 'b1' and case['id'] in parent:
                            assert list(ids) == parent[case['id']]['eager_ids'], (case['id'], 'depth-one parent drift')
                        seconds = events[-1]['seconds'] - events[0]['seconds']
                        record['samples'].append({'pair': pair, 'warmup': pair == 0, 'arm': active_arm,
                            'ids_sha256': digest(ids), 'generated_tokens': 128, 'decode_tokens': 127,
                            'first_token_seconds': events[0]['seconds'], 'decode_seconds': seconds,
                            'decode_tokens_per_second': 127 / seconds, 'whole_call_seconds': elapsed,
                            'whole_call_tokens_per_second': 128 / elapsed, 'events': events,
                            'route': route, 'fallback_reasons': fallback})
                    save()
                record['cross_depth_ids_equal'] = {arm: references[arm] == references['b1'] for arm in ARMS}
                record['medians'] = {arm: {key: statistics.median(s[key] for s in record['samples'] if s['arm'] == arm and not s['warmup'])
                    for key in ('decode_tokens_per_second', 'whole_call_tokens_per_second', 'first_token_seconds')} for arm in ARMS}
                print(json.dumps({'case': case['id'], 'medians': record['medians'], 'cross_depth_ids_equal': record['cross_depth_ids_equal']}), flush=True)
                save()
        result['memory_after_close'] = memory_stats()
        assert result['memory_after_close']['active_allocations'] == result['memory_after_close']['current_allocated_bytes'] == 0
        result['passed'] = True
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        Qwen35ParoResidentModelRunner.speculative_desired_candidate_count = original_budget
        Qwen35ParoResidentModelRunner.reclaim = original_reclaim
        result['provenance'] = collect_artifact_provenance(
            repo_root=Path.cwd(), configured_backend='hip_gfx1151', resolved_backend='hip_gfx1151', target_arch='gfx1151',
            model_path=args.model, quant='w4_paro', kv_dtype='bf16', command=[sys.executable, *sys.argv],
            build_profile='shipping PARO arithmetic and eager verifier; requested depths use shared capability planner',
            timing_protocol=result['protocol']['timing'], warmups=1, repetitions=args.repetitions,
            profiler={'enabled': False}, hipcc_version=Path('/tmp/hipengine-paro-refresh-20261023/hipcc-version.txt').read_text())
        save()

if __name__ == '__main__':
    main()
