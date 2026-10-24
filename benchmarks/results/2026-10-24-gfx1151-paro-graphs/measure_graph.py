#!/usr/bin/env python3
"""Compare public PARO eager/graph MTP and a true non-MTP baseline."""
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
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from tokenizers import Tokenizer


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
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--cases', choices=('all', 'smoke'), default='all')
    parser.add_argument('--attribution', action='store_true')
    args = parser.parse_args()
    assert args.warmups >= 1 and args.repetitions >= 1
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
    for length in (37, 255, 257, 511, 512, 513, 1024, 1025, 4096):
        cases.append({'id': f'p{length}', 'category': 'length_control', 'split': 'control', 'tokens': [9707] * length})
    if args.cases == 'smoke':
        cases = [cases[0], *[case for case in cases if case['id'] in ('p512', 'p4096')]]
    result = {'schema': 1, 'kind': 'public_paro_graph_comparison', 'attribution': args.attribution, 'protocol': {
        'backend': 'hip_gfx1151', 'quant': 'w4_paro', 'kv_storage': 'bf16', 'max_active_requests': 1,
        'max_sequence_length': 4352, 'prefix_cache': 'off', 'temperature': 0, 'ignore_eos': True,
        'output_tokens': 128, 'warmups_per_case_per_arm': args.warmups, 'samples_per_case_per_arm': args.repetitions,
        'order': 'AR/eager/graph on even pairs; graph/eager/AR on odd pairs',
        'baseline': 'Immediate parent eager verifier via graph_mode=off; all other production methods unchanged',
        'candidate': 'Default public verifier graph auto; shared bounded graph-stage ownership',
        'surface': 'LLM.stream_detailed for AR; LLM.stream_speculative_mtp_detailed for MTP',
        'decode_timing': 'Observed first-token to last-token wall; denominator is generated tokens minus first observed cumulative count. Terminal duplicate chunks excluded.',
        'first_token_timing': 'Public call start to first token, including prefill and MTP prompt priming.',
        'whole_call_timing': 'Public call start through iterator exhaustion; model construction and snapshot extraction excluded after warmup.',
        'profile': 'Shipping backend defaults; no gfx1151 PARO execution-profile plan resolves. No arithmetic or variant override.',
        'cross_route_id_equality': 'Diagnostic, not a same-schedule repeatability check.',
        'sources': sources,
    }, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': [], 'passed': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    completed_ids = []
    original_reclaim = Qwen35ParoResidentModelRunner.reclaim
    original_verify = Qwen35ParoResidentSession.verify_chain_bulk_and_commit
    active_arm = 'ar'
    verification_events = []
    def verify(session, *a, **kw):
        if active_arm == 'eager': kw['graph_mode'] = 'off'
        if args.attribution: session.runtime.device_synchronize()
        started = time.perf_counter()
        value = original_verify(session, *a, **kw)
        if args.attribution: session.runtime.device_synchronize()
        # Timing is only used in the separately synchronized attribution run.
        verification_events.append({'graph': dict(value.graph),
            'seconds': time.perf_counter() - started if args.attribution else None})
        return value
    Qwen35ParoResidentSession.verify_chain_bulk_and_commit = verify

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
                acceptance_reference = None
                for pair in range(args.warmups + args.repetitions):
                    for arm in (('ar', 'eager', 'graph') if pair % 2 == 0 else ('graph', 'eager', 'ar')):
                        active_arm = arm
                        mtp = arm != 'ar'
                        verification_events.clear()
                        params = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True, speculative_mtp=mtp, kv_storage='bf16')
                        events = []
                        ids = None
                        text = []
                        previous_count = 0
                        completed_ids.clear()
                        started = time.perf_counter()
                        stream = (llm.stream_speculative_mtp_detailed([case['tokens']], params)
                                  if mtp else llm.stream_detailed(case['tokens'], params))
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
                        assert bool(route['specdec2_mtp2_used']) == mtp, route
                        assert runner['model_runner']['active_requests'] == 0
                        if mtp:
                            assert route['specdec2_mtp2_cycles'] > 0
                            assert set(route['specdec2_mtp2_candidate_counts']) == {1}
                            if arm == 'eager':
                                assert set(route['specdec2_mtp2_execution_routes']) == {'eager'}
                                assert all(v['graph']['status'] == 'disabled' for v in verification_events)
                            else:
                                assert 'graph' in route['specdec2_mtp2_execution_routes']
                                assert any(v['graph']['status'] == 'replayed' for v in verification_events)
                                assert not any(v['graph']['status'].endswith('fallback') for v in verification_events)
                            accepted = route['specdec2_mtp2_accepted_counts']
                            if acceptance_reference is None: acceptance_reference = accepted
                            assert accepted == acceptance_reference, (case['id'], arm, 'acceptance mismatch')
                        else:
                            assert not verification_events
                        fallback = runner['routes']['fallback_reasons']
                        assert not any(k.startswith('specdec2_mtp2_recovered') for k in fallback), fallback
                        if arm not in references:
                            references[arm] = ids
                            record[arm + '_ids'] = list(ids)
                            record[arm + '_text'] = tokenizer.decode(list(ids), skip_special_tokens=False)
                        assert ids == references[arm], (case['id'], arm, 'repeat mismatch')
                        if mtp and 'eager' in references and 'graph' in references:
                            assert references['eager'] == references['graph'], (case['id'], 'eager/graph ID mismatch')
                        decode_seconds = events[-1]['seconds'] - events[0]['seconds']
                        sample = {'pair': pair, 'warmup': pair < args.warmups, 'arm': arm,
                                  'generated_tokens': len(ids), 'generated_ids_sha256': token_hash(ids),
                                  'first_token_seconds': events[0]['seconds'], 'decode_seconds': decode_seconds,
                                  'decode_tokens': 127, 'decode_tokens_per_second': 127 / decode_seconds,
                                  'whole_call_seconds': elapsed, 'whole_call_tokens_per_second': len(ids) / elapsed,
                                  'token_events': events, 'route': route, 'fallback_reasons': fallback,
                                  'verification_events': list(verification_events)}
                        record['samples'].append(sample)
                    save()
                record['ar_mtp_ids_equal'] = references['ar'] == references['graph']
                record['medians'] = {arm: {key: statistics.median(s[key] for s in record['samples'] if s['arm'] == arm and not s['warmup'])
                                          for key in ('decode_tokens_per_second', 'first_token_seconds', 'whole_call_tokens_per_second')}
                                     for arm in ('ar', 'eager', 'graph')}
                print(json.dumps({'case': case['id'], 'medians': record['medians'], 'ar_mtp_ids_equal': record['ar_mtp_ids_equal']}), flush=True)
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
        Qwen35ParoResidentSession.verify_chain_bulk_and_commit = original_verify
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
