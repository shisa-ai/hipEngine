"""Check unmodified public depth-two selection against the measured depth-two arm.

This checks outputs and ownership, not performance. No planner or resolver is
patched; the LLM constructor deliberately omits speculative_candidate_budget.
"""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from tokenizers import Tokenizer
from measure_depth import digest, runner_in


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    prompts = {}
    for source in reference['protocol']['sources']:
        path = Path(source['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
        for line in path.read_text().splitlines():
            item = json.loads(line)
            prompt = '<|im_start|>user\n' + item['messages'][0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'
            prompts[item['id']] = tokenizer.encode(prompt).ids
    result = {'passed': False, 'performance_claim': False, 'cases': [],
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        with closing(LLM(str(args.model), backend='auto', quant='w4_paro', max_active_requests=1,
                         max_sequence_length=4352, prefix_cache='off')) as llm:
            for case in reference['cases']:
                tokens = ([9707] * case['prompt_tokens'] if case['split'] == 'control'
                          else prompts[case['id']])
                assert digest(tokens) == case['prompt_ids_sha256']
                expected = next(s['route'] for s in case['samples'] if s['arm'] == 'b2')
                for repeat in range(2):
                    output = llm.generate_detailed([tokens], SamplingParams(
                        max_tokens=128, temperature=0, ignore_eos=True,
                        speculative_mtp=True, kv_storage='bf16'))[0]
                    assert list(output.generated_token_ids) == case['b2_ids'], case['id']
                    owner = runner_in(llm.live_loop_snapshot())
                    assert owner['model_runner']['active_requests'] == 0
                    routes = owner['routes']
                    route = routes['recent_completed'][-1]
                    assert route['specdec2_mtp2_used']
                    assert max(route['specdec2_mtp2_candidate_counts']) == 2
                    for key in ('specdec2_mtp2_candidate_counts', 'specdec2_mtp2_accepted_counts',
                                'specdec2_mtp2_execution_routes'):
                        assert route[key] == expected[key], (case['id'], key)
                    assert not any(k.startswith('specdec2_mtp2_recovered') for k in routes['fallback_reasons'])
                result['cases'].append({'id': case['id'], 'repeats': 2,
                                        'ids_sha256': digest(output.generated_token_ids)})
                print(case['id'], 'default depth two matches', flush=True)
        result['memory_after_close'] = memory_stats()
        assert result['memory_after_close']['active_allocations'] == 0
        assert result['memory_after_close']['current_allocated_bytes'] == 0
        result['passed'] = True
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
