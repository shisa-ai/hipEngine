"""Public GGUF prompt-sink regression; no timing claim."""
import argparse
from contextlib import closing
import json
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_gguf_mtp import _StreamingNextNPromptSink
from hipengine.generation.qwen35_gguf_moe_mtp2 import _MoeTargetHiddenSink

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--models', nargs='+', default=['Qwen3.8-27B-UD-Q4_K_M.gguf', 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'])
parser.add_argument('--lengths', nargs='+', type=int, default=[37, 1025])
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()
records = []
originals = []
for cls in (_StreamingNextNPromptSink, _MoeTargetHiddenSink):
    original = cls.consume
    originals.append((cls, original))
    def wrapper(self, _original=original, _name=cls.__name__, **kwargs):
        result = _original(self, **kwargs)
        records.append({'sink': _name, 'request_id': kwargs['request_id'],
                        'start': kwargs['chunk_start'], 'rows': kwargs['rows']})
        return result
    cls.consume = wrapper

def routes_in(value):
    if isinstance(value, dict):
        if 'routes' in value and 'recent_completed' in value['routes']:
            return value['routes']
        for child in value.values():
            found = routes_in(child)
            if found is not None:
                return found
    return None

result = {'passed': False, 'models': [], 'performance_claim': False}
try:
    for model in args.models:
        item = {'model': model, 'cases': []}
        result['models'].append(item)
        with closing(LLM('/models/gguf/' + model, backend='hip_gfx1151', quant='gguf_q4_k_m',
                         max_active_requests=1, max_sequence_length=2048, prefix_cache='off')) as llm:
            params = SamplingParams(max_tokens=16, temperature=args.temperature, seed=args.seed,
                                    speculative_mtp=True, kv_storage='bf16')
            for length in args.lengths:
                outputs = []
                for repeat in range(2):
                    records.clear()
                    output = llm.generate_detailed([[9707] * length], params)[0]
                    ids = list(output.generated_token_ids)
                    assert len(ids) == 16
                    if not records:
                        item['failure_snapshot'] = llm.live_loop_snapshot()
                    assert records, 'public MTP route did not use either prompt sink'
                    routes = routes_in(llm.live_loop_snapshot())
                    assert routes is not None
                    assert routes['recent_completed'][-1]['specdec2_mtp2_used'], routes
                    assert not any(k.startswith('specdec2_mtp2_recovered') for k in routes['fallback_reasons']), routes
                    outputs.append({'ids': ids, 'sink_chunks': list(records), 'route': routes['recent_completed'][-1]})
                assert outputs[0]['ids'] == outputs[1]['ids']
                item['cases'].append({'length': length, 'runs': outputs})
            item['final_snapshot'] = llm.live_loop_snapshot()
        item['memory_after_close'] = memory_stats()
        assert item['memory_after_close']['active_allocations'] == 0
        args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
        print(model, [(c['length'], len(c['runs'][0]['sink_chunks'])) for c in item['cases']], flush=True)
    result['passed'] = True
finally:
    for cls, original in originals:
        cls.consume = original
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
