"""Compare public dense GGUF checkpoint ownership across C1/C2 routes."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.speculative.transaction import SpeculativeStateJournal

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--candidate', action='store_true')
p.add_argument('--model', default='/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
a = p.parse_args()

def routes_in(value):
    if isinstance(value, dict):
        if 'routes' in value and 'recent_completed' in value['routes']:
            return value['routes']
        for child in value.values():
            result = routes_in(child)
            if result is not None: return result
    return None

captures = []
original_capture = SpeculativeStateJournal.capture

def capture(owner, name, *args, **kwargs):
    result = original_capture(owner, name, *args, **kwargs)
    captures.append(name)
    return result

SpeculativeStateJournal.capture = capture
result = {'passed': False, 'candidate': a.candidate, 'cases': [], 'performance_claim': False,
          'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
try:
    for width, temperature in ((1, 0.0), (1, 0.7), (2, 0.0)):
        with closing(LLM(a.model, backend='hip_gfx1151', quant='gguf_q4_k_m',
                         max_active_requests=width, max_sequence_length=2048,
                         prefix_cache='off')) as llm:
            for length in (37, 257):
                case = {'width': width, 'temperature': temperature, 'length': length, 'runs': []}
                result['cases'].append(case)
                for repeat in range(2):
                    captures.clear()
                    params = SamplingParams(max_tokens=32, temperature=temperature, seed=871,
                                            speculative_mtp=True, kv_storage='bf16')
                    outputs = llm.generate_detailed([[9707] * (length + i) for i in range(width)], params)
                    ids = [list(output.generated_token_ids) for output in outputs]
                    assert len(ids) == width and all(len(row) == 32 for row in ids)
                    routes = routes_in(llm.live_loop_snapshot())
                    assert routes is not None
                    completed = routes['recent_completed'][-width:]
                    assert len(completed) == width
                    assert all(row['specdec2_mtp2_used'] and row['specdec2_mtp2_cycles'] > 0 for row in completed), completed
                    assert not any(k.startswith('specdec2_mtp2_recovered') for k in routes['fallback_reasons']), routes
                    if a.candidate and width == 1:
                        assert 'provider' in captures and 'target' in captures, captures
                    elif width == 2:
                        assert not captures, captures
                    case['runs'].append({'ids': ids, 'routes': completed, 'captures': list(captures)})
                assert case['runs'][0]['ids'] == case['runs'][1]['ids'], case
                print(json.dumps({'width': width, 'temperature': temperature, 'length': length, 'passed': True}), flush=True)
                a.output.write_text(json.dumps(result, indent=2) + '\n')
        assert memory_stats()['active_allocations'] == 0, memory_stats()
    result['memory_after_close'] = memory_stats()
    result['passed'] = True
except BaseException as error:
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    SpeculativeStateJournal.capture = original_capture
    a.output.write_text(json.dumps(result, indent=2) + '\n')
