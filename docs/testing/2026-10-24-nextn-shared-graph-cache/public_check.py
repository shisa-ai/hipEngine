"""Observe actual NextN proposal graph use through the public dense GGUF API."""
import argparse
from contextlib import closing
import json
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', default='/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
a = p.parse_args()
original = Qwen35GGUFNextNExecutor._launch_graph_chain_device
original_close = Qwen35GGUFNextNExecutor.close
original_binding = Qwen35GGUFNextNExecutor._proposal_graph_binding
binding_errors = []
records, closes = [], []

def binding(owner, **kwargs):
    try:
        return original_binding(owner, **kwargs)
    except Exception as error:
        binding_errors.append(f'{type(error).__name__}: {error}')
        raise

def launch(owner, *args, **kwargs):
    value = original(owner, *args, **kwargs)
    records.append({'returned_device_proposal': value is not None, 'contract': owner.proposal_graph_contract()})
    return value

def close(owner):
    original_close(owner)
    cleanup = getattr(owner, 'proposal_graph_cleanup', None)
    closes.append(None if cleanup is None else cleanup())

Qwen35GGUFNextNExecutor._launch_graph_chain_device = launch
Qwen35GGUFNextNExecutor.close = close
Qwen35GGUFNextNExecutor._proposal_graph_binding = binding
result = {'passed': False, 'cases': [], 'binding_errors': binding_errors}
try:
    with closing(LLM(a.model, backend='hip_gfx1151', quant='gguf_q4_k_m', max_active_requests=1,
                     max_sequence_length=2048, prefix_cache='off')) as llm:
        params = SamplingParams(max_tokens=32, temperature=0,
                                speculative_mtp=True, kv_storage='bf16')
        for length in (37, 63, 257):
            runs = []
            for repeat in range(2):
                records.clear()
                output = llm.generate_detailed([[9707] * length], params)[0]
                assert len(output.generated_token_ids) == 32
                assert records, ('Public route did not reach NextN graph proposal execution', binding_errors, llm.live_loop_snapshot())
                assert any(r['returned_device_proposal'] for r in records), records
                assert any(r['contract']['replays'] > 0 for r in records), records
                assert not any(r['contract']['last_error'] for r in records), records
                runs.append({'ids': list(output.generated_token_ids), 'proposal_graphs': list(records)})
            assert runs[0]['ids'] == runs[1]['ids']
            result['cases'].append({'length': length, 'runs': runs})
    result['cleanup'] = closes
    assert closes
    for value in closes:
        if value is not None:
            assert value['pending_close_count'] == 0 and value['close_failure_count'] == 0
            assert value.get('pending_partial_releases', 0) == 0
    result['memory_after_close'] = memory_stats()
    assert result['memory_after_close']['active_allocations'] == 0
    result['passed'] = True
except BaseException as error:
    result['error'] = f'{type(error).__name__}: {error}'
    result['last_proposal_records'] = records
    raise
finally:
    Qwen35GGUFNextNExecutor._launch_graph_chain_device = original
    Qwen35GGUFNextNExecutor.close = original_close
    Qwen35GGUFNextNExecutor._proposal_graph_binding = original_binding
    a.output.write_text(json.dumps(result, indent=2) + '\n')
