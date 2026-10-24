"""Public MoE explicit-MTP refusal containment and subsequent-request recovery."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.generation.qwen35_gguf_moe_mtp2 import Qwen35GGUFMoEMTP2Adapter

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
result = {'passed': False, 'fault': 'temporarily cap native target context at 128 positions'}
original = Qwen35GGUFMoEMTP2Adapter._target_context_ceiling
params = SamplingParams(max_tokens=16, temperature=0, speculative_mtp=True, kv_storage='bf16')
try:
    with closing(LLM('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', backend='hip_gfx1151',
                     quant='gguf_q4_k_m', max_active_requests=1, max_sequence_length=2048,
                     prefix_cache='off')) as llm:
        before = llm.generate_detailed([[9707] * 37], params)[0]
        Qwen35GGUFMoEMTP2Adapter._target_context_ceiling = lambda self, target: min(128, original(self, target))
        try:
            llm.generate_detailed([[9707] * 127], params)
        except Exception as error:
            result['refusal'] = f'{type(error).__name__}: {error}'
            assert 'MoePromptContextUnsupportedError' in result['refusal'], result['refusal']
        else:
            raise AssertionError('Explicit MTP silently ran without candidate context room')
        finally:
            Qwen35GGUFMoEMTP2Adapter._target_context_ceiling = original
        result['snapshot_after_refusal'] = llm.live_loop_snapshot()
        after = llm.generate_detailed([[9707] * 37], params)[0]
        result['before_ids'] = list(before.generated_token_ids)
        result['after_ids'] = list(after.generated_token_ids)
        assert result['before_ids'] == result['after_ids']
        result['snapshot_after_recovery'] = llm.live_loop_snapshot()
    result['memory_after_close'] = memory_stats()
    assert result['memory_after_close']['active_allocations'] == 0
    result['passed'] = True
finally:
    Qwen35GGUFMoEMTP2Adapter._target_context_ceiling = original
    a.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
