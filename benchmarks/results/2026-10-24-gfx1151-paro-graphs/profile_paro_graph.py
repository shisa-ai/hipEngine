"""One public PARO graph request for prebuilt rocprofv3 tracing."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--length', type=int, default=37)
a = p.parse_args()
original = Qwen35ParoResidentSession.verify_chain_bulk_and_commit
verifications = []
def observe(session, *args, **kwargs):
    result = original(session, *args, **kwargs)
    verifications.append(dict(result.graph))
    return result
Qwen35ParoResidentSession.verify_chain_bulk_and_commit = observe
result = {'passed': False, 'performance_claim': False, 'prompt_tokens': a.length}
try:
    with closing(LLM(a.model, backend='hip_gfx1151', quant='w4_paro', max_active_requests=1,
                     max_sequence_length=4352, prefix_cache='off')) as llm:
        output = llm.generate_detailed([[9707] * a.length], SamplingParams(
            max_tokens=32, temperature=0, ignore_eos=True, speculative_mtp=True, kv_storage='bf16'))[0]
        result['ids'] = list(output.generated_token_ids)
        result['verifications'] = verifications
        assert len(result['ids']) == 32
        assert any(v['status'] == 'replayed' for v in verifications)
        assert not any(v['status'].endswith('fallback') for v in verifications)
        result['snapshot'] = llm.live_loop_snapshot()
    result['memory_after_close'] = memory_stats()
    assert result['memory_after_close']['active_allocations'] == 0
    result['passed'] = True
finally:
    Qwen35ParoResidentSession.verify_chain_bulk_and_commit = original
    a.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
