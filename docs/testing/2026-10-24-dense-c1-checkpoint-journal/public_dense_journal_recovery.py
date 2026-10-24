"""Exercise dense C1 journal rollback after provider repair through LLM.generate."""
import argparse
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter as Adapter

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', default='/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
a = p.parse_args()
result = {'passed': False, 'cases': [], 'performance_claim': False}
params = SamplingParams(max_tokens=24, temperature=0, speculative_mtp=True, kv_storage='bf16')
originals = {name: getattr(Adapter, name) for name in ('propose_batch', '_rollback_state_journal', '_repair_provider_state', '_repair_provider_state_device')}
try:
    with closing(LLM(a.model, backend='hip_gfx1151', quant='gguf_q4_k_m', max_active_requests=1,
                     max_sequence_length=2048, prefix_cache='off')) as llm:
        for length in (37, 257):
            baseline = llm.generate_detailed([[9707] * length], params)[0]
            for action in ('failure', 'cancel'):
                token = GenerationCancellationToken()
                pending, injected, restored = {}, [], []
                repairs = [0]
                def propose(owner, plan, *args, **kwargs):
                    rid = plan.speculative_request_ids[0]
                    row = owner.owner._row(rid)
                    pending['position'] = int(row.lease.session.position)
                    pending['prefix'] = list(row.slot.generated_ids)
                    return originals['propose_batch'](owner, plan, *args, **kwargs)
                def rollback(owner, state):
                    active = state.state_journal is not None
                    value = originals['_rollback_state_journal'](owner, state)
                    if active and injected and not restored:
                        target = owner.owner._row(state.request_id).lease.session
                        assert int(target.position) == pending['position']
                        assert state.state_journal is None and state.proposal_checkpoint is None
                        restored.append(True)
                    return value
                def make_repair(name):
                    def repair(owner, state, *args, **kwargs):
                        value = originals[name](owner, state, *args, **kwargs)
                        repairs[0] += 1
                        if repairs[0] == 2:
                            injected.append(dict(pending))
                            if action == 'failure': raise RuntimeError('injected dense journal failure after provider repair')
                            token.cancel()
                        return value
                    return repair
                Adapter.propose_batch = propose
                Adapter._rollback_state_journal = rollback
                for name in ('_repair_provider_state', '_repair_provider_state_device'):
                    setattr(Adapter, name, make_repair(name))
                try:
                    output = llm.generate_detailed([[9707] * length], replace(params, cancellation_token=token))[0]
                finally:
                    for name, method in originals.items(): setattr(Adapter, name, method)
                assert len(injected) == 1 and restored == [True], (injected, restored, repairs)
                if action == 'cancel':
                    assert output.finish_details.cancelled
                    assert list(output.generated_token_ids) == injected[0]['prefix']
                else:
                    assert len(output.generated_token_ids) == 24
                recovered = llm.generate_detailed([[9707] * length], params)[0]
                assert recovered.generated_token_ids == baseline.generated_token_ids
                result['cases'].append({'length': length, 'action': action, 'baseline_ids': list(baseline.generated_token_ids),
                    'interrupted_ids': list(output.generated_token_ids), 'recovered_ids': list(recovered.generated_token_ids),
                    'injected': injected, 'restored': restored})
                print('PASS', length, action, flush=True)
                a.output.write_text(json.dumps(result, indent=2) + '\n')
    result['memory_after_close'] = memory_stats()
    assert result['memory_after_close']['active_allocations'] == 0
    result['passed'] = True
except BaseException as error:
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    for name, method in originals.items(): setattr(Adapter, name, method)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
