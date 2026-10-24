"""Public PARO depth diagnostics against serial C1 logits at identical cycle inputs.

This is a cycle-local C1 teacher check, not an independent BF16 model oracle or
an arithmetic-profile promotion certificate. Timing is intentionally omitted.
"""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from hipengine import LLM, SamplingParams
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr, memory_stats
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter
from scripts.quant_quality.metrics import per_row_metrics
from scripts.mtp_paro_verifier_numerics import _summary, _THRESHOLDS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
tokenizer = Tokenizer.from_file(str(a.model / 'tokenizer.json'))
cases = []
for split, name in [('category', 'mtpbench-code-general-ja.jsonl'), ('heldout', 'gdn-prefill-category-heldouts.jsonl')]:
    for line in (Path('benchmarks/prompts') / name).read_text().splitlines():
        item = json.loads(line)
        text = '<|im_start|>user\n' + item['messages'][0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'
        cases.append({'id': item['id'], 'category': item['category'], 'split': split, 'tokens': tokenizer.encode(text).ids})
for length in (37, 255, 257, 511, 513, 1025, 4096):
    cases.append({'id': f'p{length}', 'category': 'length_control', 'split': 'control', 'tokens': [9707] * length})
if a.smoke:
    cases = [cases[0], cases[-1]]
result = {'passed': False, 'performance_claim': False, 'kind': 'public_depth_cycle_local_c1_teacher',
          'thresholds': _THRESHOLDS, 'cases': [], 'limitations': __doc__,
          'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.output.parent.mkdir(parents=True, exist_ok=True)
def save():
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
def copy(session, ptr, shape, dtype):
    values = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(values), DeviceBuffer(int(ptr), values.nbytes), values.nbytes, runtime=session.runtime)
    return values

depth, rows = 1, []
original_verify = Qwen35ParoMTP2Adapter._verify_target
original_budget = Qwen35ParoResidentModelRunner.speculative_desired_candidate_count

def verify(session, batch, **kwargs):
    tokens = list(batch.tokens)
    device = kwargs.get('candidate_token_ids_i32')
    if device is not None:
        candidates = copy(session, device.ptr, (len(tokens) - 1,), np.int32)
        tokens[1:] = [int(x) for x in candidates]
    assert len(tokens) == len(batch.positions)
    position = int(session.position_arr[0])
    reference = []
    for token, pos in zip(tokens, batch.positions, strict=True):
        session.step(int(token), position=int(pos), sample=True)
        reference.append(copy(session, session.lm_logits.ptr, (session.vocab_size,), np.float32))
    # Slot 1 is the existing adapter's pre-cycle target backup. The C1 teacher
    # only appended KV beyond the live prefix, so mutable state/cursors suffice.
    session.copy_slot_state(1, 0, kv_rows=0)
    session._set_slot_position(position, slot=0)
    output = original_verify(session, batch, **kwargs)
    candidate = copy(session, session.verify_lm_logits.ptr, (len(tokens), session.vocab_size), np.float32)
    reference = np.stack(reference)
    assert np.isfinite(reference).all() and np.isfinite(candidate).all()
    metrics = per_row_metrics(reference, candidate, np.argmax(reference, axis=1))
    rows.append({'positions': list(batch.positions), 'tokens': tokens,
                 'reference_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
                 'candidate_sha256': hashlib.sha256(candidate.tobytes()).hexdigest(),
                 'metrics': {k: v.tolist() for k, v in metrics.items()}})
    return output

Qwen35ParoMTP2Adapter._verify_target = staticmethod(verify)
Qwen35ParoResidentModelRunner.speculative_desired_candidate_count = lambda self, request: depth
try:
    with closing(LLM(str(a.model), backend='hip_gfx1151', quant='w4_paro', max_active_requests=1,
                     max_sequence_length=4352, prefix_cache='off')) as llm:
        for case in cases:
            for depth in (1, 2, 3):
                item = {k: v for k, v in case.items() if k != 'tokens'}
                item.update(depth=depth, prompt_tokens=len(case['tokens']), repeats=[])
                result['cases'].append(item)
                Qwen35ParoMTP2Adapter._verify_target = staticmethod(original_verify)
                baseline = llm.generate_detailed([case['tokens']], SamplingParams(max_tokens=16, temperature=0,
                    ignore_eos=True, speculative_mtp=True, kv_storage='bf16'))[0]
                item['uninstrumented_ids'] = list(baseline.generated_token_ids)
                Qwen35ParoMTP2Adapter._verify_target = staticmethod(verify)
                for repeat in range(3):
                    rows.clear()
                    output = llm.generate_detailed([case['tokens']], SamplingParams(max_tokens=16, temperature=0,
                        ignore_eos=True, speculative_mtp=True, kv_storage='bf16'))[0]
                    assert len(output.generated_token_ids) == 16 and rows
                    assert list(output.generated_token_ids) == item['uninstrumented_ids'], (case['id'], depth, 'instrumentation drift')
                    record = {'ids': list(output.generated_token_ids), 'cycles': list(rows)}
                    if item['repeats']:
                        assert record == item['repeats'][0], (case['id'], depth, 'repeat drift')
                    item['repeats'].append(record)
                print(case['id'], depth, 'repeat passed', flush=True)
                save()
    result['memory_after_close'] = memory_stats()
    assert result['memory_after_close']['active_allocations'] == 0
    summaries = {}
    for depth in (1, 2, 3):
        selected = [item for item in result['cases'] if item['depth'] == depth]
        for category in ['all', *sorted({item['category'] for item in selected})]:
            cycles = [cycle for item in selected if category == 'all' or item['category'] == category
                      for cycle in item['repeats'][0]['cycles']]
            summary = _summary(np.asarray([x for c in cycles for x in c['metrics']['kl_nats']]),
                               np.asarray([x for c in cycles for x in c['metrics']['top1_equal']]))
            summary['passed'] = all(summary[name] <= _THRESHOLDS[name + '_max'] for name in ('mean_kl', 'p95_kl', 'p99_kl', 'max_kl')) and summary['top1_agreement'] >= _THRESHOLDS['top1_min' if category == 'all' else 'per_scope_top1_min']
            summaries[f'b{depth}:{category}'] = summary
    result['summaries'] = summaries
    result['passed'] = all(s['passed'] for s in summaries.values())
    assert result['passed'], 'Cycle-local C1 teacher envelope failed; inspect summaries'
except BaseException as error:
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    Qwen35ParoMTP2Adapter._verify_target = staticmethod(original_verify)
    Qwen35ParoResidentModelRunner.speculative_desired_candidate_count = original_budget
    save()
