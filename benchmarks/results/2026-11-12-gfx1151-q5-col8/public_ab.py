"""Public scalar-copy/batched-snapshot A/B with fresh model owners."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from hipengine import LLM, SamplingParams
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.kernels.hip_gfx1151 import policy_gguf as policy
import hipengine.kernels.hip_gfx1151 as backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    prompts_path = Path('benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl')
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines()]
    original = policy._GGUF_Q5_T16_ROWTILE_COL8_SHAPES
    parent = original - {(5120, 12288), (5120, 17408)}
    original_col8 = backend.gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out
    results = []
    runtime = get_hip_runtime()
    initial_memory = memory_stats()
    for arm in ('ar', 'loop', 'batch', 'batch', 'loop'):
        counts = Counter()

        def execute(*args, **kwargs):
            counts[str((args[3], args[4], args[5]))] += 1
            return original_col8(*args, **kwargs)

        backend.gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out = execute
        policy._GGUF_Q5_T16_ROWTILE_COL8_SHAPES = parent if arm == 'loop' else original
        llm = LLM('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf', backend='hip_gfx1151',
                  kv_storage='bf16', max_sequence_length=2048, prefix_cache='off',
                  speculative_candidate_budget=3)
        params = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True,
                                speculative_mtp=arm != 'ar')
        try:
            llm.generate_detailed(['List three uses of a hash table.'], params)
            runtime.device_synchronize()
            counts.clear()
            samples = []
            for prompt in prompts:
                text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                               for m in prompt['messages']) + '<|im_start|>assistant\n'
                started = time.perf_counter()
                row = llm.generate_detailed([text], params)[0]
                runtime.device_synchronize()
                samples.append(dict(id=prompt['id'], category=prompt['category'],
                                    wall_s=time.perf_counter()-started,
                                    ids=list(row.generated_token_ids),
                                    telemetry=row.telemetry.to_json_dict()))
                print(arm, prompt['id'], samples[-1]['wall_s'], flush=True)
            results.append(dict(arm=arm, samples=samples, mixed_plan_counts=dict(counts),
                                profile=llm.resolved_execution_profile,
                                manifest=llm.execution_profile_manifest_sha256))
        finally:
            llm.close()
            policy._GGUF_Q5_T16_ROWTILE_COL8_SHAPES = original
            backend.gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out = original_col8
        results[-1]['memory_after_close'] = memory_stats()
        Path(args.out).write_text(json.dumps(dict(results=results), indent=2))
    loop = next(r for r in results if r['arm'] == 'loop')
    checks = []
    for result in results:
        if result['arm'] == 'ar':
            continue
        assert result['mixed_plan_counts'], result['arm']
        for expected, got in zip(loop['samples'], result['samples']):
            a = expected['telemetry']['diagnostics']['specdec2_mtp2']
            b = got['telemetry']['diagnostics']['specdec2_mtp2']
            same_control = all(a[key] == b[key] for key in (
                'cycles', 'accepted_draft_tokens', 'generated_draft_tokens',
                'selected_depth_histogram', 'failure_reason_counts', 'ar_step_reason_counts'))
            engaged = b['cycles'] > 0 and b['recoverable_failures'] == 0
            checks.append(dict(arm=result['arm'], id=got['id'],
                               ids_equal=expected['ids'] == got['ids'],
                               control_equal=same_control, engaged=engaged))
    source_paths = ('hipengine/kernels/hip_gfx1151/policy_gguf.py',
                    'hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.hip')
    payload = dict(results=results, checks=checks, host=platform.node(),
                   initial_memory=initial_memory, final_memory=memory_stats(),
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                   source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths},
                   prompt_sha256=hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
                   env={k:v for k,v in os.environ.items() if k.startswith(('HIPENGINE_', 'GPU_MAX_','HIP_VISIBLE'))})
    payload['passed'] = all(c['ids_equal'] and c['control_equal'] and c['engaged'] for c in checks)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    assert payload['passed'], 'public loop/batch parity or control failure'


if __name__ == '__main__':
    main()
