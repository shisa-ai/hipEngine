"""Historical gfx1151 public Gemma MTP batch smoke (2026-10-18).

Requires the target GGUF at model below and its discoverable MTP assistant.
Edit that path for your installation. This script loads the model on the GPU
and writes raw outputs under /tmp; do not commit those outputs. The original
run is recorded in the gemma-mtp-batching worklog entry.
"""
import dataclasses
import json
import time
from hipengine import LLM, SamplingParams

model = '/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf'
print('loading', flush=True)
llm = LLM(model, backend='hip_gfx1151', max_active_requests=4,
          max_sequence_length=256, prefix_cache='off', execution_profile='strict',
          engine_command_timeout_seconds=600)
prompts = ['The capital of France is', 'def add(a, b):\n    return',
           '日本の首都は', 'Count from one to five:']
results=[]
try:
    for width in (1, 2, 3, 4):
        print('width', width, flush=True)
        outputs=llm.generate_detailed(prompts[:width], SamplingParams(
            max_tokens=12, temperature=0, ignore_eos=True, speculative_mtp=True))
        row=[dataclasses.asdict(output) for output in outputs]
        print(json.dumps({'width':width,'outputs':row},ensure_ascii=False,default=str),flush=True)
        assert len(outputs)==width
        for output in outputs:
            assert len(output.generated_token_ids)==12
            assert output.telemetry.diagnostics['speculative_provider']=='gemma4_mtp'
        assert any(width in output.telemetry.diagnostics['physical_target_request_widths'] for output in outputs)
        results.append({'width':width,'outputs':row})
    with open('/tmp/gemma-mtp-public-batch-smoke.json','w') as f:
        json.dump(results,f,ensure_ascii=False,default=str)
finally:
    llm.close()
print('PASS',flush=True)
