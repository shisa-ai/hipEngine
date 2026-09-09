import sys
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
model = sys.argv[1]
with Qwen35GGUFResidentSession(
    model, compiler_version=open('/tmp/ud-hipcc-version.txt').read(),
    max_sequence_length=600, use_wmma_prefill=True, use_gemv_decode=True,
) as session:
    session.prefill([9707] * 512, use_bulk=True, bulk_attention_mode='bulk', return_logits=True)  # warm
    session.reset()
    session.prefill([9707] * 512, use_bulk=True, bulk_attention_mode='bulk', return_logits=True, record_gpu_stage_timings=True)
    t = session.last_prefill_gpu_stage_timings_ms
    for k, v in sorted(t.items(), key=lambda x: -x[1])[:25]:
        print(f"{v:9.3f} ms  {k}")
    print("TOTAL:", round(sum(t.values()), 1))
