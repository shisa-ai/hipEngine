"""Device-side prefill stage census (deduplicated).

The stage recorder accumulates each interval under every alias name
(mark(specific, stage_prefix)), so summing the result dict double counts
aliased intervals. This census drops the aliases at mark time so each
interval lands under exactly one key; the per-key values remain the
per-stage totals across layers.
"""
import sys

sys.path.insert(0, "/home/lhl/hipEngine-ud")

from hipengine.runtime.qwen35_gguf_runner import (  # noqa: E402
    _HipWallClockStageRecorder,
    Qwen35GGUFResidentSession,
)

_orig_mark = _HipWallClockStageRecorder.mark


def _mark_primary_only(self, name, *_aliases):
    return _orig_mark(self, name)


_HipWallClockStageRecorder.mark = _mark_primary_only

model = sys.argv[1]
with Qwen35GGUFResidentSession(
    model, compiler_version=open("/tmp/ud-hipcc-version.txt").read(),
    max_sequence_length=600, use_wmma_prefill=True, use_gemv_decode=True,
) as session:
    session.prefill([9707] * 512, use_bulk=True, bulk_attention_mode="bulk",
                    return_logits=True)  # warm
    session.reset()
    session.prefill([9707] * 512, use_bulk=True, bulk_attention_mode="bulk",
                    return_logits=True, record_gpu_stage_timings=True)
    t = session.last_prefill_gpu_stage_timings_ms
    total = 0.0
    for k, v in sorted(t.items(), key=lambda x: -x[1]):
        print(f"{v:9.3f} ms  {k}")
        total += v
    print(f"DEDUPED TOTAL: {total:.1f} ms")
