# Qwen4Exp X-10 routed-F16 leave-one-out

Five counterbalanced prefill pairs were measured for all 12 canonical cases on the Framework gfx1151 host. The bound arm is the named production composition; the override arm disables `HIPENGINE_QWEN4_EXP_ROUTED_F16=0` after profile binding.

- All 60 pair samples completed.
- Bound and override arms were repeat-exact.
- Cross-route logits were exact for every case.
- Disabling routed F16 is slower at every shape/category. Override/bound throughput ratios range from 0.4236 to 0.4781 by case; p4096 ratios are 0.4266-0.4313.
- The routed-F16 production component is therefore retained. The override is rejected as a performance candidate, not promoted.

Artifact: `routed-f16-ab.json`.

This is one X-10 leave-one-component-out ablation. It does not close the full final-stack protocol: the remaining component interactions, telemetry, task review, and complete-EOS/heldout quality accounting remain open.