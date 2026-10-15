# Qwen4Exp X-10 GDN tiled16 leave-one-out

The current production composition was compared with `HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0` over all 12 canonical cases using five counterbalanced pairs per case, chunk1024, and one warmup per arm.

- All 60 measured pairs completed.
- Both arms were repeat-exact and cross-route logits were exact for every case.
- Override/bound throughput ratios ranged from 0.9976 to 1.0012 by case; the component is timing-neutral at this protocol's resolution.
- Contemporaneous `rocm-smi` telemetry was sampled every 0.5 seconds: 429 samples, zero telemetry errors, temperatures from 41 C to 65 C, and the raw time-ordered stream is preserved.
- No production change is justified. The tiled16 GDN component remains selected because the leave-one-out did not establish a non-regressive measurable win for its removal.

Artifacts: `gdn-tile16-ab.json`, `telemetry.jsonl`.

This is the second X-10 leave-one-component-out ablation. Full final-stack qualification still needs the combined quality/task gate, remaining interaction-relevant ablations, and completion of the stable timing packet.