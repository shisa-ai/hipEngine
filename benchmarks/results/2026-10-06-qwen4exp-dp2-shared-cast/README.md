# DP-2 bounded two-consumer F16 ownership

The bounded native two-consumer route is correctness-qualified but is **not promoted**. A tracked-clean same-host engine A/B on Radeon 8060S/gfx1151 completes the full 28-case suite: 12 canonical cases plus 16 category-heldout cases at p1024/p4096. It records 168 measured arm runs, three repetitions per arm, one warmup per arm, exact output trajectories, and zero allocations after teardown.

## Engine A/B result

The candidate enables `HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR=1`; the parent disables it. Both arms run in one Python process with one model residency and alternating six-slot arm order. Timing charges synchronized prefill plus exactly 128 decode transitions and the final device synchronization. The model is Qwen3.8-Flash-Next UD-Q4_K_XL, production/BF16, chunk1024, on host `55ea6c509d0b49eea8de7094a1023668`.

| Shape | Prefill candidate/parent | Decode candidate/parent | Complete client wall candidate/parent |
| ---: | ---: | ---: | ---: |
| 512 | 1.000095x (+0.010%) | 1.000131x (+0.013%) | 1.000123x (+0.012%) |
| 1024 | 0.999418x (−0.058%) | 1.000615x (+0.061%) | 1.000230x (+0.023%) |
| 4096 | 0.999612x (−0.039%) | 1.000965x (+0.096%) | 1.000263x (+0.026%) |

Across the qualified p1024/p4096 rows, prefill is 0.999575x (−0.043%), decode is 1.000842x (+0.084%), and complete client wall is 1.000254x (+0.025%). Across all 28 cases, prefill is 0.999592x (−0.041%), decode is 1.000774x (+0.077%), and complete client wall is 1.000245x (+0.024%). These differences are small and do not support a model-throughput improvement claim.

The route engages only at the existing bounded scope: gfx1151 production Q8_0 dense-wide layers13–47, p1024/p4096 rows, and immediate `attn_qkv→attn_gate` or `shared_gate→shared_up` pairs. Across all measured samples it records 10,980 candidate pairs, 21,960 accepted consumers, and 28,440 casts, versus 39,420 parent casts. P512 remains on the parent with zero pair engagement. The full p4096 prompts execute four 1024-token chunks, so each candidate sample records 244 pairs and eliminates 244 casts.

## Correctness, provenance and decision

- All 168 measured output trajectories are cross-arm exact and deterministic within each arm.
- The source is tracked-clean commit `940f3b5520e45e88b86fc7c155802496af18a05a`; the artifact separately records clean worktree provenance, fixture/model/manifest hashes, command, host and ROCm identity.
- Final teardown reports zero active allocations and zero current allocated bytes.
- The native qualification artifact remains the source for full logits, Conv/GDN state, full KV, repeat bytes, isolation and native admission checks.

The operation-level cast screen remains useful evidence, but its savings do not translate into a measurable complete-engine win in this protocol. DP-2 therefore remains default-off; strict, Q4_K_M, other backends and unqualified shapes retain the parent. No production profile or model-throughput claim follows from this packet.

## Reproduce

```bash
P=/home/lhl/miniforge3/envs/therock10-staging-20260828/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$P/_rocm_sdk_devel/lib:$P/_rocm_sdk_core/lib:$P/_rocm_sdk_libraries/lib
export PATH=$P/_rocm_sdk_devel/bin:$P/_rocm_sdk_core/bin:$PATH
export HIPENGINE_HIP_ARCH=gfx1151
uv run --no-sync python scripts/qwen4exp_f16_pair_engine_ab.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --output /tmp/qwen4exp-dp2-engine-ab-20261006-clean.json
```

The full measured samples are in [engine-ab.json](engine-ab.json), with the compact comparison in [artifact.json](artifact.json). The earlier operation screen, native qualification, partial run and repair history remain in [initial.json](initial.json), [final.json](final.json), [native-qualification.json](native-qualification.json), [focused_p1024.json](focused_p1024.json) and the preceding sections of this directory.
