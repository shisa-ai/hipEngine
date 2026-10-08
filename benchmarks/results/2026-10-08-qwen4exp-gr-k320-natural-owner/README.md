# GR-up K320/full-row DPP qualification and promotion

This packet records the complete qualification of the separately registered
K320/H2560/four-branch complete-rowbatch4 GR-up DPP sibling
(`coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact`) and its
promotion to the production default for the qualified lane (Qwen3.8-Flash-Next
UD-Q4_K_XL, production profile). Physical host: Radeon 8060S/gfx1151,
`55ea6c509d0b49eea8de7094a1023668`.

The sibling is an exact-class narrowing of the promoted GR-up DPP owner
(`coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact`): it removes only
compile-time-provable generic row-validity and variable-K control for the
K=320/H=2560/four-branch shape with complete rowbatch-4 groups. The generic DPP
owner remains the registered strict fallback and the owner of every other
shape. `HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS=0` is the bisection opt-out.

## Stage 1 — synthetic operation screen (2026-10-06)

12 real-weight complete fused GR-up cases at rows 512/1024/4096, 20
counterbalanced pairs per case: **1.084–1.103x** operation speedup, geometric
mean **1.092x**, gate and mixed outputs bit-identical.
[Screen packet](../2026-10-06-qwen4exp-gr-k320-full-rows-screen.md).

## Stage 2 — natural-input owner gate

A registry hook ran the parent into the model-owned gate/mixed outputs and the
candidate into private buffers on identical natural activations, normalized
inputs and resident raw weights, through six canonical prefill cases
(chunk 4096, HIP-event timing on the launch stream):

| Case | Captures | All outputs exact | Owner speedup |
| --- | ---: | --- | ---: |
| code-p512 | 96 | yes | **1.0925x** |
| code-p1024 | 96 | yes | **1.0958x** |
| code-p4096 | 96 | yes | **1.1046x** |
| general_en-p4096 | 96 | yes | **1.1054x** |
| general_ja-p4096 | 96 | yes | **1.1062x** |
| mixed_ja_en-p4096 | 96 | yes | **1.1050x** |

576/576 captures exact (both materialized outputs bit-for-bit); activation,
normalized input and raw weight immutable in every capture; zero allocations
after teardown. [natural-owner.json](natural-owner.json).

## Stage 3 — full model-state gate

All 12 canonical cases, parent/candidate/candidate/candidate/parent arms
toggled through the runner's own default-off selector (the production
selection path is under test), 16 decode steps, full KV hashes:
**3,456 candidate calls**, exact prefill and decode logits, exact
recurrent/layout state, exact full KV bytes and token IDs, engagement in every
enabled arm, zero teardown allocations. [state-gate.json](state-gate.json).

## Stage 4 — complete-engine A/B

One model residency, all 12 canonical cases, production compact-capture path,
one warmup plus three measured repetitions per arm with balanced arm order:

| Metric | Before (K320 off) | After (K320 on) | Ratio |
| --- | ---: | ---: | ---: |
| Prefill | 339.258 tok/s | **343.010 tok/s** | **1.0111x** |
| Decode | 16.128 tok/s | 16.074 tok/s | 0.9967x |
| Aggregate client wall | 484.934 s | **483.709 s** | **0.9975x** |

4,608 candidate calls; all 12 trajectories cross-mode exact and within-mode
deterministic; zero teardown allocations. The decode ratio is arm noise: both
arms launch the bit-identical parent kernel at decode (rows=1 is outside the
admission window), and the same-residency instrument's decode noise band is
about ±1% (the dense Q8 A/B measured +1.22% decode on identical decode
kernels). [engine-ab.json](engine-ab.json).

## Promotion

The production profile binding
(`hipengine/generation/qwen4_exp_profiles.py`) now binds
`HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS=1` for the qualified lane (production,
`gguf_ud_q4_k_xl`); strict and every other lane keep the generic DPP owner.
This is a retained small-win promotion: exact-class, +1.11% measured prefill
on the same-residency instrument, −0.25% aggregate client wall, mechanistically
consistent with the owner-level 9.3–10.6% GR-up family saving (~1.5 s of owner
time per p4096 prefill).

## Reproduction

```bash
export ROCMENV=/home/lhl/miniforge3/envs/therock10-staging-20260828
export ROCMLIB=$ROCMENV/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH="$ROCMENV/bin:$ROCMLIB/bin:$PATH"
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib"
export HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=/home/lhl/hipEngine
uv run python -u scripts/qwen4exp_gr_k320_full_rows_natural_owner.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --output benchmarks/results/2026-10-08-qwen4exp-gr-k320-natural-owner/natural-owner.json
uv run python -u scripts/qwen4exp_gr_k320_full_rows_state_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt --decode-steps 16 --full-kv \
  --output benchmarks/results/2026-10-08-qwen4exp-gr-k320-natural-owner/state-gate.json
uv run python -u scripts/qwen4exp_gr_k320_full_rows_engine_ab.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt --require-cached-build \
  --output benchmarks/results/2026-10-08-qwen4exp-gr-k320-natural-owner/engine-ab.json
```
