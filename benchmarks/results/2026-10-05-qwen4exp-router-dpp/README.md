# Ordered router DPP: production prefill route

On Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`, Qwen3.8-Flash-Next UD-Q4_K_XL,
BF16 KV and chunk4096, the ordered router DPP route improves synchronized
full-prefill wall across all 12 canonical cases. One process and one model
residency, one warmup per arm per case, three counterbalanced measurements
per arm, 128 decode transitions:

| Prompt tokens | Parent tok/s | DPP tok/s | Prefill change | Mean wall saved/request |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 287.913 | 289.505 | +0.553% | 9.779 ms |
| 1024 | 312.834 | 314.353 | +0.486% | 15.820 ms |
| 4096 | 319.356 | 321.238 | +0.589% | 75.140 ms |

These are weighted token rates from complete synchronized direct-runner
prefill, including reset and the first greedy output. All four categories
and both case-order strata improve. All 72 measured output trajectories are
exact and deterministic; DPP engagement is 48 calls per candidate sample,
zero per parent sample. Device allocations return to zero after close.
The gfx1151 UD production plan selects DPP by default; strict and Q4_K_M
production keep tile4. The exact tile4 parent remains registered as fallback
and temporary rollback.

The original `code-p4096` final DPP decode sample drops to 14.460 tok/s while
its other candidate samples are 18.403/18.407 and its parent samples are
18.430/18.458/18.400. Consequently the original complete-request aggregate
at p4096 is 0.364% worse and decode is 1.495% worse. Preserve this observation;
it is not excluded from the artifact. The state gate proves zero DPP decode
calls, but that alone does not establish a causal explanation.

A focused [code-p4096 recheck](code-p4096-recheck.json), with the same clean
source, model and three-pair protocol, does not reproduce the slowdown:
weighted decode is 18.311 vs 18.326 tok/s (+0.079%), prefill is 324.547 vs
326.467 tok/s (+0.592%), and complete request wall is 0.407% lower.
All six output trajectories match, candidate engagement is 0/48 per arm
and teardown is leak-free. This is a diagnostic recheck, not a replacement
for the full-suite samples; the original slowdown's cause is unresolved.
No decode-speed or all-case complete-request non-regression claim is made.

[Engine samples](engine-ab.json) preserve timing/order/output hashes and
engagement. [State gate](state-gate.json) preserves every natural-router and
full-state hash. Public-serving, native-context, heldout task and independent
c2 certification are not asserted by this exact implementation substitution.

## Ordered router DPP operation screen

On Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`, Qwen3.8-Flash-Next UD-Q4_K_XL
real F32 router weights at layers 0/22/47 and synthetic F32 activations,
the tile4 DPP candidate improves complete logits-plus-top10-selection wall
in all nine cases. Twenty counterbalanced pairs per case:

| Rows | Parent ms | DPP ms | Wall speedup |
| ---: | ---: | ---: | ---: |
| 512 | 1.558–1.561 | 1.335–1.336 | 1.167–1.169x |
| 1024 | 3.062–3.078 | 2.615–2.626 | 1.167–1.172x |
| 4096 | 12.970–12.974 | 11.410–11.423 | 1.135–1.137x |

Both first-arm subsets improve for every case. Logits, selected expert IDs
and routing weights match the parent bit-for-bit. Zero allocations/bytes
remain after close. These are operation results, not model throughput.
Natural-input and full-canonical state checks subsequently pass: 12 cases,
1,548 full-vocabulary rows per arm through D128, exact recurrent/layout/full-KV
hashes, and 576 private natural router calls (1,081,344 rows across 48 layers).
Parent/candidate/repeat logits, int64 expert IDs and routing weights are raw-bit
exact. Candidate engagement is 0/48/0 in parent/DPP/parent arms and zero during
decode. Four GPU ownership tests also pass for row-neighbor substitution,
row permutation, reused buffers, unchanged inputs and output canaries.
These model checks qualify the T0 substitution against the live production
parent; they do not claim a new production numerical budget.

## Implementation and correctness

`f32_hidden_token_tile4_dpp_exact` preserves the parent's 256-thread FMA
ownership and reduction association. After one LDS publication barrier,
wave 0 folds strides 128/64/32 in registers and finishes 16/8/4/2/1 through
`permlanex16`/DPP instructions. The strict tile4 parent remains registered.

This is not the removed September 7 shuffle-tail candidate. That candidate
retained two inter-wave reduction barriers and finished 32..1 with a shuffle
sequence; it passed exactness but failed a consistent model-wall gain.
The September 27 warp-per-token candidate changed association and failed
the production envelope. Neither is restored unchanged.

The new RED initially fails because the wrapper/key is absent. Thirty-two
GPU tests then pass, including the existing router regressions. The new
fixtures cover tokens 1/3/5/7/17/513/1024/4096, K 1/255/2047/2048/2560/4097,
non-power-of-two expert counts, duplicated expert ties, cancellation and zero
inputs, repeat raw-bit logits/route weights and exact expert IDs. CPU-reference
`linear` comparison meets KL≤0.05 and top1≥90% on random fixtures.
A 10,000-vector host simulation confirms the register-fold tree's output bits
and demonstrates that an alternate wave-sum association would differ.

Fifty-four focused CPU/registry/profile/reference tests pass. An expanded
bundle has 109 passes and the known unrelated
`test_gfx1151_backend_aliases_gfx1100_kernel_keys` stale `pair32` expectation,
already reproduced at a clean baseline before this unit. No all-green CPU
suite claim; no repair to that unrelated file. Fixture, registry/CPU smoke,
compilation and benchmark-export checks pass.

## Trace and reproduction

Cached-only `rocprofv3 --kernel-trace` shows the DPP producer at
11.170–11.250 ms versus parent 12.781–12.859 ms, with 32 allocated VGPR,
scratch 0 and 4096 B requested dynamic LDS in both arms. The static LDS 0 field
does not include dynamic allocation. Code-object census has 29 VGPR for both;
static instructions 249→251, but parent has one publication plus eight
**dynamic loop** barriers versus the candidate's one publication barrier.
Static instruction count is not dynamic work or an occupancy counter.

```bash
uv run --no-sync python scripts/qwen4exp_router_dpp_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt \
  --require-cached-build --output /tmp/qwen4exp-router-dpp-screen.json
uv run --no-sync pytest tests/test_gpu_qwen4exp_router_dpp.py \
  tests/test_gpu_qwen35_router_plan.py -q --suite gpu
```

Use the ROCm environment in
[the worklog](../../../worklog/entries/20261005T232354.215863Z-pi-qwen4exp-router-ordered-dpp-0d1744.md).
The model path, source hashes, exact commands, 180 timed pairs, both order
subsets, output hashes, library hash, cached trace durations/resources and
static census are in [screen.json](screen.json). Raw logs/CSV/disassembly
remain in `/tmp`.

## Model gate reproduction

Use the ROCm environment from the linked operation-screen worklog and run with
cached builds from this repository's `uv` environment:

```bash
uv run --no-sync python scripts/qwen4exp_row4_state_gate.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt \
  --route-package router-dpp --prefill-chunk-size 4096 \
  --all-cases --decode-steps 128 --full-kv \
  --output /tmp/qwen4exp-router-dpp-qualification/state-gate.json
flock -n /tmp/hipengine-gfx1151-benchmark.lock \
  uv run --no-sync python scripts/qwen4exp_halo_box_campaign_ab.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt \
  --require-cached-build --route-package router-dpp --prefill-chunk-size 4096 \
  --warmups-per-mode 1 --repetitions-per-mode 3 \
  --output /tmp/qwen4exp-router-dpp-qualification/campaign-ab.json
uv run --no-sync pytest tests/test_gpu_qwen4exp_router_dpp_isolation.py -q --suite gpu
```

The gate and original full-suite benchmark use tracked-clean `ef927ba05`.
Their manifests record the parent production plan
`850829d509ebf741403e71e30450b8b7b681994f16e0f02038ba672c1f493d5b`
plus explicit counted post-binder DPP overrides. Fresh-process named DPP
resolution produces production manifest
`52865d16d4a985549cd90fe8c69a5a46b6eb4543eae17a46df46c79a70be7b89`;
strict stays
`1335b8244237ffb0cac0ec9e3eb670d17ebd9d8aca607eebb463ca291b9cfbed`.
The benchmark does not install the private natural-input hook. Raw packets
and their SHA256 values are referenced by the compact JSON files.

The post-promotion CPU route/profile/gate bundle has 117 passes, including
fresh-process default, parent rollback and diagnostic override evidence binding.
The latest full reviewed CPU tier has five pre-existing baseline failures recorded in
`worklog/entries/20261005T224525.619135Z-pi-qwen4exp-dense-split-runtime-removal-ab3694.md`;
no all-green broad-suite claim is made. GPU execution on gfx1100 is unverified.
