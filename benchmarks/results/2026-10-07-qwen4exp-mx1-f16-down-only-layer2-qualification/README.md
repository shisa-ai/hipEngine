# MX-1 F16 down-only (`f16_compact_down`) canonical layer2 qualification

This packet records the complete 18-prompt, 24-step qualification of the MX-1
**down-only** candidate against the layer2 numerical/state/task gate. The candidate
is **rejected on the production numerical envelope**: the global top-1 gate and
the binding `prefill_last` / `prefill_to_c1` scope fail. The task axis separately
lands on `requires_task_review` (deterministic cross-route divergences), which is
not the rejection basis.

This is a same-width, single-host, single-model gate. It is not a performance
claim and not a promotion.

## Candidate

MX-1's down-only configuration keeps the Q4_K gate/up leg on the production
exact iu8-risk+repair route and swaps only the Q5_1 down leg to the F16-WMMA
compact grouped owner with no repair pass:

```
HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN=0
HIPENGINE_QWEN4_EXP_Q5_1_WMMA=1
```

- `classification`: T1
- `base_profile`: production
- `candidate_key`: `(hip_gfx1151, moe_linear, gguf_q5_1,
  selected_grouped_wmma_prefill_compact_bf16_bf16_out)`
- `strict_fallback`: `qwen4_exp_q5_1_selected_wmma_iu8_risk_prefill_bf16_bf16_out`
  (registered, unchanged)
- `dispatch_count_mode`: direct_alias (the runner attribute the `q5_wmma` branch
  calls), `candidate_dispatch_calls = 3870`

The candidate manifest records the unmodified **production base**; the two
overrides are applied at run time and are recorded separately in `gate.json`.
The runner's independent gate/up predicate stays on the production route with
these overrides. This packet does not certify a named candidate manifest.

## Command, environment, host, model

```
PY=/home/lhl/miniforge3/envs/therock/bin/python
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib" \
PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH" \
PYTHONPATH=/home/lhl/hipEngine \
HIPENGINE_HIP_ARCH=gfx1151 \
$PY scripts/qwen4exp_layer2_profile_gate.py \
  --candidate f16_compact_down \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --output benchmarks/results/2026-10-07-qwen4exp-mx1-f16-down-only-layer2-qualification/gate.json
```

No `--limit`, no shortened decode/free counts, no custom fixtures or thresholds.

- Source: `e678dadd7fd42b6ad1c12f8ae571bc78ea7d15e9`, tracked-clean.
- Host: `gfx1151`, machine id `55ea6c509d0b49eea8de7094a1023668`, AMD RYZEN AI
  MAX+ 395 / Radeon 8060S (Strix Halo).
- ROCm: TheRock `10.0.0`, Python 3.12; tuned `accelerator-performance`; GPU
  performance level `high`.
- Model: unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, architecture `qwen4exp`.
- Strict manifest sha256: `1fcbf992271251220cc4707eb8d24d28506cd3714550756cf2e32f387d879670`.
- Candidate (production) manifest sha256: `be32e7193f30c4c4a65f5cbaf99b62033fd6458edd295fc2d7806e734001469b`.

## Protocol identity

- Suites: `benchmarks/prompts/mtpbench-code-general-ja.jsonl`
  (`fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a`) plus
  `benchmarks/prompts/gdn-prefill-category-heldouts.jsonl`
  (`8d86154fdb52adf5085dbf278c4097dce78d9510d1a6d04ade29a327560e5280`).
- `complete_prompt_and_heldout_suite: true`, 18 prompts, 450 teacher-forced
  rows (18 `prefill_last` + 432 `c1`/`steady`).
- 24 decode steps, 3 candidate repeats, 32 free tokens, 2 free runs, BF16 KV,
  prefill chunk 512.
- Thresholds: mean 1e-3, p95 5e-3, p99 2e-2, max 5e-2, top-1 0.99, per-scope
  top-1 0.97, review line 0.02.

## Numerical result

Global (450 rows):

| metric | value | limit | verdict |
| --- | --- | --- | --- |
| mean KL | 6.179e-4 | 1e-3 | pass |
| p95 KL | 3.203e-3 | 5e-3 | pass |
| p99 KL | 7.322e-3 | 2e-2 | pass |
| max KL | 1.7177e-2 | 5e-2 | pass |
| top-1 agreement | 0.98667 (444/450) | 0.99 | **FAIL** |

Two independent binding failures:

1. **Global top-1** is below the 0.99 gate (6 deterministic flips, 444/450).
   Every flip has top-k overlap 1.0 and the strict winner at rank 2, i.e. the
   candidate reorders the top pair; `top1_mismatches_outside_flip_eligible = 0`.
2. **Scope failure** on `shape=prefill_last` and `transition=prefill_to_c1`
   (the same 18 rows), which are the only failing scopes:

| scope | rows | mean | p95 | p99 | max | top-1 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prefill_last / prefill_to_c1 | 18 | **2.424e-3** | **1.057e-2** | 1.439e-2 | 1.535e-2 | 1.0 | **FAIL** |
| c1 / steady | 432 | 5.427e-4 | 3.045e-3 | 6.396e-3 | 1.7177e-2 | 0.98611 | pass |
| category code | 150 | 2.319e-4 | 8.301e-4 | 4.932e-3 | 7.455e-3 | 0.98667 | pass |
| category general_en | 100 | 7.654e-4 | 3.296e-3 | 6.593e-3 | 1.7177e-2 | 0.99 | pass |
| category general_ja | 100 | 7.139e-4 | 2.814e-3 | 8.020e-3 | 1.535e-2 | 0.99 | pass |
| category mixed_ja_en | 100 | 9.535e-4 | 4.409e-3 | 6.711e-3 | 9.727e-3 | 0.98 | pass |

The `prefill_last` scope exceeds the mean limit by 2.42x and the p95 limit by
2.11x. No row crosses the 0.02 review line (`rows_over_review_boundary: []`),
and `eligible_for_automatic_admission` is false.

These measurements reject the tested down-only composition. They do not
quantify the down leg's isolated contribution by comparison with the historical
gate/up-only or full-F16 packets, which change other owners and source state.

## State, determinism, lifecycle, task

- `measurement_valid: true`; engagement established (3870 direct-alias calls
  across 12 real down shapes, e.g. `(compact=390..710, experts=512, in=640,
  out=2560)`).
- Repeat determinism: **passed** (3 repeats identical per prompt, no
  mismatches).
- Request-state repeat gate: passed (finite, layout exact, metadata exact,
  repeat exact). Strict/candidate state bytes differ; that difference alone is
  not a production failure. This gate does not test same-width neighbor isolation.
- Lifecycle: passed (both strict and candidate `current_allocated_bytes` 0
  after close).
- Task gate: `requires_review` — 11/18 strict-exact, 7 deterministic
  cross-route divergences (repeat-exact, no nondeterminism). The 32-token outputs
  are prefixes, including incomplete code. No semantic-equivalence or task-quality
  verdict is established by these differences; they are not the numerical
  rejection basis.

## Decision

`status: failed`; `numerical_passed: false`; `state_passed: true`;
`task_passed: false`; `lifecycle_passed: true`;
`next: reject candidate and keep strict fallback`.

The rejection is a binding production-envelope failure (global top-1 and the
`prefill_last`/`prefill_to_c1` scope), not bit-inequality. The registered strict
fallback is retained; no default, kernel, or profile is changed.

## Limits

- One model, one quant, one host, one profile, same-width rows only. No
  isolation, c2, canonical-depth, or performance comparison.
- `performance_claim: false`. The candidate's added per-invocation 16-row map
  build and D2H readback are part of the candidate but are not measured here.
- The raw gate stdout is kept outside the tracked tree at
  `/tmp/mx1-down-gate/qual.log`.
