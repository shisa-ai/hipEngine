# DE-1 gate: decode Q8_1-DP4A linear (`decode_dp4a_linear`)

Full production profile gate for restoring the Q8_1-DP4A `logical128/t64` decode gate+up on the
calibrated safe-43 layers. Correctness evidence; `perf_claim: false`.

- **Gate:** `scripts/qwen4exp_layer2_profile_gate.py --candidate decode_dp4a_linear`
- **Host:** gfx1151, machine_id `55ea6c509d0b49eea8de7094a1023668`
- **Base commit:** `bf9dca7f4` (tree clean during measurement, `measurement_valid: true`)
- **Candidate kernel:** `gguf_q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out`
- **Strict fallback:** `gguf_q4_k_selected_dual_silu_logical128_t64_gemv_bf16_bf16_out`
- **Suite:** 18 prompts, `complete_prompt_and_heldout_suite: true` (mtpbench-code-general-ja +
  gdn-prefill-category-heldouts), 450 teacher-forced rows, 24 decode steps, 3 candidate repeats,
  32 free tokens

`bound_environment` records `HIPENGINE_QWEN4_EXP_Q4_DP4A64 = "0"` - the binder's post-recovery value
that the candidate then overrides to `"1"`. This is direct evidence of the mechanism: the flag is
set to `"1"` for production and then zeroed by the blanket `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`
sweep, which contains `Q4_DP4A64`.

## Verdict

| gate | verdict |
|---|---|
| `numerical_passed` | **true** |
| `state_passed` | **true** |
| `lifecycle_passed` | **true** |
| `task_passed` | false (13/18 strict-exact) |
| `passed` | false |
| `status` | `requires_task_review` |

## Numerical envelope

| metric | measured | threshold | margin |
|---|---:|---:|---:|
| mean KL | **2.4387e-4** | 1e-3 | **4.1x** |
| p95 KL | 1.2413e-3 | 5e-3 | 4.0x |
| p99 KL | 3.6676e-3 | 2e-2 | 5.5x |
| max KL | 6.1393e-3 | 5e-2 | 8.1x |
| top-1 agreement | **0.99111** (446/450) | 0.99 | passes |

By category: `code` 150 rows, mean KL 8.28e-5, top-1 0.9867, `passed: true`; `general_en` 100 rows,
mean KL 2.98e-4, top-1 1.0, `passed: true`.

Flip accounting: `flip_eligible_rows = 33` (7.33%), `top1_mismatches_flip_eligible = 4`,
**`top1_mismatches_outside_flip_eligible = 0`**. All four top-1 mismatches are on rows whose strict
margin was already small enough to flip. `teacher_nll_delta = 6.08e-4`,
`topk_overlap_mean = 0.9849`.

Mean KL **2.44e-4** reproduces the route's own 2026-08-29 `accepted_explicit_production_profile`
value of **2.72e-4** to within 11%.

## Determinism and isolation

`state_repeat_gate.passed = true`, `candidate_repeat_exact: true`, `repeat_mismatches: []`, with
`repeat_exact: true` on all 18 prompts. Lifecycle: both arms return to 0 current bytes and 0 active
allocations; peak 83.430 GB (strict) vs 83.455 GB (candidate).

## The five divergences

`task_passed` is strict generated-ID equality. All five divergences are deterministic cross-route
prefix differences at `free_tokens = 32`:

| prompt | category | divergence |
|---|---|---|
| `general_en_explain` | general_en | "autoregressive **decoding** process" -> "**generation** process" |
| `mixed_ja_en_review` | mixed_ja_en | "due to increased KV cache usage" -> "as KV cache usage increases" |
| `heldout_code_rate_limiter` | code | `import Dict, Deque` -> `import Dict, Deque, Tuple` (superset) |
| `heldout_general_ja_speculative` | general_ja | Japanese paraphrase, same meaning |
| `code_topological_sort` | code | strict opens `from collections import deque`; candidate opens `def topo_sort` |

Four are provably benign. The fifth differs structurally; at token 32 the candidate is still inside
its docstring, so whether it needs `deque` is not determinable from this artifact.

`docs/EXECUTION-PROFILES.md` section 6.5 requires "no material task-level regression versus strict
under a predeclared paired criterion" and task-specific checks such as code execution - not strict
generated-ID equality. `AGENTS.md` states that production-profile arithmetic wins "are not required
to match strict generated IDs". The gate reflects this: it reports `requires_task_review` rather
than `failed` when numerical passes and the task gate diverges.

**No functional criterion has been applied yet.** `free_tokens = 32` yields prefixes too short to
execute. A follow-up run with `--free-tokens 256` collects executable outputs for the section 6.5
criterion; this artifact's task verdict is therefore open, not settled.
