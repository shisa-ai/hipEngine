# The exact-iu8 risk multiplier: measured, gated, and NOT promoted

**Status: not promoted** (`2026-10-04`). Measured on gfx1151 (AMD Radeon 8060S,
Strix Halo APU, machine `55ea6c509d0b49eea8de7094a1023668`), model
`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL` (UD-Q4_K_XL).

Tightening `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` from 4.0 to 1.0 **passes the
production numerical gate with 3.3-6.4x margin** and cuts the repaired share of
rows **3.40x**, but its wall benefit does not survive the canonical protocol, so
the production profile keeps 4.0. Both halves are recorded here because the
numerical result is what makes the remaining lever worth pursuing and the wall
result is what says the multiplier is not where the time is.

## Question

Reattributing the prefill put the routed MoE block at 40.6% of it and, inside that,
a **1370 ms repair pass - 8.6% of the whole prefill**. The pass serves a Kahan-bound
criterion, and `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` sets the bound. Its own
docstring records the default 4.0 as chosen for a ">=4x margin" over a **screened
floor of 0.5-1**, so the pass is deliberately over-inclusive. How much had never
been measured.

## Cost, measured

`scripts/qwen4exp_profile_gap.py --mode prefill --risk-diagnostics`, `code-p4096`,
production profile, 1504 `q4_iu8_exact:expert_gate_up` calls and 9,856,614,400
outputs per point, three wall repetitions each:

| `RISK_MULT` | risk rows | repair rate | max rate | over-capacity | wall (s, 3 reps) | tok/s | vs 4.0 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **4.0** | 70,167,220 | **0.7119%** | 0.8057% | 0 | 17.0430 / 17.0034 / 16.9939 | 240.75 | - |
| 1.0 | 20,621,412 | **0.2092%** | 0.2389% | 0 | 16.6418 / 16.6523 / 16.6723 | 245.92 | **-2.11%** |
| 0.5 | 11,074,348 | **0.1124%** | 0.1281% | 0 | 16.5641 / 16.5645 / 16.5647 | 247.27 | -2.64% |

Within-point spread is 0.2-0.3% at 4.0 and 1.0, so the delta is about 7x its own
noise. `over_capacity_calls` is **0 at every point**: the repair buffer is never the
constraint, so the incidence is the whole story.

**The incidence is linear in the multiplier and the wall is not.** 4.0 -> 1.0 cuts
the flagged share **3.40x** but the wall only **2.11%**; 4.0 -> 0.5 cuts it
**6.33x** for 2.64%. On the recorded 1370 ms share a proportional pass would have
saved 968 ms (5.7%); the measured saving is 358 ms. **The repair kernel therefore
carries a large component that does not scale with `risk_count`** - it is launched
over `risk_capacity = compact * 2 * ffn` regardless. That is a separate, larger and
still-open lever.

## Why it is not promoted

The isolating harness above reads -2.11%. **The canonical twelve-case harness reads
flat**, and that is the protocol that produces the public row, so it is the one that
decides.

Interleaved A/B/A over the four 4K cases, one session, same host, 1 warmup and 3
repetitions per arm (`ab-A-mult1.json`, `ab-B-mult4.json`, `ab-C-mult1.json`):

| arm | mean tok/s | per-category |
| --- | ---: | --- |
| A, multiplier 1.0 | 264.11 | code 260.6-261.7, general_en 265.6-266.8, general_ja 266.5-267.9, mixed 261.5-262.3 |
| B, multiplier 4.0 | 259.46 | code 262.8 / 262.4 / **245.8**, general_en 267.6 / 268.0 / **249.7**, general_ja 268.9 / 269.4 / **252.5**, mixed 263.6 / **255.8** / **247.0** |
| C, multiplier 1.0 | 266.30 (5 of 12 samples) | code 262.6, general_en 267.7, general_ja 269.4 / 268.9, mixed 263.0 |

**Arm B is contaminated and the comparison is inconclusive.** Its third repetitions
degrade 6% progressively (245.8, 249.7, 252.5, 247.0) - a run-order or thermal event
inside the arm, not an arm effect - and its *clean* repetitions read about 0.6%
**above** the 1.0 arms (262.6 / 267.8 / 269.2 / 263.6 against A's 261.1 / 266.4 /
267.2 / 261.9). The same-arm A-against-C drift is 0.82%, which is larger than any
between-arm effect that can be defended from this run.

**The two readings are reconcilable, and the reconciliation is the real finding.**
This change's own measurement says the incidence is linear in the multiplier and the
wall is not: 3.40x fewer flagged rows buys 2.11% where a proportional model over the
recorded 1370 ms repair share predicts 5.7%. **A pass that is mostly fixed cost
cannot yield 3.4x of anything**, so the -2.11% is harness-specific (that harness's
chunking gives the repair a larger proportional share) and the canonical flat read is
the expected one. **The lever is the repair pass's launch geometry, not its
threshold** - it launches over `risk_capacity = compact * 2 * ffn` regardless of
`risk_count`.

An earlier non-interleaved canonical pair also read -0.42% at 4K, but the baseline
arm of that pair had a GTT-contending job start 20 s into it, so it is not usable
either way and is not relied on here.

## Numerical gate

`scripts/qwen4exp_layer2_profile_gate.py --candidate q4_iu8_risk_mult_1`
(classification T2, `base_profile="production"`). 18 prompts across
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` and the
`gdn-prefill-category-heldouts.jsonl` heldouts, **450 rows**, production teacher.
Raw artifact `gate.json` in this directory.

| metric | candidate | limit | margin |
| --- | ---: | ---: | ---: |
| mean KL | **3.039e-4** | 1e-3 | 3.3x |
| p95 KL | **1.785e-3** | 5e-3 | 2.8x |
| p99 KL | **4.271e-3** | 2e-2 | 4.7x |
| max KL | **7.771e-3** | 5e-2 | 6.4x |
| top-1 agreement | **0.99556** | 0.99 | pass |
| per-scope top-1 min | **0.99** | 0.97 | pass |

`scope_failures: null`. Every category passes (worst is `general_ja` at mean
4.509e-4 / max 7.771e-3) and every shape passes. `state_passed: true`,
`lifecycle_passed: true`, determinism exact with no repeat mismatches.

### The scope that rejected the sibling passes here

`repair_free_gate_up` (`2026-09-30`, `../2026-09-30-qwen4exp-repair-free-gate-up/`)
removed the repair entirely and was rejected on **`prefill_last` /
`prefill_to_c1`**, 18 rows. That is the decisive comparison, because it isolates the
prefill path where per-element drift accumulates across 48 layers:

| `prefill_last`, 18 rows | `repair_free_gate_up` (rejected) | **this candidate** | limit |
| --- | ---: | ---: | ---: |
| mean KL | 2.159e-3 ✗ | **7.891e-6** ✓ | 1e-3 |
| p95 KL | 7.776e-3 ✗ | **3.527e-5** ✓ | 5e-3 |
| p99 KL | 2.131e-2 ✗ | **1.047e-4** ✓ | 2e-2 |
| max KL | - | **1.221e-4** ✓ | 5e-2 |
| top-1 | 1.00000 | **1.00000** | 0.97 |

**274x better on mean KL than the rejected candidate at the exact scope that
rejected it**, and a 191x margin on p99. The reason is mechanical: that candidate
repaired **nothing**, while this one still repairs **29% of what 4.0 flags**. The
rejection is not evidence against this change, and reading it as such would have
discarded a measured win on a scope its own evidence never covered.

### Task gate

`requires_task_review`, 15/18 strict-exact, `candidate_repeat_exact: true` with no
repeat mismatches. The three divergences are deterministic and synonym-level:

* `heldout_code_rate_limiter` - the candidate adds a `Deque` import
  (`from typing import Dict, Deque, Tuple` against `Dict, Tuple`), i.e. the
  candidate's code is the more complete one.
* `heldout_general_ja_migration` - the opening phrase ("高負荷時の" against
  "現在、高負荷").
* `heldout_general_ja_speculative` - the same explanation of speculative decoding
  phrased differently ("生成した候補トークンを…一括" against "複数のトークンを予測し…").

`AGENTS.md` states that strict generated-ID equality is not the promotion bar, and
the gate reports `requires_task_review` rather than `failed`. This is the same
outcome the accepted `dp4a logical128` promotion recorded.

## Why the gate result still matters

It is the strongest numerical evidence this route has, and it removes a
misreading that would otherwise be easy to make. `repair_free_gate_up`
(`2026-09-30`) was rejected on `prefill_last`, and a reader could reasonably
conclude that the repair is load-bearing and the whole route is closed. It is not:
that candidate repaired **nothing**, and this one, repairing 29% of what 4.0 flags,
is **274x better than it on the very scope that rejected it**. What is closed is the
threshold as a source of *wall* time, not the route as a source of *numerical
headroom*.

The 0.5 point is untested and sits at the bottom of the screened safe range, where
the multiplier's docstring says BF16 flips begin to escape. It buys a further 0.53%
of the isolating harness's wall (16.564 s against 16.655 s), which the canonical
result above says will not transfer either.

## What the unit tests do and do not show

`tests/test_gpu_qwen4exp_q4_iu8_exact.py` now runs its exactness contract at **both
1.0 and 4.0** on skewed experts and 16-row tile tails (22 tests, all passing). The
sparse exact pass is bit-parity by construction, so this shows the repair still
reproduces the pair2 parent for the rows it serves at 1.0.

It does **not** show the whole output is parent-identical at 1.0 - a tighter bound
queues fewer rows, and the rows it declines keep the plain iu8 value. That is what
the gate measures, and why the gate and not the unit test is the binding evidence.

## Limits

One host, one GPU architecture (gfx1151, not the gfx1100 default hardware), one
model file, one quant, production profile, BF16 KV. **The wall in the cost table is
this harness's prefill wall (17.014 s at the default) and is not comparable to the
canonical-benchmark row** (16.0 s, different harness and warmup protocol); only the
three points within that table are mutually comparable. The 1370 ms repair share is
from a 2026-09-16 capture and is used only to show the wall saving is below a
proportional model. `gate.json` was produced with the production profile still at
4.0, which is what makes its teacher arm the pre-change default; re-running the
candidate now would compare 1.0 against 1.0.

## Reproduce

```bash
# cost
for MULT in 4.0 1.0 0.5; do
  HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT=$MULT python3 scripts/qwen4exp_profile_gap.py \
    --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --mode prefill --case-id code-p4096 --execution-profile production \
    --risk-diagnostics --output /tmp/risk-mult-$MULT.json
done

# numerical gate (run before the promotion, with production still at 4.0)
python3 scripts/qwen4exp_layer2_profile_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --candidate q4_iu8_risk_mult_1 --decode-steps 24 --repeat-runs 3 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --output benchmarks/results/2026-10-04-qwen4exp-q4-iu8-risk-mult-1/gate.json

# unit level, both multipliers
python3 -m pytest tests/test_gpu_qwen4exp_q4_iu8_exact.py -q
```
