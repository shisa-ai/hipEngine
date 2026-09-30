# MX-1's full F16 route: rejected — and it is worse than its own gate/up half

**Status: REJECTED** (`20260930T`). Candidate `mx1_full_route` — MX-1 as the ledger
specifies it — fails the layer2 production envelope, and fails it **worse than the
gate/up half alone**. This closes the ledger's largest remaining lever as a performance
opportunity.

## Question

`MX-1` prices the routed-expert repair at **1,219.978 ms = 7.75%** of the 15,741.963 ms
prefill union, and its route is "F16 routed experts with **no repair**": Q4_K/Q5_1/Q8_0
decoded to F16, F16 WMMA, F32 accumulate, gate/up paired, **down writes F16**. Its
completed admissibility work predicted the operand class is safe (F16 decode error at
the F16 limit; output error 8.0x-9.5x below the BF16 floor already in the path).

Reconnaissance found **both halves already ship** as F16 WMMA selected-MoE prefill
kernels, so the row's prescribed build was unnecessary:

| half | owner | intrinsic |
| --- | --- | --- |
| gate/up (Q4_K) | `gguf_q4_k_selected_dual_wmma_prefill_compact` | `wmma_f32_16x16x16_f16_w32` |
| down (Q5_1) | `qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact` | `wmma_f32_16x16x16_f16_w32` |

## Configuration

Three flags, because the down dispatch tests `exact_grouped_down` **before** `q5_wmma`:

```
4559  if down == gguf_q5_1
4571    if q5_mmq:
4603    elif exact_grouped_down:   ← production chain (exact row_publish + repair)
4751    elif q5_wmma:              ← F16 down owner
```

`HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4=0` + `EXACT_GROUPED_DOWN=0` + `Q5_1_WMMA=1`.

Engagement verified before the run: **47 F16 gate/up calls + 43 F16 down calls, zero
repair calls on either half.** The counts are not 48 each because layer 2's gate/up is
Q5_K and 5 down layers are Q8_0 — matching the ledger's tensor census (94 Q4_K / 43
Q5_1 / 2 Q5_K / 5 Q8_0 = 144).

## Result

`measurement_valid: true`; `candidate_dispatch_calls: 4230`;
`hard_gates_passed: false`; `decision.next: "reject candidate and keep strict fallback"`.

**Worse than the gate/up-only variant on every metric:**

| metric | **full MX-1 route** | gate/up-only F16 | plain iu8 | limit |
| --- | ---: | ---: | ---: | ---: |
| mean KL | **7.288e-4** | 4.584e-4 | 5.278e-4 | 1e-3 |
| p95 | **3.4452e-3** | 2.5307e-3 | 2.8062e-3 | 5e-3 |
| p99 | **1.1172e-2** | 5.1165e-3 | 5.2674e-3 | 2e-2 |
| max | **1.8181e-2** | 6.3267e-3 | 2.4688e-2 | 5e-2 |
| top-1 | **0.99333** (447/450) | 0.99778 (449) | 0.99556 (448) | 0.99 |
| **failing scopes** | **4** | 2 | 2 | — |

The failing scopes are `general_ja`, `mixed_ja_en`, `prefill_last` and
`prefill_to_c1` — the gate/up-only variant failed only the last two.

**The binding scope, `prefill_last` (18 rows):**

| metric | **full route** | gate/up-only | limit |
| --- | ---: | ---: | ---: |
| mean KL | **1.848e-3** (1.85x over) | 1.127e-3 (1.13x over) | 1e-3 |
| p95 | **7.716e-3** (1.54x over) | 5.067e-3 (1.013x over) | 5e-3 |
| p99 | 8.839e-3 ✓ | 5.665e-3 ✓ | 2e-2 |
| max | 9.119e-3 ✓ | 5.815e-3 ✓ | 5e-2 |
| top-1 | 1.0 ✓ | 1.0 ✓ | 0.97 |

Zero rows over the 0.02 review boundary; three top-1 flips.

## What this establishes

**Adding the F16 down roughly doubles the drift** — mean 4.584e-4 → 7.288e-4, p99
5.1165e-3 → 1.1172e-2, top-1 449 → 447 matches, failing scopes 2 → 4. So the F16 down
contributes error on the same order as the F16 gate/up, and the two compound.

**MX-1 is therefore refuted.** The row's full route:

1. fails the calibrated envelope;
2. fails it **worse than its own gate/up half**, so the down half cannot rescue it;
3. fails it against a bar **production meets exactly** — the iu8-risk route carries the
   repair precisely so its arithmetic is bit-identical to the strict parent, i.e. zero
   deviation by construction.

Together with the earlier `repair_free_gate_up` result (plain iu8, no repair: 2.16x over
on `prefill_last` mean), **two distinct operand classes have now been measured and both
fail.** The F16 class is the tighter of the two, and it still cannot remove the repair.

**The repair is load-bearing on the prefill path, and the 1,219.978 ms is not
recoverable by operand class.** Removing it is not reachable by configuration.

## Scope and limits

One model, one quant, one host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`),
one profile. Thresholds are the calibrated ones from `docs/EXECUTION-PROFILES.md`.

The `prefill_last` scope's 1e-3 mean limit remains the binding constraint, and the
ledger's own `prefill_last`-row work already flagged that as "a scope-and-threshold
question for the qualification packet, not a kernel question". **This result does not
resolve that question, and it is not a pass under the current envelope.** If that
threshold question were ever decided differently, the gate/up-only F16 variant at 1.13x
over would be the nearest candidate — but it would still need the down leg left exact,
which keeps its 522.992 ms of repair.

**Not promoted; no kernel written; no default changed.**

## Artifacts

- `gate.json` — full gate payload (envelope, per-scope, engagement, host, git).
- Sibling: `../2026-09-30-qwen4exp-f16-compact-gate-up/` — the gate/up-only variant.
- Worklog: `worklog/entries/20260930T153959*-qwen4exp-mx1-full-route-rejected*.md`.
