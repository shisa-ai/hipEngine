# F16 compact gate/up: rejected — and it is the tightest non-exact route measured

**Status: REJECTED** (`20260930T`). Candidate `f16_compact_gate_up` fails the layer2
production envelope. Recorded because it is MX-1's own operand class, because it
**beats every alternative route on every metric** while still failing, and because the
comparison it fails against is one production meets **exactly**.

## Question

`MX-1` is the ledger's largest remaining lever: the routed-expert repair is
**1,219.978 ms = 7.75%** of the 15,741.963 ms prefill union. Its route is "F16 routed
experts with no repair" — Q4_K/Q5_1/Q8_0 decoded to F16, F16 WMMA, F32 accumulate —
and its completed admissibility work predicted the operand class is safe: F16 decode
error sits at the F16 limit, and output error is **8.0x-9.5x below the BF16 floor
already in the path**.

Reconnaissance found the route **already ships**: `gguf_q4_k_selected_dual_wmma_prefill_compact_kernel`
decodes Q4_K to `half16_t` and calls `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` with
F32 accumulate, dual gate+up, over the compact selected-MoE ABI. So this was a
measurement, not a build.

## Configuration

`HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4=0` alone — Q4_IU8_PREFILL already defaults to 0,
and the down leg was left at production so the gate/up half is isolated. Verified
engaged before the run: **360 candidate dispatches** at the real gate/up shapes
(`[390..500, 2560, 640, 640]`), 0 repair calls.

## Result

`measurement_valid: true`; `hard_gates_passed: false`;
`decision.next: "reject candidate and keep strict fallback"`.

**Every aggregate gate passes, and better than any alternative route:**

| metric | **F16 compact** | plain iu8 | limit |
| --- | ---: | ---: | ---: |
| mean KL | **4.5837e-4** | 5.2778e-4 | 1e-3 |
| p95 KL | **2.5307e-3** | 2.8062e-3 | 5e-3 |
| p99 KL | **5.1165e-3** | 5.2674e-3 | 2e-2 |
| max KL | **6.3267e-3** | 2.4688e-2 | 5e-2 |
| top-1 | **0.99778** (449/450) | 0.99556 (448/450) | 0.99 |
| rows over review boundary | **0** | 1 | — |
| top-1 mismatches | **1** | 2 | — |

**The failure is per-scope, on the same 18 `prefill_last` rows — by a whisker:**

| `prefill_last` (18 rows) | **F16 compact** | plain iu8 | limit |
| --- | ---: | ---: | ---: |
| mean KL | **1.127e-3** (1.13x over) | 2.159e-3 (2.16x over) | 1e-3 |
| p95 KL | **5.067e-3** (1.013x over) | 7.776e-3 (1.56x over) | 5e-3 |
| p99 KL | **5.665e-3** ✓ | 2.131e-2 (1.07x over) | 2e-2 |
| max KL | **5.815e-3** ✓ | 2.469e-2 ✓ | 5e-2 |
| top-1 | 1.0 ✓ | 1.0 ✓ | 0.97 |

The 432 `c1`/`steady` rows pass at 4.305e-4 / 2.456e-3 / 4.929e-3.

## What this establishes

1. **MX-1's admissibility prediction is confirmed.** The F16 route is the *tightest*
   non-exact operand class measured here: strictly better than the plain-iu8 route on
   mean, p95, p99, max (3.9x better on max), top-1, review-boundary rows, and top-1
   flips. The 8.0x-9.5x-below-the-BF16-floor prediction shows up as real headroom.
2. **It still fails, and it fails against a bar production meets exactly.** Production's
   iu8-risk route carries the repair precisely so its gate/up arithmetic is
   **bit-identical to the strict parent** — so its deviation from the strict teacher is
   zero by construction, while this route sits at 1.127e-3 mean KL on the binding scope.
3. **So the repair is load-bearing on the prefill path in every configuration tested.**
   Two routes now fail it — plain iu8 (2.16x over) and F16 (1.13x over) — and the F16
   route is the best non-exact one available. Removing the repair is not reachable by
   operand class.
4. **The binding constraint is a scope, not a kernel.** The failure is the 18-row
   `prefill_last` scope at a 1e-3 mean limit, which the ledger's own
   `prefill_last`-row work already flagged as "a scope-and-threshold question for the
   qualification packet, not a kernel question" — 4-18 rows with no averaging to absorb
   one sensitive prompt, against 1536 rows for the same limit in the canonical protocol.
   **This result does not resolve that question and must not be read as a pass.**

## Scope and limits

One model, one quant, one host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`),
one profile, **gate/up only** — the Q5_1 down leg was left at production. The full
`MX-1` route also replaces the down leg with F16, which adds its own error and cannot
reduce this one; the full route is gated separately. Thresholds are the calibrated ones
from `docs/EXECUTION-PROFILES.md`. **Not promoted; no default changed; no kernel
written.**

## Artifacts

- `gate.json` — full gate payload (envelope, per-scope, engagement shapes, host, git).
- Worklog: `worklog/entries/20260930T153800*-qwen4exp-f16-gate-up-result*.md`.
