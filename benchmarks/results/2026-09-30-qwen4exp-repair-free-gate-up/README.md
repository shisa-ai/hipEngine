# Repair-free gate/up: rejected — the repair is load-bearing on the prefill path

**Status: REJECTED** (`20260930T`). Candidate `repair_free_gate_up` fails the layer2
production envelope. Recorded because it closes the cheapest route to the ledger's
largest remaining lever and vindicates `MX-1`'s premise by measurement.

## Question

The ledger's `MX-1` is the largest remaining lever: the routed-expert repair is
**1,219.978 ms = 7.75%** of the 15,741.963 ms prefill union, and the prescribed way to
remove it was a from-scratch F16 kernel port. Reconnaissance found the repair's own
documentation calling it a **bit-parity** mechanism — the risk kernel queues outputs
within a Kahan bound of a *BF16 rounding boundary*, and the production profile marks
each route *"bit-identical to \<parent\>"* — with the multiplier's docstring adding that
under the production contract a rare escape is *"bounded drift, **not a contract
violation**"*.

If that reading held, the whole prize would be available as a **configuration**: select
the plain iu8 owner, which publishes identical arithmetic without the risk criterion or
the repair pass. That is what this run tested.

## Configuration

| setting | value |
| --- | --- |
| `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4` | `0` (clears `exact_grouped_q4_gate`, which the dispatch tests *before* the `q4_iu8` branch) |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL` | `1` (already production, reachable only once the branch above yields) |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS` | `all` (production binds 35-47; the repair route has no layer restriction) |
| Q5_1 down | **left at production**, deliberately: it has no plain non-risk iu8 sibling, so changing it would swap the fast iu8 route for the slower bit-exact owner and confound the measurement |

This isolates the **gate/up** half — 573.211 ms of the repair.

## Result

`measurement_valid: true`; **360 candidate dispatches** at the real gate/up shapes
(`[390..500, 2560, 640, 640]`); `hard_gates_passed: false`;
`decision.next: "reject candidate and keep strict fallback"`.

**Every aggregate gate passes:**

| metric | value | limit |
| --- | ---: | ---: |
| mean KL | 5.278e-4 | 1e-3 |
| p95 KL | 2.806e-3 | 5e-3 |
| p99 KL | 5.267e-3 | 2e-2 |
| max KL | 2.469e-2 | 5e-2 |
| top-1 agreement | 0.99556 | 0.99 |

**The failure is per-scope, concentrated exactly where the repair operated:**

| scope | rows | top-1 | mean KL | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c1` / `steady` | 432 | 0.99537 | 4.598e-4 | 2.765e-3 | 4.358e-3 |
| **`prefill_last` / `prefill_to_c1`** | **18** | 1.00000 | **2.159e-3** ✗ | **7.776e-3** ✗ | **2.131e-2** ✗ |

`scope_failures: [{"dimension": "shape", "value": "prefill_last"}, {"dimension":
"transition", "value": "prefill_to_c1"}]`.

One row exceeds the 0.02 review boundary (kl 0.0247, top-1 equal, top-k overlap 1.0) and
two rows flip top-1 — both at tiny strict margins (0.0284; candidate rank 2) with top-k
overlap 1.0. Hence `eligible_for_automatic_admission: false` and `task_gate:
requires_review` (15/18 exact).

## What this establishes

**The repair is not decorative.** Its docstring's claim that a rare escape is bounded
drift is true **per element** at a 0.71% incidence (`EG-3`), but across 48 layers the
accumulated drift lands on the prefill path and exceeds the calibrated per-scope
envelope. The `prefill_last` rows are precisely those the prefill route produces, and
they are 5x the `steady` rows' mean KL.

**So `MX-1`'s premise is correct after all: removing the repair requires changing the
arithmetic.** Dropping it while keeping iu8 does not pass.

**A planned follow-on is ruled out.** The next step was to be a plain-iu8 sibling of
`q5_1_selected_wmma_iu8_risk_prefill_kernel` for the down leg (661.210 ms). That was
conditional on the gate/up half passing; the same prefill-path drift would be expected,
and it is not to be built on this evidence.

## Determinism and provenance

The run was repeated after fixing the dispatch counter, and the two runs' quality
summaries are **identical to every printed digit** (mean 5.278e-4, p95 2.806e-3, p99
5.267e-3, max 2.469e-2, top-1 0.99556). That is both the run's determinism evidence and
confirmation that the first run was in fact this candidate — the counter, not the route,
was broken.

The first run reported `candidate_dispatch_calls: 0` because the gate's counter replaces
the **registry** entry while the runner calls the plain iu8 owner as a **direct module
attribute** (`qwen4_exp_runner.py:4397`). Fixed with `direct_dispatch_target`; engagement
independently verified (4 calls at `[50, 2560, 640, 640]`, 0 risk, 0 repair) before the
re-run.

## Scope and limits

One model, one quant, one host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`),
one profile, gate/up only — Q5_1 down unchanged at production. The gate's thresholds are
the calibrated ones from `docs/EXECUTION-PROFILES.md`. **Not promoted; no default
changed; no kernel written.**

## Artifacts

- `gate.json` — full gate payload (envelope, per-scope, engagement shapes, host, git).
- Ledger: `MX-1` (the configuration, the numbers, and the consequence) and `EG-3` (its
  "sole path" claim is now measured rather than inferred).
- Worklog: `worklog/entries/20260930T145752*-qwen4exp-repair-free-gate-result*.md`.
