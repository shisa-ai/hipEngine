# tiled16_multi GDN prefill: full production-envelope admission and promotion

This packet records the admission of the registered `qwen4exp_gdn_tiled16_multi_prefill`
kernel (`gdn_recurrence_norm_gate` / `f32_state`) as the production default for
its certified suffix window on the Qwen3.8-Flash-Next UD-Q4_K_XL lane, replacing
the recovery-gated `qwen4exp_sigmoid_wave_norm_prefill` route that ran there
since the arithmetic-recovery profile froze the column-warp gate. The kernel
reassociates the gated-delta recurrence over multi-column tiles at fp32
round-off (NMSE 1.8e-14 per the GD-4 drift screens) and was measured by the
GD-4 session at **1.39-2.39x faster than the production wave_norm route at a
third of the registers** (85 vs 256 VGPR). It was gated off on this lane solely
for non-bit-exactness - a bar the production-correctness contract does not
impose.

## Stage 1 — engagement (bab72d129)

Through-the-runner engagement confirmed on the UD lane under the two runtime
env overrides: 15 multi calls per prefill at p512/p1024/p4096, trajectory heads
matching the canonical fixture, the multi-column library built and cache-warm.
[engagement-probe.json](engagement-probe.json).

## Stage 2 — layer2 production gate: PASS on the full envelope

Full protocol (18 prompts, 450 strict-teacher rows, 24 decode steps, 3 repeats,
2 free 32-token runs; source `084095c52` tracked-clean):

| Gate | Value | Limit | Margin |
| --- | ---: | ---: | ---: |
| mean KL | 3.454e-4 | 1e-3 | 2.9x |
| p95 KL | 1.599e-3 | 5e-3 | 3.1x |
| p99 KL | 5.180e-3 | 2e-2 | 3.9x |
| max KL | 1.086e-2 | 5e-2 | 4.6x |
| top-1 agreement | 0.99333 (447/450) | 0.99 | pass |

**The binding risk scope passes decisively**: `prefill_to_c1` (the
`prefill_last` scope where the F16 operand-class candidates failed at
1.13-1.85x over) runs at mean 4.149e-4 (2.4x margin), p95 1.368e-3 (3.7x), and
**top-1 agreement 1.0 - 18/18 perfect**. `steady`: 432 rows at mean 3.43e-4,
top-1 99.31%, all 3 mismatches flip-eligible, zero outside. Zero rows over the
0.02 review boundary; `scope_failures: []`; `hard_gates_passed: true`;
`eligible_for_automatic_admission: true`. 1,350 candidate dispatch calls;
state-repeat, lifecycle and measurement validity all pass. The task axis is
`requires_review` (14/18 free-run strict-exact, deterministic divergences) -
the same class the K320 promotion carried, not a rejection basis; recorded for
the human owner. [gate.json](gate.json).

This is the first arithmetic-changing candidate to pass the full production
envelope on this lane where the F16 class failed - the reassociation sits at
fp32 round-off rather than an operand-class change.

## Stage 3 — same-residency engine A/B

One residency, 12 canonical cases, compact capture, balanced arms,
counting-hook engagement (45 multi calls per case = 15 x 3 reps):

| Metric | wave_norm (before) | multi (after) | Ratio |
| --- | ---: | ---: | ---: |
| Prefill | 340.958 tok/s | **343.291 tok/s** | **1.0068x** |
| Decode | 16.326 tok/s | 16.424 tok/s | 1.0060x |
| Aggregate client wall | 480.465 s | **477.438 s** | **0.9937x** |

All 12 cases positive on prefill (+0.48% to +1.38%); within-mode deterministic;
clean teardown. [engine-ab.json](engine-ab.json).

## Promotion

- `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS` no longer zeroes
  `GDN_COLWARPS_PREFILL` for the UD lane (the gate opens for production).
- `HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT` binds the multi variant for every
  production quant (the `!= gguf_ud_q4_k_xl` exclusion removed).
- The UD manifest's 27-47 GDN row declares the multi variant with this gate as
  evidence; the multi entry is removed from the recovery `inactive` set so the
  declaration survives.
- Opt-out: `HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0` (the selector's existing
  escape hatch); the columnwarps/wave_norm/serial chain stays registered as
  the runtime fallback.
