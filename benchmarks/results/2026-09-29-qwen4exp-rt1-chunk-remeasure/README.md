# RT-1: prefill chunk 4096 is 4.94% faster than today's 1024 default

**Date:** 2026-09-29
**Host:** Radeon 8060S (gfx1151), machine-id `55ea6c509d0b49eea8de7094a1023668`,
kernel 7.1.6-1-cachyos.
**Model:** Qwen3.8-Flash-Next UD-Q4_K_XL, Q8_0, BF16 KV, production execution
profile, case `code-p4096` (4096 prompt tokens), cached build.
**Artifacts:** `chunk-{1024,2048,4096}.json` (1 rep), `rep3-chunk-{1024,4096}.json`
(3 reps).

## Result

Three interleaved-cold repetitions per arm, same process-per-arm protocol:

| chunk | samples (s) | median (s) | within-arm spread |
|---|---:|---:|---:|
| 1024 (today's default) | 15.642, 15.625, 15.613 | **15.625** | 29.3 ms |
| 4096 | 14.977, 14.854, 14.841 | **14.854** | 135.4 ms |

**−771.9 ms, −4.94%.** The separation between medians is **5.70x** the larger
within-arm spread, and the arms do not overlap (`min(1024) = 15.613 >
max(4096) = 14.977`). So the effect is established at this sample size rather
than read against zero.

A single-repetition screen also covered 2048: 15.665 / 14.957 / 14.761 s for
1024 / 2048 / 4096, i.e. −4.52% and −5.77%. **The 3-rep median supersedes the
−5.77% figure**; the single sample overstated the effect.

## The output is identical

`token_id = 248068` and `logits_sha256 = 7270a972da43ff4642e557b1` on **every**
arm at every repetition, with `fell_back_to_strict = False` and
`named_profile_intact = True`. Chunk size changes how the work is scheduled, not
what it computes — which is what makes it safe to consider at all.

## Cost

| chunk | peak allocation |
|---|---:|
| 1024 | 84,439,702,280 B |
| 2048 | 86,094,246,664 B |
| 4096 | 89,403,335,432 B |

Chunk 4096 costs **+4.96 GB peak (+5.9%)**. That is an admission trade, not a
free win, and it belongs in any decision.

## Protocol

Recovered from the recorded command in
`benchmarks/results/2026-09-28-qwen4exp-current-attribution/artifact.json`, so
all arms share the case, fixture, execution profile and compiler-version file:

```bash
qwen4exp_profile_gap.py --model-root <UD-Q4_K_XL> --mode prefill \
  --case-id code-p4096 --repetitions 3 --execution-profile production \
  --compiler-version-file /tmp/qwen4exp-ledger-20260928-current/hipcc-version.txt \
  --require-cached-build --prefill-chunk-size {1024,4096} --output <path>
```

`--require-cached-build` keeps `hipcc` out of the measured process. One process
per arm, under the benchmark lock.

## Not claimed, and what blocks promotion

- **Not promoted.** No default changed. Two concrete blockers:
  1. The ledger's own note requires re-running the numerical gate at the new
     chunk size (`X-6`). A chunk change moves every row count the gate exercises.
  2. `RT-3` records that short-request costs historically kept 1024 as the
     default, and this measurement is one long case. The right form is a
     **shape-based chunk policy** — 4096 for long prompts, 1024 for short — not
     a blanket default flip.
- **One case, one host, prefill only.** Not a category/heldout qualification and
  not a decode or MTP result.
- The earlier September 15 engine A/B recorded **+3.94%** at a larger chunk.
  That is not contradicted here — different date, and the ledger notes the
  September 14 recovery changed the composition — but this measurement does not
  confirm it either, and is not presented as doing so.

---

# X-6 gate re-run at chunk 4096: passes, and both arms are bit-identical

RT-1's first promotion blocker is the ledger's `X-6` requirement that the
numerical gate be re-run at the new chunk size — and the gate states its own
scope explicitly: *"A complete run at one chunk size never qualifies another
chunk size."* So the chunk-1024 certification does not transfer by default.

Re-run at chunk 4096 for the promoted route at canonical depth:

```bash
execution_profile_q8_wmma_prefill_layers_gate.py --route wmma --layers 16-47 \
  --prefill-chunk-size 4096 --decode-steps 128 --repeat-runs 3 \
  --require-cached-build --output gate-wmma-16-47-chunk4096.json
```

## Result

| | chunk 1024 (recorded) | chunk 4096 (this run) |
|---|---:|---:|
| teacher logits sha256 | `550bb9b8…` | **`550bb9b832bc7ba2…`** |
| candidate logits sha256 | `65370710…` | **`65370710c0b9c40f…`** |
| mean KL | 3.79552e-4 | **3.795524e-4** |
| status | complete | complete |
| `hard_gates_passed` | true | true |

**Both arms reproduce the chunk-1024 certification bit-identically.** Not merely
the mean KL agreeing to six figures — the full trajectories hash the same. So the
route's arithmetic and its drift are both **invariant to the prefill chunk size**
over 1024 → 4096.

Other gates at chunk 4096: `p95` 1.762e-3, `max` 1.530e-2, `top1_rate` 0.99354,
`repeat_deterministic: true`, `canonical_numerical_qualification_claim: true`,
teacher rows 1548, chunk recorded as 4096.

## This is not the false-pass mode

The gate previously returned **KL exactly 0.0 with `measurement_valid: true`** on
a route that never ran, so a clean pass is the failure to rule out *first*. Here
the engagement is recorded, not inferred:

- `candidate_route_engaged: true`, `teacher_route_engaged: false`
- `wmma_prefill` candidate engaged rate **0.373585** against the teacher's **0.0**
- **792** launches of `wmma_prefill_f32_f32_out`
- `selectors_observed_in_one_arm_only: []` — the attribution rests on both arms
  being observed
- `selectors_differing_between_arms: ["q4k_rowtile", "wmma_prefill"]`

The KL is nonzero, the arms differ, the teacher is genuinely the cleared exact
chain (`teacher_chain: exact coltile8_rowbatch4_f32_f32_out`), and the hashes
match a certification taken at a different chunk size in a different session.

## What this does and does not clear

**Clears** RT-1's first blocker for the promoted route: the chunk-4096
configuration passes the canonical numerical gate and reproduces the chunk-1024
result bit-identically, so the chunk change does not perturb this route.

**Does not clear** RT-1's second blocker (`RT-3`: short-request behaviour and the
+4.96 GB allocation), and does not clear `X-7`'s remaining profile blockers,
which are unchanged and reported by the gate itself: category-heldout,
strict-teacher/BF16-relative, control/isolation/graph/lifecycle, task-quality,
and engagement/repair-oracle coverage. Those are pre-existing and are not
created or worsened by the chunk change.

**Still not promoted.** No default changed.
