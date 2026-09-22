---
status: current
owns: The zbook / Radeon 8060S (gfx1151) UD-versus-plain Q4_K_M parity hypotheses, the paired A/B protocol that adjudicates them, and the two recorded root-cause fixes this campaign executes
---
# UD gfx1151 (zbook / Radeon 8060S) Optimization Plan 2

Created: 2026-09-23. Host lane: `zbook`, AMD Radeon 8060S, `hip_gfx1151`,
HIP 7.15.26333. Model pair: `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` (plain)
against `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` (UD).

**Naming.** [`UD-GFX1151-OPTIMIZE.md`](UD-GFX1151-OPTIMIZE.md) tracks the
RX 7900 XTX (`gfx1100`) lane; its filename is kept only for existing links and
does not identify the measured backend. This document is the campaign for the
actual gfx1151 lane. It supersedes neither that campaign nor
[`UD-OPTIMIZED-ROUTE-PLAN.md`](UD-OPTIMIZED-ROUTE-PLAN.md); it executes the
two fixes those campaigns recorded but did not land on this backend.

**Objective.** Close the UD-versus-plain Q4_K_M throughput gap on zbook
without regressing absolute UD rates or its production-profile quality:

- Decode: move `UD decode / plain decode` from **0.675x** toward 1.0x.
- Prefill: move `UD prefill / plain prefill` from **0.59-0.63x** toward 1.0x.
- Quality: every promoted route stays inside the calibrated production
  envelope of [`EXECUTION-PROFILES.md`](../EXECUTION-PROFILES.md) §6.1
  (mean/p95/p99/max KL ≤ `1e-3`/`5e-3`/`2e-2`/`5e-2`, top-1 ≥ 99% overall /
  97% per scope) against the incumbent production default.
- An absolute regression disqualifies a ratio win. Both arms move together
  under thermal conditions, so every adjudication is a same-window paired A/B.

## 1. Opening evidence: the gap is verified and still present

Two paired measurements, both arms in one thermal window each, same harness
and protocol:

| Run | Commit | Decode Δ (512 / 1024 / 4096) | Prefill Δ (512 / 1024 / 4096) |
| --- | --- | --- | --- |
| 2026-09-21 | `d3a5ccf02` | −32.30% / −32.15% / −33.15% | −37.59% / −42.08% / −43.46% |
| 2026-09-23 re-check | `91f56a7c8` | −32.48% / −31.85% / −32.53% | −36.65% / −40.71% / −40.46% |

Absolute medians from the 2026-09-23 re-check (tok/s; plain → UD):

| Shape | Plain prefill | UD prefill | Plain decode | UD decode |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 296.433 | 187.780 | 11.5833 | 7.8209 |
| 1024/128 | 296.940 | 176.050 | 11.3081 | 7.7060 |
| 4096/128 | 283.376 | 168.715 | 11.5020 | 7.7609 |

The re-check ran at current HEAD on an idle GPU; every per-shape CV is at or
below 0.62% (decode at or below 0.13%) and both arms pass the harness's
18/18 graph/eager id, logits and state gate. Absolute rates sit 4-5% below the
2026-09-21 pair (day-to-day spread on this power-limited host; the plain arm
lands at its recorded 294.8 tok/s floor) while the ratios reproduce within
spread.

Artifacts:

- [`benchmarks/results/2026-09-21-gfx1151-qwen38-ud-vs-plain-q4km-baseline.json`](../../benchmarks/results/2026-09-21-gfx1151-qwen38-ud-vs-plain-q4km-baseline.json)
- [`benchmarks/results/2026-09-23-zbook-ud-vs-plain-q4km-head-recheck.json`](../../benchmarks/results/2026-09-23-zbook-ud-vs-plain-q4km-head-recheck.json)

Both are `performance_claim: false` diagnostics: neither arm carries the
profile-quality and serving gates a published row requires. These are
**zbook-lane** rates; the published plain row (404.474 / 12.150) belongs to
the separate desktop `gfx1151` machine and is not a baseline here
([`docs/OPTIMIZATION.md`](../OPTIMIZATION.md) §2).

### Comparison history

The same-host plain-to-UD prefill ratio has moved from about **11.7x**
(2026-09-08, `2026-09-08-zbook-ud-prefill-route-gap.json`) to about **1.60x**
through the closed route campaign, so the prefill lever works and this
campaign continues it. The decode gap has not moved: no commit between
`d3a5ccf02` and `91f56a7c8` touched `qwen35_gguf_materialize.py`,
`qwen35_gguf_policy.py`, `gguf_linear.py`, or `hipengine/kernels/hip_gfx1151/`.

## 2. Recorded root causes

Both causes were attributed on 2026-09-21 with code evidence and both were
re-confirmed present at `91f56a7c8` on 2026-09-23.

### C1 — the dense-IQ decode policy is undeclared on `hip_gfx1151` (decode)

rocprofv3 kernel census (512-token prefill, 32 measured graph-replay decode
steps): UD spends **61.64 ms/token — 53.42% of its 115.39 ms pure decode —
in `gguf_iq_dense_strict_kernel`**, across four compile-time templates at
130.78 launches/token, the dominant template at **442.1 µs/launch**. Plain
records zero launches in that family. Second-ranked: Q5_K resident direct
GEMV at +20.22 ms/token and +84.3 launches/token. Third: Q4_K single local32
at +7.21 ms/token.

Mechanism: `_iq_dense_decode_dispatch`
(`hipengine/runtime/gguf_linear.py:8151-8184`) reads
`backend_package_capability(backend, "GGUF_IQ_DENSE_DECODE_POLICY", {})` and
returns the original dispatch unchanged on a miss. `hip_gfx1100` declares the
constant with all six dense-IQ quants on `local32_gemv_bf16_bf16_out`;
`hip_gfx1151` declares it nowhere. The local32 owners are already registered
on `hip_gfx1151` — `KernelKey(hip_gfx1151, linear, gguf_iq4_xs,
local32_gemv_bf16_bf16_out)` and its five siblings resolve, as does
`local32_pair_silu_bf16_bf16_out`. Nothing needs writing; the policy needs
declaring.

Recorded decision: **route the family, do not tune the strict kernel.**

Quality question recorded with it: `hip_gfx1100` also declares
`GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS` and `GGUF_IQ_DENSE_DECODE_STRICT_SLOTS`,
pinning specific UD-Q4_K_M slots back to the strict GEMV because the fast
owners' accumulation-order tail breached the calibrated max-row ceiling.
`hip_gfx1151` declares neither table. Whether gfx1151's all-one-wave prefill
policy runs the Q3_K hi+lo-split path the prefill pin exists for is an
argument to verify, not an established fact.

Evidence:
[`benchmarks/results/2026-09-21-zbook-ud-vs-plain-q4km-decode-attribution.json`](../../benchmarks/results/2026-09-21-zbook-ud-vs-plain-q4km-decode-attribution.json),
[worklog entry](../../worklog/entries/20260920T232551.346747Z-lhl-zbook-ud-decode-attribution-ddf2e9.md).

### C2 — per-tensor repack eligibility never reaches the production AR path (prefill, and decode layouts)

The production route audit at `91f56a7c8` plans the two files as:

| File | `repack` | Routes (gfx1151) |
| --- | --- | --- |
| plain Q4_K_M | `on` | Q4_K 288× `gguf_q4_k_t16_v1` (1 raw), Q5_K 48× `gguf_q5_k_t16_v1`, Q6_K 41× qmicro_planar + 24× `gguf_q6_k_t16_v1` |
| UD Q4_K_M | **`OFF`** | everything but 103 Q4_K on `raw-gguf-kernel`: Q5_K ×131, IQ4_XS ×117, Q8_0 ×104, Q6_K ×24, Q3_K ×7, IQ4_NL ×7, IQ3_S ×4 |

Mechanism: the loader calls `preflight_qwen35_gguf_artifact()` without
`repack_veto` (`hipengine/loading/qwen35_gguf_materialize.py:885`), the
admission layer pre-applies the model-wide raw-IQ veto, and `decode_repack` is
already false before the planner's per-tensor branch runs — so
`HIPENGINE_UD_REPACK_ELIGIBILITY`'s documented `per-tensor` default never
reaches the production AR path (both env values produce byte-identical route
tables).

Open reconciliation owed to C2: the closed route campaign recorded item-1
per-tensor repack as landed (44.2% optimized bytes, K_M), while the
production route table above still reports `repack=OFF` at HEAD. Experiment E3
step 1 resolves which path diverged before any layout change is written.

Evidence:
[`benchmarks/results/2026-09-21-gfx1151-qwen38-ud-vs-plain-q4km-baseline.json`](../../benchmarks/results/2026-09-21-gfx1151-qwen38-ud-vs-plain-q4km-baseline.json),
[worklog entry](../../worklog/entries/20260920T231552.005967Z-lhl-ud-vs-plain-q4km-gfx1151-baseline-ae4016.md).

## 3. Hypotheses

| ID | Hypothesis | Target | Prior evidence |
| --- | --- | --- | --- |
| H1 | A fresh prefill kernel census attributes the −37%..−41% prefill gap mainly to Q5_K/Q8_0/Q6_K raw-layout tensors running GEMV-shaped prefill owners instead of the WMMA/T16 prefill family | sizes E1 and ranks E3 | 2026-09-08 census found this shape of cause, but policies changed since; needs a fresh census |
| H2 | Declaring `GGUF_IQ_DENSE_DECODE_POLICY` on `hip_gfx1151` (with the two strict-slot pin tables) removes most of the 61.64 ms/token strict-IQ share and narrows the decode gap materially | decode, ~53% of pure decode at stake | gfx1100 record for these owners: 2.5-2.9x per launch, 22.76 vs 17.57 tok/s end-to-end — **gfx1100 evidence, not a gfx1151 rate** |
| H3 | Making per-tensor repack eligibility reach the production admission path restores Q4/Q5/Q6/Q8 tensors to tuned layouts and is the largest single prefill lever (plain keeps 288 Q4_K + 48 Q5_K + 65 Q6_K on tuned layouts; UD keeps 103) | prefill + decode layouts | route campaign: 0.00 GB → 44.2% optimized bytes and prefill 22.1 → 29.9 tok/s when forced; production path still reports `repack=OFF` |
| H4 | After H2, the Q5_K resident direct GEMV (+20.22 ms/token, +84.3 launches/token) and Q4_K single local32 (+7.21 ms/token) are the next decode pots and yield to owner/route changes, not inner-loop tuning | decode remainder | 2026-09-21 decode attribution ranking |
| H5 | The production route audit and live resident plan can diverge: the per-tensor policy may exist in the planner but be dropped, overridden, or bypassed at the loader/admission boundary. A runtime route manifest and selected-kernel trace must identify the first divergence before E3 is judged | prefill and decode layouts | C2's `repack=OFF` audit versus the existing per-tensor policy implementation |
| H6 | Once C1 removes the strict-IQ bottleneck, launch count and host submission may become material. Pair-family fusion, row batching, or graph-capture reuse for the remaining Q4/Q5/Q6/Q8 projections can win even when each leaf is near its practical read roof | decode launch overhead | C1 census: 130.78 strict-IQ launches/token; the paired baseline does not yet separate leaf time from submission/graph overhead |
| H7 | UD's raw Q5/Q6/Q8 tensors have layout candidates beyond the current Q4/Q5 focus: Q8 T16 decode, Q6 qmicro-planar/T16, and dense sidecar routes that E1/E3 show are still bypassed. Rank them by bytes and launch share, not by similarity to the plain arm | prefill and decode remainder | route-audit counts; registered gfx1151 Q8/Q6 families in `docs/KERNELS.md` |
| H8 | Owner thresholds and crossover bands were measured on other shapes or lanes and may leave a gfx1151 gap at 512/1024/4096. A sweep around each selected boundary, including values below, above, and unrelated to it, can recover wins without changing arithmetic | prefill and decode owner selection | gfx1151 package's measured row-band ladders and the anti-threshold rule in `OPTIMIZATION.md` §5 |

## 4. Experiment plan

Each experiment is one logical unit: implement, gate, paired-measure, record.
The paired baseline (§5.1) re-runs after every retained lever; the campaign
closes when the gap targets in §1 are met or the remaining hypotheses are
measured negative.

- [ ] **E0 — Baseline frozen.** Both paired artifacts recorded
  (`performance_claim: false`). Done 2026-09-23.
- [ ] **E1 — Fresh prefill attribution.** rocprofv3 `--kernel-trace` census of
  both arms at the matched 512-token prefill shape at current HEAD, same
  protocol as the 2026-09-21 decode census (warm build outside the profiler,
  pinned compiler-version file, `HIPENGINE_REQUIRE_CACHED_BUILD=1`). Output: a
  ranked per-kernel prefill table for both arms. Tests H1; sizes E3.
- [ ] **E2 — Dense-IQ decode policy declaration (C1 / H2).**
  - [ ] E2a — Recover the existing draft: `stash@{0}
    (pre-origin-main-merge-preserve-local-work-20260922)` holds the declaration
    (`hip_gfx1151/__init__.py` +56 lines: `GGUF_IQ_DENSE_DECODE_POLICY`,
    `GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS`,
    `GGUF_IQ_DENSE_DECODE_STRICT_SLOTS`, `GGUF_IQ_DENSE_VERIFY_POLICY`), a
    `gguf_linear.py` change, and `tests/test_unit_gguf_linear_dispatch_cache.py`.
    It was stashed before the 2026-09-22 origin merge and never re-applied.
    Review it against the post-merge tree and re-derive the hunks; do not
    delete the stash and do not blind-pop it (it also carries unrelated audit
    inventory churn from before the merge).
  - [ ] E2b — Verify the pin-relevance question first: does gfx1151's prefill
    policy run the Q3_K hi+lo-split path that `GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS`
    exists for? The answer decides whether the prefill pin applies here.
  - [ ] E2c — Land declaration + applicable pins, run the 162-row
    production-reference gate (18 prompts × 9 forced steps) before enabling,
    and confirm the dispatch table resolves the local32 owners.
  - [ ] E2d — Paired A/B of the decode arms; record ms/token family shares to
    confirm the strict-IQ share actually moved.
- [ ] **E3 — Production per-tensor repack (C2 / H3).**
  - [ ] E3a — Reconcile the recorded item-1 state against the production
    `repack=OFF` route table: locate where the admission call chain drops the
    eligibility mode (`preflight_qwen35_gguf_artifact()` call site and the
    upstream `decode_repack` flag).
  - [ ] E3b — Fix the call chain so `per-tensor` reaches the planner on the
    production AR path; add a route-table test that fails when the UD file
    plans `repack=OFF` under the default policy.
  - [ ] E3c — Execution-profile gate for every variant that changes
    arithmetic or layout, then paired A/B (prefill and decode) plus the route
    audit diff (optimized bytes before/after).
  - [ ] E3d — Verify the shipping path, not just the CPU audit: capture the
    materialization manifest, resident-byte totals, selected variant, and
    fallback reason from `hipengine.LLM.generate()` for both files. Reconcile
    those fields with the route-audit output and fail the experiment if the
    audit and launched owner disagree.
  - [ ] E3e — After per-tensor routing is reachable, rank the raw remainder by
    bytes and launch share. Explicitly check Q8_0 T16, Q6_K qmicro-planar/T16,
    Q5/Q6 sidecars, Q3_K W4A16 eligibility, and the lm-head route. Each
    candidate gets its own route-table test and paired A/B; do not widen a role
    predicate from static similarity alone.
- [ ] **E4 — Ranked decode remainder (H4 / H6 / H7).** Q5_K direct GEMV and
  Q4_K single local32 owners/routes, in census order, each behind its own gate
  and paired A/B. Before writing a new leaf, split the trace into kernel time,
  launch count, graph replay, host submission, and synchronization. If launch
  overhead becomes material after H2, test row batching, pair/dual owners, or
  graph-capture reuse as separate routing units. Re-rank after every retained
  route; adopt the strict-kernel-tuning prohibition from §6.
  - [ ] E4a — For every new owner, test rows 1, 2, 3, 4, 8, 16, and the
    production boundary shapes, plus one non-boundary shape. Preserve the
    strict fallback and verify the selected symbol in a real user request.
- [ ] **E5 — Closeout.** Final paired run, artifact, one
  `benchmarks/CHANGELOG.md` line per retained lever, worklog entries, and this
  document's status flipped to `closed` with the end-state table filled in.

### 4.1 Review additions: optimization surfaces not to lose

The confirmed root causes define the first two levers, but they do not exhaust
this lane's candidate space. Keep these checks in the active queue, ordered by
what E1-E3 actually attribute:

1. **Verify live route realization.** Treat the CPU route audit, materialization
   manifest, resident allocation, and launched kernel as separate facts. A
   planner fix is not a performance result until a normal
   `hipengine.LLM.generate()` request selects the intended owner and reports no
   fallback.
2. **Re-rank after H2.** The current Q5/Q4 decode ranking is pre-fix evidence.
   Dense-IQ routing can expose Q8/Q6, the lm-head, launch submission, or
   synchronization as the next bottleneck.
3. **Audit layout coverage.** For every raw UD tensor family, compare the
   planned layout with the registered gfx1151 consumers: Q8_0 T16, Q6_K
   qmicro-planar/T16, Q5/Q6 sidecars, Q3_K W4A16, and the lm-head-specific
   path. Report optimized bytes, resident bytes, launches, and time by family.
4. **Sweep owner crossovers.** Re-measure gfx1151 row and prompt-length bands
   below, above, and away from each threshold. Keep boundary changes separate
   from arithmetic changes so a route win is attributable.
5. **Measure launch and fusion economics.** If leaf time no longer explains the
   gap, test row batching, pair/dual projection owners, launch reuse, and graph
   capture as independent levers. Compare total decode wall and launch count,
   not kernel time alone; preserve strict ownership and synchronization.
6. **Track memory and thermal effects.** Record resident bytes, peak allocation,
   and temperature/clock state with every retained route result. A repack win
   that increases pressure enough to lower absolute UD throughput is not a ratio
   win.

## 5. Measurement protocol

### 5.1 Paired baseline (adjudicates every hypothesis)

Two arms, back to back, one thermal window, idle GPU:

```bash
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
  python3 scripts/qwen38_gfx1151_readme_sweep.py \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --output /tmp/ud-pair/plain_q4_k_m.json
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
  python3 scripts/qwen38_gfx1151_readme_sweep.py \
    --model /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
    --output /tmp/ud-pair/ud_q4_k_m.json
```

Protocol defaults are binding: `--prompt-lengths 512 1024 4096`,
`--decode-tokens 128`, `--max-sequence-length 8192`, `--warmups 1`,
`--repetitions 3`; production/BF16 C1 session, bulk WMMA prefill, graph-replay
decode. The harness refuses to emit timings unless both its eager and captured
trajectories pass the 18/18 id, logits and state gate. Record host
`machine_id`, commit, dirty count, and per-shape CV; adjudicate only against
the same-window control, never against a row from another host or date.

### 5.2 Kernel attribution

`rocprofv3 --kernel-trace` through `scripts/gguf_decode_graph_rocprof_driver.py`
(for decode: 512-token prefill, 4 eager warm steps, capture, 8 warm replays,
0.5 s idle gap, 32 measured replays windowed to the trailing
`advance_decode_position` occurrences) and the matched-shape prefill census
for E1. Warm build outside the profiler against a pinned compiler-version
file so the profiled process spawns no `hipcc`. The unprofiled warm run of the
same driver must reproduce the paired-baseline rates before the census is
trusted (2026-09-21 reproduced to 0.3% plain / 2.6% UD).

For E1, retain launch counts, grid/block geometry, selected symbol, resident
layout, allocation bytes, and memory-throughput/occupancy counters when the
profiler exposes them. Run the unprofiled prefill driver at 256, 768, 2048,
and 4096 tokens in addition to the matched 512-token census; this avoids
turning a threshold-point result into a general claim. Record whether each arm
reaches the planned owner through the normal `hipengine.LLM.generate()` path.
After H2, separate kernel time from launch, graph-replay, host-submission, and
synchronization time before choosing an E4 owner or fusion.

### 5.3 Route audit (layout adjudication)

```bash
python3 scripts/gguf_quant_route_audit.py \
  /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf
```

CPU-only. Note: the UD file's `nextn map: validation=FAILED (diagnostic)` is
the audit validating the draft block without the artifact pin, not a runtime
refusal.

### 5.4 Quality gates (bind every arithmetic or route change)

- Production-reference 162-row gate (18 prompts × 9 forced steps, production
  config, candidate production vs incumbent production) under the §6.1
  envelope; the tokenized category/heldout suite with deterministic repeats.
- Leaf-level comparisons cannot resolve these routes: both candidates write
  bf16 (1.66e-03 output-rounding floor). Only end-to-end gate scoring is
  evidence.
- Reference-error rule from the route campaign: score candidate-production vs
  incumbent-production — not against our own incumbent alone, not against an
  external engine's teacher.

## 6. Rules

- **Route, do not tune.** The strict IQ GEMV is not to be optimized (recorded
  decision, 2026-09-21). Levers are declarations, routes, layouts, and owners.
- **gfx1100 numbers are mechanism evidence, not gfx1151 rates.** Any transfer
  is a hypothesis until measured on this lane in a paired A/B.
- **Same-window pairs only.** zbook is power- and thermal-limited; cross-date
  or cross-host rate comparisons are diagnostics, not adjudications.
- **No speed claim without its gates.** `performance_claim: true` requires the
  profile-quality and serving gates; the two opening artifacts are explicitly
  diagnostic.
- **One lever per unit, committed after validation.** No bundling a routing
  change with a kernel edit.
- **Absolute regression disqualifies a ratio win** — for both arms.

## 7. Status scoreboard

| Lever | Hypothesis | Paired decode Δ | Paired prefill Δ | Quality gate | State |
| --- | --- | ---: | ---: | --- | --- |
| E0 baseline (2026-09-23) | — | −32.5 / −31.9 / −32.5% | −36.7 / −40.7 / −40.5% | n/a (diagnostic) | recorded |
| E1 prefill census | H1/H5/H7/H8 | — | — | n/a | open |
| E2 decode policy declaration | H2 | — | — | required | open (draft in stash) |
| E3 production per-tensor repack | H3/H5/H7 | — | — | required | open |
| E4 Q5/Q4 decode owners | H4/H6 | — | — | required | open |

## 8. Out of scope / carried

- **gfx1100 lanes** (W7900, RX 7900 XTX): separate campaigns
  ([`UD-GFX1151-OPTIMIZE.md`](UD-GFX1151-OPTIMIZE.md),
  [`UD-OPTIMIZED-ROUTE-PLAN.md`](UD-OPTIMIZED-ROUTE-PLAN.md)); this campaign
  neither ports to nor measures on them beyond citing mechanism evidence.
- **UD MTP on gfx1151** remains unmeasured; the U6 certificate is gfx1100.
  No MTP rate or promotion follows from this campaign.
- **`UD-Q4_K_S` speed-claim eligibility** (2/20 `general_ja` divergences) is
  unchanged and is not waived here; K_S can be added as a fourth arm after
  K_M's levers land.
- **Route-plan item 15** (reconcile `EXECUTION-PROFILES.md` §6 wording with
  the 2026-09-09 production-reference ruling) stays owed to that document; the
  §6.1 production reference used here is the lead's recorded ruling.
- **W4A16 decode routing** stays closed: measured to win only from 12 rows and
  lose 4x at rows=1, so decode cannot route through it.

## 9. References

- Baseline entry: [`worklog/entries/20260920T231552.005967Z-lhl-ud-vs-plain-q4km-gfx1151-baseline-ae4016.md`](../../worklog/entries/20260920T231552.005967Z-lhl-ud-vs-plain-q4km-gfx1151-baseline-ae4016.md)
- Decode attribution: [`worklog/entries/20260920T232551.346747Z-lhl-zbook-ud-decode-attribution-ddf2e9.md`](../../worklog/entries/20260920T232551.346747Z-lhl-zbook-ud-decode-attribution-ddf2e9.md)
- Prefill route gap that opened the original campaign: [`worklog/entries/20260908T042726.067135Z-lhl-ud-prefill-route-gap-9240cf.md`](../../worklog/entries/20260908T042726.067135Z-lhl-ud-prefill-route-gap-9240cf.md)
- Route campaign: [`UD-OPTIMIZED-ROUTE-PLAN.md`](UD-OPTIMIZED-ROUTE-PLAN.md)
- gfx1100 campaign: [`UD-GFX1151-OPTIMIZE.md`](UD-GFX1151-OPTIMIZE.md)
- UD bring-up campaign: [`UD-QUANTS.md`](UD-QUANTS.md), [`QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md`](QWEN38-UD-Q4KM-GFX11-CAMPAIGN.md)
- Integration record: [`UD-MAIN-INTEGRATION.md`](UD-MAIN-INTEGRATION.md)
- Evidence rules: [`docs/OPTIMIZATION.md`](../OPTIMIZATION.md), [`docs/BENCHMARK.md`](../BENCHMARK.md), [`docs/EXECUTION-PROFILES.md`](../EXECUTION-PROFILES.md)