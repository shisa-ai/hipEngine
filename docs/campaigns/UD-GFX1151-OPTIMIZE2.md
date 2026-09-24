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

- Decode: move `UD decode / plain decode` from **0.675x** toward the
  bytes-read parity point of about **1.045x** (§1.1). 1.0x is not the finish
  line: UD streams fewer weight bytes per token than plain, so at 1.0x UD is
  still slower than plain per byte read.
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

### 1.1 Bytes-read parity

Decode on this host is weight-streaming bound, so the decode target is set by
bytes read per token, not by the plain arm's rate. Summed from the GGUF tensor
tables (layers 0-63 plus `output.weight`; `token_embd` and the NextN block
excluded because decode does not stream them):

| File | Weight bytes/token | Composition (GB) |
| --- | ---: | --- |
| plain Q4_K_M | 16.09 GB | Q4_K 10.50, Q6_K 4.45, Q5_K 1.04, F32 0.10 |
| UD Q4_K_M | 15.39 GB | Q5_K 4.83, IQ4_XS 4.76, Q4_K 3.49, Q6_K 1.47, IQ4_NL 0.33, Q3_K 0.27, IQ3_S 0.15, Q8_0 0.07, F32 0.01 |

Plain at 11.58 tok/s streams about 186 GB/s. UD at the same effective rate
would decode about 12.1 tok/s, **about 1.045x plain**. Report effective GB/s
for each arm, and for each kernel family in a census, next to tok/s. This is
an inferred ceiling from byte counts, not a measured rate.

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
production route table above still reported `repack=OFF` at HEAD. **Resolved
by E3 (2026-09-25):** the planner was never the diverged path — production
resolved `decode_repack=None` to the env default and applied per-tensor
eligibility all along (E1's live probe was the true view). The admission
report coerced `None` to `False` and pre-applied the model-wide raw-IQ veto,
and `gguf_quant_route_audit.py` pre-applied the same veto itself, so both
report surfaces recorded `repack=OFF`. Evidence:
[`benchmarks/results/2026-09-25-zbook-e3-route-audit-repack-truth.json`](../../benchmarks/results/2026-09-25-zbook-e3-route-audit-repack-truth.json).

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
| H9 | UD's gate/up and residual work runs unfused. Plain spends 32.40 ms/token in one fused Q4_K dual local32+SiLU owner (62 launches/token) and folds the residual into its Q4_K/Q6_K owners; UD has no fused dual and pays a separate SiLU-mul and residual add (62 launches/token each). Fusing by per-layer gate/up **type pair** removes launches and the separate passes | decode launches and wall-minus-kernel gap | 2026-09-21 census; UD per-layer gate/up pairs: IQ4_XS/IQ4_XS 20, Q5_K/Q5_K 9, Q4_K/Q4_K 5, Q3_K/Q3_K 1, IQ4_NL/IQ4_NL 1, mixed 28 |
| H10 | UD stores the 48 GDN layers' `ssm_alpha`/`ssm_beta` as Q8_0 (plain: F32). The fused alpha/beta+conv decode owner admits only `quant_key == "f32"` (`_try_launch_dense_f32_alpha_beta_conv_decode`), so UD falls back to a Q8_0 dual split GEMV plus a separate conv launch | decode, ~0.45 ms/token and ~48 launches/token | census: UD 0.92 ms Q8_0 dual split (47.5 launches) + 0.19 ms conv (46.5) vs plain 0.66 ms fused |
| H11 | UD's norm launches cost 1.62 ms/token against plain's 0.38 at the same 124.97 launches/token. A different norm variant is being selected, plausibly because the residual is not folded into the preceding owner | decode, ~1.2 ms/token | 2026-09-21 census `norms` family; cause not yet attributed |
| H12 | gfx1100 runs IQ4_XS/IQ4_NL prefill on the coop64 W4A16 owner and Q3_K/IQ3_S on coop32 (all bit-exact); gfx1151's own `GGUF_IQ_DENSE_PREFILL_POLICY` (`hip_gfx1151/__init__.py:2071`) sends all seven dense-IQ quants to the one-wave owner. IQ4_XS alone is 4.76 GB of UD. No registration port is needed: `register_gfx1151_kernels()` mirrors the coop/coop64 keys and all 14 of them resolve for `hip_gfx1151` — only the four per-quant policy overrides are gfx1100-only, and the coop owners have never been exercised on this backend | prefill | gfx1100 package comments on `GGUF_IQ_DENSE_PREFILL_POLICY`: IQ4_NL 2.0-2.2x one-wave, IQ4_XS 1.5-2.1x as coop32 and a further 1.4-1.6x at coop64 over 512-1024 rows, Q3_K/IQ3_S 1.4-1.5x; **gfx1100 evidence, not a gfx1151 rate** |

## 4. Experiment plan

Each experiment is one logical unit: implement, gate, paired-measure, record.
The paired baseline (§5.1) re-runs after every retained lever; the campaign
closes when the gap targets in §1 are met or the remaining hypotheses are
measured negative.

- [x] **E0 — Baseline frozen.** Both paired artifacts recorded
  (`performance_claim: false`). Done 2026-09-23.
- [x] **E1 — Fresh prefill attribution.** rocprofv3 `--kernel-trace` census of
  both arms at the matched 512-token prefill shape at current HEAD, same
  protocol as the 2026-09-21 decode census (warm build outside the profiler,
  pinned compiler-version file, `HIPENGINE_REQUIRE_CACHED_BUILD=1`). Output: a
  ranked per-kernel prefill table for both arms. Tests H1; sizes E3.
  **Done 2026-09-24, `performance_claim: false`.**
  ([artifact](../../benchmarks/results/2026-09-24-zbook-ud-plain-q4km-prefill-census.json))
  - **H1 refuted as stated**: both arms run WMMA prefill families throughout;
    misc GEMV is 0.2% of the UD window. The +952.4 ms window delta (plain
    1687.9 ms / 1740 dispatches vs UD 2640.3 ms / 1264, ratio 0.6393) is
    ranked instead as **dense-IQ one-wave W4A16 at 48.18% of the UD window**
    (1272.1 ms, 135 launches, IQ4_XS template `<0>` alone 1020.1 ms at 8719
    µs/launch ×117) and **Q5_T16 WMMA at 24.81%** (655.0 ms). Q4_T16 (346.9
    ms vs plain 971.5) and Q6_T16 (66.8 vs 402.7) are *cheaper* on UD — the
    composition shift, not a GEMV fallback, is the gap.
  - **Sizes E9 (H12) as the largest single prefill lever**: the 48.18% is
    exactly the one-wave owner E9's coop/coop64 overrides target. Mechanism
    arithmetic only (gfx1100 record, not a gfx1151 rate): 1.5× on that family
    → ratio ≈ 0.76, 2× → ≈ 0.84; parity still needs the Q5_T16 pot (E4/E6).
  - **Live route probe passes both arms** through `hipengine.LLM.generate()`
    (campaign §5.2): plain resolves its T16/dual WMMA owners; UD resolves
    `dense_wmma_w4a16_prefill_bf16_bf16_out` for all four dense-IQ quants
    (234/14/14/8 resolves) — the same owner as census symbol
    `gguf_iq_prefill_wmma_kernel`, registered by `gguf_iq_wmma_prefill.py`.
  - **E3a input (H5 live fact)**: the shipped path resolves *T16 layout keys*
    for UD (`gguf_q4_k_t16_v1`, `gguf_q5_k_t16_v1`,
    `gguf_q6_k_t16_qmicro_planar_v1`, `gguf_q8_0_t16_v1`) while the committed
    route audit records `repack=OFF` with those families on
    `raw-gguf-kernel`. E3a must locate which surface diverged before any
    layout change is written.
  - **H7 ranked low for prefill** (Q8 71.5 ms / 2.7%, Q6 66.8 ms / 2.5%);
    **H8 shows no threshold artifact**: the unprofiled 256→4096 ladder drifts
    monotonically (UD/plain 0.643 → 0.596), so the widening long-prompt gap is
    a row-scaling effect, not a cusp at 512.
  - Protocol note: the first identical census (2026-09-23) lived under
    `/tmp` and was destroyed by the 2026-09-24 host reboot; the recorded
    re-run from `~/ud-e1-census/` reproduced its ranked shares to within
    ~1%. `--kernel-trace` yields timing/geometry only — no bandwidth counters,
    so per-family GB/s from §1.1 remains unmeasured.
- [ ] **E2 — Dense-IQ decode policy declaration (C1 / H2).**
  - [x] E2a — Recover the existing draft: stash
    `9a381b0c1223597e5605ac17dedf99f43a5de66a`
    (`pre-origin-main-merge-preserve-local-work-20260922`; `stash@{0}` on
    2026-09-23, but the stash stack is shared and the index drifts, so address
    it by SHA) holds the declaration
    (`hip_gfx1151/__init__.py` +55 lines: `GGUF_IQ_DENSE_DECODE_POLICY`,
    `GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS`,
    `GGUF_IQ_DENSE_DECODE_STRICT_SLOTS`, `GGUF_IQ_DENSE_VERIFY_POLICY`), a
    `gguf_linear.py` change, `tests/test_unit_gguf_linear_dispatch_cache.py`,
    and gate-script edits: `scripts/gguf_iq_local32_decode_gate.py` (adds a
    `--backend` argument for gating this backend's declaration) and
    `scripts/gguf_ud_combined_stack_gate.py`. It was stashed before the
    2026-09-22 origin merge and never re-applied.
    Review it against the post-merge tree and re-derive the hunks; do not
    delete the stash and do not blind-pop it (it also carries unrelated audit
    inventory churn from before the merge).
    **Done 2026-09-24:** hunks re-derived and applied after review (the stash
    is intact and un-popped; a patch copy lives in `~/ud-e1-census/`). The
    stash's comment promised
    `tests/test_unit_gfx1151_iq_dense_policy_parity.py` but carried only its
    dispatch-cache half; the full parity test was recovered from pre-merge
    commit `29c4792d9` (the WIP that produced the stash) and passes 22/22
    with the declaration applied. The declaration moves three names out of
    the gfx1100-only capability ledger, so
    `test_gfx1151_capability_ledger_covers_gfx1100_only_live_reads` was
    updated 25→22 and the transferred rows moved to a dated note in
    `docs/archive/20260909-GFX1151-GFX1100-TRANSFER-AUDIT.md`. Route probe
    through `hipengine.LLM.generate()` confirms local32 fires (124
    selections), strict fallback resolves for the 4 pinned IQ3_S slots and
    the deliberately-unrouted Q3_K, prefill W4A16 unchanged. Guard chain
    green (compileall, pytest, fixtures, 3 smokes).
  - [x] E2b — Verify the pin-relevance question first: does gfx1151's prefill
    policy run the Q3_K hi+lo-split path that `GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS`
    exists for? The answer decides whether the prefill pin applies here.
    **Answer 2026-09-24: yes, the pin applies.** (1) gfx1151's
    `GGUF_IQ_DENSE_PREFILL_POLICY` routes `gguf_q3_k` to the one-wave owner
    `dense_wmma_w4a16_prefill_bf16_bf16_out` (rows 8-131072), whose launcher
    `hipengine_gguf_iq_wmma_prefill_bf16_bf16_out` dispatches
    `case 3 → launch_iq_wmma_prefill<3>` (Q3_K is quant 3). (2) The split is
    a property of the quant, not the variant: `SPLIT_LO = (Q == 3 || Q == 5
    || Q == 6)` is constexpr in both the one-wave kernel
    (`gguf_iq_wmma_prefill.hip:218`) and the coop kernel (`:378`), so every
    prefill variant gfx1151 can select runs Q3_K through the hi+lo split
    path - the arithmetic class the pin exists for. (3) The production
    binding is present: `_iq_dense_mmq_strict_slots()`
    (`qwen35_gguf_runner.py:20463`) reads
    `GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS` for the backend keyed by
    `(file_type, artifact preset)`, and `_iq_dense_prefill_dispatch` checks
    the pinned slot before policy admission. The recovered declaration's
    prefill pin (`layers.0.ffn_up` on UD-Q4_K_M) therefore transfers and is
    exercised by `test_gfx1151_prefill_keeps_the_strict_owner_for_the_pinned_slot`.
  - [x] E2b′ — Leaf timing before the gate: time each local32 owner against
    the strict GEMV on every real UD dense-IQ shape on gfx1151 (min-of-N,
    interleaved). This gives the gfx1151 per-launch factor that the
    pre-registered prediction below depends on, cheaply.
    **Pre-registered prediction:** at a 2.5-2.9x per-launch factor, the
    61.64 ms/token strict share falls to 21-25 ms/token, UD wall moves from
    about 125 to 85-88 ms/token, and paired decode lands at **0.98-1.02x**.
    E2 alone is expected to close most of the decode gap; a result well below
    that band means another cost is hiding, and E4 starts from that residual.
    Include `local32_pair_silu_bf16_bf16_out` for the 20 IQ4_XS/IQ4_XS
    gate/up layers (H9) in the same declaration if its gate passes with it.
    **Done 2026-09-24:** `scripts/gguf_iq_local32_decode_leaf.py` timed all
    13 real dense-IQ shapes (rows 1/2/4, min-of-N interleaved; the two Q3_K
    shapes report `local32=false` and are excluded — Q3_K stays deliberately
    unrouted). Artifact: `~/ud-e1-census/e2b-leaf-timing.json`.
    - rows==1 per-launch factor over the 11 measured shapes: **mean 2.51,
      range 1.89–3.96**. The mean sits at the floor of the registered
      2.5–2.9 band and the distribution is bimodal: **IQ4_XS (117 of the
      124 routed slots) averages 2.06 (1.89–2.19), below the band**, while
      IQ4_NL (7 slots) averages 3.77 (3.47–3.96). The effective factor for
      the dominant family is therefore ~2.0x, not 2.5x, so the prediction
      below is at risk on its dominant term; E2d's paired A/B adjudicates
      the actual decode ratio, not this input.
    - local32-vs-strict rows==1 output: max relative difference 2.94e-3
      (reassociation class, screened by the KL gates in E2c), correlation
      ≈ 1.
    - pair arm (`blk.1` IQ4_XS/IQ4_XS gate/up): `local32_pair_silu` is
      **bit-exact to the unfused chain**, 2.01x faster than the two strict
      GEMVs, and 1.02x faster than the chain (~10 µs/pair). The pair row
      ships in this declaration; the E2c probe gate resolves its owner
      through the resident session (`local32_pair_silu_bf16_bf16_out` in
      the candidate owner list).
  - [x] E2c — Land declaration + applicable pins, run the 162-row
    production-reference gate (18 prompts × 9 forced steps) before enabling,
    and confirm the dispatch table resolves the local32 owners.
    **Done 2026-09-24.** Declaration + both pin tables landed with E2a (the
    pins run in both arms). Gate:
    `gguf_ud_combined_stack_gate.py --backend hip_gfx1151 --category-heldout
    --decode-tokens 9` — gfx1151 mode pairs the shipped decode policy
    (candidate) against the all-strict decode incumbent with the shipped
    prefill route held constant in both arms, so the paired difference is
    exactly the E2 lever. **PASS** over pooled 180 positions (18 prompts ×
    (9 forced steps + 1 prefill-last row); the 162 forced-step rows are the
    campaign's count): KL mean 4.68e-5 (≤1e-3), p95 2.72e-4 (≤5e-3), p99
    7.03e-4 (≤2e-2), max 8.55e-4 (≤5e-2), top1 1.0000 (≥0.99), every scope
    top1 1.0000 (≥0.97; code 60 / general_en 40 / general_ja 40 /
    mixed_ja_en 40), candidate logits finite, no diagnostic above the p99
    envelope. The calibration arm (incumbent vs all-strict, same forced
    tokens) measures mean 1.84e-4 / max 1.06e-2 — the candidate decode
    re-route moves logits ~3.9x **less** than the prefill-route choice
    production already ships. Artifact:
    `~/ud-e1-census/e2c-162row-gate-run1.json`; deterministic repeat in
    flight at record time. The six-quant local32 probe gate
    (`gguf_iq_local32_decode_gate.py --backend hip_gfx1151`) also PASSes:
    KL mean 2.33e-4, max 4.93e-3, top1 1.0, finite, prefill KL 0, candidate
    owners incl. `local32_pair_silu`.
    Dispatch confirmation through `hipengine.LLM.generate()` (48 decode
    steps, `~/ud-e1-census/e2c-llm-route-probe.log`, `RESULT e2a: PASS`):
    local32 singles **3948** resolves, **pair dual 940** (~20/step = all 20
    IQ4_XS/IQ4_XS layers), strict fallback **517** (pinned IQ3_S + declared
    Q3_K), prefill W4A16 intact **134**, verify-row owner 0 (no verifier
    rows in this window).
  - [x] E2d — Paired A/B of the decode arms; record ms/token family shares to
    confirm the strict-IQ share actually moved.
    **Done 2026-09-24.** Same-window paired A/B
    (`gguf_iq_dense_decode_ab.py --backend hip_gfx1151 --repetitions 3
    --steps 32`; arms alternate in one process, each captures its own decode
    graph, owner assertions enforced): decode **8.1089 → 10.8090 tok/s,
    ratio 1.3330x** (CV 0.76% / 0.13%); prefill 188.93 / 188.81 tok/s,
    ratio 0.9994x — the lever is decode-only as designed. Incumbent owners
    W4A16-prefill + strict gemv; candidate adds `local32_gemv` and
    `local32_pair_silu`. Artifact: `~/ud-e1-census/e2d-paired-ab.json`.
    Decode census (E0 protocol — 512-token prefill, 4 warm, capture, 8 warm
    replays, 0.5 s gap, 32 graph steps, rocprofv3 1.3.5 `--kernel-trace`,
    trailing-`advance_decode_position` window):
    **strict-IQ family 61.64 → 5.36 ms/token (−91.3%)** — the share moved.
    The work landed on the IQ4 local32 family at 25.80 ms/token: pair dual
    `gguf_iq4_xs_local32_dual_silu<2>` 9.74 ms/token at 19.38 launches/token
    (all 20 IQ4_XS/IQ4_XS layers), IQ4_XS singles 14.40, IQ4_NL 1.66;
    combined IQ route 31.16 ms/token vs 61.64 strict-only. Residual strict
    5.36 ms/token = exactly the 11 non-local32 slots (Q3_K 7 declared
    strict + IQ3_S 4 pinned) at 10.66 launches/token. Pure window
    115.39 → 85.53 ms/token; 807.97 launches/token. Unprofiled warm decode
    **10.89 tok/s vs E0's 8.085 on the identical same-host protocol
    (1.347x)**. Q5_K direct GEMV (20.50 ms/token) and Q4_K single local32
    (14.08) are unchanged and now rank first and second — E3/E4/E6's pots.
    Evidence bundle (OPTIMIZATION §2 fields): `benchmarks/results/2026-09-24-
    zbook-gfx1151-ud-iq-decode-policy-e2d.json` (`performance_claim: false`;
    the loop's `ud_plain_parity` is the paired UD/plain claim vehicle).
- [x] **E3 — Production per-tensor repack (C2 / H3).** Executed 2026-09-25;
  resolution recorded in the scoreboard row and the C2 reconciliation note
  above. The unit changed report surfaces only — runtime allocation is
  byte-identical — so no profiled A/B is owed and the parity metric is
  untouched (iteration logged with metric unchanged).
  - [x] E3a — Reconcile the recorded item-1 state against the production
    `repack=OFF` route table: locate where the admission call chain drops the
    eligibility mode (`preflight_qwen35_gguf_artifact()` call site and the
    upstream `decode_repack` flag). **Found two report-surface drops:**
    admission's `repack_veto is None` branch coerced `decode_repack=None` to
    `False` and pre-applied the model-wide raw-IQ veto without consulting
    `resolve_ud_repack_eligibility()`; `gguf_quant_route_audit.py`
    pre-applied the same veto itself. The planner (`plan_qwen35_gguf_
    materialization`) was correct all along.
  - [x] E3b — Fix the call chain so `per-tensor` reaches the planner on the
    production AR path; add a route-table test that fails when the UD file
    plans `repack=OFF` under the default policy. **Fixed in admission +
    audit** (shared `resolve_gguf_decode_repack` in `qwen35_gguf_policy`);
    RED→GREEN: `test_unit_qwen35_gguf_decode_repack_semantics` (3 new E3
    tests), live UD-file gate `test_e3_ud_file_default_policy_plans_repack_on`,
    route-audit unit file incl. a pinned `model-wide` rollback-seam test.
  - [x] E3c — Execution-profile gate for every variant that changes
    arithmetic or layout, then paired A/B (prefill and decode) plus the route
    audit diff (optimized bytes before/after). **No variant changes
    arithmetic or layout on the shipped path** (report-surface fix only), so
    the profiled A/B is not owed; route-audit prefix/postfix diff recorded —
    plain control: 0 differing scalars on both backends; UD
    `decode_repack_enabled` False→True, raw_gguf residents 395→136
    (12.60→6.23 GB), accepted planned 16.08→15.85 GiB (gfx1100) /
    16.08→15.19 GiB (gfx1151).
  - [x] E3d — Verify the shipping path, not just the CPU audit: capture the
    materialization manifest, resident-byte totals, selected variant, and
    fallback reason from `hipengine.LLM.generate()` for both files. Reconcile
    those fields with the route-audit output and fail the experiment if the
    audit and launched owner disagree. **PASS both arms:** the route probe
    through `hipengine.LLM.generate()` resolves the T16 census owners
    (q4_k_t16/q5_k_t16/q6_k_qmicro_planar/q8_0_t16 prefill) plus the dense-IQ
    W4A16 census owner on the UD arm and the T16 prefill family on plain,
    `RESULT PASS` with exit 0 for each arm (logs:
    `e3-llm-route-probe-{ud,plain}.log`), agreeing with the postfix audit.
  - [ ] E3e — After per-tensor routing is reachable, rank the raw remainder by
    bytes and launch share. **Bytes rank recorded** (postfix audit, gfx1100):
    IQ4_XS ×117, IQ4_NL ×7, Q3_K ×7, IQ3_S ×4, Q4_K tied-source ×1 = 136 raw
    residents / 6.23 GB. Launch-share re-rank after E4's re-census; each
    candidate (Q8_0 T16 decode, Q6_K variants, Q5/Q6 sidecars, Q3_K W4A16,
    lm-head route) still gets its own route-table test and paired A/B — do not
    widen a role predicate from static similarity alone.
- [x] **E4 — Ranked decode remainder (H4 / H6 / H7).** Q5_K direct GEMV and
  Q4_K single local32 owners/routes, in census order, each behind its own gate
  and paired A/B. Before writing a new leaf, split the trace into kernel time,
  launch count, graph replay, host submission, and synchronization. If launch
  overhead becomes material after H2, test row batching, pair/dual owners, or
  graph-capture reuse as separate routing units. Re-rank after every retained
  route; adopt the strict-kernel-tuning prohibition from §6. Run E4 after E3:
  **premise correction (E3, 2026-09-25):** the original note here said UD's
  103 Q4_K tensors plan `kernel:gguf_q4_k` rather than plain's
  `gguf_q4_k_t16_v1` — that came from the stale route audit. E3's postfix
  audit shows **both** arms plan their Q4_K tensors as
  `kernel:gguf_q4_k_t16_v1` (UD 103×, plain 288×), so the layout-consequence
  reading of H4 is refuted: UD's Q4_K single local32 launching about 167 µs
  against plain's 120 µs at the same T16 layout is a genuine owner-tuning
  candidate, not a C2 artifact. Re-census after E3 before writing any Q4_K
  owner change (E3 changed report surfaces only, so census deltas vs E2d
  indicate drift, not layout change).
  **Census + split + re-rank recorded 2026-09-25** (iteration 7; artifact
  `benchmarks/results/2026-09-25-zbook-e4-decode-recensus.json`): E2d
  reproducibility within 0.2% on every top kernel with identical
  launches/token (runtime-neutrality of E3 confirmed empirically); split —
  kernel 91.1% (UD) / 92.5% (plain) of wall, non-kernel ≈10.3/11.4 µs per
  launch, so launch overhead is material but not dominant and kernel pots
  rank first; re-rank — (1) Q5 selected-down direct GEMV 20.47 ms/tok @
  242.9 µs/launch with no plain counterpart (plain's Q5 work runs in
  `q5_k_t16_dense_tile8_gemv` at 111.2 µs/launch, UD tile8 111.5 — owner
  speed identical where both run it; composition 131 vs 48 Q5_K tensors),
  (2) Q4_K single local32 157.9 vs plain 119.5 µs/launch at the same T16
  layout, (3) IQ4_XS local32 family 24.15 ms/tok (E9/E2-adjacent space).
  Candidates each stay open behind their own gates (E4a shape matrix, strict
  fallback, symbol verified in a real request, route-table test, paired
  same-window A/B).
  - [x] E4a — **Executed 2026-09-25 (iteration 8): route lever (b), leaf
    tuning (a) not needed.** The census pot was a policy-table miss, not an
    owner defect: gfx1151's `GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE` carried
    exactly one Q5_T16 shape — `ssm_out` (6144,5120), which is plain's only
    Q5 tensor — so UD's other six decode shapes (119 tensors: ffn_up/down,
    attn_q/qkv/v/gate) fell through to the 242.9 µs direct owner (census
    item 1, 20.47 ms/tok). Rows=1 screen, tile8 vs direct vs the
    `gguf_quant_gemv` reference: **bit-exact on all seven production
    shapes**; tile8 wins ffn_up 1.16x (393.0→339.8 µs), ffn_down 1.28x
    (415.7→324.9), attn_gate 1.36x (186.2→136.5), attn_qkv 1.39x
    (284.1→204.7), attn_q 1.18x (292.0→248.1); **attn_v kept direct**
    (47.6 vs 47.8 µs, 0.99x — no win). One lever: five rows added to the
    gfx1151 table. Route conservation through `hipengine.LLM.generate()`:
    direct 4410→504 resolves, tile8 1953→5859, total 20599 unchanged —
    every moved resolve landed on tile8, no silent fallback; plain arm
    untouched (its sole Q5 shape was already routed). Guard green: unit
    contract RED→GREEN (`test_gfx1151_backend_aliases_gfx1100_kernel_keys`,
    pre-existing `ssm_out` C1 route test still passes), 8-shape GPU
    exactness node 83.7 s, full default-tier suite + fixtures + three
    smokes all exit 0, stash 9a381b0c untouched. Shape-matrix note: the C1
    route is rows==1-only by construction (`_t16_c1_variant_dispatch`
    guards rows!=1, non-t16 ABI, and native-batch sessions), so rows 2–16
    keep their rowtile/direct owners unchanged; no new kernel means the
    strict-fallback clause is n/a, and the real-request symbol check is the
    route probe above. Paired A/B `ud_plain_parity` **pending** per lead
    directive: three consecutive windows invalidated by campaign 5.1's 2%
    prefill@512 visit check (5.71% / 5.39% / 6.33%; plain_1 303–314 vs
    settled plain_2 287–295; a 60-minute clean GPU idle did not settle the
    first visit). The check binds plain's prefill, which this rows==1
    decode-route edit cannot affect.
  - [x] E4b — **Executed 2026-09-25 (iteration 9): item 2 adjudicated
    composition, negative result; item 1's move confirmed.** Post-E4a
    re-census (same E0 protocol; artifact
    `benchmarks/results/2026-09-25-zbook-e4b-post-e4a-recensus.json`):
    the direct GEMV symbol left the top 12 (tile8 now owns all Q5 at
    24.02 ms/tok, 116.2 launches/tok; Q5 family 25.22 → 24.02 ms/tok and
    −10.7 launches/tok; UD pure 85.53 → 84.83 vs plain 79.81 unchanged;
    plain control totals identical at 571.59 launches/tok). Item 2
    (Q4_K single local32, UD 158.0 vs plain 119.5 µs/launch) is **not an
    owner defect**: both arms run the same kernel symbol and
    same-kernel+same-shape speed is arm-identical (tile8 111.2/111.5
    precedent); the gap is role/shape skew plus per-tensor quant scatter
    — plain pairs 65/65 ffn layers as (Q4_K,Q4_K) into 62 dual-SiLU
    launches/tok, UD has only 5 (Q4_K,Q4_K), 20 (IQ4_XS,IQ4_XS) pairable
    via the IQ dual (19.4/tok in-window), 9 (Q5_K,Q5_K), and **13
    mixed-quant gate/up layers unpairable under every same-quant rule**
    (26 singleton launches). Closing that scatter would re-plan the
    artifact's per-tensor quants — a bytes/arithmetic change under the
    execution-profile gate, not a code-route lever. Re-rank: item 3 =
    IQ4_XS local32 family 24.25 ms/tok (dual_silu 9.80 + gemv<2,1> 7.51
    + gemv<4,1> 6.94) — UD-only quants with no plain counterpart, so a
    within-UD variant/shape question, E9-adjacent.
  - [x] E4c — **Executed 2026-09-25 (iteration 10): item 3 adjudicated
    composition, negative result; E4 complete.** Full IQ4 tensor-role
    map reconciles the census: `dual_silu` 19.4 launches/tok ≈ the
    **20 (IQ4_XS,IQ4_XS) pairable ffn layers, all already on the fused
    owner** (97%; remainder = per-token block-type variation); the gemv
    singles ≈ inherent single-tensor roles (`ffn_down` IQ4 ×27 with no
    partner by definition; attention roles ×32 — attn_gate 12, attn_q 8,
    attn_qkv 6, ssm_out 5, attn_output 1) plus the 22 IQ4-involved
    mixed-quant gate/up layers — pairable count fully consumed (0
    unpaired IQ4 pairs). The `<2,1>`/`<4,1>` split is shape-class
    selection with no same-shape alternative to beat. **E4 closes:
    item 1 route-landed (E4a), items 2-3 negative composition
    (E4b/E4c)**; paired A/B parity stays pending per the lead's
    directive; E6 inherits the mixed-layer sizing.
- [ ] **E6 — Gate/up and residual fusion by type pair (H9).** After E2 and E3,
  re-census the FFN block. Route same-type pairs to existing dual owners first
  (IQ4_XS pair-SiLU if not already landed in E2; Q5_K
  `q5_k_t16_dense_dual_silu`, which needs E3's T16 layout). Then size the 28
  mixed-type layers (E4c sizing confirmation 2026-09-25: exactly 28 per-block
  mixed pairs — 22 IQ4-involved, 6 K-quant; inventory in the E4b/E4c
  artifacts): plain's Q6/Q4 mixed-pair owner is the precedent for a
  two-quant dual. Residual folding into the IQ local32 and Q5_K owners is a
  separate unit. Judge each on decode wall and launches/token, not kernel time
  alone.
  **Step 1 adjudicated 2026-09-25 (E6a, iteration 11):** IQ4_XS pair-SiLU is
  already firing (19.4 launches/tok, landed in E2, confirmed by E4c) and the
  Q5_K same-quant dual was screened at the production shape (5120, 17408)
  rows=1 — **bit-exact but 0.93× versus the 2×tile8 chain (705.6 vs 653.6
  µs/layer), so the route is NOT enabled** (the dual mirrors the direct owner's
  schedule; post-E4a singles run tile8). Dormancy causes and the re-screen
  clearing command are in `docs/REFACTOR.md`; one-layer leftovers (Q5/Q6/Q3
  same-quant singles) are below any route's fixed cost. **No same-quant route
  gap remains; step 1 closes.**
- [ ] **E7 — GDN alpha/beta on the fused path (H10).** Make UD's Q8_0
  `ssm_alpha`/`ssm_beta` reach the fused alpha/beta+conv owner, either by
  planning a load-time Q8_0→F32 expansion (exact in weight value: an fp16
  scale times an int8 is representable in f32) or by a Q8_0 input variant of
  the fused owner. Key eligibility on the stored dtype and shape, never on the
  artifact identity. Check the prefill alpha/beta route for the same split.
  Confirm the fused symbol fires through `LLM.generate()`.
- [ ] **E8 — Attribute the norm-cost gap (H11).** From the existing 2026-09-21
  traces first (no new GPU run needed): which norm symbols UD selects versus
  plain, their µs/launch, and what forces the variant. If it follows from the
  unfused residual, it rides with E6; otherwise it is its own routing unit.
- [ ] **E9 — Cooperative IQ prefill owners on gfx1151 (H12).** The
  coop/coop64 keys already resolve on `hip_gfx1151` — `register_gfx1151_kernels()`
  mirrors every non-excluded `hip_gfx1100` key, and all 14 coop-family keys
  were verified present — so there is no registration port to write. The unit
  is: apply gfx1100's four per-quant overrides to gfx1151's
  `GGUF_IQ_DENSE_PREFILL_POLICY`, verify bit-exactness against the one-wave
  owner on the real UD shapes (first exercise of these owners on this
  backend; the bit-exact record is gfx1100's), then measure. Q3_K's coop32
  route uses the hi+lo split path, so it inherits E2b's pin question. Paired
  prefill A/B at the §5.1 shapes plus a short prompt.
- [ ] **E5 — Closeout** (runs last, after E6-E9). Final paired run, artifact, one
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

Two arms in one thermal window, idle GPU, plain/UD visits interleaved (ABBA
order in the block below):

```bash
# One thermal window, arms ordered plain, UD, UD, plain (ABBA), idle GPU:
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
  python3 scripts/qwen38_gfx1151_readme_sweep.py \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --output /tmp/ud-pair/plain_1.json
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
  python3 scripts/qwen38_gfx1151_readme_sweep.py \
    --model /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
    --output /tmp/ud-pair/ud_1.json
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
  python3 scripts/qwen38_gfx1151_readme_sweep.py \
    --model /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
    --output /tmp/ud-pair/ud_2.json
HIPENGINE_HIP_ARCH=gfx1151 GPU_MAX_HW_QUEUES=2 \
  python3 scripts/qwen38_gfx1151_readme_sweep.py \
    --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
    --output /tmp/ud-pair/plain_2.json
```

Protocol defaults are binding: `--prompt-lengths 512 1024 4096`,
`--decode-tokens 128`, `--max-sequence-length 8192`, `--warmups 1`,
`--repetitions 3`; production/BF16 C1 session, bulk WMMA prefill, graph-replay
decode. The harness refuses to emit timings unless both its eager and captured
trajectories pass the 18/18 id, logits and state gate. Record host
`machine_id`, commit, dirty count, and per-shape CV; adjudicate only against
the same-window control, never against a row from another host or date.

Keep the command block's plain, UD, UD, plain (ABBA) order rather than a
single back-to-back pair. Absolute rates moved 4-5% between days, and
a single pair cannot separate drift inside the window from the change under
test. Report both visits of each arm together; a within-arm disagreement
wider than the recorded CV band invalidates the window. When
a lever touches a row-count threshold (for example the dense-IQ prefill
`min_rows=8`), add one short prompt (16-64 tokens) to the sweep so the
row-band behavior is exercised away from the 512/1024/4096 points.

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
- Bit-exact levers take the fast lane. A route, layout, or fusion change whose
  token ids, forced-step full-vocab logits, and state fingerprints are
  bit-identical to the incumbent on the 162-row production-reference gate (for
  example E7's exact Q8_0→F32 expansion, or E9's coop owners if they reproduce
  gfx1100's bit-exact record) records that identity check in place of the KL
  envelope run: identity across the gate's forced rows implies KL = 0 on them.
  Free-run ids on the 18-prompt suite alone do not establish that — greedy
  stability can hide off-trajectory logit changes. The envelope gate binds
  whenever the arithmetic changes.

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
- **Unresolvable zbook swings are pending, not blockers.** When a gate
  comparison on this host stays outside the recorded CV band for reasons that
  cannot be separated from power/thermal throttle, record the item as pending
  for the desktop gfx1151 lane and move on to the next experiment; do not
  chase the number on zbook. (Lead directive, 2026-09-23.)

## 7. Status scoreboard

| Lever | Hypothesis | Paired decode Δ | Paired prefill Δ | UD launches/token | UD wall − kernel (ms/token) | Quality gate | State |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| E0 baseline (2026-09-23) | — | −32.5 / −31.9 / −32.5% | −36.7 / −40.7 / −40.5% | 856.41 (plain 571.59)¹ | 9.81 (plain 6.54)¹ | n/a (diagnostic) | recorded |
| E1 prefill census | H1/H5/H7/H8 | — | census window 0.6393 (1687.9 vs 2640.3 ms, not a paired A/B) | 1264/window (plain 1740) | — | n/a (attribution) | recorded 2026-09-24: H1 refuted, IQ one-wave 48.18% + Q5_T16 24.81% rank the gap; route probe PASS; T16-live-vs-audit-OFF divergence owed to E3a |
| E2 decode policy declaration | H2 | window invalidated (see State) | window invalidated; A/B prefill 0.9994x | 807.97 (E0-protocol census) | 6.28 (unprofiled wall 91.81 − pure 85.53) | **PASS**: 162-row ×2 bit-identical (KL mean 4.68e-5, max 8.55e-4, top1 1.0000 overall/per-scope), local32 probe gate, LLM.generate route probe | executed 2026-09-24: paired A/B decode **1.3330x** (8.1089 → 10.8090 tok/s, CV 0.76%/0.13%), strict-IQ 61.64 → 5.36 ms/token (−91%), unprofiled warm 1.347x vs E0; `ud_plain_parity` **pending** per lead directive 2026-09-25 (set pending and move on) — three windows invalidated by campaign 5.1's 2% prefill@512 visit check under evening host drift (probe start 314 vs 286-292 floor; the 04:48 baseline agreed to 0.5%); iteration 5 logged with metric unchanged; record in the E2d artifact + E2 checkboxes |
| E3 production per-tensor repack | H3/H5/H7 | — (report-surface unit; runtime allocation byte-identical) | — | — | — | **PASS**: RED→GREEN unit tests, live UD-file gate, full-suite broad run + focused repair, route-audit plain 0-diff control, `LLM.generate` route probe PASS ×2 arms | executed 2026-09-25: **H5 divergence located** — the planner was always per-tensor-correct; admission coerced `decode_repack=None`→False plus a model-wide raw-IQ veto, and the route-audit script pre-applied the same veto, so both report surfaces recorded `repack=OFF`. Fix = shared `resolve_gguf_decode_repack` + eligibility-gated veto; `model-wide` seam reproduces the pre-fix record, plain control 0 differing scalars; UD report: raw_gguf 395→136 residents (12.60→6.23 GB), planned 16.08→15.85 GiB (gfx1100) / →15.19 GiB (gfx1151), routes Q4_K 103×`q4_k_t16`, Q5_K 131×`q5_k_t16`, Q6_K 24×`q6_k_qmicro_planar`, Q8_0 104×`q8_0_t16`. No perf claim owed (no runtime change); metric untouched → iteration logged. Artifact: `benchmarks/results/2026-09-25-zbook-e3-route-audit-repack-truth.json` |
| E4 Q5/Q4 decode owners | H4/H6 | — (census attribution only; no retained owner yet) | census window 0.9202 (86.24 vs 93.73 ms/tok graph-mode profiled, not a paired A/B) | 807.97 launches/token (plain 571.59) | — | re-census reproducibility PASS: top-6 kernels ≤0.2% vs E2d, launches/token identical, pure −0.14% | census recorded 2026-09-25 (iteration 7): trace split kernel 91.1% UD / 92.5% plain of wall, non-kernel ≈10.3 µs per launch UD / 11.4 plain — kernel pots rank first; re-rank (1) Q5 selected-down direct GEMV 20.47 ms/tok @242.9 µs/launch, no plain counterpart (both arms' tile8 ≈111 µs/launch; composition 131 vs 48 Q5_K), (2) Q4_K single local32 157.9 vs plain 119.5 µs/launch same T16 layout, (3) IQ4_XS local32 family 24.15 ms/tok (E9-adjacent); premise correction committed `73e5e4300`; artifact `benchmarks/results/2026-09-25-zbook-e4-decode-recensus.json`; each candidate stays open behind route-table test + paired A/B + E4a shape matrix |
| E4a Q5_T16 route to tile8 | H4 | **pending** (3 windows invalidated, lead directive: set pending and move on) | pending (same windows) | — (route unit; same tensors and launches/token, different owner) | — | **PASS**: rows=1 tile8==direct==`gguf_quant_gemv` bit-exact on 7 production shapes + control (GPU node 83.7 s), unit contract RED→GREEN, full guard green (suite+fixtures+3 smokes), `LLM.generate` route-probe conservation 20599 resolves unchanged | executed 2026-09-25 (iteration 8): five rows added to gfx1151 `GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE` (ffn_up 1.16x, ffn_down 1.28x, attn_gate 1.36x, attn_qkv 1.39x, attn_q 1.18x at rows=1; attn_v kept direct at 0.99x) — closes census item 1's route half (direct 242.9 µs/launch vs tile8 137–248 by shape); plain arm untouched; `ud_plain_parity` pending: windows invalidated 5.71/5.39/6.33% vs the 2% plain-prefill@512 visit check, 60-min idle did not settle plain_1; metric unchanged, iteration logged |
| E4b Q4_K single composition check | H4/H6 | — (adjudication unit; no retained change) | — (re-census window, not a paired A/B) | 807.97 (plain 571.59, both identical to E4) | — | census + quant-map evidence: same kernel symbol both arms, tile8 cross-arm speed-identity precedent (111.2/111.5), per-block gate/up quant map recorded in the artifact | adjudicated 2026-09-25 (iteration 9): item 2 **NEGATIVE — composition** (role/shape skew + 13 mixed-quant unpairable layers), not an owner defect; post-E4a effect confirmed (direct symbol out of top-12, Q5 family 25.22→24.02 ms/tok, UD pure 85.53→84.83 vs plain 79.81); re-rank item 3 = IQ4_XS family 24.25 ms/tok (UD-only, E9-adjacent); `ud_plain_parity` pending per lead directive (windows 5.71/5.39/6.33% vs 2% check); artifact `benchmarks/results/2026-09-25-zbook-e4b-post-e4a-recensus.json` |
| E4c IQ4_XS family composition check | H4 | — (adjudication unit; no retained change) | — (re-census window, not a paired A/B) | 807.97 (plain 571.59) | — | tensor-role map reconciles census: dual 19.4/tok = 20 pairable layers all fused (97%, remainder block-type), singles = inherent roles (down ×27, attention ×32) + 22 mixed layers, pairable count fully consumed | adjudicated 2026-09-25 (iteration 10): item 3 **NEGATIVE — composition**, no same-quant route gap; `<2,1>`/`<4,1>` split is shape-class selection with no same-shape alternative; **E4 complete** (item 1 route-landed E4a, items 2-3 negative E4b/E4c); mixed-layer sizing confirmed at exactly 28 (22 IQ4 + 6 K-quant) and handed to E6; `ud_plain_parity` pending per lead directive; metric unchanged, iteration logged |
| E6a same-quant pair routes | H9 | — (route screen; nothing enabled, no retained change) | — | 807.97 (unchanged by construction — route declined) | — | **PASS**: dual == 2×tile8+silu_mul chain bit-exact at (5120,17408) rows=1 (screen + existing unit oracle), allocate-once timing ×200, guard green, docs gates 4/4 | adjudicated 2026-09-25 (iteration 11): Q5_K same-quant dual **screened and REJECTED — 0.93×** (705.6 vs 653.6 µs/layer) since E4a moved singles to tile8; IQ4 pair-SiLU already firing (19.4/tok, E2); dormancy causes (poisoned Q4 variant inheritance, tile8 dispatch predicate, dead 5620 branch) + re-screen clearing command recorded in `docs/REFACTOR.md`; E6 step 1 closes with no same-quant route gap; `ud_plain_parity` pending per lead directive; metric unchanged, iteration logged |
| E6 gate/up + residual fusion | H9 | — | — | — | — | required | open |
| E7 GDN alpha/beta fused path | H10 | — | — | — | — | bit-exact lane if exact | open |
| E8 norm-cost attribution | H11 | — | — | — | — | n/a | open |
| E9 coop IQ prefill owners | H12 | — | — | — | — | bit-exact lane if exact | open |

¹ From the 2026-09-21 decode census (512-token prefill, 32 graph-replay
steps), not the 2026-09-23 re-check.

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