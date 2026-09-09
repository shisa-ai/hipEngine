# gfx1100 → gfx1151 Transfer Audit for Qwen3.8-27B (2026-09-09)

This audit lists the merged Qwen3.8-27B `Q4_K_M` gfx1100 work that the
gfx1151 pass must check, in the order the pass should check it. The release
plan it serves: finish the remaining gfx1100 optimizations, close gfx1100,
run this gfx1151 pass, then ship "finished" Qwen3.8-27B support on both
backends.

Two structural facts shape every row:

- **The gfx1151 backend shares the gfx11 device sources** and adds peer
  registrations/capabilities in `hipengine/kernels/hip_gfx1151/__init__.py`
  (`docs/KERNELS.md` backend table). "Shared source" means the kernel body is
  available; it does not mean the variant is admitted, default, or qualified
  on gfx1151. Each backend admits an independent subset, and all 2026-09
  qualification evidence is from the W7900 / RX 7900 XTX hosts.
- **gfx1151 is a unified-memory APU** (Radeon 8060S, 40 CUs, ~221 GB/s
  measured practical bandwidth, ~64-120 GiB pool; `docs/ROOFLINE-gfx1151.md`).
  There is no 24 GiB VRAM wall, so the gfx1100 capacity campaign's motivation
  does not transfer. What transfers from that campaign is decode/prefill
  *speed* and memory-lifetime hygiene, not token-count ladders.

Do not assume W7900-derived thresholds, floors, or tile choices are optimal
on gfx1151 (`docs/ROOFLINE-gfx1151.md` "gfx1151 preset" guidance). Every
"verify" row below means: run the gfx1151 measurement, do not inherit the
gfx1100 number.

## Status legend

- **Verify-admission** — shared-source kernel/route; check it is admitted,
  selected, and non-regressive on gfx1151 shapes.
- **Re-qualify** — route exists on both backends but its numerical/serving
  qualification is gfx1100-host-only; re-run the applicable gate on gfx1151.
- **Speed-only** — the gfx1100 motivation was the 24 GiB wall; on gfx1151
  only the performance aspect is relevant.
- **N/A-capacity** — capacity-ladder work whose premise (24 GiB VRAM) does
  not exist on the unified-memory APU. Confirm it does not regress; do not
  re-run ladders.
- **gfx1151-first** — work that is *easier* or already done on gfx1151; check
  the reverse direction instead.

## A. Decode path (c=1 and small-batch GEMV)

| # | gfx1100 work (evidence) | gfx1151 action | Notes |
| --- | --- | --- | --- |
| A1 | T16 rows=1 decode owners at/near the DRAM-bandwidth floor; nasone32 VDR load-reuse leaf screen rejected 1.11-1.44x slower (`benchmarks/results/2026-09-09-gfx1100-qwen38-q4km-vdr-load-reuse-rejected.json`) | **Verify-admission** | Re-derive the floor on gfx1151: kernel time vs weight bytes at the ~221 GB/s measured practical roof. If the gfx1151 T16 owners sit at that floor, decode GEMV is done and no GEMV donor can win; if they miss it, the gap is the top gfx1151 decode target. |
| A2 | Q4 selected gate/up pair-reuse dual owner retained on gfx1100/W7900 only (native-c8 +4.97%, 2026-09-05 audit packet D1; `docs/KERNELS.md` Q4/Q5/Q6 T16 selected row) | **Verify-admission** | The package floor `GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS` is gfx1100-capability data. Check whether gfx1151 admits the pair-reuse owner, and re-tune the geometry gate (x_rows/rows) for the 40-CU grid rather than copying the gfx1100 pins. |
| A3 | nasone32 RDNA3.5-guarded donors: dequant-float matvec, D=256 tile override (comparison doc "Optimization Review", dismissed for gfx1100) | **gfx1151-first** | Tracked as P3.2 in `docs/QWEN38-GFX1151-PARITY-CAMPAIGN.md`. These guards *select* on gfx1151. Low-priority confirmation screen against the A1 floor result; the gfx1100 VDR rejection is the prior. |
| A4 | strix-llama.cpp gfx1151-specific HIP optimizations | **gfx1151-first** | P1.2 (2026-08-29) audited the Q4_K_M-active deltas as Vulkan coopmat tuning with no HIP port required. Re-check only if strix-llama.cpp merged new HIP deltas after `5f851647`. |

## B. Prefill path

| # | gfx1100 work (evidence) | gfx1151 action | Notes |
| --- | --- | --- | --- |
| B1 | PP8192 gap attributed to projection prefill (~63% of timed prefill is T16 WMMA projection; GDN only ~8%) — worklog `pp8192-gap-attribution-8bb338` | **Verify-admission** | The gfx1151 prefill shape distribution differs (parity campaign P2/P2.3 fixed small-row bands). Re-run the kernel-time table on gfx1151 before assuming the same projection/GDN split. |
| B2 | Q8_1-activation integer MMQ bulk-prefill screen (top open gfx1100 perf item; see "Open gfx1100 items" below) | **gfx1151-first** | gfx1151 already has the qualified T2 composite `gguf_q4_k_q8_1_selected_prefill` (planar-Q6 K17408/N5120 down, K5120/N1024 narrow-V, rows 17-48; `docs/KERNELS.md`). If the gfx1100 screen produces a winner, port it back through the shared-source idiom rather than re-deriving it. |
| B3 | Oracle-free direct INT8 prefill attention screened, memory structure proven, blocked at 4.6x prefill cost (`benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-blocked.json`) | **Re-qualify after unblock** | The coder's INT8-PREFILL lane owns the kernel-speed fix on GPU1. Once the gate can flip on gfx1100, run the gfx1151 numerics/speed qualification; the APU's bandwidth profile may shift the break-even versus the oracle route. |

## C. KV storage, DMS, and capacity

| # | gfx1100 work (evidence) | gfx1151 action | Notes |
| --- | --- | --- | --- |
| C1 | DMS INT8 and DMS-BF16 codecs beat dense decode on W7900 (wave-grouped GQA producer, wave-group6 split-K, chunked keep-scan, in-place finalize, hoisted per-layer staging; commits `d1737d8dc`..`805cea3e3`) | **Re-qualify (speed)** | The speed win is the transferable part. Run the dense-vs-DMS A/B (`scripts/` A/B used in `28a72f6ad`) on gfx1151 shapes; the wave-group geometry (6-wave producer) is a W7900-tuned choice and needs a gfx1151 sweep. |
| C2 | `layer_outer` DMS prefill packing, single-plane hidden stream (default on), route-scoped hidden-plane alias, merged-lane ladder to 232,448 prompt tokens (commits `22ded5522`..`e68b34338`) | **N/A-capacity** | Motivation was the 24 GiB wall. Confirm the default-on single-plane stream does not regress gfx1151 speed; skip the ladders. |
| C3 | Non-DMS pure-INT8 route single-plane hidden alias, 131,072 → 155,648 tokens (`benchmarks/results/2026-09-09-rx7900xtx-int8-layer-outer-hidden-alias-capacity.json`) | **N/A-capacity** | Same as C2. Keep the gate's rollback semantics intact on gfx1151; no ladder. |
| C4 | INT8 KV storage routes (pure INT8 with FP32 scales; mixed-layout default) | **Re-qualify** | Qualification (numerics, no persistent BF16 mirror audit) is gfx1100-host-only. On gfx1151 the INT8 KV value proposition is bandwidth (halved KV bytes), not capacity, so qualify decode speed and the production-profile numerics gates, not token counts. |
| C5 | Capacity/concurrency ladders: oversubscribed queue at capacity 8, sustained-horizon D128/D24/D512 budgeting, DMS INT8 edge/requested ladders (2026-09-06/07 artifacts) | **N/A-capacity** | Re-budget only if the gfx1151 serving pool gets its own declared-capacity policy; the APU pool is shared with the host, so the budgeting model itself differs (GTT/HSA, not dedicated VRAM). |

## D. MTP / speculative decode

| # | gfx1100 work (evidence) | gfx1151 action | Notes |
| --- | --- | --- | --- |
| D1 | C1 MTP scratch-lifetime and graph-fallback repair; native C1 context qualification harnesses; capacity-bound dense C1 native verification (commits `6c01f1f1c`, `fea8425e6`, `6971b5bb9`) | **Re-qualify** | Run the same native C1 harnesses (4K/8K/16K target checks, 8K/33-output B3) on gfx1151. The repair is runtime logic and should transfer, but the graph-bucket admission is per-backend capability data. |
| D2 | C1 MTP 63.48 tok/s / 1.769x true-AR baseline beats nasone32 (comparison doc) | **Verify-admission** | gfx1151 MTP has its own campaigns (scaling, structural differential). Do not import gfx1100 MTP numbers; the gfx1151 gate is MTP >= 1.15x own AR at every width C1-C8 (scaling campaign goal 1). |

## E. Measurement playbook for the gfx1151 pass

Follow `docs/ROOFLINE-gfx1151.md` section 6.1 and the `docs/ROOFLINE.md`
decision tree; do not optimize against a single counter.

1. `rocprofv3 --kernel-trace` first. Rank the kernel-time table before
   choosing any target ("do not assume attention or GEMV is the bottleneck
   without the kernel-time table").
2. Classify each top family into a regime: memory-bound decode GEMV, compute-
   bound prefill WMMA, or launch/stack-bound small shapes.
3. Check each family against its own roof: weight bytes vs ~221 GB/s measured
   practical bandwidth for decode; 59.4 TFLOP/s BF16 WMMA (INT8 same class,
   INT4 2x) for prefill; wall vs kernel-sum vs launch-count decomposition for
   serving shapes (the P1.3/P3.1 method).
4. Only for families that miss their roof: targeted `pmc: SQ_WAVES` plus VGPR,
   scratch, and LDS audit from the trace. Keep CU vs WGP units straight (40
   CUs = 20 WGPs; fewer than 40 resident workgroups cannot fill the GPU).
5. rocprof-compute on gfx1151 is conditional: GL0/max-mclk issues and
   RDNA3.5 support marked upcoming (`docs/ROCM-AI.md`). Do not build a claim
   on its counters without the corrections noted there.
6. Profiling discipline: prebuild `.so` caches outside the profiler and run
   with a compiler-version file plus `require_cached`; never compile JIT code
   inside a profiled child (`docs/KERNELS.md` trap).
7. Admission: use the busy-card/ownership guard before any gfx1151 benchmark
   (the 2026-09-08 external-benchmark contamination is the recorded
   counterexample).

## F. Reporting rules for the pass

- gfx1151 and gfx1100 are independent lanes. Never report a gfx1151 absolute
  rate as an old→new comparison against a gfx1100 number (AGENTS.md evidence
  policy); the gfx1151 comparators are the frozen C1-C8 external-engine
  cells, and per-family roofline headroom.
- Every retained gfx1151 result needs the standard artifact under
  `benchmarks/results/` plus `benchmarks/README.md` and
  `benchmarks/CHANGELOG.md` rows naming the physical gfx1151 host.
- MTP speedup claims need the full multi-prompt suite and a true same-protocol
  no-MTP baseline (AGENTS.md anti-gaming); verifier-derived rows are
  diagnostic only.
- Do not reopen gfx1151 AR decode: it leads C3-C8 and its C1/C2 blockers are
  closed under the parity campaign's named-blocker rule (scaling campaign
  scope note). This audit adds checks, not re-litigation.

## G. Open gfx1100 items that gate the handoff

These must reach a verdict (retained / rejected / parked-with-blocker) before
gfx1100 is declared closed; each then enters this audit as a row above.

1. **Q8_1-activation integer MMQ bulk-prefill screen (GPU0 candidate).**
   nasone32-style raw-Q4_K + Q8_1 integer MMQ with the `efa4e8641` load-reuse
   idiom versus the retained float T16 WMMA prefill owners, on the Qwen3.8-27B
   bulk-prefill shapes (dual+SiLU gate/up pair, shared-B down). The
   2026-09-09 VDR leaf kernels are the load-reuse starting point; the
   gfx1151 `mmq64x64` composite (B2) is the consumer reference. Prior
   boundary: the 2026-08-12 source-shaped Q4T16 integer-MMQ rejection
   (`benchmarks/results/2026-08-12-qwen36-27b-q4t16-source-shaped-dense-mmq-rejected.json`)
   bars repeating the T16-payload I128/J128 or I64/J64 dataflow; this screen
   is only worthwhile because the raw-Q4_K + load-reuse consumer is a
   materially different dataflow.
2. **INT8 prefill attention kernel speed (coder, GPU1).** Tile/split-K the
   direct INT8 prefill kernel toward AOTriton-competitive speed, then the
   production-profile numerics campaign, then the capacity ladder re-run
   (worklog `int8-direct-prefill-capacity-b1f842` "Next").
3. **TG-gap attribution.** Projections are exonerated by the VDR screen;
   attention decode at long context and GDN decode remain unprofiled for the
   TG128 34.31/29.82 vs 36.67/35.70 deficit (worklog
   `engine-comparison-donor-screen-c608a0` "Next").
