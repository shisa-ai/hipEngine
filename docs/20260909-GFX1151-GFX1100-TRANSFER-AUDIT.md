# gfx1100 → gfx1151 Transfer Audit for Qwen3.8-27B (2026-09-09)

This audit lists the merged Qwen3.8-27B `Q4_K_M` gfx1100 work that the
gfx1151 pass must check, in the order the pass should check it. The release
plan it serves: finish the remaining gfx1100 optimizations, close gfx1100,
run this gfx1151 pass, then ship "finished" Qwen3.8-27B support on both
backends.

**Current release checklist:** [section K](#k-v050-scope-and-identity-decision),
which supersedes [section J](#j-current-v050-release-checklist) as the live list
by fixing the pass's scope and dispositioning section J's items under it.
Sections A-I preserve the earlier audit and its dated corrections; section J
records the September 12 source/evidence pass; section K is the v0.5.0 scope
decision and the identity ruling that closes audit finding 3.

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

## H. Status after the 2026-09-12 review

Reviewed at commit `f3b20c1f2` on the physical gfx1151 host (Ryzen AI MAX+ 395 /
Radeon 8060S, ROCm `10.0.0`, `GPU_MAX_HW_QUEUES=2`).

What this review settled:

- **A1 (decode GEMV floor) is closed by measurement.** Dense decode streams about
  89% of the practical read roof, so the row no longer gates anything.
- **C4 (INT8 KV) is closed as a rejection, reproduced.** The `2026-08-15`
  gfx1151 quality failure reproduces to the full printed precision of that
  artifact on today's kernels, so it is representation-owned and no gfx1151
  kernel work can be expected to clear it.
- **B2 (integer MMQ) is reversed.** gfx1151 is the donor, and the open item is
  on the gfx1100 side.
- **D1 narrowed to one named key**, and the whole gfx1100-only surface on the
  Qwen3.8-27B dense path is five enumerated variants.
- **New finding:** the PARO runtime pins its KV/attention resolution to the
  gfx1100 backend key, so on gfx1151 it cannot reach the one gfx1151 override
  that changes the reduction rather than the geometry. Tracked in
  `docs/REFACTOR.md`.

### The gfx1100-only inventory is mechanical, not a reading exercise

`hipengine/kernels/hip_gfx1151/__init__.py` aliases the whole gfx1100 key space
and then subtracts an explicit list. At this commit the two backends hold
**1309 shared variants, 103 gfx1100-only variants, 7 gfx1151-only variants,
239 declared exclusions (`_GFX1151_ALIAS_EXCLUSIONS`), and 18 body overrides
(`_GFX1151_OVERRIDES`)**. The declared exclusion list is larger than the
gfx1100-only set because it also names keys that gfx1151 never had, and because
the 18 overrides keep their key while replacing the body. Reproduce:

```bash
python3 - <<'PY'
import collections, importlib
from hipengine.kernels.registry import registered_keys
for m in ("hipengine.kernels.hip_gfx1100", "hipengine.kernels.hip_gfx1151"):
    importlib.import_module(m)
by = collections.defaultdict(set)
for k in registered_keys():
    by[k.backend].add((k.layer, k.quant, k.variant))
a, c = by["hip_gfx1100"], by["hip_gfx1151"]
print(len(a & c), "shared;", len(a - c), "gfx1100-only;", len(c - a), "gfx1151-only")
PY
```

The 103 gfx1100-only variants are concentrated off the dense path: by layer they
are 46 `linear`, 19 `moe_linear`, and 38 spread across 24 other layer keys (at
most five variants each); by quant they are 24 `gguf_q5_k`, 23 `gguf_q6_k`,
15 `gguf_iq3_xxs`, and 14 `bf16`. Every gfx1100-only variant carries a written
reason in the gfx1151 file, so this is deliberate scope, not drift.

Exactly **three** gfx1100-only variants sit on the Qwen3.8-27B dense path, and
each one has a named gate that would clear it. Two more were on this list until
2026-09-12, when their named gates were satisfied and the keys were admitted:
`linear / gguf_q4_k_t16_v1 / dense_rowtile_col4_bf16_bf16_out` and
`linear+residual / gguf_q4_k_t16_v1 / dense_rowtile_bf16_residual_bf16_out`
([qualification artifact](../benchmarks/results/2026-09-12-gfx1151-qwen38-27b-q4km-dense-rowtile-withheld-variants-qualified.json)).

| Key | Recorded reason |
| --- | --- |
| `linear_attn_alpha_beta+chain_conv+snapshot / f32 / bf16_k5120_n48_c10240_k4_exact_state_rows_tloop` | Screened only on gfx1100; gfx1151 keeps three independent leaves |
| `linear_state_pair_copy / f32 / chunked_i32` | W7900-only until gfx1151 receives independent transaction and launch-overhead gates |
| `paged_kv_write / gguf_q4_k_m / mixed_bf16_shared_batch_spans` | Qualified only for the W7900 dense-H5120 N1 graph; gfx1151 keeps scalar append aliases |

The two admitted keys keep their shared bodies rather than an override, and
their strict fallbacks (`dense_rowtile_bf16_bf16_out`, and the rowtile +
`gguf_bf16_add` chain for the composite) stay registered.

The 18 overrides are the healthy pattern — gfx1151 keeps the key and swaps in a
40-CU-tuned body (`gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out`,
`qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16`,
`qwen35_router_logits_bf16_f32w_auto_256`, and so on). One override changes the
*reduction* rather than the geometry, and it is the one PARO cannot reach; see
the `docs/REFACTOR.md` entry and the note below.

The rows below are the ones whose status changed.

### A1 (decode GEMV floor) — closed by measurement

Dense decode is at the memory roof, so this row no longer gates anything. The
standard `Q4_K_M` artifact carries 16.091 GB of AR-active weights per token
(GGUF tensor table: 17.096 GB total, minus the 0.290 GB `blk.64` NextN/MTP
block, minus the full 0.715 GB embedding table, plus one 5120-row embedding
read). Measured AR decode is **12.213 tok/s** at 512/128, so the kernel streams
**196.5 GB/s ≈ 88.9%** of the ~221 GB/s practical read roof in
`docs/ROOFLINE-gfx1151.md` §3.2. The residual ~11% is shared with attention,
GDN, sampler, and launch gaps, so it is not recoverable from the GEMV owners.
Evidence: [`Q4_K_M AR/prefill refresh`](../benchmarks/results/2026-09-12-gfx1151-qwen38-27b-q4km-ar-prefill-refresh.json).
The published `Q4_K_S` lane lands in the same place (13.069 tok/s over a
16.12 GB file ≈ 95% of roof).

### C4 (INT8 KV re-qualification) — re-checked on current kernels, rejection confirmed

On gfx1151 the INT8 KV value proposition is **bandwidth, not capacity** (there
is no 24 GiB wall), and the lane currently has none of it:

- `hipengine/models/qwen35.py` records the gfx1151 `int8_per_token_head`
  capability as **rejected** on `2026-08-15` ("minimum-prompt top-1 agreement
  0.7778 is below the 0.90 gate" at 1024/8), so `resolve_kv_capability`
  resolves the request to `effective_kv_storage="bf16"` with
  `runtime_action="fallback_bf16"`. The gate is fail-closed and the reason is
  recorded, which is correct behavior.
- gfx1100 records the **same** quant/KV/scale contract as **qualified** with
  scope `explicit_no_mirror_direct_c4` (`max_direct_rows=4`), and the gfx1100
  route has since been through the per-layer oracle repair (`2026-09-10`) and
  the physical-c4 promotion (`2026-09-11`).
- **Re-checked on 2026-09-12 and the rejection holds.** The run used the same
  artifact (the rejection's recorded `artifact_sha256` is the file present on
  this host) and the same suite (its recorded `prompts_sha256` is the current
  suite), through `scripts/qwen35_native_mixed_kv_suite.py --backend hip_gfx1151
  --diagnostic-kv-capability --candidate-kv-storage int8_per_token_head
  --kv-scale-dtype fp32 --require-no-bf16-mirror`. 512/8 passes (mean KL
  2.19e-04, max KL 1.23e-02, top-1 1.0, all 11 prompts). 1024/8 fails, and it
  fails with **every aggregate and per-prompt statistic reproducing the
  `2026-08-15` rejection to that artifact's full printed precision** — same two
  failing prompts (`mixed_ja_en_review`, `mixed_v1`), same mean KL
  `0.041977959056962555`, same max KL `3.446476164561575`, same minimum top-1
  `0.7777777777777778`. That is across the paged-attention geometry pins and
  small-row owner work landed since August, and across a compiler change from
  AMD clang 23.0.0git to HIP 7.15.26333.
- The failure is **representation-owned, not kernel-owned**, and it is
  localized: nine of eleven prompts pass with a worst max-KL of 9.37e-04, and
  the whole failure sits in one mixed-language review prompt and one synthetic
  mixed prompt. The `2026-08-15` protocol declares those prompts
  heldout/control for map selection, so this is not clearable by reselecting a
  layer map. Leave the capability rejected and re-open only on a materially new
  input-independent representation signal. Artifact:
  [`INT8-KV re-check`](../benchmarks/results/2026-09-12-gfx1151-qwen38-27b-int8-kv-recheck-rejected.json).
- INT8 K/V is therefore **not** a gfx1151 lever for this artifact, and the
  remaining decode-bandwidth headroom stays where A1 put it.
- The two lanes also measured **different model files** for the same nominal
  quant (`7e78da5d…c6fe169`, 17,106,775,008 bytes on gfx1151 versus
  `7b2aec3b…cc89f1b`, 17,106,773,984 bytes on gfx1100), so a gfx1100 result
  does not by itself transfer to gfx1151.

### D1 (MTP graph-bucket admission) — narrowed to one named key

The transferable gfx1100-only MTP item is a single registration:
`paged_kv_write / gguf_q4_k_m / mixed_bf16_shared_batch_spans`, the W7900
dense-H5120 N1 **graph** verifier KV-batching owner (row 5 of the table above).
gfx1151 excludes it and keeps scalar append aliases. Everything else in the D1
row is per-backend capability data already read through
`backend_package_capability` (for example
`GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS` is `8` on both backends).

### New finding: PARO on gfx1151 cannot reach the gfx1151 paged-attention body

`hipengine/runtime/qwen35_paro.py` pins `_PAGED_KV_REGISTRY_BACKEND =
"hip_gfx1100"` (line 213) and passes it to all four of its KV/attention resolve
sites: `resolve_paged_kv_write` (~2122), `resolve_paged_attn_decode` (~2259 and
~4212), and `resolve_paged_attn_prefill` (~3778). The `Qwen35ParoDecodeState`
docstring states the assumption outright ("Kernel selection still flows through
the registry/wrappers added in the gfx1100 backend tree"), and PARO is a live
gfx1151 lane, so this is on the shipped path.

Concretely it defeats the single semantics-changing override in the table
above: `paged_attn_decode / w4_paro / bf16_context_batch_c1_exact_spans`
resolves to the gfx1100 c1-exact kernel, while every other gfx1151
paged-attention caller gets
`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans` — the body the
gfx1151 package pins to "the c4/c8-proven 256-thread shape" on the generic
reduction. Both bodies are validated, so this is a cross-backend
arithmetic-consistency risk rather than a known defect, but it is the kind of
divergence a gfx1151 PARO-vs-GGUF paged-attention gate would flag. Tracked in
`docs/REFACTOR.md` with its removal condition.

### B2 (integer MMQ) — the direction is reversed, and the gfx1100 side is the open one

gfx1151 owns `GGUF_Q6_DENSE_INTEGER_MMQ_PREFILL_POLICY` (planar Q6,
`t16_q8_1_planar_integer_mmq64x64`, rows 17-48 on two shapes) and the
`t16_q8_1_planar_integer_mmq64x64_bf16_bf16_out` registration that gfx1100 does
not have. The gfx1100 Q8_1-activation MMQ screen therefore remains an
**open gfx1100 item**, and gfx1151 is the donor rather than the recipient.

### A2 (selected pair-reuse geometry) — still open

gfx1151 admits the Q4 selected-expert pair-reuse dual owner
(`GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8`) but the owner's geometry
(`x_rows=8/rows=64`) is shared source pinned by a gfx1100 measurement, and the
40-CU grid is not the 96-CU grid it was tuned on. This is a MoE
(Qwen3.6-35B-A3B) item, not a Qwen3.8-27B one, and it stays low priority.

### A3, A4, B1, B3, C1, C2, C3, C5, D2 — unchanged

No new evidence in this review. C1/C2/C3/C5 remain N/A-capacity or
speed-only rows, and D2 stays governed by the gfx1151 scaling campaign's
own-AR rule.

### Reproducibility notes for the published gfx1151 rows

- The `Q4_K_M` lane is reproducible on this host: the file at
  `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` matches the evidence identity
  `7e78da5d…c6fe169` exactly.
- The published `Q4_K_S` row is **not** reproducible here: its artifact
  (`22200efcd98a7aeeaf83f59b0f1400b055d9e0437900e26b930ef2d42a3eb3f9`,
  16,121,359,328 bytes) is absent, and the local `UD-Q4_K_S` file is a
  different artifact (`75bc9c8a…`, 15,358,213,024 bytes). Re-running that row
  needs the original file re-fetched.

## I. Pin inventory for the Qwen3.8-27B dense path (2026-09-12, commit `7ed5c5559`)

Amended 2026-09-12 at commit `eb7236fb5`: the highest-value target below was
reframed and closed by a same-file comparator measurement. The pin counts and
the six-mechanism table are unchanged.

Section H established that the gfx1100-only *registry* surface is mechanical and
carries a written reason per key. This section does the same for the *capability*
surface and separates the mechanisms, because they have different clearing costs.
The registry counts below are re-derived at commit `7ed5c5559` and are unchanged
from section H.

### Two surfaces, counted

The registry surface, reproduced with the snippet in section H: **1309 shared,
103 gfx1100-only, 7 gfx1151-only, 239 declared exclusions, 18 body overrides.**

The capability surface is separate and larger. `backend_package_capability`
reads a module-level constant from the backend package by name, so a name defined
on one backend and not the other silently takes the call site's default. Counting
every `GGUF_*` name read by a module on this path:

| Relation between the two backend packages | Count |
| --- | ---: |
| Defined on gfx1151 only | 50 |
| Defined on both, different value | 31 |
| Defined on both, same non-default value | 23 |
| Defined on gfx1100 only | 18 |
| Absent on both (the call-site default governs) | 2 |
| **Total distinct names** | **124** |

The dense path is **net ahead on gfx1151**: it defines 50 capabilities that
gfx1100 does not, against 18 in the other direction. The `hip_gfx1100` literals
in the runtime are nominal source markers (route map in `docs/KERNELS.md`), and
they are not evidence of a gfx1151 deficit.

### Six mechanisms, and which a gfx1151 change can clear

| Mechanism | On this path | Pin lives in | Clearable from the gfx1151 package alone? |
| --- | ---: | --- | --- |
| Live code pin | 1 | `runtime/qwen35_paro.py:213` | no — measured load-bearing, see below |
| Registry exclusion | 3 of 103 | `_GFX1151_ALIAS_EXCLUSIONS` | yes |
| Capability gate defined on gfx1100 only | 18 | the gfx1100 package | yes — define the name on gfx1151 |
| Capability gate declined on gfx1151 | 6 | the gfx1151 package | yes, but three are measured rejections |
| Geometry pin | shared device source | the `.hip` launcher | **no** |
| Row-count floor or ceiling | 13 names (9 differ) | both packages | yes — it is a per-backend value |

Only the geometry pin is structural. The 103 registry exclusions each carry a
written reason in the gfx1151 package; **none of the 18 gfx1100-only capability
gates appears anywhere in the gfx1151 package** — no constant, no comment, no
exclusion entry. Twelve of the 18 do carry a gfx1100-side comment, but it states
the gfx1100 qualification rather than a gfx1151 verdict, and the remaining six
have no comment at all. That asymmetry is the inventory's main defect: a name that
exists only on gfx1100 reads as an ordinary default at every call site, so a
gfx1151 regression from its absence is invisible in review.

### The PARO pin is load-bearing, not a stale default

The `_PAGED_KV_REGISTRY_BACKEND = "hip_gfx1100"` pin at
`runtime/qwen35_paro.py:213` was cleared and re-tested on 2026-09-12 at commit
`fa4209f79`. Threading the session backend into the four paged-KV / paged-attention
resolutions is a one-line change that reaches the gfx1151 override at
`hip_gfx1151/__init__.py:3199`, and it **fails**:

| Arm | c=2 / p512 / d128 native_batch vs independent c1 |
| --- | --- |
| Session backend threaded (gfx1151 `fixed256` body) | **25/137**, row 0 diverges at decode token 25 |
| Session backend threaded, repeat run | **25/137**, bit-identical to the first |
| Historical gfx1100 pin (`c1_exact` body) | **137/137** |
| `per_row` control | **137/137** |

Measured on the physical gfx1151 host with Qwen3.6-35B-A3B PARO. The gfx1151
override is correct for the GGUF route that registers it and wrong for this PARO
path, which keeps its own batch-vs-c1 equality contract on the gfx1100 c1-exact
body. The pin now carries that measurement as a source comment and is guarded by
`tests/test_unit_qwen35_decode_state.py::test_qwen35_decode_state_paged_kv_backend_stays_pinned_to_gfx1100`.
The clearing cost in the table above is therefore **no**, not yes: this is the
second mechanism on this path that a gfx1151 package change cannot safely reach,
for a measured reason rather than a structural one.

The 18 gfx1100-only gates are not spread evenly. Eight cover prefill
(`GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES`, `GGUF_Q4_T16_GROUPED_PAIR_ROWS6_POLICY`,
`GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS`, `GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES`,
`GGUF_RAW_K_PREFILL_ROLE_VARIANTS`, and the three `GGUF_T16_F16_ROCBLAS_*` names),
eight cover the SPECDEC2 verifier (`GGUF_SPECDEC2_*`), and two are unclassified
(`GGUF_C8_Q5_RAW_MMQ_SSM_OUT`, `GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS_BY_MAX_SEQUENCE_LENGTH`).
One of the prefill eight is not this artifact's: the raw-K prefill family admits
quants `{gguf_q8_0, gguf_q5_k, gguf_q6_k}` and never `gguf_q4_k_t16_v1`, so
`GGUF_RAW_K_PREFILL_ROLE_VARIANTS` is N/A for Qwen3.8-27B `Q4_K_M` and matters
only to raw-Q5/Q6 artifacts. Every one of the 18, plus
`GGUF_SPECDEC2_MTP2_PHYSICAL_C1`, now carries a gfx1151 verdict in section K's
ledger; that ledger is the replacement for this paragraph's grouping.

### The geometry pin is in shared device source, not in a package

The A2 row's geometry gate resolves to `launch_q4_dual_pairreuse_direct` in
`kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.hip:6868`, which fixes
`block(128)`, caps `rows` at 64, and requires `rows % x_rows == 0`. gfx1151
admits the owner at the same floor (`GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8`
on both backends) and does not override the body, so the 40-CU part runs the
96-CU part's launch geometry. This is the one pin a gfx1151 package change cannot
reach: retuning it means either editing shared source that gfx1100 also uses, or
registering a gfx1151 override body as the 18 existing overrides do.

### Dense prefill scratch policy diverges by file type, not by backend

The two dense scratch gates are keyed on `(geometry, file type)`, and the two
backends populate different file types:

| Capability | gfx1100 | gfx1151 |
| --- | --- | --- |
| `GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES` | `MOSTLY_Q4_K_M`: `min_rows=1`, `priority_min_rows=4096`, `hidden_inplace_min_rows=4096`, `priority_min_live_stages=5` | absent |
| `GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES` | `MOSTLY_Q4_K_M`: 1,024-row ceiling at capacity ≥1,024 | `MOSTLY_Q4_K_S`: 4,096-row ceiling at capacity ≥4,096, 1,024 at ≥8,192 |

Both consequences land on the Qwen3.8-27B dense path:

- **Dense scratch liveness aliasing is off on gfx1151.** The dense branch reads
  `GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES` and returns `None` when the
  geometry/file-type key is missing. gfx1151's
  `GGUF_PREFILL_SCRATCH_LIVENESS_MIN_ROWS = 768` is read only in the MoE branch,
  so it does not substitute here.
- **A Q4_K_M request gets no row ceiling on gfx1151.** The ceiling exists on
  gfx1100 specifically to stop a 4,096-row auto query chunk from overrunning
  metadata buffers sized at allocation (the 2026-09-09 INT8 comparison-protocol
  crash recorded in `_dense_prefill_scratch_row_cap`'s docstring). On gfx1151 the
  dict has a `MOSTLY_Q4_K_S` key instead, so the same file type resolves to
  `None` — meaning no clamp, not a tighter one.

Neither is a measured gfx1151 result; both are consequences of the file-type key.
They are the two concrete items a gfx1151 change can clear without touching
shared source.

### Highest-value gfx1151 targets, in order

1. **Give the Q4_K_M lane its own published comparator — closed 2026-09-12, and
   the original framing of this item was wrong.** This item previously read that
   the `352.426/364.443/367.993` prefill cells "come from the Q4_K_S campaign"
   and that the Q4_K_M lane therefore had no same-artifact external cell. Both
   statements are false. Those cells come from
   `benchmarks/results/2026-08-15-gfx1151-qwen38-27b-p0-baseline.json`, whose
   `model` block is the Q4_K_M file, sha256 `7e78da5d…c6fe169` — the same
   artifact this section measures. The Q4_K_S campaign borrowed them, and says
   so in its own key names (`llama_hip_q4km_prefill_tok_s`), while measuring a
   separate `same_file_llama_prefill_512` for its own file. The P0 rows also
   carry `true_ar_tok_s` (`12.151/12.065/11.508`) and a uniform
   `token_sha256_i64`, so the Q4_K_M lane had a token-exact same-file comparator
   on **both** axes. What it lacked was publication, which the 2026-09-12
   refresh supplies: [`same-file llama.cpp comparator`](../benchmarks/results/2026-09-12-gfx1151-qwen38-27b-q4km-same-file-llama-comparator.json).

   The real finding is the opposite of a gap. On this file hipEngine does not
   lead llama.cpp HIP at every working shape: it trails prefill at 4K/128 by
   1.73% at the explicit-token-array tier and by 3.49% at the `llama-bench`
   tier, and it trails llama.cpp Vulkan decode at every shape by 4.1-6.4%. The
   `Q4_K_S` section's "beats the llama.cpp backends at every working shape"
   prose remains true for `Q4_K_S` against this comparator, which is the harder
   prefill baseline of the two, but it must not be read as covering the Q4_K_M
   row.
2. **Close the 18 undocumented capability gates.** Each needs a recorded gfx1151
   verdict — retune, decline, or N/A — at the same standard the 103 registry
   exclusions already meet. This is a ledger task.
3. **Settle whether the missing Q4_K_M row ceiling is safe.** This is the one item
   on the list that is a correctness question rather than a ledger or benchmark
   one: gfx1100 clamps this file type to 1,024 rows specifically to keep a
   4,096-row auto query chunk inside metadata buffers sized at allocation, and
   gfx1151 does not. Either the gfx1151 owner-slots arena removes the hazard, or
   it does not and the clamp needs a `MOSTLY_Q4_K_M` key.
4. **The pair-reuse geometry (A2)** stays the lowest-priority structural item and
   remains a Qwen3.6-35B-A3B MoE concern, not a Qwen3.8-27B one.

Decode is not on this list. A1 closed it at 88.9% of the practical read roof, and
C4 closed the INT8 K/V route as representation-owned.

## J. Current v0.5.0 release checklist

Updated September 12, 2026 against `1c9f2471c61b00199c8d47481db75c3e854ebe6c`,
including the pull from local `6150ace4b`. This is a source/evidence audit,
not a new gfx1151 GPU qualification. **The transfer review is not fully
closed.** Shared source is not independent backend qualification, and an
unqualified optimization need not block release when its existing fallback
is safe and the limitation is explicit.

### Release safety and default-behavior checks

- [ ] **Qualify or backend-gate inherited Q4 fused-prefill retiles.**
  `hipengine/runtime/gguf_linear.py::_q4_t16_dual_silu_retile_enabled`
  defaults on, and fused-pair dispatch selects row64/row128 variants when
  registered. gfx1151 aliases both registrations without an override or
  exclusion. The [W7900 promotion artifact](../benchmarks/results/2026-08-31-w7900-q4km-fused-q4-prefill-retiles-retained.json)
  instead says peer backends retain prior owners. Resolve this actual
  source/evidence scope mismatch with independent gfx1151 shape/model,
  numerical and performance gates or backend-owned admission. The later
  row48 capability gate does not gate row64/row128. No gfx1151 numerical
  regression is demonstrated by this source audit.
- [x] **Resolve the production-default / automatic-MTP interaction.**
  The [current socket and route packet](../benchmarks/results/2026-09-12-gfx1151-qwen38-serving-mtp-closure.json)
  confirms that omitted-profile production runs AR, including explicit MTP
  requests. The FP32 production manifest does not match historical FP16 MTP
  certificates. Strict C1/K3 remains available at capacity1/4, context1-67,
  natural25 and greedy sampling; K4 and ignore_eos correctly fall back.
  Three complete category runs and blocking/SSE cancellation/refill pass.
- [x] **Verify the shipped production composition in the declared greedy scope.**
  FP32 state replaces the failed unbounded FP16 default; planar-Q6 integer-MMQ
  is confined to target verification after an AR-prefill scope failure.
  The [final public packed gate](../benchmarks/results/2026-09-12-gfx1151-qwen38-public-packed-final-qualified.json)
  passes 8,716 teacher-forced rows at KL0/top1 100%, with three repeats,
  static C2/C4/C8, sparse slots, C8-to-C1 retirement and p512/D128.
  The [headline preflight](../benchmarks/results/2026-09-12-gfx1151-qwen38-final-headline-refresh.json)
  adds exact graph/eager IDs, final logits and states over 18 prompts.
  Real-socket blocking/SSE and cancellation/refill pass and allocations drain.
  This does not claim arbitrary long-context MTP, non-greedy sampling, SLO
  certification, or full-repository release closure.
- [x] **Decide the Q4_K_M scratch-row clamp question from section I.**
  Section K records the no-clamp decision and focused selection-invariant
  tests. This closes the known allocation/chunk mismatch at source level;
  the confirming 1K/4K/8K GPU boundary matrix, including tails and packed
  requests, is still outstanding and is not claimed as passed.
- [ ] **Audit the newly default-on layer-outer route's reachable scope.**
  `7444dd705` promotes `HIPENGINE_GGUF_PACKED_LAYER_OUTER=1` globally, with
  W7900 INT8 evidence. Establish which gfx1151 artifact/KV combinations can
  actually enter the one-shot or resumable executor, and assert that rejected
  Q4_K_M INT8 falls back to BF16 without engaging it. Any reachable INT8/DMS
  diagnostic scope needs independent layer-boundary state/KV, ragged/tail,
  hidden-alias, cancellation/allocation-failure, decode-handoff and speed
  checks before qualification. Validate executor telemetry from the leased
  session after `1d02aa05f`; do not mistake a fallback run for this gate.
  Include `_gguf_int8_prefill_slot_local_aotriton_enabled`: it also defaults
  on globally, explicitly changes reduction order, and cites W7900-only
  qualification. Ordinary BF16 reachability must be distinguished from
  transient-oracle INT8 reachability for both settings.
  Keep native sampling outside a completion claim until the documented
  [zero-capacity packed-sampler failure](../worklog/entries/20260911T220357.309195Z-lhl-p6-native-sampling-blocked-16d72c.md)
  is repaired and independently checked. Its existing failure is not a
  measured gfx1151 reproduction, and a skipped sampling arm is not a pass.
- [ ] **Close release validation and wording gaps.**
  [Test migration](testing/TEST-MIGRATION.md) still records eight source-pin
  failures and a failing published-command gate, and explicitly does not
  claim full GPU/live execution. Repair the focused failures without blind
  hash refreshes; the milestone/release run must use `--suite all`, not
  unit-only default discovery. Scope `CHANGELOG.md`'s INT8 c4 and layer-outer
  gains to their measured artifact/backend, and reconcile its broad
  "Everything below was tested ... on ... both" introduction and automatic
  MTP wording with the actual per-backend evidence.
  Refresh `docs/KERNELS.md`'s small-row single-wave description against the
  actual gfx1151 low-VGPR/shared-B2W2 override.

### Optimization transfer decisions still open

- [x] **Give all 18 gfx1100-only capability settings a gfx1151 verdict.**
  Closed by the section K ledger, which covers those 18 plus
  `GGUF_SPECDEC2_MTP2_PHYSICAL_C1`: qualified, rejected with evidence, pending a
  named gate, or N/A for this artifact, one row per name. The applicable dense
  prefill pair/row48 settings, scratch liveness, the three F16 rocBLAS settings,
  and the verifier settings are all in it, and no W7900 threshold was copied.
  Re-review the [unequal Q4 pair decision](../worklog/entries/20260815T191659.162367Z-pi-qwen38-gfx1151-p4-unequal-q4-pair-69a5bb.md):
  its exact positive leaf screen was declined before integration solely
  below a projected 1% request-saving threshold. The ledger's verdict for that
  setting is "declined" on the gfx1151 default, so today's small-win policy
  still warrants an integrated non-regression decision rather than automatic
  rejection under that historical threshold — the ledger records the reopen
  condition, it does not settle the measurement.
- [ ] **Finish the three remaining dense-path registry decisions.**
  Fused alpha/beta+conv+snapshot, chunked state-pair copy, and N1 graph
  batched KV append remain excluded and need their transaction/state and
  graph/launch gates. The Q4 col4 and residual rowtiles are admitted with
  shape crossover and full-model evidence in section K; their admission
  packet now links the explicit CPU-reference floor and cached-only gfx1151
  actual-weight rocprofv3 trace. Those two evidence gaps are closed.
  Every candidate also needs the applicable
  production review if strict equality fails, plus a same-host performance
  decision and strict fallback.
- [ ] **Close prefill and optional DMS follow-ups with scoped verdicts.**
  Refresh the gfx1151 prefill family trace before choosing a transfer
  target. Direct INT8 prefill remains conditional on representation
  admission and a speed premise; DMS speed/lifetime checks are separate
  from public dense BF16 qualification. Do not rerun the XTX capacity
  ladders on the APU. The selected-expert pair-reuse geometry is a
  Qwen3.6-35B-A3B MoE follow-up, not this dense release checklist.
- [ ] **Make every published artifact reproducible or name the limitation.**
  The Q4_K_M same-file comparator is closed, but section H records the
  original Q4_K_S artifact as absent from this host. Re-fetch it for any
  claimed current-head Q4_K_S refresh; local UD-Q4_K_S is not a substitute.

### Reviewed outcomes not to reopen without a new premise

- [x] Registry inventory reproduced on this HEAD with `.venv/bin/python`:
  **1309 shared, 103 gfx1100-only, 7 gfx1151-only**. Most exclusions are
  outside dense Qwen3.8; do not describe all 103 as missing dense ports.
- [x] gfx1151 has its own prefill/GDN/attention overrides and planar-Q6
  integer-MMQ admission. Integer MMQ is a gfx1151 donor, not a missing
  gfx1100-to-gfx1151 transfer.
- [x] The September 12 Q4_K_M INT8 quality recheck reproduces the rejection.
  Preserve artifact-scoped BF16 fallback; gfx1100 physical-c4 INT8 evidence
  does not qualify the different gfx1151 model file.
- [x] Decode-floor attribution and the same-file Q4_K_M llama.cpp comparator
  have measured verdicts (sections H/I). This is not a new measurement or
  a claim that post-default-flip performance is unchanged.
- [x] The PARO gfx1100 KV registry pin is deliberately retained after the
  failed gfx1151-body substitution (`6150ace4b`, section I). It is not an
  outstanding one-line backend correction for dense Qwen3.8.

The tracked scripts `scripts/qwen38_q4_dense_rowtile_gfx1151_screen.py` and
`scripts/qwen38_gfx1151_q4_dense_rowtile_gate.py` produced the two Q4 rowtile
admission packet in section K. Single-arm diagnostics are supported after the
KeyError repair in `d4c8e12b1`; they do not constitute a two-arm gate.

Audit verification: the focused current tests
`tests/test_unit_execution_profile_defaults.py` and
`tests/test_unit_speculative_mtp_serving_capability.py` pass together
(60 tests). They establish resolver/evidence contracts, not the requested
public-server or hardware gates above.

## K. v0.5.0 scope and identity decision

Recorded September 12, 2026 at `f1bdd2332`. Section J is the source/evidence
checklist; this section fixes the *scope* that checklist runs under and rules on
the profile-identity question section I left open. Where K and J disagree, K
governs.

### Scope for this pass

- **gfx1151 only.** Every open item below is a gfx1151 action. gfx1100 is out of
  scope: no gfx1100 change, no gfx1100 qualification, and no gfx1100 row in this
  audit is a work item. A gfx1100 artifact is cited as the reference
  composition or the prior, never as the target.
- **Test artifacts — two, not three.**
  - Primary: Qwen3.8-27B GGUF `Q4_K_M` (`7e78da5d…c6fe169`, 17,106,775,008
    bytes) on the physical gfx1151 host.
  - Secondary: Qwen3.6-35B-A3B GGUF `UD-Q4_K_M`
    (`/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`), the MoE lane.
  - **Qwen3.6-27B `Q4_K_M` is being retired.** Its published gfx1100 rows are
    to be removed from final validation, so it is not a test target here and its
    absence from this host is not a limitation to record.
- **The MoE lane has no equivalent transfer audit.** Sections A-I enumerate the
  dense Qwen3.8-27B `Q4_K_M` path only. Qwen3.6-35B-A3B is in scope as a test
  artifact, but its gfx1100-to-gfx1151 transfer surface has never been
  enumerated, so the MoE items in this section are a first pass and not a
  complete inventory.
- **Backlog rule.** Anything else that surfaces is qualified only if its
  artifact is already on this host. Otherwise it gets a backlog entry with a
  named reopen condition, not a fetch and not an investigation.
- **Excluded and owned elsewhere:** the test-tier migration
  ([`docs/testing/TEST-MIGRATION.md`](testing/TEST-MIGRATION.md)). Section J's
  release-validation item bundles that work together with release-wording fixes;
  the wording half stays in scope here, the migration half does not.

### Qwen3.6-27B and Qwen3.8-27B are one dispatch identity

`hipengine/kernels/policy.py` states the rule outright: "Human-facing model
names are provenance, not dispatch keys. Backend packages use these immutable
identities to admit only model families with the exact weight/state topology
that was gated, while compatible finetunes and renamed exports inherit the same
policy."

Both dense 27B files resolve to `QWEN35_DENSE_H5120_GEOMETRY`
(`architecture="qwen35"`, 64 blocks, H5120, FFN 17408, 24/4 heads, K/V 256, SSM
6144/16/128, conv 4, time-step rank 48). `GGUFModelGeometry` is a plain frozen
dataclass, so the policy key is the structural identity of those 20 fields and
nothing else. Both files therefore also share the profile key
`(model="qwen3_5_gguf", backend, quant="gguf_q4_k_m")`.

The two dense profile modules are consequently **not** two plans for two models.
They are one dense-qwen35 plan per backend, filed under misleadingly
model-named modules:

| Module | Registers | What it actually is |
| --- | --- | --- |
| `qwen36_gguf_gfx1100_profiles.py` | dense + MoE, gfx1100 | the gfx1100 dense-qwen35 plan |
| `qwen38_gguf_profiles.py` | dense, gfx1151 | the gfx1151 dense-qwen35 plan |
| `qwen36_gguf_profiles.py` | MoE, gfx1151 | the gfx1151 MoE-qwen35 plan |

**Ruling: audit finding 3 of `20260911T204040.594305Z-gfx1151-audit-qwen38-dense-path-map-050600`
("a Qwen3.8-27B file on gfx1100 resolves to the Qwen3.6 dense plan") is closed
as not a defect.** One plan per backend is the intended design. **The profile
key must not be split per model name** — splitting it would contradict
`policy.py` and would silently drop one file onto the migration path.

Tracked refactor: rename the three modules to architecture scope
(`qwen35_dense_*` / `qwen35_moe_*`) so a future reader does not re-report the
file names as a defect. The rename changes names only, never the key.

### New work item: A/B the gfx1100 dense composition against the gfx1151 one

The candidate is not a retiring model's configuration. The gfx1100 plan cites
two evidence artifacts, and the one behind the production selection that
differs — the fused R28 `linear_pair_silu` variant — is a **Qwen3.8-27B
`Q4_K_M` gfx1100 measurement**
(`benchmarks/results/2026-09-02-w7900-q4km-k3-c7-fused-r28-periodic-strict-retained.json`,
model `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,773,984 bytes). Only the strict plan's
`_DENSE_STRICT_EVIDENCE` is a Qwen3.6-27B publication, and that model is being
retired. So the A/B compares a Qwen3.8-evidenced candidate against the gfx1151
plan on this artifact, which is exactly the comparison that decides whether the
backends need different compositions at all.

Because the plan is geometry-keyed, "the gfx1100 defaults" is a runnable
candidate on this artifact, not a different model's configuration. The two
per-backend `production` plans differ in both binder env flags and variant
selections (the gfx1100 strict plan's `linear_pair_silu` fallback is
`dense_dual_rowtile_bf16_bf16_out`):

| | gfx1100 dense plan | gfx1151 dense plan |
| --- | --- | --- |
| `linear_pair_silu` | `dense_dual_wmma_prefill_row32_bf16_bf16_out` | `dense_dual_rowtile_bf16_bf16_out` |
| `linear` (C2/C3 scopes) | — | `dense_rowtile_bf16_bf16_out` |
| `gdn_chain_recurrent_rmsnorm_gate` | — | `bf16_c1_exact_state_rows_tloop_fp16state` |
| `linear_attn_chain_conv_decode` | `bf16_c1_exact_state_rows_tloop` | `bf16_c1_exact_state_rows_tloop` |
| binder env | `FP16_RECURRENT_STATE=0`, `VERIFY_CAPTURE_PREFILL_GDN=1`, `VERIFY_F32_RESIDUAL=1`, `VERIFY_F32_POST_NORM=1`, `Q4_T16_DUAL_SILU_PRODUCTION_R28=1`, `C8_Q6_DP4A_GROUPED=1` | `FP16_RECURRENT_STATE=1`, `VERIFY_CAPTURE_PREFILL_GDN=1`, `VERIFY_PRODUCTION_Q4_ROWTILE=1` |

Since `8899172e5` an omitted profile resolves to `production` on both backends,
so the gfx1151 plan is the only thing standing between a no-flag Qwen3.8-27B
user and the gfx1100 composition, and the two have never been measured against
each other.

- [ ] **Run the gfx1100-composition A/B on gfx1151 as part of initial
  qualification.** Same host, same artifact, same KV route, same protocol; the
  gfx1151 plan is the control, the gfx1100 plan's reachable binder is the
  candidate. Keep the winner. Where the gfx1151 composition wins, that
  measurement *is* the gfx1151 gate. Where the gfx1100 composition wins, add a
  gfx1151 backend gate that admits it deliberately rather than leaving it
  inherited by accident.
- Two reachability facts, verified at this commit, constrain that A/B:
  - **Two of the gfx1100 binder's flags are inert on gfx1151 by construction.**
    `GGUF_SPECDEC2_Q4_DUAL_SILU_PRODUCTION_R28_POLICY` and
    `GGUF_SPECDEC2_Q4_DUAL_SILU_ROWTILE_POLICY` are defined only in the gfx1100
    package, and
    `hipengine/runtime/gguf_linear.py::_q4_t16_physical_dual_silu_variant`
    reads them through `backend_package_capability(backend, name, {})`, so on
    gfx1151 they resolve to `{}` and the row-28 retile never engages. A naive
    "set the gfx1100 env" A/B measures only the reachable subset. Either name
    that subset in the result or define the policies on gfx1151 first.
  - **The gfx1100 binder does not publish its manifest hash.** The gfx1151
    binder sets `HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256`; the gfx1100 one
    does not. Record the manifest hash in the A/B artifact by hand so the two
    arms are identifiable after the fact.
- `graph_policy` is **manifest metadata only** — no runtime branch reads it
  (`grep -rn graph_policy hipengine/` finds only the profile registrations and
  an unrelated PARO local). The two plans declare different values
  (`specdec2_eager_c1_exact_qwen38_c8_q6_dp4a` versus `specdec2_eager_c1`) but
  that difference is not enforced anywhere, so it must not be read as evidence
  that the compositions differ in decode-graph behaviour. Relevant to section
  J's "verify the shipped production composition" item.

### Carried forward from section J, re-scoped

- [x] **Q4_K_M scratch-row clamp.** Decided 2026-09-12: **no clamp is needed
  on gfx1151 for correctness**, so the `MOSTLY_Q4_K_M` key is not added. The
  cap on gfx1100 is a memory policy, and its shrink of the allocation is what
  created the 2026-09-09 overrun; with no cap the gfx1151 allocation and every
  chunk site derive from the same resolution. See the subsection below for the
  invariant, the test that holds it at the 1K/4K/8K boundaries, and the one
  measurement still outstanding.
- [x] Production-default / automatic-MTP interaction: scoped closure in section J.
- [x] Shipped production composition: declared greedy numerical/serving scope
  closed in section J; broader quality and release axes remain separate.
- [ ] Inherited Q4 fused-prefill retiles (section J) — stays.
- [ ] Layer-outer and slot-local AOTriton reachable scope (section J) — stays.
  `HIPENGINE_GGUF_PACKED_LAYER_OUTER` and
  `_gguf_int8_prefill_slot_local_aotriton_enabled` both default on globally with
  W7900-only evidence, which is exactly the inheritance this scope exists to
  test.
- [ ] **Qwen3.6-35B-A3B MoE lane.** In scope as the secondary artifact, and the
  selected-expert pair-reuse geometry (section H's A2) moves here from the
  backlog. gfx1151 admits the Q4 selected-expert pair-reuse dual owner
  (`GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8` on both backends), but the
  owner's geometry (`x_rows=8`, `rows` capped at 64, `block(128)`) is shared
  source pinned by a gfx1100 measurement on a 96-CU part and is not overridden
  on gfx1151. The gate resolves to `launch_q4_dual_pairreuse_direct` in
  `kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.hip`, which a gfx1151
  package change cannot reach: retuning means either editing shared source or
  registering a gfx1151 override body as the 18 existing overrides do. Note
  also that this host holds `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, while the W7900
  MoE rows are standard `Q4_K_M` — a different artifact, so re-measure rather
  than inherit.
- [x] gfx1100-only capability inventory (sections I/J) — **documented**, all 19
  names now carry a gfx1151 verdict in the ledger below. Derived at this commit
  over `hipengine/**/*.py`: **47** `GGUF_*` names are defined on gfx1100 and
  absent on gfx1151; **19** of those are live-read through
  `backend_package_capability` anywhere in the package, and
  `GGUF_SPECDEC2_MTP2_PHYSICAL_C1` (`generation/qwen35_gguf_mtp2.py`) is missing
  from section J's list of 18. The other 28 sit in the gfx1100 package's
  `__all__` without being read that way; most are internal composition blocks
  folded into a parent policy in the same file (`GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY`
  is merged into a combined policy a few lines below its definition). The
  section's defect (section I: no gfx1100-only name appears anywhere in the
  gfx1151 package) is answered by the ledger below, which records a verdict for
  all 19 and why the ledger is written in this document rather than in the
  package. With the MoE lane in scope, section J's "do not treat
  MoE/raw-quant-only settings as dense `Q4_K_M` gaps" now applies only to the
  primary dense artifact, not to Qwen3.6-35B-A3B.
- [ ] Three dense-path registry decisions (sections H/J) — stays.
- [x] Test-tier migration half of section J's release-validation item — **not
  this pass's work**; in flight under another owner.
- [ ] Release-wording half of the same item — **stays**. `CHANGELOG.md`'s INT8
  c4 and layer-outer gains need scoping to their measured artifact and backend,
  and its "tested … on … both" introduction and automatic-MTP wording need
  reconciling with the per-backend evidence. `docs/KERNELS.md`'s small-row
  single-wave description needs refreshing against the actual gfx1151
  low-VGPR/shared-B2W2 override.
- [x] gfx1100 items G2 and G3 (section G) — **out of scope**, backlog.

### The two withheld dense Q4T16 rowtiles are admitted (2026-09-12)

Admission and the measured leaf/full-model gates are complete. The
[CPU-reference and trace follow-up](../benchmarks/results/2026-09-12-gfx1151-qwen38-rowtile-cpu-reference-followup.json)
closes the review's two missing kernel gates: ten synthetic CPU-reference
cases pass at 100% top-1 and maximum KL 0.00919, and the uncontended cached-only
actual-weight trace names both families at rows 2/3/4 with zero scratch.
This does not qualify the separate production-profile composition.

| Step | Result |
| --- | --- |
| Leaf screen (`scripts/qwen38_q4_dense_rowtile_gfx1151_screen.py`) | 30 cases, every arm bit-exact against the retained owner. `col4` at K5120/N1024 rows 2-4 is 1.005-1.257x the retained `dense_rowtile_bf16_bf16_out` (6 tensors, 21 counterbalanced pairs each, 377/378 pairs won); the fused `dense_rowtile_bf16_residual_bf16_out` at K17408/N5120 rows 2-4 is 1.007-1.011x `rowtile + gguf_bf16_add` (4 tensors, 245/252 pairs won). |
| Soak gate (`scripts/qwen38_gfx1151_q4_dense_rowtile_gate.py --rows 2,3,4 --max-new-tokens 32`) | rows 2/3/4 `eq_ok`, candidate and control arms token-identical, native c-aware decode, no serial fallback. |
| Full-suite gate (`--rows 2,4 --windows --max-new-tokens 8`) | All 10 mtp-bench prompts across `code`/`general_en`/`general_ja`/`mixed_ja_en`, at rows 2 and 4, `eq_ok` and token-identical between arms. |
| Verdict | Both keys admitted: `_GFX1151_ALIAS_EXCLUSIONS` no longer withholds them, and the shipped registry reproduces the admitted arm's tokens with no mutation. |

Artifact: [`dense rowtile qualification`](../benchmarks/results/2026-09-12-gfx1151-qwen38-27b-q4km-dense-rowtile-withheld-variants-qualified.json).
Both are sub-window wins: the two families carry 0.46% and 9.97% of the 16.091 GB
per-token AR-active weight stream, which projects to roughly 0.06% and 0.09% of
decode, so this is not a topline move. The gate harness also needed a fix: the
batch diagnostic created an `LLM` per arm without closing it, so multi-arm runs
leaked a full model each and triggered a host-wide OOM kill.

### Longer-horizon FP16 state qualification fails (2026-09-12 UTC)

The new [D128 packed-state packet](../benchmarks/results/2026-09-12-gfx1151-qwen38-fp16-packed-d128-rejected.json)
covers 6,394 teacher-forced rows, all ten canonical prompts plus eight
category heldouts, static C4/C8, dynamic C8-to-C1 retirement, sparse C8,
and a 512-token-prompt/D128 group. Three repeats, isolation, and dispatch
identity pass. Mean/p95/p99 KL passes, but maximum KL **0.22994** exceeds
the binding **0.05** ceiling. Failures include ordinary C4/C8 and sparse
continuations, plus the 512-token prompt at decode step 28.

This is FP16 versus FP32 storage on the same packed production arithmetic,
not the named strict-profile denominator. The independent named C1 check also
failed, and the binder now uses FP32 state. After additionally confining
integer-MMQ to target verification, the final public packed gate passes all
8,716 rows exactly. Historical D24 verifier cells keep their original
manifest identity; they are not relabeled as the new production profile.

### The Q4_K_M scratch-row clamp is not needed on gfx1151 (2026-09-12)

This was the section's one correctness question. Decision: **no clamp**, the
`MOSTLY_Q4_K_M` key is not added, and the route is safe without it. The
reasoning is that the gfx1100 cap is a *memory* policy, and the overrun it is
associated with was caused by the cap itself.

- The cap landed in `38e5b85ce` (2026-09-07) to stop session scratch being
  sized by the declared context below 4,096 (~1 MiB per declared token against
  a 64 KiB/token KV payload). Its recorded measurements are memory: BF16 3,840
  request peak 21.820 → 19.375 GiB, declared-context boundary 3,840 → 40,960
  tokens, capped and uncapped runs token-identical.
- The 2026-09-09 overrun (fixed in `2082f1353`) was a consequence of that
  shrink: the cap pinned the allocation at 1,024 rows while the auto policy
  resolved a 4,096-row query chunk, and the chunk sites had to be taught to
  honor the cap. The failure mode requires a cap that makes the allocation
  smaller than the natural chunk.
- The invariant that matters is `chunk_rows <= allocated_rows`, because
  `for_chunk` writes `rows` entries into buffers sized by
  `_prefill_scratch_rows`. Without a cap, the allocation is
  `max(selector_linear(capacity), selector_full(capacity))` and every selector
  is monotone in tokens and bounded by capacity, so a request with
  `rows <= capacity` can never select more than the allocation. The outer chunk
  loop additionally takes `min(_prefill_scratch_rows(rows), bulk_scratch.rows)`,
  and `_chunk_ranges` only ever shortens the last chunk, so tails are covered.
- Measured on the selection logic by the new cases in
  `tests/test_unit_gguf_prefill_chunk_row_cap.py`: at capacities
  1,024/4,096/8,192/32,768 with tails of 0/1/3 rows and 1/2-row requests, the
  invariant holds; for capacities above the 1,025-token tuning floor the
  gfx1151 allocation equals the 4,096-row auto query chunk exactly, so the
  uncapped route is self-consistent rather than merely unbounded; and the
  gfx1100 contrast case shows the clamp there is load-bearing (allocation
  1,024 against a 4,096-row natural chunk).

One limitation is recorded rather than papered over: this is a proof of the
selection invariant plus the policy history, not a GPU run of the boundary
matrix. The device-side metadata writes are bounded by the same `rows` value
that the invariant bounds, and the 2026-09-09 crash was a selection mismatch
rather than a device-side accounting error, so the static argument covers the
known failure mode — but a confirming gfx1151 run at 1K/4K/8K with tails and
packed requests is still the measurement that would close it empirically.

Separately, and not part of this correctness question: without the cap, gfx1151
scratch still tracks the declared context below 4,096. That is the same memory
behaviour the gfx1100 cap removed, and on this APU it is a capacity/boundary
question for the gfx1151 lane (section J's capacity items, backlog C2/C3/C5),
not a correctness one.

### gfx1100-only capability ledger: one gfx1151 verdict per name (2026-09-12)

Nineteen `GGUF_*` capability names are defined on gfx1100, absent on gfx1151,
and live-read through `backend_package_capability` somewhere in the package.
The original table contained **12 declined and 7 N/A** dispositions.
The September 12 UTC follow-up admits row48, leaving **11 declined, 7 N/A,
and 1 admitted**; 18 names remain gfx1100-only. The inventory is not
independent measurement of every disabled optimization. The unequal-pair
integrated follow-up now finds a real regression: all 18 prompts are exact
but every measured complete-prefill row band is slower, including 1.0-1.2%
lower throughput at 512/1K/4K. See
[integrated rejection](../benchmarks/results/2026-09-12-gfx1151-qwen38-unequal-pair-integrated-rejected.json).
This is the section I defect closed: none of them appeared in the gfx1151
package, so each absence read as an ordinary call-site default. Each now has a
gfx1151 verdict below. Nothing in this pass changes a shipped value.

The ledger is written here rather than as an inline comment in the gfx1151
package on purpose: four tests pin that package's bytes
(`test_unit_laguna_h8a_source_default`, `test_unit_laguna_h8b_source_default`,
`test_gpu_laguna_h7u_parallel_moe_compaction`,
`test_gpu_laguna_h7y_swa_lane_major_cache`), so an inline ledger would force a
re-pin of four other campaigns' source audits for a comment-only change.
`tests/test_unit_gfx1151_backend.py::test_gfx1151_capability_ledger_covers_gfx1100_only_live_reads`
keeps this table and the live-read set in step instead.

Verdict vocabulary: **N/A** means the name cannot select anything on gfx1151;
**declined** means the gfx1151 default disables a W7900-measured optimization
and the retained owner runs instead. Every declined row is fail-closed (the
call site's default is the conservative value) and every row names its reopen
condition. Reachability is stated against the two in-scope artifacts: Qwen3.8-27B
`Q4_K_M` dense and Qwen3.6-35B-A3B `UD-Q4_K_M`.

| Name | gfx1151 read | What runs instead | Verdict |
| --- | --- | --- | --- |
| `GGUF_Q4_T16_GROUPED_PAIR_ROWS6_POLICY` | `{}` — and unreachable | its only call site is nested inside the absent `GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS` gate, so no gfx1151 route reaches it; the rows6 sibling is registered but unselectable | N/A |
| `GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES` | `{}` → identity miss | retained pair route; `dense_unequal_dual_wmma_prefill_bf16_bf16_out` is registered but never selected | declined |
| `GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS` | `48` | row48 at rows33-48; row64/row128 above the band; exact unfused fallback | admitted |
| `GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES` | `{}` → `None` | no scratch liveness aliasing (more scratch, no arithmetic change) | declined |
| `GGUF_RAW_K_PREFILL_ROLE_VARIANTS` | `{}` | computed `coltile*_rowbatch*` geometry; all three W7900 entries are `gguf_q6_k`, whose raw coltile family gfx1151 declines upstream | N/A |
| `GGUF_T16_F16_ROCBLAS_SOLUTION_VERSION_PREFIX` | `""` → guard fails | the F16 rocBLAS pair route is not selected; its kernel is not registered on gfx1151 | N/A |
| `GGUF_T16_F16_ROCBLAS_SOLUTION_INDICES` | `{}` | as above; the indices are keyed to one rocBLAS build hash | N/A |
| `GGUF_T16_F16_ROCBLAS_PAIR_ONLY_POLICIES` | `{}` | as above | N/A |
| `GGUF_C8_Q5_RAW_MMQ_SSM_OUT` | `False` | C8 Q5-K MMQ SSM-out route closed; `q5_raw_mmq_target_session` gets no library and the retained owner runs | declined |
| `GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS_BY_MAX_SEQUENCE_LENGTH` | `{}` | gfx1151's own flat `GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS = 8` | N/A |
| `GGUF_SPECDEC2_MTP2_PHYSICAL_C1` | `False` | legacy singleton; the call site's own comment records that gfx1151 evidence retains it | declined |
| `GGUF_SPECDEC2_PRODUCTION_PHYSICAL_PROMPT_STREAMING` | `False` | no production prompt streaming | declined |
| `GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXTRA_ROWTILE_SHAPES` | `()` | no extra verifier rowtile shapes | declined |
| `GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q5_ROWTILE_ROWS` | `()` | no Q5 verifier rowtiles | declined |
| `GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS` | `()` | no Q6 verifier rowtiles | declined |
| `GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS` | `{}` | doubly off: also ANDed with the Q6 rowtile name above | declined |
| `GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS` | `()` | no padded target-verify row counts | declined |
| `GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS` | `()` | no exact-rowtile target rows | declined |
| `GGUF_SPECDEC2_NATIVE_TARGET_CACHE_CAPACITY_POLICIES` | `()` | gfx1151's own `GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT` / `_GRAPH_MAX_CONTEXT = 65544`, already the cache capacity | N/A |

Three properties make the declined rows safe rather than gaps:

- **Every one is fail-closed.** The call-site defaults are `False`, `()`, `{}`,
  or `0`, so an absent name closes a route. No name in this set has a
  permissive default that an absent definition could leave open.
- **Every gated route has a registered owner on gfx1151.** The retained
  rowtile16/pair/singleton/legacy-verifier owners are registered and are what
  the gfx1151 evidence for these artifacts was measured on.
- **The numeric rows are W7900-measured values that must not be copied.** The
  rocBLAS solution indices are bound to one library build
  (`5.2.0.dabb6df2b98`); the row48 crossover (rows 33-48) and the per-context
  server admission (`{768: 13}`) are W7900 measurements, and the gfx1151
  package already carries its own admission number.

Named reopen conditions, one per declined group:

| Group | Reopen when |
| --- | --- |
| Q4 prefill geometry (unequal pair, row48) | an independent gfx1151 crossover measurement at the affected shapes and row bands exists (the admitted rowtile qualifications above are the pattern: leaf screen plus complete-model token gate) |
| Dense scratch liveness | an independent gfx1151 scratch-lifetime measurement reproduces the W7900 field set (`priority_min_rows`, `hidden_inplace_min_rows`, `priority_min_live_stages`) on the gfx1151 geometry/file-type identity |
| C8 Q5 raw MMQ | an independent gfx1151 c=8 Q5-K SSM-out measurement |
| SPECDEC2 production physical routes | a gfx1151 production-profile MTP verifier measurement on the full mtp-bench category suite |

Two consequences are recorded rather than fixed. First, the gfx1151 package
registers three variants that no gfx1151 policy can select
(`dense_rowtile16_w2_grouped_rows6_bf16_bf16_out`,
`dense_unequal_dual_wmma_prefill_bf16_bf16_out`, and
`dense_dual_wmma_prefill_row48_bf16_bf16_out`): they are shared aliases, so the
registration is inert rather than wrong, and it becomes reachable if a reopen
condition above is met. Second, the dense scratch policy divergence is a memory
observation, not a correctness one — no aliasing means more scratch, never less.

### Backlog (named reopen condition, no work this pass)

- **A3** nasone32 RDNA3.5-guarded donors — the gfx1100 VDR rejection is the
  prior. Reopen only if the A1 floor conclusion changes.
- **A4** strix-llama.cpp HIP deltas — reopen only if upstream merges new HIP
  work after `5f851647`.
- **B1** prefill family attribution, **B3** direct INT8 prefill speed, **C1**
  DMS speed A/B — reopen when a prefill or DMS transfer target is chosen.
- **C2/C3/C5** capacity ladders — N/A-capacity on the APU; confirm
  non-regression only.
- **D2** MTP numbers — governed by the gfx1151 scaling campaign's own-AR rule.
- **Other gfx1100-only capability names outside the two artifacts.** 2 `PARO_*`
  names are defined on gfx1100, absent on gfx1151, and — like the 28 `GGUF_*`
  names that are not live-read — carry no comment, constant, or exclusion entry
  anywhere in the gfx1151 package. This is the same defect class as the ledger
  above, outside this release's two artifacts. No artifact that reaches those
  names is on this host, so it is backlog by the rule above. The 19 live-read
  `GGUF_*` names no longer fall in this class: the ledger above covers them.
- **G2/G3** gfx1100 open items — gfx1100, out of scope.
