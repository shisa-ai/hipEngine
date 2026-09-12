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
**1307 shared variants, 105 gfx1100-only variants, 7 gfx1151-only variants,
241 declared exclusions (`_GFX1151_ALIAS_EXCLUSIONS`), and 18 body overrides
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

The 105 gfx1100-only variants are concentrated off the dense path: by layer they
are 47 `linear`, 19 `moe_linear`, and 10 `laguna_attention_prefill`; by quant
they are 24 `gguf_q5_k`, 23 `gguf_q6_k`, 15 `gguf_iq3_xxs`, and 14 `bf16`. Every
gfx1100-only variant carries a written reason in the gfx1151 file, so this is
deliberate scope, not drift.

Exactly **five** gfx1100-only variants sit on the Qwen3.8-27B dense path, and
each one has a named gate that would clear it:

| Key | Recorded reason |
| --- | --- |
| `linear / gguf_q4_k_t16_v1 / dense_rowtile_col4_bf16_bf16_out` | W7900-only until gfx1151 receives an independent shape crossover and full-model gate |
| `linear+residual / gguf_q4_k_t16_v1 / dense_rowtile_bf16_residual_bf16_out` | W7900-only pending independent gfx1151 boundary/model gates |
| `linear_attn_alpha_beta+chain_conv+snapshot / f32 / bf16_k5120_n48_c10240_k4_exact_state_rows_tloop` | Screened only on gfx1100; gfx1151 keeps three independent leaves |
| `linear_state_pair_copy / f32 / chunked_i32` | W7900-only until gfx1151 receives independent transaction and launch-overhead gates |
| `paged_kv_write / gguf_q4_k_m / mixed_bf16_shared_batch_spans` | Qualified only for the W7900 dense-H5120 N1 graph; gfx1151 keeps scalar append aliases |

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

The registry surface, reproduced with the snippet in section H: **1307 shared,
105 gfx1100-only, 7 gfx1151-only, 241 declared exclusions, 18 body overrides.**

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
| Registry exclusion | 5 of 105 | `_GFX1151_ALIAS_EXCLUSIONS` | yes |
| Capability gate defined on gfx1100 only | 18 | the gfx1100 package | yes — define the name on gfx1151 |
| Capability gate declined on gfx1151 | 6 | the gfx1151 package | yes, but three are measured rejections |
| Geometry pin | shared device source | the `.hip` launcher | **no** |
| Row-count floor or ceiling | 13 names (9 differ) | both packages | yes — it is a per-backend value |

Only the geometry pin is structural. The 105 registry exclusions each carry a
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
only to raw-Q5/Q6 artifacts.

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
   verdict — retune, decline, or N/A — at the same standard the 105 registry
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
