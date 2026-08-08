# hipEngine Kernel Catalog and Port Playbook

This doc is both the live kernel catalog and the mechanics for landing a kernel in hipEngine — writing/porting kernels under `kernels/<backend>/`, the JIT build layer, gotchas specific to this repo, and the correctness gate a kernel must pass.

**Kernel work happens in this tree.** hipEngine is substantively different from `~/amd-gpu-tuning/` (torch-free runtime, four-axis registry, `KVLiveSpans` ABI, verifier/small-batch-shaped kernels), so new kernels, fused variants, micro-tuning, `rocprofv3 --kernel-trace` iteration, VGPR/occupancy and `__launch_bounds__` sweeps, and fusion experiments all live here. `~/amd-gpu-tuning/nano-vllm-amd/` and the parent docs remain read-only *lineage references*: cite source file + commit when porting an idea, reuse prior evidence, but do the development and profiling in-tree. Profiling inside the hipEngine tree is expected — prebuild the `.so` and use a precomputed compiler-version file + `require_cached` so the profiled process does not spawn `hipcc` (see the JIT/rocprof notes below).

See also:
- `docs/PLAN.md` "Kernel Port Strategy" — authoritative source inventory, split plan, per-family targets.
- `docs/TESTING.md` — RED/GREEN, CPU-reference fixtures, and math-correctness gates.
- `~/amd-gpu-tuning/AGENTS.md` — audit-first-via-rocprofv3, time-share/occupancy/iters-per-thread/VGPR discipline.
- `~/amd-gpu-tuning/LESSONS-LEARNED.md` — device-code gotchas and kernel lineage results.
- `~/amd-gpu-tuning/docs/OPTIMAL.md` — current optimal Qwen3.5/PARO native engine route and flags.
- `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md` — Qwen3.5/PARO design history and evidence rows.

## Status legend

| Status | Meaning |
| --- | --- |
| **hipEngine landed** | Source lives in this repo, is registered or runnable through hipEngine, and has this repo's tests/smokes. |
| **CPU reference landed** | Torch-free NumPy oracle lives in `hipengine/kernels/cpu_reference/`; it is correctness infrastructure, not a HIP port. |
| **Lineage green** | Implemented/validated in `~/amd-gpu-tuning/nano-vllm-amd/`; source for hipEngine's copy+partition+retype port, but not yet landed here. |
| **Lineage dirty / experimental** | Observed in the parent checkout's uncommitted worktree or R&D notes. Treat as a lineage reference only; promote to a default hipEngine path via in-tree implementation + bit-exact RED test + the correctness gate, not by waiting on the parent. |
| **Planned** | Architecture path is decided, but no hipEngine implementation yet. |

## hipEngine-landed kernels and oracles

This is the authoritative list of kernels/oracles that exist in this repo today. Empty backend family packages under `hipengine/kernels/hip_gfx1100/*/` are placeholders, not implemented kernels.

### Maple packed linear kernels (**hipEngine landed, gfx11 bring-up**)

`hipengine/kernels/hip_gfx1100/quant/maple_ternary.{hip,py}` implements the
framework-independent storage contract published in
`deepgrove-ai/mlx-lm-deepgrove/mlx_lm/ternary.py`: ternary projections pack 16
LSB-first 2-bit codes per U32 word with `value = row_alpha * (code - 1)`, while
embeddings and the exact lm-head use LSB-first affine 4-bit/group-64 storage.
The family exposes generic BF16 ternary GEMV, split-weight fused Q/K/V GEMV,
selected-expert dual gate/up and single down GEMV, exact expert-major batched
gate/up/down, affine4 embedding lookup, and affine4→FP32 lm-head GEMV. The
expert-major consumers take generic stable-compaction `expert_start` and
`sorted_lanes` metadata and write directly back to original row/route order.
The generic group-scatter family now also registers an int32 selected-ID
parallel count/prefix/stable-scatter variant for Maple's router ABI. These
kernels register under both `hip_gfx1100` and the peer `hip_gfx1151` alias for
quant key `maple_ternary2`.

Independent NumPy fixtures cover pack/dequant order and BF16 boundaries. The
gfx1151 gate is BF16-bit exact for embedding, generic/fused QKV, and selected
expert outputs; affine4 FP32 logits pass at `atol=rtol=2e-4`. A cache-only
`rocprofv3 --kernel-trace` smoke observes
`maple_ternary_gemv_kernel` at grid 128/local32, VGPR32, SGPR128, LDS0,
scratch0, and **3,527 ns** duration on Radeon 8060S/gfx1151.

The P1 expert-major prefill path is BF16-bit exact to the row/route NumPy oracle
and passes all 18 natural+heldout final hidden/KV/span-state hashes with KL 0
and 90/90 token/top-1 agreement. A cached dirty-tree gfx1151 trace confirms the
final kernels ran as `maple_selected_ternary_dual_grouped_kernel<128, 1>`
(local128, VGPR72, LDS1024, scratch0) and
`maple_selected_ternary_grouped_kernel<32, 1>` (local32, VGPR48, LDS512,
scratch0). Stable metadata costs 0.444 ms/request. The expert family changes
**276.150 -> 254.179 ms (1.086x)** and traced prefill320 changes **498.442 ->
476.730 ms (1.046x)** in the implementation trace. Clean qualification retains
**726.421/679.632/650.745 tok/s** at 128/320/512 with 18/18 state hashes and
90/90 positions exact. Exact 2-/4-lane cooperative schedules regress, leaving
gate/up scalar unpack/reduction as the measured blocker to the 97.708-ms P1
target. The row/route gather remains an environment-controlled rollback.

The P3 dense token-tile sweep leaves the original tile-8 QKV/O consumers
unchanged. Exact tile 16/32 candidates preserved the production 2,048-wide
reduction bit-for-bit but measured **731.182/571.923 tok/s** versus tile 8
**744.116 tok/s**, with **0/16** paired wins for each candidate. The regression
is consistent with larger dynamic LDS (8/16 KiB versus 4 KiB) and longer block
residency dominating the saved 512-byte weight-row reloads. The required direct
BF16-WMMA probe is also rejected: native K16 association changes **106/256
FP32** word partials and **43/655,360 BF16** production-shape outputs. A cached
trace names `maple_ternary_gemm_wmma_kernel` at local32/VGPR48/SGPR128/LDS0/
scratch0, and extracted ISA confirms `v_wmma_f32_16x16x16_bf16`. All temporary
selectors, alternate exports, and WMMA probes were removed; do not repeat these
scalar-tile or direct-WMMA schedules.

The D0 c1 router registers a retained one-dispatch last-block composite. Its
256 expert blocks preserve the existing logit tree; a four-byte owned atomic
counter identifies the final block, which executes the unchanged FP32
softmax/stable-top-8 body and wraps the counter to zero. The clean two-resident
18-prompt natural+heldout qualification improves the exact two-dispatch
rollback **139.538 -> 145.321 tok/s (+4.14%)**, saves **0.301 ms** at the paired
median, and wins **1,127/1,152** timed pairs. All **1,296/1,296** tokens/top
logits, **36/36** native-start/final state pairs, and **2,592/2,592** counter
checks are exact; close returns zero ownership. Cached tracing names
`maple_router_topk_single_dispatch_kernel` at local256/VGPR16/SGPR128/LDS3584/
scratch0, cuts **24 launches/token**, and measures the refreshed short profile
at **5.527-ms wall / 4.859-ms kernels / 271 launches / 180.935 tok/s**. The
two-dispatch route remains registered as the exact rollback.

The retained/default D0 affine4 head registers
`maple_affine4_gemv/group64_wave32_exact`. One local32 wave computes four
virtual production partials per lane and reconstructs the exact stride
64/32/16/8/4/2/1 tree with shuffles, removing the original four-wave LDS
exchange and barriers without output-row tiling. At the real 151,936x2,048
head it is FP32-bit exact across all logits and improves **1.527 -> 1.020 ms
(1.496x, 48/48 wins)**. Clean two-resident 18-prompt qualification improves
the exact group64 rollback **143.679 -> 153.409 tok/s (+6.77%)**, saves
**0.442 ms** at the paired median, and wins **1,146/1,152** pairs. All
**1,296/1,296** tokens/top logits, **36/36** native-start/final state pairs,
**2,592/2,592** counter checks, and lifecycle are exact. Cached tracing names
`maple_affine4_gemv_wave32_exact_kernel` at local32/VGPR16/SGPR128/LDS0/
scratch0, **0.968 ms/step** in the final trace. D0's behavior-neutral host tail
also snapshots the two default-off fusion selectors once per step rather than
once per layer, deleting 46 environment reads without changing this **271-
launch** trace. Four-process fixed-token A/B improves **200.279 -> 202.580
tok/s (+1.15%)**; the separate trace process is **5.018-ms wall / 4.550-ms
kernels / 0.468-ms host gap = 199.293 tok/s**. Production now selects wave32
directly after the final roadmap audit removed the temporary environment seam;
the original group64 primitive remains registered and directly tested.

The retained/default D1 batched head registers
`maple_affine4_gemv/group64_batched_rowreuse_exact` for c2/c4/c8. One local128
block owns one vocabulary row across all request rows, loads each packed
word/scale/bias once, and replays the original 128-thread FP32 tree independently
for every request. Production-shape c2/c4/c8 logits are bit-exact to the original
all-row kernel. The clean fixed-helper gate improves aggregate throughput
**218.818/261.099/299.181 -> 250.481/346.365/428.063 tok/s**; all nine timing
samples, **18/18** natural/category-heldout trajectories, sparse and reclaimed
slots, and lifecycle are exact. Cached c8 tracing names
`maple_affine4_gemv_batched_rowreuse_exact_kernel<8>` at local128/VGPR96/
SGPR128/scratch0, measures the head **10.490 -> 3.734 ms (2.809x)**, and reduces
wall **25.925 -> 19.296 ms** with the same 293 launches/batch. Production now
selects row reuse directly at supported widths after the final roadmap audit;
unsupported widths retain the registered original all-row route, and direct
kernel tests compare both exact primitives.

`hipengine/kernels/hip_gfx1100/attention/maple_attention.{hip,py}` adds the
unfused attention/KV chain: device span publication, per-head standard
QK-RMSNorm plus rotate-half partial RoPE, BF16 K/V append, and online-softmax
GQA decode. Both write and attention consume all required `KVLiveSpans`
pointers (`base_offsets`, `live_counts`, `token_positions`, `evict_mask`) plus
`row_positions`; the same token-granular ring represents SWA-512 and bounded
global cache by capacity. Span wrap, nonidentity physical offsets, Q/K values,
K/V bytes, and attention output are BF16-bit exact to the NumPy oracle. The
cache-only gfx1151 trace reports QK/RoPE/KV-write at local32/VGPR24/scratch0,
**5,771 ns**, and attention at local32/VGPR16/scratch0, **3,607 ns** on the tiny
fixture. The batched ring-prefill attention reads the complete live causal
prefix across chunk boundaries; its prefix-aware fixture is BF16-bit exact and
a cached gfx1151 trace names `maple_attention_prefill_ring_kernel` at **6,452
ns**, VGPR16, LDS0, scratch0. The clean post-P1 prefill320 trace measures 48
chunk/layer calls at **63.993 ms/request (13.55% of kernel time)**, local128,
VGPR16, LDS0, scratch0. This local128 body owns one `(query head,row)` per block,
rereads each KV stream for all four GQA query heads, and barriers throughout the
per-key reduction. P2 adds a separately registered exact GQA4 body: one wave32
owns `(KV head,row)`, loads each K/V row once, and emulates all 128 virtual
threads through the original 64/32/16/8/4/2/1 LDS stages plus the original
weighted-value/FMA boundary. It consumes every `KVLiveSpans` pointer and keeps
local128 as rollback. The production-shape primitive is BF16-bit exact at the
256-row chunk boundary; the binding M5 gate passes 18/18 state hashes, 90/90
positions, KL 0, and exact lifecycle. Clean cached tracing names
`maple_attention_prefill_ring_gqa4_wave32_kernel` at local32/VGPR64,
dynamic-LDS512 (static trace field LDS0), scratch0 and measures **63.993 ->
21.916 ms (2.920x)** at unchanged launch count. P2 qualified 128/320/512 at
**749.175/741.368/754.000 tok/s**, up **3.13%/9.08%/15.87%** over P1. P4's
unchanged arithmetic recertifies **750.854/741.890/754.458 tok/s** with
5,355,881,852-byte residency and carries native prefill safely through the
retained 520/770-token physical-state gates. Batched decode uses disjoint
per-request rings and separate SWA/global capacity owners; wrapped positions
remain inside their request arena after position 512. The wrapped c=3 primitive
is BF16-bit exact;
cached tracing reports batched QK/RoPE/KV write at **3,967 ns** (VGPR24) and
batched attention at **20,197 ns** (VGPR16), both LDS0/scratch0.

`hipengine/kernels/hip_gfx1100/moe/maple_moe.{hip,py}` supplies the unfused MoE
control/tail around selected ternary GEMV: BF16-hidden/BF16-weight router logits
with FP32 accumulation, all-expert softmax, stable top-k, and selected
renormalization; trained clamp-7 SwiGLU; and selected weighted sum plus residual
with both published BF16 boundaries. Tiny gfx1151 fixtures preserve selected IDs
exactly, bound FP32 route weights to one ULP, and preserve BF16 activation/residual
bytes. Cache-only tracing
reports router local256/VGPR8/LDS3584/scratch0 at **19,156 ns**, clamp-7
local256/VGPR32/scratch0 at **2,605 ns**, and weighted residual
local256/VGPR16/scratch0 at **2,885 ns**.

The router also retains the two-dispatch parallel variant
`maple_router_topk_parallel_bf16` (grid-over-experts coalesced dot followed by
parallel softmax/stable-top-k). It first cut the serial router from 277 -> 48
us/call and decode from 12,758 -> 6,132 us (2.08x), with the packed gate at max
KL 0.0139, top-1 18/18, and exact router IDs. It is now the explicit D0 rollback
rather than the default (`benchmarks/results/2026-08-07-gfx1151-maple-router-parallel.json`).

`hipengine/runtime/maple.py` composes those registered/unfused families with the
existing direct-weight PARO RMSNorm and two-stage FP32 argmax into a resident
24-layer c=1 runner. Immutable packed weights stay device-resident; SWA and
global metadata reset without clearing stale cache bytes because absolute
`token_positions` gate every read. Both span-owner identities are published even
when their capacities are equal; the earlier top-ID-1112 diagnostic preceded
that correctness fix and is not retained evidence. P4 keeps global layers
chunk-batched beyond SWA-512 while each sliding layer restores pre-chunk span
metadata, batches the safe prefix, and serializes post-wrap attention rows. The
520/770-token gates preserve physical K/V, spans, final state, and continuation.

`MapleBatchRunner.from_runner()` now shares the c1 checkpoint owner and exposes
request-local span/cache views for native prompt admission. Public
`MapleResidentModelRunner` uses fixed sparse slots, D0 c1 for one active row,
and D1 c2/c4/c8 otherwise; completion/rollback resets only that slot. The clean
public protocol reaches **123.131/165.697/202.038/214.788 aggregate tok/s** at
c1/c2/c4/c8, with all 15 repeated natural/heldout trajectory sets,
physical-c8 singleton preservation, staggered slot reuse, and lifecycle exact.
The final roadmap audit removes the duplicate `MapleContinuousBatcher` owner;
the low-level D1 benchmark drives `MapleBatchRunner.batch_step` directly, while
all admission/reclaim orchestration stays on the public runner. Evidence:
`benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json`.

The corrected 18-position packed-formula gate passes at max KL **0.013508** and
18/18 top-1; the pinned Transformers `trust_remote_code` same-weight gate passes
at max KL **0.004719** and 18/18 top-1, with device greedy argmax exact. The
post-fix public `LLM.generate_detailed()` route resolves model ID →
`hip_gfx1151` / `maple_ternary2`, consumes the exact 18-token chat prompt,
produces a coherent 37-token answer, stops on real EOS 151645, repeats the same
IDs/text in one resident process, and returns tracked allocation to zero after
`close()`. The observed **4.365 s cold / 0.703 s resident-repeat** walls are
bring-up diagnostics only, not retained throughput claims. Evidence:
`benchmarks/results/2026-08-05-gfx1151-maple-ternary2-correctness.json` and
`benchmarks/results/2026-08-05-gfx1151-maple-public-e2e-smoke.json`.

### Laguna gfx1151 decode transfer screen

The retained gfx1100 current-P4 Laguna head-RMSNorm + partial-RoPE + BF16
KV-write composites are now registered under the peer `hip_gfx1151` key. The
shared gfx11 source is compiled as a native gfx1151 code object; the global and
SWA variants retain the complete `KVLiveSpans` ABI and the registered unfused
head-RMSNorm/RoPE then KV-write chain. The clean p512/d128 candidate improves
**11.466687 -> 11.483587 tok/s (+0.147%)**, with all three candidate samples
above every control sample, so gfx1151 selects it by default. The long-context
profiler retains `--no-head-kv-fusion` as the explicit rollback.

The native gfx1151 fixture is bit-exact at global positions 0/255/256/4095 and
SWA positions 0/511/512/1023, including F32 query/key outputs, BF16 K/V bytes,
and all live-span metadata. `rocprofv3 --kernel-trace` observes both composite
specializations at local256, VGPR16, SGPR128, LDS0, and scratch0. The separately
registered global wave-0 reduction remains excluded on gfx1151 because its
gfx1100 full-model gate was rejected.

The next transfer registers the complete exact gfx1100 split
global/SWA attention bundle under gfx1151. This includes the score producers,
ordinary and softplus-gated reducers, SWA tile16 crossover, and wave-local
reducers; the profiler selector requests the measured gfx1100 thresholds
global 127, SWA 65, and tile16 257 as one inseparable candidate. The clean
full-model p512/d128 gate improves **11.485885 -> 14.533955 tok/s (+26.538%)**,
so gfx1151 now selects the full threshold/reducer bundle by default.
Native gfx1151 tests match serial attention against the CPU oracle across
global/SWA wraps and prove the gated/wave-local reducers bit-exact to their
registered unfused/retained chains. Cached tracing observes all required
producer/reducer families with zero kernel scratch.

The exact dense-prefix global owner now also has a retained/default
local1024 specialization. It preserves the eight-wave denominator tree
and every scalar FP32 operation while widening only independent QK and
double-buffered value transport from 16 to 32 wave32s. Live lengths
513/576/639 are F32- and gated-BF16-byte exact to local512 and improve the
21x100 leaf **2.126%/3.823%/4.157%**. Cached gfx1151 tracing names
grid40/local1024, VGPR48, SGPR128, reported LDS512, and scratch0:
[`global local1024 leaf`](../benchmarks/results/2026-07-31-gfx1151-laguna-global-local1024-leaf.json).
Fake-owner selection proves that only dense-prefix idle-buffer requests reach
local1024 and that local512 remains the exact fallback:
[`runtime admission`](../benchmarks/results/2026-07-31-gfx1151-laguna-global-local1024-runtime-correctness.json).
All seven exact p512/d128 candidate runs win
**22.358675 -> 22.383414 tok/s (+0.11065%)** with unchanged state and
residency. Tracked-clean selector-unset production reaches
**22.378602 tok/s**, and the 127-transition census cuts global attention
**0.453932 -> 0.402996 ms/token (-11.221%)** and complete kernel work
**0.049632 ms/token**. gfx1151 defaults local1024; the comparison-only profile
seam is removed and local512 remains the exact non-dense/eviction/peer
fallback:
[`production`](../benchmarks/results/2026-07-31-gfx1151-laguna-global-local1024-production.json),
[`census`](../benchmarks/results/2026-07-31-gfx1151-laguna-post-global-local1024-wall-reprofile.json),
[`retention`](../benchmarks/results/2026-07-31-gfx1151-laguna-global-local1024-retained.json).

The gfx1151-only
`linear_pair/gguf_q4_k/pack8_dual_decode_bf16_bf16_out` key now reuses the
existing exact local32 pack8 dual body for Laguna c=1 gate/up. It owns all
**47 shared plus one leading-dense pair/token**; registry, backend, shape, or
layout miss retains two singleton launches. Direct K3072/N1024, K1024/N3072,
and K3072/N12288 leaves are BF16 byte-exact and improve two singletons by
**20.25%/24.81%/12.48%**. Seven resident p512/d128 pairs improve
**19.556271 -> 19.645185 tok/s (+0.4547%, -0.2314 ms/token)** with exact
trajectories and lifecycle. Cache-only tracing names
`gguf_q4_k_pack8_dual_prefill_out_kernel<unsigned short>` for exactly
**5,969 shared + 127 dense** calls at local32/VGPR96/SGPR128/LDS512/scratch0,
reduces compute dispatches **816 -> 768/token**, and cuts the complete Q4
pack8 family **3.018303 -> 2.836943 ms/token (-6.01%)**.

The exact gfx1151 `linear_pair_silu/gguf_q4_k` successor retains both
independent BF16 projection boundaries in registers, consumes them with the
same SiLU-product expression, and writes only the BF16 intermediate. It owns
the same **47 shared plus one leading-dense chains/token**, removes another
**48 launches/token** and **483,328 bytes/token** of temporary gate/up
write-read traffic, and retains the pair-plus-SiLU chain as the unfused
fallback. Actual-weight 21x100 leaves improve shared
**0.014770 -> 0.012433 ms (-15.824%)** and dense
**0.474136 -> 0.469647 ms (-0.947%)** with zero BF16 mismatches. All seven
resident p512/d128 pairs improve **20.756829 -> 20.810024 tok/s (+0.2563%,
-0.12315 ms/token)** with exact trajectory/state/lifecycle. Native tracing
names the `true` specialization at local32/VGPR96/SGPR128/LDS512/scratch0.
Evidence:
[`dual-Q4 plus SiLU retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-pack8-dual-silu-retained.json).

### Laguna gfx1151 exact cached-only qrow4 prefill

The global/SWA `laguna_attention_prefill` family now registers cached-only
qrow4 variants that consume complete `KVLiveSpans` after the existing writer
has rounded and appended one M128 K/V tile. The gfx1151 scheduler pre-appends
all complete global tiles and only pre-wrap SWA tiles; partial tiles, wrapped
SWA, staged verifier transactions, gfx1100, and unmeasured backends retain
attend-then-append. This avoids destroying SWA keys that remain visible across
a ring wrap.

The M128 GPU fixture is F32 byte-identical to production qrow4 for both layer
types. Cached tracing names global `<4,true>` and SWA `<4,true,true>` at
local32, VGPR64, SGPR128, zero LDS, and zero scratch. The implementation-tree
screen improves global **1.305x** and SWA **1.142–1.186x** across pp512 tile
positions. Seven paired full-model runs improve **507.391 -> 528.771 tok/s
(1.042x)** with every pair positive and complete output/state exactness; 1K
and 4K diagnostics remain exact at **1.103x/1.047x**. Clean selector-unset
publication completed at **526.451 tok/s** before the later exact Q6 qmicro
production checkpoint raised the topline to **530.447 tok/s**. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-preappend-{candidate,production}.json`.

The separately registered cached-metadata qrow4 candidate exploits the same
preappend contract but derives visibility solely from complete
`KVLiveSpans`, removing current-vs-cache source selection while preserving the
ordered dot, wave32 reduction, online softmax, PV order, and every F32 output
bit. Eleven-sample four-position leaf timing improves SWA **1.108–1.128x** at
every tile and global **1.010–1.052x** from position 128 onward; global
position 0 retains the established cached body. The qualified 12-full/36-SWA
model improves **14.6024 -> 13.3230 ms (1.096x)** and projects **15.353 ms**
pp512 saving. Cached tracing names global `<4,true,true>` and SWA
`<4,true,true,true>` at local32/VGPR64/SGPR128/LDS0/scratch0. The gfx1151
runtime policy now selects the candidate for every safe SWA tile and global
tiles from position 128, retaining the established cached body for global
position 0. Seven alternating one-owner pp512 pairs improve
**533.507 -> 542.785 tok/s (+1.739%, 7/7 wins)** with identical complete model
state. Clean selector-unset production reaches **542.088 tok/s** median and
**542.022 tok/s** minimum; cached tracing observes 12 global start-0, 36 global
metadata-only, and 144 SWA metadata-only calls while cutting attention
**175.802 -> 160.123 ms (-8.92%)**. gfx1100 and unmeasured backends remain
unchanged. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-cached-meta-{candidate,default,production}.json`.

The next exact global-only candidate extends that metadata-only body from four
to six adjacent query rows. It is F32 byte-identical on the production M128
preappend fixture. Eleven-sample leaf timing is neutral at global start 0 and
improves qrow4 by **1.202x/1.262x/1.278x** at starts 128/256/384. Keeping
qrow4 at start 0 and for every SWA call models **13.3281 -> 12.8211 ms
(1.0395x)** across one global plus three SWA layer slots, or **6.083 ms**
across the 12/36 production layer split. Cached tracing names global
`<6,true,true>` at local32/VGPR88/SGPR128/LDS0/scratch0 versus qrow4 VGPR64.
The analogous SWA qrow6 body lost **10.9–18.4%** at all four positions and was
removed completely. Seven alternating complete-state pp512 pairs then improve
qrow4 **546.056 -> 548.774 tok/s (+0.498%, 7/7 wins)** with identical
logits/hidden/KV/token/cursor. gfx1151 therefore defaults only the qualified
global positions to qrow6, with `prefill_global_qrow6=false` as explicit
qrow4 rollback; other backends remain unchanged. Clean selector-unset
512/1K/4K reaches **547.064/513.180/428.628 tok/s**, improving the preceding
M2048 packet by **0.376%/1.359%/4.518%**. Cached tracing observes 12
global-qrow4 / 36 global-qrow6 / 144 SWA-qrow4 calls at pp512 and cuts
attention **158.702 -> 152.406 ms (-3.97%)**. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-global-qrow6-{candidate,default,production}.json`.

The next exact specialization exploits a stronger fact already established by
the same preappend schedule: before the first ring wrap, a complete initial
tile has `token_positions[logical_slot] == logical_slot` and no eviction.
Separately registered dense-initial global/SWA bodies still consume the full
`KVLiveSpans` ABI, preserve `base_offsets` physical mapping, and validate
boundary metadata, but remove every per-token position/eviction load and
branch. All three candidates are F32-bit exact at starts 0/128/256/384.
Eleven-sample leaf timing improves cached metadata by **1.130–1.229x** for
global qrow4, **1.107–1.176x** for global qrow6, and **1.062–1.124x** for SWA
qrow4. The qualified production-shaped policy falls **12.8348 -> 11.8695 ms
(1.0813x)** per four-layer pattern, modeling **11.584 ms** pp512 saving.
Cached tracing names global qrow4 `<4,true,true,true>` at local32/VGPR64,
global qrow6 `<6,true,true,true>` at local32/VGPR88, and SWA qrow4
`<4,true,true,true,true>` at local32/VGPR64; all use zero LDS and scratch.
Runtime promotion still requires strict complete-initial-tile qualification
and a full-model complete-state gate. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-candidate.json`.
That gate is now satisfied. The runtime requires one consecutive complete
M128 tile, capacity/no-wrap safety, untouched eviction metadata, and the
existing non-verifier preappend schedule. Seven matched complete-model pairs
move cached-metadata rollback **552.144 -> 559.539 tok/s (+1.339%)**, save
**12.255 ms** at the medians, and preserve logits, hidden states, KV bytes,
token/logit, cursor, and allocation lifecycle exactly. gfx1151 now defaults
the capability with `prefill_dense_initial=false` as explicit rollback;
gfx1100 and all unsafe shapes retain the prior exact paths. Clean
selector-unset publication remains the next gate. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-default.json`.
Clean selector-unset publication reaches **559.290/523.090/439.044 tok/s** at
512/1K/4K, improving the previous production **1.420%/1.118%/1.607%**.
Cached tracing independently reaches **559.225 tok/s**, observes exactly
12 global-qrow4 / 36 global-qrow6 / 144 SWA-qrow4 dense-initial launches, and
cuts attention **153.226 -> 141.846 ms (-7.43%)**. The global qrow4/qrow6 and
SWA qrow4 resources remain local32, zero LDS/scratch, and VGPR64/88/64.
Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-dense-initial-production.json`.

The next bounded composite keeps that same dense-initial/KVLiveSpans
qualification but replaces positions 128/256/384 with exact BF16-to-F32 cache
widening, zero-workspace strided-batch F32 hipBLASLt QK/PV contractions, and a
causal F32 softmax. Start 0 remains on qrow4 because the BLAS route loses
there; partial, wrapped, explicitly evicted, verifier, decode, unsupported
head, and context-above-512 paths retain the established kernels. The CPU
fixture proves BF16 widening exactly, softmax against NumPy, and complete
attention within `rtol=2e-3, atol=2e-4`. Across 21 samples per qualified shape,
global context 256/384/512 improves **0.3785/0.5869/0.8003 ->
0.2823/0.3453/0.4365 ms**, and SWA improves
**0.6195/1.0079/1.4014 -> 0.3626/0.4634/0.6015 ms**, winning every pair.
Seven complete pp512 diagnostics improve **576.076 -> 602.518 tok/s** median
with 6/7 wins and deterministic state per mode. The changed F32 association is
quality-gated: pp512 all-exact KL improves **0.003246 -> 0.002214** and top-1
stays 2930. Cached tracing names the widen kernel at local256/VGPR24/LDS0 and
the causal-softmax kernel at local256/VGPR16/LDS32; both use SGPR128 and zero
scratch, followed by eight QK and eight PV library contractions. The route
owns **23,068,672 bytes** of scratch and no library workspace. gfx1151 now
selects it by capability with an explicit session rollback. Clean
selector-unset publication reaches **623.050/563.399/462.430 tok/s** at
512/1K/4K, improving the prior production **7.907%/3.307%/0.590%**. Corrected
cached attribution measures **82.763 ms** total pp512 attention versus
**143.669 ms** before the route. The complete category lane remains max KL
**0.049542582**, **316/320** top-1; the route-specific pp512 all-exact KL
improves **0.003246 -> 0.002214** with top-1 2930. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-hipblaslt-{candidate,production}.json`.

Replicating the eight widened KV heads into 48/72 query-head-major scratch is
closed. It permits one QK and one PV call, but raises route scratch
**23.1 -> 56.6 MB**. All 32 zero-workspace heuristics per contraction were
swept; the best qualified 48-layer model regresses **75.380 -> 105.483 ms
(+39.94%)**, with zero wins at contexts 256/384/512. The CPU-reference route
remains within **4.10e-8** absolute error. All candidate code is removed.
Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-replicated-heads-rejected.json`.

The successor packs only the 4.7-MB F32 query/output tile into head-major
order. K/V remains unreplicated, and one eight-way wide QK plus one wide PV
batch replaces sixteen calls. All 32 zero-workspace algorithms were screened
per shape. The qualified 48-layer leaf model improves **74.976 -> 71.169 ms
(-5.08%)**, with 21/21 wins at every production context and at most
**4.10e-8** absolute error. Seven pp512 pairs improve **621.806 -> 627.217
tok/s (+0.870%, 6/7 wins)**. All-exact KL improves **0.002214 -> 0.002097**,
production-vs-candidate KL is **0.000119**, and top-1 stays 2930. Cached
tracing names the query-pack/output-unpack kernels at local256/VGPR24/SGPR128,
zero LDS/scratch, around one QK and one PV contraction. gfx1151 enables the
capability. Clean selector-unset 512/1K/4K improves
**623.050/563.399/462.430 -> 629.101/566.858/463.903 tok/s
(+0.971%/+0.614%/+0.318%)**. The committed all-family trace cuts pp512
attention **82.763 -> 73.330 ms (-11.40%)** and dispatches
**4,145 -> 2,417**, with query-pack/output-unpack still local256/VGPR24/
SGPR128/LDS0/scratch0. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-packed-query-{candidate,production}.json`.

The successor causal-softmax production default maps exactly one score row to one
wave32 workgroup. It removes the prior local256 kernel's eight LDS partials
and four workgroup barriers while preserving complete `KVLiveSpans`
qualification, causal visibility, the F32 score/output ABI, and the
packed-query QK/PV contractions. The qualified 48-layer attention model
improves **72.738 -> 62.755 ms (-13.73%)**; seven complete pp512 pairs improve
**614.668 -> 620.032 tok/s (+0.873%, 6/7 wins)**. The changed reduction tree
passes its explicit quality gate: all-exact KL improves
**0.002097 -> 0.001796**, production-to-candidate KL is **0.0000971**, and
top-1 remains 2930. Cached tracing names the retained kernel at
local32/VGPR24/SGPR128/LDS0/scratch0. gfx1151 enables the capability with the
block256 body retained as explicit numerical rollback. Clean selector-unset
512/1K/4K improves **629.101/566.858/463.903 ->
632.618/568.845/464.606 tok/s (+0.559%/+0.351%/+0.152%)**; corrected
tracing cuts pp512 attention **73.330 -> 69.983 ms (-4.56%)** at unchanged
**2,417** dispatches. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-attention-wave-softmax-{candidate,production}.json`.

WPF-H5B independently screens the complete packed route on gfx1100/W7900 for
Laguna Q2 XL. No device body is ported: the transfer reuses exact BF16-cache
widening, complete `KVLiveSpans`, packed two-call F32 hipBLASLt QK/PV, wave32
causal softmax, direct packed-query production, and the packed output gate.
Start 0, partial, wrapped, evicted, verifier, decode, and unsupported routes stay
exact. The basic eight-QK/eight-PV composition regresses. Screening all four
available gfx1100 heuristics selects QK indices **2/1/3** at contexts
**256/384/512** and PV index **2** for both 48/72 heads. The selected-context
48-layer leaf improves **109.897 -> 62.655 ms (1.754x)** with max-row KL
**1.10e-15**, top-1 **100%**, and max abs **4.84e-8**. An explicit natural-M512
owner passes KL **0.000429**, token **2930**, deterministic complete state/KV,
and lifecycle. Cached tracing names exactly **144** widen/QK/softmax/PV stacks,
keeps **48** start-0 exact calls, and moves attention **488.304 -> 60.669 ms
(8.049x)** while complete kernel sum falls **3,001.692 -> 2,603.520 ms
(-13.265%)**.

Runtime qualification fails. The deterministic split-local M512 extension keeps
all 18 committed natural prompts as suffixes, and instrumentation observes all
**10,512** expected candidate launches with the six measured algorithm pairs.
The complete 576-step gate reaches maximum KL **0.444675 > 0.05** at **564/576
(97.917%)** top-1 despite **1.148x** diagnostic prefill, deterministic repeats,
and exact lifecycle. Remove the gfx1100 package policy/map, generic map seam,
and resident propagation; do not add these changed-association routes to the
catalog's production path. The existing leaf remains separately usable and
exact qrow4/M128 remains production. Evidence:
`benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-{candidate,rejected}.json`.

The long-context successor extends that packed-query/library design without
allocating a full-prefix score plane. Three separately registered dense-initial
primitives widen one logical BF16 K/V block to F32, update per-row online
softmax maximum/denominator and block-relative weights, then merge the F32 PV
numerator into persistent bounded state. The runtime uses 4,096-key blocks only
for complete global M128 tiles beginning at 4K; decode, verifier, partial,
wrapped, evicted, SWA, and unmeasured-backend paths retain complete
`KVLiveSpans` fallbacks. A two-block CPU fixture matches the established
attention route within tolerance. Cached gfx1151 tracing names the block
widener at local256/VGPR24/LDS0/scratch0, online tile softmax at
local32/VGPR16/LDS0/scratch0, and numerator merge at
local256/VGPR24/LDS0/scratch0. Complete-model 4K/16K/64K gates improve the
transitional full-score owner **0.121%/0.614%/7.516%**; mandatory 128K
improves **88.073 -> 99.100 tok/s (+12.521%)** while scratch falls
**4,298,113,024 -> 143,753,216 bytes (-96.655%)**. gfx1151 enables
`LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT`; the full-score owner remains a
temporary rollback. Evidence:
`benchmarks/results/2026-07-27-gfx1151-laguna-lc1-block4096-hipblaslt-production.json`.

The LC-2 rolling-SWA successor keeps the same tensorized arithmetic across ring
wraps without preappending over still-visible history. A new local256 gather
maps absolute positions `start-511..start-1` through complete
`KVLiveSpans`, rounds the 128 current F32 K/V rows through the established BF16
boundary, and emits one fixed 639-key F32 union. A local32 wave softmax applies
the exact row-dependent 512-token diagonal window before packed PV. The
M128/start>=512/consecutive/non-evicted gfx1151 route is separate; decode,
verifier, partial, evicted, and unmeasured paths retain the generic fallback.
The wrap fixture matches qrow2 within `rtol=2e-3, atol=2e-4`, with leaf maximum
absolute error **3.818e-8**. All 32 QK/PV algorithms select **25/18** and
improve **3.313 -> 0.684 ms (4.846x)**. Cached tracing reports gather
local256/VGPR32 and softmax local32/VGPR24, both SGPR128/LDS0/scratch0.
Complete 4K/16K/64K gates improve **23.418%/17.341%/8.072%**; mandatory 128K
improves **99.100 -> 103.520 tok/s (+4.460%)** with exact state and
**33,554,432 bytes** bounded scratch. gfx1151 enables
`LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT`. Evidence:
`benchmarks/results/2026-07-28-gfx1151-laguna-lc2-swa-hipblaslt-production.json`.

LC-3 keeps SWA at M128 and widens only complete global matrix chunks. At one
4K K/V block, exhaustive M128/M256/M512/M1024/M2048 screens reduce inclusive
global cost per row **43.309 -> 35.008 -> 32.123 -> 28.329 -> 26.382 us**;
M2048 selects QK15/PV2, uses **1,796,734,976 bytes** bounded scratch, and
stays within maximum absolute **4.284e-8** of qrow6. The companion 2K-context
screen selects QK15/PV1. SWA widening is rejected because the `511 + M` union
increasingly computes masked query/current pairs: per-row cost rises
**5.370 -> 21.124 us** from M128 to M2048. The gfx1151 capability
`LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS=2048` therefore applies only to complete
global M2048 chunks; SWA, partial tails, verifier, decode, evicted, and
unmeasured paths remain M128. Complete 4K/16K/64K improves
**5.615%/20.473%/39.112%** and mandatory 128K improves
**103.520 -> 149.684 tok/s (+44.594%)**, with exact tokens/cursors and full
lifecycle recovery. Evidence:
`benchmarks/results/2026-07-28-gfx1151-laguna-lc3-global-m2048-production.json`.

LC-4 specializes only the metadata-qualified dense-initial global block widen.
Because production allocates global KV in identity physical order, the
retained kernel directly streams `logical_start * width + index` and removes
per-element position, eviction, block-table, and physical-slot work. Paired
M2048/4K timing improves **0.250249 -> 0.234780 ms (-6.181%)**, stays
bit-exact to the generic widen, and leaves attention within maximum absolute
**3.446e-8** of qrow6. Trace resources fall VGPR24 -> VGPR16 with local256,
SGPR128, LDS0, and scratch0. The 4K/16K/64K aggregate A/B is mixed within
one-run noise; mandatory 128K is neutral at **149.249 versus 149.684 tok/s
(-0.291%)**, with exact state and lifecycle. gfx1151 promotes
`LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE`; complete `KVLiveSpans` and the
generic span-aware kernel remain the fallback. Evidence:
`benchmarks/results/2026-07-28-gfx1151-laguna-lc4-dense-contiguous-cache.json`.

The subsequent Q4 row64/local256 screen is rejected and removed. Eight waves
split 64 routed rows while preserving 32 FP32 accumulators/lane. Natural
routing nevertheless reduces M256/M512 tiles only **5.44%/16.84%**.
Cooperatively reconstructing one shared 128-column weight tile regresses the
actual layer-1 leaf **116.98%/103.34%**; retaining production's direct
per-column decode still regresses **28.31%/19.29%**. Both variants are BF16
bit exact against row32 and pass the CPU quality gate, but no kernel, wrapper,
mode, or test remains. Do not retry row64 without a variable-row or persistent
mechanism that avoids both padding and local256 residency loss. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-q4-row64-local256-rejected.json`.

### Laguna gfx1151 exact router token reuse

The shared `moe/router.hip` family now registers
`router_logits/f32/bf16_hidden_token_tile_{8,16}` alongside the existing
four-token route. Each wider tile preserves every token/expert's K traversal,
per-thread products, 256-thread reduction tree, and F32 output. At
M128/M256/M512, tile 8 improves the exact leaf by **1.346/1.304/1.341x** and
is the gfx1151 package default; tile 4 remains explicit rollback and the
unmeasured-backend default. Cached gfx1151 tracing names the tile-8 kernel at
local256, VGPR32, SGPR128, 8 KiB dynamic LDS, wave32, and zero scratch.
Production-shaped F32 logits, selected IDs, scaled routing weights, and
complete MoE BF16 output are byte-exact. Clean seven-pair production improves
tile-4 rollback **497.625 -> 503.349 tok/s (+1.150%)**, wins every pair, and
keeps all tile-8 samples above 500 (**minimum 501.698 tok/s**). Cached
all-family tracing measures **504.631 tok/s** and cuts router
**30.658 -> 23.315 ms**. Evidence:
`benchmarks/results/2026-07-26-gfx1151-laguna-router-token-tile8-production.json`.

The c=1 multi-expert projection screen is rejected and fully removed. To
avoid the 12-KiB wave strides of a literal lane-per-expert schedule, local256
candidate blocks cached the K3072 BF16 activation in 6 KiB LDS and assigned
8/16/32 experts per block with 32/16/8 adjacent K lanes per expert. All three
passed the CPU quality gate, stayed finite, preserved top-1/top-10, and had
maximum KL about `1.2e-13`. They nevertheless regressed the retained exact
one-workgroup-per-expert wave-0-tree projection by **8.77%/11.81%/28.97%**
in HIP-event time and **8.73%/11.77%/28.91%** in synchronized wall. The
monotonic loss closes activation-reuse/multi-expert tiling for the resident
expert-major F32 matrix; another attempt needs a different weight layout or a
persistent owner that preserves the 256-expert parallel grid:
[`rejected expert tiles`](../benchmarks/results/2026-07-31-gfx1151-laguna-router-expert-tiles-rejected.json).

The corresponding bounded persistent output-projection owner is also rejected
and fully removed. Exact local256 owners walked whole K6144/K9216 output
columns at grid stride, reducing 3,072 completion atomics to 40/80 while
preserving every BF16 boundary and reduction. The maximum occupancy-safe
grid80 still regresses full/SWA leaves **1.170%/1.542%** and adds
**0.158479 ms/token** across 12 full plus 36 SWA calls. The compiled resource
footprint permits only two blocks/CU, so 120/160 are rejected by the occupancy
guard. Do not retry this persistent composition without a higher-occupancy
reduction or lower-byte projection representation:
[`rejected persistent output owner`](../benchmarks/results/2026-07-31-gfx1151-laguna-f16-persistent-output-rejected.json).

The finer-grained c=1 shared-branch event split is rejected and fully removed
as well. Retaining post-router shared gate/up but holding shared down until
selected gate/up is exact and changes no kernels or launches, yet the rough
complete p512/d128 pair regresses **23.074630 -> 22.939050 tok/s (-0.58757%,
+0.256145 ms/token)**. The secondary join becomes critical. Keep the retained
post-router full-branch release; do not retry this event split without first
reducing shared-down bytes:
[`rejected shared-down split`](../benchmarks/results/2026-07-31-gfx1151-laguna-shared-down-after-selected-gate-rejected.json).

The diagnostic-only prefill routing replay also captures the normalized F32
weight beside every selected expert ID. Normal generation allocates neither
capture plane. `scripts/laguna_routing_replay.py` reports the final-one and
final-two route-mass distributions plus eligible/removed-lane fractions at a
frozen threshold grid, so any routed-width experiment starts from measured
model-wide work instead of a guessed cutoff.

### CPU-reference primitive oracles (**CPU reference landed**)

Registered by `hipengine.kernels.cpu_reference.register_cpu_reference_kernels()` under `KernelKey("cpu_reference", <layer>, "fp16")`:

- `embed`
- `rmsnorm`
- `shared_gate_combine+residual+rmsnorm` and `weighted_sum+shared_gate+residual+rmsnorm` (rounded MoE-tail boundary oracle)
- `linear`
- `qkv_proj`
- `rotate`
- `attention_decode`
- `full_attn_prefill` and `full_attn_prefill_varlen` (append-then-attend causal GQA + sigmoid gate oracles)
- `o_proj`
- `lm_head`

Fixture coverage currently includes `rmsnorm`, `linear`, `rotate`, masked `attention_decode`, and causal-GQA `full_attn_prefill`; varlen full-attn is covered by direct NumPy unit tests. Run committed fixtures with `python3 scripts/check_fixtures.py`.

Laguna gfx1151 source-F16 decode also has separately registered rows==1
`onebarrier_*` single/triple siblings. They preserve the local256 grid,
arithmetic, and VGPR16/LDS512/scratch0 resources while removing the generic
reducer's second broadcast barrier. All six natural roles are byte-exact and
improve **0.57-1.71%** at the leaf; the weighted family moves
**31.316 -> 31.097 ms/token (-0.698%)**. gfx1151 rows==1 runtime ownership is
default-on after seven exact same-session p512/d128 pairs improve
**14.758912 -> 14.800191 tok/s (+0.280%)** and cached whole-model tracing
records all **18,288 = 144 x 127** expected candidate calls with zero retained
decode GEMVs. `HIPENGINE_LAGUNA_F16_DECODE=gemv` remains the exact LD-2
rollback. Evidence:
[`retained one-barrier decode`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-onebarrier-retained.json).
The follow-up exact local128 owner is closed and removed: reconstructing the
same eight logical wave sums with four physical waves/output is byte-exact but
regresses every natural role and moves weighted family time
**31.039 -> 39.045 ms/token (+25.79%)**. Candidate and retained codegen are
both VGPR16/LDS512/scratch0, so the lost physical concurrency—not registers or
LDS—is the blocker. Evidence:
[`rejected local128 exact owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-local128-exact-rejected.json).
The exact local256 block2 follow-up is also closed and removed. It keeps all
eight waves and the complete reduction tree for each output, but shares each
workgroup across two adjacent columns. All six natural roles remain byte-exact
and regress **4.42-33.12%**; weighted family time moves
**31.571 -> 33.466 ms/token (+6.00%)**. The one-workgroup-per-output grid is
therefore part of the gfx1151 performance contract, not just the eight-wave
reduction. Evidence:
[`rejected exact block2 owner`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-block2-exact-rejected.json).
The successful LD-2 successor preserves that complete geometry and specializes
only the three production K widths at compile time. K3072/K6144/K9216 keep
every thread-local accumulation, shuffle, ordered wave sum, and store bit-
exact while fully unrolling the 12/24/36-iteration K loop. All six natural
roles improve **15.91-26.93%** and the weighted family moves
**30.952 -> 24.482 ms/token (-20.90%)**. Seven exact p512/d128 pairs improve
the retained one-barrier owner **14.786076 -> 16.391201 tok/s (+10.856%)**.
Cached tracing records all **18,288 = 144 x 127** intended calls with zero
fallback, local256/VGPR24/LDS512/scratch0. gfx1151 now selects fixed-K
automatically; `HIPENGINE_LAGUNA_F16_DECODE=onebarrier` is the exact rollback
and other backends remain unchanged. Evidence:
[`retained fixed-K decode`](../benchmarks/results/2026-07-28-gfx1151-laguna-f16-fixedk-retained.json).

Laguna gfx1151 selected-MoE decode now also has exact natural-shape resident-
T16 siblings: Q4 gate/up fixes `x_rows=1, rows=10, K3072, N1024`, while Q4
and planar-Q6 down fix ten distinct intermediate rows at `K1024, N3072`.
They preserve the local128 grid, thread-local K traversal, FMA/reduction order,
resident bytes, and BF16 store. Actual-weight leaves improve
**1.63%/21.19%/10.33%** with zero BF16 mismatches. Seven complete p512/d128
pairs improve **16.850003 -> 16.976046 tok/s (+0.748%)**, and cached tracing
records all **5,969/3,048/2,921** intended role calls with zero generic
selected-T16 fallback. Resources are local128/SGPR128/LDS512/scratch0 with
VGPR **200/104/80**. gfx1151 selects these siblings only for the admitted c=1
shape; other shapes/backends retain the generic exact routes. Evidence:
[`retained natural selected decode`](../benchmarks/results/2026-07-28-gfx1151-laguna-selected-natural-decode-retained.json).

The exact gfx1151 gate/up successor splits each resident 16-column T16 tile
across two local128 8-column workgroups while preserving every output
column's K ownership, FMA sequence, wave tree, ordered four-wave sum, and BF16
store. Actual layer-1 leaves improve **5.35-7.13%** and remain byte-exact;
tile4 and two separate single-projection launches are rejected at
**+10.51%/+9.15%**. Seven complete p512/d128 pairs improve
**16.991621 -> 17.007001 tok/s (+0.091%)**, all seven win, and all state is
exact. Cache-only tracing records **5,969** tile8 calls with zero natural
tile16/generic fallback at local128/VGPR96/SGPR128/LDS512/scratch0, versus
VGPR200 for the 16-column owner. gfx1151 defaults only the admitted natural
gate/up shape; peers/non-natural shapes retain registered exact fallbacks.
Evidence:
[`retained selected tile8 decode`](../benchmarks/results/2026-07-28-gfx1151-laguna-selected-natural-tile8-retained.json).

The exact tile8 parallel-tail sibling assigns the eight independent ordered
four-wave reductions and BF16 stores to lanes 0..7 instead of serializing all
columns on thread 0. It leaves the resident bytes, grid/local128 geometry,
K/FMA ownership, wave32 shuffle trees, and wave0..3 additions unchanged. On
actual layer-1 K3072/N1024 gate/up weights, all 21 counterbalanced pairs improve
**0.130259 -> 0.128862 ms (-1.072%)** with zero BF16 mismatches. Seven
same-session resident p512/d128 pairs all improve
**19.998518 -> 20.007478 tok/s (+0.0448%)** with exact generated state and
lifecycle, so gfx1151 selects the parallel tail by default. Cached tracing names
the `true` specialization at
local128/VGPR96/SGPR128/LDS512/scratch0 with a plausible 104.996-us fixture
duration. Evidence:
[`tile8 parallel-tail leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-tail-leaf.json),
[`retained resident gate`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-tail-retained.json).

The registered exact tile8 parallel-tail SiLU sibling consumes the two
independently BF16-rounded gate/up values inside the tail and writes the same
BF16 intermediate directly. This removes one launch plus 20,480 B each of
gate/up writes and reads per selected layer without changing any GEMV
arithmetic. On actual layer-1 K3072/N1024 weights, the complete current
two-launch chain improves **0.131058 -> 0.129529 ms (-1.167%)**; all 21
counterbalanced pairs win and the intermediate has zero BF16 mismatches.
Cached tracing keeps grid16384x10/local128,
VGPR96/SGPR128/LDS512/scratch0. All seven same-session resident p512/d128
pairs improve **20.008491 -> 20.063975 tok/s (+0.2773%, -0.1382 ms/token)**
with exact generated state and lifecycle, removing **47 compute
launches/token**, so gfx1151 promotes the fused route. Evidence:
[`tile8 parallel-tail SiLU leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-silu-leaf.json),
[`retained resident gate`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-parallel-silu-retained.json).

The routed Q4 expert gate/up catalog now also registers decode and prefill
consumers for quant key `gguf_q4_k_t16_dual_interleaved_v1`. One 4,736-byte
tile interleaves the two former 2,368-byte T16 records and replaces both
resident allocations. The c=1 fused tile8 decoder and generalized rows 2-31
fallback preserve each ordinary projection's exact K/FMA/reduction/BF16
boundary; M32+ uses the exact D8
MMQ128x32/wave-column/direct/double-buffer prefill owner. Materialization,
profile accounting, ownership, and teardown are covered by direct device
tests. Cached tracing reports local128/VGPR80/LDS512/scratch0 for the exact
short-row decoder and local128/VGPR96/LDS3072/scratch0 for production MMQ.
Same-revision production improves decode
**22.130173 -> 22.260802 tok/s (+0.59027%)** and pp512
**654.569 -> 655.535 tok/s (+0.14757%)** with unchanged
**79,022,522,196-byte** residency. Tracked-clean selector-unset publication
reproduces **22.262504/656.990 tok/s**:
[`paired expert production`](../benchmarks/results/2026-07-31-gfx1151-laguna-q4-t16-dual-interleaved-production.json).

An exact local32 replay is rejected and removed. One physical wave
reconstructed the retained local128 tile8 body's four logical wave32 K/FMA
chains and reductions in order, deleting 512 B of LDS and the block barrier
without changing BF16 output. On actual layer-1 K3072/N1024 weights it
regressed **0.126660 -> 0.188025 ms (+48.45%)** and won 0/9 pairs. The four
physical waves are required to hide this memory/decode work; do not retry
local32 logical-wave replay:
[`rejected wave32 replay`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-tile8-wave32-replay-rejected.json).

The separately registered natural selected-down parallel-tail siblings retain
all four physical waves and assign the 16 independent ordered wave0..3 sums
and BF16 stores to lanes 0..15. Actual Q4 and planar-Q6 down leaves are
byte-exact and improve **3.125%/0.940%**, with 20/21 and 21/21 paired wins.
Cached tracing preserves local128/LDS512/scratch0 and VGPR **104/80**.
The resident gate and clean publication passed, and gfx1151 selects these
natural shapes by default:
[`selected-down parallel-tail leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-selected-down-parallel-tail-leaf.json).

The exact weighted successor preserves that complete route-parallel producer
grid and every per-route BF16 output. One self-resetting completion counter
per 16-column output tile lets only the last of ten route producers replay the
registered slot-order F32 weighted FMA chain. This is not the rejected serial
top-10 owner: all **1,920 producer workgroups** remain. Natural Q4 and
planar-Q6 leaves improve **3.940%/3.752%** with byte-exact per-route and
routed outputs. Cached tracing keeps grid1920/local128,
VGPR **104/80**, SGPR128, LDS512 B, and scratch0. Seven same-resident
p512/d128 pairs improve
**22.071805 -> 22.139076 tok/s (+0.30479%, 7/7 wins)** with exact state,
so gfx1151 promotes the composite. Tracked-clean selector-unset publication
reaches **22.119461 tok/s (+0.25472%)** and the complete census proves
**529 -> 482 dispatches/token**, with selected down plus weighting
**4.888128 -> 4.819087 ms/token (-1.412%)**:
[`parallel weighted retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-selected-down-parallel-weighted-retained.json).

The gfx1151 Q4 selected-down weighted owner now pairs adjacent-column payload
transport without changing arithmetic ownership. One nibble byte, aligned
`d`/`dmin` FP16 pairs, and scale/min byte pairs feed two unchanged per-column
F32 FMA chains; the four-wave sums, BF16 route outputs, completion-counter
ownership, and slot-order weighted result remain byte-exact. The actual
E256/K1024/N3072/top-10 leaf improves
**0.058665 -> 0.058097 ms (-0.967%)**. Cache-only tracing records
**57.107 -> 56.827 us**, local128/SGPR128/LDS512/scratch0, and allocated VGPR
**104 -> 80**. Seven same-resident p512/d128 pairs improve
**22.762554 -> 22.793632 tok/s (+0.13653%, 5/7 wins)** with exact trajectory,
unchanged **79,066,169,172-byte** residency, and complete lifecycle recovery.
gfx1151 defaults the paired payload; the scalar-column registered weighted
owner is the explicit constructor rollback:
[`paired Q4 down retention`](../benchmarks/results/2026-07-31-gfx1151-laguna-q4-selected-down-paircoeff-retained.json).

A physical-local256 selected-down/D9 successor is rejected and removed. It
kept the first 191 weighted output tiles on the unchanged local128 owner, then
launched only the final 16-column tile as ten local256 route blocks so the
last producer could run the exact wave-0 D9 tree with 256 physical lanes and
without a second output-tile completion tier. All selected/routed/hidden/norm
BF16 fields were byte-exact, but actual Q4 and planar-Q6 leaves regressed
**5.625%/5.188%** versus retained selected-down plus standalone D9. Together
with the earlier local128 logical-lane rejection, this closes
selected-down-integrated D9 absent a different consumer or persistent design:
[`rejected final-tile local256 owner`](../benchmarks/results/2026-07-31-gfx1151-laguna-selected-down-lasttile256-moe-tail-rejected.json).

The separately registered `argmax/f32/top1_i64_publish_control` primitive
preserves the retained two-stage top-1 reduction and minimum-index tie break,
then lets stage 2's winning thread publish the next embedding token plus the
next scratch/KV row positions. gfx1151 Laguna c=1 consumes those scalars only
when the following host token and serial position match the synchronized
winner; forced tokens, registry misses, rows/verifier, and peer backends retain
ordinary host publication. The GPU fixture matches the baseline ID and FP32
value bits and all control values. Seven same-resident p512/d128 pairs improve
**22.853913 -> 22.868721 tok/s (+0.06479%, 7/7 wins)**. Cache-only tracing
proves **482 unchanged model kernels + two D2H copies** per steady transition
instead of **482 + five**, with the candidate stage 2 at
local256/VGPR16/SGPR128/LDS0/scratch0 and **2.043 us median**:
[`argmax control retention`](../benchmarks/results/2026-07-31-gfx1151-laguna-argmax-control-publish-retained.json).
Tracked-clean selector-unset production at `7d85771a8` measures
**22.865539 tok/s / 43.733936 ms/token**, up **0.04105%** and
**0.017954 ms/token** from the preceding clean checkpoint with exact state,
unchanged residency, and a remaining **0.904414-ms/token** same-GGUF Vulkan
gap:
[`argmax control production`](../benchmarks/results/2026-07-31-gfx1151-laguna-argmax-control-publish-production.json).

### gfx1100 HIP kernels (**hipEngine landed**)

The source-F16 catalog now also includes a separately registered
`linear_quad/fixedk_onebarrier_bf16_f32_out` c=1 diagnostic in
`hipengine/kernels/hip_gfx1100/linear/laguna_f16_projection.hip`, exposed by
`launch_f16_weight_linear_quad(...)`. One grid selects Q, K, V, or gate while
preserving every output column's retained local256 K order and reduction tree;
the registered triple plus singleton remains the unfused fallback. The
focused K3072 gfx1151 device fixture is F32-bit exact. Cached tracing names
`laguna_f16w_quad_fixedk_onebarrier_gemv_kernel<3072>` at
local256/VGPR24/SGPR128/LDS512/scratch0 and measures **8.095 us** on a tiny
33+8+9+7-output fixture versus **20.277 us** for the two retained launches.
All seven same-resident p512/d128 candidates win
**21.944420 -> 22.026384 tok/s (+0.37351%, -0.169573 ms/token)** with exact
state and unchanged residency, so gfx1151 now defaults the quad; peer
backends and explicit disable retain triple plus singleton.

The attention catalog also has a separately registered rows==1
`attention_projection+head_rmsnorm+partial_rotary+kv_write` composite with
`fp16_weight+laguna_f32_weight` and global/SWA fixed-K variants. It keeps one
local256 block per exact Q/K/V/gate output column, then uses a per-head
completion counter so only the final Q producer runs the retained head
RMSNorm/RoPE body and only the final combined K/V producer runs the retained
K/V norm/RoPE/BF16 append body. The existing `linear_quad` plus head/KV
composite remains the unfused fallback. Natural Q48/global and Q72/SWA
fixtures match every projection F32 bit, rotated F32 bit, BF16 K/V byte,
`KVLiveSpans` field, and counter reset. Cached tracing names
`laguna_f16_projection_head_rmsnorm_partial_rotary_write_kv_bf16_kernel` at
local256/VGPR24/SGPR128/LDS512/scratch0. Seven same-resident p512/d128 pairs
are throughput-flat but mechanically positive at
**22.016010 -> 22.017120 tok/s (+0.00504%)**, with a paired-median
**0.002932-ms/token** saving and five of seven wins. Tracked-clean production
is aggregate-flat at **22.007742 tok/s (-0.1097%)**, while the complete
127-transition census proves **625 -> 577 dispatches/token**, a shorter
**45.699715 -> 45.660100-ms/token** span, and
**2.006962 -> 1.882766 ms/token** span-minus-kernel time. gfx1151 therefore
selects the exact mechanical win; peer backends and explicit disable retain
the exact two-launch chain. Evidence:
[`projection/head/KV retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-f16-projection-head-kv-retained.json),
[`production and census`](../benchmarks/results/2026-07-30-gfx1151-laguna-f16-projection-head-kv-production.json).

The same source-F16 catalog now registers
`linear+add+rmsnorm/fp16_weight+gguf_f32_weight/fixedk_onebarrier_bf16_out`.
One local256 block retains each K6144/K9216 attention-output column's fixed-K
dot and BF16 store; the last of 3,072 producers observes every store, executes
the existing local256 residual-add/RMSNorm tree, and resets a stream-reused
completion counter. The registered fixed-K projection plus add/RMSNorm chain
remains the unfused fallback. Global and SWA device fixtures match every
projection/residual/norm BF16 byte and the reset counter. Cached gfx1151
tracing reports local256/VGPR24/SGPR128/LDS512/scratch0. Seven same-resident
p512/d128 pairs all improve **22.005296 -> 22.062263 tok/s (+0.25888%)**,
with a **0.113153-ms/token** paired-median saving and unchanged residency.
Tracked-clean production reaches **22.063262 tok/s (+0.25227%)** and the
complete census proves **577 -> 529 dispatches/token**,
**45.660100 -> 45.543776 ms/token** span, and
**1.882766 -> 1.793306 ms/token** span-minus-kernel time. gfx1151 selects the
exact composite in production:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-f16-output-add-rmsnorm-retained.json),
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-f16-output-add-rmsnorm-production.json).
### Moonshine source-F16 projection baselines (**hipEngine landed**)

The current runtime, CUDA-review findings, and ordered gfx1151 transfer gates
are maintained in [`MOONSHINE.md`](MOONSHINE.md). Kernel catalog entries below
remain the arithmetic/source authority; CUDA measurements never select HIP
geometry without an independent gfx1151 gate.

`linear/moonshine_projection.{hip,py}` provides raw-pointer FP16-input,
FP16-weight, FP16-output single, paired, and triple projections with FP32
accumulation. The single wrapper also has an explicit row-precompute key for
encoder-frame K/V materialization. The keys are
`moonshine_projection/single_fp32_accum`,
`moonshine_lm_head/tied_fp32_accum`,
`moonshine_projection_rows/single_fp32_accum`,
`moonshine_projection_bias/single_fp32_accum`,
`moonshine_projection_pair/pair_fp32_accum`,
`moonshine_cross_kv_precompute/pair_head_major_fp32_accum`, and
`moonshine_qkv_proj/triple_fp32_accum`, all under `quant="fp16"`; gfx1151
uses the peer backend alias and native `gfx1151` code object. The tied LM-head
entry preserves the singleton reduction under a distinct HIP kernel symbol so
whole-token profiles can separate its 30.671-MB stream from other 416-wide
projections. Phase-3 runtime uses the separately registered
`tied_wave8_fp32_accum` symbol: one local256 block owns eight independent
wave32 vocabulary rows. The one-row-per-block local256 wrapper remains the
explicit fallback. The cross-K/V variant preserves the same dot products but writes direct resident
`[heads,frames,52]` storage instead of row-major `[frames,416]`, avoiding a
separate transpose or temporary frame buffer. Four-row output matches the
transposed NumPy projection within max absolute error `3.052e-5`; cache-only
tracing names the head-major pair at 17.073 us, local256/VGPR16/LDS512/scratch0.

The peer `cuda_sm120a/linear/moonshine_projection.{cu,py}` C1c port preserves
the same nine raw-pointer FP16 keys under `quant="fp16"` without aliasing HIP
wrappers or adding backend branches: single, row-precompute single,
bias-aware single, triple QKV, row pair, head-major cross K/V, tied LM head
(plain + wave8), and the C1d fc1 gated-SiLU / fc2 bias+residual boundaries.
It builds only with `-arch=sm_120a` and uses full-mask `__shfl_down_sync`
warp reductions. On an RTX PRO 6000 Blackwell (GPU0) all projection families
pass the independent NumPy oracle within 2.0e-3 (all finite); the 36,864-row
tied LM head matches and wave8 equals the plain tied path, and the head-major
cross-K/V matches the transposed oracle layout. An Nsight Systems cache-only
trace observes all seven C1c kernels and no compiler child; single-run
launch-level medians are 1.7/2.6/2.4 us single/triple/bias, 31.4 us pair,
33.0 us cross-K/V head-major, and 31.0/7.0 us LM tied/wave8. These are
bring-up diagnostics, not a performance promotion; the tied-vs-wave8 LM gap is
a C3 leaf hint pending enclosing-layer/generation evidence.

The production-shape fixture covers hidden 416, batch-one single/triple, and a
40-row paired cross-K/V baseline against the independent NumPy oracle. Maximum
absolute error is `3.052e-5`, all outputs are finite, and the cache-only
`rocprofv3` smoke names all three kernels at local256/VGPR16/LDS512/scratch0.
Measured four-row diagnostic durations are 20.759/56.788/69.772 us for
single/pair/triple; these are bring-up diagnostics, not promoted performance
claims. The bias-aware sibling adds each FP16 bias to the FP32 reduced dot
before the FP16 boundary, matching decoder `fc1`/`fc2`; its cache-only
four-row trace is 10.620 us at the same local256/VGPR16/LDS512/scratch0 tuple.

The Phase-3 gfx1151 decode route retains every key above as a fallback but uses
measured production geometry: local32 for triple QKV, bias-aware fc1, and
head-major cross-K/V; local64 for bias-aware fc2; and the existing vectorized
`dense_gemv_out_fp16` local64 kernel for the 24 unbiased singleton Q/O calls.
The generic Moonshine reduction now returns directly from wave32 and removes an
unneeded second block barrier for larger groups; thread-0 arithmetic is
unchanged. A 15x20-launch event matrix measures single/triple/fc1/fc2/cross at
3.436/4.963/12.373/6.665/52.251 us versus the original local256
5.783/10.324/25.142/7.035/165.463 us. Existing dense three-dispatch triple,
WMMA, rocBLAS GEMM-ex, and inclusive hipBLASLt candidates are slower. Full
fixture/token/selected-region promotion evidence remains in the Moonshine
experiment ledger rather than treating these leaf numbers as a standalone
speed claim. Fixture token equality is required through first EOS and at every
captured boundary position; uncaptured teacher-forced continuation after EOS is
reported diagnostically because it is neither generated ASR output nor a Tier-B
boundary.

`norm/moonshine_layernorm.{hip,py}` registers
`moonshine_layernorm/fp16/fp32_stats`. One local256 block computes the FP32
mean and centered variance in two passes, then writes the weighted FP16
boundary. The hidden-416 seven-row fixture is byte-equal to the NumPy oracle.
Cache-only gfx1151 tracing names `moonshine_layernorm_fp16_kernel` at 14.948 us,
local256/VGPR24/LDS512/scratch0. Moonshine uses LayerNorm; this does not reuse
or alter the Qwen/PARO RMSNorm math. The separately registered
`moonshine_residual+moonshine_layernorm/rounded_fp32_stats` composite writes the
exact rounded FP16 residual boundary, synchronizes the workgroup, and computes
the following FP32-statistics LayerNorm in the same launch. The production
hidden-416 boundary is byte-exact/all-close to the primitive chain and improves
5.384 -> 3.657 us (1.47x), local256/VGPR24/LDS512/scratch0.

`fused/moonshine_glue.{hip,py}` registers explicit FP16 primitives for device
int64 embedding lookup, rounded residual add, pair-interleaved partial RoPE,
fixed self-cache append, and deterministic lowest-ID FP16-logit argmax. It also registers
`moonshine_partial_rope+moonshine_self_cache/interleaved_fixed_append`; the
separate RoPE and cache keys remain its required unfused fallback. Positions
0/1/63/193 and logical 8x52 heads are byte-equal to the NumPy oracle, and the
composite is byte-equal to the two-kernel chain. Cache-only gfx1151 tracing
reports embedding/residual/RoPE/cache/composite at
3.326/1.403/2.645/1.844/1.763 us, local256, LDS0, scratch0, and maximum VGPR24.
The 36,864-way argmax is tie-stable and traces at 31.219 us,
local256/VGPR16/LDS3072/scratch0; it uses no caller scratch allocation.

The peer `cuda_sm120a/fused/moonshine_glue.{cu,py}` C1 port preserves these
six raw-pointer FP16 keys without aliasing HIP wrappers or adding backend
branches to model code. It builds only with the architecture-qualified
`-arch=sm_120a` target and uses CUDA warp32-compatible block reductions. On an
RTX PRO 6000 Blackwell (GPU0), embedding, rounded residual, lowest-ID argmax,
partial RoPE, fixed cache append, and the bounded RoPE+cache composite pass the
independent NumPy oracle at positions 0/1/63/193; the composite is byte-equal
to the separate CUDA chain. An Nsight Systems cache-only trace observes all six
expected kernels. Single-run diagnostic medians are 0.576 us residual, 1.024 us
embedding, 8.256 us argmax, and 1.168/0.784/1.200 us for RoPE/cache/composite
(the last three are four-instance medians). These are bring-up diagnostics, not
a complete decoder or performance promotion.

The peer `cuda_sm120a/norm/moonshine_layernorm.{cu,py}` C1b port preserves the
HIP LayerNorm family: `moonshine_layernorm/fp16/fp32_stats` and the fused
`moonshine_residual+moonshine_layernorm/rounded_fp32_stats` composite, both
raw-pointer, no backend branches. The CUDA kernels use the same ordered FP32
warp-butterfly plus cross-warp shared reduction (full-mask `__shfl_down_sync`)
and the residual+LayerNorm launch writes the rounded FP16 boundary before
computing FP32 statistics over that same buffer, exactly like the HIP
reference. On an RTX PRO 6000 Blackwell (GPU0) the hidden-416 kernels pass the
independent NumPy FP32-stats oracle across decoder and encoder row counts
1/7/40/207/1248 (allclose 3.0e-3), the fused residual boundary is byte-exact to
the primitive chain, and all outputs are finite. A cache-only Nsight Systems
trace observes both expected kernel identities with no compiler child;
per-row-bucket durations for rows 1/7/40/207/1,248 are
1.952/1.952/2.016/2.336/4.768 us (LayerNorm) and
2.080/2.144/2.240/2.592/5.440 us (residual+LayerNorm). A batch-timed
CUDA-event schedule screen (2000-launch batches on GPU0) shows 256 threads is
the measured best below 768 rows and 128 threads is best from 768 upward for
both kernels (256 is roughly 1.4-1.5x slower at 1,248 rows), so the wrappers
auto-select 128 threads for rows >= 768 while keeping 256 for decoder/short
buckets; an explicit ``threads=`` always overrides. A thread-sweep correctness
gate covers threads 32/64/128/256 across hidden 52/416 and rows 1/7, plus
poisoned-output/epsilon and constant/extreme-row coverage. Against the
model-derived CUDA synthetic and ``audio-konichiwa`` fixtures, final decoder
LayerNorm output is byte-exact at positions 0/1/8/32/64/128/193 (opt-in GPU
gate). These are bring-up diagnostics, not a performance promotion.

`fused/moonshine_mlp.{hip,py}` registers
`moonshine_gated_silu/fp16/value_gate_split`: it consumes the bias-aware FP16
`fc1` boundary as `[value,gate]`, evaluates SiLU in FP32, multiplies in FP32,
and writes FP16. The complete unfused production-shape chain is bias-aware
`416->3328` projection, gated SiLU to 1664, bias-aware `1664->416`
projection, and the registered residual primitive. It is byte-equal to the
NumPy decoder-MLP-plus-residual oracle. Cached gfx1151 tracing reports
51.497/4.289/12.664/2.966 us for fc1/activation/fc2/residual; the activation is
local256/VGPR16/LDS0/scratch0. Phase 3 adds exact
`moonshine_mlp_fc1/bias_gated_silu_fp32_accum` and
`moonshine_mlp_fc2_residual/bias_rounded_residual_fp32_accum` composites. The
first computes paired value/gate rows with the original FP32 reduction and FP16
boundaries before SiLU; the second preserves the FP16 projection boundary before
the rounded residual. Complete boundaries improve 15.265 -> 9.636 us and 9.278
-> 6.899 us; the selected whole MLP+next-norm chain improves 30.604 -> 22.116 us
(1.38x). Resources are local32/VGPR16/LDS0/scratch0 and
local64/VGPR16/LDS512/scratch0. Primitive fallbacks remain registered.

The peer `cuda_sm120a/fused/moonshine_mlp.{cu,py}` C1d port preserves the
standalone `moonshine_gated_silu/fp16/value_gate_split` primitive: it consumes
the bias-aware FP16 `fc1` boundary as `[value,gate]`, evaluates SiLU in FP32,
multiplies in FP32, and writes the FP16 activation, one thread per activated
element. It builds only with `-arch=sm_120a` and uses the CUDA warp32-safe
launch geometry. On an RTX PRO 6000 Blackwell (GPU0) the kernel matches the
independent NumPy oracle across decoder and encoder row counts
1/7/40/207/1248 (allclose 1.0e-3, all finite), and the complete unfused
production-shape chain (bias-aware fc1, gated SiLU, bias-aware fc2, residual)
matches the NumPy decoder-MLP-plus-residual oracle (allclose 5.0e-3). A
cache-only Nsight Systems trace observes the single gated-SiLU kernel identity
five times (one per row bucket) with no compiler child; single-run launch-level
medians are about 0.9 us. The C1d fused MLP boundaries registered in the C1c
projection module are also gated: the fused fc1 (bias-aware projection + paired
gated SiLU) and fused fc2 (bias-aware projection + rounded residual) chain
matches the NumPy decoder-MLP-plus-residual oracle byte-exact on GPU0. These
are bring-up diagnostics, not a performance promotion.

`attention/moonshine_attention.{hip,py}` registers
`moonshine_self_attention/fp16/fixed_cache_logical_dim` and
`moonshine_cross_attention/fp16/resident_masked_logical_dim`, with matching
explicit CPU fallback keys. One wave32 owns each of the eight heads, reduces
only the logical 52 dimensions, and maintains FP32 online-softmax max,
denominator, and output state without 56-dimension padding or score scratch.
Self-cache past lengths 0/1/2/8/32/64/128/193 and masked resident cross-cache
lengths 40/207/1248 pass the NumPy oracle. The maximum-length smoke is finite;
self output is byte-equal and masked cross output has max absolute error
`4.768e-7` and relative L2 `7.961e-6`. Cache-only gfx1151 tracing names the
self/cross kernels at 96.982/558.650 us for lengths 194/1248, local32,
VGPR32/SGPR128/LDS0/scratch0. These remain the explicit correctness fallbacks.

The Phase-3 gfx1151 cross-attention default is the separately registered
`resident_masked_parallel_tokens` route. One local256 block owns each head;
eight waves process interleaved masked tokens and merge FP32 online-softmax
partials in 2,048 bytes of LDS. It keeps logical dimension 52, resident
head-major FP16 K/V, scratch0, and no score plane. Clean 15x20-launch timing
improves fallback to selected at 40/24, 40/40, 207/105, 207/207, and
1,248/1,248 frames by 3.27x/3.64x/6.22x/5.93x/6.71x. Full synthetic and six
padded real Tier-B gates retain exact generated IDs through EOS, 100% logit
top-1, zero timed allocation, and clean teardown. The clean decoder profile
reduces eight past-1 cross calls from roughly 0.23 ms to 0.060 ms and moves the
long-cache bottleneck to self attention. Detailed evidence is in the Moonshine
experiment ledger's `results/2026-07-31-hip-phase3-cross-attention.md`.

The Phase-3 self-attention route registers a branch-specialized one-wave
candidate plus `fixed_cache_parallel_tokens` at local64/128/256. The runtime
keeps the exact fallback at position 0, uses two waves at position 1, four waves
at positions 2-3, and eight waves from position 4 through 193. These are general
cache-length buckets, not fixture/token conditions. At positions 1/2/3/4/8/32/
64/128/193, selected leaf speedups are 1.07x/1.15x/1.26x/1.33x/1.57x/3.05x/
4.13x/5.26x/6.03x. The local256 route uses VGPR32/SGPR128/LDS2,048/scratch0
and no score plane. Full synthetic and all six padded real Tier-B gates retain
exact generated IDs, 100% logit top-1, zero timed allocation, and clean
teardown. Clean position-193 profiling cuts self attention from 0.999 ms /
57.4% to 0.192 ms / 20.0%, while the selected event median falls from 1.951 to
1.122 ms. Evidence is in the experiment ledger's
`results/2026-07-31-hip-phase3-self-attention.md`.

`runtime/moonshine.py` composes these primitives into the complete unfused
resident decoder. Code objects are explicitly prepared before timed work;
validated FP16 encoder hidden state and int32 masks upload once; eight
head-major cross-K/V pairs precompute once; then each sequential token runs
embedding, eight self/cross/MLP layers, final LayerNorm, tied FP16 LM projection,
and lowest-ID argmax without a tracked allocation. A diagnostic callback can
snapshot the 25 per-position layer/final boundaries without changing the
default chain. The retained synthetic 40-frame fixture checks 310 cross,
boundary, logit, and self-cache tensors at positions 0/1/8/32/64/128/193,
plus every one of 194 greedy selections. All tokens are exact; maximum boundary
absolute error is `0.75`, maximum relative L2 is `0.003506`, selected-logit
KL max/mean is `1.538e-5/8.824e-6`, top-1 is 100%, timed tracked allocations
are zero, and teardown returns 129,686,968 resident bytes to zero. This is the
correctness fallback; synchronized timing and baseline publication are a
separate Phase-2 gate.

The retained Phase-3 runtime now combines tuned projections, wave8 LM head,
masked cross attention, cache-bucketed self attention, residual+LayerNorm, and
the two MLP composites. It issues 103 kernels/token versus the 135-kernel Phase-2
fallback. Clean past-1 timing is 0.861 ms HIP event / 0.915 ms wall; the six-file
decoder-only median is 5.449 ms with exact generated IDs. Past-1 aggregate
kernel time is 0.767 ms. Detailed bounded-fusion evidence is in the experiment
ledger's `results/2026-07-31-hip-phase3-bounded-fusions.md`; fixed-address graph
capture/replay remains the next structural step.

### gfx1100 HIP kernels (**hipEngine landed**)

WPF-1 adds separately registered exact raw-Q5_K/Q6_K prefill
rowbatch4/8/16/32 primitives in `quant/gguf_k_gemv.{hip,py}`. Fixed grid Y lets
one workgroup reconstruct each encoded weight once per row slab while
preserving every row's scalar thread-local K order and wave/cross-wave
reduction tree. Arbitrary positive lengths, including partial tails through 33
rows, are BF16/F32-bit exact. The original ten-role M128 screen improved
**1.268-2.412x** with rowbatch4 and **1.340-3.347x** with rowbatch8.

The WPF-1W extension is exact on all ten actual UD-Q2_K_XL roles. Their
**unweighted diagnostic** one-each event sum moves rowbatch8 **45.1883 ms** to
rowbatch16 **41.2040 ms (1.0967x)** and rowbatch32 **39.2782 ms (1.1505x)**;
these ratios are not end-to-end forecasts. Rowbatch16 beats eight on eight
roles; rowbatch32 wins the other eight but leaves N48/N72 to retained smaller
slabs and explicit crossover/bisection. Cached
W7900 tracing names all eight new Q5/Q6 BF16/F32 bodies at local128. RB16 uses
VGPR40/SGPR128/LDS512/scratch0 in rocprof (code-object VGPR38, SGPR69/70, no
spills/private). RB32 uses VGPR80/SGPR128/LDS1024/scratch0 (code-object VGPR73,
SGPR107, zero VGPR/private spills, but 14/5 Q5/Q6 SGPR spills). The measured
RB32 timings include that scalar-spill cost. HIP's occupancy query still admits
**8 workgroups / 32 wave32 waves per CU**, the hardware maximum, for every new
body; runtime `SQ_WAVE_CYCLES` is unavailable on this profiler stack.

The gfx1100 package now selects rowbatch32 by default only around Laguna bulk
row-layer execution. The shared-weight M128 gate proves rowbatch16, rowbatch32,
and an RB32 repeat bit-exact across all 48 hidden boundaries, logits, active
K/V, every `KVLiveSpans` field, IDs, positions, and lifecycle. Diagnostic
complete-pass wall is RB8 **1.4096 s**, RB16 **1.3335 s**, and RB32 **1.3020
s**. Clean paired RB32 improves pp512/pp1K **79.023/73.610 -> 85.174/78.946
tok/s (+7.783%/+7.249%)**, every candidate sample wins, and selector-unset
publishes **85.481/79.555 (+7.408%/+6.768% over the RB8 headline)**. Explicit
zero/4/8/16/32 remain on gfx1100 while gfx1151 excludes every key; rows <=8 and
unsupported shapes/dtypes/layouts/quants retain the registered scalar/backend
fallback. M256/M512 must retrace RB32 grid Y, launch waves, duration, and
runtime occupancy; the same code object's SGPR spills are compile-time
invariant, but M128 does not establish achieved scheduling at wider grids.
Evidence: [`rowbatch4/8 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-rowbatch-primitive.json) ·
[`rowbatch8 production`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-rowbatch8-production.json) ·
[`rowbatch16/32 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-rowbatch16-32-primitive.json) ·
[`rowbatch32 production`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-rowbatch32-production.json).

WPF-C1 selected matrix256/attention128 as the preceding gfx1100 package
capacity while keeping explicit M128 fallback and M512 as the WPF-2 comparator.
One clean rotating screen measures M128 **85.028/78.672**, M256
**85.855/79.526 (+0.973%/+1.086%)**, and M512 **85.826/79.507
(+0.939%/+1.062%)** at 512/1K. M256 had the lowest direct-route aggregate wall
and planned row/MoE scratch **219,514,912 bytes**, half M512's **439,021,600**.
Complete logits, all 48 hidden boundaries, K/V/live spans, shared-prefix
routing, repeats, and lifecycle are exact. The cached M256 trace records RB32
at observed grid Y4/Y8 with unchanged local128/VGPR80/LDS1024/scratch0 and
selected-IQ grids 1,280/2,560. Its clean package-resolved publication is
**86.239/80.452 tok/s (+0.887%/+1.128% over matrix128/RB32)**. Explicit
`prefill_chunk_size` continues to override package capacity.
Evidence: [`matrix256 production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-matrix256-retained.json).

WPF-2 added exact rowbatch8 siblings for grouped IQ2/IQ3 gate/up and grouped
IQ3 single/down execution, including fused post-SiLU variants. They preserve
each row's group order, FMA sequence, wave shuffle tree, serial wave
accumulation, BF16 projection boundary, and arbitrary tails. The grouped
gate/up K partition was not the production owner: actual tiny activations
exposed 11/30,720 values where local256/group8 differed from retained
local64/pair16 arithmetic. Post-WPF-2b cleanup removes those unowned Laguna
rowbatch8/fused-SiLU gate variants while preserving the base/rowbatch4/adaptive/
auto grouped-dual chain used by Qwen3.5 GGUF. Laguna retains route-major or
pair16 gate/up, then one stable expert compaction/gather feeds IQ3 rowbatch8 or
IQ4 auto down before the registered sorted-lane weighted reducer restores token
order.

Actual layer 1 (IQ2->IQ3) and layer 47 (IQ3->IQ4) are bit-exact at post-SiLU,
every routed down row, and final MoE output. A shared-weight M128/M256/M512 gate
preserves all 48 hidden boundaries, logits, K/V and every `KVLiveSpans` field,
routing, repeats, and lifecycle at KL0/top-1 100%. Direct -> grouped improves
M256 **86.175/79.924 -> 96.643/89.049 tok/s (+12.147%/+11.417%)** and M512
**86.129/79.887 -> 98.289/90.555 (+14.118%/+13.354%)** at 512/1K. M512 beats
the independently measured grouped M256 rows by **1.703%/1.691%**, so the
WPF-2 checkpoint selected matrix512/attention128 plus `grouped_exact`;
explicit M128/M256 and `direct` remain exact rollbacks, while unsupported
quant/key misses fail closed to the registered exact route-major chain. Its
clean package-resolved publication reached **99.230/91.559 tok/s** at 512/1K,
**+15.064%/+13.806%** over the preceding M256/RB32 packet. The complete cached
trace names 180 IQ3 rowbatch8 calls at local128/VGPR48/SGPR128/LDS512/scratch0
and eight IQ4 calls at local128/VGPR64/SGPR128/LDS512/scratch0 across both
shapes, with no compiler under the profiler. Versus the clean M256 trace,
selected IQ falls **27.160%/27.122%**, kernel sum **12.137%/11.446%**, and span
**12.269%/11.560%**.
Evidence: [`WPF-2 grouped-IQ production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-grouped-iq-matrix512-retained.json) ·
[`WPF-2 grouped-IQ correctness`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-grouped-iq-exact-correctness.json).

WPF-2b screened separately registered local64/pair16 IQ2 fused-SiLU
rowbatch4/8 primitives. One output-column/expert block preserves production's
pair16 K ownership, two-wave shuffle tree and serial wave sum, BF16 gate/up
projection boundaries, and fused SiLU rounding while retaining each decoded
gate/up pair across routed rows. Synthetic K3072 uneven tails through 31 rows
are BF16-bit exact to the production local64/pair16 tile2 body. Across all 46
actual M512 IQ2 gate/up layers, the inclusive route-major gate/up + post-SiLU
gather control totals **1343.915 ms**; rowbatch4/8 total **603.706/482.040 ms
(2.226x/2.788x)**, every layer wins, and every output is BF16-bit exact.
Cached primitive tracing names rowbatch8 at
local64/VGPR104/SGPR128/LDS512/scratch0 versus production tile2 VGPR136. The
losing rowbatch4 wrapper/key/instantiation is removed after publication;
rowbatch8 is the sole retained pair16 grouped primitive.

The registry-driven rowbatch8 owner is now the gfx1100 package default for
bulk rows; c=1 and unsupported gate/up quant keys retain route-major exact
fallbacks, and `grouped_exact` remains the preceding explicit rollback.
Complete state matches at KL 0 through all 48 hidden boundaries and full
K/V/`KVLiveSpans`. Clean 512/1K publication improves
**99.230/91.559 -> 118.705/107.804 tok/s (+19.626%/+17.743%)**. Cached tracing
cuts gate/up **62.549%/62.850%**, total selected IQ **45.021%/45.343%**, and
kernel span **16.309%/15.133%** with unchanged **1,479/2,962** dispatches.
Evidence:
[`WPF-2b primitive`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-pair16-grouped-gate-up-candidate.json) ·
[`WPF-2b production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-pair16-grouped-gate-up-production.json).

WPF-3 adds separately registered exact qrow4 SWA prefill and a C256-qualified
policy. One local32 wave keeps one query head, production's two-pass maximum,
denominator, value-FMA order, reconstructed 128-thread dot tree, and complete
`KVLiveSpans` visibility independently for four adjacent causal query rows,
while reusing each decoded BF16 K/V row. Wave32 remains the fallback unless
`rows == 128` and absolute start is at least 256. The 508..515 wrapped,
evicted, ragged-seven fixture and all four M128 positions are F32-bit exact.
The qualified four-slice sum improves **21.059 -> 9.389 ms (2.243x)**. Cached
tracing names qrow4 at local32/VGPR72/SGPR128/LDS0/scratch0. Complete M512
state matches all 48 hidden boundaries, logits, full K/V spans, repeat, and
lifecycle at KL0. A dirty same-weight 512/1K gate improves
**118.296/106.751 -> 131.852/124.817 tok/s (+11.459%/+16.923%)** during
candidate admission. The gfx1100 package now exports the qualified policy as
its default. A no-override M512 gate is KL0/bit-exact through all 48 boundaries
and full K/V spans; explicit local128 remains rollback. Clean selector-unset
512/1K improves **118.705/107.804 -> 131.919/125.960 tok/s
(+11.131%/+16.842%)**. Cached tracing cuts SWA **55.411%/59.449%**, all
attention **50.675%/52.612%**, and kernel span **9.643%/14.228%** at unchanged
**1,479/2,962** dispatches. Evidence:
[`WPF-3 exact qrow4 candidate`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-exact-candidate.json) ·
[`WPF-3 default promotion`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-default-promotion.json) ·
[`WPF-3 production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-exact-production.json).

The separately registered online qrow4 body remains a changed-association
arithmetic ceiling, not a gfx1100 production owner. Its complete
18-prompt/576-step M512 lane improves natural-prompt prefill **117.170 ->
118.335 tok/s (+0.995%)** and h16/h32 E2E **0.764%/0.609%**, but fails the
mandatory quality gate at maximum KL **0.394600** despite **564/576** top-1.
Poolside, same-mode determinism, and lifecycle pass. Keep the exact C256 policy
as gfx1100 default; do not delete online qrow2/qrow4 registrations because
gfx1151 owns them independently. Evidence:
[`WPF-3 online rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-swa-qrow4-online-rejected.json).

WPF-1T adds separately registered exact Q5_K/Q6_K output-column candidates in
`quant/gguf_k_gemv.{hip,py}`. `(2,16)` and `(4,8)` hold the RB32 product at 32
FP32 accumulators/thread while sharing each BF16 activation load/conversion
across two or four adjacent output columns. Every output independently retains
local128 K ownership, ordered FMAs, wave32 shuffles, and serial wave-0..3
accumulation. Synthetic BF16/F32 outputs are byte-exact through 17- and 33-row
tails. Across all 15 unique actual M512 `(quant, dtype, K, N)` configurations,
representing 381 production invocations, both candidates are exact and faster.
The weighted RB32/`(2,16)`/`(4,8)` sums are **2699.147/2220.526/1828.710 ms**;
`(4,8)` improves **1.476x (-32.249%)**, with individual wins **1.124-3.472x**.
It compiles at local128/VGPR72/SGPR50/LDS512/private0 with zero spills; RB32 is
VGPR73/SGPR107/LDS1024/private0 with 14 Q5 / 5 Q6 SGPR spills. Cached tracing
names all eight quant/dtype/geometry instantiations at plausible 13.52-23.64 us
on the tail fixture and spawns no compiler. gfx1100 exports a package-owned
`coltile` shape policy for divisible-by-four full RB32 slabs. Exactly four
`(quant, output, K, N)` keys select `(2,16)`, reducing the all-`(4,8)` weighted
family **1828.710 -> 1791.936 ms (-2.011%)**; every other eligible key keeps
`(4,8)`. The frozen seven-pair gate improves **+0.545%/+0.459%** at 512/1K. A
package-path repeat remains exact and positive at **+0.382%/+0.242%** but misses
the repeated 1K `>0.3%` magnitude threshold, so no broader shape owns
`(2,16)`. Explicit `rowbatch`, smaller slabs, unsupported widths, and gfx1151
remain exact fallbacks. The public Laguna variant constructor/setter is removed.
A no-override natural M512 gate is KL0/bit-exact through all 48 hidden boundaries
and full K/V/`KVLiveSpans`. Same-weight rotating 512/1K promotion improves
**131.491/124.949 -> 169.046/157.420 tok/s (+28.561%/+25.987%)**, with every
pair/repeat exact and lifecycle recovery. Clean selector-unset publication
reaches **169.253/159.229 tok/s (+28.301%/+26.412%)** versus the preceding exact
packet. Cached tracing names **1,524** coltile calls including warmup and cuts
dense/shared **38.546%/38.875%**, kernel sum **21.978%/20.935%**, and kernel
span **21.893%/20.852%** at 512/1K. The restored clean 4K gate reaches
**123.084 tok/s** with deterministic ID 7772, positions, lifecycle, and
allocation recovery. The 150-tok/s short gate passes, but 16K+ remains closed
below the 800/700 512/4K stretch gate. Evidence:
[`WPF-1T candidate`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile4-rowbatch8-candidate.json) ·
[`WPF-1T default promotion`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile4-rowbatch8-default-promotion.json) ·
[`WPF-1T production`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile4-rowbatch8-production.json) ·
[`WPF-1T role policy`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-coltile-role-policy.json).

The existing explicit P6 signed-byte IQ2 MMQ32 primitive is now rejected as a
Laguna runtime after actual M512 repricing. Over all 46 IQ2 gate/up layers its
quantizer-inclusive leaf sum improves exact **1297.436 -> 388.901 ms (3.336x)**
and every layer is faster. A temporary matrix512/attention128/RB32 session
reaches diagnostic **122.135/110.761 tok/s (+23.082%/+20.972%)** at 512/1K,
but the complete 18-prompt/576-step lane reaches maximum KL **0.683239** at
**565/576 (98.090%)** top-1. Sparse exact P6 repair is non-viable at maximum
**85.946%** BF16 mismatch coordinates and **99.496%** touched active expert
output rows, versus frozen **5%/20%** stop rules. No runtime owner, repair
queue, selector, or default was added; retain the P6 key only as explicit
primitive/ceiling evidence. That IQ2 result does not adjudicate WPF-1R; the
separate raw-Q5/Q6 screen below closes on its own actual tensors.
Evidence: [`P6 M512 rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-p6-iq2-mmq-matrix512-rejected.json).

WPF-H1 adds separately registered gfx1100-only, source-faithful Q5_K
primitives in `quant/gguf_k_mmq_prefill.{hip,py}` from llama.cpp
`c0bc8591e`. The producer expands hipEngine's BF16 boundary to F32, then emits
the same K-major 144-byte DS4 record: four FP16 `(scale, pre-quantization sum)`
pairs and 128 signed bytes. Its maximum actual M512/K9216 workspace is
**5,308,416 bytes**. The consumer uses local256 I128/J128/K256 ownership,
eight RDNA3 integer-WMMA waves, one raw-Q5 expansion per K256 slab, separate
dot/min terms, and a 57,856-byte dynamic-LDS launch class; no weight sidecar.
CPU byte oracles pass at 127/128/129-row producer tails, and BF16/F32 output
fixtures pass at M17/N72, M127/N128, and M129/N128. Final aligned resources are
producer local256/VGPR24/scratch0 and consumer local `(32,8)`/VGPR192/scratch0.
The eight-role/235-call M512 leaf improves **1,562.932 -> 97.110 ms (16.094x)**,
but the complete 18-prompt/576-step gate rejects runtime ownership at maximum
KL **4.162014** with **561/576** top-1 despite **1.348x** natural-prompt prefill.
The temporary constructor switch, activation scopes, workspace owner, package
capability, and dispatch policy are removed; gfx1151 remains excluded and the
qualified primitive stays explicit evidence only. Evidence:
[`leaf`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-k-source-mmq-candidate.json) ·
[`complete rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-k-source-mmq-rejected.json).

WPF-H3 adds separately registered gfx1100-only IQ3_XXS/IQ4_XS selected-down
consumers in `quant/gguf_iq_source_mmq_prefill.{hip,py}`, ported from
llama.cpp HIP `c0bc8591e` MMQ/load-tile/vector-dot/MMA ownership. The existing
strict producer emits caller-owned K-major 144-byte DS4 Q8_1 records; the new
raw-IQ consumer expands each expert-major K256 block directly, with no weight
sidecar. Compact row starts remain unpadded, while a host-built tile map pads
each expert independently to J128 without materializing padded activation rows.
The actual M512 workspace is **5,898,240 bytes**, reusable per selected-down
activation. Both quants pass empty/uneven/129-row tail fixtures. Immediate WMMA
fragment consumption removes IQ4's spill and lowers IQ3/IQ4 resources from
VGPR208 and VGPR256/scratch8 to **VGPR152** and **VGPR248/scratch0**; both use
local `(32,8)` and dynamic-LDS57,856. Across all **45 IQ3 + 2 IQ4** actual
M512 layers, exact grouped down moves **565.437 -> 115.951 ms (4.877x)** and
every layer wins. IQ3 alone reaches **111.016 ms**, **27.145% below** the
matched llama.cpp **152.380-ms** IQ3 trace. Maximum mean KL is **0.000756** and
minimum top-1 **97.578%**; preserve layer 47's disclosed **3.307 max row KL /
2048 max abs** outlier. Runtime natural-prompt prefill improves **152.276 ->
181.556 tok/s (1.192x)**, but complete quality rejects the route at maximum KL
**0.373028** with **567/576** top-1. An IQ3-source/IQ4-exact followup still
reaches maximum KL **0.372917**, isolating source IQ3 arithmetic. Runtime
ownership, tile128 metadata, capabilities, and focused integration tests are
removed; exact grouped down remains production and gfx1151 stays excluded.
Evidence:
[`leaf`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-candidate.json) ·
[`complete rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-rejected.json).

WPF-H4 adds separately registered gfx1100-only raw-Q6-to-F16 and composite
rocBLAS primitives in `quant/gguf_q6_k_f16_rocblas_prefill.{hip,py}`, ported
from llama.cpp HIP `c0bc8591e` `ggml-cuda.cu`/`convert.cu`/`dequantize.cuh`
ownership. One local64 launch expands every raw 210-byte K256 block to caller-
owned row-major F16 while independent blocks cast the resident BF16 activation
to F16. `rocblas_gemm_ex` then uses F16 A/B/C/D and F16 compute before the
registered output cast returns BF16 or F32. The unfused dequant plus ordinary
BF16-to-F16 cast remains a numerically equivalent primitive chain. No
persistent weight sidecar is created; one session can reuse three planes
bounded at **97,517,568 bytes** for M512.

The actual six-shape/**144-call** M512 inventory improves retained exact
coltile **174.351 -> 14.349 ms (12.151x, -91.770%)**. Every shape wins, all
outputs are finite, maximum mean KL is **3.441e-5**, and minimum top-1 is
**97.852%**. The candidate is **0.571 ms / 3.825% below** the matched llama.cpp
**14.919865-ms** source-stack recount, while remaining only a matched-family
attribution rather than a paired-process wall claim. Replacing convenience
`hgemm` with source-matching F16-compute `rocblas_gemm_ex` moves the stack
**15.119 -> 14.950 ms**; fusing the two independent producers into one launch
then reaches **14.349 ms**. A four-elements-per-thread cast followup regressed
the stack to **15.873 ms** and is removed. Cached tracing names the fused
producer at local64/VGPR16/SGPR128/LDS0/scratch0, all expected rocBLAS bodies,
and the BF16/F32 result casts. The temporary default-off owner passed natural
M512 at KL **0.000721933**, top-1 **100%**, deterministic complete state, and
exact teardown; integrated tracing observed exactly 144 producer/GEMM/cast
stacks. Complete changed-arithmetic quality nevertheless reaches maximum KL
**0.338657 > 0.05** at **567/576** top-1 despite **1.042x** diagnostic prefill.
The context-local owner, rocBLAS handle, 97,517,568-byte workspace, package
capabilities, and selector are removed. Exact coltile stays production,
gfx1151 remains excluded, and only the separately registered leaf/unfused chain
remain. Evidence:
[`leaf`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-candidate.json) ·
[`complete rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-rejected.json).

WPF-H5A adds separately registered gfx1100-only exact-value raw-Q5-to-F32 and
composite F32-SGEMM primitives in
`quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}`. One local64 launch expands each
176-byte Q5_K block with the established F32 scale/min expression while
independent blocks widen resident BF16 activations exactly to F32. rocBLAS
SGEMM consumes caller-owned row-major F32 input/weight planes and writes F32;
the BF16-output wrapper adds the registered F32-to-BF16 cast. The unfused raw
dequant plus ordinary BF16-to-F32 cast remains a numerically equivalent
primitive chain. No persistent weight sidecar exists, and one M512 session can
reuse three planes bounded at **195,035,136 bytes**. Edge-pattern dequant bytes
and BF16 widening are bit-exact to independent CPU values; BF16/F32 output
fixtures pass the KL/top-1 gate and allocation recovery. Cached tracing over
all eight real Q5 shape classes names 16 fused producers, 16 rocBLAS SGEMMs,
and eight result casts; the producer is local64/VGPR16/SGPR128/LDS0/scratch0.
Actual-role timing keeps the regressive F32 K3072/N48 gate on exact coltile and
selects the candidate for the other seven shapes. The resulting **235-call**
policy moves weighted events **1,256.936 -> 221.137 ms (5.684x, -82.407%)** and
synchronized wall **1,223.263 -> 231.966 ms (5.273x)**. All outputs are finite
at maximum mean/max-row KL **1.59e-9/5.79e-8** and top-1 **100%**. The stack
remains **3.751x** llama.cpp's matched Q5 trace. Its default-off owner allocates
one admitted 195,035,136-byte plane set and passes natural M512 at KL
**0.0003742**, top-1 **100%**, token **2930**, deterministic complete state, and
teardown. Complete quality nevertheless reaches max KL **1.143627** at
**564/576** top-1 despite **1.330x** diagnostic prefill. Remove the temporary
owner/workspace/capabilities/tests; exact coltile stays production and gfx1151
is excluded. Evidence: [`rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-rejected.json) ·
[`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-candidate.json).

WPF-H5C reuses that bit-exact producer but replaces rejected SGEMM association
with separately registered local128 F32-weight ordered consumers. H5C's **8x4**
and **4x8** geometries keep 32 accumulators/thread and exactly reproduce raw
coltile's per-thread K sequence, `fmaf` order, wave32 shuffle tree, serial
wave-0..3 sum, and BF16/F32 store. H5E retains that sequence with
**4x16/8x8/16x4** and 64 accumulators/thread; 1x64/2x32 are removed after every
role regresses. The raw-Q5 composite has the required unfused producer-plus-
`f32_weight` primitive chain; gfx1151 excludes all W7900 geometries. Rows17/33
tails and all eight actual M512 roles are byte-exact. The final-source 235-call
policy moves H5D **1,085.630 -> 951.876 ms (1.141x, -12.320%)** by events and
**1,040.166 -> 961.993 ms (1.081x, -7.515%)** by synchronized wall. A stronger
15-repeat N6144 adjudication confirms 16x4 on both clocks. Scratch remains one
projection-local **150,994,944-byte** plane with no persistent sidecar. Complete
package-default M512 state is KL0/byte-exact through all 48 boundaries, logits,
K/V, repeat, and teardown. Selector-unset production reaches
**184.997/172.104/131.496 tok/s** through 4K. Cached tracing observes exactly
235 local64/VGPR16 producers and 235 consumers; 32-accumulator bodies remain
local128/VGPR72/LDS512/scratch0 and new 64-accumulator bodies are
local128/VGPR136/SGPR128/LDS1024/scratch0. H5F retains constant-48 12x4 only
for F32 N48 at **1.187%/0.496%** event/wall. H5G retains exact
8x10/16x5/8x12/12x8 consumers on five roles; its strong gate cuts H5F
**892.586/896.357 -> 815.474/829.319 ms (-8.639%/-7.479%)** by event/wall.
Constant-80/96 bodies are local128/VGPR168/200/SGPR128/LDS1536/scratch0.
Complete state remains KL0 and clean 512/1K/4K reaches
**188.393/175.042/132.743 tok/s (+2.192%/+2.055%/+1.329%)** over H5F. H5H
closes the register boundary: constant-112 is VGPR232/LDS2048/scratch0 but loses
every role; constant-128 is VGPR256/LDS2048 with 28–52 B scratch and also loses
every role. All seven temporary bodies are removed. The public per-session
selector is removed; benchmark A/B uses scoped mutation. Evidence:
[`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-production.json) ·
[`candidate`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-candidate.json) ·
[`H5H boundary rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-register-boundary-rejected.json) ·
[`post-H5G residual`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5g-residual.json).
The retained request reclassification puts exact Q6 coltile at **177.047 ms /
146 calls** versus matched llama.cpp **14.920 ms**, selecting a raw-Q6 exact-F32
producer feeding this existing ordered `f32_weight` family as WPF-H5I. The
separately registered gfx1100 producer forms the signed integer `scale*quant`
product before the single F32 super-scale multiply, then Q6 composites reuse the
retained ordered consumers through the required producer-plus-primitive
fallback chain. An independent signed-Q6 oracle and raw-coltile rows17/33 gate
are BF16/F32 byte-exact across every geometry; cached tracing records the
producer at local64/VGPR16/SGPR128/LDS0/scratch0. The all-role screen and
15-repeat adjudication retain only `8x4`, `16x4`, and `16x5`; seven non-owning
Q6 composite surfaces are removed while Q5 production is unchanged. Across all
**146** actual calls, strong producer-inclusive event timing moves **194.758 ->
119.751 ms (1.626x, -38.513%)** and synchronized wall moves **189.722 ->
121.353 ms (1.563x, -36.037%)**. Four roles select a candidate; both long-K
roles and the wide-N F32 role retain exact raw coltile. Q5 and Q6 share the
existing **150,994,944-byte** serial plane/library with no new allocation.
Complete M512 state is KL0/byte-exact across all 48 boundaries, logits, K/V plus
live spans, repeat, and teardown. Cached tracing records exactly **143** Q6
producers, **143** ordered consumers, and **3** raw-coltile fallbacks: Q6 falls
**177.047 -> 110.170 ms (-37.774%)** and request kernel sum falls **2,667.034
-> 2,600.260 ms (-2.504%)**. Clean selector-unset 512/1K/4K reaches
**191.713/178.080/134.411 tok/s (+1.762%/+1.736%/+1.256%)** over H5G. The
quant-keyed package policy is promoted on gfx1100 without a quant branch or
public per-session selector; gfx1151 remains fail-closed
([H5I production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-production.json) ·
[H5I candidate](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-candidate.json)).

Post-H5I physical attribution reconciles **2,600.260 ms** exactly: Q5
**922.619 ms**, exact IQ3/IQ4 selected down **556.749 ms**, attention **471.150
ms**, gate/up **469.311 ms**, Q6 **110.170 ms**, and remaining **70.261 ms**.
The IQ family is 45 K1024/N3072 rowbatch8 calls at **530.864 ms** plus two
K1024/N3072 IQ4 calls at **25.886 ms**. A natural M512 routing capture counts
**230,400** IQ3 lanes, **9,844** active `(layer,expert)` instances, and
**33,547** rowbatch8 reconstruction batches. WPF-H5J therefore specializes the
existing exact body rather than changing arithmetic: retain one decoded IQ3
segment across the complete expert row range while replaying the unchanged
8-row accumulator/reduction/store phases, and right-size IQ4's sole populated
wave to local32. This differs from the removed rowbatch16
(VGPR256/40-byte scratch), rejected output tile 2 (~50% slower), and rejected
source-MMQ association. No sidecar/allocation or runtime default is admitted
before every **45+2** actual layer wins both event and synchronized-wall clocks
at exact bytes
([post-H5I residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5i-residual.json)).
The admitted gfx1100-only leaves preserve retained output bytes and the
independent NumPy dequant oracle exactly across empty, uneven, and rowbatch-tail
expert ranges. The binding actual-weight/routing gate catches a one-BF16-ULP
IQ4 mismatch in the separately compiled constant-K body; a generated seed-258
RED reproduces it. Remove that duplicate and let the K1024 wrapper launch the
retained exact dynamic body at local32. The final **45 IQ3 + 2 IQ4** gate is
byte-exact and both-clock positive on every layer: event sum moves **567.274 ->
500.176 ms (-11.828%)** and synchronized wall **567.056 -> 500.448 ms
(-11.746%)**. Cached W7900 tracing names IQ3 at
local128/VGPR40/SGPR128/LDS512/scratch0 and distinguishes the shared IQ4 body at
local128 versus local32/VGPR64/LDS512/scratch0. Complete M512 state is
KL0/byte-exact through all 48 boundaries, logits, K/V/live spans, repeat, and
teardown. Integrated tracing records all **45+2** package-selected calls,
reducing selected down **556.749 -> 497.145 ms (-10.706%)** and request kernel
sum **2,600.260 -> 2,532.020 ms (-2.624%)**. Clean selector-unset 512/1K/4K is
**196.103/181.859/137.169 tok/s (+2.290%/+2.122%/+2.052%)** over H5I. gfx1100
defaults only the bounded K1024/N3072 keys; every miss and gfx1151 retain the
preceding exact route, with no new allocation, workspace, or sidecar. H5K's
scratch-free rowbatch12/16 extensions lose both clocks on all **45/45** actual
IQ3 layers, regressing event/wall sums **+6.893%/+5.771%** and
**+10.770%/+9.870%**; every temporary body/wrapper/key/test is removed. The
unchanged request is again Q5-led at **919.697 ms**, of which ordered consumers
own **904.399 ms** and two roles **741.721 ms**. WPF-H5L admits an exact linear
weight-tile-major workgroup mapping for six material roles; F32 N48/N72 retain
H5G after the final-source both-clock gate. The 235-call producer-inclusive
family moves **882.963/887.364 -> 486.892/474.348 ms
(-44.857%/-46.544%)** by event/wall, with exact bytes, lifecycle recovery, and
unchanged local128/VGPR72-200/LDS512-1536/scratch0 classes. Complete M512 state
is KL0/byte-exact across all 48 boundaries, logits, K/V/live spans, repeat, and
teardown with no new allocation. Integrated tracing selects **235** producers,
**188** candidates, and **47** H5G fallbacks; Q5 moves **919.697 -> 466.986 ms
(-49.224%)** and request kernel sum **2,532.020 -> 2,074.261 ms (-18.079%)** at
unchanged **1,862** dispatches. Clean package-default 512/1K/4K promotes
**237.956/217.888/157.366 tok/s (+21.342%/+19.812%/+14.725% over H5J)**. The
matched M512 gap is **2.917x**; every miss and gfx1151 retain exact fallback.
Post-H5L attribution ranks matched attention/Q5/IQ-down gaps at **437.720/
408.035/338.619 ms**. H5M's separately registered source-qualified exact qrow4
body chooses required current/cache loads before the unchanged two-pass
arithmetic. It is F32-bit exact at dense starts 0/128/256/384 and 508..515
wrap/eviction/ragged cases; production starts 256/384 improve event/wall
**4.324%/4.354%** at local32/VGPR72/SGPR128/LDS0/scratch0. Complete M512 state
is KL0/byte-exact. The bounded package role selects exactly **72** H5M calls
alongside **48** global and **72** wave32 calls, cutting qrow4 **268.720 ->
260.500 ms (-3.059%)**, attention **459.445 -> 450.790 (-1.884%)**, and request
sum **2,074.261 -> 2,060.485 (-0.664%)** at unchanged **1,862** dispatches.
Clean package-default 512/1K/4K is **238.565/218.182/158.138 tok/s
(+0.256%/+0.135%/+0.490% over H5L)** with no new allocation or sidecar; matched
M512 narrows to **2.90983x**. Explicit routes, shape/registration misses, and
gfx1151 retain WPF-3. The production-identical post-H5M request reconciles
**2,060.485 ms / 1,862 dispatches** in a **2,086.586-ms** span. Matched gaps
rank attention/Q5/IQ-down at **429.065/406.709/336.162 ms**. Attention's 48
global, 72 SWA-wave32, and 72 source-qualified-qrow4 calls consume
**80.707/109.583/260.500 ms**; qrow4 remains **57.79%** of attention, with
**111.604/148.896 ms** at starts 256/384. H5N's separately registered exact
first-fill specialization derives identity-ring position/visibility without
`token_positions`, `evict_mask`, or `live_counts` reads while retaining cached
`base_offsets`, the full `KVLiveSpans` ABI, source rounding, dot tree, two-pass
arithmetic, attend-before-append schedule, and H5M fallback. It matches
H5M/wave32 bytes and improves start 256 **1.147x/1.144x event/wall**, start 384
**1.166x/1.163x**, and their sums **1.158x/1.156x**, with unchanged
local32/VGPR72/SGPR128/LDS0/scratch0 resources. Complete M512 state is KL0 and
integrated tracing selects all 72 calls, cutting qrow4/attention/request sum
**13.918%/8.087%/1.687%**. Runtime ownership is still rejected: clean 4K is
**-0.217%**, and a seven-repeat adjudication confirms **7/7** H5N samples below
H5M (**158.152 -> 157.832 tok/s, -0.202%**). Remove the temporary role policy,
retain only the leaf, and keep H5M production. The 508,944,384 source-level
generic predicates remain diagnostic, not a physical-load claim. WPF-H5O then
returns to Q5's retained **465.660-ms** family/**406.709-ms** matched gap through
a distinct exact representation in
`quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}`. Its 64-byte-aligned **320-byte**
block holds 256 unpacked quant bytes plus eight F32 scale and eight F32 minimum
coefficients. Every reconstructed F32 bit and rows17/33 H5L/H5G output byte
matches. Cached tracing is scratch-free: producer/expand VGPR16, consumers
VGPR80-200/LDS512-1536. Nevertheless **0/8** actual roles win both clocks;
producer-inclusive weighted event/wall regresses **477.022/473.054 ->
606.780/614.512 ms (+27.202%/+29.903%)**. The source-level 2.034x logical-byte
model fails because coefficient loads and reconstruction ALU dominate. Remove
every H5O symbol/key/test; H5L/H5G and gfx1151 remain unchanged. Do not retry
coefficient reconstruction without a distinct operation-count premise. WPF-H5P
cross-screens one unmeasured post-schedule geometry combination: H5F's exact
4x16/16x4/8x8 64-accumulator predecessors under H5L weight-major traversal.
Four of five roles lose at least one clock and are removed. Keep only BF16
K6144/N3072 `16x4`: it preserves the F32 producer and every K/FMA/wave/store
operation, is byte-exact at rows17/33 and on actual weights, and physically
reduces local128 **VGPR168/LDS1536 -> VGPR136/LDS1024** at scratch0. The
final-source producer-inclusive 12-call role falls **31.306 -> 29.329 ms
(-6.315%)** by HIP events and **30.890 -> 29.898 ms (-3.211%)** by synchronized
wall. Complete state is KL0/byte-exact and integrated tracing selects exactly
**12** calls, cutting role/Q5/request sum **5.800%/0.572%/0.187%**. The first
clean 512 packet is **-0.189%**, but a frozen seven-repeat adjudication is
**+0.176%**. Source-default 512/1K/4K is **+0.093%/-0.019%/-0.054%** and the
final frozen 1K/4K adjudication remains **-0.030%/+0.014%**. Reject runtime
ownership; remove the eager/package surfaces, retain the exact leaf, and keep
H5L production. WPF-H5Q consumes H5J's already-produced active-expert list
under a separately registered exact persistent traversal. P64/P128 alone win
all **45/45** actual IQ3 layers on both clocks; the predeclared robust rule keeps
P64 and removes seven instantiations. Final-source event/wall sums fall
**492.847/491.518 -> 481.081/483.823 ms (-2.387%/-1.565%)**, every output byte
matches H5J, sampled CPU-oracle values agree, and tracing records local128/
VGPR48/SGPR128/LDS512/scratch0/grid-y64. K1024 decode, rowbatch8 arithmetic/
reduction/store order, metadata, allocation, layout, and H5J fallback remain
unchanged. Complete state is KL0/byte-exact and integrated tracing selects all
**45** P64 IQ3 calls, cutting IQ-down/request sum **3.255%/0.491%**. The
scoped default-off clean gate improves 512/1K/4K
**+0.702%/+0.278%/+0.370%**, with 3/3 paired wins each. Selector-unset
publication confirms **+0.663%/+0.355%/+0.267%**, again 3/3 wins each, and
promotes **239.981/219.494/158.693 tok/s**. Only the gfx1100 K1024/N3072/E256
IQ3 variant+ABI entries change; H5J remains fallback, IQ4 is unchanged, and
gfx1151 remains fail-closed. The production-identical post-H5Q trace sums
**2,050.376 ms / 1,862 dispatches** against matched llama.cpp HIP **724.299
ms**; residual gaps rank attention **431.450 ms**, Q5 **409.559 ms**, IQ down
**320.157 ms**, and gate/up **59.253 ms**. WPF-H5R adds separately registered
exact cached-only two-pass global/SWA qrow4 candidates behind the existing safe
preappend schedule for complete M128 tiles. Both replay production bytes and
complete `KVLiveSpans`. Global's local256 dot/denominator/normalized-PV
reconstruction reaches local32/VGPR248/LDS8192/scratch0 and loses all starts at
**0.636–0.926x** on both clocks; remove its export/key/exclusion/test case. SWA
remains local32/VGPR64/LDS0/scratch0 and wins starts 0/128/256/384 on both
clocks. Including equal append cost, its actual 144-call event/wall sums fall
**337.277/334.031 -> 126.687/125.764 ms (-62.438%/-62.350%, 2.662x/2.656x)**.
Complete M512 state is KL0/byte-exact; corrected one-queue tracing records all
**144** write->H5R pairs at unchanged **1,862** dispatches and cuts the SWA
schedule/request sum **63.767%/9.690%**. Selector-unset one-queue 512/1K/4K is
**+11.340%/+4.848%/+0.746%**, with 3/3 paired wins each, exact state, and
unchanged ownership, promoting **267.205/230.441/160.221 tok/s**. The earlier
uncapped speed rows are superseded. Post-H5R one-queue attribution reconciles
**1,851.695 ms / 1,862 dispatches** and reranks exact Q5 first at **482.339 ms**
versus matched llama.cpp **58.951 ms**, a **423.388-ms** gap. WPF-H5S screens
separately registered persistent row-group partitions **1/2/4/8/16/32** for the
six H5L roles while preserving the producer/plane, geometry, arithmetic, launch,
workspace, and fallbacks. All bytes match; all 36 cached symbols have expected
grids, scratch0, unchanged LDS, and only +8 VGPR. No role wins both clocks.
Best aggregate P32 regresses producer-inclusive event/wall
**459.018/473.034 -> 565.864/566.290 ms (+23.277%/+19.714%)**; remove every
candidate surface and retain H5L/H5G. H5T maps H5Q's four logical K256
partitions into one wave while preserving P64/rowbatch8 and every arithmetic
boundary. Named register planes eliminate an initial scratch104 spill and
reach local32/VGPR96/LDS0/scratch0 with exact bytes. Actual event/wall moves
**474.107/485.298 -> 475.945/469.677 ms (+0.388%/-3.219%)**; only **12/45**
layers win both clocks. Remove the symbol/key/test and retain H5Q. H5U's
separate gfx1100 cached-source global leaf preserves the production local256
arithmetic and wins every standalone start, with weighted event/wall
**101.535/101.899 -> 84.124/84.622 ms (1.207x/1.204x)**. Default-off
M512/C4096 is KL0; tracing records **48 H5U + 144 H5R** pairs at unchanged
**1,862** dispatches and cuts global schedule **15.494%**. Matched direct M512
improves **0.849% with 5/5 wins**, but source-default ownership is rejected:
the final balanced role-ineligible 1K adjudication is **-0.00257% with 2/8
wins**. Remove all global runtime-policy plumbing and retain only the leaf.
H5V screens the largest retained family without reopening H5S persistence or
H5P geometry. Its separate local32 Q5 body preserves H5L bytes at rows17/33/M512
for all six roles and traces at SGPR128/scratch0 with unchanged LDS and only +8
VGPR. But sequential logical-K replay loses retained four-wave parallelism:
**0/6** roles wins both clocks, and producer-inclusive weighted event/wall
regresses **464.968/466.267 -> 492.423/493.754 ms (+5.905%/+5.895%)**. Remove
the body, exports, wrappers, keys, gfx1151 exclusions, and focused test; retain
H5L/H5G and require a new operation/data-reuse premise before revisiting this
schedule. H5W admits three exact Q6 weight-major composite wrappers/keys over
already-retained local128 16x5-BF16, 16x4-BF16, and 16x5-F32 symbols, matching
**142/143** H5I-selected calls with no HIP body/symbol, launch, allocation,
workspace, sidecar, or package-policy change. Rows17/33/M512 are byte-exact.
Cached tracing records producer->consumer order at VGPR136-168/LDS1024-1536/
scratch0 and expected grids. Every final-source role wins both clocks, moving
producer-inclusive weighted event/wall **87.859/81.559 -> 70.756/67.795 ms
(-19.466%/-16.876%)**. Default-off runtime is KL0/byte-exact across all 48
boundaries and complete state. Cached integration records exact **142 H5W + one
H5I + three raw** consumers at unchanged **1,862** request / **289** Q6
dispatches; Q6/request sum falls **121.306/1,851.695 -> 92.636/1,803.036 ms
(-23.635%/-2.628%)**. Default-off one-queue 512/1K/4K improves
**+1.830%/+1.492%/+1.061%** with 3/3 wins each. Selector-unset confirms
**+1.785%/+1.532%/+1.100%**, promoting **271.526/234.020/161.853 tok/s** and
narrowing matched M512 **2.59795x -> 2.55661x**. Preserve H5I F32-N72 and raw
long-K/wide-N fallbacks; H5W is package production. Post-H5W attribution
reconciles **1,803.036 ms / 1,862 dispatches** and returns to Q5, now
**476.433 ms** versus llama.cpp HIP **58.951 ms**. H5X admits a distinct exact
plane layout rather than persistence, geometry, compression, or K-ownership.
The linear local256/VGPR16/scratch0 producer stores full-F32 `[tile][k][col]`;
matching local128 consumers replay unchanged scalar FMAs/reductions with the
same VGPR/LDS/scratch0 while physical ISA moves **8/12/16 `global_load_b32` ->
2/3/4 `global_load_b128`** per K iteration. Rows17/33 plane bits and all six
actual M512 outputs are exact. Four roles / **151 calls** win both clocks; remove
BF16 K3072/N12288 and K9216/N3072 surfaces and retain H5L for their **37**
calls. Six-role selected event/wall falls **465.863/467.511 -> 458.615/459.712
ms (-1.556%/-1.668%)**; final-source winners fall **265.784/266.992 ->
258.653/258.959 (-2.683%/-3.009%)** with 4/4 wins. Default-off complete state
is KL0 and byte-exact. Four counter-rotated cached request segments record exact
**151 H5X + 37 H5L + 47 H5G** ownership at unchanged **1,862/470** request/Q5
dispatches and cut median Q5/request/span **1.911%/0.657%/0.886%**. Clean
512/1K/4K improves **+0.439%/+0.468%/+0.518%**, 3/3 wins each. Selector-unset
confirms **+0.531%/+0.310%/+0.327%**, again 3/3 each, promoting
**273.366/235.061/162.533 tok/s (+0.678%/+0.445%/+0.421% over H5W)** and
narrowing matched M512 to **2.53940x**. Retain four eager aliases and promoted
role entries while preserving two H5L roles, N48/N72 H5G, all Q6 routes, and
every miss; H5X is package production. The corrected external-comparator trace
uses C4096/direct M512 and measures **278.062 tok/s**, with **1,831.568 ms /
1,862 dispatches** versus llama.cpp HIP **724.299 ms**. Q5 remains the largest
**407.137-ms** gap and rowbatch8/10 own **346.501 ms**. Current ISA confirms one
scalar BF16 activation load per logical row even where H5X already vectorizes
weights. **WPF-H5Y** therefore freezes each role's H5X/H5L weight layout and
arithmetic but packs exact BF16 bits as aligned `[row_group][k][row_slot]`
records. Static load instances model **4.521B -> 0.920B (-79.65%)**. The
standalone gate admits all six roles: rows17/33/512 plane/output bytes are exact,
physical b64/b128 plus bounded tail loads preserve each weight-load class and
consumer resources at scratch0, and the **188-call** pack-inclusive event/wall
aggregate falls **462.608/455.971 -> 263.014/274.237 ms
(-43.145%/-39.856%)**. The bounded **161,120,256-byte** default-off owner
passes complete M512 at KL0/byte-exact. Paired tracing records exact
**188 packs + 235 weight producers + 188 H5Y + 47 H5G**, cutting
Q5/request/span **47.204%/9.685%/9.770%**. Default-off 512/1K/4K improves
**10.939%/9.051%/5.920%**, 3/3 wins each. Selector-unset confirms
**10.862%/8.969%/5.829%**, again 3/3 each, promoting canonical
**303.140/256.139/171.830 tok/s (+10.892%/+8.967%/+5.720% over H5X)**.
Matched C4096/direct-M512 reaches **306.305 tok/s / 1,658.386-ms** kernel sum
and narrows llama.cpp HIP to **2.26632x**. Gaps rank IQ-down/attention/Q5 at
**339.558/239.624/188.153 ms**; exact IQ3 alone is **486.381 ms / 45 calls**.
WPF-H5Z admits an orthogonal exact activation-resident output-column P256 leaf.
It keeps H5Q P64, local128/four-wave K, rowbatch8 arithmetic, and all metadata/
allocation/fallback semantics while retaining one K8 activation tile across
sequential outputs. All five screened leaves are byte-exact; only P256/P512 win
all **45/45** actual IQ3 layers on both clocks, and the frozen max-min rule keeps
P256. Selection event/wall falls **481.013/487.809 -> 454.128/455.001 ms
(-5.589%/-6.725%)**; final source confirms **478.606/486.167 ->
459.818/451.737 ms (-3.926%/-7.082%)**. P256 traces at local128/VGPR112/
SGPR128/LDS512/scratch0. Device ISA has exactly eight b128 activation records
before the sequential output loop, then the unchanged two d16 and three b32 IQ3
records. Remove P32/P64/P128/P512. Its bounded default-off owner reuses H5Q's
active-expert ABI with no allocation/workspace change. Natural M512 is KL0 and
byte-exact across all state/repeat. Four cached H5Q/H5Z/H5Z/H5Q requests retain
**2,050** dispatches and exact **45 IQ3 + two IQ4** topology, moving IQ3/request/
span **488.610/1,625.126/1,650.283 -> 477.168/1,603.812/1,624.882 ms
(-2.342%/-1.312%/-1.539%)**. Default-off 512/1K/4K improves
**+1.801%/+1.334%/+0.900%**, 3/3 wins each. Selector-unset confirms
**+1.819%/+1.452%/+0.872%**, again 3/3, promoting H5Y/H5Z at canonical
**307.658/259.947/173.562 tok/s (+1.490%/+1.486%/+1.008% over H5Y/H5Q)**.
Matched H5Z C4096/direct-M512 reaches **311.622 tok/s / 1,628.336-ms** kernel
sum and narrows llama.cpp HIP to **2.22765x**. Gaps rank IQ-down/attention/Q5 at
**325.570/235.310/182.882 ms**; IQ3 is **472.416 ms / 45 calls**. Immediate
exact IQ ownership/geometries are already screened. WPF-H6A exact dense-initial
cached-only metadata elision now qualifies a bounded default-off owner for its
H5R-derived SWA and H5U-derived global leaves. Natural M512 is KL0 and byte-exact
across all 48 boundaries, complete logits/KV/`KVLiveSpans`, and repeat at
unchanged **161,120,256-byte** workspace. Four paired cached requests preserve
**2,050** dispatches and exact **48 H6A global + 144 H6A SWA** write-before-
attention topology, moving attention schedule/request-sum/span
**254.976/1,627.696/1,653.806 -> 170.086/1,560.817/1,581.621 ms
(-33.294%/-4.109%/-4.365%)**. Resources remain global local256/VGPR40/scratch0
and SWA local32/VGPR64/scratch0; no compiler runs under profiling. Default-off
512/1K/4K improves **307.071/259.710/173.388 -> 312.331/261.467/173.954 tok/s
(+1.713%/+0.677%/+0.326%)**, 3/3 wins each. Selector-unset confirms
**307.158/260.161/173.375 -> 312.781/261.591/173.997 tok/s
(+1.831%/+0.550%/+0.359%)**, again 3/3, promoting H6A at canonical
**312.781/261.591/173.997 tok/s (+1.665%/+0.633%/+0.251% over H5R/H5Y/H5Z)**.
The binding post-H6A C4096/direct-M512 row is **326.174 tok/s**, **+92.414%**
over campaign start and **+4.670%** over H5Z. The old llama.cpp row bound only
the launcher and is superseded as synthetic evidence. A clean patched c0bc8591
rebuild with the implementation hash and **5/5 top-1 2930** markers measures
exact natural/C4096/BF16 llama.cpp HIP at **696.342 tok/s**; synthetic pp512 is
**711.410 tok/s**, consistent with the user's **714.07**. The exact wall gap is
**2.13488x**. Current/llama kernel sums are **1,568.190/718.241 ms**; residuals
rank IQ-down/Q5/attention/gate-up/Q6 at
**336.609/187.223/147.249/93.203/79.112 ms**.

WPF-H6B screens the new exact IQ3 data-layout operation selected after H6A.
Complete aligned 16-byte `{float scale, int8 magnitude[8], padding}` records and
all **45/45** actual-layer outputs match H5Z/CPU bytes. Physical tracing is
scratch-free at producer local256/VGPR24/LDS0 and consumer local128/VGPR104/
LDS512, but LLVM elides dead padding into one b96 rather than the frozen b128
record load. More importantly, producer-inclusive H5Z -> H6B event/wall
regresses **462.301/450.204 -> 575.804/587.342 ms (+24.552%/+30.461%)**, with
**0/45** both-clock wins. Remove the producer/consumer symbols, wrappers,
registry keys, exclusions, and RED test; retain H5Z and do not retry this
representation without a materially different operation-count/data-movement
premise
([H6B rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-signed-magnitude-segment-plane-rejected.json) ·
[post-H6A matched residual / H6B target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6a-matched-residual.json) ·
[H6A production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-dense-initial-cached-exact-attention-production.json) ·
[H6A candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-dense-initial-cached-exact-attention-candidate.json) ·
[post-H5Z matched residual / H6A target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5z-matched-residual.json) ·
[H5Z production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-production.json) ·
[H5Z candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-candidate.json) ·
[post-H5Y residual / H5Z target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5y-matched-residual.json) ·
[H5Y production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-activation-tile-k-row-production.json) ·
[H5Y candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-activation-tile-k-row-candidate.json) ·
[post-H5X matched residual / H5Y target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h5x-matched-residual.json) ·
[H5X production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-tile-k-col-production.json) ·
[H5X candidate](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-tile-k-col-candidate.json) ·
[post-H5W residual / H5X target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5w-residual.json) ·
[H5W production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-production.json) ·
[H5W candidate](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-candidate.json) ·
[H5W target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-weight-major-target.json) ·
[H5V rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-one-wave-k-partitions-rejected.json) ·
[H5V target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-one-wave-k-partitions-target.json) ·
[H5U runtime rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-runtime-rejected.json) ·
[H5U leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-candidate.json) ·
[H5U target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-target.json) ·
[H5T rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-one-wave-k-partitions-rejected.json) ·
[H5T target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-one-wave-k-partitions-target.json) ·
[H5S rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-persistent-row-group-rejected.json) ·
[post-H5R residual / H5S target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5r-residual.json) ·
[H5R production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-production.json) ·
[H5R SWA leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-candidate.json) ·
[post-H5Q residual / H5R target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5q-residual.json) ·
[H5Q production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-active-expert-persistent-production.json) ·
[H5Q leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-active-expert-persistent-candidate.json) ·
[H5Q target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-active-expert-persistent-target.json) ·
[H5P rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-weight-major-occupancy-runtime-rejected.json) ·
[H5P leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-weight-major-occupancy-retune-candidate.json) ·
[H5P target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-weight-major-occupancy-retune-target.json) ·
[H5O rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-factorized-exact-plane-rejected.json) ·
[H5O target](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-factorized-exact-plane-target.json) ·
[H5N runtime rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-dense-first-fill-runtime-rejected.json) ·
[H5N leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-dense-first-fill-exact-candidate.json) ·
[post-H5M residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5m-residual.json) ·
[H5M production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-sourcequal-exact-production.json) ·
[H5M leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-sourcequal-exact-candidate.json) ·
[post-H5L residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5l-residual.json) ·
[H5L production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-production.json) ·
[H5L candidate](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-candidate.json) ·
[post-H5K residual](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5k-residual.json) ·
[H5K rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-larger-resident-rowbatch-rejected.json) ·
[H5J production](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-production.json) ·
[H5J leaf](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-candidate.json)).

Post-H6B Q5 decomposition is **214.346-ms H5Y consumers (87.2%)**, **25.385-ms
producers (10.3%)**, **4.567-ms activation packs (1.9%)**, and **1.552-ms
fallback (0.6%)**. The dominant BF16 K9216/N3072 and F32 K3072/N9216 leaves
already emit **188/156** VOPD sites with aligned loads and scratch0, so H6C does
not repeat Q5 producer, broad geometry, ownership, VOPD, or plane screens.
**WPF-H6C exact special-IQ3 expert-major fused-SiLU rowbatch4** now admits
one separately registered K3072/N1024/E256 instantiation of the existing
expert-major template. One block owns an expert/output column, reuses exact raw
gate/up segments across four compact rows, and preserves RT1's scalar FMAs,
wave32 trees, serial wave-0..7 sums, gate/up BF16 boundaries, SiLU expression,
and BF16 output. On actual layer-47 weights and natural M512 routing, complete
bytes match the route-major control and fair control-post-gather versus
candidate-pre-gather event/wall moves **32.691/32.724 -> 15.458/15.438 ms
(-52.716%/-52.825%, 2.115x/2.120x)**. Cached trace is
local256/VGPR72/LDS512/scratch0 at physical grid 1024x256; code-object metadata
is VGPR71/SGPR58/fixed-LDS256/private0/spill0. A bounded role map resolves only
`(layer47, gguf_iq3_xxs)` at exact model shape and reuses existing
`expert_down`/`expert_gate` scratch. Complete natural M512 is KL0/byte-exact
across all state and repeat at unchanged **600,141,856-byte** total scratch.
Four cached requests preserve **2,050** dispatches and exact 46-IQ2/one-H6C
plus 45-H5Z/two-H5J topology; gather-inclusive special time falls **32.127 ->
15.030 ms (-53.215%)**. Default-off 512/1K/4K improves
**+1.148%/+0.796%/+0.560%**; selector-unset publication confirms
**+1.326%/+0.897%/+0.490%**, 3/3 wins each, promoting
**316.106/263.864/174.840 tok/s**. Fixed natural-M512/C4096 improves
**325.211 -> 328.863 tok/s (+1.123%, 5/5 wins)** and narrows exact llama.cpp HIP
**696.342** to **2.11742x**
([H6C production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-production.json) ·
[H6C runtime candidate](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-runtime-candidate.json) ·
[H6C leaf](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-candidate.json) ·
[H6C target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-target.json)).

The clean post-H6C source-default refresh reaches **329.563 tok/s**, **+94.413%**
over campaign start and **2.11293x** behind exact llama.cpp HIP **696.342**.
Its representative cached request is **1,546.351 ms / 2,050 dispatches** in a
**1,567.000-ms** span. Fresh gaps rank IQ-down/Q5/attention/Q6/gate-up at
**334.482/186.766/146.896/79.035/74.403 ms**; H5Z is **480.299 ms / 45 calls**.
**WPF-H6D exact row-interleaved IQ3 VOPD** is the retained gfx1100 IQ3 source
default through the existing raw active-expert ABI. The helper interleaves low
`sum0..sum7`, then high `sum0..sum7` inside each unchanged j0..3 step; H5Z/H5Q
remain registered rollback. Cached metadata/ISA keeps **72 FMAs, 13 global
loads, 52 DS operations, two barriers, SGPR58, private0, and spill0**, forms
**17** FMA/FMA pairs, and cuts issue slots **72 -> 55**, function slots **859 ->
775**, metadata VGPR **107 -> 99**, and runtime VGPR **112 -> 104** at
LDS512/scratch0. Complete natural-M512 state is KL0/byte-exact across all **48**
boundaries, logits, K/V/`KVLiveSpans`, and repeat at unchanged
**161,120,256-byte** workspace / **600,141,856-byte** scratch. Four cached
requests retain exact **45 H5Z or 45 H6D + two H5J** topology at **2,050**
dispatches and cut IQ3/request/span **2.564%/0.251%/0.861%**. Selector-unset
512/1K/4K is **+1.207%/+0.657%/+0.492%**, 3/3 wins each; fixed C4096/M512 is
**329.327 -> 332.308 tok/s (+0.905%, 5/5 wins)** and **2.09547x** behind exact
llama.cpp HIP. **92/92** guards pass. The clean promoted-source reprofile follows
([H6D production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-production.json) ·
[H6D candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-candidate.json) ·
[post-H6C residual / H6D target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6c-matched-residual.json)).

The clean post-H6D source-default refresh reaches **332.992 tok/s**, **+96.436%**
over campaign start and **2.09117x** behind exact llama.cpp HIP **696.342**.
Its representative cached request is **1,530.211 ms / 2,050 dispatches** in a
**1,551.216-ms** median span. Fresh gaps rank IQ-down/Q5/attention/Q6/gate-up
at **320.874/186.614/146.882/78.701/72.663 ms**. The trace preserves exact
**45 H6D + two H5J**, **46 IQ2 + one H6C**, and **48 global + 144 SWA H6A**
topology with H6D local128/VGPR104/LDS512/scratch0 and no compiler activity.

**WPF-H6E exact Q6 activation-tile-K-row transfer** is admitted as three
separately registered gfx1100 H5W siblings. The coltile16 rowbatch5 BF16
K3072/N1024, rowbatch4 BF16 K1024/N3072, and rowbatch5 F32 K3072/N1024 leaves
reuse H5Y's retained pack and Q6's exact row-major F32 producer while preserving
local128/four-wave K ownership, scalar `fmaf` order, wave32 tree, serial wave
sum, stores, workgroup order/count, maps, allocation, and workspace. All
rows17/33/M512 output/plane bytes and sampled CPU values pass; the broader
retained bundle is **35/35**. On actual weights, producer-inclusive weighted
event/wall moves **65.969/66.187 -> 58.085/58.217 ms (-11.952%/-12.042%,
1.136x/1.137x)** with 3/3 both-clock wins. Disassembly records 16 unchanged b32
weight loads plus one b64 activation record and the required rowbatch5 u16 tail;
trace resources exactly match H5W at **VGPR136/168, LDS1024/1536, scratch0** and
matching grids. No compiler runs under profiling. Complete natural-M512 is
KL0/byte-identical across all 48 boundaries, logits, K/V/`KVLiveSpans`,
repeat, and teardown at unchanged **161,120,256-byte** workspace /
**600,141,856-byte** scratch. Four cached requests preserve all other families while H5W controls record 2,050 dispatches and H6E candidates
record 2,192 from the exact 142 added packs. Q6/request-sum/span moves **92.867/
1,545.837/1,572.498 -> 84.000/1,541.912/1,563.696 ms (-9.549%/-0.254%/
-0.560%)**. Selector-unset H5W rollback -> H6E source 512/1K/4K improves
**318.215/266.225/176.015 -> 319.854/267.357/176.470 tok/s
(+0.515%/+0.425%/+0.259%)**, 3/3 exact wins. Fixed C4096/direct-M512 improves
**332.443 -> 333.329 tok/s (+0.266%, 5/5 wins)** and is **2.08905x** behind
llama.cpp HIP. H6E is Q6 source default; H5W/H5I stay registered rollback and
promotion adds no body, allocation, workspace, sidecar, or selector
([H6E production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-k-activation-tile-k-row-production.json) ·
[H6E candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-k-activation-tile-k-row-candidate.json) ·
[post-H6D target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6d-matched-residual.json)).

The clean H6E source refresh reaches **334.512 tok/s**, **+97.333%** over
campaign start and **2.08166x** behind llama.cpp HIP; IQ down remains the
largest **320.074-ms** family gap. **WPF-H6F exact IQ3 paired-output reduction
amortization** is now the retained gfx1100 IQ3 source default on H6D's prior
**465.480 ms / 45 calls**. It keeps P256/P64/local128, rowbatch8, all per-output
segment decode/FMA/wave0..3/store operations, loads, addresses, and active
traversal while carrying two strided outputs through one reduction epoch.
Compiled output stride is **0x200** versus H6D **0x100** with two barrier
instructions in each loop body, physically proving **24 -> 12 dynamic barriers
per rowbatch (-50%)**. Candidate metadata/runtime is private0/spill0/scratch0 at
**VGPR146/152, LDS256/512**, and local/grid remains **128 / 32768x64**. Frozen
rows1/7/8/9/M512, P64/P65, pair boundaries, and CPU bytes pass **9/9**. All
**45/45** actual-layer outputs are exact and win both clocks: event/wall moves
**445.316/436.801 -> 352.255/360.918 ms (-20.898%/-17.372%,
1.264x/1.210x)**.

The source map adds no ABI, allocation, or launch path; H6D/H5Z/H5Q remain
registered active-expert rollbacks. Complete natural-M512 is KL0 and byte-exact
across all **48/48** hidden boundaries, logits, K/V/`KVLiveSpans`, repeat, and
lifecycle at unchanged **161,120,256-byte** workspace / **600,141,856-byte**
scratch. Four cached requests retain **2,192 dispatches** and substitute exact
**45 H6D -> 45 H6F**, cutting IQ3/request-sum/span
**464.484/1,540.306/1,567.420 -> 366.610/1,458.072/1,479.670 ms
(-21.072%/-5.339%/-5.598%)**. Selector-unset 512/1K/4K improves
**320.079/267.093/176.521 -> 336.830/278.753/181.563 tok/s
(+5.234%/+4.365%/+2.856%)**, 3/3 exact wins each; **156/156** retained guards
pass. Fixed C4096/direct-M512 improves **333.248 -> 352.761 tok/s (+5.856%,
5/5 wins)** and is **1.97397x** behind llama.cpp HIP. The clean promoted-source
refresh reaches **353.798 tok/s**, **+108.710%** over campaign start and
**1.96819x** behind llama.cpp HIP. Its representative cached request is
**1,435.431 ms / 2,192 dispatches** in a **1,460.237-ms** median span. Fresh gaps
rank IQ-down/Q5/attention/gate-up/Q6 at **221.737/191.928/149.544/75.429/
71.249 ms**. Exact topology retains **45 H6F + two H5J**, **46 IQ2 + one H6C**,
**48 global + 144 SWA H6A**, and unchanged H5Y/H6E routes with no H6D escape.

**WPF-H6G exact Q5 one-step K-record prefetch is rejected with every candidate
surface removed.** The frozen rows17/33/M512 cross-product preserves complete
H5Y outputs, activation/weight planes, sampled CPU values, policy, workspace,
and gfx1151 exclusion. Candidate/control metadata remains private0 at identical
VGPR **194/162**, but code-object ISA places `s_waitcnt vmcnt(0)` after each
**13/4-instruction** next-record load group with **zero current FMAs in between**.
On actual BF16 K9216/N3072 row-major 12x8 and F32 K3072/N9216 tile-K-col 8x10
weights, direct weighted event/wall regresses **194.591/194.547 ->
203.237/204.091 ms (+4.443%/+4.906%)** and producer/pack-inclusive regresses
**217.265/217.342 -> 225.464/226.243 (+3.774%/+4.095%)**. Both roles lose both
clocks. Keep H5Y and do not retry compiler-scheduled K-record prefetch without a
new physical mechanism
([H6G rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-record-prefetch-rejected.json) ·
[post-H6F residual / H6G target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6f-matched-residual.json) ·
[H6F production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-paired-output-reduction-production.json)).

The post-H6G Q6 audit rejects persistent exact-F32 residency before code, then
rejects **WPF-H6H bounded source-F16 fallbacks** at complete quality. The
three-shape owner borrowed **97,517,568 bytes** inside H5Y/H6E's existing
**161,120,256-byte** serial scratch with no allocation and reused the retained
H4 producer/rocBLAS/cast body unchanged. Natural M512 passes at KL
**0.000685**, top-1 **100%**, token **2930**, deterministic repeat, all **48/48**
hidden boundaries changed, and clean lifecycle. The quality-only 18-prompt/
**576-step** gate then reaches max KL **0.411789 > 0.05** at **565/576
(98.09%)** top-1; all steps exercise changed arithmetic and Poolside separately
passes at KL **0.000157**. Per the frozen stop rule, run no topology or wall
timing and remove the policy, context ABI, conditional eager library/rocBLAS
handle, and RED test. Preserve the separately registered H4 leaf and all exact
H6E/H5I/raw production
([H6H rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-f16-raw-fallback-rejected.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q6-f16-raw-fallback-target.json)).

**WPF-H6I exact IQ3 triple-output reduction amortization** is now the retained
gfx1100 IQ3 source default through the existing `grouped_raw_iq_active_experts`
ABI/raw allocation/`grouped_iq_prefill` library. The leaf remains stride
**0x300**, 216 FMAs, private0/spill0/scratch0 at metadata/runtime
**VGPR164/168, LDS384/512**, and exact/both-clock positive on **45/45** actual
layers. Complete natural M512 is KL0 and byte-exact across logits, final/post
hidden, all **48/48** boundaries, K/V/`KVLiveSpans`, repeat, and teardown at
unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch. Four
cached requests preserve **2,192 dispatches** and every non-IQ3 family while
substituting exact **45 H6F -> 45 H6I**; IQ3/request-sum/span falls
**9.559%/1.906%/2.200%** with H6I local128/VGPR168/LDS512/scratch0. Selector-
unset H6F rollback -> H6I source at 512/1K/4K gains
**2.304%/1.650%/0.719%**, 3/3 exact wins each; fixed C4096/M512 gains
**2.036% with 5/5 wins**, reaches **360.154 tok/s**, and is **1.93346x** behind
llama.cpp HIP **696.342**. **192/192** guards pass. Keep H6F/H6D/H5Z/H5Q as
registered rollback and reprofile clean production before selecting another
kernel target
([H6I production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-production.json) ·
[H6I candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-candidate.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-target.json) ·
[post-H6I residual / H6J target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6i-matched-residual.json)).

The clean H6I source refresh reaches **359.963 tok/s**, **+112.347%** over
campaign start and **1.93448x** behind exact llama.cpp HIP **696.342**. Its
representative cached request is **1,409.540 ms / 2,192 dispatches** in a
**1,433.072-ms** median span. Fresh gaps rank Q5/IQ-down/attention/gate-up/Q6
at **194.868/187.775/150.723/78.345/71.906 ms**. Exact topology retains
**45 H6I + two H5J**, **46 IQ2 + one H6C**, **48 global + 144 SWA H6A**, and
unchanged H5Y/H6E routes with no H6F escape.

**WPF-H6J exact dense-initial SWA qrow4 unscaled-dot replay is rejected.** The
local32 leaf matches complete H6A bytes and sampled CPU rows at starts
0/128/256/384, leaves all five `KVLiveSpans` fields immutable, and recovers
allocations. Cached ISA proves four second-pass K-load and 20 wave-reduction
sites removed, four LDS stores plus four loads, unchanged exp/PV/store sites,
and metadata VGPR54/LDS8192/private0/spill0. rocprof records local32,
grid2304x32, LDS8192, scratch0, but runtime **VGPR248** versus H6A VGPR64. Every
start loses both clocks: weighted H6A -> H6J moves **95.924 -> 133.542 ms event
(0.718x)** and **97.607 -> 139.600 ms wall (0.699x)**. Skip runtime ownership,
remove every HIP/Python/key/exclusion/test surface, retain H6A, and do not retry
full 4x512 LDS score replay without a materially different occupancy-preserving
mechanism
([rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-swa-dot-replay-rejected.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6i-matched-residual.json)).

**WPF-H6K exact IQ3 quadruple-output reduction amortization is rejected.** The
frozen **9/9** matrix and all **45/45** actual-layer outputs are byte-exact to
H6I. ISA realizes stride **0x400**, **288** useful FMAs, and fixed-N3072 **4 ->
3 epochs / 8 -> 6 dynamic barriers (-25%)**. Metadata/runtime is private0/
spill0/scratch0 at VGPR **193/200**, LDS **512/512**, local128, and unchanged
grid32768x64; the isolated M512 rocprof smoke improves **650.724 -> 629.001
us**. That physical saving does not survive the binding all-layer distribution:
**0/45** wins both clocks, aggregate event regresses **329.061 -> 339.509 ms
(+3.175%, 0.969x)**, and synchronized wall regresses **332.027 -> 337.538 ms
(+1.660%, 0.984x)**. Remove every candidate and RED surface, skip runtime
ownership, retain H6I, and do not retry wider output grouping without a
materially occupancy-preserving mechanism
([rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-quadruple-output-reduction-rejected.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-quadruple-output-reduction-target.json)).

**WPF-H6L exact IQ2 pair16 grouped gate/up rowbatch16 decode amortization** is
now the retained gfx1100 IQ2 source default. It instantiates the frozen WPF-2b
template only at K3072/N1024/E256 `<16>` and preserves local64/pair16 K
ownership, one output/expert block, every row's FMA/two-wave sum/BF16 gate-up
boundary/SiLU/store, activation/useful work, grid, allocation, and workspace.
The boundary/CPU matrix passes **10/10** and all **46/46** actual layers are
byte-exact/both-clock positive. Code-object metadata is VGPR112/SGPR86/LDS256/
private0/spill0/wave32; cached rocprof records local64/grid65536x256, runtime
VGPR112/LDS512/scratch0, and no compiler activity.

Natural-M512 control/candidate/repeat is KL0 and byte-exact across logits, all
**48/48** hidden boundaries, complete K/V/`KVLiveSpans`, and teardown at
unchanged **161,120,256-byte** workspace / **600,141,856-byte** scratch. Four
cached requests preserve **2,192 dispatches** and replace exactly **46
rowbatch8 -> 46 H6L** with unchanged H6C/H6I/Q5/Q6/attention topology. IQ2/
request-sum/span moves **460.772/1,424.447/1,452.975 ->
377.540/1,351.047/1,372.593 ms (-18.064%/-5.153%/-5.532%)**. Selector-unset
rowbatch8 rollback -> H6L source at 512/1K/4K improves
**343.370/282.905/182.706 -> 362.826/295.544/188.636 tok/s
(+5.666%/+4.468%/+3.246%)**, 3/3 exact wins each. Fixed natural C4096/M512
improves **360.451 -> 381.893 tok/s (+5.949%, 5/5 wins)** and is **1.82340x**
behind llama.cpp HIP **696.342**; **212/212** guards pass. Keep rowbatch8 as the
same-ABI registered rollback and reprofile clean production before selecting
the next matched residual target
([production](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-production.json) ·
[candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-candidate.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-target.json)).

The clean H6L source refresh reaches **381.977 tok/s**, **+125.334%** over
campaign start, **+6.116%** over clean H6I, and **1.82299x** behind matched
llama.cpp HIP **696.342**. Its representative cached request is
**1,326.062 ms / 2,192 dispatches**. Q5/IQ-down/attention/Q6 gaps are
**194.004/189.827/151.442/72.392 ms**; H6L moves gate/up to **393.895 ms**, or
**7.498 ms faster** than llama.cpp's **401.393 ms**. Exact topology is **46 H6L
+ one H6C**, **45 H6I + two H5J**, **48 global + 144 SWA H6A**, and unchanged
H5Y/H6E. No compiler runs under cached profiling.

**WPF-H6M exact explicit wait-split Q5 K-record pipelining is rejected with
every candidate surface removed.** Frozen rows17/33/M512 and both actual roles
preserve complete H5Y output/activation/weight-plane bytes, sampled CPU quality,
policy, workspace, and gfx1151 exclusion. Cached ISA realizes the new physical
premise—not H6G's failed scheduling—with exact **13/4 next-record global loads,
32 useful current-record `v_fmac_f32` sites, no intermediate wait or loaded-value
use, then one `s_waitcnt vmcnt(0)`**. Metadata/runtime remains private0/scratch0
at VGPR **194/200** BF16 and **162/168** F32, local128, LDS1536, and exact M512
grids.

That physical overlap is slower. On actual BF16 K9216/N3072 row-major 12x8 and
F32 K3072/N9216 tile-K-col 8x10 weights, direct weighted event/wall moves
**194.618/195.249 -> 205.367/205.331 ms (+5.523%/+5.164%)** and the 70-call
producer/pack-inclusive aggregate moves **215.590/216.860 -> 227.873/227.347
(+5.697%/+4.836%)**; both roles lose both clocks. Skip runtime ownership,
remove HIP/Python/key/exclusion/RED surfaces, retain H5Y/H6L production
**381.977 tok/s**, and close Q5 geometry/plane/ownership/compiler-managed plus
explicit wait-split premises before returning to a distinct residual family
([H6M rejection](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-q5-k-record-wait-split-rejected.json) ·
[post-H6L residual / H6M target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-post-h6l-matched-residual.json)).

**WPF-H6N exact global dense-initial fixed-512 score arena is the retained
gfx1100 source default.** The generic role parser admits both H6A/H6N keys
without backend/quant branches, ABI, allocation, workspace, or public-selector
changes. Complete natural M512 is KL0/byte-exact across logits, all **48/48**
hidden boundaries, K/V/spans, repeat, and teardown. Four cached requests
preserve **2,192 dispatches** and replace exactly **48 H6A global with 48 H6N**,
retaining 144 H6A SWA and all other normalized kernels. Global/attention/
kernel-sum/span move **57.126/169.556/1,320.178/1,346.667 -> 31.969/148.140/
1,305.325/1,327.300 ms (-44.038%/-12.631%/-1.125%/-1.438%)** at local256/
VGPR40/scratch0. Fresh selector-unset fixed C4096/M512 improves **381.772 ->
387.571 tok/s (+1.519%, 5/5 wins)** and is **1.79668x** behind llama.cpp HIP.
Selector-unset 512/1K/4K is **-0.054%/+0.199%/+0.054%**, exact/finite and
lifecycle-clean, with 4K winning **3/3**. H6A global remains registered
rollback, H6A SWA remains source, gfx1151 stays excluded, and **81/81** guards
pass
([H6N production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-production.json) ·
[candidate/runtime](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-candidate.json) ·
[target](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-global-dense-initial-score-arena512-target.json)).

The clean post-H6N cached reprofile is **386.959 tok/s / 1,309.339 ms / 2,192
dispatches** versus matched llama.cpp HIP **696.342 tok/s / 718.241 ms**.
Q5/IQ-down/attention/Q6 gaps are **195.952/191.674/126.480/72.532 ms**;
gate/up is **3.851 ms faster** than llama.cpp. Q5's exact geometry, plane,
ownership, and prefetch/wait-split routes are closed. H6N supplies the distinct
attention interval. **WPF-H6P exact staged-wave-publication triple-output IQ3
is now a retained standalone gfx1100 leaf** targeting only H6I's register
liveness; H6I remains source production pending runtime qualification.

The frozen H6I body carries `acc_a[8]`, `acc_b[8]`, and `acc_c[8]` together into
`reduce_local128_triples_batched`. H6P instead computes and wave-reduces A,
writes lane-0 partials without a barrier, ends that accumulator scope, repeats
B/C, then uses one publication barrier, original per-output wave0..3 sums/
stores, and the reuse barrier. Cached ISA preserves **216** FMAs, stride
`0x300`, two barrier instructions/eight dynamic barriers, 23 global loads,
local128/LDS512/grid32768x64, and scratch0. Metadata/runtime VGPR falls
**164/168 -> 107/112** with private0/spill0; SGPR metadata rises **56 -> 78**,
DS instructions rise **60 -> 156**, and code grows **6,264 -> 8,360 bytes**.
The runtime register-file wave ceiling therefore moves **9 -> 13**, but the
added publication traffic makes full timing mandatory rather than inferring a
win from occupancy.

The frozen rows1/7/8/9/M512/P64/P65 matrix passes **9/9** against H6I and sampled
CPU bytes; all **45/45** actual layers win both clocks. H6P now also qualifies
as a bounded default-off owner through the existing `grouped_raw_iq_active_experts`
ABI, raw allocation, and `grouped_iq_prefill` library. Generic role matching is
tightened from K1024/N3072 to exact K1024/N3072/E256; E255 and every other
shape/registration/backend miss fail closed without backend/quant branches.

Complete natural M512 is KL0/byte-exact across logits, all **48/48** hidden
boundaries, K/V/`KVLiveSpans`, repeat, and teardown. Four cached requests retain
**2,192 dispatches** and exact **45 H6I or 45 H6P + two H5J** down topology;
IQ3/request-sum/span moves **335.561/1,350.501/1,377.064 -> 326.309/1,346.568/
1,368.182 ms (-2.757%/-0.291%/-0.645%)** at H6I/H6P runtime VGPR **168/112**.
Default-off 512/1K/4K gains **+0.764%/+0.416%/+0.242%**, 3/3 exact wins each;
fixed C4096/M512 gains **+0.141% (4/5 wins)**. Workspace/scratch stay
**161,120,256/600,141,856 bytes**, **246/246** guards pass, and H6I remains
source pending separate publication
([H6P candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-staged-wave-publication-candidate.json) ·
[post-H6N residual / target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6n-matched-residual.json)).

The later source publication supersedes the pending-source language above: H6P
is now the retained gfx1100 IQ3 source, and the clean fixed natural C4096/M512
refresh is **389.145 tok/s / 1,302.492 ms**, **1.77515x** behind freshly rerun
matched llama.cpp HIP **690.791 tok/s / 714.008 ms**. **WPF-H6Q exact
compact-shuffle-loop staged-wave IQ3 is now a retained standalone leaf.** Its
separate `#pragma clang loop unroll(disable)` helper preserves every H6P dynamic
shuffle/order, **216** FMAs, stride `0x300`, 23 global loads, three staged
planes, two/eight barriers, local128/grid32768x64/LDS512, and bytes. Static
bpermutes fall **120 -> 24**, code **8,360 -> 6,620 bytes**, metadata/runtime
VGPR **107/112 -> 95/96**, while LDS loads/stores remain **24/12** and
private/spill/scratch remain zero. Frozen **9/9** and all **45/45** actual layers
are exact and win both clocks: event **329.124 -> 313.405 ms (-4.776%, 1.050x)**
and wall **326.037 -> 317.946 ms (-2.481%, 1.025x)**, with minimum layer wins
**1.038x/1.016x**. H6Q qualifies through the existing
`grouped_raw_iq_active_experts` ABI/raw allocation/library and is now the
retained source default; H6P is explicit same-ABI rollback. Complete natural
M512 is KL0/byte-exact across logits, all **48/48** hidden boundaries, K/V/
`KVLiveSpans`, repeat, and teardown. Four cached requests preserve **2,192
dispatches** and substitute exact **45 H6P -> 45 H6Q**, cutting IQ3/request-sum/
span **4.725%/0.487%/1.076%**. Selector-unset 512/1K/4K gains
**+0.730%/+0.571%/+0.359%**, 3/3 wins each; fixed C4096/M512 gains **+0.467%
(5/5 wins)** at **390.887 tok/s**. Workspace/scratch remain unchanged and
**156/156** guards pass.

The previous clean H6Q source reprofile remains **390.947 tok/s / 1,301.236 ms
/ 2,192 dispatches**. **WPF-H6R exact DPP peer-exchange staged-wave IQ3 is now
the retained source default; H6Q is explicit same-ABI rollback.** Candidate-local
permlanex16/DPP helpers change only H6Q's peer operation; the physical leaf
remains byte-exact and both-clock positive on all **45/45** actual layers, with
zero bpermutes, exact **24 permlanex16 + 96 DPP**, unchanged 216
FMAs/LDS/barriers/stride/grid, and metadata/runtime VGPR **101/104** at
private0/spill0/scratch0. Complete natural M512 is KL0/byte-exact through all
**48/48** hidden boundaries, complete K/V and `KVLiveSpans`, repeat, and
teardown. Four production-identical cached requests preserve **2,192
dispatches** and change only **45 H6Q -> 45 H6R** calls; IQ3/request-sum/span
falls **13.837%/3.578%/4.014%**. Fresh selector-unset 512/1K/4K gains
**3.793%/3.274%/1.992%**, all 3/3 exact wins, while fixed C4096/M512 improves
**391.307 -> 407.780 tok/s (+4.210%, 5/5)**. H6R reuses raw allocation and
`grouped_iq_prefill`; ABI, workspace/scratch, sidecar, and dispatch count remain
unchanged. gfx1151 fails closed and **219/219** guards pass. Clean committed H6R
reprofiling reaches **407.091 tok/s / 1,247.252 ms / 2,192 dispatches**, versus
campaign-start **169.516 tok/s / 3,001.692 ms** and matched llama.cpp HIP
**690.791 tok/s / 714.008 ms**. Q5/attention/IQ-down/Q6 gaps are now
**197.358/127.879/125.185/72.769 ms**. Q5 remains mechanism-closed; H6A SWA is
the largest actionable exact leaf at **117.506 ms / 144 calls**. Select
one-shot target-only **WPF-H6S exact DPP peer-exchange dense-initial SWA qrow4**:
retain H6A's local32/grid2304x32/LDS0 ownership, BF16 cache reads, two QK passes,
ordered F32 adds, softmax/PV order, complete bytes, and `KVLiveSpans`; replace
only offset16/8/4/2/1 bpermute peer sources with H6R's permlanex16 then DPP
8/4/2/1. Admission requires exact **12 remaining bpermutes + 8 permlanex16 + 32
DPP**, unchanged 12 global u16 loads/4 exp/no barriers, code <=8,000 bytes,
metadata/runtime VGPR <=80/80, private0/spill0/scratch0, and every
starts0/128/256/384 plus weighted 144-call both-clock win. Remove all H6S
surfaces on any miss
([post-H6R residual / H6S target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6r-matched-residual.json) ·
[H6R production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-dpp-peer-exchange-production.json) ·
[H6R candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-dpp-peer-exchange-candidate.json) ·
[post-H6Q target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6q-matched-residual.json) ·
[H6Q production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-compact-shuffle-loop-production.json)).

The binding one-shot screen now rejects H6S and removes every candidate
implementation/test/key/exclusion surface. Complete starts0/128/256/384 outputs
are byte-exact/finite with immutable spans and clean lifecycle. Codegen realizes
exact **12 bpermutes + 8 permlanex16 + 32 DPP**, unchanged 12 global u16 loads,
four exp, FMA/store counts, and zero barriers; code falls **7,044 -> 6,676
bytes**, metadata VGPR **64 -> 59**, and rocprof reports local32/grid2304x32/
VGPR64/LDS0/scratch0 with no compiler activity. Despite that physical success,
every start regresses event and wall; weighted 144-call event moves
**94.696 -> 108.850 ms (+14.946%, 0.870x)** and wall
**96.707 -> 112.761 ms (+16.601%, 0.858x)**. Skip runtime qualification, retain
H6A SWA/H6N global and clean H6R **407.091 tok/s**, and do not retry DPP
attention peer exchange without a materially new premise
([H6S rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-dpp-peer-rejected.json)).

**WPF-H6T exact fused-DPP-add staged-wave IQ3** is now the retained gfx1100 IQ3
source default; H6R remains explicit same-ABI rollback. The leaf still passes
**9/9** and **45/45** actual layers both clocks while converting H6R's **72
`v_add_f32_dpp` + 24 row-shift-1 `v_mov_b32_dpp` -> 96 DPP adds + zero moves**.
It retains 24 permlanex16, 216 FMAs, 23 global loads, **24 LDS b128 loads + 12
two-address stores**, two/eight barriers, stride `0x300`, and local128/
grid32768x64/LDS512 at metadata/runtime VGPR **101/104**, private0/spill0/
scratch0; slots/code fall **1,399 -> 1,384 / 8,016 -> 7,920 bytes**. Complete
natural M512 is KL0 and byte-exact across **48/48** hidden boundaries, complete
K/V/`KVLiveSpans`, repeat, and teardown. Four cached requests preserve **2,192**
dispatches and replace exact **45 H6R -> 45 H6T**; IQ3/request-sum/span move
**267.433/1,284.605/1,313.165 -> 261.844/1,283.120/1,304.737 ms
(-2.090%/-0.116%/-0.642%)**. Fresh selector-unset fixed C4096/M512 improves
**407.600 -> 408.900 tok/s (+0.319%, 5/5)** and is **1.68939x** behind matched
llama.cpp HIP **690.791**; fresh 512/1K/4K gains **+0.351%/+0.423%/+0.176%** at
**383.162/308.780/193.629 tok/s**, all **3/3** exact wins. Change only the source-
map value: the nine-entry ABI, allocation, workspace, total scratch, dispatch
count, and gfx1151 exclusion stay unchanged; **144/144** source guards pass
([H6T production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-production.json) ·
[H6T candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-candidate.json) ·
[H6T target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-fused-dpp-add-target.json)).

**WPF-H6U exact DPP-add wave reduction for Q6 activation-row consumers** is the
retained gfx1100 Q6 source default; H6E remains explicit rollback. The exact
**11/11** leaf replaces **320/400/400 `ds_bpermute_b32`** with **64+256 / 80+320 /
80+320 permlanex16 + `v_add_f32_dpp`**, cutting physical code/slots and runtime
VGPR **136/168/168 -> 112/144/144** at unchanged LDS/private0/spill0/scratch0.
Complete natural M512 is KL0 and byte-exact across all **48/48** hidden
boundaries, complete logits, K/V/`KVLiveSpans`, repeat, and teardown. Four cached
requests preserve **2,192** dispatches and substitute exact **2/46/94 H6E ->
H6U** consumers; consumer/Q6/request-sum/span move
**54.144/86.958/1,276.589/1,305.317 -> 48.443/81.029/1,274.060/1,295.123 ms
(-10.529%/-6.817%/-0.198%/-0.781%)** with H6U runtime VGPR **144/112/144** and
scratch0. Fresh selector-unset fixed C4096/M512 improves **409.485 -> 411.704
tok/s (+0.542%, 5/5)** and is **1.67788x** behind matched llama.cpp HIP
**690.791**; fresh 512/1K/4K gains **+0.524%/+0.427%/+0.286%** at
**384.637/309.813/194.321 tok/s**, all **3/3** exact wins. Only three selected-
map values change; F32 N72 fallback, allocation/workspace/scratch/dispatches,
and gfx1151 remain unchanged, and **153/153** guards pass
([H6U production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q6-dpp-wave-reduction-production.json) ·
[H6U candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q6-dpp-wave-reduction-candidate.json) ·
[post-H6T residual / H6U target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6t-matched-residual.json)).

Clean committed H6U reprofiling reaches **410.220 tok/s / 1,232.836 ms / 2,192
dispatches**, **+141.994%** over campaign start and **1.68395x** behind matched
llama.cpp HIP **690.791 tok/s / 714.008 ms**. **WPF-H6V exact DPP-add Q5 wave
reduction is rejected and fully removed.** All six H5Y production roles are
byte-exact. Codegen replaces exact **160/480/400/480/400/400 bpermutes** with
**32/96/80/96/80/80 permlanex16 + 128/384/320/384/320/320 DPP adds**, zero
moves, fewer code slots/VGPR, identical FMA/global-load/LDS-store/barrier/global-
store counts, and scratch0. The 188-call weighted event/wall improves
**269.681/271.908 -> 267.729/267.342 ms (-0.724%/-1.679%)**, but only **3/6**
roles win both clocks. BF16 K3072/N1024 regresses **+12.795%/+13.346%**, BF16
K6144/N3072 misses event by **0.560%**, and F32 K3072/N6144 regresses
**+4.137%/+1.757%**. The frozen universal all-role gate therefore fails. Skip
runtime qualification, remove HIP/Python/key/export/test/gfx1151-exclusion
surfaces without tuning, retain H5Y/H6U policy/allocation/fallbacks, and close
this exact universal transfer
([H6V rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-q5-dpp-wave-reduction-rejected.json) ·
[target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6u-matched-residual.json)).

**WPF-H6W exact late-start dense-initial SWA qrow4 aligned global-score-record
replay is the retained gfx1100 SWA source default; H6A is explicit rollback.**
The admitted local32/VGPR56/LDS0/scratch0 leaf emits one b128 score store/load,
removes all four second-QK loads and 20 reduction sites, and cuts code/slots
**7,044→4,984 B / 1,345→871**. Runtime borrows the aligned
**18,874,368-byte** prefix of the existing Q5 F32 plane with same-stream
projection → attention → FFN lifetime and no allocation/workspace growth.
Source promotion changes only one selected-map SWA value and adds the complete
H6A rollback map; H6N global, starts0/128 fallback, runner/KV ABI, and kernel
bodies do not change. Complete M512 remains KL0/exact across all **48/48**
boundaries, logits, K/V/spans, repeat, and lifecycle. Production-identical
cached topology remains exact **48 H6N + 72 H6A + 72 H6W** at **2,192**
dispatches and cuts selected late SWA/attention/kernel-sum/span
**23.808%/12.344%/1.319%/2.018%**. Fresh selector-unset fixed M512 gains
**+1.515% (5/5)** at **417.421 tok/s**; 512/1K/4K gains
**+1.304%/+0.736%/+0.153%**, all **3/3**, with unchanged workspace/scratch,
gfx1151 exclusion, and **115/115** guards
([production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-global-score-replay-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-global-score-replay-candidate.json)).

Clean post-H6W reprofiling reaches **416.891 tok/s / 1,214.475 ms / 2,192
dispatches**, **1.65700x** behind matched llama.cpp HIP **690.791 tok/s /
714.008 ms**. Q5/IQ-down/attention/Q6 gaps are **198.017/119.429/105.690/
66.788 ms**. Q5's representative **223.393-ms** H5Y consumers remain dominant,
but static-shape source-MMQ rejects all six material shapes at max KL **0.585291–4.622387**
and the three dominant exact-value F32/SGEMM shapes reject at **0.402533–
0.846753** over the complete 18-prompt/576-step category-heldout gate. H6V
remains closed without role subsetting; producer/prefetch/geometry routes are
not reopened.

Select target-only **WPF-H6X exact workgroup-resident IQ3_XXS grid table** on
the **264.602-ms / 45-call** H6T body. Current `load_iq3_segment` issues two
lane-divergent constant/global `IQ3_XXS_GRID[256]` loads per segment and waits.
Natural M512 routing has **33,547 rowbatch8 epochs / 103,056,384 segment
calls**, modeling **824,451,072** table-load wave instructions and **105.530
GB** logical table bytes. A separate gfx1100 sibling must use all 128 threads to
copy the exact 256-entry uint32 table into **1,024 bytes LDS** once/workgroup,
barrier, and source only those two lookups from LDS. Selection arithmetic moves
global table loads to **5,898,240 wave instructions / 0.755 GB (-99.2846%)**
while adding **737,280 barriers**, **1.0731%** of H6T's existing dynamic main
barriers; this is not physical-traffic or speed evidence.

Freeze RED before executable changes. Preserve exact table values, aux/sign/
scale decode, **216 FMAs, 24 permlanex16, 96 DPP adds**, three staged scopes,
wave publication/serial sum/store, rowbatch8, P256/P64 traversal, grid, ABI,
raw allocation, and workspace. Physical admission requires exact **19 static
global loads, two coalesced table preloads, six LDS table reads, three barriers,
1,408-byte metadata LDS / <=1,536 runtime LDS, VGPR <=101/104**, and zero
private/spill/scratch. Rows1/7/8/9/M512, P64/P65/reversed/tails/CPU, cached named
execution with no compiler, and every **45/45** actual-layer event+wall win are
binding. Any miss removes all H6X implementation/test/key/export/gfx1151-
exclusion surfaces without tuning; H6T stays source and runtime qualification is
separate
([post-H6W residual / target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6w-matched-residual.json)).

H6X is **rejected at the frozen physical gate before profiler/timing**. Its
**10/10** cached exact matrix preserves all table/output/CPU/lifecycle facts,
and ISA realizes global loads **23→19**, six LDS reads, one coalesced preload
store, barriers **2→3**, metadata LDS **384→1,408 bytes**, 216 FMAs, 24
permlanex16, 96 DPP adds, and private0/spill0/scratch0. Code/slots move
**7,920/1,384→7,944/1,381**, but metadata VGPR rises **101→103**, exceeding the
frozen **≤101** ceiling. By contract, skip cached trace and all-45 timing, remove
every H6X implementation/test/key/export/gfx1151-exclusion surface without
tuning or rerun, and retain H6T source
([H6X rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-grid-lds-rejected.json)).

Select target-only **WPF-H6Y exact IQ3 packed-prefix b32 load** without tuning
or reopening H6X. H6T's **264.602-ms / 45-call** body emits exact **8 b128 + 9
b32 + 6 d16 = 23** static global loads. In each of three scopes, the two d16
loads read adjacent block bytes0..1 FP16 scale and bytes2..3 selector pair.
H6Y must load those four bytes once as little-endian b32, recover the exact bits
in registers, and leave aux/table/sign/magnitude/scale/FMA/reduction/store order
unchanged. Expected codegen is **8 b128 + 12 b32 + zero d16 = 20** global loads,
unchanged DS/barriers/LDS384/runtime512/216 FMAs/24 permlanex16/96 DPP/stride,
metadata/runtime VGPR **≤101/104**, and private/spill/scratch0. Natural routing
models **412,225,536 fewer prefix global-load wave instructions** at unchanged
**52.765 GB** logical bytes; this is selection arithmetic only. Freeze RED, then
require complete FP16-bit/row/P64/P65/CPU/lifecycle bytes, cached named execution
without compiler, and every layer **1..45** plus aggregate to win both clocks
under 5/15/5. Any miss removes H6Y without tuning/rerun; H6T remains source
([post-H6X residual / H6Y target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6x-rejection-matched-residual.json)).

H6Y is **rejected at the frozen physical gate before profiler/timing**. The
correct rolling-window implementation loads `block + group32*8 + 2*local8`,
takes selectors from bits16..31, and broadcasts wave-lane0's scale bits0..15.
Its cached exact matrix passes **11/11**, including all 12 finite FP16 classes
and all 256 selector bytes. ISA realizes global loads **23→20**, unchanged two
barriers/LDS384/216 FMAs/24 permlanex16/96 DPP, and cuts code/slots
**7,920/1,384→7,872/1,357**. However the scale exchange adds three
`ds_bpermute_b32` instructions to the frozen unchanged-DS set and metadata VGPR
rises **101→106**, exceeding the frozen **≤101** ceiling. Skip cached trace and
all-45 timing by contract; remove every implementation/test/key/export/gfx1151-
exclusion surface without tuning or rerun, and retain H6T source
([H6Y rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-packed-prefix-b32-rejected.json)).

**WPF-H6Z exact late-start global qrow4 aligned score/weight replay is now the
retained gfx1100 global source default; H6W remains the explicit H6N-global
rollback.** The source change modifies only `global_m128_c4096_first_fill_exact`;
SWA remains H6A early/H6W late. H6Z groups only starts256/384, preserves H6N's
complete arithmetic and `KVLiveSpans` ABI, and remains local32/grid1536x32 at
metadata/runtime VGPR **47/48**, LDS0/private0/spill0/scratch0. It borrows the
same aligned **18,874,368-byte** H6W prefix of the existing **150,994,944-byte**
Q5 F32 plane, passes its strict **12,582,912-byte** extent, and adds zero
allocation, workspace, sidecar, or dispatch. Starts0/128 retain H6N; wrong
shape, metadata, registration, backend, or unbound ownership fails closed.

Fresh source-selected natural M512 is KL0/top-1 100% and byte-exact across
complete logits, final/post hidden, all **48/48** hidden boundaries, K/V plus
every `KVLiveSpans` field, repeat, and lifecycle. Four cache-only source requests
preserve **2,192** dispatches and production topology **24 H6N + 24 H6Z + 72
H6A + 72 H6W**. Late global falls **23.894→12.231 ms (-48.812%)**, attention
**125.254→116.041 (-7.355%)**, kernel sum **1,214.563→1,205.023 ms (-0.785%)**,
and span **1,241.814→1,227.056 ms (-1.188%)**, with zero compiler process.

Fresh selector-unset fixed natural C4096/M512 improves **417.180→420.785 tok/s
(+0.864%, 5/5)**. H6Z cannot dispatch in C512/C1024 sessions, so those are
unchanged-path controls at **390.831/311.543 tok/s**; binding C4096/4K improves
**194.478→194.694 (+0.111%, 2/3)**. The exact fixed/event/span wins retain H6Z
under the cycle-wall policy despite aggregate 4K noise. Keep H6W/H6N and H6A
rollbacks; gfx1151 remains excluded and **126/126** guards pass
([H6Z production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-score-weight-replay-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-global-score-weight-replay-candidate.json)).

Clean committed H6Z reprofiling reaches **423.233 tok/s / 1,195.702 ms / 2,192
dispatches**, **+149.671%** over campaign start and **1.63218x** behind matched
llama.cpp HIP **690.791 tok/s / 714.008 ms**. The exact topology remains **24
H6N + 24 H6Z + 72 H6A + 72 H6W**, the kernel span is **1,217.373 ms**, and
profiling spawns no compiler. Current Q5/IQ-down/attention/Q6 gaps are
**198.740/116.810/93.654/66.495 ms**; those four explain **98.756%** of the
**481.694-ms** kernel gap. Gate/up is already **1.929 ms faster** than llama.cpp.

Select target-only **WPF-H7A exact late-start SWA scaled-score replay** on the
**62.562-ms / 72-call** H6W body. H6W stores one aligned `float4` of unscaled
dots per logical slot, computes `dot * scale` for max in pass one, then repeats
the same multiply after loading the record in pass two. A separate gfx1100
sibling must compute each visible scaled score once, use that same F32 bit
pattern for max and the record, and replay `exp(score - max)`. The natural-M512
schedule removes **255,135,744** duplicate scale multiplications with zero byte,
workgroup, record, allocation, workspace, ABI, or result change; this is target
operation-count rationale, not a speed claim.

Freeze RED first. Preserve H6W's one-wave/one-head/qrow4 ownership, every QK
product/reduction, score/max/exp/denominator/PV/output bit, aligned
**18,874,368-byte** plane, complete `KVLiveSpans`, starts0/128 fallback, package
policy, and gfx1151 exclusion. Physical admission requires four second-pass
scale-subtract FMA sites removed (**total `v_fma_f32` <=52 versus 56**),
unchanged eight u16 loads, one b128 record load/store, 32 bpermutes, four exp
sites, code **<=4,984 B / <=871 slots**, metadata/runtime VGPR **<=54/56**, and
LDS0/private0/spill0/scratch0. Complete starts256/384 H6W/CPU/scaled-record/
span/poison/lifecycle bytes, cached named execution without compiler, and both
starts plus weighted **72-call** event+wall wins under one immutable 5/15/5
screen are binding. Any miss removes every H7A surface without tuning/rerun;
runtime/source qualification stays separate
([post-H6Z residual / H7A target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h6z-matched-residual.json)).

H7A is **rejected at its first binding complete-byte gate**. Structural,
registry, gfx1151-exclusion, and strict-preflight checks pass, and the frozen
body compiles once. Complete outputs are finite but differ from H6W at
**80,469/1,179,648** elements for start256 and **100,075/1,179,648** for
start384, with maximum absolute **4.656613e-9 / 3.7252903e-9**. The target
premise missed an ISA numerical boundary: H6W compiles each replay
`dot * scale - max` as `v_fma_f32`, while a stored scaled score rounds the
multiply before subtraction. That changes exp and PV bits, so exactness fails
regardless of the small magnitude. Stop before code-object candidate
adjudication, rocprof, or timing; do not waive, tune, or rerun. Remove the RED,
HIP/Python/key/export/gfx1151-exclusion surfaces and retain byte-identical H6W/
H6Z production **423.233 tok/s / 1,195.702 ms**
([H7A rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-swa-scaled-score-replay-rejected.json)).

Fresh post-H7A production reaches **422.602 tok/s** with exact state/lifecycle;
a compiler-free representative trace is **1,200.759 ms / 2,192 dispatches** and
retains **45 H6T calls / 263.748 ms**. Select target-only **WPF-H7B exact lane-
parallel IQ3 final-row publication**, materially distinct from the closed H6X/
H6Y load reductions. H6T's thread0 final phase emits **24 `ds_load_b128` + 24
`global_store_d16_hi_b16`** sites to serially handle eight rows. A separate
sibling leaves the complete IQ3/decode/FMA/DPP/LDS-publication path unchanged,
then assigns lanes0..7 one row each. Every row retains the identical serial
wave0→1→2→3 F32 association and same three BF16 destinations before the
unchanged trailing barrier.

Across **34,352,128** natural publication phases, modeled DS-load and global-
store wave issues each move **824,451,072→103,056,384 (-87.5%)**, with unchanged
logical bytes; this is issue-count rationale, not speed evidence. Freeze RED
before code. Require complete H6T/CPU/poison/lifecycle bytes across boundary and
M512 cases plus all **45/45** actual layers. Static admission is exact **3 b128
LDS loads + 3 d16 stores**, unchanged 23 global loads/12 LDS stores/two barriers/
216 FMAs/24 permlanex16/96 DPP, code **<=7,920 B / <=1,384 slots**, metadata/
runtime VGPR **<=101/104**, LDS **384/<=512 B**, and private/spill/scratch0.
Cached named execution without compiler and all 45 layers plus aggregate both-
clock wins under one immutable 5/15/5 screen are binding. Remove H7B on any miss
without tuning/rerun; runtime/source qualification remains separate
([post-H7A residual / H7B target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h7a-rejection-matched-residual.json)).

H7B is **rejected at the first compiled physical-resource gate**. The frozen
boundary/preflight matrix passes **10/10** with complete H6T/CPU bytes and clean
lifecycle. Its first code object realizes exact **23 global loads / 3
`ds_load_b128` / 12 `ds_store_2addr_b32` / 3
`global_store_d16_hi_b16` / 2 barriers / 216 FMAs / 24 permlanex16 / 96 DPP**,
and cuts code/slots **7,920/1,384→5,916/994**. Metadata remains LDS384,
private0, spill0, but VGPR rises **101→108**, violating the frozen **≤101**
ceiling. Stop before rocprof/runtime resources/all-layer timing; do not tune,
recompile, or rerun. Remove the RED plus every H7B HIP/Python/key/export/gfx1151
surface and retain H6T/H6Z production **422.602 tok/s / 1,200.759 ms**
([H7B rejection](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-lane-parallel-final-rows-rejected.json)).

Fresh post-H7B production is **422.947 tok/s** with compiler-free
**1,199.578 ms / 2,192 dispatches**. The remaining three generic raw-Q6 calls
own median **28.474 ms / 34.904% of Q6**: K12288/N3072 BF16 **10.347 ms**,
K3072/N9216 F32 **9.925 ms**, and K9216/N3072 BF16 **7.988 ms**. Standalone
registry-only **WPF-H7C exact raw-Q6 DPP-add wave reduction is admitted** as two
Q6-only siblings in `quant/gguf_k_gemv.{hip,py}`. H6H's source-F16/rocBLAS route
remains quality-closed; H7C keeps raw source values and exact F32 association.

The frozen matrix passes **22/22** across all three roles and rows1/7/8/9/M512.
The first BF16/F32 object emits exact zero bpermutes, **32 permlanex16 + 128 DPP
adds**, unchanged 24 global loads/one store/eight b128 LDS stores/two LDS loads/
one barrier/32 ordered FMAs, and cuts code/slots **4,840/843→4,228/681** and
**5,040/909→4,452/749**. Metadata/runtime VGPR is **60/64** and **55/56**;
LDS512 and private/spill/scratch0 are unchanged. Cached named execution covers
exact grids **98,304x64 / 589,824x32** with zero compiler.

The immutable actual-weight 5/15/5 screen improves all three roles on event and
wall. Layer-0 down is **14.866/14.868→14.741/14.750 ms**, layer-47 Q is
**10.752/10.795→10.705/10.700 ms**, and layer-47 output is
**11.630/11.639→11.537/11.547 ms**. Aggregate event/wall improves
**37.248/37.303→36.983/36.998 ms (-0.712%/-0.817%)** with exact bytes and
lifecycle.

H7C now also has a qualified bounded default-off runtime owner. gfx1100 exports
`GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS` for exactly the three M512
`(quant, output ABI, rows, K, N)` keys, while
`GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS` and the live
`GGUF_RAW_K_PREFILL_ROLE_VARIANTS` remain empty. `gguf_linear` consumes the map
generically and validates registration; wrong row/quant/K/N, malformed map,
missing key, and gfx1151 all retain the existing generic coltile route. No
runner, allocation, ABI, wrapper, registry, kernel, or gfx1151 change is needed.

Complete natural M512 state is KL0/byte-exact across logits, final/post hidden,
**48/48** hidden boundaries, full KV/spans, and repeat at unchanged
**161,120,256 / 600,141,856-byte** ordered workspace/total scratch. Four
cache-only C512 requests preserve **2,192 dispatches** and exact family counts;
controls own **2 BF16 + 1 F32** generic calls and candidates own the matching
three H7C calls at unchanged offsets/resources. Selected raw-Q6/Q6/span medians
improve **28.543/81.457/1,280.898→28.220/81.105/1,279.005 ms** with zero
compiler. Matched C4096/M512 improves **420.701→420.914 tok/s (+0.0505%, 4/5)**,
and 512/1K/4K medians improve **+0.0552%/+0.0274%/+0.0179%** with exact state
and complete lifecycle.

Source promotion changes only `GGUF_RAW_K_PREFILL_ROLE_VARIANTS` to copy H7C;
the named empty map remains explicit rollback. Fresh source-selected M512 state
is KL0/byte-exact across **48/48** boundaries and full KV/spans. A fresh four-run
trace preserves **2,192 dispatches** and exact **2 BF16 + 1 F32** ownership while
selected raw-Q6/Q6/span improves
**28.583/81.639/1,283.417→28.376/81.470/1,280.788 ms** with zero compiler. Fresh
source aggregate results are mixed: fixed C4096/M512 is
**419.433→418.487 tok/s (-0.225%, 2/5)**; 512/1K/4K is
**+0.0925%/+0.0372%/-0.0488%**. Retain source under the cycle-wall policy based
on repeatable selected-subwindow/span gains and the immutable all-role leaf
screen, while recording the fixed/4K rows as aggregate noise rather than wins.
The last clean committed checkpoint remains **422.947 tok/s / 1,199.578 ms**
until the required post-commit reprofile
([H7C production](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-raw-q6-dpp-wave-reduction-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-raw-q6-dpp-wave-reduction-candidate.json) ·
[target](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-post-h7b-rejection-matched-residual.json)).

Fresh post-H7C production reaches **422.786 tok/s** with cache-only
**1,197.499/1,219.043-ms** representative kernel sum/span, **2,192 dispatches**,
and zero compiler. Q5/IQ-down/attention/Q6 gaps are
**196.915/117.620/93.693/66.653 ms**. The isolated **WPF-H7D** Q5 schedule probe
closes row-interleaved VOPD as an exact next leaf: both scalar-order variants
compile to **52 paired FMAs** at VGPR122, and forced pairing is illegal under
gfx1100 `src0` VGPR-bank assignment. Do not create a production H7D surface or
retry Q5 scheduling without a new layout/codegen premise.

Standalone **WPF-H7E IQ3-only two-plane residual-D4 source-MMQ is admitted** in
`quant/gguf_iq_source_mmq_prefill.{hip,py}`. The new separately named
I128/J128/K256 IQ3 consumer reuses the qualified
`gguf_q8_0_mmq128_quantize_bf16_d4x2` producer and accumulates both activation
planes while each raw-IQ3 tile is staged. Existing one-plane IQ3/IQ4 functions
remain byte-frozen, H6T/IQ4 remain production source, and gfx1151 is excluded.
The frozen correctness matrix passes **9/9** across rows1/7/8/9/M512 and expert
tails with complete overwrite, independent CPU quality, immutable metadata,
finiteness, and lifecycle.

The first object is local `(32,8)`, grid `(24,mmq_total_rows/128)`, dynamic
LDS57,856, code **31,564 B**, metadata VGPR/SGPR **148/44**, private/spill/
dynamic-stack0; cached execution reports runtime VGPR **152**, LDS launch
57,856, scratch0, the intended symbol, and zero compiler. ISA contains exact
**128 integer WMMAs / five barriers / 64 BF16 stores**. The one-shot
producer-inclusive 5/15/5 screen wins both clocks on every **45/45** actual IQ3
layer and aggregate event/wall
**247.297/260.672→186.732/180.752 ms (-24.491%/-30.659%)**. Max leaf KL is
**0.000487**, minimum top-1 **99.941%**, and lifecycle recovers.

A temporary bounded default-off owner selected only exact-M512 IQ3, reused
`expert_gate_up`, and added zero scratch. Natural-M512 state passed at KL
**0.000224** / top-1 **100%**; cached tracing proved **45 H6T → 45 tile128 + 45
producer + 45 H7E**, runtime VGPR152/scratch0, and diagnostic IQ-down
**269.921→208.298 ms**.

The complete gate rejects runtime ownership. All **18 prompts / 576 steps**
exercise H7E-derived state at M512; max KL is **5.630805 > 0.05** and general-
Japanese top-1 is **115/128 = 89.844% < 90%** (suite **531/576 = 92.188%**).
Same-mode repeats are deterministic, free-running equality is **21/54 h16** and
**6/54 h32**, and Poolside fallback/lifecycle pass. Remove the live/qualified
maps, residual plan/launch route, tile128 export, optional libraries, and runtime
test; skip promotion timing. Keep only the standalone H7E leaf as diagnostic
evidence. H6T/IQ4 remain production at **422.786 tok/s**
([H7E rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-complete-quality-rejected.json) ·
[candidate](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-source-mmq-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7c-matched-residual-iq3-d4x2-target.json)).

**WPF-H7G exact padded-row Q5 compute is the retained complete-map rollback
under H7H source** in `quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}`. Its four
exact-role exports and wrappers cover natural-M512 BF16 K3072/N12288 `c8r12`,
BF16 K6144/N3072 `c16r5`, F32 K3072/N6144 `c16r5`, and F32 K3072/N9216
`c8r10`; its `r4/r8` values retain H5Y arithmetic. The complete H5Y eight-role
map remains a second named rollback. The
standalone rows1/7/8/9/M512 matrix passes **23/23**, first-object dual/scalar
FMA sites become **91/5, 66/14, 66/14, 73/7**, and integrated leaf wall
improves **136.993 -> 129.092 ms (-5.767%)** with exact H5Y bytes.

Complete source-selected natural-M512 state is KL0/byte-identical across logits,
all **48/48** hidden boundaries, K/V/`KVLiveSpans`, repeat, scratch, and
teardown. Cache-only source tracing records exact **2/12/12/35 = 61** H7G calls
at local128/LDS1536/scratch0/runtime-VGPR168/200 with zero compiler executable.
Fresh H5Y -> H7G fixed C4096/M512 improves
**420.569 -> 423.981 tok/s (+0.811%, 5/5)**; 512/1K/4K improves
**+0.962%/+0.410%/+0.201%**, all **3/3** exact wins. Clean production reaches
**424.845 tok/s / 1,192.424 ms / 2,192 dispatches**, **1.62598x** behind
matched llama.cpp HIP. Q5 falls **255.229 -> 248.888 ms**; the remaining
Q5/IQ-down/attention/Q6 gaps are **190.574/118.366/93.960/66.873 ms**.
Workspace/allocation and gfx1151 remain unchanged; **104/104** final guards
pass
([H7G production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-padded-compute-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-padded-compute-candidate.json)).

Retained-source **WPF-H7H exact full-group Q5 compute** reuses the qualified
unconditional-compute body through separately named gfx1100 exports, primitive
and composite registry keys for the two divisible natural-M512 roles. H5Y
remains unfused fallback, the complete H7G map remains named rollback, and
gfx1151 is excluded. BF16 K3072/N1024 `c8r4` owns **92 calls / 24.093 ms** and
BF16 K9216/N3072 `c12r8` owns **35 / 80.144 ms**.

RED-first leaf correctness passes **13/13**. The production object emits `r4`
**15 dual + 5 scalar** FMA sites at **3,584 B / 587 slots / VGPR72 / LDS512**,
and `r8` **47+9** at **9,728 B / 1,588 slots / VGPR194 / LDS1,536**; both stay
private/spill/dynamic-stack/scratch0. Fresh selector-unset H7G -> H7H source is
KL0/exact across **48/48** boundaries, full state, and repeat at unchanged
workspace/scratch. Cache-only source tracing records exact **61 H7G + 127 H7H**
Q5 calls among **2,925** dispatches on one queue/stream at runtime VGPR
**72/200**, scratch0, and zero compiler. Fixed C4096/M512 improves **423.045 ->
426.745 tok/s (+0.874%, 5/5)**; clean 512/1K/4K gains
**+1.042%/+0.896%/+0.477%**, all 3/3. Clean production reaches **427.407 tok/s
/ 1,185.096 ms / 2,192 dispatches**, **1.61624x** behind matched llama.cpp HIP;
Q5 falls **248.888 -> 237.185 ms**. Final exact guards pass **83/83** plus
**65/65** runner/backend/registry nodes
([production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-group-compute-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7g-matched-full-group-q5-target.json)).

Target-only **WPF-H7I exact raw-Q6 full-group compute** transfers H7H's proven
predicate-elision premise to H7C's separate raw-Q6 body. The three and only
natural-M512 H7C roles are BF16 K12288/N3072 `c4r8`, F32 K3072/N9216 `c2r16`,
and BF16 K9216/N3072 `c4r8`; all row groups are exactly full. They own
**28.482 ms / 34.776%** of current Q6. H7I must remove only the inner
`row < rows` compute mask while preserving H7C's outer group guard, final live-
row store guard, raw-Q6 decode, 32 ordered FMAs, DPP/LDS reduction, ABI, map,
workspace, and generic rollback.

The immutable first object cuts BF16/F32 code **4,228 -> 4,060 / 4,452 ->
4,032 bytes**, slots **681 -> 623 / 749 -> 631**, scalar row comparisons
**9 -> 2 / 17 -> 2**, and raises dual-FMAC sites **1 -> 10 / 1 -> 11**. It
keeps 24 global loads, one global store, 32 permlanex16, 128 DPP adds, one
barrier, LDS512, private0/spill0/scratch0; metadata VGPR **69/64** stays within
the frozen 72 ceiling. The first-and-only all-role screen is byte-exact and
improves weighted H7C -> H7I event/wall **35.840/34.854 -> 20.323/21.974 ms
(-43.295%/-36.954%)**. No H7I source, export, wrapper, key, test, capability,
or live-map change exists. Freeze RED first with strict M512/full-group
preflight and complete H7C fallback; runtime/source remain separate
([post-H7H residual / H7I target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json)).

Standalone **WPF-H7I is admitted** as two separately registered gfx1100-only
raw-Q6 wrappers/exports in `quant/gguf_k_gemv.{hip,py}`. The BF16 wrapper owns
both K12288 and K9216/N3072 roles; the F32 wrapper owns K3072/N9216. Both require
rows exactly 512 and threads128 before HIP loading. H7C remains the complete
fallback and live package map, no H7I capability/source owner exists, and
gfx1151 explicitly excludes both keys.

RED-first correctness passes **22/22** across exact M512 candidate equality and
rows1/7/8/9 H7C/CPU/poison/finite/lifecycle fallback. The first repository code
object reproduces selected BF16/F32 code/slots **4,060/623** and **4,032/631**,
row comparisons **2/2**, dual/scalar FMAC **10/14 and 11/16**, metadata VGPR
**69/64**, SGPR **44/54**, LDS512, and private/spill/scratch0 while leaving H7C
physical fields unchanged. The non-adjudicative actual-weight replay remains
byte-exact and improves weighted event/wall **35.432/34.617 -> 20.089/21.762
ms (-43.302%/-37.135%)**. Cache-only rocprof records exact **2 BF16 + 1 F32**
H7I names at local128, runtime VGPR72/64, LDS512, scratch0, and zero compiler.

The complete exact three-role package capability is now a qualified bounded
runtime owner while `GGUF_RAW_K_PREFILL_ROLE_VARIANTS` remains H7C. Complete
M512 H7C/H7I/repeat state is KL0 and byte-exact across logits, all **48/48**
hidden boundaries, full KV/`KVLiveSpans`, unchanged scratch, and teardown.
Fixed C4096/M512 improves **426.583 -> 429.000 tok/s (+0.567%, 5/5)**; clean
512/1K/4K gains **+0.763%/+0.441%/+0.194%**, all 3/3. Cache-only integration
records exact **2 BF16 + 1 F32 H7I**, zero H7C, **2,925** total dispatches, and
zero compiler. Keep H7C as named complete source/rollback until the separate
source-default gate
([H7I candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json)).

Retained-source **WPF-H7I exact raw-Q6 full-group compute** atomically copies
the complete three-role H7I capability into `GGUF_RAW_K_PREFILL_ROLE_VARIANTS`;
the complete H7C map remains named rollback and the empty generic map remains
fallback. No kernel/wrapper/registry, selector, allocation/workspace, or gfx1151
change accompanies source promotion. Fresh selector-unset H7C -> H7I state is
KL0/exact across **48/48** boundaries, complete logits/KV/`KVLiveSpans`, repeat,
scratch, and teardown. Fixed C4096/M512 improves **427.903 -> 429.434 tok/s
(+0.358%, 5/5)**; clean 512/1K/4K gains **+0.455%/+0.309%/+0.322%**.

Cache-only source tracing records exact **2 BF16 + 1 F32 H7I**, zero H7C,
**2,925** dispatches, local128/LDS512/runtime-VGPR72/64/scratch0, and zero
compiler. Every clean profiled request preserves **2,192** dispatches and that
three-call topology. Clean production reaches **431.310 tok/s / 1,172.241 ms**,
**1.60161x** behind matched llama.cpp HIP; raw-Q6 falls **81.900 -> 74.409 ms**.
Final exact guards pass **93/93** plus **65/65** runner/backend/registry nodes
([production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-production.json) ·
[candidate/runtime](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-compute-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7h-matched-raw-q6-full-group-target.json)).

Out-of-tree **WPF-H7J exact Q5 full-grid bounds specialization is rejected**.
It removed only provably redundant outer/final bounds from H7H's two strict
M512 full-group instantiations and changed no repository surface. The frozen
single 5/15/5 actual-weight screen is byte-exact/finite/lifecycle-clean with
zero compiler, but the 92-call `c8r4` role is **0.99954x event / 0.99127x
wall**. The 35-call `c12r8` role and weighted aggregate are positive; both roles
were predeclared inseparable, so `r8`-only salvage is inadmissible. Keep H7H
source and do not reopen or subset H7J
([rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q5-full-grid-bounds-rejected.json)).

Target-only **WPF-H7K exact late-start SWA score-to-weight publication** is the
next materially distinct gfx1100 boundary in
`attention/laguna_kv_attention.hip`. H6W's starts256/384 score-only replay owns
**72 calls / 62.627 ms**, **54.309%** of current attention. H7K must preserve
H6W's first-pass unscaled-dot records/max updates, fused `dot*scale-max`, lane-0
token-order denominator accumulation, all-lane token-order unnormalized PV,
and final divide. It may only overwrite each aligned `float4` score record with
four weights in a middle pass, then consume those weights in a separate PV pass.

The dynamic rationale removes **255,135,744** lane-0 weight broadcasts and adds
**128,065,536** aligned record operations / **2.049 GB** logical record traffic;
this is not a physical-traffic or speed claim. Freeze starts256/384 together,
strict M128/C512/window512/H72/KV8/D128 preflight, the existing aligned
**18,874,368-byte** plane, H6W/H6A fallback, and unchanged runtime/source policy.
Before timing, the first repository object must retain local32/grid2304x32,
8 u16 K/V loads, 16 F32 query loads, 16 F32 output stores, 2 b128 record loads
and stores, 28 bpermutes, 4 exponentials, 56 FMAs, code <=5,500 B, slots <=950,
metadata/runtime VGPR <=54/56, and LDS/private/spill/scratch/barrier0. Then
require complete H6W/CPU bytes, complete finite/nonnegative causal record bytes,
immutable `KVLiveSpans`, poison/lifecycle, named cache-only trace, and one
5/15/5 screen where both starts and the 72-call aggregate win event and wall.
No start/layer/prompt subset, recompile, tuning, or favorable rerun is allowed.
No H7K source/test/key/export/exclusion exists before RED
([post-H7J residual / H7K target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7j-matched-swa-weight-publication-target.json)).

**WPF-H7K is rejected at the frozen first-object physical gate.** The first
same-source repository object is **674,512 bytes** and SHA-256
`4cdcf8ea...ef67`. H6W remains exact at **4,984 B / 871 slots / VGPR54**. H7K
meets code/slot/VGPR bounds at **5,048 B / 875 / 54**, emits exact 8 u16 K/V
loads, 16 output stores, 28 bpermutes, four exponentials, 56 FMAs, two b128
record stores, and LDS/private/spill/scratch/barrier0. The binding failure is
record reads: both aligned `float4` phases scalarize to **0 b128 + 2 b32 load
sites**, violating the frozen requirement for two b128 record loads. Do not
recompile or rewrite after this miss. Skip candidate correctness, trace, and
timing; remove all H7K body/export/wrapper/key/test/gfx1151 surfaces and retain
H6W/H6Z production
([rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-swa-weight-publication-physical-rejected.json)).

Target-only **WPF-H7L exact IQ3 full-batch/live-tail split** is the next
materially distinct boundary in `quant/gguf_iq_selected_prefill.hip`. The
actual 45-layer route has **230,400** live IQ3 rows across **33,547** rowbatch8
iterations. Exactly **24,650 batches (73.479%) / 197,200 rows (85.590%)** are
full; the **8,897** final tails carry **33,200** live rows but H6T still runs
**37,976 inactive slots (14.150% of current compute slots)** through its dot and
publication bodies.

H7L must split each expert at `begin + floor((end-begin)/8)*8`: complete batches
retain H6T's eight-row activation loads, 216 ordered FMAs, 24 permlanex16, 96
direct DPP adds, wave publication, serial wave0..3 sums, and BF16 stores. At
most one tail computes/publishes only its 1..7 live rows while preserving each
row's same eight magnitude FMAs, scale multiply, permlanex16+DPP 8/4/2/1 order,
serial sum, and store. No padding/layout, compaction, metadata, allocation,
workspace, ABI, package/runtime/source, H6T, or gfx1151 change is allowed. The
source-operation model removes up to **4.200B inactive FMA + 2.333B inactive
exchange wave operations**; it is not a physical-issue or speed claim.

Freeze RED before implementation over empty/uneven experts, tail sizes1..7,
rows1/7/8/9/M512, P64/P65/reversed traversal, complete H6T/CPU bytes, poison,
finite output, and lifecycle. The first object must retain local128/grid32768x64,
metadata/runtime VGPR <=101/104, LDS384/512, private/spill/scratch0, a bounded
live-tail loop, and code <=14,000 B / slots <=2,400 before named cache-only
trace. Consume one all-45 actual-layer 5-warmup/15-counter-rotated/5-launch
screen; every layer and aggregate must win event and wall. No layer/tail/prompt
subset, tuning, recompile, or favorable rerun is admissible. No H7L
implementation/test/key/export/exclusion exists before RED
([post-H7K residual / H7L target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7k-matched-iq3-live-tail-target.json)).

**WPF-H7L is rejected at the first-object physical gate.** The frozen
**511,008-byte** host object preserves the H6T function at **7,920 B / 1,384
slots / metadata VGPR101 / SGPR78 / LDS384 / spill0**. The H7L function is
**49,592 B / 9,082 slots / metadata VGPR133 / SGPR107 / 270 SGPR spills**. It
therefore misses four declared limits: code <=14,000 B, slots <=2,400, metadata
VGPR <=101, and spill0. Its passing fields—LDS384, local ceiling128, wave32,
private0, VGPR-spill0, dynamic-stack0, and scratch-instruction0—do not permit
partial salvage.

Do not change loop spelling, recompile, or retain a favorable tail/layer subset
after this first-object miss. Skip candidate correctness, rocprof, and timing;
remove the H7L kernel/export/wrapper/registry/test/gfx1151 surfaces and preserve
H6T source/package/runtime/workspace exactly
([H7L physical rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-live-tail-physical-rejected.json)).

**WPF-H7M exact IQ3 two-wave/two-K256-partition replay is rejected
out-of-tree.** It maps logical H6T partitions 0/2 and 1/3 to two physical waves while
retaining each K256 dot/DPP tree and serial partition0..3 publication. Physical
preselection occurs before timing: both source forms are private/spill/scratch0;
register activation is **13,132 B / 2,172 slots / VGPR166 / LDS384**, and the
selected lower-VGPR LDS form is **12,744 B / 2,171 / VGPR113 / LDS16,768**.

The selected local64/grid16384x64 body is exact on all **45/45** production
layers but has **0/45** both-clock wins. H6T -> H7M event regresses **246.763 ->
392.180 ms (0.629x)** and wall **261.551 -> 377.358 ms (0.693x)**. No
repository body/export/wrapper/key/test/exclusion exists; keep local128/four-wave
H6T and close one-/two-wave K-partition collapse absent a new reuse operation
([H7M rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-two-wave-k-partition-rejected.json)).

**WPF-H7N exact raw-Q6 c16r4 direct ordered consumption is rejected
out-of-tree.** H7N instantiates a full-group DPP c16r4 consumer for all three
H6U roles and removes their activation-pack and Q6-to-F32 producer launches.
The same immutable BF16/F32 object is **8,900/8,872 B / 1,393/1,390 slots /
VGPR112 / SGPR60 / LDS1,024 / spill0**, with exact 64-FMA, 64-permlanex16, and
256-DPP-add structure. A pre-timing analyzer formula was reconciled from an
incorrect 96 to the emitted **68 load sites** against that unchanged object,
without source change or recompile.

All three actual-role outputs are byte-exact and lifecycle-clean, but each
inclusive comparison loses both clocks by **3.95–5.46x**. The 142-call event
aggregate regresses **48.267 -> 233.861 ms (0.206x)** and wall **48.520 ->
231.238 ms (0.210x)**. Add no body/export/wrapper/key/test/exclusion, retain
H6U/H7I plus **431.310 tok/s**, and do not retry direct raw-Q6 ordered-consumer
replacement without a new cross-tile reuse/decode operation
([H7N rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-c16r4-direct-rejected.json)).

**WPF-H7O exact raw-Q6 full-group geometry crossover is rejected out of
tree.** H7O swaps H7I's constant-32 role geometry without changing decode or
arithmetic: BF16 c4r8 -> c2r16 and F32 c2r16 -> c4r8. The immutable first
object passes all frozen physical gates. BF16 is **4,060 B / 634 slots /
VGPR64 / SGPR54**, F32 is **4,032 B / 620 / VGPR69 / SGPR44**, and both are
local128/LDS512/private/spill/scratch0 with exact **32 ordered FMAs / 32
permlanex16 / 128 DPP adds / 24 global loads / one barrier**.

Every actual-role output is byte-exact and lifecycle-clean. Both BF16 roles win
(**1.077x/1.063x** and **1.093x/1.080x** event/wall), but F32 loses
(**0.912x/0.913x**). Aggregate event/wall improves **21.909/21.905 ->
21.314/21.488 ms**, but the predeclared all-three-role rule forbids favorable
BF16-only salvage after timing. Add no body/export/wrapper/key/test/exclusion,
retain H7I plus **431.310 tok/s**, and do not reopen H7O subsets
([H7O rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-full-group-geometry-crossover-rejected.json)).

**WPF-H7P candidate-distance-only IQ3 D4x2 boundary repair is rejected without
repository code or candidate timing.** An out-of-tree F32-publication probe
keeps H7E's D4x2 producer/consumer arithmetic unchanged and audits every output
against exact H6T for all **45** natural-M512 IQ3 layers. BF16 mismatch is
**16,306,295 / 707,788,800 (2.30384%)**. Candidate-to-boundary distance alone
has risk/recall **6.234%/43.799%** at 1/16 cell, **12.467%/55.611%** at 1/8,
and **24.931%/68.070%** at 1/4; the last still leaves **5,206,620** mismatches.
A 1.0-cell threshold repairs **99.719%** of all outputs yet misses **14,702**
and has an ideal zero-overhead speedup of only **0.592x** versus exact. No
threshold is complete. Add no guard, queue, exact-repair leaf, key, runtime, or
source owner. Do not retry boundary distance without a materially different
prompt-independent error-size signal; H6T remains production at **431.310
tok/s**
([H7P rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-d4x2-boundary-repair-rejected.json)).

**WPF-H7Q/H7R third-plane residual certificates are rejected without repository
code or candidate timing.** H7Q's D4x2/D4x3 disagreement set is sparse at
**2.30385%** and catches **99.7364%** of mismatches, but leaves **42,981** wrong;
adding D4x3 boundary distance becomes complete only at **99.7205%** repair
density. H7R instead outward-rounds producer residual maxima and multiplies
prompt-independent exact-IQ3 L1 sidecars over K64/K128/K256/K1024 bins. All
four zero-margin bounds capture every observed mismatch, but select
**74.5071%/81.1992%/86.4963%/92.8985%** of outputs. Break-even allows only
**30.6591%** exact repair before any guard work. The best free-guard ideal is
**0.695x** exact and the declared read-ceiling best is **0.610x**. Add no
sidecar, guard, queue, sparse-exact leaf, key, runtime, or source owner; retain
H6T at **431.310 tok/s** and do not implement H7Q/H7R
([H7Q/H7R rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-residual-certificates-rejected.json)).

Post-H7R, select target-only **WPF-H7S exact raw-Q6 c2r32 packed-activation
cross-row reuse**. This is not H7N's one-launch row-major c16r4 route or H7O's
constant-32 H7I geometry crossover. It targets the separate **142-call** H6U
family currently composed as tile-K-row activation pack + exact Q6-to-F32
producer + ordered DPP consumer, weighted at **48.267/48.520 ms** event/wall.
H7S keeps the existing caller-owned activation plane and H6U fallback, packs
strict M512 into sixteen rowbatch32 groups, then lets each local128 workgroup
decode two raw-Q6 columns and apply them to 32 aligned BF16 rows.

Every output retains `k = tid + 128n`, the exact signed `scale*quant` conversion,
ordered FMA sequence, H6U permlanex16+DPP 16/8/4/2/1 tree, wave publication,
serial wave0..3 sum, and BF16/F32 store. The 64-accumulator source model changes
H7N's **16 Q6 decodes + 4 scalar activation loads = 68 global-load sites** to
**2 decodes / 8 Q6-field sites + four b128 activation records = 12 sites**.
Across the 2/46/94 production roles, logical input bytes model **0.937x** the
current three-launch chain but **3.107x** H7N; H7N's 4–5x loss despite its lower
byte model is why decode/load instruction count, not minimum bytes, selects
c2r32. Removing only the producer models **142 fewer dispatches/request** and
**2,192 -> 2,050** total. None of these source models is a physical or speed
claim.

Freeze RED before implementation for all three strict M512 roles together,
complete primitive/composite H6U and sampled CPU bytes, exact rowbatch32 pack,
poison/finite/lifecycle, rows511/513 and wrong-shape rejection, registry/source
isolation, unchanged workspace/maps, and gfx1151 absence. The first object must
show local128/wave32, LDS1,024, private/spill/scratch0, one barrier, 64 ordered
FMAs, 64 permlanex16, 256 DPP adds, four activation `global_load_b128` sites,
eight raw-Q6 field loads, code <=14,000 B, slots <=2,400, VGPR <=136, and SGPR
<=96. Named cache-only tracing must show exactly pack then H7S consumer with no
Q6-to-F32 producer and zero compiler. Then one immutable actual-weight 5-warmup/
15-counter-rotated/5-launch screen must improve every role and the weighted
142-call aggregate on event and wall. Any miss removes all H7S surfaces; no
role/c4r16/c8r8/prompt subset, tuning, recompile, or favorable rerun is allowed.
Runtime/source qualification remains separate
([post-H7R residual / H7S target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7r-matched-raw-q6-cross-row-reuse-target.json)).

Reject **WPF-H7S** after the first and only immutable all-role screen. Its first
object realizes the declared form exactly: BF16/F32 code is **5,912/5,884 B**
at **864/860 slots**, both **VGPR112 / SGPR24 / LDS1,024 / spill/scratch0**,
with **4 b128 + 8 raw-Q6 loads, 64 ordered FMAs, 64 permlanex16, 256 DPP
adds**, and one barrier. GREEN passes **8/8**; named tracing shows exactly pack
then H7S consumer for all roles, no F32 producer, and zero compiler. Complete
outputs are byte-exact and finite. The economics fail decisively: role
speedups are **0.290/0.320**, **0.402/0.411**, and **0.293/0.304** event/wall,
and weighted H6U→H7S regresses **49.193→149.544 ms event (0.329x)** and
**49.721→146.161 ms wall (0.340x)**. Remove every body/export/wrapper/key/RED/
gfx1151 exclusion, retain exact H6U and **431.310 tok/s**, and do not salvage a
role or nearby c4r16/c8r8 geometry after timing
([H7S rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-raw-q6-cross-row-reuse-rejected.json)).

Reject **WPF-H7T quality-gated late-start QK-only tensorized score replay** at
the complete quality gate. Its sole repository object satisfies the frozen
physical contract: global/SWA consumers are **3,968/4,224 bytes**, **683/744
slots**, metadata **VGPR49/53 / SGPR50/44**, local/wave32, and LDS/private/
spill/scratch0. GREEN passes **10/10**. Named cache-only tracing records all
four global/SWA starts256/384 chains in exact key-widen→query-pack→one-QK→
consumer order on one queue/stream, with no BLAS PV, value widening, standalone
softmax, or compiler.

The full committed **18-prompt / 576-step** lane executes exactly
**7,008/7,008** H7T launches with frozen algorithms 1/3 at contexts384/512.
Finiteness, **562/576 (97.569%)** top-1, every-category top-1 >=90%, same-mode
repeat determinism, oracle, and lifecycle all pass. Maximum KL reaches
**0.393845 > 0.05**, however, and all four categories exceed the ceiling. Run
no H7T 5/15/5 admission timing; remove the key-widen/consumer bodies, exports,
wrappers, registry keys, zero-allocation owner, RED, and gfx1151 exclusions.
Retain exact H6W/H6Z/H6A/H6N production and **431.310 tok/s**, and forbid
family/start/head/layer/prompt subset, BLAS-retune, rewrite, recompile, or
favorable-rerun salvage
([H7T rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-qk-only-score-replay-quality-rejected.json) ·
[H7T target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7s-qk-only-score-replay-target.json)).

Select target-only **WPF-H7U exact stable parallel MoE active compaction** in
`moe/group_scatter.{hip,py}`. Current gfx1100 intentionally selects
`active_experts`: one local256 workgroup counts by scanning all **5,120** lanes
for each of 256 experts, serially publishes starts/active IDs, then scans lanes
again for each active expert. The clean production trace records **47 calls /
25.187 ms**. The separately registered `active_experts_source_rows_parallel`
wrapper already launches one local256 count workgroup per expert, one local256
fixed-256 Blelloch prefix with wave ballots, and one local256 stable ballot-
scatter workgroup per expert. It preserves ascending lane order and exact
`expert_start`, active IDs/count, sorted lanes, source rows, and weights.

H7U changes only the gfx1100 package capability after admission: **47 serial →
141 parallel** stages (net **+94**, modeled request total **2,286**), while the
47-call **7.717-ms** packed-hidden gather, MMQ tile map, router, gate/up/down
arithmetic, allocation, and workspace remain unchanged. Prior gfx1151
production measured exact metadata/MoE output, **7/7** paired wins, and
**2.564-ms** parallel tracing; that is transfer rationale, not gfx1100
admission. Freeze RED before any gfx1100 owner change. Then require complete
natural-M512 all-47 metadata and full-state bytes, private/spill/scratch0
physical inspection, exact 47 count + 47 prefix + 47 scatter / zero serial /
unchanged 47 gather cache-only trace, and one immutable all-layer plus aggregate
both-clock 5/15/5 screen. No layer/expert/routing-pattern/length subset,
local-size retune, rewrite, recompile, or favorable rerun is allowed. Production
remains **431.310 tok/s** and no W7900 candidate result exists
([H7U target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7t-parallel-moe-compaction-target.json)).

The unchanged registered H7U sibling now passes gfx1100 standalone admission.
One bounded package-only `LAGUNA_MOE_GROUP_COMPACT_H7U_MODE="parallel"`
capability turns RED **2/7** into GREEN **9/9** without changing HIP, wrappers,
registry keys, allocation, or workspace; `LAGUNA_MOE_GROUP_COMPACT_MODE`
remains absent so production resolves serial. The cached gfx1100 object reports
count/prefix/scatter code **988/2,104/1,796 B**, metadata VGPR **10/17/31**,
SGPR **18/18/42**, local256/wave32, LDS **2,048/2,120/80 B**, and zero private,
spill, dynamic stack, or scratch instruction.

Natural M512 proves exact all-47 metadata and packed-hidden bytes plus complete
48-boundary/logit/KV/span state. Selected-region runtime tracing reports
**0.304/0.176/0.673 ms** for 47 count/prefix/scatter calls (**1.153 ms total**),
zero serial, unchanged **47 / 7.865-ms** gather, and **2,286 application
dispatches** on one queue/stream. The immutable all-layer 5/15/5 screen wins
**47/47** both-clock and moves aggregate event/wall **20.508/20.701 →
1.297/1.445 ms (15.813x/14.331x)**. Keep this capability default-off until
separate runtime/source gates qualify fixed and 512/1K/4K requests
([H7U candidate](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-candidate.json)).

H7U is now the gfx1100 source default. Replace the bounded package seam with
`LAGUNA_MOE_GROUP_COMPACT_MODE="parallel"`; `set_group_compact_mode("serial")`
remains the exact rollback and backend peers remain independent. Bounded fixed
C4096/M512 is exact at **430.412→436.602 tok/s (+1.438%, 5/5)**, and clean
source-default 512/1K/4K improves **+1.371%/+1.245%/+0.626%** with **3/3** wins
per length.

The source-selected trace names **47** count, **47** prefix, and **47** scatter
calls, zero serial, unchanged 47 gather, **2,286 dispatches**, one queue, and
zero compiler in all five requests. Runtime LDS allocations are reported in
512-byte granules as **2,048/2,560/512 B**; the frozen object records logical
**2,048/2,120/80 B**. The three stages total **1.155 ms**, versus the prior
serial **25.187 ms**. Clean matched production is **437.189 tok/s** and the
representative kernel sum is **1,160.833 ms**
([H7U production](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-parallel-moe-compaction-production.json)).

The next target-only kernel is **WPF-H7V dequantized-Q6 H6U full-batch/live-
tail** in `quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}`. The current local128
DPP body has two `row < rows` predicates inside every exact role and one outer
row/output guard. At M512, rowbatch4 has 128/128 full groups; each rowbatch5
call has 102 full groups plus one remainder-2 group. Weighted across 2/46/94
calls, **1,757,184/1,763,328 workgroups (99.652%)** are full.

H7V must keep one predicate-free full kernel plus the unchanged H6U kernel as
its rowbatch5 tail/fallback. Expected source topology is 142 H7V full launches,
96 H6U tail launches, unchanged 142 activation packs and 143 exact Q6-to-F32
producers, and **2,382 request dispatches**. The full object must retain
local128/wave32, LDS **1,024/1,536 B**, exact 64/80 FMA and permlanex16/DPP
counts, one barrier, serial four-wave sum/store, and private/spill/scratch0,
with no row-bound compare and no worse code/slots/VGPR than H6U. Freeze RED and
all-role timing before implementation; no candidate has run
([H7V target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7u-q6-full-batch-live-tail-target.json)).

Reject **WPF-H7V** after its first immutable all-role screen. The predicate-free
object passes every declared physical gate: BF16-r4 is **5,808 B / 873 slots /
VGPR108**, BF16-r5 **6,960 / 1,001 / 139**, and F32-r5 **6,928 / 996 / 139**,
all no worse than H6U and exact at **64/80 FMAs, 64/80 permlanex16, 256/320 DPP
adds, one barrier, LDS1,024/1,280, private/spill/scratch0**. GREEN passes
**9/9**. Full-request rocprof names exact **142 packs + 143 producers + 142 H7V
full + 96 H6U tail consumers**, **2,382 dispatches**, one queue/stream, runtime
VGPR112/144 and LDS1,024/1,536, scratch0, with zero compiler.

Complete outputs are byte-exact/finite/lifecycle-clean. The r4 role improves
**1.00255x/1.00495x**, but BF16-r5 is **0.97542x/0.97618x** and F32-r5
**0.97309x/0.97499x** event/wall. Weighted H6U→H7V therefore regresses
**47.949→48.680 ms event (0.985x)** and **48.522→49.162 ms wall (0.987x)**.
Delete every H7V body/export/wrapper/key/RED surface, retain H6U and **437.189
tok/s**, and forbid r4-only salvage, rewrite, recompile, or favorable rerun
([H7V rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-q6-full-batch-live-tail-rejected.json)).

Reject **WPF-H7W exact H6T output-partition P128 crossover** in
`quant/gguf_iq_selected_prefill.{hip,py}` after the sole immutable timing
screen. Its one **469,056-byte** first repository object passes every frozen
physical bound. P128 and H6T P256 both compile to **7,920 B / 1,384 slots /
VGPR101 / SGPR78 / physical LDS384 / private0 / spill0** and identical **216
useful FMAs, 24 permlanex16, 96 DPP adds, 24 LDS b128 loads, 12 LDS stores, and
two barriers**. GREEN passes **12/12** across rows1/7/8/9/M512, P64/P65,
partition/routing tails, H6T/CPU bytes, poison/repeat/finite/lifecycle, strict
shape rejection, source policy, and backend isolation.

Cache-only rocprof names exact **45 H7W P128 + two unchanged IQ4** calls at
local128/grid16,384×64/runtime-VGPR104/LDS512/scratch0, **2,286 application
dispatches**, one queue/stream, and zero compiler. All 45 actual-weight outputs
are byte-exact and allocation-clean, but only **16/45** layers win both clocks.
H6T P256→H7W P128 moves event **260.663→261.392 ms (+0.280%, 0.99721x)** and
synchronized wall **260.731→262.135 ms (+0.538%, 0.99464x)**. The modeled
50% workgroup and activation-record reductions do not translate into speed.

Remove the H7W HIP export, Python symbol/wrapper/registry key, gfx1151
exclusion, and RED. Retain H6T P256 as source/default and production **437.189
tok/s**; run no runtime/source qualification. Do not salvage
layer/expert/routing/prompt/length subsets, retune output partitions, rewrite
the body, recompile, or favorably rerun this family
([H7W rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-iq3-output-p128-rejected.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7v-iq3-output-p128-target.json)).

Select target-only **WPF-H7X exact H6W one-slot BF16 K/V software pipeline** in
`attention/laguna_kv_attention.hip`. Current H6W
`laguna_swa_attention_prefill_qrow4_dense_initial_global_score_replay_exact_bf16_kernel`
is **72 calls / 62.656 ms**, **54.198%** of attention. Its current physical
shape is local32/wave32, **4,984 B / 871 slots / VGPR54 / SGPR40 / LDS0 /
private0 / spill0**; runtime reports VGPR56/LDS0/scratch0. ISA has eight static
BF16 load sites and eight `vmcnt` waits: each four-load K or V clause drains
`vmcnt(3→0)` before current-slot QK or PV work.

H7X may add only one separately named gfx1100 H6W-equivalent body/export/wrapper/
registry key and one gfx1151 exclusion. Preload the initial K/V slot and rotate
one alternate four-BF16 register set so next-slot loads overlap current exact
arithmetic. Do not change masking, score-record bytes/order, max/exp/denominator
order, broadcasts, PV order, divide, output, ABI, spans, allocation/workspace,
dispatches, package owner, or H6W fallback. At natural M512, both K and V have
**63,866,880 / 64,032,768 (99.7409%)** prefetchable slots per request.

Before build, freeze starts256/384 and fail-closed starts0/128/wrong-shape RED.
The sole first object must be local32/wave32, code≤8,000 B, slots≤1,400,
metadata/runtime VGPR≤64/64, metadata SGPR≤64, LDS/private/spill/scratch0, no
barriers, and must show next-slot K and V loads before substantial current-slot
QK/PV work and delayed waits without premature use. GREEN must prove complete
H6W and independent CPU bytes, exact score records, poison/repeat/finiteness,
immutable spans, and lifecycle. Cache-only tracing must name **72 H7X + 72 H6A
+ 24 H6N + 24 H6Z**, **2,286 application dispatches**, one queue/stream, and
zero compiler. Consume one all-72 actual-state 5/15/5 screen only after those
gates; starts256, starts384, and the weighted aggregate must each win event and
wall. On any miss remove every H7X/RED/exclusion surface without subset,
prefetch-distance retune, rewrite, recompile, or favorable rerun. Runtime/source
qualification remains separate; no candidate or H7X speed claim exists
([H7X target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7w-swa-kv-prefetch-target.json)).

Reject **WPF-H7X** from `attention/laguna_kv_attention.hip` at first-object
physical analysis. The sole **674,768-byte** object has H7X at **5,320 B / 931
slots / VGPR54 / SGPR44 / LDS0 / private0 / spill0** and unchanged H6W at
**4,984 B / 871 / VGPR54 / SGPR40**. H7X remains under every resource ceiling
and preserves exact **32 bpermutes / one b128 score-record load+store / four exp
/ 56 FMA / 41 FMAC / 16 output stores / zero barriers or scratch**.

Codegen emits four four-u16 clauses: K prologue, K steady-next, V prologue, and
V steady-next. Both steady-next clauses are `4 × global_load_u16` followed
immediately by `s_waitcnt vmcnt(3)`, then the rest of the drain; there are
**zero instructions** between final load and first wait. The required K/QK and
V/PV software-pipeline overlap is absent, so resource cleanliness cannot admit
the leaf. Consume no candidate correctness, trace, timing, runtime, or source
gate. Delete the body/export/Python symbol+wrapper+key/gfx1151 exclusion and
RED; retain exact H6W and **437.189 tok/s**. Do not rewrite, recompile, retune
prefetch distance, salvage any subset, or favorably rerun
([H7X rejection](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-swa-kv-prefetch-physical-rejected.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7w-swa-kv-prefetch-target.json)).

Select target-only **WPF-H7Y exact H6W lane-major BF16 K/V mirror loads** in
`attention/laguna_kv_attention.hip`. Production remains **437.189 tok/s /
1,153.347 ms / 2,286 dispatches**, **1.58007x** behind matched llama.cpp HIP.
H6W remains **72 calls / 62.656 ms**. Its natural head bytes are logically
`[part4][lane32]` for this wave-owned reduction: every K or V pass emits four
coalesced `global_load_u16` sites and four staged `vmcnt` waits so lane *l*
receives dimensions `l + 32p`.

H7Y may add only one separately named gfx1100 H6W-equivalent leaf consuming
caller-provided `[lane32][part4]` mirrors. The exact mapping is
`mirror[base + lane*4 + part] = natural[base + part*32 + lane]`; each 8-byte
lane record is naturally aligned. One `uint2`/`global_load_b64` must recover
the same four BF16 bits in pass one and independently in pass two, before the
unchanged ordered QK reduction, unscaled score record, max, fused scale-minus-
max, exp/denominator, lane-0 broadcast, token-order PV, divide, and stores.
No GQA-head sharing, key split, changed association, or H7X prefetch scheduling
is reopened.

Across starts256/384, 32 qrow4 groups, 72 heads, and 36 SWA layers, each pass
executes **64,032,768** logical wave slots. H6W's eight K+V load sites model
**512,262,144** dynamic load-issue slots; H7Y's required two b64 sites model
**128,065,536**, removing **384,196,608 (-75%)** plus the same number of wait-
issue slots. The **32,784,777,216-byte** logical attention payload is unchanged.
This is an operation model only, not physical traffic or speed evidence.

Freeze RED before code. The sole first object must contain exactly **2
`global_load_b64`, 0 `global_load_u16`, ≤2 vmcnt waits**, unchanged one b128
score-record load/store, 32 bpermutes, four exp, 56 FMA, 41 FMAC, 16 output
stores, barrier0, and local32/wave32/code≤5,200 B/slots≤900/VGPR≤56/SGPR≤48/
LDS-private-spill-scratch0. Prove the 128-element transpose round-trip,
starts256/384 complete H6W+CPU output, records, poison/repeat/finiteness,
immutable `KVLiveSpans`, and lifecycle. Then consume one all-72 actual-state
5/15/5 screen with mirrors prepared outside timing; each start and aggregate
must win event and wall. Any miss removes H7Y without subset, packing-width
retune, rewrite, recompile, or favorable rerun.

H7Y passes every standalone gate and remains a separately registered explicit
leaf. The sole object is **4,900 B / 855 slots / metadata VGPR54 / SGPR40 /
spill0**, with exactly **2 `global_load_b64`, 0 `global_load_u16`, 2 vmcnt
waits** and unchanged b128-record/bpermute/exp/FMA/FMAC/output-store counts.
GREEN passes **6/6**. An actual-cache trace names exact **72 H7Y** calls at
local32/grid2304x32/runtime-VGPR56/LDS0/scratch0 on one queue with zero compiler.
All 72 actual-layer outputs and score planes are byte-exact.

The immutable all-72 H6W→H7Y screen improves start256 **23.739→23.681 ms event
/ 23.719→23.703 wall**, start384 **32.868→32.577 / 32.840→32.614**, and
aggregate **56.607→56.259 ms event (-0.616%) / 56.559→56.317 wall (-0.428%)**.

The separately RED-gated bounded runtime owner now qualifies default-off. It
allocates exact **72 MiB / 72 buffers** for SWA K/V mirrors and replaces each
natural SWA writer with one fused natural+lane-major writer, adding no dispatch.
The first writer object is **1,724 B / 357 slots / metadata VGPR23 / SGPR53**,
with exactly two F32 loads, four BF16 stores, and LDS/private/spill/scratch0;
the retained natural writer is byte-identical. Focused GREEN is **7/7**.
Complete M512 is KL0 and byte-exact across all **48/48** boundaries, logits,
K/V/spans, repeat, and lifecycle.

The named request is exact **144 fused writers + 72 H7Y + 72 H6A + 24 H6N + 24
H6Z / 2,286 dispatches**, one queue/stream and zero compiler. Runtime resources
are H7Y local32/grid2304x32/VGPR56/LDS0/scratch0 and writer local256/
grid1024x128/VGPR24/LDS0/scratch0. Writer-inclusive fixed C4096/M512 improves
**436.120→436.785 tok/s (+0.152%)**; clean 512/1K/4K medians improve
**+0.0530%/+0.1217%/+0.0043%**.

Selector-unset source promotion is rejected at its first binding fixed median.
Complete source M512 remains KL0/byte-exact, but H6Z/H6W rollback→H7Y source
moves **436.403→436.275 tok/s (-0.0294%, 0.99971×; 2/5)**. Run no later source
length/trace/post-commit gate and allow no favorable rerun. Restore H6Z/H6W
production **437.189 tok/s** and retain the bounded H7Y owner default-off
([source rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-source-rejected.json) ·
[runtime](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-runtime-candidate.json) ·
[standalone](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-swa-lane-major-cache-candidate.json) ·
[target](../benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-post-h7x-swa-lane-major-cache-target.json)).

No kernel is added for the post-H7Y Q5/attention repair audit. H5A SGEMM's
BF16 mismatch density is only **0.0914%**, but complete candidate-distance
coverage is **100%** and its F32 mismatch density is **96.737%**. The retained
H2 source-F16-WMMA attention leaf changes **21.276%** of post-gate BF16 values
and touches **99.908%** of current exact qrow4/head workgroups. Even ideal
linear exact repair plus the retained **20.971-ms** candidate is **136.255 ms**,
slower than current H6N/H6Z/H6A/H6W **115.385 ms** before detection/queue/merge
cost. Keep both leaves diagnostic-only, add no repair kernel/runtime owner, and
do not treat single-prompt mismatch sparsity as a dispatch policy
([repair-audit rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-attention-repair-audits-rejected.json)).

No new device body is selected for **WPF-H8A exact resident global-Q5 F32
cache**. The target reuses the current exact tile-K-col16 producer at owner
setup for all 12 full-attention `attn_q` and 12 `attn_output` tensors, publishing
an immutable 24-entry raw-pointer→F32-plane map. Each plane is **75,497,472
bytes**; total ownership is **1,811,939,328 bytes (1.6875 GiB)**. Request-time
execution retains the exact activation pack and H7G padded-compute consumer but
removes the measured **24 producer calls / 5.596 ms**, modeling **2,286→2,262
dispatches**. A stricter owner+child dummy-allocation run completes exact M512
with token2930, **4.167 GB** free, zero compiler, and lifecycle recovery.

Freeze Python owner/registry RED while pinning the HIP source and cached object
unchanged. Require all-or-nothing allocation/sharing and reverse teardown,
complete bytes for all 24 actual planes, rows1/7/8/9/M512 H7G output identity,
complete state, exact **24 setup / zero request producer** trace, and positive
fixed plus 512/1K/4K medians before any default-off admission. The existing
pack+transient producer+H7G chain is the mandatory unfused fallback; gfx1151,
partial maps, wrong shapes, and unavailable memory fail closed. Full-family,
role-only, layer, prompt, token, and favorable-rerun subsets are forbidden
([H8A target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-resident-q5-global-f32-cache-target.json)).

H8A now qualifies as a bounded default-off Python owner with no new device body,
JIT key, or changed HIP source/object. Its two registered resident composites
reuse the exact tile-K-row pack and H7G coltile16 padded-compute primitives.
The weight-owning session publishes one immutable **24-entry / 1,811,939,328-
byte** raw-pointer map after all retained producers finish; explicit children
share it read-only and never own/free it. Every actual plane matches a fresh
producer over all **75,497,472 bytes**. Complete M512 and all **48/48** hidden
boundaries/logits/KV/spans/repeat are exact with lifecycle recovery.

The named trace records **24 setup producers** before requests and exact **zero
request producers + 24 target packs + 24 H7G consumers / 2,262 application
dispatches** on one queue/stream with zero compiler. Setup is excluded from
request timing. Fixed C4096/M512 improves **436.765→438.368 tok/s (+0.367%,
5/5)** and clean 512/1K/4K medians improve **+0.748%/+0.332%/+0.257%**, each
3/3. At this bounded checkpoint source remained false pending its separate RED;
gfx1151 and every partial/wrong-shape/allocation-failed route remain fail-closed
([H8A runtime](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-q5-global-f32-cache-runtime-candidate.json) ·
[H8A target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h7y-resident-q5-global-f32-cache-target.json)).

H8A now owns the gfx1100 source path after selector-unset complete-plane/state,
fixed, length-transfer, topology, and clean committed production gates. Fixed
transient-H7G rollback→source improves **435.272→437.286 tok/s (+0.463%, 5/5)**;
clean 512/1K/4K improves **+0.290%/+0.142%/+0.215%**, all 3/3. Clean commit
`c4ea62347` reaches **440.353 tok/s**, and five exact profiled requests record
**2,262 dispatches**, **1,151.215-ms** representative sum / **1,174.598-ms**
median span, exact **24 setup / zero request coltile16 producers**, one
queue/stream, and zero compiler. Retain transient H7G as explicit opt-out and
all fail-closed fallbacks; source promotion changes no device body or object
([H8A production](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-q5-global-f32-cache-production.json)).

No new device body is selected for **WPF-H8B exact scoped activation-pack
reuse**. An execution-unchanged natural-M512 audit records **330** H5Y/H6U
BF16 tile-K-row pack calls and finds **95** consecutive equal-key immutable
runs with **107** redundant calls. The complete classes are 12 full-attention
Q/K/V triples (**24 removed**), 35 SWA K/V pairs (**35**), 46 shared-Q5 gate/up
pairs (**46**), one dense-Q5 gate/up pair, and one layer-47 shared-Q6 gate/up
pair. Token2930/position511, finite complete state, lifecycle, and zero-compiler
checks pass.

Freeze one scope-local producer cache before implementation. Its key is exact
input/activation pointers, rows, K, row batch, and stream; publication occurs
only after a successful pack. Scope exit, a changed key/stream, producer
failure, disabled policy, wrong shape, c=1, non-M512, registry miss, and backend
miss all execute the retained producer. The HIP source/object, pack body,
H7G/H7H/H6U consumers, F32 producers/resident planes, allocation, workspace,
and arithmetic remain unchanged. The measured profile model removes
**2.342313 ms** and changes **330→223 packs / 2,262→2,155 dispatches**; the
zero-cost wall ceiling **441.242 tok/s (+0.202%)** is not a speed result.
Require RED-first scope/failure/all-95-run exactness, complete M512 state,
cache-only named topology, fixed and 512/1K/4K positive medians, then a separate
source-default gate. No attention/shared/layer/role/prompt/length subset or
favorable rerun is admissible
([H8B target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-activation-pack-reuse-target.json)).

H8B now qualifies bounded default-off. The backend-neutral one-entry scope
cache and complete runtime ownership pass **103/103** retained tests without a
new compiler process; the HIP source/object and every pack/dequant/compute body
remain unchanged. Complete M512 executes exact **223 packs (24 resident + 199
transient)** and preserves token2930/position511 plus every frozen state digest.
The committed cache-only trace proves **330→223 packs / 2,262→2,155
application dispatches**, removed geometry `{r4:46,r5:60,r12:1}`, unchanged
non-pack kernel names/counts, expected H8A producer/consumer resources, one
queue/stream, and zero compiler.

Fixed C4096/M512 improves **438.412→438.919 tok/s (+0.116%, 4/5)**. Clean
512/1K/4K improves **406.770→407.374 (+0.148%)**, **321.625→322.189
(+0.175%)**, and **198.586→198.888 tok/s (+0.152%)**, all exact and
lifecycle-clean. Keep the source capability false and H8A production at
**440.353 tok/s** until a separately frozen source-default gate passes
([H8B runtime](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-runtime-candidate.json) ·
[H8B target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-activation-pack-reuse-target.json)).

H8B now owns the gfx1100 source path after selector-unset state, fixed,
length-transfer, topology, and clean committed production gates. Fixed disabled
rollback→source improves **438.114→439.243 tok/s (+0.258%, 5/5)**; clean
512/1K/4K improves **+0.109%/+0.0097%/+0.055%**. Clean commit `6b9411b15`
reaches **440.893 tok/s**, and five exact profiled requests record **2,155
dispatches**, **1,146.420-ms** median sum / **1,166.621-ms** span, one
queue/stream, and zero compiler. Retain explicit disabled full-pack rollback;
source promotion changes no device body, JIT object, allocation, workspace, or
output byte
([H8B production](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-scoped-activation-pack-reuse-production.json)).

WPF-H8C's exact dual-weight shared-Q5 leaf passes physical admission at
local128, metadata/runtime VGPR **134/136**, LDS **1,024 B**, and zero
private/scratch/spills. Its ISA contains one 64-bit activation-record load for
two independent weight streams, and rows17/33/M512 gate/up outputs are
BF16-byte exact to H7H plus sampled CPU references. The cache-only named trace
records the intended symbol once at **808.921 µs**, with two producers, one
pack, one queue/stream, and zero compiler activity.

Do **not** register or port this leaf. The required actual-weight screen rejects
it before runtime ownership: all **46/46** Q5 shared gate/up pairs remain exact
and finite, but only **14/46** win both clocks. Summed H7H→H8C consumer event
time moves **27.8051→27.8323 ms (0.9990×)** and synchronized wall moves
**28.0210→28.0053 ms (1.0006×)**. Per the frozen no-subset/no-rerun rule, the
HIP body, wrapper/key, gfx1100 capabilities, gfx1151 exclusion, and RED test are
removed; H8B/H7H remains the sole source path
([H8C rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-shared-q5-dual-consumer-rejected.json) ·
[H8C target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8b-shared-q5-dual-consumer-target.json)).

Do **not** add a complete-class Q6 exact-value F32-SGEMM composite as
**WPF-H8D**. The retained local64 exact Q6-to-F32 producer, cast kernels, and
rocBLAS SGEMM are individually valid, but the required all-six-shape/**144-call**
M512 screen fails before target publication. Five shapes win both clocks and
the weighted diagnostic improves **74.099→40.969 ms event (1.809×)** and
**74.469→41.232 ms wall (1.806×)**. The F32 K3072×N72 role instead regresses
**0.4325×/0.4456×** event/wall. Exact complete activation operands, sampled
real-weight operands, primitive quality, lifecycle, and zero-compiler gates
pass, but they cannot waive the complete-class timing rule. Add no Q6-SGEMM
registry key, owner, capability, or test; skip the 576-step gate and forbid
post-screen 143-call salvage
([H8D rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-k-f32-sgemm-complete-class-rejected.json)).

Do **not** restore gfx1100's removed packed-F32 attention package map as
**WPF-H8E**. The existing backend-neutral hipBLASLt owner and four heuristics
per QK/PV contraction remain valid standalone machinery, but a source-unchanged
all-**128-combination** screen finds four numerical classes per global/SWA
shape. A fixed synthetic closest-output rule selects six complete shape keys,
changes five H5B mappings, and remains faster at every selected shape; matched
C4096/M512 is finite/deterministic at KL **0.000231** and token2930.

That primitive evidence cannot waive full-model quality. All **10,512** expected
changed-association stacks run across 18 prompts/**576 teacher-forced steps**,
but maximum KL is **0.391103** versus **0.05**, despite **563/576** top-1 and
diagnostic **1.3086×** prefill. Keep gfx1100 without
`LAGUNA_PREFILL_ATTENTION_HIPBLASLT*` capability/map ownership, add no RED or
registry/runtime surface, and prohibit a post-result alternate-class rerun.
H6N/H6Z/H6A/H6W exact attention remains production
([H8E rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-quality-selected-f32-attention-algorithms-rejected.json)).

No new device body is selected for **WPF-H8F exact resident shared-Q5 F32
cache**. Reuse the retained coltile8/rowbatch4 producer once at owner setup for
all **92** layer1–46 `ffn_gate_shexp`/`ffn_up_shexp` raw-Q5 K3072×N1024
tensors, then publish those planes through the existing raw-pointer map to the
unchanged H7H consumer. H8B's 46 shared activation packs remain unchanged. The
clean trace measures exactly 92 targeted producer launches/request at
**3.439745 ms median**; request dispatches model **2,155→2,063**.

The live target audit holds the exact **92 allocations / 1.078125 GiB** beside
H8A and M512, reaches token2930/position511 with finite logits and clean
lifecycle, and leaves **2.802734 GiB** free. Extend the H8A map all-or-nothing
from **24→116** planes; a shared-class miss/failure frees every new buffer and
retains the 24-plane H8A plus registered producer/H7H fallback. Freeze complete
plane bytes, CPU edge values, rows1/7/8/9/M512 output identity, full state,
named setup/request topology, fixed and 512/1K/4K both-clock gates before any
source attempt. HIP source/object changes and partial-plane/layer/role/prompt/
token/route/length/favorable-rerun salvage are forbidden
([H8F target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8e-resident-shared-q5-f32-cache-target.json)).

Do **not** retain the H8F resident wrapper/owner. Reusing the admitted
coltile8 producer and H7H device body is mathematically and topologically valid:
focused tests pass **6/6**, all **116/116** planes match fresh producers, M512 is
KL0/exact through all **48** boundaries and KV state, and the named request has
**0 shared producers / 46 packs / 92 H7H consumers / 2,063 dispatches** at the
retained local128/VGPR72/LDS512/scratch0 consumer resources. Fixed M512 wins
**+0.1162% (5/5)**, but the binding first clean 512/1K/4K medians are
**+0.3421%/-0.0710%/-0.00171%**. Remove the Python composite/key, plane-kind
extension, 92-plane owner, gfx1100 capabilities, gfx1151 exclusion, and RED
test under the no-length-subset/no-rerun rule. The HIP source/object remains
unchanged and H7H/H8B stays production
([H8F rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-resident-shared-q5-f32-cache-rejected.json)).

Do **not** transfer the existing dense-initial global qrow6 body to gfx1100 as
**WPF-H8G**. The registered primitive remains valid for its qualified sibling
RDNA3 route, but a cached complete W7900 starts128/256/384 screen fails both
current-output identity and all-class timing. Qrow6 wins start128
**1.0492×/1.0362×** event/wall, yet differs from H6N/H6Z by
**730,971/742,825/749,888 F32 bits**, and starts256/384 lose both clocks. The
weighted 36-call route regresses **15.869→21.545 ms event (0.7365×)** and
**16.078→21.803 ms wall (0.7374×)**. Keep start0 and all measured starts on the
current H6N/H6Z schedule; add no gfx1100 capability, dispatch branch, owner, or
RED, and forbid favorable start128-only salvage
([H8G rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-global-qrow6-transfer-rejected.json)).

Do **not** retain **WPF-H8H exact prefill attention+softplus dual publication**.
Its four first objects preserve every H6N/H6Z/H6A/H6W output bit: cache-only
all-start tests are F32-context and BF16-gate exact, complete `KVLiveSpans` is
unchanged, lifecycle recovers, and cached tracing names every candidate with
positive duration and zero compiler. The retained control bodies are source-
identical.

The physical gate fails before timing. Runtime VGPR is H6N **40≤48**, H6Z
**88>56**, H6A **80>72**, and H6W **80>64**; local sizes remain
**256/32/32/32**, with zero LDS/scratch. The immutable policy forbids resource
rewrite or favorable route subset, so remove all four bodies, C/Python wrappers,
registry keys, gfx1151 exclusions, and RED coverage. Keep the registered
unfused H6N/H6Z/H6A/H6W plus standalone softplus chain; add no owner,
capability, allocation, workspace, or source policy
([H8H target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8g-prefill-attention-softplus-dual-publication-target.json) ·
[H8H rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-prefill-attention-softplus-dual-publication-physical-rejected.json)).

**WPF-H8I exact stream-ordered Q5 partition accumulation is target-only.** It
must add separately registered local32 four-stage siblings for all six current
H7G/H7H Q5 roles, not alter the retained controls. Partition `p`, lane `i`
visits `p*32+i+128*n`; stage 0 publishes `0.0f + partial`, stages 1/2 load-add-
store the next partial, and stage 3 retains the exact BF16/F32 publication.
This preserves all **20,085,760** compute waves and every scalar/reduction/final
sum association while replacing **5,021,440** local128 workgroups/barriers.

The physical trade is binding: four launches per consumer produce **752**
partition launches and **2,155→2,719** application dispatches, plus **7.546875
GiB** extra M512 global traffic and ≤**24 MiB** accumulation storage. Runtime
may only borrow aligned inactive request scratch; standalone tests may allocate
explicitly. Freeze strict shape/pointer/preflight before HIP loading, gfx1151
fail-closed, local32/LDS0/private-spill-scratch0 and current+8 VGPR ceilings,
exact partition/final-output and named-trace gates, then require every role and
the weighted **188-call** aggregate to win event and synchronized wall. Keep
registered H7G/H7H plus producer/pack fallback; forbid any subset, resource
rewrite, recompile, or favorable rerun
([H8I target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8h-streamed-q5-partitions-target.json)).

Do **not** retain H8I. The complete first object passes: all **24** stage
specializations are exact at partition and final-output boundaries, local32,
LDS/private/scratch/spill-free, and within role VGPR ceilings at runtime maxima
**80/200/168/200/168/168**. Its physical validity does not transfer to speed.
Every actual-weight role loses both event and synchronized wall; the weighted
188-call aggregate regresses **222.555→289.013 ms event (+29.861%)** and
**225.438→286.922 ms wall (+27.273%)**. Remove the HIP body/exports, Python
wrappers/keys, gfx1151 exclusions, and RED test. Keep H7G/H7H plus exact
producer/pack fallback and add no owner, allocation, capability, or source map
([H8I rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-stream-ordered-q5-partitions-rejected.json)).

**WPF-H8J exact IQ3 four-workgroup occupancy is target-only.** Retained H6T
owns **45 calls / 264.377 ms**, **96.783%** of current IQ-down, at local128,
metadata/runtime VGPR **101/104**, LDS **384/512**, and scratch0. RDNA3's
1,536-VGPR SIMD budget gives 14 register-limited waves, but four waves per
workgroup round this down to **3 complete workgroups / 12 resident waves**.
A separately named H6T-equivalent sibling fixes only `launch_bounds(128,4)` and
must compile at **≤96 VGPR** to admit **4 workgroups / 16 waves (+33.333%)**.
This is an occupancy model, not speed evidence.

Freeze RED before executable changes. Preserve all 23 global loads, 216 FMAs,
24 permlanex16, 96 DPP adds, 24 LDS b128 loads, 12 LDS stores, two barrier
sites, serial wave sum, BF16 bytes, raw ABI, H6T fallback, allocation/workspace,
and dispatch count. Require metadata/runtime VGPR≤96, LDS384/512,
private/spill/scratch0, complete boundary/CPU/all-45 bytes, and a named
compiler-free trace. The single immutable 5/15/5 actual-routing screen must win
every layer and the aggregate on event and synchronized wall. Do not sweep
launch bounds or salvage any layer/expert/prompt/length/rewrite/recompile/rerun
subset
([H8J target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8i-iq3-four-workgroup-occupancy-target.json)).

**WPF-H8J is rejected at its first-object physical gate.** The sole AMD-clang22
build emits H6T and H8J as relocation-normalized identical **7,920-byte /
1,384-slot** bodies with the same **23 global loads / 216 FMAs / 24
permlanex16 / 96 DPP adds / 24 LDS b128 loads / 12 LDS stores / 2 barriers**.
Both metadata records remain **VGPR101 / SGPR78 / LDS384 / private0 / spill0 /
wave32**. Thus `launch_bounds(128,4)` does not cross the required ≤96-VGPR
boundary: 1,536/101 permits 15 register waves but only **3 complete four-wave
workgroups**. Apply the frozen no-resource-rewrite/no-rerun rule, skip runtime
trace, correctness, and timing, remove the candidate and RED, and retain H6T
production
([H8J rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-four-workgroup-occupancy-physical-rejected.json)).

**WPF-H8K exact IQ3 uniform-rowbatch4 triple-output ownership is target-only.**
Replay the immutable all-45 natural routing: **230,400 useful rows** map under
H6T rowbatch8 to **24,650 full + 8,897 tail = 33,547 epochs**, **268,376
compute slots**, and **37,976 padded slots (14.150%)**. Fixed rowbatch4 maps the
same counts to **53,907 full + 7,639 tail = 61,546 epochs**, **246,184 slots**,
and **15,784 padded slots (6.411%)**. It therefore removes **22,192 slots
(−8.269%)**, **2,454,257,664 modeled FMA-wave operations**, and
**1,363,476,480 exchange-wave operations**, but adds **28,670,976 output-triple
and 57,341,952 barrier epochs**. These are source-operation counts, not speed.

The one separately named gfx1100 body must keep H6T local128/P256/four-wave/
three-output ownership, raw ABI, active-expert traversal, each row's exact IQ3
decode/eight ordered FMAs/permlanex16+DPP tree/serial wave sum/BF16 store, and
all registered fallbacks. Only uniform row ownership changes **8→4**. The
explicit resident activation plus per-scope accumulator model falls **40→20
dwords**, but compiler output must independently prove metadata/runtime VGPR
**≤96**, LDS **≤192/256 B**, code/slots **≤7,920/1,384**, and zero
private/spill/scratch. Freeze RED first and consume one all-45 5/15/5 both-clock
screen. Do not compile alternate row batches or salvage a layer, expert,
routing, prompt, token, length, output partition, rewrite, recompile, or
favorable rerun
([H8K target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8j-iq3-rowbatch4-triple-output-target.json)).

**WPF-H8K is rejected at its frozen named-trace resource gate.** Complete
rows1/3/4/5/7/8/9/M512, P64/P65, uneven/empty, H6T/CPU bytes, poison, repeat,
and lifecycle pass. The sole immutable build passes first-object metadata and
ISA at **VGPR70 / SGPR58 / LDS192 / private/spill0**, **4,916 B / 882 slots**,
**19 global loads / 108 FMAs / 12 permlanex16 / 48 DPP adds / 12 LDS b128
loads / 6 dual-address stores / 2 barriers / scratch0**.

A cache-only natural-M512 selected-region trace preserves exact token2930 and
all logits/hidden/KV digests, names **45 H8K / 0 H6T / 2 IQ4** calls within
**2,155 application dispatches**, one queue/stream, and zero compiler. Runtime
resources are local128/grid32768×64/**VGPR72/LDS512/scratch0**. The sole failure
is runtime LDS **512 > frozen 256 B**. The first profiler attempt emitted zero
rows because the system roctx lacked profiler resume/pause; the exact cached
retry used the existing SDK roctx override without source, object, or resource
change. Apply the no-resource-gate-rewrite/no-rerun rule: skip the all-45 5/15/5
timing and all owner/state/length/source gates, remove candidate plus RED, and
retain H6T production
([H8K rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-rowbatch4-triple-output-runtime-lds-rejected.json)).

**WPF-H8L exact IQ3 lossless 12-bit codebook packing is target-only.** H6T's
read-only 256-entry `iq3xxs_grid` stores four magnitudes per uint32, and the
complete value alphabet is **4/12/20/28/36/44/52/62**. Encode each coordinate
as a 3-bit code in one uint16 and reconstruct exactly with
`magnitude = 4 + 8*code + 2*(code == 7)`. Independent enumeration proves all
**256/256** entries reconstruct and all 256 packed words remain unique. Table
storage falls **1,024→512 bytes** without moving the table into LDS.

Across the immutable natural-M512 all-45 routing, **103,056,384** segment
decodes model **105,529,737,216→52,764,868,608 logical table bytes (−50%)** at
the same **824,451,072** wave loads. This is an address-width operation count,
not a cache-traffic or speed claim. The one separately named H6T-derived body
must retain local128/grid32768×64, P64/P256, rowbatch8/triple-output ownership,
raw weight and active-expert ABI, **216 ordered FMAs / 24 permlanex16 / 96 DPP
adds**, serial wave sums, **24 LDS b128 loads / 12 stores / two barriers**, and
BF16 bytes. Its sole table delta is fixed **8 b128 + 9 b32 + 6 d16 → 8 b128 +
3 b32 + 12 d16** plus 24 code extracts.

Freeze RED first. Require all 256 table values plus rows1/7/8/9/M512, P64/P65,
empty/uneven/reordered routing, sampled CPU, and all **45/45** actual layers.
The first object must be VGPR≤128, LDS384/512, code≤10,000 B, slots≤1,800, and
private/spill/scratch0; then one 5/15/5 all-layer screen must win every layer
and aggregate on event and synchronized wall. Do not compile another table
width/formula/load source or salvage any layer/expert/routing/prompt/token/
length/body/recompile/favorable-rerun subset
([H8L target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8k-iq3-codebook12-target.json)).

**WPF-H8L is rejected by its first and only all-45 timing screen.** Complete
all-entry, rows1/7/8/9/M512, P64/P65, uneven/empty, H6T/CPU, poison, repeat,
and lifecycle checks pass. Its sole object realizes **8 b128 + 3 b32 + 6
d16_b16 + 6 u16**, **216 FMAs / 24 permlanex16 / 96 DPP adds / 24 LDS b128
loads / 12 stores / two barriers**, at **8,740 B / 1,523 slots / VGPR111 /
SGPR78 / LDS384 / private/spill/scratch0**. The exact-state selected-region
trace records **45 H8L / 0 H6T / 2 IQ4**, 2,155 application dispatches, one
queue/stream, and runtime **VGPR112/LDS512/scratch0**.

The binding 5/15/5 screen is byte-exact on every actual layer, but **0/45** win
both clocks. H6T→H8L aggregate event moves **260.043597→290.495983 ms
(+11.710493%, 0.895171×)** and synchronized wall moves
**260.756712→290.437301 ms (+11.382483%, 0.897807×)**. Apply the frozen no-
width/formula/load-source/layer/rewrite/recompile/rerun rule: add no owner or
source policy, remove candidate plus RED, and retain H6T production
([H8L rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-codebook12-all45-timing-rejected.json)).

**WPF-H8M exact IQ3 sign-folded BF16 codebook is target-only.** Current H6T
loads six divergent uint32 unsigned-grid entries across its triple-output body,
converts their 24 magnitudes, then applies signs through exactly **24
`v_cmp_eq_u32` + 24 `v_cndmask_b32`** sites. Independently enumerate the fixed
index `(sign_nibble << 8) | grid_index`: all **4,096/4,096** uint64 records are
exact and unique, with four little-endian signed BF16 magnitudes per record and
SHA-256 `fd6a3253...eb90f`.

This deliberately trades bytes for ALU. Read-only table storage grows
**1,024→32,768 bytes**. Across the immutable **103,056,384** all-45 segment
decodes, wave loads remain **824,451,072**, while modeled logical table bytes
grow **105,529,737,216→211,059,474,432 (+100%)**. These are operation/address-
width counts, not cache traffic or speed. The fixed candidate must change H6T's
static loads **8 b128 + 9 b32 + 6 d16_b16 → 8 b128 + 3 b32 + 6 b64 + 6
d16_b16**, expand the 24 BF16 values to identical F32 bits, and remove every
sign compare/select site. It must not change scale decoding, **216 ordered dot
FMAs / 24 permlanex16 / 96 DPP adds**, serial wave sums, **24 LDS b128 loads /
12 stores / two barriers**, raw ABI, P64/P256, rowbatch8/triple-output ownership,
or BF16 outputs.

Freeze RED before executable work. Compile one table dtype/layout/index/body
once. Require all records plus rows1/7/8/9/M512, P64/P65, empty/uneven/reordered
routing, H6T and sampled CPU bytes, poison/repeat/lifecycle, and all **45/45**
actual layers. The first object must have exactly six b64 table loads, zero sign
cmp/cndmask sites, code≤8,500 B, slots≤1,500, metadata/runtime VGPR≤101/104,
LDS384/512, and private/spill/scratch0. Then one 5/15/5 all-layer screen must
win every layer and aggregate on event and synchronized wall. Do not compile
signed-int8/F16/F32/split/LDS alternatives or salvage any cache placement,
layer/expert/routing/prompt/token/length/body/recompile/favorable-rerun subset
([H8M target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8l-iq3-signed-bf16-codebook-target.json)).

**WPF-H8M is rejected at the frozen first-object metadata-VGPR gate.** Complete
all-record, rows1/7/8/9/M512, P64/P65, uneven/empty, H6T/CPU, poison, repeat,
and lifecycle checks pass **4/4** from one cache-only object after exactly one
build. The candidate realizes the exact prescribed **8 b128 + 3 b32 + 6 b64 +
6 d16_b16** loads, zero sign compare/cndmask sites, **216 FMAs / 24
permlanex16 / 96 DPP adds / 24 LDS b128 loads / 12 stores / two barriers**, and
private/spill/scratch0. It cuts code/slots **7,920/1,384→7,768/1,321** and all
24 byte conversions associated with the unsigned table.

The sole failure is metadata VGPR **101→102 > frozen 101**; SGPR78/LDS384 and
all other gates pass. Do not relax the resource bound or rewrite/recompile the
body after observing it. Skip named runtime trace, all-45 timing, runtime/state/
length/source work; remove candidate plus RED and retain H6T production
([H8M rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-iq3-signed-bf16-codebook-physical-rejected.json)).

**WPF-H8N exact Q5 paired-rowgroup twin-team F32-weight staging is target-only.**
After H8M, Q5 remains first at **230.429 vs 58.314 ms**, a **172.115-ms** gap.
The complete six-role H7G/H7H class is **188 calls / 203.861 ms**. H8N adds
one separately named local256 family: two independent logical local128 teams
retain the production `tid=0..127`, `k=tid+128*n`, scalar `fmaf`, wave32
16/8/4/2/1 tree, serial wave0→1→2→3 sum, and BF16/F32 store. Team 0 alone
loads each exact F32 K128×COL slab into ping-pong LDS; both teams consume it
concurrently for adjacent existing activation row groups.

Freeze six geometries before code: team tiles `8×4`, `8×8`, `16×5`, `8×8`,
`16×5`, and `8×10` for BF16 K3072/N1024, BF16 K3072/N12288, BF16
K6144/N3072, BF16 K9216/N3072, F32 K3072/N6144, and F32 K3072/N9216.
This deliberately replaces the two VGPR200 production tiles with known
64-accumulator `8×8` ownership; the others retain their current accumulator
shape. Aggregate logical F32-plane bytes model
**807,571,292,160→407,862,509,568 (−49.495%)** and workgroups
**5,021,440→2,689,792 (−46.434%)**, with all **1,433,445,335,040 useful FMAs**
unchanged. Unlike H5S's serial persistent loop, each H8N pair removes one useful
weight load before either team advances. The cost is **5,021,440→90,764,032
barrier epochs (18.075×)** and per-role fixed LDS **9,216/10,240/18,944/
10,240/18,944/10,752 bytes**. These are source counts, not cache traffic or
speed evidence.

RED must precede executable edits. One build must meet local256, per-role
metadata/runtime VGPR ceilings **80/144/176/144/176/176**, exact fixed LDS, and
private/spill/scratch0. Require odd-pair tails plus rows1/4/5/7/8/9/10/12/13/
17/33/M512, H7G/H7H and sampled CPU bytes, poison/repeat/lifecycle, all six
named cache-only traces, then one 5/15/5 actual-weight screen where every role
and the weighted 188-call aggregate win event and synchronized wall. No role,
dtype, shape, layer, prompt, token, length, geometry, buffer, K-tile, resource
rewrite, recompile, or favorable-rerun salvage is admissible
([H8N target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8m-q5-twin-team-weight-staging-target.json)).

**WPF-H8N is rejected by its first and only all-role both-clock screen.** One
build passes complete rows1/4/5/7/8/9/10/12/13/17/33/M512 control-byte and
sampled-CPU correctness **5/5** with zero further compiler activity. The five
object instances satisfy all fixed local256 resource gates: metadata/runtime
VGPR is **75/80**, **133/136**, or **165/168** within the six frozen ceilings;
LDS is exactly **9,216/10,240/18,944/10,240/18,944/10,752 bytes**, with two
static barriers and private/spill/scratch0. A compiler-free trace names all six
shape-qualified roles at their exact M512 grids on one queue/stream.

The binding actual-weight 5/15/5 screen remains byte-exact but yields **0/6**
roles positive on both clocks. The weighted 188-call H7G/H7H→H8N aggregate
event moves **212.741699→370.566397 ms (+74.186066%, 0.574099×)** and
synchronized wall moves **224.094835→365.406702 ms (+63.058957%, 0.613275×)**.
Thus the modeled 49.495% logical-weight-byte reduction is overwhelmed by LDS
staging and the 18.075× barrier-epoch cost. Apply the frozen no-role/dtype/
shape/geometry/buffer/K-tile/rewrite/recompile/rerun rule: add no runtime or
source owner, remove candidate plus RED, and retain H7G/H7H production
([H8N rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-twin-team-weight-staging-rejected.json)).

**WPF-H8O exact gfx1100 after-router least-priority shared/routed MoE branch
concurrency is rejected at its binding fixed gate.** The generic event path and
W7900 priority range **(+1, −1)** execute the exact frozen two-queue schedule.
One all-48-boundary warmup per arm and all 14 timed requests preserve complete
logits, final/post hidden, K/V plus `KVLiveSpans`, token2930, position511,
finiteness, the 24-plane H8A sidecar, **161,120,256/600,141,856-byte** workspace/
scratch ownership, and session/owner lifecycle with zero compiler activity.

The queue-matched serial control→candidate median is **438.603566→436.513735
tok/s (-0.476474%, 0.995235×)** and the candidate wins **0/7** pairs, failing
both frozen timing requirements. Do not run the named trace or clean
512/1K/4K transfer, and do not salvage eager release, normal priority, another
queue count, layer/length subset, event boundary, rewrite, recompile, or a
favorable rerun. Remove the descriptor and RED, keep all gfx1100 concurrency
capabilities false, preserve gfx1151's separate source policy, and retain the
one-queue exact serial path
([H8O rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-after-router-low-priority-moe-concurrency-rejected.json) ·
[H8O target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8n-moe-shared-after-router-low-priority-target.json)).

**WPF-H8P lossless Q5 signed-int16 power-of-two plane is analytically rejected;
no kernel exists.** The proposed transient format would store 256 signed-int16
mantissas and eight exponents per raw Q5_K block, reducing the exact expanded
record **1,024→520 bytes (−49.219%)** while reconstructing each weight with an
integer conversion and binary scale rather than H5O's coefficient arithmetic.

The representation invariant fails before implementation. A deterministic
first-65,536-block production audit checks **16,777,216** values from
`blk.0.attn_q.weight`; 32-value exponent groups fit only **10.085%**, and even
one exponent per value fits only **53.868%**. Block 0/subblock 0/lane 17 is
F32 `0x3d72fd00 = 62205 × 2^-20`. The magnitude is odd, making `2^-20` its
largest exact binary scale, and 62,205 exceeds signed-int16. Therefore no fixed
group split can satisfy bit identity. Do not add a producer, consumer, registry
key, package owner, RED, build, or timing screen for this format; rerank from
unchanged H7G/H7H/H8B production
([H8P analytical rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q5-fixed16-power2-plane-analytical-rejected.json)).

**WPF-H8Q exact Q6 int16-product plus tiled-F32-scale transient plane is
rejected at its first-object physical gate.** The transient producer and three
H6U-equivalent consumers pass **15/15** exact tests, including both legal
product extrema **−4,064/+4,096**, all roles at rows17/33/M512, sampled CPU
values, poison/repeat/lifecycle, preflight, registry/fallback, and gfx1151
isolation. The local64 producer is **VGPR14/LDS0/private0/spill0**. The local128
consumer object contains the intended int16 conversions, scale multiplies, and
uniform scalar/readfirstlane scale loads with no private segment, spills, or
scratch.

The immutable consumer metadata is nevertheless **VGPR169/136/169** for BF16
K3072/N1024, BF16 K1024/N3072, and F32 K3072/N1024. Those counts already exceed
the frozen runtime ceilings **160/128/160** by **9/8/9**, so all three roles
fail before a runtime trace. Do not perform the producer-inclusive 5/15/5 timing
screen, add a runtime owner, or reinterpret the zero-cost traffic ceiling as a
result. Honor the frozen no-role/dtype/shape/layout/scale/resource-rewrite/
recompile/rerun rule; remove candidate plus RED and retain H8B production
**440.893 tok/s / 2,155 dispatches**
([H8Q rejection](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-q6-int16-product-plane-physical-rejected.json) ·
[H8Q target](../benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8p-q6-int16-product-plane-target.json)).

The W7900 Laguna parity kernel campaign is **tabled** at this boundary; no H8R
kernel is selected. A later campaign must use the component ranking and
high-leverage source-transfer gates in
[`LAGUNA-PARITY-STATUS.md`](LAGUNA-PARITY-STATUS.md), not continue Q5 packing,
IQ codebook/occupancy, qrow/prefetch, or H8Q resource-salvage variants.

WPF-1B now adds a separately registered raw-resident Q5_K/Q6_K MMQ32
primitive in `quant/gguf_k_mmq_prefill.{hip,py}`. One local128 workgroup stages
one K32 interval for 32 raw output columns and 32 producer rows, then reuses
both tiles through packed integer dots. A single row-major K128 Q8_1 block
stores four FP32 scales/sums plus 128 int8 values, so the same producer pack can
feed both quant families without a weight sidecar. All actual M128 roles are
finite at maximum KL **5.6163e-5** and minimum top-1 **96.094%** versus the
then-retained rowbatch8. Inclusive quantize+MMQ improves every N>=1024 role by
**4.798-16.452x**; the N48/N72 gates lose and remain fallback-only. Cached
W7900 tracing records the producer at local256/VGPR24/LDS0/scratch0 and Q5/Q6
MMQ at local128/VGPR48/56/SGPR128/LDS3072/scratch0. A temporary default-off
`raw_k_prefill_mmq` execution owner lazily allocated one bounded Q8_1
workspace, cached one `(pointer, rows, K, stream)` producer pack, and consulted
registered Q5/Q6 crossover policies. It selected only N>=1024; N48/N72 and all
key/shape/backend misses retained the active exact rowbatch owner. The D4
runtime screen improved the
then-retained rowbatch8
**79.119/73.781 -> 129.572/116.116 tok/s** at 512/1K, and its M128 full-state
sample passes at KL **0.034789**, but the mandatory 18-prompt/576-step lane
rejects it at maximum KL **0.624304** despite **97.743%** top-1. The gfx1100
package default remains exact and is now rowbatch32. Keep D4 default-off only as
an accuracy-refinement baseline; finer producer scales or bounded correction
must pass the same complete lane before promotion.
Evidence: [`WPF-1B primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-primitive.json) ·
[`D4 runtime rejection`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d4-runtime-rejected.json).

The separately registered D8/S8 accuracy sibling stores eight FP32 scales/sums
per K128 block (**192 bytes**) and applies independent per-K16 activation scales
to each Q6 weight scale and each half of a Q5 subblock. D4 source/keys remain
unchanged as rejection evidence. D8 passes **9/9** focused GPU cases and all ten
actual M128 roles at maximum KL **5.9453e-5** / minimum top-1 **96.875%**.
Inclusive quantize+MMQ improves every N>=1024 role **5.009-15.848x**; N48/N72
still lose and remain fallback-only. Cached W7900 tracing names D8 producer at
local256/VGPR24/LDS0/scratch0 and Q5/Q6 bodies at local128/VGPR48/56,
LDS3584/scratch0. The temporary default-off owner selected this D8/S8
policy/ABI and allocated **2,359,296 bytes** at M128; rejected D4 remains
available only through its explicit primitive keys. Shared-weight M128 full
state passes at KL **0.002081**, same top-1, and byte-exact candidate repeat
across all 48 hidden boundaries/KV/live spans. Clean default-off A/B moves
**79.179/73.808 -> 129.083/115.802 tok/s** at 512/1K, but the complete
576-step lane rejects D8 at maximum KL **0.400292** despite **98.264%** top-1.
The package default therefore remains MMQ-off and is now exact rowbatch32.
D4->D8 improves only
275/576 teacher steps while worsening 301/576 and leaves 28 steps above the KL
limit in both variants. The completed two-stage residual screen below is the
last blind precision variant; WPF-1R subsequently rejects sparse exact repair
for this Q5/Q6 owner.
Evidence: [`D8 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8-primitive.json) ·
[`D8 runtime rejection`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8-runtime-rejected.json).

The separately registered D8R8/S8 residual sibling represents each BF16 value
as `d*q + dr*qr` per K16 while retaining exact original sums for Q5 minimum
correction. Its K128 block is **352 bytes** with two int8 planes. All ten actual
M128 roles pass at maximum KL **8.241e-7** and **100%** top-1. Inclusive
quantize+MMQ improves every N>=1024 role **3.673-10.989x**; N48/N72 lose and
remain fallback-only. Cached tracing names the producer at local256/VGPR32,
LDS0/scratch8 and both Q5/Q6 leaves at local128/VGPR96/LDS5120/scratch0. The
temporary default-off owner selected this D8R8/S8 policy/ABI and allocated
**4,325,376 bytes** at M128. Shared-weight M128 full state passes at
KL **0.009152**, same top-1, and byte-exact candidate repeat across all 48
hidden boundaries/KV/live spans. Clean default-off A/B moves
**79.022/73.686 -> 123.466/111.324 tok/s** at 512/1K. The complete 18-prompt,
576-step lane nevertheless rejects D8R8 at maximum KL **0.964321** despite
**562/576 (97.569%)** top-1; category top-1 remains above 90%. Reducing maximum
actual-role KL roughly 68x versus D8 while worsening the autoregressive maximum
closes D16 and further blind residual precision screens. D4/D8/D8R8 remain
explicit primitive diagnostics and package production remains exact
rowbatch32. WPF-1R below closes the runtime premise rather than adding bounded
repair.
Evidence: [`D8R8 primitive`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8r8-primitive.json) ·
[`D8R8 runtime rejection`](../benchmarks/results/2026-07-28-gfx1100-laguna-q2-xl-q5-q6-mmq32-d8r8-runtime-rejected.json).

The M512 WPF-1R screen captures all **381/381** raw-Q5/Q6 projection tensors;
333 satisfy the existing D8R8 policy and 48 narrow N48/N72 gates remain exact.
Measured BF16 mismatches are only **0.500-5.270%** of coordinates, but they
already touch **72.266-100%** of encoded output-weight rows in every eligible
tensor. Coordinate repair would reread **0.160-1.686x** the complete exact
RB32 family, so all 333 fail the frozen 20% touched-row stop and 331 also fail
the 25% read stop before conservative risk detection. The per-tensor max-error
BF16-midpoint envelope reaches **9.142-93.418%** uncertain coordinates,
**99.512-100%** touched rows, and **2.925-29.894x** exact-family reads; all 333
fail all three conservative limits. Finite output, BF16/F32 RNE agreement,
complete inventory, token 2930, position 511, and allocation recovery pass.
No queue, repair kernel, overflow route, runtime mode, or timing/quality lane is
added. The rejected public selector, lazy workspace/library owner,
policy/context dispatch, and owner-focused tests are now removed. Exact RB32
remains production; direct D4/D8/D8R8 wrappers and primitive keys remain only
as published ceiling/rejection evidence and have no runtime policy owner.
Evidence: [`WPF-1R raw-Q5/Q6 repair rejection`](../benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q5-q6-d8r8-repair-density-rejected.json).

| Layer key | Quant key | Source | Public wrapper | Current gate |
| --- | --- | --- | --- | --- |
| `laguna_attention_prefill` variant `source_f16_wmma_q8_gqa8_spans` | `bf16` K/V with F32 query/output | `hipengine/kernels/hip_gfx1100/attention/laguna_flash_attention_prefill.{hip,py}`, ported from llama.cpp HIP `c0bc8591e` `fattn-mma-f16.cuh` | `laguna_flash_attention_prefill_f16_wmma_bf16_spans(...)` | WPF-H2 standalone gfx1100 leaf; runtime rejected. One local `(32,4)` block owns eight queries by eight GQA heads, stages aligned BF16 K/V packs into a 272-byte LDS row, and executes K64 online-softmax F16 WMMA while resolving all `KVLiveSpans` fields. Global/sliding fixtures pass at max mean KL **2.53e-10**, min top-1 **94.12%**. Weighted M512 attention moves **490.919 -> 21.719 ms (22.603x)**, nominally matching llama.cpp's **21.725-ms** main+fixup family; traced resources are VGPR208/LDS18,432/scratch0. The attempted 144-block stream/fixup path is removed after non-finite output and a **1.751x** slowdown. The complete runtime gate reaches max KL **1.804860 > 0.05** at **564/576** top-1 despite **1.027x** diagnostic prefill; F32-PV/global-only/SWA-only followups fail. Runtime ownership is removed and exact qrow4/M128 remains production. Evidence: `benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-source-flash-attention-rejected.json` and the standalone candidate artifact. |
| `laguna_attention_prefill` variant `swa_context_rows_qrow4_cached_exact_spans` | `bf16` K/V with F32 query/output | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_swa_attention_prefill_qrow4_cached_exact_bf16_spans(...)` | WPF-H5R gfx1100 production leaf. After the existing writer safely preappends one complete no-wrap M128 tile, one local32 wave reads all historical/current K/V from BF16 cache while replaying H5M/wave32 logical-slot/four-row, ordered-dot, maximum, denominator/PV, division/store, and complete `KVLiveSpans` semantics. Starts 0/128/256/384 are byte-exact, local32/VGPR64/LDS0/scratch0, and include equal append cost while moving the actual 144-call event/wall sums **337.277/334.031 -> 126.687/125.764 ms (2.662x/2.656x)**. Complete state is KL0; corrected one-queue tracing records all 144 write->H5R pairs and selector-unset 512/1K/4K improves **+11.340%/+4.848%/+0.746%** with 3/3 wins each, promoting **267.205/230.441/160.221 tok/s**. The exact global sibling was removed after losing all starts at **0.636–0.926x**. Runtime ownership is production; partial/wrapped/verifier/miss paths retain H5M/wave32 and gfx1151 is excluded. [`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-production.json) · [`candidate`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-swa-preappend-cached-exact-candidate.json). |
| `moe_linear` variants `selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out` / `selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out` | `gguf_iq3_xxs` / `gguf_iq4_xs` | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.{hip,py}` | strict K1024 resident-segment IQ3 wrapper and local32 launch of the retained exact IQ4 body | WPF-H5J gfx1100 production. IQ3 decodes each immutable eight-value segment once per active expert/output block and replays the retained rowbatch8 arithmetic. IQ4 reuses the exact dynamic physical body at local32; a generated RED removed a separately compiled constant-K duplicate after a one-BF16-ULP mismatch. All 47 actual M512 layers are byte-exact and both-clock positive, moving event/wall sums **567.274/567.056 -> 500.176/500.448 ms (-11.828%/-11.746%)**. Cached resources are IQ3 local128/VGPR40/LDS512/scratch0 and shared IQ4 local32/VGPR64/LDS512/scratch0. Complete state is KL0; integrated selected down falls **10.706%** and clean 512/1K/4K reaches **196.103/181.859/137.169 tok/s**. gfx1100 defaults only K1024/N3072; every miss and gfx1151 retain exact fallback. H5K rejects/removes scratch-free rowbatch12/16 after both lose all 45 actual layers by **5.8–10.8%**. [`production`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-production.json) · [`H5K rejection`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-larger-resident-rowbatch-rejected.json) · [`leaf`](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-candidate.json). |
| `moe_linear` variant `selected_grouped_prefill_compact_k1024_active_expert_p64_activation_resident_out_p256_rowbatch8_bf16_bf16_out` | `gguf_iq3_xxs` | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.{hip,py}` | `GGUF_IQ3_ACTIVATION_RESIDENT_OUTPUT_PARTITIONS[256](...)` | WPF-H5Z gfx1100 production. It reuses H5Q's active-expert P64 ABI but inverts output ownership: each local128 block retains one rowbatch8 K8 BF16 activation tile and processes 12 strided output columns sequentially. All bytes, IQ3 decode/FMA/reduction/store order, metadata, allocation, and H5Q fallback are unchanged. P256/P512 alone win all 45 layers; max-min retains P256. Complete state is KL0; paired tracing cuts IQ3/request/span **2.342%/1.312%/1.539%** at unchanged **2,050** dispatches. Selector-unset 512/1K/4K improves **+1.819%/+1.452%/+0.872%**, 3/3 wins each, promoting **307.658/259.947/173.562 tok/s**. Cached resources are VGPR112/SGPR128/LDS512/scratch0; eight b128 activation records precede the unchanged 2-d16/3-b32 loop. H5Q remains rollback and gfx1151 is excluded. [`production`](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-production.json) · [`candidate`](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-activation-resident-output-sweep-candidate.json). |
| `moe_linear` variant `selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out` | `gguf_iq3_xxs`, `gguf_iq4_xs` | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_source_mmq_prefill.{hip,py}`, ported from llama.cpp HIP `c0bc8591e` MMQ family | `gguf_iq{3_xxs,4_xs}_selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out(...)` | WPF-H3 standalone gfx1100 leaf; runtime rejected. Reuses caller-owned strict DS4 bytes and compact expert metadata, pads only tile ownership to J128, and reads raw expert-major weights. Actual M512 **45 IQ3 + 2 IQ4** layers improve **565.437 -> 115.951 ms (4.877x)**; IQ3 is **27.145% below** matched llama.cpp. Complete runtime quality reaches max KL **0.373028** at **567/576** top-1 despite **1.192x** diagnostic prefill; IQ3-source/IQ4-exact still reaches **0.372917**. Runtime ownership is removed and exact grouped down remains production. Local `(32,8)`, dynamic-LDS57,856, IQ3/IQ4 VGPR152/248, scratch0; gfx1151 excluded. |
| `dequant` variant `raw_f16_source_local64`, `dequant_cast` variant `raw_f16_bf16_input_source_local64`, and `linear` variants `f16_rocblas_source_bf16_{bf16,f32}_out` | raw `gguf_q6_k` | `hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_f16_rocblas_prefill.{hip,py}`, ported from llama.cpp HIP `c0bc8591e` Q6 conversion and F16 rocBLAS route | raw dequant, fused dequant/input-cast producer, and complete BF16/F32-output wrappers | WPF-H4 standalone gfx1100 leaf; runtime rejected. One local64/VGPR16/LDS0/scratch0 producer writes caller-owned F16 weight/input planes, F16-compute `rocblas_gemm_ex` writes F16 output, and one cast returns BF16/F32. The unfused registered producer chain remains fallback. Actual six-shape/144-call M512 improves exact coltile **174.351 -> 14.349 ms (12.151x)**, **3.825% below** matched llama.cpp. Complete runtime quality reaches max KL **0.338657** at **567/576** top-1 despite **1.042x** diagnostic prefill. The temporary owner/workspace/capabilities are removed; exact coltile remains production and gfx1151 stays excluded. Evidence: `benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-rejected.json` and the standalone candidate artifact. |
| `dequant` variant `raw_f32_exact_local64`, `dequant_cast` variant `raw_f32_bf16_input_exact_local64`, and `linear` variants `f32_rocblas_exact_values_bf16_{bf16,f32}_out` | raw `gguf_q5_k` | `hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}` | exact-value raw dequant, fused dequant/input-widen producer, and complete BF16/F32-output wrappers | WPF-H5A standalone gfx1100 leaf; runtime rejected. Raw-Q5 expansion and BF16 widening are bit-exact; output passes at max mean/max-row KL **1.59e-9/5.79e-8** and top-1 **100%**. The local64/VGPR16/LDS0/scratch0 producer writes bounded caller-owned F32 planes; no sidecar exists and gfx1151 is excluded. With N48 exact fallback, the 235-call policy moves **1,256.936 -> 221.137 ms (5.684x)** events and **5.273x** wall, but remains **3.751x** llama.cpp. Natural M512 passes at KL **0.0003742**, but complete quality reaches max KL **1.143627** at **564/576** top-1 despite **1.330x** diagnostic prefill. The temporary owner/workspace/capabilities are removed; exact coltile remains production. Evidence: `benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-rejected.json` and the standalone candidate artifact. |
| `linear` `f32_weight` variants `ordered_coltile4_rowbatch{8,16}`, `ordered_coltile8_rowbatch{4,10,12}`, `ordered_coltile12_rowbatch{4,8}`, and `ordered_coltile16_rowbatch{4,5}` for BF16/F32 output plus raw-`gguf_q5_k`/`gguf_q6_k` `f32_ordered_*` composites | transient exact F32 weights / raw `gguf_q5_k` or `gguf_q6_k` | `hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.{hip,py}` | exact producer plus production-ordered local128 consumer; primitive and raw-quant composite registrations | WPF-H5C..H5J gfx1100 exact production. Q5/Q6 rows17/33 and all actual roles are byte-exact with one **150,994,944-byte** max plane/no sidecar. H5G owns Q5 constant-80/96; H5H rejects/removes 112/128 at the VGPR256 spill cliff; H5I promotes four exact-Q6 roles. Complete state is KL0 and clean H5J production is **196.103/181.859/137.169 tok/s** through 4K. gfx1100 defaults the quant-keyed route and gfx1151 remains fail-closed. H5L adds separately registered `ordered_weight_major_*` / `f32_ordered_weight_major_*` Q5 siblings for BF16 `8x4/8x12/16x5/12x8` and F32 `16x5/8x10`. Only linear workgroup ownership changes. The six-role/235-call policy is exact and moves event/wall **882.963/887.364 -> 486.892/474.348 ms (1.813x/1.871x)**; F32 N48/N72 stay H5G. Complete state is KL0, integrated Q5 falls **49.224%**, and clean production reaches **237.956/217.888/157.366 tok/s**. H5L is the gfx1100 package default; every miss and gfx1151 retain H5G. |
| `embedding` variant `lookup_bf16_out` | `gguf_q4_k`, `gguf_q5_k` | `hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_embedding.hip` | `gguf_q4_k_embedding_bf16_out(...)`, `gguf_q5_k_embedding_bf16_out(...)`, registry-driven `launch_gguf_embedding(...)` | Net-new gfx11 raw-Q4_K/Q5_K row dequant for Laguna target/DFlash roots. Synthetic Q4_K lookup is BF16-exact vs CPU, invalid token IDs preserve caller output rows, and the real S 2.1 root chain (Q4 embedding -> F32-weight RMSNorm -> Q6T16 full logits -> GPU argmax) gives embedding/norm max-abs `0`, logits KL `6.87e-13`, top-1 `81364 == 81364`, and finite 100,352-way logits on gfx1151. Cached `rocprofv3 --kernel-trace` shows `gguf_q4_k_embedding_bf16_out_kernel` at `9.818 us`, 16 VGPR, zero scratch/LDS, followed by the expected norm, Q6T16, and argmax kernels. The pinned Q2 XL Q5_K root is BF16-exact against direct GGUF dequant for four repeated/unique rows; cached W7900 tracing names `gguf_q5_k_embedding_bf16_out_kernel` at `12.440 us`. `scripts/laguna_root_probe.py`; `/tmp/laguna-root-rocprof-result.json`. |
| `linear` variants `pack8_gemv_decode_bf16_{bf16,f32}_out`; `linear_pair` variants `pack8_gemv_decode_bf16_{bf16,f32}_out` | raw `gguf_q5_k`, `gguf_q6_k` | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip` | `gguf_q5_k_pack8_gemv_decode_bf16_{bf16,f32}_out(...)`, `gguf_q5_k_pair_pack8_gemv_decode_bf16_{bf16,f32}_out(...)`, `gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out(...)` | The singleton local128 schedule hoists each output-pack's exact Q5 d/dmin and packed scale/min coefficients to 1,024 B LDS while preserving the established per-thread K order and reduction tree. The registered pair uses the same forced-inline block body and independent `blockIdx.y` gate/up workgroups, so each output remains BF16-bit exact to two singleton launches. Synthetic K3072/N1024 pair output and actual Laguna layer-1 CPU-quant/full-MoE oracles pass; the focused bundle reports 107 passes. Actual `blk.1` shared gate/up counterbalanced wall is **28.148 -> 16.373 us/pair (-41.83%; 1.719x)**. Cached W7900 tracing names `gguf_q5_k_pair_pack8_decode_out_kernel` at **12.4 us median** in isolation and **19.321 us median** across 46 all-layer calls, local128/VGPR72/LDS1024/scratch0. The dirty full-model short trace removes 46 launches/token (1,010 -> 964), moves the pair family **1.561 -> 0.891 ms/token (-42.94%)**, kernel sum **18.983 -> 18.240 ms (-3.92%)**, and span **22.878 -> 21.963 ms (-4.00%)** with exact d32 IDs/lifecycle. Laguna c=1 requests only registered decode pairs and retains two singleton launches on registry/shape miss; rows>1 are unchanged. The BF16 shared pair's clean category gate promotes D5 at **44.501 h32 decode tok/s (+3.30%)** with exact suite/state/lifecycle. The retained F32 unequal-width sibling flattens each query+per-head-gate pack range into one grid, preserves both singleton output buffers byte-for-byte at K3072 N6144+48 and N9216+72, and retains the same fail-closed c=1-only dispatch. Actual `blk.0/1` query+gate pairs improve **44.827 -> 34.499 us (-23.04%)** and **56.790 -> 42.123 us (-25.83%)**. Clean cached traces confirm exactly 47 F32 pair launches and **964 -> 917 dispatches/token**, local128/VGPR48/SGPR128/LDS1024/scratch0. Short/512/1K/near-4K kernel sum improves **3.13%/1.91%/1.70%/1.21%**, median span improves **3.21%/2.15%/1.95%/1.38%**, and profiled child throughput improves **3.89%/2.87%/1.81%/1.48%**, with exact IDs/finite logits/teardown. The complete suite promotes D6 at **45.433 h32 decode tok/s (+2.09%)** and **11.921 h32 E2E (+0.52%)**, with every category positive and prefill neutral. The new raw-Q6 F32 sibling flattens both pack ranges and preserves the established singleton arithmetic for 47 equal-width K/V pairs plus layer 47's unequal query/gate pair. Production synthetic K3072 N1024+1024 and N9216+72 outputs are byte-exact, and actual global/SWA K/V plus layer-47 query/gate improve **37.36%/36.65%/8.80%**. A dirty cached full-model trace names 48 `gguf_q6_k_pair_pack8_decode_out_kernel` calls, removes **48 launches/token (917 -> 869)**, runs local128/VGPR56/SGPR128/LDS512/scratch0, moves the Q6 F32 family **1.465 -> 0.934 ms/token (-36.26%)**, total kernel sum/span **17.689 -> 17.256 ms (-2.45%)** / **21.274 -> 20.750 ms (-2.46%)**, and preserves exact IDs/finite logits/teardown. Clean context/category promotion promotes D7 at **46.409 h32 decode tok/s (+2.15%)** with exact suite/state/lifecycle. The next exact Q5 attention-output tile16 candidate remained BF16-bit exact and scratch-free at K6144/K9216 N3072, but its best local256/VGPR88/LDS2048 schedule regressed actual-weight HIP-event time **16.69%/19.04%**; it was removed before full-model measurement (`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q5-output-tile16-rejected.json`). |
| `linear`, `linear_pair`, `linear_triple` mixed and `tiled_*` variants | `fp16_weight` | `hipengine/kernels/hip_gfx1100/linear/laguna_f16_projection.hip` | single BF16/F32 activation to BF16/F32 output; BF16-to-F32 dual/triple; exact rows>1 8x4/16x4 tiled single/triple; direct WMMA diagnostics plus compensated WMMA single/triple with a quality-gated gfx1151 SWA default; model-neutral `launch_f16_weight_linear{,_pair,_triple}(...)` | Source-F16 Laguna Q/K/V/gate/O projection foundation. Decode keeps the original 256-thread GEMV reduction. The LPF-1 tile preserves that thread-local K sequence and reduction tree while reusing each activation across four output columns and each weight across 8 rows below M16 or 16 rows at/above M16. Synthetic rows 2/3/4/5/17 are bit-exact for F32 and BF16 output. Clean same-session rows 2..128 are exact and all faster, with a 2.0538x weighted profile, 55 rows 23.460->48.760 tok/s (2.0784x), and 128 rows 23.374->50.240 (2.1494x). The clean two-repeat ten-prompt gate moves the prior bulk-GEMV row 23.333->48.560 tok/s (+108.12%), TTFT 3.481->1.692 s, and h32 E2E 5.719->8.717 tok/s while all IDs/categories/Poolside/lifecycle gates pass and decode stays neutral. gfx1151 therefore defaults to tiled from two rows; rows=1 and unsupported backends retain GEMV. Cached gfx1151 trace names `laguna_f16w_tiled_exact_kernel<unsigned short, 16>` at 3.798 ms for the 55x9216x3072 O shape, workgroup 256, grid 196608x4, 96 VGPR, 128 SGPR, 512 B LDS, and zero scratch. A faster reassociated 16x16 WMMA control reached 60.65 tok/s but changed three of ten free-running trajectories and was removed rather than admitted. AR-O2 now has a separately registered replacement candidate that reads the same resident row-major F16 bytes directly, converts BF16 activations to F16 in registers, and accumulates 16x16x16 F16 WMMA into FP32 without a sidecar or inference-time staging. Seeded M16/M17/M32/M64/M128/M256/M512 synthetic F32 output passes the CPU gate at KL <=0.05/top-1 >=90%, BF16 output is the exact RNE boundary of its FP32 result, and triple dispatch matches three CPU matrices. The current tile shares BF16->F16 activation fragments across four output waves through bounded LDS; M256/M512 additionally share each weight fragment across two/four row groups. Cached gfx1151 tracing confirms local128/VGPR104/LDS4096/scratch0 at M128, local256/VGPR104/LDS10240/scratch0 at M256, and local256/VGPR136/LDS17408/scratch0 at M512. The clean production screen improves every M16-512 full/SWA family; at M128 exact -> WMMA is **12.772/18.505 -> 1.800/2.474 ms (7.094x/7.479x)** and the 12-full/36-SWA sum improves **7.403x**, with nonzero KL `3.60e-16`, top-1 100%, finite output, and exact teardown. The complete ten-prompt model gate rejected this schedule despite **53.447 -> 73.637 prefill tok/s (1.3778x)** because maximum teacher-forced KL reached **0.097062** (>0.05); suite top-1 was 317/320, three prompt trajectories differed deterministically, the Poolside first-token oracle passed, and teardown was exact. Its direct all-layer runtime/category route is removed, and only that separately registered direct leaf stays diagnostic. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-screen.json` and `benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-category-rejected.json`. The new separately registered `wmma_comp_*` variant computes each WMMA K16 partial from zero and Kahan-accumulates those FP32 partials while retaining the same resident F16 bytes and BF16-to-F16 register conversion. Synthetic M16/M128/M256/M512 single output and triple dispatch pass the CPU quality gate. The scratch-free M128 RT4 trace names `<float,4,true>` / `<unsigned short,4,true>` at **65.002/60.393 us** for K512/N35, local128/VGPR152/LDS2048/scratch0. The clean M16-512 production screen improves every full/SWA family; at M128 exact -> compensated is **12.802/18.502 -> 2.190/2.893 ms (5.847x/6.396x)** and the weighted 12-full/36-SWA sum improves **819.707 -> 130.422 ms (6.285x)**. The clean three-repeat ten-prompt gate promotes the quality-safe SWA-only boundary while every broader full-model/component schedule remains rejected. Exact tiled -> compensated SWA QKV/gate/O from M16 moves weighted prefill **53.388 -> 69.037 tok/s (+29.313%)**, TTFT **1.529 -> 1.187 s (-22.377%)**, and h16/h32 E2E **+17.004%/+11.663%** with neutral decode. All 320 teacher-forced logits are finite at maximum KL **0.043888**, top-1 is **318/320 (99.375%)**, every category is >=96.875%, and Poolside oracle/repeats/lifecycle pass. gfx1151 `auto` selects only this scoped route; full-attention, M2-15, rows=1, and unmeasured backends remain exact, with explicit `tiled`/`gemv` rollback. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-comp-screen.json` and `benchmarks/results/2026-07-23-gfx1151-laguna-f16-wmma-comp-swa-retained.json`. The original production-head fixture remains against FP32 CPU matmul; its decode trace shows single BF16->F32 `102.431 us`, single BF16->BF16 `104.756 us`, dual `129.122 us`, and triple `127.198 us`, all 16 VGPR, 512 B LDS, zero scratch. |
| `linear` runtime variant `hipblaslt_scaled` | resident `fp16_weight`, BF16 activation, FP32/BF16 output | `hipengine/core/hipblaslt.py`, `hipengine/runtime/laguna_f16_hipblaslt.py`, `hipengine/runtime/laguna_gguf_runner.py` | session-cached zero-workspace FP16-input/F16-weight/FP32-output hipBLASLt contraction plus scaled-row cast/restore | LAP-6 gfx1151 rows>1 default; exact tiled/GEMV remain fallback and decode is unchanged. One producer-row cast is shared by Q/K/V/gate, O casts in place, and the post-embedding token-ID allocation aliases row scales so scratch does not grow. A real same-session pp512 diagnostic compounded with D4x3 MMQ improves **127.831 -> 154.321 tok/s (1.207x)** with next token 2930 in both modes and seven cached descriptors. The clean compounded category gate admits the route at max KL **0.040724836**, **317/320** top-1, flat decode, and exact lifecycle recovery. |
| `linear` variant `pack8_wmma_prefill_bf16_bf16_out` | resident rank-2 `gguf_q4_k` pack8 | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.{hip,py}` | direct qweight/F32-scale/F32-min 64x16 wave32 WMMA consumer | LAP-5 gfx1151 rows>=16 default for Laguna dense/shared Q4. It consumes existing resident bytes with no sidecar/cache rebuild, converts Q4 weights and BF16 activations to FP16 WMMA operands, accumulates in FP32, and preserves the BF16 boundary. The synthetic pack8 output is BF16-bit identical to raw-Q4 WMMA and passes its CPU-reference quality gate. M512/K3072/N1024 improves **1.2695 -> 0.2407 ms (5.275x)**; compounded pp512 reaches **163.881 tok/s**, next token 2930. Cached gfx1151 trace names the 64x16 kernel at **23.244 us** on the boundary fixture, local32/VGPR88/SGPR128/LDS0/scratch0. Exact pack8 FMA remains fallback; the clean compounded category gate admits this route with the D8/D4 expert and hipBLASLt defaults. |
| `head_rmsnorm+partial_rotary` variants `positions_f32`, `positions_packed_query_f32`; `attention_gate` softplus variants | `laguna_f32_weight`, `f32` | existing exact `fused/gguf_ops.hip` head-normalize/rotate body plus `fused/laguna_attention.hip` | `materialize_laguna_rope_tables(...)`, `launch_laguna_head_rmsnorm_rope(...)`, `laguna_softplus_head_gate_f32_{,bf16_}out(...)` | Host tables use the independent Transformers-validated Laguna CPU YaRN/plain equations and absolute indices. gfx1151 production-shape tests cover partial-64 YaRN with 48 Q/8 KV heads and full-128 plain SWA with 72 Q/8 KV heads, dim 128. The packed-query sibling changes only destination addressing for aligned M128 tiles; generic rows and all key rows are F32-bit exact to `positions_f32`. Eleven complete pp512 pairs remain state-exact and improve **647.210 -> 650.651 tok/s (+0.532%, 7/11 wins)**. The clean selector-unset trace names 48 packed producers at local256/VGPR16/SGPR128/LDS0/scratch0, removes all **144 / 4.907-ms** query transposes, cuts dispatches **2,273 -> 2,129**, and improves producer-plus-pack **20.530 -> 16.666 ms (-18.82%)**. Clean 512/1K/4K reaches **654.249/579.699/468.608 tok/s**. Existing cached traces for the ordinary fused head RMSNorm+RoPE are `13.505/13.426 us`; FP32/BF16 softplus broadcast is `2.485/1.764 us`, all zero scratch. |
| `attention_gate` variants `softplus_broadcast_{bf16,fp16_via_bf16}_packed_tiles_out` | mixed generic/head-major `f32` attention output | `hipengine/kernels/hip_gfx1100/fused/laguna_attention.{hip,py}` | `laguna_softplus_head_gate_f32_{bf16,fp16_via_bf16}_packed_tiles_out(...)` | Exact gfx1151 packed-output production boundary. Qualified dense-initial library PV tiles remain head-major in their destination slices; one two-dimensional softplus kernel reads that mixed layout and emits the ordinary generic BF16 or FP16-via-BF16 tensor. The focused mixed-layout fixture is bit-exact to the generic gate. Direct `(row, head)` block mapping removes runtime division; the rows384 trace is local128/VGPR8/LDS0/scratch0 and costs **0.178974 ms** versus **0.212719 ms** for the generic gate. Eleven complete-state pp512 pairs are exact at **647.920 versus 645.735 tok/s** independent medians. The repaired clean all-family trace removes all **144 / 3.703-ms** output transposes, cuts dispatches **2,417 -> 2,273**, and improves transpose plus gate **11.240 -> 10.318 ms (-8.20%)**. |
| `laguna_kv_write`, `laguna_attention_decode`, `laguna_attention_prefill` global/SWA span variants | `bf16` | in-tree `attention/laguna_kv_attention.hip`; global keeps the proven block-256 page-table structure | `allocate_laguna_kv_cache(...)`, scalar and rows forms of `laguna_global_{write_kv,attention_*}` / `laguna_swa_{write_kv,attention_*}` | The owner allocates 12 global caches at admitted context and 36 physical 512-token rings: exact 4K payload is 264 MiB across 243 tracked payload/metadata allocations, with teardown returning counters to baseline. Every scalar/bulk writer and reader consumes complete `KVLiveSpans`; bulk rows infer consecutive absolute positions from the span's query-position scalar. Production 48/8 and 72/8 GQA decode matches direct FP32 CPU attention through 510/511/512/513 and repeated wraps. The bulk gate seeds 508 rows, then proves an eight-row 508..515 global/SWA chunk bit-exact against append+decode per row, including permuted physical offsets, explicit eviction, and future ring overwrites that must not hide earlier-row keys. Cached gfx1151 tracing records bulk global/SWA attention at `1058.825/1672.777 us` versus eight scalar readers totaling about `1396/5547 us`; bulk writers are `1.683-59.231/1.563-49.854 us`, with expected kernel names. The older 1026-row scalar trace remains `1.603/1.523 us` median writers, `730.650 us` global read, and `1123.828 us` SWA boundary median. Full resident lengths 1/2/7/55/65 plus five B+1 rows now match serial live-span metadata and every live BF16 K/V row exactly (`2026-07-22-gfx1151-laguna-bulk-prefill-verifier-correctness.json`). LPF-5 adds a separately registered diagnostic SWA prefill `swa_context_rows_wave32_exact_spans`: one wave reconstructs the baseline 128-thread stride-64/32/16..1 reduction tree exactly while removing per-token block barriers. The 508..515 wrap/eviction fixture is F32 byte-exact to both baseline bulk and scalar attention. A balanced production 128-row/512-window leaf probe improves **20.434 -> 9.229 ms (2.214x)**; cached tracing confirms the candidate at **9.123 ms median**, workgroup 32, 32 VGPR, 128 SGPR, zero LDS/scratch versus baseline **20.355 ms**, workgroup 128, 16 VGPR, 1,024 B LDS, zero scratch. The clean shared-weight full-model gate promotes this variant on gfx1151 after exact complete logits/hidden/cursors and 512/1K/4K gains of **+8.31%/+12.85%/+14.06%**; a prior complete timing pass independently reproduced **1.082/1.128/1.140x**. LPF-5 backend capability metadata initially selected wave32 exact only on gfx1151, while explicit baseline rollback and unmeasured-backend defaults remained; AR-O5's context-qualified qrow2 selector then overlaid only its measured M128/start>=128 slices and otherwise delegates back to wave32. AR-O3 now lets one M256/M512 matrix transaction feed unchanged global/SWA attention in resident-position-backed slices of at most 128 rows: a cached gfx1151 fixture seeds 0..383 and matches ordinary 128+2 chunks byte-for-byte for all context, K/V, and span state through positions 511/512/513, without a per-slice host metadata copy or kernel-signature change. The clean repeated 512/1K/4K full-state screen promotes matrix512/attention128 on gfx1151: M128 **64.997/60.385/49.540** -> M512 **69.069/63.925/51.989 tok/s (+6.266%/+5.862%/+4.943%)**, aggregate wall **1.05183x**, with complete logits/hidden/KV/span/cursor/repeat/lifecycle equality. Exact M512 row/MoE scratch is **411,953,168 bytes** under the unchanged 2-GiB admission floor; unmeasured backends retain matrix128. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-matrix-chunk-retained.json`. The first AR-O5 exact SWA query-head-group candidate reused each BF16 K/V row across all nine query heads, but VGPR rose **32 -> 104** and production M128/full-window time regressed **9.179 -> 9.858 ms (+7.41%)**; the 508..515 wrap fixture regressed **1.054 -> 2.945 ms (+179.53%)** despite byte-exact output. All qgroup9 code is removed; retain wave32 and pursue query-row/online-softmax tiling instead (`benchmarks/results/2026-07-23-gfx1151-laguna-swa-qgroup9-rejected.json`). The next explicit `swa_context_rows_qrow2_exact_spans` leaf preserves each adjacent query row's exact scan/reduction/softmax order while sharing BF16 K/V loads. It is byte-exact on the 508..515 fixture and runs local32/VGPR56/SGPR128/LDS0/scratch0. Dirty full-window medians improve exact wave32 at M32/M55/M64/M122/M128 by **1.118/1.217/1.145/1.092/1.163x**. The first M-only three-repeat model screen improves 512/1K/4K **1.057%/1.199%/1.040%**, but empty-context M128 is only **0.876x** and every first category repetition regresses. A dedicated prior-context sweep reaches **1.048x** at 128, so conservative `swa_context_rows_qrow2_m128_c128_exact_spans` delegates to wave32 unless rows==128 and absolute start>=128; shorter residual tiles remain on measured wave32. The final exact-selector gate improves 512/1K/4K **69.031/63.969/52.017 -> 69.647/64.745/52.557 tok/s (+0.893%/+1.212%/+1.040%)** with complete logits/hidden/KV/span/cursor/repeat/lifecycle equality. Its ten-prompt category gate is exact and non-regressive at **0.999652x prefill** and **0.999917/0.999999x h16/h32 E2E**. At that stage gfx1151 defaulted to the context-qualified selector; after the online-qrow2 promotion it remains the primary exact rollback and still delegates empty/short/partial/verifier rows to wave32, while unmeasured backends remain unchanged (`benchmarks/results/2026-07-23-gfx1151-laguna-swa-qrow2-retained.json`). The clean post-promotion trace confirms SWA duration falls **9.38%/9.00%/8.99%** at 512/1K/4K while global is flat and reaches **16.823 s / 21.62%** at 4K. The vendored AOTriton GPU/head-dim-256 control passes, but native V3 and per-query-head V2 both return `hipErrorInvalidValue` for Laguna head-dim-128, including a global-only M512 query tile. Do not add a shape expansion or adapter selector; the remaining global lane requires an in-tree raw-pointer tiled causal kernel with complete `KVLiveSpans` (`benchmarks/results/2026-07-23-gfx1151-laguna-post-qrow2-global-screen.json`). The explicit `global_context_rows_qrow2_online_spans` candidate now streams one BF16 K/V row across two adjacent query rows and maintains online max/denominator/output state without whole-context score LDS. Six M128/context4096 prior-context screens are finite and within max-abs `5.96e-8` of the CPU-validated retained route; at 3968 prior rows, synthetic 128-way KL is `1.14e-15` with 100% top-1. Balanced leaf medians improve **0.994/3.352/8.858/20.217/42.299/86.273 -> 0.267/0.725/1.660/3.538/7.223/14.705 ms (3.720-5.867x)**. Cached gfx1151 tracing names the online kernel at **14.807 ms**, local32/grid1536x64/VGPR48/SGPR128/LDS0/scratch0 versus retained **86.752 ms**, local256/grid12288x128/VGPR40. The clean three-repeat full-model screen moves 512/1K/4K **69.751/64.756/52.584 -> 71.475/68.281/64.076 tok/s (+2.472%/+5.444%/+21.854%)**, with finite logits, maximum KL `0.007589`, top-1 9/9, exact cursors, deterministic state, and exact lifecycle. The complete category gate promotes it on gfx1151: weighted prefill improves **69.310 -> 69.529 tok/s (+0.315%)**, h16/h32 E2E improves **+0.184%/+0.125%**, every category is positive, maximum KL is `0.030836`, top-1 is 317/320, and Poolside/repeats/lifecycle pass. Exact global prefill remains explicit rollback and the default on unmeasured backends (`benchmarks/results/2026-07-23-gfx1151-laguna-global-qrow2-online-retained.json`). The retained-default reprofile cuts global duration **79.49%/81.62%/82.53%** and kernel sum **2.80%/5.33%/18.03%** at 512/1K/4K; 4K global is now **4.61%**, SWA **15.21%**, total attention **19.82%**, and span residual **0.208%**. Defer another grouped-head global route under its `1.048x` perfect-removal ceiling; permit one bounded SWA online-qrow2 screen under SWA's `1.179x` ceiling (`benchmarks/results/2026-07-23-gfx1151-laguna-post-global-online-all-family-profile.json`). That explicit `swa_context_rows_qrow2_online_spans` leaf now replaces exact qrow2's two ring scans with one online max/denominator/output scan while preserving complete span/ring visibility. M128/full-window improves **7.893 -> 2.552 ms (3.093x)** and start508 wrap **8.676 -> 2.987 ms (2.904x)** with max-abs `3.45e-8`, synthetic KL `1.11e-15`, and top-1 100%. Cached tracing names the candidate at **2.559 ms**, local32/VGPR56/SGPR128/LDS0/scratch0. The clean repeated full-model gate moves 512/1K/4K **71.354/68.156/63.995 -> 76.226/74.538/70.885 tok/s (+6.828%/+9.364%/+10.766%)**, with max full-vocabulary KL `0.016558`, top-1 9/9, exact IDs/cursors, deterministic state, and exact lifecycle. The complete category gate promotes it on gfx1151: weighted prefill improves **69.011 -> 69.761 tok/s (+1.086%)**, h16/h32 E2E improves **+0.616%/+0.420%**, every category is positive, maximum KL is `0.042924`, top-1 is 316/320, and Poolside/repeats/lifecycle pass. Exact context-qualified qrow2 and wave32 remain rollback/fallback, and unmeasured backends retain prior defaults (`benchmarks/results/2026-07-23-gfx1151-laguna-swa-qrow2-online-retained.json`). The separately registered `swa_context_token4_exact_spans` decode candidate uses four waves for four independent logical-slot dots, stores unscaled dots/physical slots in 4,120 B dynamic LDS, and then preserves baseline-order max, contracted score-minus-max, denominator, and value accumulation. All seven KV tests pass byte-exactly through 510/511/512/513 and repeated wrap 1024/1025 with an explicit eviction. Cached W7900 tracing measures the six candidate calls at **237.722 us median** versus baseline **792.747 us (-70.01%; 3.335x)**, local128/VGPR24/static-LDS0/scratch0. Clean full-model traces preserve exact IDs/state/lifecycle and move short/512/1K/near-4K SWA **4.202/27.776/27.846/27.901 -> 2.118/13.111/13.096/13.104 ms/token (-49.60%/-52.80%/-52.97%/-53.03%)**. The complete ten-prompt category gate promotes gfx1100 token4 by backend capability at **43.081 h32 decode tok/s (+10.919% vs D3)** and **11.760 h32 E2E (+2.724%)**, with prefill within -0.223% and explicit baseline rollback; gfx1151/unmeasured backends retain baseline. All other rows are correctness/dispatch diagnostics, not a retained target-throughput claim. |
| `laguna_attention_decode` variant `global_context_wmma_gqa6_k64_three_term_raw_numerator_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_wmma_gqa6_k64_three_term_raw_numerator_bf16_spans(...)` | gfx1151 diagnostic-only global decode primitive. One local256 block owns each GQA6/K64 split; three BF16 query/probability terms feed independent F32 WMMA QK/PV accumulators and a local128 FP64 raw-numerator merge. The evicted live513/576/639 CPU/control gate has max F32 error **1.49e-8** and **0/1/2** gated BF16 mismatches; leaves improve **43.45-47.08%**. Cached trace records partial local256/VGPR96/LDS4608/scratch0 and merge local128/VGPR24/LDS0/scratch0. The complete 18-prompt/576-step gate is finite with exact span/reset/lifecycle state and **97.05%** top-1, but max KL **2.623766** is **52.48x** the ceiling. The selector/harness are removed and production remains unchanged; retain only as evidence that unqualified cooperative association is inadmissible (`benchmarks/results/2026-07-29-gfx1151-laguna-global-three-term-wmma-rejected.json`). |
| `laguna_attention_decode` variant `global_context_fused_exact_gated_mixed32_local512_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_fused_exact_gated_mixed32_local512_*_bf16_spans(...)` | Retained gfx1151 exact global decode default. The 32-owner grid grows local256 to local512 so sixteen waves partition independent QK and value transport, while softmax issue and denominator ownership remain frozen at eight waves to preserve FP32 association. Live513/576/639 context F32 and gated BF16 are byte-identical; 21x100 leaves improve **26.11%/23.43%/30.51%**. Cached tracing confirms grid32/local512, VGPR48/SGPR128/LDS512/scratch0. Seven resident p512/d128 pairs all improve **20.581562 -> 20.726022 tok/s (+0.70189%, -0.33865 ms/token)** with exact trajectory/state/lifecycle. gfx1151 capability selects it through live4000; local256 and peer-backend routes remain unchanged rollback (`benchmarks/results/2026-07-30-gfx1151-laguna-global-mixed32-local512-retained.json`). |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_dim32_vstage64_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_dim32_vstage64_bf16_spans(...)` | gfx1151 exact long-global decode owner above the retained 6,000-live-slot fused bound. One wave32 score owner loads each K row once for all six GQA queries; a local256 normalization phase preserves the generic eight-wave exp32 max/sum tree; four D32/local512 PV blocks per KV head stage V64 once and replay it across all six queries while preserving each output's chronological scalar-F32 FMA chain. The live4,097/16,448/65,664/131,200 leaf is F32/BF16 byte-exact and improves **73.47%/80.94%/80.77%/81.04%**. Capacity8,192/live4,224 with explicit evictions and a future-position slot is also byte-identical to the generic complete-`KVLiveSpans` route. At live16,448, cached tracing measures **0.305/0.280/0.766 = 1.351 ms/layer**, with score/normalize/PV resources **VGPR16/56/32**, **LDS0/512/11,264**, and scratch0. The directional production gate improves d4K/d16K/d64K/d128K **39.96%/117.15%/262.23%/326.73%** with exact recurrent trajectories and no new resident allocation; tracked-clean d16K confirms **16.756 tok/s (+116.72%)** within **0.20%** of the dirty row. The generic gated split route remains registered fallback and short fused routes remain owners through live6,000. Evidence: `benchmarks/results/2026-08-01-gfx1151-laguna-long-global-gqa6-dim32-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_bf16_spans(...)` | Registered metadata-aware rollback for the gfx1151 exact long-global owner and the default on peer backends. It combines token-loop4 score ownership with exact D32/V80 staged PV, preserving the complete F32/BF16 boundary while reducing V64 stages/barriers by 20%. Its original live4K-128K and production gates remain in `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-vstage80-retained.json`; explicit eviction selects this route. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_bf16_spans(...)` | gfx1151 exact long-global production owner while dense-initial metadata is valid. It retains the full `KVLiveSpans` ABI but removes token/base/eviction metadata and the physical-slot plane from score, exact denominator, and D32/V80 partial PV. Live4,097/16,448/65,664/131,200 is F32/BF16 byte-exact and improves the metadata-aware leaf **4.407%/7.492%/9.163%/9.678%**. Cached resources are score local32/VGPR40/LDS0, denominator local256/VGPR56/LDS512, PV local512/VGPR32/LDS14,336, all scratch0. Complete d16K/d64K/d128K improves **19.481/12.848/9.839 -> 19.791/13.575/10.239 tok/s (+1.592%/+5.657%/+4.067%)** with exact trajectories, noise-flat short guards, and full lifecycle recovery. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-dense-v80-identity-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_bf16_spans(...)` | gfx1151 exact dense-prefix long-global owner only at live >=65,536. It preserves D32/V80 ownership, 16-byte vector width, LDS staging, ordered scalar-F32 FMA chain, complete `KVLiveSpans` ABI, and rounding while marking the one-pass BF16 V loads non-temporal. The formal live4K/16K/64K/128K leaf changes **+0.181%/-3.625%/-1.449%/-1.988%** and is F32/BF16 byte-exact, so the temporal owner remains selected below the measured 64K production crossover. Complete d64K/d128K improves **13.575/10.239 -> 13.673/10.316 tok/s (+0.724%/+0.751%)**; 512/1K/4K are noise-flat and 16K is unchanged by dispatch. Cached tracing names the `<80,true,true>` PV template at local512/VGPR32/SGPR128/LDS14,336/scratch0. Explicit eviction and peer backends retain the temporal V80 route. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-dense-v80-nontemporal-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_dense_prefix_nontemporal_key_value_bf16_spans(...)` | gfx1151 production successor to the V-only non-temporal sibling at live >=65,536. The score lane replaces four scalar BF16 K reads with one aligned 8-byte non-temporal load and exact bit unpack; D32/V80 ownership, arithmetic, V bypass, complete `KVLiveSpans`, rounding, and temporal eviction fallback are unchanged. Against V-only bypass, the repeated exact leaf improves **3.263%/3.578%** at live65,664/131,200 and regresses below the inactive threshold. Complete d64K/d128K improves **13.673/10.316 -> 13.849/10.446 tok/s (+1.287%/+1.265%)**; unchanged 512/1K/4K/16K routes are noise-flat and all trajectories/lifecycle pass. Trace names score `<true,true>` at local32/VGPR40/LDS0 and PV `<80,true,true>` at local512/VGPR32/LDS14,336; both use SGPR128 and scratch0. Evidence: `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-dense-v80-nontemporal-key-value-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch8_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch8_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Registered exact rollback for the gfx1151 prefetch16 owner. Each output wave preloads eight probability/BF16-V pairs before replaying the unchanged chronological F32 FMA chain. Its original exact and production promotion evidence remains in `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-dense-v80-prefetch8-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch16_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage80_prefetch16_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Registered exact rollback for the gfx1151 V128 owner. Each output wave preloads sixteen probability/BF16-V pairs before replaying the unchanged chronological F32 FMA chain. Its original exact and production evidence remains in `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-dense-v80-prefetch16-retained.json`; explicit eviction selects metadata-aware temporal V80. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage128_prefetch16_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage128_prefetch16_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Registered scalar-probability V128 rollback for unaligned score strides. V128 reduces stage/barrier rounds by 37.5% versus V80 while retaining D32 ownership, prefetch16, non-temporal K/V, exact score/denominator and chronological PV arithmetic, complete `KVLiveSpans`, and BF16 rounding. Its original complete d16K/d64K/d128K promotion improved **20.035/14.232/10.975 -> 20.135/14.404/11.093 tok/s (+0.502%/+1.206%/+1.082%)** with noise-flat short routes and full lifecycle recovery. V160 is removed after its 128K gain collapses to 0.291%; V80 remains the eviction/peer rollback. Evidence: `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-dense-v128-prefetch16-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage128_probability_vec4_prefetch16_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage128_probability_vec4_prefetch16_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Current gfx1151 dense-initial exact long-global owner above the 6,000-live fused bound when the score stride is divisible by four. It loads four contiguous FP32 probabilities, applies the unchanged inverse lane-wise, and writes one float4 into the shared V128 stage without changing any values or the chronological prefetch16 PV recurrence. Exact live4K/16K/64K/128K leaves improve **1.107%/2.944%/3.453%/3.437%**. Complete d16K/d64K/d128K improves **20.135/14.404/11.093 -> 20.181/14.620/11.233 tok/s (+0.226%/+1.500%/+1.262%)**, with noise-flat short routes, exact trajectories, and full lifecycle recovery. Trace names candidate PV `<128,true,true,16,true>` at local512/VGPR56/SGPR128/LDS22,528/scratch0 and measures **2,491.917 -> 2,353.578 us (-5.551%)**. Unaligned strides select scalar V128; explicit eviction and peer backends retain V80. Evidence: `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-v128-probability-vec4-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage64_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_tokenloop4_deferrednorm_dim32_vstage64_bf16_spans(...)` | Registered exact rollback for gfx1151 V80 and the measured token-loop4 endpoint for peer backends. One local32 wave retains all six GQA query vectors and visits four consecutive KV tokens, reducing the score grid 4x while preserving each token's four products, wave reduction, scale, denominator, and ordered D32/V64 PV. Live4,097/16,448/65,664/131,200 F32/BF16 leaves are byte-exact and improve **8.55-9.53%** over the prior score owner. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-tokenloop4-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_exact_gated_gqa6_deferrednorm_dim32_vstage64_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_exact_gated_gqa6_deferrednorm_dim32_vstage64_bf16_spans(...)` | gfx1151 production successor to the normalized-score GQA6 path. It preserves the same exp32 max/denominator trees, stores unnormalized exponentials in the existing score plane and 48 inverse denominators in the physical-scratch tail, then multiplies once in the retained D32/V64 PV loader without changing its chronological scalar-F32 FMA chain. Live4,097/16,448/65,664/131,200 F32 context and BF16 gated output are byte-exact while the leaf improves **0.512%/0.864%/0.755%/0.638%**. Cached live16,448 tracing records **0.287-ms score + 0.249-ms denominator + 0.765-ms PV**, local **32/256/512**, VGPR **16/56/32**, LDS **0/512/11,264 B**, and scratch0. The complete one-session gate improves prior exact d4K/d16K/d64K/d128K **0.036%/1.078%/1.490%/2.190%** to **21.670/16.971/9.214/5.725 tok/s** with identical generated hashes and no residency/allocation delta. The normalized sibling remains exact rollback. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-deferrednorm-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim32_vstage64_ctx4096_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_bf16_spans(...)` | Registered D32 rollback and peer-backend default for the admitted 4,096-token context-split arithmetic. The original gfx1151 quality and trajectory gates remain in `2026-08-02-gfx1151-laguna-long-global-late4-ctx4096-retained.json`; gfx1151 production now selects the byte-identical D64 sibling. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim32_vstage64_ctx4096_compensated_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim32_vstage64_ctx4096_compensated_bf16_spans(...)` | Registered compensated-D32 rollback. Its isolated layer-28 quality schedule passes at max KL **0.007761** and **127/127 top-1**; gfx1151 production now selects the byte-identical compensated-D64 sibling. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-layer28-compensated-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_bf16_spans(...)` | Registered one-token-score D64 rollback for the gfx1151 ctx4096 owner. Two output waves share each staged probability/V64 tile, halving the D32 PV grid while preserving every output's chronological scalar-F32 order. It is byte-identical to D32 at live4K/16K/64K/128K and cuts the active leaf **17.258%/10.665%/10.944%**. Local512/VGPR32/LDS19,456B/scratch0. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-dim64-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_compensated_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_compensated_bf16_spans(...)` | Registered one-token-score compensated D64 rollback. It is byte-identical to compensated D32 and cuts the active leaf **19.729%/12.989%/13.425%** at 16K/64K/128K. The historical complete-model D64 promotion improved **17.433/9.868/6.321 -> 17.731/10.120/6.470 tok/s**, preserving every established trajectory and lifecycle. Local512/VGPR32/LDS19,456B/scratch0. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-dim64-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_bf16_spans(...)` | Registered normalized-score rollback for the gfx1151 ctx4096 owner. It replaces only the one-token score producer with exact four-token GQA6 ownership; normalization, D64/V64 partial PV, merge, gate, and BF16 store are unchanged. Live4,097/16,448/65,664/131,200 is F32/BF16 byte-exact and improves **3.93%/9.49%/9.23%/9.69%**. Cached tracing records score local32/VGPR40/LDS0/scratch0 and **4,097 -> 1,025** workgroups at live4,097. Historical d128K improved **8.438 -> 8.689 tok/s (+2.977%)**. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-ctx4096-tokenloop4-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_compensated_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_compensated_bf16_spans(...)` | Registered normalized-score compensated rollback. It preserves the admitted Kahan merge and changes only score ownership. The leaf remains byte-exact and improves **4.38%/9.05%/8.96%/9.36%** at live4K/16K/64K/128K. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-ctx4096-tokenloop4-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_bf16_spans(...)` | Registered metadata-aware fallback for uncompensated layers 32/36/40/44 at live >=98,304 after any explicit eviction. It reuses the retained parallel exact-max denominator, applies each reciprocal in D64/V64 partial PV, and specializes the wrapper-enforced 256-token block mapping with shift/mask arithmetic. Score resources are local32/VGPR40/LDS0/scratch0; denominator is local256/VGPR56/LDS512/scratch0 and PV local512/VGPR32/LDS19,456/scratch0. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-tokenloop4-fixed256-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_bf16_spans(...)` | Registered metadata-aware compensated fallback for layer 28 after explicit eviction. It preserves the admitted Kahan partial and merge arithmetic; the normalized token-loop4 sibling remains the deeper rollback. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-tokenloop4-fixed256-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_bf16_spans(...)` | Registered cached dense-prefix rollback for uncompensated layers 32/36/40/44. It retains the full `KVLiveSpans` ABI but carries identity through score, exact denominator, and D64/V64 partial PV: no physical-slot plane is published or replayed, and BF16 values are addressed directly by logical token. Ordinary leaves improve **2.369-11.337%** over the metadata-aware sibling and d128K improves **9.716 -> 9.839 tok/s (+1.259%)** over the score-only dense checkpoint with exact recurrent state and unchanged lifecycle. Score local32/VGPR40/LDS0, denominator local256/VGPR56/LDS512, PV local512/VGPR32/LDS19,456; all scratch0. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-dense-value-identity-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_bf16_spans(...)` | Registered cached dense-prefix rollback for compensated layer 28. It preserves the admitted Kahan partial/merge path while using direct identity denominator/value transport; the complete compensated leaf improves **2.317-8.948%** at live4K-128K. Explicit eviction selects the metadata-aware sibling. Evidence: `benchmarks/results/2026-08-02-gfx1151-laguna-long-global-dense-value-identity-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Registered prefetch1 rollback for uncompensated layers 32/36/40/44. It changes only aligned K/V cache policy and retains exact score, denominator, partial association, merge/gate/rounding, and full ABI. Its formal ordinary leaf and original promotion evidence remain in `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-ctx4096-nontemporal-key-value-retained.json`; explicit eviction selects the deeper metadata-aware temporal fallback. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Registered prefetch1 rollback for compensated layer 28. It preserves the admitted Kahan partial/merge arithmetic and changes only aligned K/V cache policy. Its original exactness and promotion evidence remains in `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-ctx4096-nontemporal-key-value-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_prefetch4_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_prefetch4_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Current gfx1151 dense-initial owner for ordinary layers 32/36/40/44 at live >=98,304. Each output wave loads four chronological probability/BF16-V operands before replaying the unchanged F32 recurrence. Formal live4K/16K/64K/128K improves prefetch1 **26.822%/4.691%/5.867%/4.536%** at F32/BF16 byte equality. Cached tracing names `<false,64,true,true,4>` at local512/VGPR32/SGPR128/LDS19,456/scratch0. Together with compensated prefetch16 it advances d128K **10.839382 -> 10.974722 tok/s (+1.249%)**, while 512/1K/4K/16K/64K are noise-flat and lifecycle is exact. Evidence: `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-ctx4096-operand-prefetch-retained.json`. |
| `laguna_attention_decode` variant `global_context_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_prefetch16_dense_prefix_nontemporal_key_value_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_split_gated_gqa6_dim64_vstage64_ctx4096_tokenloop4_deferrednorm_compensated_prefetch16_dense_prefix_nontemporal_key_value_bf16_spans(...)` | Current gfx1151 dense-initial owner for compensated layer 28 at live >=98,304. Sixteen operands overlap the longer Kahan dependency chain without changing its chronological arithmetic. Formal live4K/16K/64K/128K improves prefetch1 **31.166%/10.335%/10.274%/9.360%** at F32/BF16 byte equality. Cached tracing names `<true,64,true,true,16>` at local512/VGPR40/SGPR128/LDS19,456/scratch0. Explicit eviction retains the metadata-aware compensated fallback. Evidence: `benchmarks/results/2026-08-03-gfx1151-laguna-long-global-ctx4096-operand-prefetch-retained.json`. |
| `laguna_attention_decode` variant `global_context_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | `laguna_global_attention_decode_wmma_qk_three_term_mixed32_exp32_producer_max_exact_pv_bf16_spans(...)` | gfx1151 diagnostic-only global decode primitive. The retained 32-block mixed owner replaces only scalar QK with three-term BF16 WMMA and keeps producer maxima, exp32 ordered softmax, staged scalar F32 PV, gate, and stores in one launch. The evicted live513/576/639 CPU/control gate has max F32 error **5.59e-9** and **1/1/0** gated BF16 mismatches. Cached 9x50 leaves improve **15.93-20.91%**; tracing names local256/VGPR104/LDS512/scratch0 at 32 blocks. The authoritative 18-prompt/576-step recurrent gate is finite with exact state/lifecycle and **564/576 (97.92%)** top-1, but max KL is **0.741272**, or **14.83x** the ceiling. The selector/harness are removed and production remains unchanged. Retain the primitive only as evidence that tile-local ownership is mechanically profitable but three-term QK association is inadmissible (`benchmarks/results/2026-07-29-gfx1151-laguna-global-single-launch-wmma-qk-rejected.json`). |
| `laguna_sigmoid_router_topk` variant `correction_bias`; scalar/rows Laguna selected/shared MoE chain | `f32` router state; Q4T16/Q6T16 or raw IQ2_XS/IQ3_XXS/IQ4_XS routed experts; Q4 pack8 or raw Q5/Q6/Q8 shared expert | in-tree `moe/laguna_router.hip`; reused `quant/gguf_t16_selected_gemv.hip`, GGUF dense kernels, SiLU/add primitives; `runtime/laguna_moe.py` | `laguna_sigmoid_correction_topk_f32(...)`, `laguna_weighted_sum_rows_bf16_f32w(...)`, `run_laguna_moe_{c1,rows}(...)` | The router preserves separate unbiased sigmoid and correction-only selection buffers, stable lower-ID ties, top-10/256, normalized uncorrected weights, and a separate 2.5-scaled output. The plan resolves exact gfx1100/gfx1151 keys and validates rank-3 source/T16/raw-IQ strides. Production 3072/1024 tests run Q4T16 gate/up, separate BF16 SiLU, Q4/Q6 T16 down, routed sum, and an always-on Q4-pack8/Q4-or-Q6 shared branch with no Qwen shared gate; scalar routed/shared/combined outputs stay within relative-L2 `0.02` of the raw-GGUF CPU oracle. The bounded rows scratch scales every intermediate by token count; a three-row chain is BF16-bit-exact to three scalar runs for both Q4 and Q6 selected/shared down layouts. Cached gfx1151 tracing shows the new three-row weighted reducer at `2.725 us`, one rows-form router/select launch, rows-form selected Q4T16 dual/down, and Q4 pack8 shared prefill kernels; all intended families execute. Earlier c=1 traces remain logits/select `10.820/33.623 us`, Q4T16 dual `250.550 us`, Q6T16 down `121.628 us`, Q4 shared gate/up `20.799/34.264 us`, and Q6 shared down `76.183 us`. Full resident lengths 1/2/7/55/65 plus five B+1 rows are exact for logits, final/pre-final hidden, and all six taps. The pinned Q2 XL layer-1 IQ2/IQ3/Q5/Q6 and layer-47 IQ3/IQ4/Q6/Q8 chains stay within relative-L2 0.02 of direct source-byte CPU oracles; three-row execution is BF16-bit-exact to scalar replay. A full 814-weight W7900 smoke executes two all-layer tokens with finite logits and clean teardown. These are correctness/dispatch diagnostics, not a retained model-throughput claim. |
| `linear` variants `pack8_gemv_decode_bf16_{bf16,f32}_out`; `linear_pair` variants `pack8_gemv_decode_bf16_{bf16,f32}_out` | raw `gguf_q5_k`, `gguf_q6_k` | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip` | `gguf_q5_k_pack8_gemv_decode_bf16_{bf16,f32}_out(...)`, `gguf_q5_k_pair_pack8_gemv_decode_bf16_{bf16,f32}_out(...)`, `gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out(...)` | The singleton local128 schedule hoists each output-pack's exact Q5 d/dmin and packed scale/min coefficients to 1,024 B LDS while preserving the established per-thread K order and reduction tree. The registered pair uses the same forced-inline block body and independent `blockIdx.y` gate/up workgroups, so each output remains BF16-bit exact to two singleton launches. Synthetic K3072/N1024 pair output and actual Laguna layer-1 CPU-quant/full-MoE oracles pass; the focused bundle reports 107 passes. Actual `blk.1` shared gate/up counterbalanced wall is **28.148 -> 16.373 us/pair (-41.83%; 1.719x)**. Cached W7900 tracing names `gguf_q5_k_pair_pack8_decode_out_kernel` at **12.4 us median** in isolation and **19.321 us median** across 46 all-layer calls, local128/VGPR72/LDS1024/scratch0. The dirty full-model short trace removes 46 launches/token (1,010 -> 964), moves the pair family **1.561 -> 0.891 ms/token (-42.94%)**, kernel sum **18.983 -> 18.240 ms (-3.92%)**, and span **22.878 -> 21.963 ms (-4.00%)** with exact d32 IDs/lifecycle. Laguna c=1 requests only registered decode pairs and retains two singleton launches on registry/shape miss; rows>1 are unchanged. The BF16 shared pair's clean category gate promotes D5 at **44.501 h32 decode tok/s (+3.30%)** with exact suite/state/lifecycle. The retained F32 unequal-width sibling flattens each query+per-head-gate pack range into one grid, preserves both singleton output buffers byte-for-byte at K3072 N6144+48 and N9216+72, and retains the same fail-closed c=1-only dispatch. Actual `blk.0/1` query+gate pairs improve **44.827 -> 34.499 us (-23.04%)** and **56.790 -> 42.123 us (-25.83%)**. Clean cached traces confirm exactly 47 F32 pair launches and **964 -> 917 dispatches/token**, local128/VGPR48/SGPR128/LDS1024/scratch0. Short/512/1K/near-4K kernel sum improves **3.13%/1.91%/1.70%/1.21%**, median span improves **3.21%/2.15%/1.95%/1.38%**, and profiled child throughput improves **3.89%/2.87%/1.81%/1.48%**, with exact IDs/finite logits/teardown. The complete suite promotes D6 at **45.433 h32 decode tok/s (+2.09%)** and **11.921 h32 E2E (+0.52%)**, with every category positive and prefill neutral. The new raw-Q6 F32 sibling flattens both pack ranges and preserves the established singleton arithmetic for 47 equal-width K/V pairs plus layer 47's unequal query/gate pair. Production synthetic K3072 N1024+1024 and N9216+72 outputs are byte-exact, and actual global/SWA K/V plus layer-47 query/gate improve **37.36%/36.65%/8.80%**. A dirty cached full-model trace names 48 `gguf_q6_k_pair_pack8_decode_out_kernel` calls, removes **48 launches/token (917 -> 869)**, runs local128/VGPR56/SGPR128/LDS512/scratch0, moves the Q6 F32 family **1.465 -> 0.934 ms/token (-36.26%)**, total kernel sum/span **17.689 -> 17.256 ms (-2.45%)** / **21.274 -> 20.750 ms (-2.46%)**, and preserves exact IDs/finite logits/teardown. Clean context/category promotion promotes D7 at **46.409 h32 decode tok/s (+2.15%)** with exact suite/state/lifecycle. The next exact Q5 attention-output tile16 candidate remained BF16-bit exact and scratch-free at K6144/K9216 N3072, but its best local256/VGPR88/LDS2048 schedule regressed actual-weight HIP-event time **16.69%/19.04%**; it was removed before full-model measurement (`benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-q5-output-tile16-rejected.json`). D12 adds separately registered raw-Q5 BF16-output `wave32x2_gemv_decode_bf16_bf16_out` singleton and F32 unequal-pair siblings; the F32 singleton wrapper is an unregistered bit-oracle. One local32 wave reconstructs the four baseline logical-thread groups for two outputs, preserving every `[t,t+128]` accumulation, four 16..1 trees, and the final 0..3 group addition while replacing coefficient LDS/barriers with wave shuffles. K256/K512, K6144/K9216 N3072, and K3072 N6144+48/N9216+72 are bit-exact; a shared-weight 16-step gate matches all 48 hidden rows, logits, complete K/V/live spans, reset, and lifecycle. Cached W7900 trace records local32/VGPR96/LDS0/scratch0 and expected 1,536/4,644 workgroups. Formal 50-warmup/15x200 actual-weight leaves improve **13.63-24.80%** in HIP events and **10.39-23.73%** in synchronized wall. Clean short/512/1K/near-4K output/query-gate, kernel sum, span, and profiled child rows all improve. The counterbalanced full suite moves h32 decode **47.046 -> 48.987 tok/s (+4.124%)** and E2E **11.997 -> 12.117 (+1.001%)**, with every category positive and unaffected prefill neutral. gfx1100 backend metadata defaults both role siblings; gfx1151, rows>1, registry/shape misses, and explicit disable retain pack8 (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d12-q5-wave32x2-{correctness,retained}.json`). The removed D13 Q5 shared pair+SiLU composite was BF16-bit exact and shortened its inclusive boundary **5.46-6.93%** at local256/VGPR88/LDS1536/scratch0, but its body regressed **9.28-11.07%** and every clean context's total kernel sum regressed **0.390-0.707%**. Its HIP body/export, wrapper/registration, runtime selector, and tests are removed; the pair+separate-SiLU chain remains the only route (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d13-q5-shared-silu-rejected.json`). The later quality-gated P1 Vulkan-style raw-Q5 local64 row4 screen approximately halves all four actual output/query-gate event windows (**48.12-50.38%**) at local64/VGPR96/LDS32/scratch0, but combined/output-only/query-gate-only 18-prompt maximum KL is **0.461353/0.893206/1.35822** versus the `0.05` ceiling. Its bodies, wrappers, selectors, and tests are removed; exact D12 remains the only retained c=1 route (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p1-q5-row4-rejected.json`). The retained exact D12 fixed-metadata sibling reads each output pair's fixed 16-byte Q5 metadata with two wave-uniform `s_load_b128` operations per superblock. It preserves all eight accumulator chains and reductions while removing 32 coefficient `ds_bpermute` operations and reducing logical VGPR **89 -> 72** with zero LDS/scratch. First/last global and Q5-SWA output/query-gate rows are bit exact and improve event **19.80-25.19%** and synchronized wall **17.59-24.07%**; the 16-transition full-state gate and cached two-transition trace pass with **94+94** candidate calls at local32/VGPR72/LDS0/scratch0 and no coefficient-publication c=1 Q5 calls. Both clean process orders improve Q5 **22.68-23.12%**, kernel sum **2.35-6.34%**, span **2.17-5.58%**, and child throughput **2.26-4.41%** at every context. Both complete 18-prompt orders move h32 **54.476 -> 57.711 tok/s (+5.938%)** with every train/heldout category decode and E2E row positive. gfx1100 defaults fixed metadata with independent role rollback; rows>1 and unsupported backends retain the existing exact routes (`benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-q5-fixed-metadata-{correctness,retained}.json`). A later exact raw-Q6 fixed-metadata pair transferred the standalone singleton's local128 cooperative `d*scale` schedule and improved first/last actual pair event/wall **21.92-40.52% / 22.60-39.01%**. Full state and resources passed, and clean short/512/1K Q6-family, kernel-sum, and span rows all improved. It was nevertheless removed under the frozen no-waiver gate because two-order short profiled-child throughput changed **53.759 -> 53.248 tok/s (-0.951%)**, beyond the -0.5% guard; near-4K and category work were stopped (`benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-q6-fixed-metadata-pair-{correctness,rejected}.json`). The retained BF16 shared-Q5 pair instantiates the fixed-metadata wave32x2 block for sparse layers 1-46. Production K3072/N1024+N1024 output is byte-exact. Codegen contracts **701 -> 508** static instructions, removes all three barriers and rounded **1,024 -> 0 B** LDS at logical VGPR **72 -> 73** with no spills. Formal first/last actual pairs improve event **27.45-27.61%** and wall **26.88-27.48%**. Full logits, all 48 hidden/47 routed boundaries, active K/V/`KVLiveSpans`, reset, lifecycle, and promoted default-vs-pack8 rollback are exact. Cached tracing records 46 candidate calls/token at local32/VGPR80/SGPR128/LDS0/scratch0, unchanged **723 dispatches/token**, and no generic shared-pair calls. Both clean orders improve shared-pair work **45.99-47.13%**, kernel sum **1.32-3.02%**, span **1.43-2.62%**, and profiled child **0.89-3.33%**. Both complete 18-prompt orders move h32 **59.500 -> 60.942 tok/s (+2.425%)**, with every train/heldout category decode positive and all guards passing. gfx1100 defaults the local32 sibling with explicit pack8 rollback; layer 47 Q6, rows>1, misses, and unsupported backends retain existing routes (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-shared-q5-fixed-metadata-{correctness,retained}.json`). |
| `attention_projection_quad` variants `mixed_pack8_gemv_decode_bf16_f32_out`, `mixed_q6_fixed_meta_pack8_gemv_decode_bf16_f32_out`, `mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out`, `mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out` | raw role tuples `Q5/Q6/Q6/Q5` and `Q6/Q8/Q8/Q6` | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.{hip,py}` | role-qualified registry wrappers selected by `launch_laguna_mixed_attention_projections(...)`; gfx1100 local32 default with explicit local128 fixed-Q6, generic-mixed, and pair/singleton rollback | Retained Laguna c=1 default. Layers 0-46 flatten four independent fixed-metadata Q5 wave32 pair owners with generic local128 Q6 packs; layer 47 flattens generic local128 Q6 query/gate plus Q8 K/V packs. Actual layers 0/1/46/47 are F32-bit exact and improve inclusive event **4.52-16.57%** / wall **3.65-14.23%**. A 16-transition shared-weight gate matches full logits, all 48 hidden/47 routed boundaries, active K/V and every `KVLiveSpans` field, reset, and lifecycle. Cached tracing records 47 Q5/Q6 plus one Q6/Q8 call per transition, **772 -> 723 dispatches/token**; Q5/Q6 is local128/VGPR88/SGPR128/LDS512/scratch0 and Q6/Q8 is local128/VGPR56/SGPR128/LDS512/scratch0. Both clean process orders improve the projection family **2.02-3.35%**, kernel sum **0.09-0.35%**, span **0.69-1.56%**, and child throughput **1.06-2.92%** at every context. Both complete 18-prompt orders move h32 **57.833 -> 58.425 tok/s (+1.024%)**, with every train/heldout category decode positive and every E2E/prefill/TTFT guard passing (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-attention-{correctness,retained}.json`). The retained fixed-Q6 sibling leaves Q5 and Q8 ownership unchanged while cooperatively publishing the Q6 packs' exact `d*scale` metadata. Actual layers 0/1/46/47 are F32-bit exact and improve complete projection event **9.61-41.52%** / wall **8.50-38.85%**. A 16-transition gate matches full logits, all hidden/routed boundaries, active K/V/`KVLiveSpans`, reset, and lifecycle. Cached tracing records the expected 94+2 calls over two transitions at unchanged **723 dispatches/token**; Q5/Q6 is local128/VGPR88/SGPR128/LDS1024/scratch0 and Q6/Q8 is local128/VGPR48/SGPR128/LDS1024/scratch0. Both clean process orders improve projection work **8.08-10.10%**, kernel sum **0.73-1.26%**, span **0.57-1.49%**, and profiled-child throughput **0.01-0.84%**. Both complete 18-prompt orders move h32 **58.466 -> 59.211 tok/s (+1.275%)**, with every train/heldout category decode positive and all E2E/prefill/TTFT guards passing. gfx1100 defaults the fixed-Q6 variant with explicit `use_mixed_q6_fixed_meta_attention=False` generic-mixed rollback (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-q6-fixed-metadata-{correctness,retained}.json`). The Q5/Q6-only all-local32 sibling is now the retained gfx1100 default: each Q5 wave reuses the fixed-address helper, while each Q6 wave preserves four original local128 partitions and their exact final add order without LDS. Production outputs, full model state, and explicit default-vs-local128 rollback are bit-exact; first/last actual layers improve event **11.39-14.77%** and wall **11.24-15.72%**. Cached tracing records 47 local32/VGPR80/LDS0/scratch0 calls plus one retained layer-47 fixed-Q6 Q6/Q8 call at unchanged **723 model kernels/token**. Both clean orders improve projection/kernel/span/child **7.00-8.12% / 0.49-2.12% / 0.45-2.77% / 0.20-1.29%**; both complete category orders move h32 **60.900 -> 61.732 tok/s (+1.367%)** with every train/heldout category positive. Explicit disable or registry miss restores local128 fixed-Q6 without a quant/backend branch (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-local32-projection-{correctness,retained}.json`). A separately registered W7900 diagnostic now pairs one exact Q5 and Q6 output pair per local32 wave and reuses each BF16 activation register. K256/1024/3072/9216 synthetic boundaries and all 47 actual layers are F32-bit exact; CPU top-1 is 100%, actual first/last event/wall improve **2.86-7.16% / 2.45-5.43%**, and repository codegen is logical/allocated VGPR **92/96**, SGPR **70/128**, LDS/private/spills/scratch0, **1,245 instructions / 7,208 bytes**, zero barriers. Cached tracing names grid/local **99,072/32** with no compiler under profiling. A temporary explicit/default-off gfx1100 owner passed shared-weight 16-transition full logits/IDs, all **48 hidden + 47 routed boundaries**, active K/V and every `KVLiveSpans` field, reset/re-prefill, and lifecycle byte-exact at KL0/top-1 100%. Cache-only full-model tracing recorded exactly **47 candidate calls + one unchanged layer-47 Q6/Q8 call/token**, zero candidate prefill calls, and **723 model kernels/token**; global/SWA candidates remained grid **99,072/148,608**, local32/VGPR96/LDS0/scratch0. The frozen clean short gate rejects runtime selection: both orders improve the Q5/Q6 projection **5.31%/5.85%** and kernel sum **0.42%/1.44%**, but order A regresses dispatch span **1.265%** and order B regresses profiled-child throughput **0.681%**, beyond the 0.5% guards. Remaining contexts/categories stop; capability/session/CLI/dispatch integration is removed and production files again match primitive commit `0de4f45d8`. The exact key remains gfx1151-excluded and diagnostic (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-{correctness,runtime-correctness,rejected}.json`). |
| `linear`, `linear_pair`, `linear_triple` mixed and `tiled_*` variants | `fp16_weight` | `hipengine/kernels/hip_gfx1100/linear/laguna_f16_projection.hip` | single BF16/F32 activation to BF16/F32 output; BF16-to-F32 dual/triple; exact rows>1 8x4/16x4 tiled single/triple; model-neutral `launch_f16_weight_linear{,_pair,_triple}(...)` | Source-F16 Laguna Q/K/V/gate/O projection foundation. Decode keeps the original 256-thread GEMV reduction. The LPF-1 tile preserves that thread-local K sequence and reduction tree while reusing each activation across four output columns and each weight across 8 rows below M16 or 16 rows at/above M16. Synthetic rows 2/3/4/5/17 are bit-exact for F32 and BF16 output. Clean same-session rows 2..128 are exact and all faster, with a 2.0538x weighted profile, 55 rows 23.460->48.760 tok/s (2.0784x), and 128 rows 23.374->50.240 (2.1494x). The clean two-repeat ten-prompt gate moves the prior bulk-GEMV row 23.333->48.560 tok/s (+108.12%), TTFT 3.481->1.692 s, and h32 E2E 5.719->8.717 tok/s while all IDs/categories/Poolside/lifecycle gates pass and decode stays neutral. gfx1151 therefore defaults to tiled from two rows; rows=1 and unsupported backends retain GEMV. Cached gfx1151 trace names `laguna_f16w_tiled_exact_kernel<unsigned short, 16>` at 3.798 ms for the 55x9216x3072 O shape, workgroup 256, grid 196608x4, 96 VGPR, 128 SGPR, 512 B LDS, and zero scratch. A faster reassociated 16x16 WMMA control reached 60.65 tok/s but changed three of ten free-running trajectories and was removed rather than admitted. AR-O2 now has a separately registered replacement candidate that reads the same resident row-major F16 bytes directly, converts BF16 activations to F16 in registers, and accumulates 16x16x16 F16 WMMA into FP32 without a sidecar or inference-time staging. Seeded M16/M17 synthetic F32 output passes the CPU gate at KL <=0.05/top-1 >=90%, BF16 output is the exact RNE boundary of its FP32 result, and triple dispatch matches three CPU matrices. Cached gfx1151 tracing names the F32/BF16 leaves at **2.364/2.124 us**, local32, VGPR32, LDS0, scratch0. It remains explicit-only pending production M16-512 timing and the complete quality lane. The original production-head fixture remains against FP32 CPU matmul; its decode trace shows single BF16->F32 `102.431 us`, single BF16->BF16 `104.756 us`, dual `129.122 us`, and triple `127.198 us`, all 16 VGPR, 512 B LDS, zero scratch. |
| `head_rmsnorm+partial_rotary` variant `positions_f32`; `attention_gate` softplus variants | `laguna_f32_weight`, `f32` | existing exact `fused/gguf_ops.hip` head-normalize/rotate body plus `fused/laguna_attention.hip` | `materialize_laguna_rope_tables(...)`, `launch_laguna_head_rmsnorm_rope(...)`, `laguna_softplus_head_gate_f32_{,bf16_}out(...)` | Host tables use the independent Transformers-validated Laguna CPU YaRN/plain equations and absolute indices. gfx1151 production-shape tests cover partial-64 YaRN with 48 Q/8 KV heads and full-128 plain SWA with 72 Q/8 KV heads, dim 128. Cached traces: fused head RMSNorm+RoPE `13.505/13.426 us` and FP32/BF16 softplus broadcast `2.485/1.764 us`; all zero scratch, with 16/24 VGPR respectively. |
| `head_rmsnorm+partial_rotary+kv_write` global/SWA variants `*_f32_bf16_spans` | `laguna_f32_weight` | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip` | `laguna_{global,swa}_head_rmsnorm_rope_write_kv_f32_spans(...)`; `LagunaGGUFResidentSession(..., use_head_kv_fusion=True)` | Current-P4 retained gfx1100 default recomposition of the exact historical head/KV body over retained split attention and gated reducers. One local256 block owns each query/KV head, preserves the established FP32 RMSNorm/partial-RoPE arithmetic, writes F32 Q/K plus RNE BF16 K/V, and consumes all five `KVLiveSpans` fields. Global page-256 and SWA ring/wrap boundary fixtures, all 48 hidden/47 routed outputs, full logits, active K/V/spans, reset, and lifecycle are exact. First/last actual global/SWA layers improve inclusive event **33.05-39.36%** and wall **33.41-39.13%**. Cached tracing names global/SWA siblings at local256, VGPR16, SGPR128, dynamic LDS1024, scratch0. gfx1151, rows/prefill, explicit disable, and registry miss retain the registered two-launch chain. Clean short/512/1K/near-4K kernel sum improves **0.14-0.87%** after a predeclared reverse short confirmation resolves the first +0.035% noise row to pooled **-0.462%**; span and profiled child improve at every context. The exact two-order 18-prompt gate moves h32 decode **51.872 -> 52.391 tok/s (+1.001%)**, with every train/heldout category decode positive and all E2E/prefill/TTFT guards passing. gfx1100 capability metadata defaults the fused body with explicit rollback (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-head-kv-{correctness,retained}.json`). A separately registered post-wave0 global sibling keeps local256 and the complete current-P4 ABI while wave 0 exactly replays the RMS tree in two dynamic barriers. Synthetic page-boundary fixtures match retained P4 and the registered unfused chain bit-for-bit and pass CPU KL/top-1; all **12/12** actual global layers preserve F32 Q/K, BF16 K/V, and every span field. Direct cached-C layer-0/44 event/wall improve **20.57-22.55%**. Clang-22 contracts **814 -> 662 instructions**, **3,976 -> 3,296 B**, logical VGPR **15 -> 12**, and SGPR **69 -> 67**; cached tracing names one local256/VGPR16/SGPR128/static-LDS0+dynamic-1024/scratch0 call at **6.440 us** with no compiler. The same source regresses SWA **7.21-7.38%**, so SWA is excluded and retained P4 remains its sole route. A temporary false/default-off gfx1100 owner passed exact shared-weight bulk prefill, all **48 hidden + 47 routed** boundaries, 16 decode transitions, active K/V and every span field, reset/re-prefill, and lifecycle at KL0/top-1 100% with zero allocation delta. Cache-only tracing recorded exactly **12 candidate global + 36 retained-P4 SWA calls/token**, zero retained-global decode calls, unchanged **45 IQ3 wave10 / 678 model kernels/token**, local256/VGPR16/SGPR128/dynamic-LDS1024/scratch0, and no compiler. The frozen short gate rejects ownership despite global-family and kernel-sum wins in both orders: order A regresses profiled-child throughput **0.859%**, and order B regresses dispatch span **0.810%**. Favorable pooled child/span rows cannot waive either per-order failure. Long contexts/categories stop; capability/plan/session/CLI seams are removed, current-P4 global/SWA and the registered unfused chain remain the only runtime routes, and the exact candidate stays diagnostic (`benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-head-wave0-tree-{design,correctness,runtime-correctness,rejected}.json`). |
| `laguna_kv_write`, `laguna_attention_decode`, `laguna_attention_prefill` global/SWA span variants | `bf16` | in-tree `attention/laguna_kv_attention.hip`; global keeps the proven block-256 page-table structure | `allocate_laguna_kv_cache(...)`, scalar and rows forms of `laguna_global_{write_kv,attention_*}` / `laguna_swa_{write_kv,attention_*}` | The owner allocates 12 global caches at admitted context and 36 physical 512-token rings: exact 4K payload is 264 MiB across 243 tracked payload/metadata allocations, with teardown returning counters to baseline. Every scalar/bulk writer and reader consumes complete `KVLiveSpans`; bulk rows infer consecutive absolute positions from the span's query-position scalar. Production 48/8 and 72/8 GQA decode matches direct FP32 CPU attention through 510/511/512/513 and repeated wraps. The bulk gate seeds 508 rows, then proves an eight-row 508..515 global/SWA chunk bit-exact against append+decode per row, including permuted physical offsets, explicit eviction, and future ring overwrites that must not hide earlier-row keys. Cached gfx1151 tracing records bulk global/SWA attention at `1058.825/1672.777 us` versus eight scalar readers totaling about `1396/5547 us`; bulk writers are `1.683-59.231/1.563-49.854 us`, with expected kernel names. The older 1026-row scalar trace remains `1.603/1.523 us` median writers, `730.650 us` global read, and `1123.828 us` SWA boundary median. Full resident lengths 1/2/7/55/65 plus five B+1 rows now match serial live-span metadata and every live BF16 K/V row exactly (`2026-07-22-gfx1151-laguna-bulk-prefill-verifier-correctness.json`). LPF-5 adds a separately registered diagnostic SWA prefill `swa_context_rows_wave32_exact_spans`: one wave reconstructs the baseline 128-thread stride-64/32/16..1 reduction tree exactly while removing per-token block barriers. The 508..515 wrap/eviction fixture is F32 byte-exact to both baseline bulk and scalar attention. A balanced production 128-row/512-window leaf probe improves **20.434 -> 9.229 ms (2.214x)**; cached tracing confirms the candidate at **9.123 ms median**, workgroup 32, 32 VGPR, 128 SGPR, zero LDS/scratch versus baseline **20.355 ms**, workgroup 128, 16 VGPR, 1,024 B LDS, zero scratch. The clean shared-weight full-model gate promotes this variant on gfx1151 after exact complete logits/hidden/cursors and 512/1K/4K gains of **+8.31%/+12.85%/+14.06%**; a prior complete timing pass independently reproduced **1.082/1.128/1.140x**. Backend capability metadata selects wave32 exact only on gfx1151, while explicit baseline rollback and unmeasured-backend defaults remain. The separately registered `swa_context_token4_exact_spans` decode candidate uses four waves for four independent logical-slot dots, stores unscaled dots/physical slots in 4,120 B dynamic LDS, and then preserves baseline-order max, contracted score-minus-max, denominator, and value accumulation. All seven KV tests pass byte-exactly through 510/511/512/513 and repeated wrap 1024/1025 with an explicit eviction. Cached W7900 tracing measures the six candidate calls at **237.722 us median** versus baseline **792.747 us (-70.01%; 3.335x)**, local128/VGPR24/static-LDS0/scratch0. Clean full-model traces preserve exact IDs/state/lifecycle and move short/512/1K/near-4K SWA **4.202/27.776/27.846/27.901 -> 2.118/13.111/13.096/13.104 ms/token (-49.60%/-52.80%/-52.97%/-53.03%)**. The complete ten-prompt category gate promotes gfx1100 token4 by backend capability at **43.081 h32 decode tok/s (+10.919% vs D3)** and **11.760 h32 E2E (+2.724%)**, with prefill within -0.223% and explicit baseline rollback; gfx1151/unmeasured backends retain baseline. The exact D10 token8 sibling passed empty/short/adversarial/full/wrap/eviction and complete shared-weight state gates, and improved every clean short/512/1K/near-4K SWA, kernel-sum, span, and profiled-child row. It nevertheless failed the predeclared complete-suite non-regression gate: aggregate h16 E2E changed **-0.055%**, general-English h16 decode/E2E changed **-0.535%/-0.254%**, and other h16 category E2E rows also dipped. The token8 wrapper, registry entry, kernel, tests, and selector were removed; gfx1100 remains on retained token4 with baseline fallback and no D10 rollback debt (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d10-swa-token8-rejected.json`). D17 later recomposed D10's exact token8 arithmetic only inside an all-or-none D15-derived head/KV and attention/gate bundle. It passed empty/wrap/eviction/adversarial fixtures, all-48-layer state/KV/lifecycle, resource ceilings, and every clean short/512/1K/near-4K mechanical row at **679 dispatches/token**. The complete suite moved h32 decode **48.971 -> 50.668 tok/s (+3.465%)** and improved every category's h16/h32 decode/E2E, but aggregate TTFT changed **+0.795%**, outside the frozen 0.5% guard. Per the predeclared any-failure rule, all D17 kernels/exports/wrappers/registrations/selector/tests are removed without standalone D10/D14/D15 debt; D12 remains canonical at **48.987 tok/s** (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d17-attention-boundaries-rejected.json`). P2.1 adds default-off `global_context_split_exact_spans` and `swa_context_split_exact_spans` siblings: independent local32 score blocks write caller-owned FP32 score/int32 physical-slot scratch, then local256/local128 reducers reproduce the retained global/token4 association exactly. Synthetic boundaries select global `>=127` and SWA `>=65`; actual layer 0/44 and 1/47 context-128 event/wall windows improve **8.53-13.31%**, full logits/hidden/routed/KV/`KVLiveSpans`/reset/lifecycle are exact, and HSACO metadata reports score/reducer VGPR **8/19** global and **7/11** SWA with zero private scratch. Clean short/512/1K/near-4K attention improves **15.66-23.28%**, complete kernel sum **2.67-16.11%**, span **4.65-14.63%**, and profiled-child throughput **1.19-17.58%**. The exact two-order 18-prompt gate moves h32 decode **50.093 -> 51.436 tok/s (+2.681%)** and E2E **12.098 -> 12.158 (+0.496%)**, with every train/heldout category positive and prefill/TTFT inside guard. gfx1100 capability metadata now defaults global `>=127` and SWA `>=65`; below-threshold calls, explicit disable, gfx1151, and unsupported backends retain the registered readers (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-split-exact-{correctness,retained}.json`). The retained-default gfx1100 exact SWA tile16 refinement assigns one local256 block to 16 score slots, preserves every wave32 dot, and feeds the unchanged split reducer. Two crossover screens select live `>=257`; actual layers 1/47 improve hot event **0.36%/0.44%** and wall **0.78%/0.36%** at complete F32 byte equality. A 150-transition gate matches logits, all 48 hidden/47 routed boundaries, active KV/`KVLiveSpans`, reset, and lifecycle exactly. Cached tracing records the score producer at local256/VGPR32/LDS0/scratch0 and grid `72 x 17`, followed by the existing reducer; no allocation is added. The global tiled sibling is removed after later regressions. Two process orders improve pooled 512/1K/near-4K SWA attention **0.571%/0.344%/0.208%** and total attention **0.461%/0.272%/0.056%**, while the complete fallback gate is exact/non-regressive. gfx1100 defaults live `>=257`; explicit disable retains P2.1 (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-swa-tile16-{correctness,retained}.json`). The removed P2.2 online siblings used tile32 FP32 `(m,l,o)` partials and an ascending stable merge. They were primitive-close and improved actual context-128 global/SWA event windows **52.56-66.56%**, but failed the complete quality gate in combined, global-only, and SWA-only modes at maximum KL **1.77384/1.16169/1.64542**; all kernels, wrappers, workspace, selectors, and tests are removed (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-online-rejected.json`). P4.1 adds separately registered exact gated split reducers: they retain the F32 context, apply the existing FP32 softplus per-head gate, and write the identical RNE BF16 gated context in the reducer. The registered unfused chain remains the below-threshold, explicit-disable, registry-miss, and non-gfx1100 fallback. Actual first/last global/SWA layers at live 128/257 are bit-exact and improve inclusive event **3.00-10.05%** and wall **2.89-9.60%** with no new allocation. The 18-prompt two-order gate moves h32 decode **51.497 -> 51.825 tok/s (+0.637%)**; every train/heldout category's h16/h32 decode improves, while E2E/prefill/TTFT remain within guards. Cached tracing names the global/SWA gated reducers at local256/local128, VGPR24, LDS512, scratch0, with the expected retained score producers and no standalone gate (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-split-gate-{correctness,retained}.json`). A post-reaudit default-off SWA sibling screened removal of only the redundant block barriers in the thread-0 four-score maximum scan. Live 65/257 fixtures and a 16-transition all-state gate are bit exact; first/last actual SWA layers at live 70/128/257/512 improve inclusive event **0.18-0.50%** and wall **0.18-0.45%** in every row. Cached tracing records `laguna_swa_attention_split_exact_gated_reduce_kernel<false>` at local128/VGPR24/LDS512/scratch0. The frozen clean gate rejects it: two short process orders pool reducer/SWA **-0.0008%/-0.0216%**, kernel sum **+0.446%**, and median dispatch span **+1.173%**, beyond the 0.5% guard. The candidate instantiation, exports, wrappers, registry keys, selector, and tests are removed before category work; only the original reducer remains (`benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-no-max-sync-{correctness,rejected}.json`). A distinct default-off wave-local sibling then removes the reducer's remaining consumer barriers and LDS: each logical wave independently replays the identical scalar max/denominator order, broadcasts four weights with width-32 shuffles, and preserves each dimension's slot-order FMA chain. Live 65/257 fixtures, full logits, all 48 hidden/47 routed boundaries, active K/V plus every span field, reset, and lifecycle are exact. First/last actual SWA layers at live 70/128/257/512 improve event/wall **4.87-18.91% / 4.84-18.96%** in every row. Cached tracing names the candidate 72 times at local128/VGPR24/SGPR128/LDS0/scratch0 and no retained reducer calls. Two process orders improve reducer/SWA **4.63-5.22% / 4.24-4.55%**, kernel sum **0.94-1.98%**, and span **0.61-1.69%** at short/512/1K/near-4K. The exact two-order 18-prompt gate moves h32 **52.211 -> 52.514 tok/s (+0.580%)** with every train/heldout category decode positive and all E2E/prefill/TTFT guards passing. gfx1100 defaults wave-local; explicit false and unsupported backends retain shared statistics (`benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-wave-local-{correctness,retained}.json`). A separately registered local64 packed-dim2 reducer now keeps all 72 one-head workgroups while each lane carries two adjacent dimensions. Empty/tied/extreme/live-65..512/wrap/evict fixtures are F32/BF16 bit-exact to the retained local128 wave-local reducer. Static/cached resources are local64/wave32/grid4608, logical/allocated VGPR18/24, SGPR50/128, LDS0/private0/spill0/scratch0. The frozen layers-1/46/47 live-70/128/257/512 50-warmup/15x200 screen improves every full score+reducer event row **0.275-0.685%** and wall row **0.295-2.294%**. Its temporary default-off gfx1100 selector preserves 16-transition full logits/hidden/routed/KV/span/reset/lifecycle bytes; cached tracing records **36** dim2 calls/token, zero retained/shared reducer calls, and unchanged **723 model kernels/token**. The frozen two-order clean gate rejects runtime promotion: short reducer/SWA improve **0.244%/0.060%**, but context-512 reducer/SWA regress **0.073%/0.247%**. The 1K/near-4K runs are stopped, categories are skipped, and selector/capability integration is removed; the separately registered primitive and gfx1151 key exclusions remain diagnostic (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-swa-local64-dim2-reducer-rejected.json`). All other rows are correctness/dispatch diagnostics, not a retained target-throughput claim. |
| `laguna_sigmoid_router_topk` variants `correction_bias`, `correction_bias_compact_wave32`; scalar/rows Laguna selected/shared MoE chain | `f32` router state; Q4T16/Q6T16 or raw IQ2_XS/IQ3_XXS/IQ4_XS routed experts; Q4 pack8 or raw Q5/Q6/Q8 shared expert | in-tree `moe/laguna_router.hip`; reused `quant/gguf_t16_selected_gemv.hip`, GGUF dense kernels, SiLU/add primitives; `runtime/laguna_moe.py` | `laguna_sigmoid_correction_topk_{f32,compact_wave32_f32}(...)`, `laguna_weighted_sum_rows_bf16_f32w(...)`, `run_laguna_moe_{c1,rows}(...)` | The router preserves separate unbiased sigmoid and correction-only selection buffers, stable lower-ID ties, top-10/256, normalized uncorrected weights, and a separate 2.5-scaled output. The plan resolves exact gfx1100/gfx1151 keys and validates rank-3 source/T16/raw-IQ strides. Production 3072/1024 tests run Q4T16 gate/up, separate BF16 SiLU, Q4/Q6 T16 down, routed sum, and an always-on Q4-pack8/Q4-or-Q6 shared branch with no Qwen shared gate; scalar routed/shared/combined outputs stay within relative-L2 `0.02` of the raw-GGUF CPU oracle. The bounded rows scratch scales every intermediate by token count; a three-row chain is BF16-bit-exact to three scalar runs for both Q4 and Q6 selected/shared down layouts. Cached gfx1151 tracing shows the new three-row weighted reducer at `2.725 us`, one rows-form router/select launch, rows-form selected Q4T16 dual/down, and Q4 pack8 shared prefill kernels; all intended families execute. Earlier c=1 traces remain logits/select `10.820/33.623 us`, Q4T16 dual `250.550 us`, Q6T16 down `121.628 us`, Q4 shared gate/up `20.799/34.264 us`, and Q6 shared down `76.183 us`. Full resident lengths 1/2/7/55/65 plus five B+1 rows are exact for logits, final/pre-final hidden, and all six taps. The pinned Q2 XL layer-1 IQ2/IQ3/Q5/Q6 and layer-47 IQ3/IQ4/Q6/Q8 chains stay within relative-L2 0.02 of direct source-byte CPU oracles; three-row execution is BF16-bit-exact to scalar replay. A full 814-weight W7900 smoke executes two all-layer tokens with finite logits and clean teardown. A separately registered exact D11-derived diagnostic now replaces only the repeated block-wide selector with eight wave-local top-10 lists and a register-resident wave-0 merge. Hidden-17/3072 random/tie/extreme fixtures and every actual Q2 XL router preserve logits, both score buffers, selected IDs, both weight buffers, and repeated self-reset state byte-for-byte. Static resources are wave32/VGPR64/SGPR42/LDS680/private0/spill0 plus 1,024 B dynamic projection scratch; cached tracing names the local256 candidate at allocated VGPR64/scratch0. Fresh all-47-layer event/wall improves split **23.26%/23.23%** and old D11 **4.83%/4.84%**. Its temporary default-off runtime owner passed shared-weight 16-transition byte identity and cached 47-call/**723 -> 676 model-kernel/token** tracing with one self-resetting four-byte counter. The frozen two-order short clean gate nevertheless reverses the isolated win: full-model router-family time regresses **14.42%/13.69%**, complete kernel sum regresses **0.736%/1.422%**, and order-B dispatch span regresses **1.370%**. Remaining contexts and categories stop; runtime selector/counter integration is removed, the registered split chain is again the only runtime route, and the exact sibling remains diagnostic (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-{correctness,runtime-correctness,rejected}.json`). The removed D11 persistent-composite screen preserved hidden-17/3072 and all 47 actual Q2 XL router outputs bit-for-bit, and actual-weight HIP events improved its isolated 47-layer router window **0.820 -> 0.661 ms (-19.37%)**. It also removed 47 launches/token and improved clean short dispatch span/profiled child, but the predeclared every-context mechanical gate failed: three clean short pairs put pooled kernel sum at **17.269 -> 17.277 ms/token (+0.046%)**. The old D11 composite kernel, wrapper, registry entry, selector, counter, and tests remain removed; the exact registered split projection plus correction-only selector remains the only runtime route. Historical evidence is preserved in the D11 design/correctness/rejection artifacts. A distinct standalone `correction_bias_compact_wave32` sibling now assigns eight experts to each lane in one stateless wave, with no projection fusion, counter, workspace, or launch contraction. Random/tie/extreme/signed-zero fixtures are field-bit exact to control and pass the CPU gate at KL **5.97e-16** / top-1 **100%**; all 47 actual projected-logit/correction rows are exact. Static/cached resources are local32/wave32/grid32, logical/allocated VGPR **70/72**, SGPR **18/128**, LDS/private/spill/scratch0, and no barriers. The repeated repository actual window improves event **0.397646 -> 0.295123 ms (-25.783%)** and wall **0.397976 -> 0.295419 ms (-25.770%)**. The gfx1100 primitive remains admitted and gfx1151-excluded. Its temporary default-off c=1 owner passes 16-transition full state/ownership and cache-only **47 retained projection + 47 candidate selector / 723-model-kernel/token** tracing, but the frozen short gate reverses the isolated win in both orders: selector time regresses **30.58%/27.60%**, complete router time **16.89%/14.08%**, kernel sum **1.787%/0.591%**, and profiled-child throughput **1.587%/1.619%**; order-A span also regresses **4.363%**. Remaining contexts/categories stop and runtime/session/CLI selection is removed; `correction_bias` is again the only runtime route (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-{correctness,runtime-correctness,rejected}.json`). |
| `moe_tail+next_rmsnorm` variants `laguna_aggregate_gguf_f32_weight_out`, `laguna_aggregate_wave0_tree_gguf_f32_weight_out` | `bf16` routed/shared/residual state plus F32 GGUF norm weight | `hipengine/kernels/hip_gfx1100/fused/paro_combine.{hip,py}`, `runtime/laguna_{moe,gguf_runner}.py` | `laguna_aggregate_moe_tail_next_rmsnorm_{gguf,wave0_tree_gguf}_bf16_out(...)`; c=1 registry-driven retained D9 route with add/add/RMS fallback | D9 preserves both BF16 add boundaries, writes post-MoE hidden plus the next `attn_norm`/final `output_norm`, and reuses the exact local256 RMS accumulation/tree. Hidden 17/3072 synthetic outputs are byte-exact to the three-kernel chain (KL 0, top-1 100%); rows>1, explicit disable, and registry miss retain the unfused route. A shared-weight actual Q2 XL gate compares all 47 sparse boundaries, full logits/argmax bits, complete K/V/live spans, reset, and lifecycle exactly through 16 decode steps. Matched dirty cached traces show exactly 47 candidate calls, **869 -> 775 dispatches/token**, local256/VGPR16/SGPR128/LDS1024/scratch0, kernel sum **17.296 -> 17.288 ms/token (-0.042%)**, span **20.702 -> 20.383 ms (-1.545%)**, and profiled child **43.890 -> 45.003 tok/s (+2.536%)**. Clean short/512/1K/near-4K kernel sum/span/child rows all improve; the complete category gate promotes D9 at **47.132 h32 decode tok/s (+1.560% vs D7)** and **12.038 h32 E2E (+0.555%)** with every category positive. The scalar body is independently promoted on gfx1151: native hidden17/3072 is byte-exact, cached resources are local256/VGPR16/SGPR128/LDS1024/scratch0, and a rollback/fused/rollback p512/d128 gate measures **14.529573/14.555265/14.525706 tok/s**, removing **94 launches/token** and moving cumulative post-merge decode **+26.935%**. The separately registered exact wave-0 RMS-tree sibling keeps both BF16 boundaries and every FP32 association, but replays stride-128/64/32 in wave 0 and strides 16..1 with shuffles. Focused RED/GREEN and hidden17/3072 edges pass; ten CPU-reference cases are KL0/top-1 100%, and all **47/47** actual sparse-layer hidden/norm outputs are byte-exact. Repeated layers 1/47 improve event/wall **2.78-2.87%**. Clang-22 contracts **9 -> 2 barriers**, **266 -> 225 instructions**, and **1,404 -> 1,276 B** at logical VGPR14/SGPR24/LDS1024/private/spills0; cache-only tracing names local256/VGPR16/SGPR128/LDS1024/scratch0 with no compiler. A temporary false/default-off gfx1100 c=1 owner passed shared-weight bulk prefill, all **48 hidden + 47 routed** boundaries, 16 decode transitions, full logits/IDs, active K/V and every live-span field, reset, and lifecycle at exact KL0/top-1 100% with zero allocation delta. Cache-only full-model tracing recorded exactly **47 candidate calls/token**, zero retained-D9 calls, unchanged **45 IQ3 wave10 calls / 678 model kernels/token**, local256/VGPR16/SGPR128/LDS1024/scratch0, and no compiler. The frozen short gate rejects runtime ownership: order A improves D9 **3.056%** but regresses kernel sum **0.0506%**, span **0.799%**, and child throughput **1.477%**; order B regresses D9 **0.843%**. Pooled favorable rows cannot waive either order, so 512/1K/3968 and categories stop. Capability/plan/session/CLI seams are removed; the retained D9 body and registered add+add+RMSNorm chain again own runtime, while the wave-0 primitive remains diagnostic (`benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-moe-tail-wave0-tree-{design,correctness,runtime-correctness,rejected}.json`). Retained D9 artifacts: `benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-moe-tail-next-rms-{correctness,retained}.json`, `benchmarks/results/2026-07-28-gfx1151-laguna-d9-moe-tail-retained.json`. |
| `weighted_sum+moe_tail` variant `laguna_top10_routed_hidden_out` | BF16 selected expert rows/shared/post-attention plus F32 scaled routing weights | `hipengine/kernels/hip_gfx1100/fused/paro_combine.{hip,py}` | `laguna_weighted_top10_routed_hidden_bf16_out(...)`; registered weighted-sum + D9 and registered RMSNorm remain the production fallbacks | Separately registered W7900 diagnostic primitive; runtime owner removed after clean rejection. One local32 feature-parallel producer performs ten ordered `fmaf`s, writes the exact observable routed BF16 row, and preserves both Laguna BF16 add boundaries into hidden; the unchanged registered local256 RMSNorm remains a second launch. Random/rounding-edge/signed-zero routed/hidden/norm fields are byte-exact to weighted-sum + D9 and pass the direct NumPy KL/top-1 gate. All 47 actual Q2 XL layers remain byte-exact; the 94-call repository window improves event/wall **12.664%/12.664%**. Codegen is logical/allocated VGPR **23/24**, SGPR **18/128**, LDS/private/spills/scratch0, 134 instructions, 804 bytes, zero barriers. A temporary default-off owner preserved full logits/IDs, all **48 hidden + 47 routed boundaries**, K/V/`KVLiveSpans`, reset, and lifecycle over 16 transitions at KL0/top-1 100%; cache-only tracing recorded **47 producer + 47 adjacent registered RMS calls/token** at unchanged **723 model kernels/token**. The frozen clean short gate rejects selection: boundary time improves **19.883%/12.390%** in orders A/B, but order B regresses total kernel sum **0.320%** and span **6.342%**. Remaining contexts/categories stop; capability/plan/session/CLI/dispatch integration is removed and production files match primitive commit `1a10af227`. The key stays gfx1151-excluded; all runtime paths retain registered control (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-{correctness,runtime-correctness,rejected}.json`). |
| `moe_linear` variant `selected_dual_t16_silu_q8_1_dp4a_gemv_decode_bf16_bf16_out` | GGML `q8_1` activations with `gguf_q4_k_t16_v1` selected gate/up | existing `quant/gguf_q4_k_gemv.hip` quantizer and registered `quant/gguf_t16_selected_gemv.hip` fused leaf | standalone prequantized selected Q4T16 gate/up+SiLU leaf | Synthetic leaf gates cover KL <= 0.05/top-1 >= 90% versus the BF16 path and bit-exact fused-versus-split Q8_1 SiLU rounding. The temporary Laguna AR-O1 producer-quantize route improved every short prefill shape and full-category performance, but failed model quality at maximum teacher-forced KL `0.17156` (>0.05). Its env/session selector, scratch, and runtime dispatch are removed; retain this leaf only as independently tested kernel evidence. |
| `moe_linear` variant `selected_t16_grouped_smallm_bf16_bf16_out`; `moe_group_compact` variant `active_experts` | `gguf_q4_k_t16_v1` / `gguf_q6_k_t16_v1` selected down plus generic F32-weight/BF16-row metadata | `quant/gguf_t16_selected_gemv.{hip,py}`, `moe/group_scatter.{hip,py}`, `runtime/laguna_moe.py` | exact C16xR4 grouped Q4/Q6 down; deterministic one-pass device compact metadata; staged count/prefix/scatter fallback | AR-O1 candidate. Mixed 0/1/2/3/4/5/8 buckets and production K1024/N3072 Q4/Q6 leaves are BF16-bit exact to direct T16 and pass the CPU GGUF oracle; the complete synthetic Laguna MoE output is bit-exact. Dirty balanced full-model selection rejects grouped gate/up and indexed no-gather input, but packed grouped down improves rows 32/55/64/122/128 by 2.71/4.79/5.02/6.62/6.84%; raw grouped loses 1.15% at 16, so the candidate falls back to direct below 32. A clean full-shape and complete category gate is required before default promotion. |
| `moe_linear` variant `selected_t16_expert_major_wmma_comp_bf16_bf16_out` | `gguf_q4_k_t16_v1` / `gguf_q6_k_t16_v1` | `quant/gguf_k_t16_selected_prefill.{hip,py}`, reused `moe/group_scatter.{hip,py}`, `runtime/laguna_moe.py` | explicit BF16 expert-major Q4/Q6 compensated WMMA single projection; exact GEMV/grouped chain remains fallback | Vulkan-transfer quality candidate. One wave owns one active expert/output-16 tile and walks the expert's natural compact rows in M16 groups; every K16 WMMA partial starts from zero and is Kahan-accumulated in FP32. The Q4/Q6 CPU fixture passes finite output, KL <=0.05, and top-1 >=90%; synthetic complete-MoE composition stays within KL 0.05. Cached gfx1151 traces name Q4/Q6 template instantiations at local32, zero LDS/scratch, and 184/200 VGPR. On seven synthetic production-dimension shapes, packed gather plus two Q4 gate/up launches improves the direct dual path from 1.052x at M32 to 5.777x at M512; Q6 down regresses at M32-64 and crosses to 1.420-2.381x at M122-256 and 2.179x at M512. The clean three-repeat full-model screen improves explicit M32/55/64/122/128/256/512 by 1.197/1.345/1.423/1.717/1.745/2.087/2.304x and reaches 176.001 tok/s at M512. Explicit M122 fails top-1 (0/3) despite max KL 0.017681; the admitted adaptive policy therefore keeps the exact route below M128. M128/M256/M512 pass finite/max-KL 0.001725/top-1 9/9/exact-cursor/deterministic-state/lifecycle gates. Explicit `adaptive_expert_major_wmma_comp` selects the candidate at M128+ for the complete gate; public defaults remain unchanged (`benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-screen.json`). The gate rejects the all-layer route despite **73.046 -> 130.557 tok/s (1.787x)** weighted prefill and positive performance in every category: 320 teacher-forced steps reach maximum KL **0.527791** (>0.05) at **314/320** top-1. The Poolside short-row fallback, determinism, neutral decode, and lifecycle pass, so exact grouped-small-M remains default (`benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-category-rejected.json`). The full-suite component bisection also rejects Q4 gate/up-only and Q4/Q6 down-only: they improve prefill **1.545x/1.095x** but reach max KL **0.988050/1.183662** at **312/320** and **311/320** top-1. Both are numerically worse than combined KL **0.527791**, showing partial cancellation. The final architecture-derived scopes also reject: global-only improves prefill **1.115x** but reaches max KL **0.628301** at **310/320** top-1, while SWA-only improves **1.505x** but reaches max KL **1.205779** at **312/320**. All temporary runtime/category/component/scope selectors and full-model harnesses are removed. Retain only this registered leaf, its CPU-quality/registry tests, cached trace, and rejection evidence as diagnostics (`benchmarks/results/2026-07-24-gfx1151-laguna-expert-major-wmma-layer-family-rejected.json`). |
| `laguna_attention_decode` variant `swa_context_fused_exact_gated_*_dpp_qk_dense_ring_*_fixed512_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | dense saturated-ring sibling of the retained local512/V128 output-sharded DPP-QK owner | gfx1151 selects this exact variant only at `live_count=capacity=512` while dense-initial metadata remains valid. Identity physical addressing and all-slot visibility compile out per-slot metadata traffic and the shared physical plane, lowering LDS **43,008 -> 40,960 B** and improving the byte-exact 21x100 leaf **25.555%**. Pre-saturation, explicit eviction, and peer backends retain generic DPP-QK `KVLiveSpans` handling. Evidence: `benchmarks/results/2026-07-30-gfx1151-laguna-swa-dense-ring-retained.json`. |
| `laguna_attention_decode` variant `global_context_fused_exact_gated_*_dpp_qk_dense_prefix_*_fixedshape_spans` | `bf16` KV, F32 query/gate | `hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.{hip,py}` | dense-prefix sibling of the retained global mixed40/local512 DPP-QK owner | gfx1151 selects this exact variant only at capacity4096/live<=4000 while dense-initial metadata remains valid. Identity physical addressing and prefix visibility compile out token/base/eviction metadata plus shared physical-slot publication/replay without changing the full `KVLiveSpans` ABI. Live513/576/639 F32 context and gated BF16 are byte-exact and improve **15.293%/19.972%/19.907%** at grid40/local512/VGPR48/scratch0. Explicit eviction and peer backends retain the generic owner. Evidence: `benchmarks/results/2026-07-30-gfx1151-laguna-global-dense-prefix-retained.json`. |
| `smoke_add` | `fp16` registry key, FP32 buffers | `hipengine/kernels/hip_gfx1100/smoke/smoke_add.hip` | `smoke_add_f32(...)` | `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` → `max_abs=0.0` on W7900 |
| `metadata_cast` variant `i32_to_i64` | `gguf_qwen35` | `hipengine/kernels/hip_gfx1100/runtime/state.hip` | `copy_i32_to_i64(...)` | NativeSpecCycle N1 device top-1 adapter. `tests/test_runtime_state_plan.py::test_copy_i32_to_i64_matches_exact_cpu_cast` is exact for `0`, `-1`, `7`, and `INT32_MAX`. Reusable B1/B2 graphs supersede the rejected one-shot capture. The retained gfx1151 six-step trace records 12 `copy_i32_to_i64_kernel` calls at **1.002-1.323 us**, 8 VGPR, 128 SGPR, zero scratch/LDS, workgroup/grid X 256; the complete verifier is **24.891 ms host / 21.674 ms kernels / 3.218 ms residual** and 940 calls/step. N1 reaches **80.132 tok/s** and public N3 **80.099** versus **70.020** direct commit with exact full-suite semantics. W7900's retained reusable N1 remains **122.667 tok/s**. Artifacts: `benchmarks/results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json` and `benchmarks/results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json`. |
| `rmsnorm` | `bf16` | `hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip` | `qwen35_rmsnorm_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16` → `max_abs=0.0`, `bit_mismatch=0`; `rocprofv3` shows `qwen35_rmsnorm_kernel`, computed `DurationNs=6560`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0` on W7900 |
| `add_rmsnorm`, `add_rmsnorm_f32`, `head_rmsnorm` | `bf16` | same | `qwen35_add_rmsnorm_bf16(...)`, `qwen35_add_rmsnorm_f32_bf16(...)`, `qwen35_head_rmsnorm_f32_bf16(...)` | Build/registration tests landed; launch wrappers are source-family peers of `qwen35_rmsnorm_kernel` and share the same `.so` |
| `rmsnorm`, `add_rmsnorm` variants `paro_out`, `paro_out_fp16` | `bf16`, `w4_paro` | same | `paro_rmsnorm_out_bf16(...)`, `paro_add_rmsnorm_out_bf16(...)`, `paro_rmsnorm_out_fp16(...)`, `paro_add_rmsnorm_out_fp16(...)` | `python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 norm/add/residual (`fp16_*_mismatch=0`); `rocprofv3` shows BF16 `paro_rmsnorm_out_kernel<uint16_t>`/`paro_add_rmsnorm_out_kernel<uint16_t>` and FP16 `paro_rmsnorm_out_kernel<_Float16>` (`DurationNs=5800`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=1024`) / `paro_add_rmsnorm_out_kernel<_Float16>` (`DurationNs=5320`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=1024`) on W7900 |
| `add_rmsnorm` variant `bf16_out_staged_f32_local256` | `gguf_f32_weight` | `hipengine/kernels/hip_gfx1100/fused/gguf_ops.{hip,py}` | `gguf_add_rmsnorm_bf16_f32_weight_staged_f32_local256(...)` | Separately registered W7900 diagnostic; runtime ownership is absent. One local256 block preserves the existing unrounded BF16+BF16 F32 add, square accumulation, reduction tree, RNE residual, and weighted norm while staging the unrounded value in dynamic LDS instead of reloading both BF16 inputs. Synthetic hidden 256/1024/3072/4096 and all 48 actual Q2 XL boundaries are BF16-bit exact to registered control; the 10x1024 CPU gate is KL max **1.40e-5**, top-1 **100%**. Repository hidden-3072 synthetic event/wall improves **2.74%/2.69%** and the complete actual 48-call window improves **3.53%/3.70%**. Codegen is local256/wave32, logical/allocated VGPR **15/16**, SGPR **18/128**, dynamic LDS **13,312 B**, private/spills/scratch0, 296 instructions, 1,384 bytes, and nine static barriers representing the same nine dynamic synchronization points as control. Cache-only tracing names two expected calls at **3.88/4.80 us**, local256/VGPR16/scratch0, with no compiler under profiling. The exact key is gfx1151-excluded; registered `bf16_out` remains every runtime/rows/prefill/backend fallback (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-correctness.json`). |
| `paged_kv_copy` variants `head_major_spans`, `head_major_dense_prefix_spans` | `bf16` | `hipengine/kernels/hip_gfx1100/attention/paged_kv_write.{hip,py}` | `qwen35_copy_paged_kv_bf16_to_head_major_{,dense_prefix_}spans(...)` | Nathan-review P0 retained gfx1151 prefill route. One raw-pointer local256 copy gathers paged token-major BF16 K/V into a reusable `[kv_head, capacity, head_dim]` pair before AOTriton; the generic key follows non-identity page tables, while the production dense-prefix key compiles identity addressing out. Persistent `KVLiveSpans` KV, native attention, and strided AOTriton remain exact fallbacks. Lengths 1/255/256/257 with permuted pages, sentinels, and dense/generic comparison are byte-exact; full-model p512 matches every hidden/GDN/KV/logit boundary, and forced allocation denial matches the same state. Cached gfx1151 tracing names 12 expected dense copies at local256/VGPR16/SGPR128/LDS0/scratch0. Copy-inclusive full prefill is neutral at 512 and improves 4K/32K/64K **0.616%/3.383%/7.001%**; default ownership is bounded to rounded capacity <=65,792 tokens. Evidence: `benchmarks/results/2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json`. |
| `rmsnorm`, `add_rmsnorm`, `bf16_add`, `gate_repeat_value`, `gate_mul`, `head_rmsnorm+partial_rotary` GGUF helpers | `gguf_f32_weight`, `bf16`, `f32_out` | `hipengine/kernels/hip_gfx1100/fused/gguf_ops.hip` | `gguf_rmsnorm_bf16_f32_weight(...)`, `gguf_rmsnorm_bf16_f32_weight_out_f32(...)`, `gguf_add_rmsnorm_bf16_f32_weight(...)`, `gguf_bf16_add(...)`, `gguf_gate_repeat_value_bf16(...)`, `gguf_gate_mul_bf16(...)`, `gguf_qwen35_head_rmsnorm_partial_rotary_{position,positions}_f32_weight(...)` | `HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt python3 -m pytest tests/test_gguf_ops.py tests/test_qwen35_gguf_full_attention_gpu.py -q` → synthetic BF16 add, F32-weight RMSNorm/add-RMSNorm, GGUF F32-weight scalar/multi-position head RMSNorm+RoPE, c=1 full-attention gate/value repeat, resident full-attention CPU-bridge oracle, and GGUF AOTriton final-row oracle pass (`5 passed` after task #46); `gguf_rmsnorm_bf16_f32_weight_out_f32(...)` is the MTP-GGUF M2.5 fp32 post-`output_norm` seed target and matches CPU RMSNorm with `rtol/atol=1e-6` in `tests/test_gguf_ops.py::test_gguf_ops_bf16_add_and_f32_weight_rmsnorm`; `rocprofv3 --kernel-trace` on gfx1151 shows `gguf_rmsnorm_bf16_f32_weight_out_f32_kernel(unsigned short const*, float const*, float*, float, long)` with `DurationNs=3887`. Existing prefill trace shows `gguf_head_rmsnorm_partial_rotary_positions_f32_weight_kernel`, BF16 prompt-KV writer, AOTriton `attn_fwd`, and `gguf_gate_mul_bf16_kernel`. Used by the Qwen3.5 GGUF full-stack runner because GGUF norm weights are F32 tensors, not PARO BF16 delta weights. |
| `router_logits`, `router_select`, `router_topk_shared` variants `out`, `out_fp16_hidden`, `prefill_sigmoid_out`, `prefill_sigmoid_out_fp16_hidden`, `coop_out`, `coop_out_fp16_hidden`; `router_topk_split_shared` variants `coop_out`, `coop_out_fp16_hidden`, `coop_out_bf16_hidden`, `coop_out_bf16_hidden_persistent` | `bf16`, `fp16`, F32 weights/`fp32` select, `w4_paro` shared route | `hipengine/kernels/hip_gfx1100/moe/router.hip` | `qwen35_router_logits_bf16(...)`, `qwen35_router_logits_fp16(...)`, `qwen35_router_select(...)`, `qwen35_router_topk_shared_out_{bf16,fp16}(...)`, `qwen35_router_topk_shared_sigmoid_out_{bf16,fp16}(...)`, `qwen35_router_topk_shared_coop_out_{bf16,fp16}(...)`, `qwen35_router_topk_split_shared_coop_out_{bf16,fp16}(...)`, `qwen35_router_topk_split_shared_coop_out_bf16_f32w(...)`, `qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent(...)` | `python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → BF16 and FP16-hidden top-k/routing plus P3.2 sigmoid-router logits vs NumPy oracle (`selected_match=True`, `fp16_selected_match=True`, `sigmoid_selected_match=True`, `sigmoid_fp16_selected_match=True`, sigmoid logits max abs `0.0`/`4.77e-07`); `rows=1 --hidden-size 256` also validates the opt-in cooperative wrappers (`coop_selected_match=True`, `coop_fp16_selected_match=True`). P9.D1 adds the GGUF split expert/shared decode cooperative wrapper; `tests/test_qwen35_router_plan.py -k split_shared_coop_bf16_matches_cpu_router` matches CPU router logits/top-k and `rocprofv3` shows `qwen35_router_topk_split_shared_coop_out_kernel<unsigned short>` (`End-Start=17440 ns`). The gfx1100 F32-weight extension is byte-exact to the current 512-thread logits plus 256-thread select chain at `hidden=2048, experts=256, top_k=8`; its cached W7900 trace shows the active 256-thread kernel at 40 VGPR, zero scratch, 512-byte LDS, and 10.6 us median. Clean commit `4c743994` promotes it by default after 4K graph decode improves `97.234 -> 98.273 tok/s` (+1.07%) with exact IDs/final values and unchanged memory. The persistent-counter extension removes exactly 40 four-byte reset nodes/token; its cache-cycled fused leaf improves `14.667 -> 10.444 us` (-28.79%), the expected `<unsigned short, true>` trace remains at 40 VGPR/zero scratch/512-byte LDS, the source-dirty admission improves `98.936 -> 100.711 tok/s` (+1.79%), and clean `0ec2a813` confirms `98.812 -> 100.446 tok/s` (+1.65%) with exact IDs/final values. `rocprofv3 --kernel-trace` all-layer 512 prefill shows FP16 hidden `qwen35_router_logits_token_tile_kernel<_Float16,4>` ran 40 times (`8.683 ms` total, avg `217.1 us`) plus block-parallel select on W7900; tokens `<4` stay on the original one-token logits kernel by default. D1.5's cooperative decode producer is gated by `HIPENGINE_PARO_ROUTER_TOPK_COOP=1`; it is correct but rejected as default after 512/128 and 4K/128 graph replay regressions. P3.2's shared-gate sigmoid producer is gated by `HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED=1`; it is correct and prefill/legacy-only, but rejected as default after neutral 512/4K E2E results. |
| `moe_group_count`, `moe_group_prefix`, `moe_group_compact`, `moe_group_scatter`, `moe_group_scatter_gather`, `moe_gather_packed_hidden`, `moe_wmma_tile_map`, `moe_mmq_tile_map` | generic selected-expert and `w4_paro` grouped/compact MoE metadata plus packed-hidden gather | `hipengine/kernels/hip_gfx1100/moe/group_scatter.hip` | `qwen35_moe_group_count(...)`, `qwen35_moe_group_prefix(...)`, `qwen35_moe_group_compact_active(...)`, `qwen35_moe_group_compact_active_source_rows(...)`, stable `*_parallel(...)` siblings, `qwen35_moe_group_scatter(...)`, `qwen35_moe_group_scatter_gather_lowp(...)`, `qwen35_moe_gather_packed_hidden_lowp(...)`, `qwen35_moe_wmma_tile_map(...)`, `qwen35_moe_mmq32_tile_map(...)` | `python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → grouped metadata/gather fixture passes (`prefix_match=True`, `lane_match=True`, `packed_match=True`, `tile_match=True`); `rocprofv3 --kernel-trace` shows `qwen35_moe_group_count_kernel` (`DurationNs=6640`), `qwen35_moe_group_prefix_kernel` (`11601`), `qwen35_moe_group_scatter_gather_kernel` (`11241`), `qwen35_moe_gather_packed_hidden_kernel` (`5360`), and `qwen35_moe_wmma_tile_map_kernel` (`2561`) on W7900. The generic one-pass compact-active extension deterministically emits starts, active IDs/count, stable lanes, expert IDs, and F32 routing weights and retains count/prefix/scatter as its unfused fallback; its dedicated GPU CPU-oracle fixture passes. The Laguna MMQ sibling additionally emits `compact_to_source = sorted_lane // top_k`, and the tile-map body supports 32-row padding without a scalar D2H. Its gfx1151 CPU-oracle fixture passes; cached tracing records compact/source metadata at **3.566 us** and the 32-row tile map at **2.084 us**, both VGPR16/scratch0. The exact stable parallel sibling replaces the serial 256x5,120 lane scan with per-expert count and ballot-ordered scatter workgroups; production-shape metadata and complete MoE BF16 output are byte-identical. Its M512/top10/E256 leaf improves **0.348880 -> 0.058969 ms (-83.10%)**, and clean seven-repeat resident pp512 improves **490.824 -> 497.408 tok/s (+1.341%)** with all paired wins. Cached production tracing measures **500.325 tok/s** and parallel count/prefix/scatter at **0.277/1.520/0.767 ms** across 47 layers. The prefix follow-up replaces its remaining one-thread loop with a one-block exclusive scan plus ballot active-ID compaction: cached prefix time falls **32.34 -> 2.404 us/layer**, local256/VGPR24/LDS2560B/scratch0, with exact complete output and a projected **1.407 ms** pp512 saving. gfx1151 defaults to parallel; serial remains rollback and other backends stay unchanged. Expert compact WMMA remains a separate Qwen/PARO lane. |
| `router_logits`, `router_select`, `router_topk_shared` variants `out`, `out_fp16_hidden`, `prefill_sigmoid_out`, `prefill_sigmoid_out_fp16_hidden`, `coop_out`, `coop_out_fp16_hidden`; `router_topk_split_shared` variants `coop_out`, `coop_out_fp16_hidden`, `coop_out_bf16_hidden`, `coop_out_bf16_hidden_persistent` | `bf16`, `fp16`, F32 weights/`fp32` select, `w4_paro` shared route | `hipengine/kernels/hip_gfx1100/moe/router.hip` | `qwen35_router_logits_bf16(...)`, `qwen35_router_logits_fp16(...)`, `qwen35_router_select(...)`, `qwen35_router_topk_shared_out_{bf16,fp16}(...)`, `qwen35_router_topk_shared_sigmoid_out_{bf16,fp16}(...)`, `qwen35_router_topk_shared_coop_out_{bf16,fp16}(...)`, `qwen35_router_topk_split_shared_coop_out_{bf16,fp16}(...)`, `qwen35_router_topk_split_shared_coop_out_bf16_f32w(...)`, `qwen35_router_topk_split_shared_coop_out_bf16_f32w_persistent(...)` | `python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → BF16 and FP16-hidden top-k/routing plus P3.2 sigmoid-router logits vs NumPy oracle (`selected_match=True`, `fp16_selected_match=True`, `sigmoid_selected_match=True`, `sigmoid_fp16_selected_match=True`, sigmoid logits max abs `0.0`/`4.77e-07`); `rows=1 --hidden-size 256` also validates the opt-in cooperative wrappers (`coop_selected_match=True`, `coop_fp16_selected_match=True`). P9.D1 adds the GGUF split expert/shared decode cooperative wrapper; `tests/test_qwen35_router_plan.py -k split_shared_coop_bf16_matches_cpu_router` matches CPU router logits/top-k and `rocprofv3` shows `qwen35_router_topk_split_shared_coop_out_kernel<unsigned short>` (`End-Start=17440 ns`). The gfx1100 F32-weight extension is byte-exact to the current 512-thread logits plus 256-thread select chain at `hidden=2048, experts=256, top_k=8`; its cached W7900 trace shows the active 256-thread kernel at 40 VGPR, zero scratch, 512-byte LDS, and 10.6 us median. Clean commit `4c743994` promotes it by default after 4K graph decode improves `97.234 -> 98.273 tok/s` (+1.07%) with exact IDs/final values and unchanged memory. The persistent-counter extension removes exactly 40 four-byte reset nodes/token; its cache-cycled fused leaf improves `14.667 -> 10.444 us` (-28.79%), the expected `<unsigned short, true>` trace remains at 40 VGPR/zero scratch/512-byte LDS, the source-dirty admission improves `98.936 -> 100.711 tok/s` (+1.79%), and clean `0ec2a813` confirms `98.812 -> 100.446 tok/s` (+1.65%) with exact IDs/final values. `rocprofv3 --kernel-trace` all-layer 512 prefill shows FP16 hidden `qwen35_router_logits_token_tile_kernel<_Float16,4>` ran 40 times (`8.683 ms` total, avg `217.1 us`) plus block-parallel select on W7900; tokens `<4` stay on the original one-token logits kernel by default. D1.5's cooperative decode producer is gated by `HIPENGINE_PARO_ROUTER_TOPK_COOP=1`; it is correct but rejected as default after 512/128 and 4K/128 graph replay regressions. P3.2's shared-gate sigmoid producer is gated by `HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED=1`; it is correct and prefill/legacy-only, but rejected as default after neutral 512/4K E2E results. A separately registered Laguna-only c=1 sibling under `router_logits/f32/bf16_hidden_wave0_tree` now preserves one local256 block/expert and every dot association while replacing the retained nine dynamic reduction barriers with one publication plus exact wave-0 replay. Hidden17/3072 synthetic and ten independent CPU cases pass (max KL **5.63e-16**, top-1 **100%**); all **47/47** actual Q2 XL router outputs are F32-bit exact and finite, and layers 1/47 improve direct event/wall **8.04-8.13% / 8.02-8.12%**. Clang-22 keeps logical VGPR22/private/spills0, reduces SGPR **30 -> 28**, and contracts **226 -> 220 instructions**. Cache-only tracing names one grid/local **65,536/256** candidate at allocated VGPR24/SGPR128/static-LDS0+dynamic-1024/scratch0 with no compiler. The exact key is gfx1151-excluded. A temporary false/default-off gfx1100 c=1 owner passed shared-weight bulk prefill, **16** decode transitions, full logits/IDs, all **48 hidden + 47 routed** boundaries, active K/V and every span field, reset/re-prefill, ownership, and teardown at KL0/top-1 100% with zero allocation delta. Cache-only tracing recorded exactly **47 candidate calls/token**, zero retained decode projections, **47 retained tile4 prefill projections**, unchanged **45 IQ3 wave10 calls / 678 model kernels/token**, local256/VGPR24/SGPR128/static-LDS0+dynamic-LDS1024/scratch0, and no compiler. The frozen short gate rejects ownership: order A improves projection/kernel/span **7.527%/0.793%/3.767%** but regresses profiled-child throughput **0.619%**; order B improves projection/child **6.307%/1.977%** but regresses kernel sum **0.612%** and span **4.211%**. Favorable pooled projection/kernel/span/child rows cannot waive either order. The runner stops before 512/1K/3968 and categories; capability/plan/session/CLI seams are removed. Retained `bf16_hidden` plus the unchanged registered selector again owns every runtime/rows/prefill/backend route, while the exact primitive remains diagnostic (`benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-router-projection-wave0-tree-{design,correctness,runtime-correctness,rejected}.json`). |
| `moe_group_count`, `moe_group_prefix`, `moe_group_compact`, `moe_group_scatter`, `moe_group_scatter_gather`, `moe_gather_packed_hidden`, `moe_wmma_tile_map` | generic selected-expert and `w4_paro` grouped/compact MoE metadata plus packed-hidden gather | `hipengine/kernels/hip_gfx1100/moe/group_scatter.hip` | `qwen35_moe_group_count(...)`, `qwen35_moe_group_prefix(...)`, `qwen35_moe_group_compact_active(...)`, `qwen35_moe_group_scatter(...)`, `qwen35_moe_group_scatter_gather_lowp(...)`, `qwen35_moe_gather_packed_hidden_lowp(...)`, `qwen35_moe_wmma_tile_map(...)` | `python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → grouped metadata/gather fixture passes (`prefix_match=True`, `lane_match=True`, `packed_match=True`, `tile_match=True`); `rocprofv3 --kernel-trace` shows `qwen35_moe_group_count_kernel` (`DurationNs=6640`), `qwen35_moe_group_prefix_kernel` (`11601`), `qwen35_moe_group_scatter_gather_kernel` (`11241`), `qwen35_moe_gather_packed_hidden_kernel` (`5360`), and `qwen35_moe_wmma_tile_map_kernel` (`2561`) on W7900. The generic one-pass compact-active extension deterministically emits starts, active IDs/count, stable lanes, expert IDs, and F32 routing weights and retains count/prefix/scatter as its unfused fallback; its dedicated GPU CPU-oracle fixture passes. Expert compact WMMA remains a separate Qwen/PARO lane. |
| `selected_dual_pack8_gemv`, `selected_pack8_gemv` variants `strided`, `transposed`, `*_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_selected_dual_pack8_*_bf16(...)`, `gemv_awq_selected_pack8_*_bf16(...)`, `gemv_awq_selected_dual_pack8_*_fp16(...)`, `gemv_awq_selected_pack8_*_fp16(...)` | `python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 dual/single, strided/transposed (`dual_mismatch=0/0`, `single_mismatch=0/0`, `fp16_dual_mismatch=0/0`, `fp16_single_mismatch=0/0`); selected dual kernels support `rows = x_rows * lanes_per_token` for batched c1-style prefill gate/up; `rocprofv3` shows FP16 selected GEMV kernels with `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64` on W7900 |
| `moe_linear` variants `selected_gemv_decode_bf16_bf16_out`, `selected_dual_silu_gemv_decode_bf16_bf16_out`, `selected_dual_silu_gemv_decode_tile2_grid64_bf16_bf16_out`, `selected_dual_silu_gemv_decode_tile2_grid64_local64_reduce_bf16_bf16_out`, `selected_gemv_decode_tile1_bf16_bf16_out`, `selected_dual_silu_gemv_decode_tile1_bf16_bf16_out`, `selected_gemv_decode_k1024_wave4_bf16_bf16_out`, `selected_gemv_decode_k1024_wave4_signbit_bf16_bf16_out`, `selected_gemv_decode_tile4_bf16_bf16_out`, and `selected_weighted_down_gemv_decode_bf16_bf16_out` | `gguf_iq2_xs`, `gguf_iq3_xxs`, `gguf_iq4_xs` raw rank-3 GGUF expert weights | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.hip` | `gguf_iq2_xs_selected_gemv_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_dual_silu_gemv_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_dual_silu_gemv_tile2_grid64_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_dual_silu_gemv_tile2_grid64_local64_reduce_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_gemv_tile1_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_dual_silu_gemv_tile1_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_gemv_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_dual_silu_gemv_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_gemv_k1024_wave4_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_gemv_k1024_wave4_signbit_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_gemv_tile4_bf16_bf16_out(...)`, `gguf_iq3_xxs_weighted_selected_down_bf16_bf16_out(...)`, `gguf_iq4_xs_selected_gemv_bf16_bf16_out(...)`, `gguf_iq4_xs_weighted_selected_down_bf16_bf16_out(...)` | IQ2_XS decode is CPU-oracle exact at synthetic K=256/3072, including Laguna's full K=3072,N=1024 projection shape and the fused BF16 projection/SiLU boundary. The exact branchless selector decoder maps the only grid codes 0/1/2 with `8 + 17*code + (code >> 1)` and applies sign by FP32 sign-bit OR. The retained pair16/local64 schedule loads two adjacent selectors together and shares their scale decode. Tile2 then computes two adjacent output columns per workgroup while sharing BF16 activation loads/conversions and preserving independent weight/reduction boundaries. At E256/K3072/N1024/top-10, rotating-distinct selected single/dual-SiLU move from branchless-local256 `49.200/78.784` through pair16 tile1 `33.296/56.922` to tile2 `30.955/55.964 us` (-37.08/-28.96% cumulative); matched tile1 -> tile2 improves all rotating/hot/repeated leaves by 2.12-8.82%. Tile2 is now the default; explicit tile1 four-axis variants remain the exact rollback/fallback. Cache-only GPU1 rocprof records local64/LDS512B/scratch0 with tile2 VGPR80/136. Artifacts: `benchmarks/results/2026-07-22-gpu1-iq2-xs-branchless-decode.json`, `benchmarks/results/2026-07-22-gpu1-iq2-xs-pair16-local64.json`, and `benchmarks/results/2026-07-22-gpu1-iq2-xs-output-tile2.json`. A later Laguna P1 row4 screen retains no new symbol: exact tile4 was BF16-bit equal but mixed/neutral on actual layers (**-1.41% to +0.89%** event/wall), while a source-backed Vulkan nested-FMA row4 sibling reduced VGPR **136 -> 72** yet regressed actual event/wall **8.38-10.90%**. Both local64/LDS512/scratch0 candidates were removed; exact tile2 remains the retained wave32 c=1 IQ2 route (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p1-iq2-row4-rejected.json`). A later gfx1100 `-mwavefrontsize64` build of the unchanged local64 source passed exact actual-layer/state/trace admission and lowered VGPR **136 -> 96**, but failed its frozen clean gate: pooled 1K IQ2 regressed **0.404%** and 512 child throughput regressed **0.562%**. Its build/wrapper/route/selector/tests are removed before categories; only historical correctness/rejection evidence remains (`benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-wave64-{correctness,rejected}.json`). A new retained gfx1100 c=1 sibling instead stores each selector's eight canonical unsigned magnitudes in one 64-bit constant-table entry while retaining parity `popc`, every FMA/reduction, and both BF16/SiLU boundaries. It adds only 3 KiB of code-object constants and no weight sidecar; the hot leaf contracts from 1,246 to 986 disassembly lines, logical VGPR **132 -> 110**, uint-to-float conversions **66 -> 10**, and multiplies **78 -> 14**, with zero spill and unchanged LDS. First/last actual IQ2 layers are BF16-bit exact and improve events **30.78-33.73%** and wall **30.00-33.43%**. Full model state/lifecycle are exact; cached model tracing records 92 c=1 calls at local64/VGPR112/LDS512/scratch0 while all 46 bulk-prefill calls remain on the retained VGPR136 route. Two clean process orders improve the IQ2 family **20.31-21.54%** and kernel sum **1.30-3.70%** at every context; both complete 18-prompt orders move h32 **52.650 -> 54.540 tok/s (+3.590%)** with every train/heldout category decode and E2E row positive. gfx1100 defaults expanded magnitudes, while explicit `False`, rows>1, and unsupported backends retain compact-grid tile2 (`benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-grid64-retained.json`). A separately registered gfx1100 fixed-local64 reduction diagnostic now keeps that retained grid64 dot body and both wave32 K owners while replacing only the four-accumulator transport: four `permlanex16` exchanges plus 16 DPP row shifts preserve each `+16,+8,+4,+2,+1` tree, and thread 0 keeps the exact `+0,+wave0,+wave1` tail. K256/K1024/K3072 synthetic and all **46** actual Laguna IQ2 layer pairs are BF16-bit exact; the independent CPU gate is KL max **1.99e-5**, top-1 **100%**. Repeated actual layers 1/45 improve event **1.27%/1.64%** and wall **1.15%/1.45%**. Codegen is local64/wave32, logical/allocated VGPR **110/112**, SGPR **31/128**, fixed/allocated LDS **32/512 B**, private/spills/scratch0, **858 instructions / 4,888 bytes**, zero `ds_bpermute`, one barrier. Cache-only tracing names the distinct candidate twice at grid `32768x10`, with a warm **31.600 us** duration and no compiler. The exact key is gfx1151-excluded and runtime-unselected; retained grid64 remains every production/rows/prefill/backend fallback (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-correctness.json`). Correctness-first UD-Q3_K_M AR primitives. IQ3 uses the qwen-kernel 256-thread/8-value-work-unit rowtile1 mapping with computed sign parity and a fused dual gate+up+SiLU path; IQ4 provides selected-single plus a top-k routing-weighted down composite that BF16-rounds each route projection before applying its weight in fallback order; its `top_k=8,in_features=512` default is local128. Direct selected kernels preserve `rows=x_rows*lanes_per_token`; single primitives remain the unfused fallbacks. `HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-q3.txt HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. python3 -m pytest tests/test_gguf_iq_gemv.py -q` passes all 17 cases on RX 7900 XTX/gfx1100: synthetic c=1/repeated-route and multi-row cases plus real-row selected outputs are bit-exact vs the torch-free CPU dequant/BF16 oracle (including IQ3 `in_features=2048` and IQ4 `in_features=512` real rows), fused IQ3 has `0` BF16-bit mismatches vs the two-single chain for c=1 and multi-row inputs, and weighted IQ4 is BF16-bit exact vs selected-single+weighted-sum for c=1 and multi-token inputs, including invalid route IDs. A cached-build `rocprofv3 --kernel-trace` of the same module on GPU1 shows all four expected symbols: selected IQ3 `3.12-4.88 us` (`VGPR=24`), fused IQ3 `4.12-5.96 us` (`VGPR=48`), selected IQ4 `4.28-15.20 us` (`VGPR=72`), while a final focused c=1/multi-token trace shows exact weighted IQ4 at `16.80/15.92 us` (`VGPR=80`); all have `SGPR=128`, `Scratch_Size=0`, and `LDS_Block_Size=512`. Resident c=1 and direct multi-row dispatch now resolves these exact four-axis keys: IQ3 uses fused dual-SiLU when present and otherwise two selected-single launches plus SiLU; IQ4 down emits one routing-weighted token row and switches to the unweighted shared/residual combine, while a missing weighted key retains selected-single + weighted-combine fallback. The public preset is `gguf_ud_q3_k_m`; its registry-selected generator now selects exact fully-bulk GDN/full-attention prefill plugins, while decode-graph policy declares IQ3 selected, IQ4 selected, and IQ4 weighted-down symbol groups. GPU1 direct-AR artifact `benchmarks/results/2026-07-19-hipengine-qwen36-35b-a3b-ud-q3km-direct-correctness.json` pins raw rank-3 residency, deterministic public `Hello -> [11]` parity with llama.cpp, finite logits/no torch import, weighted-vs-unfused teacher-forced bit equality (`KL=0`, top-1 `1.0`), and c>1 native/row-bulk bit equality. The former fully-bulk quarantine is superseded: mixed-64, all-row layer 0/3/4/40, 1K/4K attention-boundary, and full 4K serial-vs-bulk gates are bit-exact (`KL=0`, top-1 `1.0`); accepted GPU1 prefill is `218.598 tok/s` at 512 and `211.936 tok/s` at 4K. Evidence: `benchmarks/results/2026-07-20-gpu1-q3-exact-fully-bulk-prefill.json`. The retained direct baseline (`benchmarks/results/2026-07-19-gpu1-hipengine-qwen36-35b-a3b-ud-q3km-direct-baseline.json`) measures `19.452/99.015 tok/s` at 512/128, exact IQ traffic `424,280,064 bytes/token`, and a selected 16-replay graph window with 708 dispatches/token, 8.892 ms/token summed kernels, 1.747 ms/token IQ kernels, and no IQ scratch. The retained task-15 D0 decode trace shows the current task-19 weighted IQ4 composite at 37 launches/token, 1.00359 ms/token (11.29%), local128, VGPR 80, and zero scratch; it already contracts all top-8 slots sequentially and makes the review's per-slot local64 D1A obsolete. Revised D1B's exact wave-per-slot/four-output sibling was bit-exact, VGPR 32, and scratch-free, but was fully removed after the real 37-layer family regressed `16.0574 -> 18.0649 ms` (+12.50%) despite a cache-hot micro win. Bounded D1C then retained an explicit wave-uniform IQ3 super-block index: logical/allocated VGPR fell `42/48 -> 37/40`, vector `mad_u64` address chains `2 -> 0`, and the real 624-dispatch IQ3 family fell `11.4966 -> 11.2614 ms` (-2.05%) with zero scratch and exact 512/1K/4K IDs/logits; counterbalanced 512/128 graph wall moved `100.334 -> 100.536 tok/s` (+0.20%). Laguna Q2 XL applies that wave-uniform base to the otherwise identical IQ3 selected-down leaf: an actual `E256/K1024/N3072/top-10` weight screen is bit-exact and improves `98.011 -> 97.373 us` (-0.65%); clean full-model rocprof moves the 45-call family `4.040 -> 4.002 ms/token` (-0.94%) with zero scratch. The exact K1024 reduction has only four live wave32 units, so the wrapper now defaults that selected-single shape through local128 while preserving explicit local256 rollback; actual weights are bit-exact and the clean paired median improves 43.61%. Clean cached profiling confirms local128/VGPR32/LDS512/scratch0, moves the same family `4.002 -> 2.258 ms/token` (-43.57%), and reduces total kernel sum `23.097 -> 21.302 ms/token` (-7.77%). The full ten-prompt suite promotes the new 38.301 tok/s h32 headline with every category positive and exact correctness/lifecycle. The next exact Laguna c=1 composite contracts IQ3 down plus scaled routing in one registered leaf, preserves each local128 projection's BF16 boundary and the serial FMA order, and retains selected-single plus weighted-sum for bulk/fallback execution. Synthetic token rows and actual weights are bit-exact; clean local128/VGPR32/LDS512/scratch0 profiling removes 45 launches/token, moves IQ3 down plus selected reduction `2.392 -> 2.115 ms/token` (-11.61%), and cuts total kernel sum 1.43%. The complete suite promotes the **38.840 tok/s** h32 headline with every category decode/E2E positive and exact correctness/lifecycle. The retained-default P0 gfx1100 K1024 wave4 producer assigns one local32 wave to each `(route, output)`, reconstructs the local128 baseline's four independent wave partitions in registers, and hands BF16 route rows to the existing slot-order reducer. Actual layer-1/layer-45 inclusive events improve **36.34%/32.89%** versus the serial weighted composite and beat the exact row4 producer's **18.91%/14.96%** gains. It is full-model exact for logits, all 48 hidden boundaries, all 47 routed outputs, active KV/`KVLiveSpans`, reset, and lifecycle through 16 steps; cached tracing confirms 30,720 workgroups at local32/VGPR88/LDS0/scratch0. Clean short/512/1K/near-4K profiles improve family/kernel-sum/span/child throughout, and the counterbalanced category gate moves h32 **48.780 -> 50.254 tok/s (+3.022%)** with every category/horizon positive. gfx1100 defaults wave4; `serial_weighted` and unmeasured backends stay fallback (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p0-iq3-wave4-retained.json`). A separately registered gfx1100 K1024 wave4 sign-bit sibling now preserves that exact grid, four K256 accumulators, shuffle trees, partition-add order, BF16 route rows, and unchanged weighted reducer while inserting each sign directly into FP32 bit 31 with no LUT/load. Exhaustive 128-selector/256-grid GPU and CPU gates are byte-exact. Repository Clang-22 codegen contracts **527 -> 499 instructions**, logical SGPR **28 -> 18**, and sign compares/cndmasks **32/32 -> 0/0** while remaining allocated VGPR88/LDS0/scratch0. Formal layers-1/45 actual weights improve producer event/wall **8.12-8.17% / 7.33-9.15%** and inclusive producer+reducer **6.42-8.55% / 6.20-7.80%** with zero mismatches. Its temporary default-off runtime schedule is full-state exact through 16 transitions and cache-only tracing records 45 candidate producers plus 47 unchanged reducers/token, zero retained-wave4 decode producers, and 723 model kernels/token. The frozen clean gate rejects runtime promotion despite producer/inclusive improvements in both short orders: dispatch span regresses **0.571%/1.931%** and order-A profiled-child throughput regresses **1.124%**, outside the 0.5% guards. Remaining profiles/categories are skipped and schedule/CLI selection is removed; retained wave4 stays canonical while the separately registered primitive remains diagnostic (`benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq3-signbit-{correctness,runtime-correctness}.json`, `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq3-signbit-rejected.json`). The retained explicit-only gfx1100 Laguna DFlash tile4 leaf shares each K1024 BF16 activation group across four adjacent IQ3 down outputs while preserving every tile1 accumulator and reduction order. Actual layers 1/45 at 50 selected lanes are BF16-bit exact and improve inclusive events **37.35-38.24%**; the slower tile2 sibling was removed. Full prefill, verifier taps/captures/logits/acceptance, active KV/`KVLiveSpans`, reset, and lifecycle are exact. Cached four-cycle profiling names 45 tile4 calls/cycle at local128/VGPR40/LDS512/scratch0, moves the IQ3 family **11.646 -> 7.726 ms/cycle (-33.66%)**, target-verify kernel sum **64.874 -> 60.968 ms (-6.02%)**, and target-verify wall **73.955 -> 70.220 ms (-5.05%)**. Two process-order pairs retain h32 DFlash **32.307 -> 33.834 tok/s (+4.725%)** and h128 **27.790 -> 29.050 (+4.536%)**, with every category/heldout decode and E2E row positive. Tile1 remains fallback and automatic DFlash stays off at only **0.6915x/0.6338x** true AR; tile4 remains explicit rather than broadening to unmeasured models/backends. Quant math/layout references are qwen-kernel `52e240f9c6d91750d0e5e692976cfb67fd9bc603` and llama.cpp `1ebf790cda38d827559548f67b0469189690cc8c`. |
| Laguna exact IQ3 K1024 ten-wave fused weighted-down primitive (retained gfx1100 default) | `moe_linear/gguf_iq3_xxs/selected_weighted_down_gemv_decode_k1024_wave10_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.{hip,py}`, `tests/test_laguna_iq3_wave10_fused.py` | One local320 output workgroup groups the ten retained route-parallel wave32 owners. Every wave preserves the wave4 IQ3 decode, four K256 accumulators, FMA/shuffle/partition-add order, and BF16 route rounding; ten route values occupy fixed 20-byte LDS, then thread 0 replays the registered `+0.0` slot-order F32-weight `fmaf` reducer after one barrier. The retained wave4 producer plus registered weighted sum remains the mandatory unfused fallback. Exhaustive selector/grid, BF16/routing edges, invalid experts, and independent CPU KL/top-1 pass; all **45/45** actual Laguna IQ3 outputs are byte-exact. Frozen layers 1/45 improve inclusive event **8.51%/7.68%** and wall **7.18%/6.53%**. Integrated Clang-22 codegen is local320/wave32, **495 instructions / 2,860 B**, logical/allocated VGPR **83/88**, logical SGPR26, fixed/allocated LDS **20/512 B**, private/spills/scratch0, 20 shuffles, and one barrier. Cache-only tracing names the distinct symbol twice at grid/workgroup **983,040/320**, VGPR88/LDS512/scratch0, with a warm **26.120 us** call and no compiler. gfx1151 aliasing is excluded. The gfx1100 owner passes shared-weight bulk prefill, all **48 hidden + 47 routed** boundaries, 16 decode transitions, active K/V and every span field, reset/re-prefill, ownership, teardown, and a no-argument-default versus explicit-wave4 replay at KL0/top-1 100%. Cache-only full-model tracing records **45 candidate + two unchanged reducers / 678 model kernels/token**, 45 retained IQ3 prefill calls, zero candidate prefill/wave4/serial decode calls, and local320/VGPR88/LDS512/scratch0 resources. All eight clean context orders improve inclusive IQ3 **9.71-11.90%**, kernel sum **0.398-1.082%**, and span **0.813-1.998%**, with child throughput inside guard. The complete two-order 18-prompt gate moves h32 **62.318 -> 63.270 tok/s (+1.528%)**, every train/heldout category improves at both horizons, and E2E/prefill/TTFT guards pass. gfx1100 now defaults the composite; explicit wave4, exact-key miss, rows/prefill, and unsupported backends retain the registered unfused chain. Evidence: `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-{correctness,runtime-correctness,retained}.json`. |
| Laguna exact IQ3 K1024 wave10 fused sign-bit primitive (runtime-rejected) | `moe_linear/gguf_iq3_xxs/selected_weighted_down_gemv_decode_k1024_wave10_signbit_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.{hip,py}`, `tests/test_laguna_iq3_wave10_signbit_fused.py` | Separately registered W7900 sibling of the retained local320 wave10 body. It preserves ten physical route waves, four K256 accumulators, activation/weight addresses, FMA/shuffle/partition order, route BF16 rounding, 20-byte LDS tuple, one barrier, slot-order weighted `fmaf`, final BF16 store, and launch topology; only IQ3 signs use the exact load-free FP32 sign-bit helper. RED/GREEN, all 128 selectors/256 grid entries, BF16/routing edges, invalid experts, and ten-case CPU KL/top-1 pass. All **45/45** actual Laguna outputs are byte-exact; layers 1/45 improve event **8.95%/8.39%** and wall **8.36%/7.57%**. Integrated Clang-22 codegen contracts **495 -> 454 instructions / 2,860 -> 2,700 B / logical SGPR26 -> 18**, removes 32 sign compares/cndmasks, and stays logical/allocated VGPR **86/88**, fixed/allocated LDS **20/512 B**, private/spills/scratch0, 20 shuffles, and one barrier. Cache-only tracing names the distinct symbol twice at grid/local **983,040/320**, VGPR88/LDS512/scratch0, with a warm **27.480 us** call and no compiler. gfx1151 aliasing is excluded. A temporary false/default-off `wave10_signbit_fused` owner passed shared-weight bulk prefill, all **48 hidden + 47 routed** boundaries, 16 decode transitions, active K/V and every span field, reset/re-prefill, ownership, and teardown at KL0/top-1 100%. Cache-only full-model tracing recorded **45 candidate + two unchanged reducers / 678 model kernels/token**, 45 retained IQ3 prefill calls, zero candidate prefill/retained-wave10/wave4/serial decode calls, and unchanged local320/VGPR88/LDS512/scratch0 resources. Both short orders improve IQ3 **6.06-6.83%**, kernel sum **1.17-1.81%**, span **1.05-5.04%**, and child throughput. At context 512, both orders still improve IQ3 **5.04-6.52%** and kernel sum **0.014-0.896%**, but order-B span regresses **0.862%**, beyond the frozen +0.5% guard; favorable pooled span **-0.736%** cannot waive it. The gate stops before 1K/3968/categories, and capability/schedule/CLI ownership is removed while the separately registered primitive remains diagnostic. Retained wave10, wave4+weighted, and serial runtime paths are unchanged. Evidence: `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-{correctness,runtime-correctness,rejected}.json`. |
| Laguna IQ4_XS top-10/K1024 weighted composite (runtime-rejected diagnostic) | `moe_linear/gguf_iq4_xs/selected_weighted_down_gemv_decode_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.{hip,py}`, `tests/test_laguna_iq4_weighted_composite.py` | Pre-existing exact gfx1100 primitive certified for Laguna E256/top-10/K1024/N3072 without duplicating or changing its device body, wrapper, package export, or four-axis key. Each route preserves the selected-single dot/reduction and BF16 boundary, then the composite replays ten F32 routing weights in fallback order before final BF16 rounding; the registered selected-single plus weighted reducer remains the unfused fallback. Dedicated **7/7** package/key/backend, signed-zero/subnormal/finite BF16, signed/zero routing, repeated/invalid expert, registered-fallback byte-identity, and independent CPU gates pass (KL max **1.58e-69**, top-1 **100%**). Both real layers 46/47 are byte-exact; the repeated 50-warmup/15x300 isolated boundary improves event/wall **28.59-34.22%**. Clang-22 codegen is local256/wave32, **492 instructions / 2,580 B**, logical/allocated VGPR **78/80**, logical SGPR44, fixed/allocated LDS **32/512 B**, private/spills/scratch0, five shuffles, and two barriers. Cache-only tracing names exactly two grid/local **786,432/256** calls with VGPR80/LDS512/scratch0 and no compiler; gfx1151 aliasing is excluded. A temporary false/default-off gfx1100 owner passes shared-weight bulk prefill, all **48 hidden + 47 routed** boundaries, 16 transitions, active K/V plus every span field, reset/re-prefill, ownership, and lifecycle at KL0/top-1 100%. Full-model tracing records **2 IQ4 candidates + 45 retained IQ3 wave10 calls / 676 model kernels/token**, zero IQ4 split decode calls/reducers, exact resources/IDs/teardown, and no compiler. The frozen short gate nevertheless rejects ownership in both orders: inclusive IQ4 regresses **25.515%/25.191%**, complete kernel sum regresses **0.202%/0.303%**, and span regresses **0.681%/2.198%**; child throughput improves but cannot waive those failures. The gate stops before 512/1K/3968/categories, and capability/session/plan/CLI integration is removed while the certified primitive remains diagnostic. The registered selected-single plus reducer is again the only Laguna IQ4 runtime route; canonical throughput remains **63.270 tok/s / 678 kernels**. Evidence: `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-{design,correctness,runtime-correctness,rejected}.json`. |
| `moe_linear` variants `selected_gemv_decode_bf16_bf16_out` and `selected_dual_silu_gemv_decode_bf16_bf16_out` | `gguf_q3_k` raw rank-3 GGUF NextN expert gate/up weights | `hipengine/kernels/hip_gfx1100/quant/gguf_q3_k_gemv.hip` | `gguf_q3_k_selected_gemv_bf16_bf16_out(...)`, `gguf_q3_k_selected_dual_silu_gemv_bf16_bf16_out(...)` | Task #30 raw-pointer blk.40 kernels. The block decoder mirrors llama.cpp `block_q3_K` (110 bytes/256 values) and consumes resident `[E,N,row_bytes]` bytes without repack. The single primitive is the unfused fallback; dual gate/up preserves both BF16 projection boundaries before SiLU. `HIP_VISIBLE_DEVICES=1 ... pytest -q tests/test_gguf_q3_k_selected_gemv.py` reports 6 passed against independent NumPy dequant/selected math, including a real blk.40 row. Cache-only rocprof confirms single/dual symbols at local256, VGPR 16/24, LDS512B, scratch0. Quant reference: llama.cpp `1ebf790cda38d827559548f67b0469189690cc8c`; selected schedule is hipEngine-original. |
| `moe_linear` grouped scalar variants `selected_dual_grouped_prefill_compact_bf16_bf16_out`, `selected_dual_grouped_prefill_compact_rowbatch4_bf16_bf16_out`, `selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out`, `selected_dual_grouped_prefill_compact_adaptive_bf16_bf16_out`, `selected_dual_grouped_prefill_compact_auto_bf16_bf16_out`, `selected_grouped_prefill_compact_bf16_bf16_out`, `selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out`, `selected_grouped_prefill_compact_auto_bf16_bf16_out`, and `selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out`; compact variants `selected_dual_wmma_prefill_compact_bf16_bf16_out` / `selected_wmma_prefill_compact_bf16_bf16_out` | `gguf_iq2_xs`, `gguf_iq3_xxs`, `gguf_iq4_xs` raw rank-3 GGUF expert weights | `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.hip` | `gguf_iq2_xs_selected_dual_grouped_prefill_compact_{adaptive,auto,rowbatch4}_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out(...)`, `gguf_iq2_xs_selected_dual_wmma_prefill_compact_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_dual_grouped_prefill_compact_{auto,rowbatch4}_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(...)`, `gguf_iq3_xxs_selected_dual_{grouped,wmma}_prefill_compact_bf16_bf16_out(...)`, `gguf_iq4_xs_selected_dual_{grouped,wmma}_prefill_compact_bf16_bf16_out(...)`, `gguf_iq4_xs_selected_{grouped,wmma}_prefill_compact_bf16_bf16_out(...)`, `gguf_iq4_xs_selected_grouped_prefill_compact_auto_bf16_bf16_out(...)`, `gguf_iq4_xs_selected_grouped_prefill_compact_k512_wave32_bf16_bf16_out(...)` | Task-11 raw-IQ grouped-prefill path. **IQ2_XS compact prefill:** selected dual scalar and rowbatch4 are BF16-bit exact to the selected-single oracle at K=3072, including N=1024; compact WMMA passes with KL max `0.0003223`, top-1 `1.0`, and max-relative `0.0078125`. Exact branchless magnitude/sign decoding improves every representative E256/K3072/N1024 scalar case by 21.21-32.26% and every rowbatch4 case by 13.90-27.21%, with VGPR80/88, LDS512B, and scratch0. K=3072 auto now uses one block-uniform device decision from each expert's compact count: batch1 for one row, batch2 for two, and batch4 from three rows onward; once the global average reaches four rows/expert it preserves the original rowbatch4 symbol. Against unconditional rowbatch4, the retained auto policy improves every 16/32/64-token balanced/hot/Zipf leaf by 0.64-13.09%, including balanced 16 tokens `1.378 -> 1.198 ms` (-13.09%); dense 128/512-token calls are mechanically the same rowbatch4 kernel. Adaptive and scalar are BF16-bit exact, and the adaptive leaf is VGPR88/LDS512B/scratch0. Standalone rowbatch2 never won; rowbatch8 won only balanced five-row experts (-4.15%) but regressed the other 14 leaves by 12.25-96.50%, so both external candidates were removed. A pair16 grouped-prefill candidate was exact and cut populated-expert scalar leaves substantially, but was restored to group8 after balanced 16-token scalar regressed 5.25% and short rowbatch4 cases regressed up to 3.61%. Artifacts: `benchmarks/results/2026-07-22-gpu1-iq2-xs-branchless-decode.json`, `benchmarks/results/2026-07-22-gpu1-iq2-xs-pair16-local64.json`, and `benchmarks/results/2026-07-22-gpu1-iq2-xs-adaptive-rowbatch.json`. **Laguna K=3072 extension:** IQ3 scalar/rowbatch schedules now process the extra 128 eight-value groups in the same per-thread order as selected decode without retaining a second gate/up segment; IQ4 grouped and both WMMA templates admit K=3072 directly. Synthetic K=3072 scalar IQ3/IQ4 is BF16-bit exact to selected decode, rowbatch4 is exact to scalar, and WMMA passes KL/top-1 (`IQ3 KL max 0.001332/top-1 1.0`; `IQ4 KL max 1.34e-10/top-1 1.0`). Cache-only GPU1 trace confirms every grouped/WMMA symbol with zero scratch. All registered IQ grouped/WMMA variants consume the existing compact-MoE ABI; at K<=2048 scalar kernels use a static `(expert,out-column)` grid, skip empty experts before weight reads, dequantize each row segment once, and loop sorted routed rows without a host scalar read. IQ3/IQ4 dual scalar plus IQ4 down are BF16-bit exact to selected-single fallbacks across empty/uniform/hot/repeated buckets and counts `1/15/16/17`, including production `K=2048` IQ3 and blk.39 IQ4 gate/up. GPU1 resource traces show scalar allocated VGPR `48/112/64`; general kernels use LDS `512 B`, while K512 wave32 down uses LDS `0`; all have zero scratch. The scalar route is now default-on for eligible assignment counts; `HIPENGINE_GGUF_IQ_GROUPED_PREFILL=0` retains direct rollback, small assignments stay direct, and the three Q6_K down layers use their exact selected primitive. Mixed-64-token full-model native-row-bulk parity is exact (`KL=0`, top-1 `1.0`, max abs `0`); 4096/1 IDs/logit are exact and non-regressive. The retained 512 profile moves raw-IQ time `994.668 -> 613.995 ms` (-38.27%), total kernel sum `4396.145 -> 4078.667 ms` (-7.22%), emits no selected-region memory copies, and adds only `2.663 ms` count/prefix/scatter. Formal paired 512/128 wall is flat within spread (`16.648 -> 16.685 tok/s`, +0.22%), so this is a verified sub-window/default promotion rather than a topline throughput claim. Raw-IQ WMMA remains runtime-rejected after changing the sampled token and collapsing to `0.370 tok/s`; exact scalar output-tile-2 also raised IQ4 VGPR `64 -> 104` and regressed matched leaf time by about 50%, so both remain test-only/closed. **2026-07-20 exact K512 IQ4 wave32:** grouped down has only 16 active 32-value subblocks, so a new four-axis local32 sibling preserves the populated wave's dot/shuffle and the old +0 cross-wave boundary while removing 96 idle lanes, three zero waves, LDS, and barriers; an auto four-axis key selects this leaf only at K=512 and delegates general shapes to the still-registered local128 key. Hot/cold/tail fixtures and balanced production shapes at `4,096/8,192/32,768` compact rows have zero BF16 mismatches. GPU1 micro medians fall 68.91–72.76%; the full 4K trace moves IQ4 down `1,666.039 -> 502.039 ms` (-69.87%), grouped IQ total -41.88%, and total kernel sum -15.31%, with local32/VGPR64/LDS0/scratch0. Exact 512/mixed-4K throughput moves `573.288/523.321 -> 693.325/613.576 tok/s`; artifact: `benchmarks/results/2026-07-20-gpu1-q3-exact-iq4-wave32-prefill.json`. **2026-07-20 exact IQ3 row batching:** the local256 sibling interleaves four independent compact rows through one pair of barriers while preserving each row's lane dot, wave32 shuffle tree, and serial wave-0..7 sum. A package-local auto key keeps RT1 below four rows/expert, where rowbatch4 regresses, and selects rowbatch4 from the measured crossover. Production `E=256,K=2048,N=512` outputs are BF16-bit exact at 1/2/3/4/8/16/32/128 rows per expert; real 32/128-row micro medians improve 13.43/15.74%. The final 4K trace cuts IQ3 `1,093.856 -> 961.231 ms` (-12.12%) and total kernel sum `5,971.059 -> 5,827.503 ms` (-2.40%) with local256/VGPR80/LDS512B/scratch0 and unchanged 8,705 launches. Final 512 is `774.185 tok/s` (counterbalanced aggregate +0.25%, within spread); two exact mixed-4K runs median `684.499 tok/s` (+2.10%). Artifact: `benchmarks/results/2026-07-20-gpu1-q3-exact-iq3-rowbatch4-prefill.json`. Math/schedule reference: qwen-kernel `52e240f9c6d91750d0e5e692976cfb67fd9bc603`; quant tables: llama.cpp `1ebf790cda38d827559548f67b0469189690cc8c`. |
| `moe_linear` variant `selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out` | `gguf_iq2_xs` raw rank-3 gate/up plus caller-owned D4-Q8_1 activations | `hipengine/kernels/hip_gfx1100/quant/gguf_iq2_xs_mmq_prefill.{hip,py}` | `gguf_iq2_xs_selected_dual_mmq32_prefill_q8_1_d4_bf16_bf16_out(...)`, `build_iq2_xs_mmq32_metadata(...)` | Explicit populated-prefill primitive derived from llama.cpp HIP `1ebf790c` `mmq.cuh`/`mma.cuh`. One local128 block covers a 32-row x 32-column expert tile; raw IQ2 signed bytes/scales are expanded once per K256 into 10,240 B LDS and reused across four RDNA3 integer-WMMA minitiles. Dedicated counts `1/15/16/17/31/32/33/64` pass max-relative <=0.05 and the repository KL/top-1 gate; representative E256 checks have KL max <=0.00453/top-1 >=0.98125. Cache-only GPU1 E256/K3072/N1024/top-10 exact-auto -> quantizer-inclusive MMQ32 improves all 256-token cases by 22.49-28.76% and all 512-token cases by 45.03-49.86%; 16-64 tokens and 128-token hot/Zipf regress, so exact adaptive/rowbatch remains the fallback. Rocprof: local128/VGPR104/LDS10240B/scratch0; D4 quantizer local256/VGPR24/scratch0. Retained as an explicit four-axis diagnostic only: actual M512 pricing is **3.336x** over 46 IQ2 gate/up layers, but complete quality rejects it at max KL **0.683239** and sparse P6 repair rejects **85.946%** mismatch / **99.496%** touched-row density. No runtime owner or default exists. Artifacts: `benchmarks/results/2026-07-23-gpu1-iq2-xs-mmq32-prefill.json`, `benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-p6-iq2-mmq-matrix512-rejected.json`. |
| `rotate+selected_dual_pack8_gemv` variants `strided`, `strided_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(...)`, `gemv_awq_selected_dual_pack8_strided_rotate_out_fp16(...)` | `python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16/FP16 (`mismatch=0`, `fp16_mismatch=0`, `fp16_max_abs=0.0`); `rocprofv3` shows FP16 `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<_Float16,false>` with `DurationNs=21523`, `Scratch_Size=0`, `LDS_Block_Size=320`, `Workgroup_Size_X=64` on W7900 |
| `rotate+dual_pack8_gemv` variants `transposed`, `transposed_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_dual_pack8_transposed_rotate_staged_bf16(...)`, `gemv_awq_dual_pack8_transposed_rotate_staged_fp16(...)` | `python3 scripts/smoke.py --mode paro-pack8-rotate-staged-hip --rows 1 --hidden-size 128 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16/FP16 staged rotations and outputs (`mismatch=0`, `rotated_mismatch=0`, `fp16_mismatch=0`, `fp16_rotated_mismatch=0`); D1.1 opt-in graph fixture also matched generated tokens/logits (`final_kl=0`) and `rocprofv3` showed `gemv_awq_dual_pack8_transposed_rotate_staged_kernel<_Float16,true>` with `Scratch_Size=0`, `LDS_Block_Size=512`, `VGPR_Count=104`; 512/128 decode regressed `115.450 -> 110.457 tok/s`, so the runtime default remains off. Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d11-rotate-dual-pack8-fusion-rejected.json` |
| `pack8_gemv`, `dual_pack8_gemv` variants `strided`, `transposed`, `*_fp16`; `pack8_gemm` variants `fusedw4_prefill_fp16`, `fusedw4_prefill_strided_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_pack8_*_bf16(...)`, `gemv_awq_dual_pack8_*_bf16(...)`, `gemv_awq_pack8_*_fp16(...)`, `gemv_awq_dual_pack8_*_fp16(...)`, `awq_fusedw4_prefill_fp16(...)`, `awq_fusedw4_prefill_strided_fp16(...)` | `python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact generic BF16 and FP16 single/dual (`single_mismatch=0/0`, `dual_mismatch=0/0`, `fp16_single_mismatch=0/0`, `fp16_dual_mismatch=0/0`); fused-W4 route fixture gate passes (`max_kl=0.02233`, top-1 `1.0`); `rocprofv3 --kernel-trace` all-layer 512 prefill after the dual-launch wiring shows `awq_fusedw4_prefill_dual_fp16_kernel<32,32>` ran 40 times (`21.957 ms` total, avg `548.9 us`, `Scratch_Size=0`) for paired transposed Q/K and QKV/Z projections, while `awq_fusedw4_prefill_fp16_kernel<32,32,false>` ran 50 times (`14.795 ms` total) for strided V/O/linear-out projections on W7900 |
| `marlin_k_gemv` variant `fma_fp16` | `w4_paro` qweight-neutral Marlin-K FP16 rows==1 decode | `hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.hip` | `gemv_paro_marlin_k_fma_fp16(...)` | `python3 scripts/smoke.py --mode paro-marlin-k-hip --rows 2 --hidden-size 128 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact vs pack8 CPU oracle (`mismatch=0`, `max_abs=0`); `rocprofv3 --kernel-trace` shows `gemv_paro_marlin_k_fma_kernel<_Float16>` (`DurationNs=6720`, `VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`) on W7900. Model fixture gates pass with default qweight-neutral replacement (`max_kl=0.0395688706`, top-1 `1.0`; graph/eager generated IDs match, final KL `0`), and D2.1 3-run diagnostic improves 512/128 and 4K/128 decode by `+5.57%/+5.61%` while lowering tracked peak by `0.411 GiB` vs `HIPENGINE_PARO_MARLIN_K_REPLACE=0`. |
| `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` variants `out`, `out_fp16`, `out_f32` | `bf16`, `fp16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/fused/paro_silu.hip` | `silu_mul_dual_out_bf16(...)`, `silu_mul_dual_out_fp16(...)`, `silu_mul_separate_out_bf16(...)`, `silu_mul_separate_out_fp16(...)`, `silu_mul_separate_out_f32(...)`, `silu_mul_dual_rotate_out_bf16(...)`, `silu_mul_dual_rotate_out_fp16(...)`, `silu_mul_pair_rotate_out_bf16(...)`, `silu_mul_pair_rotate_out_fp16(...)` | `python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 dual SiLU and dual/pair rotate (`*_mismatch=0`, `*_fp16_mismatch=0`); `rocprofv3` shows FP16 `silu_mul_dual_out_kernel<_Float16>` (`DurationNs=1680`, `Scratch_Size=0`), `silu_mul_dual_rotate_out_kernel<_Float16>` (`DurationNs=11960`, `Scratch_Size=0`, `LDS_Block_Size=64`), and `silu_mul_pair_rotate_out_kernel<_Float16>` (`DurationNs=8480`, `Scratch_Size=0`) on W7900. `silu_mul_separate_out_{fp16,bf16}` is a hipEngine-original variant that takes two separate `[rows, features]` buffers (gate, up) instead of a packed `[rows, 2*features]` input; used by the W4 PARO dense shared expert where gate/up have distinct rotations and can’t share a packed layout. `silu_mul_separate_out_f32` computes BF16 gate/up into FP32 output for the GGUF llama-compat verifier selected-intermediate diagnostic; `tests/test_paro_silu_plan.py::test_silu_mul_separate_out_f32_matches_cpu_reference` matches NumPy FP32 SiLU on gfx1151, and cached `rocprofv3 --kernel-trace` shows `(anonymous namespace)::silu_mul_separate_out_f32_kernel(...)` with `DurationNs=2027`, `Scratch_Size=0`, `VGPR_Count=16`. |
| `weighted_sum`, `weighted_lanes_sum`, `weighted_lanes_sum+shared_add`, `weighted_sum+shared_gate+residual`, `shared_gate_combine`, `shared_gate_combine+residual`, plus MoE-tail `shared_gate_combine+residual+rmsnorm` / `weighted_sum+shared_gate+residual+rmsnorm` | `bf16`, `fp16`, `w4_paro` with FP32 weights/gate logits | `hipengine/kernels/hip_gfx1100/fused/paro_combine.hip` | `weighted_sum_out_{bf16,fp16}_f32w(...)`, `weighted_lanes_sum_out_{bf16,fp16}_f32w(...)`, `weighted_lanes_sum_shared_add_out_bf16_f32w(...)`, `weighted_sum_shared_gate_combine_residual*_{bf16,fp16}_f32w(...)`, `shared_gate_combine*_{bf16,fp16}(...)`, `shared_gate_combine_residual_rmsnorm_{gguf_bf16,paro_bf16,paro_fp16}_out(...)`, `weighted_sum_shared_gate_combine_residual_rmsnorm_{gguf_bf16,paro_bf16,paro_fp16}_out(...)` | `HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-q3.txt HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. python3 -m pytest tests/test_moe_tail_next_rmsnorm.py -q` → all 12 aggregate/slot-weighted BF16/FP16 cases are bit-exact for both raw residual and normalized output against the unfused combine→RMSNorm chain over hidden 17/2048 and tokens 1/3; CPU-reference checks also pass. A cached GPU1 `rocprofv3 --kernel-trace` shows all six fused specializations at local256, `LDS_Block_Size=1024`, `Scratch_Size=0`, and allocated VGPR `16–56`. The GGUF decode chain now promotes only the stable selected-aggregate specialization: 37 Q3 boundaries replace combine→next-RMS pairs, reduce graph dispatches `708 -> 671` per token, and improve counterbalanced 512/128 and 4K/128 graph decode `100.195 -> 101.216 tok/s` (+1.02%) and `107.366 -> 108.383 tok/s` (+0.95%). Layers 34/38 and slot-weighted Q4/PARO paths retain the unfused fallback because the one-block weighted specialization regressed its production boundary `0.249685 -> 0.446607 ms` (+78.87%) over decode16. The original combine smoke remains `python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 weighted/shared/residual combine including batched selected weighted + shared/residual and grouped sorted-lane accumulation (`*_mismatch=0`, `*_fp16_mismatch=0`); `rocprofv3` shows weighted-lane and batch combine kernels, including `weighted_lanes_sum_out_kernel<unsigned short>` (`DurationNs=2080`), `weighted_lanes_sum_out_kernel<_Float16>` (`2000`), and `shared_gate_combine_residual_batch_out_kernel<_Float16>` (`2120`) on W7900. The Laguna grouped BF16 composite preserves the ten slot-order FMAs, explicit selected BF16 boundary, shared BF16 add, and final BF16 RNE exactly: the shuffled 3-token/top-4/19-feature `libm.fmaf` fixture is bit-exact, and cached gfx1151 tracing names `weighted_lanes_sum_shared_add_out_kernel<unsigned short>` at `1.803 us`, local128, VGPR8, LDS0, scratch0. Runtime promotion still requires inclusive full-model timing; registered weighted-lane plus elementwise-add primitives remain its unfused fallback. |
| `awq_wmma` compact selected dual/single pack8 | `bf16`, `fp16`, `w4_paro` compact grouped MoE | `hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip` | `gemm_awq_selected_dual_pack8_wmma_compact_{bf16,fp16}(...)`, `gemm_awq_selected_pack8_wmma_compact_{bf16,fp16}(...)` | `python3 scripts/smoke.py --mode paro-awq-wmma-compact-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → tiny compact AWQ WMMA fixture passes (`dual_mismatch=0`, `single_mismatch=0`, FP16 mismatches `0`); `rocprofv3` shows compact dual/single WMMA kernels for BF16/FP16, e.g. BF16 dual `DurationNs=10520`, BF16 single `6361`, FP16 dual `6760`, FP16 single `5161` on W7900 |
| `dense_gemv`/`dense_dual_gemv` variants `out`, `out_fp16`, `out_wmma`, `out_fp16_wmma` | `bf16`, `fp16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip` | `dense_gemv_out_bf16(...)`, `dense_dual_gemv_out_bf16(...)`, `dense_gemv_out_fp16(...)`, `dense_dual_gemv_out_fp16(...)`, `dense_gemv_out_bf16_wmma(...)`, `dense_dual_gemv_out_bf16_wmma(...)`, `dense_gemv_out_fp16_wmma(...)`, `dense_dual_gemv_out_fp16_wmma(...)` | `python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 single and FP16 single/dual GEMV (`mismatch=0`, `fp16_mismatch=0`, `dual_fp16_mismatch=0`, max abs `0.0`); `HIPENGINE_HIP_ARCH=gfx1100 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt PYTHONPATH=. pytest -q tests/test_dense_gemv_plan.py tests/test_dense_gemv_wmma.py` → `6 passed`, including FP16/BF16 WMMA-vs-naive checks on verifier-shaped rows. `rocprofv3` shows FP16 `dense_gemv_out_kernel<_Float16>` (`DurationNs=3440`, `Scratch_Size=0`, `LDS_Block_Size=1024`) and `dense_dual_gemv_out_kernel<_Float16>` (`DurationNs=4040`, `Scratch_Size=0`) on W7900. WMMA variants are diagnostic/default-off for verifier A/B (`HIPENGINE_VERIFY_DENSE_GEMV_WMMA=on`) because 2026-05-25 W7900 microbench rejected the skinny shapes (`5x5120x48` GEMV `0.0066 ms` vs WMMA `0.0322 ms`). |
| `lm_head` variants `fp16_argmax_bf16`, `fp16_argmax_bf16_rows_i32`; `argmax` variant `f32_rows_i32`; `topk` variant `f32_rows_i32` | `w4_paro` BF16 hidden + FP16 checkpoint head; DFlash verifier row top-1 and drafter top-k | `hipengine/kernels/hip_gfx1100/linear/lm_head.hip` | `lm_head_fp16_argmax_bf16(...)`, `lm_head_fp16_argmax_bf16_rows_i32(...)`, `argmax_f32_rows_i32(...)`, `topk_f32_rows_i32(...)` | Scalar: `python3 scripts/smoke.py --mode lm-head-hip --hidden-size 32 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact id/logit (`index_match=True`, `abs=0.0`); `rocprofv3 --kernel-trace` shows `lm_head_fp16_logits_kernel`, `argmax_stage1_kernel`, and `argmax_stage2_kernel` with `Scratch_Size=0` on W7900. Row DFlash path: `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_accept_chain_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --debug-top1-readback` → row lm-head ids `[4, 6, 6, 16]` and target-top1/accept summaries match CPU oracle; `rocprofv3 --kernel-trace` on gfx1151 shows `lm_head_fp16_logits_rows_kernel` (`DurationNs=3687`, `Scratch_Size=0`), `argmax_rows_stage1_i32_kernel` (`1883–9538 ns`, `Scratch_Size=0`), and `argmax_rows_stage2_i32_kernel` (`1683–6893 ns`, `Scratch_Size=0`). DFlash top-k bring-up: `HIPENGINE_HIP_ARCH=gfx1151 python3 - <<'PY' ... topk_f32_rows_i32(...) ...` validates sorted row top-3 with stable lower-id tie breaks (`[[5,1,2],[2,3,0]]`). |
| `mtp_router_topk_softmax` variant `256x8` | `f32` Qwen3.6 MTP router logits/weights | `hipengine/kernels/hip_gfx1100/speculative/mtp.hip` | `mtp_router_topk_softmax_f32(...)` | The gfx11 body assigns one expert to each of 256 threads and repeats deterministic shared-memory pair reduction for top8, preserving lower-index ties and exact FP32 softmax order. The generic `topk_f32_rows_i32 + mtp_softmax_topk_f32` chain remains the unsupported-shape fallback. `tests/test_mtp_input_fusion_kernel.py::test_mtp_router_topk_softmax_matches_generic_topk_path` is exact for values, ids, and routing with 12 equal maxima across wave32 boundaries. Matched W7900 `rocprofv3` over 1,000 steady calls improves **94.516 -> 5.395 us/call (-94.29%)**, **80 B/thread scratch -> 0**, and **48 -> 40 VGPR** at 2.5 KiB LDS. Clean canonical on/off/on preserves three x 240 IDs/214 cycles/16 accepts and improves complete wall **16.202 -> 15.919/15.951 ms/cycle**, capture-adjusted wall **13.962 -> 13.839/13.846**, proposer update **1.222 -> 1.107/1.106**, and MTP throughput **65.188 -> 66.303/66.259 tok/s**. Clean final-child tracing confirms router **115.948 -> 10.741 us/call**, proposer host **1.465 -> 1.328 ms**, and complete host **16.317 -> 16.215 ms**. [`artifact`](../benchmarks/results/2026-07-20-w7900-paro-mtp-n4plus-parallel-router-topk.json). The source is shared by gfx1100/gfx1151; gfx1151 performance remains independently unverified. |
| `sampler` variants `processors_rows`, `temperature_rows_i32`, `temperature_top_logprobs_rows_i32`, `topk_temperature_rows_i32`, `top_p_temperature_rows_i32` | `f32` logits | `hipengine/kernels/hip_gfx1100/sampling/sampler.hip` | `apply_processors_f32_rows(...)`, `sample_temperature_f32_rows_i32(...)`, `sample_temperature_top_logprobs_f32_rows_i32(...)`, `sample_topk_temperature_f32_rows_i32(...)`, `sample_top_p_temperature_f32_rows_i32(...)` | S6/S7 native sampler: finite-clamping logits processors for logit bias, repetition/presence/frequency penalties, suppress-token ids, and step-indexed min-token/EOS masks from compact per-row lists; row-wise full-vocab `top_k=0`; full-vocab `top_logprobs` for `top_k=0`; bounded `1 <= top_k <= 64` with optional host-order top-p/min-p filters over the bounded candidate set; bounded `top_logprobs <= top_k <= 64`; and correctness-first exact full-vocab top-p/min-p temperature sampling. Uses per-row seed, counter-based RNG, selected id/logprob, retained-count reporting, and optional top-logprob metadata. `HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. python3 -m pytest tests/test_gpu_sampler_kernel.py -q` → `10 passed` on W7900/GPU0. Supported PARO c=1 requests and scheduler-owned c>N serial per-slot sampled rows use the native sampler by default; `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` forces host sampling for rollback. True batched c>N, GGUF, bounded `top_logprobs > top_k`, and dynamic forced/repair/JSON/thinking-budget processors still fall back to host sampling. |
| `dflash_accept_chain` variant `i32` | `w4_paro` DFlash chain verifier metadata | `hipengine/kernels/hip_gfx1100/speculative/dflash_accept.hip` | `dflash_accept_chain_i32(...)` | `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_accept_chain_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --debug-top1-readback` → reject/partial/full single-request patterns, multi-request real `TargetVerifyBatch` rows, and budgeted no-bonus cases match `TargetVerifyBatch.accept_from_top1` / `TargetAcceptSummary.from_accept_result`; `rocprofv3 --kernel-trace` on gfx1151 shows `dflash_accept_chain_i32_kernel` count=5, `DurationNs≈4007–17032`, `Scratch_Size=0`. |
| `dflash_commit_chain` variant `i32` | `w4_paro` DFlash verified state/KV/output commit | `hipengine/kernels/hip_gfx1100/speculative/dflash_commit.hip` | `dflash_commit_chain_i32(...)` | `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_commit_chain_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → reject/partial/full, budgeted no-bonus, and multi-request prefix-offset fixtures copy only accepted linear state, K/V path rows, hidden taps, output ids, and context metadata; rejected suffix rows stay zero/`-1` and no accepted-prefix target re-forward is used. `rocprofv3 --kernel-trace` on gfx1151 shows `dflash_commit_chain_i32_kernel` count=5, `DurationNs≈3206–11903`, `Scratch_Size=0`. |
| `dflash_prepare_noise_inputs` variants `bf16_i32`, `f16_to_bf16_i32`; `dflash_add` variant `bf16`; `dflash_concat_rows` variants `f32`, `bf16`; `dflash_rmsnorm` variant `bf16`; `dflash_silu_mul` variant `bf16`; `dflash_dense` variants `bf16_to_bf16`, `bf16_to_f32`; `dflash_head_rmsnorm_rotary` variant `f32_bf16`; `dflash_gqa_attention` variant `f32_bf16` | `w4_paro` DFlash drafter root/query prep and correctness-first tiny decoder block path | `hipengine/kernels/hip_gfx1100/speculative/dflash_drafter.hip` | `dflash_prepare_noise_inputs_bf16_i32(...)`, `dflash_prepare_noise_inputs_f16_to_bf16_i32(...)`, `dflash_add_bf16(...)`, `dflash_concat_rows_{f32,bf16}(...)`, `dflash_rmsnorm_bf16(...)`, `dflash_silu_mul_bf16(...)`, `dflash_dense_bf16_to_{bf16,f32}(...)`, `dflash_head_rmsnorm_rotary_f32(...)`, `dflash_gqa_attention_f32_bf16(...)` | `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_drafter_root_query_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` validates root/mask prep, FP16 target-embedding to BF16 conversion, BF16 add/concat, standard direct-weight RMSNorm, BF16 dense projections, SiLU*up, direct-weight head RMSNorm+rotary, non-causal GQA, and a deterministic one-layer tiny DFlash decoder block. Tiny block native top-k matches `fixtures/dflash/drafter_root_query_parent_fixture.json` parent/PyTorch top-k exactly (`[[5,9,6],[8,2,5]]`), with native-vs-parent logits `max_abs=4.802e-03`. `rocprofv3 --kernel-trace` on gfx1151 shows all DFlash drafter kernels plus `topk_rows_i32_kernel` with `Scratch_Size=0`; representative durations: add `1403–1643 ns`, concat `1563–2244 ns`, RMSNorm `2164–8015 ns`, dense BF16→BF16 `1723–2445 ns`, dense BF16→F32 `1403–2645 ns`, SiLU `1723–2204 ns`, head rotary `2846–3046 ns`, GQA `4208–4889 ns`, top-k `51497 ns`. |
| `dflash_qkv_proj` variant `bf16_mixed_indexed_v`; `dflash_head_rmsnorm_rotary` variant `f32_bf16_indexed_key` | `w4_paro` DFlash/MTP graph-safe proposer cache writes | `hipengine/kernels/hip_gfx1100/speculative/dflash_drafter.hip` | `dflash_qkv_proj_bf16_mixed_indexed_v(...)`, `dflash_head_rmsnorm_rotary_indexed_key_f32(...)` | `HIPENGINE_HIP_ARCH=gfx1100 python3 scripts/dflash_drafter_root_query_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` validates that the indexed QKV producer writes V into `value_cache_base[cache_slot + row]`, indexed head RMSNorm+RoPE writes K into `key_cache_base[cache_slot + row]`, direct Q/K/query outputs match the existing direct-output kernels, and sentinel cache rows are preserved. Quicksort MTP D32 with `HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE=1` stays exact with accepted lengths `[3,3,2,0,2,0,0,1,3,0,2,0,2]`. The D32 9-prompt suite is exact but no-held as a speed row (`0.9237x -> 0.9215x`, wall `19.376 -> 19.412 ms/cycle`), so the runtime flag remains default-off and this is retained as M12.7 graph-body infrastructure only. Artifact: `benchmarks/results/2026-06-12-hipengine-mtp-proposer-indexed-kv-write-nohold.json`. |
| `dflash_key_rmsnorm_rotary` variant `f32_bf16`; `dflash_update_kv_metadata` variant `i32` | `w4_paro` DFlash incremental draft context K/V materialization | `hipengine/kernels/hip_gfx1100/speculative/dflash_drafter.hip` | `dflash_key_rmsnorm_rotary_f32(...)`, `dflash_update_kv_metadata_i32(...)`, composed by `materialize_dflash_draft_kv_append_from_projected(...)` | `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_context_kv_materializer_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` appends projected hidden rows in two cycles, projects only new rows, applies direct K norm/RoPE, writes fixed draft K/V cache, updates positions/live count, and matches full-context NumPy reference (`key_abs=2.384e-07`, BF16 values exact; suffix sentinel rows preserved). `rocprofv3 --kernel-trace` on gfx1151 shows dense BF16→F32 count=4 `1322–3768 ns`, key RMSNorm+RoPE count=4 `1683–3447 ns`, dense BF16→BF16 count=4 `1282–2084 ns`, metadata update count=2 `1162–1684 ns`, all `Scratch_Size=0`. |
| `moe_linear` variants `selected_dual_wmma_prefill_compact_{bf16,fp16}_{bf16,fp16}_out` (P8.4) | `gguf_q4_k` raw rank-3 expert weights, compact grouped-MoE gate+up prefill | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_selected_prefill.hip` | `gguf_q4_k_selected_dual_wmma_prefill_compact_{bf16,fp16}_{bf16,fp16}_out(...)` | Mirrors `gemm_awq_selected_dual_pack8_wmma_compact_kernel<scalar_t>` and consumes the same compact-MoE ABI emitted by `qwen35_moe_group_count/prefix/scatter_gather/wmma_tile_map`: `x[compact_rows,in_features]`, `expert_start_compact`, `wmma_expert_start`, `tile_expert`, raw Q4_K expert tensors `[E,out_features,row_bytes]`, and a concatenated row-major output `[compact_rows,out_features_a+out_features_b]`. Inner loop swaps AWQ `(qweight,qzeros,scales)` for raw `block_q4_K` dequant (144 B per 256-K block; 8 subblocks of 32; two 16-wide WMMA K-tiles per subblock). `__launch_bounds__(32,2)`, one wave32 per 16x16 compact tile, output feature counts must be multiples of 16 to preserve the AWQ A/B tile split. No new compact-MoE ABI and no sidecar/repack. Wrapper import/registry smoke passes; HIP source compiles to `gguf_q4_k_selected_prefill.so` on W7900. `tests/test_gguf_q4_k_selected_wmma_prefill.py` covers registry/build-plan/contract checks plus BF16/FP16 compact-MoE correctness against CPU `gguf_quant_gemv(..., Q4_K)` across multiple experts, uneven row counts, padding, empty experts, and tile-boundary shapes (`10 passed` narrow; adjacent Q4/dispatch bundle `94 passed`). `rocprofv3 --kernel-trace` on the tiny selected test confirms `gguf_q4_k_selected_dual_wmma_prefill_compact_kernel<unsigned short>` launches (`DurationNs=18438`). |
| `moe_linear` variants `selected_dual_pack8_gemv_decode_compact_{bf16,fp16}_{bf16,fp16}_out` (P9.B1) | `gguf_q4_k` raw rank-3 expert weights, compact grouped-MoE gate+up GEMV decode | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_selected_pack8_gemv.hip` | `gguf_q4_k_selected_dual_pack8_gemv_decode_compact_{bf16,fp16}_{bf16,fp16}_out(...)` | Mirrors `paro_awq_gemv.hip::gemv_awq_selected_dual_pack8_strided_kernel` (PARO-style pack8 row layout, `__launch_bounds__(128, 4)`, 4 wave32 wave-level reduction across `xchg[8*8]`) but consumes the P8.4 compact-MoE ABI: `x[compact_rows,in_features]`, `expert_start_compact[E+1]`, raw Q4_K expert tensors `[E,out_features,row_bytes]`, row-major concatenated output `[compact_rows,out_features_a+out_features_b]`. Per-block hoists `d * scale[sb]` and `dmin * min[sb]` for all 8 output channels into shared memory (`s_scale[64]` + `s_min[64]`) so the 256-K-per-block inner loop stays in registers. Expert id is recovered via a linear scan over `expert_start_compact` inside the kernel (decode rows=1 per active expert lane; `num_experts` is small in practice). Grid: `((out_features_a + out_features_b) / 8, compact_rows)`. Constraints: `in_features % 256 == 0`, `out_features_a % 8 == 0`, `out_features_b % 8 == 0`; no new compact-MoE ABI and no sidecar/repack. Wrapper import/registry smoke passes; HIP source compiles to `gguf_q4_k_selected_pack8_gemv.so` on W7900. Inline GPU smoke vs CPU `gguf_quant_gemv(..., Q4_K)` on (compact_rows=16, in=256, out_a=16, out_b=32, E=3, expert layout [8,0,8]): BF16 max|d|=`0.801`, max_rel(eps=1)=`0.0038`; FP16 max|d|=`0.058`, max_rel=`0.00047` -- within BF16/FP16 output-rounding tolerance. Formal compact-MoE correctness fixture lands in task #24 (P9.B5). |
| `linear` variants `pack8_gemv_decode_{bf16,fp16}_{bf16,fp16,f32}_out` (P9.B4) | `gguf_q4_k` raw GGUF weights, dense decode-shaped pack8 GEMV (single output) | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_pack8_gemv.hip` | `gguf_q4_k_pack8_gemv_decode_{bf16,fp16}_{bf16,fp16,f32}_out(...)` | Mirrors PARO `gemv_awq_pack8_kernel` single-output structure: `__launch_bounds__(128, 4)` 4 wave32 waves per block, per-block hoist of `d * scale[sb]` and `dmin * min[sb]` into shared memory (`s_scale[64]` + `s_min[64]`), inner k loop strides 256 elements per Q4_K block with register-resident dequant, wave-level `__shfl_down` reduce + cross-wave `xchg[4*8]` sum. Mixed input/output dtypes: BF16/BF16 and FP16/FP16 for attention QKV/O surfaces; BF16/F32 and FP16/F32 for the lm-head logits projection when the tied output is Q4_K. Grid: `(out_features / 8, rows)`. Constraints: `in_features % 256 == 0`, `out_features % 8 == 0`; no new ABI and no resident weight repack. Inline GPU smoke vs CPU `gguf_quant_gemv(..., Q4_K)` on (rows=4, in=512, out=32): BF16/BF16 max|d|=`0.939`, max_rel(eps=1)=`0.0035`; FP16/FP16 max|d|=`0.061`, max_rel=`0.00044`; BF16/F32 (lm-head) max|d|=`6.1e-5`, max_rel=`2.5e-6` (essentially bit-exact); FP16/F32 max|d|=`7.2e-5`, max_rel=`2.1e-5`. Formal correctness fixture in `tests/test_gguf_q4_k_pack8_gemv_decode.py` (P9.B5, 23 tests pass on W7900). |
| Laguna exact Q4 LM-head local32 fixed metadata (retained gfx1100 default) | `linear/gguf_q4_k/local32_fixed_meta_gemv_decode_bf16_f32_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_pack8_gemv.{hip,py}`, `tests/test_gguf_q4_k_pack8_gemv_decode.py` | One wave owns two adjacent c=1 output rows and reconstructs the retained local128 partitions `lane+{0,32,64,96}` independently, preserving every FMA, wave tree, and 0..3 partition add. Production K3072/N100352 F32 logits are bit-exact. Repository codegen is local32/wave32, logical/allocated VGPR **70/72**, logical SGPR61, LDS/private/spills/scratch0, **443 instructions / 2,520 bytes**, zero barriers, and 40 shuffles. The frozen 50-warmup/15-counterbalanced/200-launch actual-weight gate improves event **449.31 -> 314.34 us (-30.04%)** and synchronized wall **449.17 -> 352.87 us (-21.44%)**. Cached `rocprofv3` names the sibling at grid/workgroup **1,605,632/32**, VGPR72/LDS0/scratch0, 265.44 us. gfx1151 aliasing is explicitly excluded. The gfx1100 owner passes 16-transition full state and cached one-candidate/zero-retained tracing at unchanged 723 model kernels/token. Both clean process orders improve the LM head **29.07-30.79%** and complete kernel sum **0.34-1.10%** across short/512/1K/near-4K. Both complete category orders move paired h32 decode **61.675 -> 61.992 tok/s (+0.512%)** with every train/heldout category positive, so gfx1100 now defaults local32; explicit `False`, bulk-prefill/verifier projection, rows>1, unsupported backend, and key miss retain local128. Evidence: `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-correctness.json`, `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-runtime-correctness.json`, and `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-retained.json`. |
| Laguna exact Q6 local32 standalone primitive (runtime-rejected diagnostic) | `linear/gguf_q6_k/standalone_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.{hip,py}`, `tests/test_gguf_q6_k_local32_standalone.py` | Exposes the Q6 local32 helper already retained in the mixed-attention quad. One wave owns two c=1 BF16 output rows and carries original local128 partitions `lane+{0,32,64,96}` independently, preserving every `k/k+128` FMA, wave tree, 0..3 partition add, and BF16 store. Synthetic K256/1024/3072/9216/12288 and N2/8/1024/3072 boundaries plus all 50 actual Laguna Q6 tensors are bit-exact; CPU-reference KL mean is **2.96e-5** with top-1 **100%**. Repository codegen is local32/wave32, logical/allocated VGPR **75/80**, logical SGPR18, LDS/private/spills/scratch0, **451 instructions / 2,816 bytes**, zero barriers, and 56 shuffles. All six repeated actual-weight endpoints improve **11.42-22.32% event** and **11.46-20.51% wall**. Cached primitive `rocprofv3` names grid/workgroup **49,152/32**, VGPR80/LDS0/scratch0 at 11.000 us. gfx1151 aliasing is excluded. A temporary default-off gfx1100 owner passed byte-exact 16-transition state and cached **50-call/723-kernel** tracing, but the frozen short clean gate rejected runtime selection: order A regressed kernel sum/span **0.153%/2.092%**, while order B regressed profiled-child throughput **0.642%**. Runtime capability/session/CLI/library selection is removed; F32 mixed attention, rows/prefill, Q4/Q5/Q8, key misses, and gfx1151 remain unchanged. Evidence: `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-correctness.json`, `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-runtime-correctness.json`, and `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-rejected.json`. |
| Laguna exact paired-output SWAR Q5 primitive (runtime-rejected diagnostic) | `linear/gguf_q5_k/wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out`; matching `linear_pair`; `attention_projection_quad/.../mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.{hip,py}`, `tests/test_laguna_q5_swar_pair.py` | One local32 wave retains two adjacent output rows and every FP32 FMA/reduction/store boundary while packing each row pair's `qh`/`qs` bytes into low/high uint8 lanes and sharing `0x0F0F` nibble plus `0x0101` high-bit operations. Direct BF16/F32, unequal-pair BF16/F32, and mixed-Q5/Q6 F32 siblings are separately exported; only the direct BF16, pair BF16, and mixed production keys are registered, with all retained keys as exact fallbacks and gfx1151 excluded. Synthetic K256/512/1024/3072/6144/9216 and independent CPU gates pass; all **47 attention-output + 47 mixed + 46 shared-Q5** actual boundaries are byte-exact. Repeated endpoints improve output **7.92-10.11%**, mixed **5.91-7.52%**, and shared **5.70-5.99%** in HIP events with every wall row positive. Integrated codegen is local32/wave32, LDS/private/spills/scratch0, zero barriers: BF16 direct **470 instructions / 2,688 B / logical VGPR71**, BF16 pair **478 / 2,724 / 71**, F32 pair **461 / 2,640 / 71**, mixed **923 / 5,388 / 75**. Cache-only tracing allocates VGPR72 direct/pair and VGPR80 mixed. The prior temporary all-role owner is removed after both clean short orders regress mixed projection **2.278%/1.694%** and shared gate/up **0.902%/0.439%**, but its immutable traces independently improve the 47-call attention-output role **1.952%/2.046%**. A later false/default-off output-only owner leaves mixed/shared/query-gate controls unchanged, passes byte-exact 16-transition state, and traces **47 candidate calls/token** plus zero excluded SWAR roles at **678 model kernels/token**. The frozen short gate still rejects runtime selection: output-family and kernel-sum time improve **4.001%/3.705%** and **1.452%/0.509%** in orders A/B, but profiled-child throughput regresses **1.061%/1.035%**, outside the -0.5% guard. Runtime capability/session/CLI integration is removed before long contexts/categories; canonical h32 remains **63.270 tok/s**. Evidence: `benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q5-swar-pair-correctness.json`, `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-pair-rejected.json`, `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-design.json`, `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-runtime-correctness.json`, and `benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-rejected.json`. |
| `linear` variants `pack8_gemv_decode_bf16_{bf16,f32}_out` | `gguf_q5_k` raw GGUF weights, dense Laguna decode projections | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.{hip,py}` | `gguf_q5_k_pack8_gemv_decode_bf16_{bf16,f32}_out(...)` | Exact raw-Q5 pack8 schedule for Laguna Q2 XL. One local128 block owns eight output columns and hoists all `d*scale`/`dmin*min` coefficients for each 256-value superblock into 1,024 B LDS while preserving the generic kernel's per-thread K order and wave/cross-wave reduction. The focused W7900 gate is bit-exact to the existing raw pack8 BF16/F32 outputs at the production 48/72/1,024-column K3072 shapes. Actual-weight screening moves Q5 Q/O/gate/shared/dense representatives by 62.9-80.5%; cached rocprof reports local128, VGPR48/72, LDS1024, scratch0. Full 32-token quality and serial-vs-bulk logits/hidden/taps/KV checks pass. The clean ten-prompt W7900 gate moves default h32 decode 19.565 -> 35.419 tok/s (+81.04%) with every category and E2E row positive; `benchmarks/results/2026-07-23-gfx1100-laguna-q2-xl-dense-decode-retained.json`. Laguna P3 later tested a bit-lossless Q5 T16 replacement plus an exact local32 wave32x2 sibling: both are byte exact, but the existing T16 leaf regresses actual-layer events **16.46-20.51%** and the wave32x2 sibling regresses **5.28-9.07%** while raising VGPR **96 -> 104**; candidate source/dispatch/tests are removed and raw wave32x2 remains canonical (`benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p3-q5-t16-repack-rejected.json`). |
| `linear` variants `pack8_gemv_decode_{bf16,fp16}_{bf16,fp16,f32}_out` plus top-1/gather variants (P9.B4b, added during P9.B5; llama-compat q8_1/dp4a top-1 added 2026-07-01) | `gguf_q6_k` raw GGUF weights, dense decode-shaped pack8 GEMV (single output) and resident-draft lm-head top-1 | `hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_pack8_gemv.hip` | `gguf_q6_k_pack8_gemv_decode_{bf16,fp16}_{bf16,fp16,f32}_out(...)`, `gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(...)`, `gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32(...)`, X8 sidecar `gguf_q6_k_x8_gemv_decode_q8_1_dp4a_top1_{stage1,gather}_f32(...)`, diagnostics `gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_{scalehoist,pack16}_gather_f32(...)` | Q6_K dense GEMV decode added to enable the Q6_K F32 lm-head correctness fixture required by P9.B5 (the qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M tied output weight is Q6_K). Mirrors the Q4_K dense kernel's structure exactly with the inner k loop swapped for raw Q6_K block dequant (per-16-K int8 scales x fp16 super-scale, 2 high bits per element; same per-block scale hoist as the P9.B2 Q6_K selected variant). The exact BF16 top-1/gather variant preserves logits->top-k semantics while avoiding full-logits materialization for resident device-chain `top_k==1`. The q8_1/dp4a top-1/gather variant consumes GGML q8_1 activations and is accuracy-traded/default-off for the llama-compat draft lm-head path. The `pack8_scalehoist` diagnostic keeps pack8's `vocab/8` final reduce but hoists Q6_K `d*scale` values into shared memory; correctness passes and `rocprofv3` confirms `gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_scalehoist_stage1_kernel` launches (`Workgroup_Size_X=128/64`), but same-session compat smoke rejects it (**68.65 -> 68.54 tok/s**, `draft_initial` **2.482 -> 2.485 ms/output**), so it remains evidence-only. The `pack16` diagnostic handles 16 vocab rows per stage1 block; correctness passes, but smoke rejects it (**71.74 -> 71.72 tok/s**, `draft_initial` **2.479 -> 2.487 ms/output**) and rocprof shows stage1 slower than pack8 (**3.603 -> 3.684 ms/cycle**). Constraints: `in_features % 256 == 0`, `out_features % 8 == 0`; pack16 requires `out_features % 16 == 0`; q8_1 variants also require prequantized q8_1 activation blocks. Tested in `tests/test_gguf_q6_k_pack8_gemv_decode.py`; BF16/F32 lm-head smoke (rows=2, in=256, out=16) max\|d\|=`1.9e-5`, max_rel=`8.1e-7`; q8_1/dp4a top-1 matches the CPU q8_1/Q6_K oracle and rocprofv3 confirms `gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_stage1_kernel` ran (`Workgroup_Size_X=128`, fixture duration `6116 ns`). |
| P9.B6 decode-dispatch wiring | `linear` and `moe_linear` opt-in routing (`HIPENGINE_GGUF_GEMV_DECODE`) | `hipengine/runtime/gguf_linear.py`, `hipengine/runtime/qwen35_gguf_runner.py` | `gemv_decode_session(...)`, `set_gemv_decode_enabled(...)`, `gguf_gemv_decode_enabled(...)`, `Qwen35GGUFResidentSession.use_gemv_decode` | Sibling of `wmma_prefill_session(...)` that controls the `rows == 1` decode rewrite via a new `_gemv_decode_dispatch` and the new `_try_run_post_attention_moe_c1_compact_gemv` runner helper. Same kwarg/session/env precedence as the WMMA prefill toggle. When on: dense Q8_0 `rows == 1` projections rewrite `pack8_gemv_*_out` -> `pack8_gemv_decode_*_out` (P9.B3 single + dual via the existing `launch_gguf_linear` / `launch_gguf_linear_pair` pair fusion), dense Q6_K rewrites use the P9.B4b kernel, and qwen35moe selected MoE c=1 decode routes through the compact scheduler (P8.6 `group_count`/`group_prefix`/`group_scatter_gather`; no `wmma_tile_map`) into the P9.B1 / P9.B2 GEMV decode kernels. The runner inherits the existing weighted-lane combine + shared-gate residual combine primitives from the bulk WMMA compact path. Q4_K dense (LAYOUT_Q4_K_PACK8 with separate `qweight`/`scales`/`mins` allocations) and Q6_K lm-head pack8 (same separate allocations) are intentionally not rewritten -- the P9.B kernels read raw GGUF block bytes and the layouts are incompatible without a runtime weight repack; documented as a P9.D follow-up. Registry-miss + scratch-miss safety: both rewrites use `registry.is_registered(key)` (an exact-key check, not `resolve` with fallback) so a partial build or a stripped scratch transparently falls back to the legacy decoder. Validation: `tests/test_gguf_gemv_decode_dispatch.py` (13 tests) covers default-off, kwarg/session/env opt-in, session-restore semantics, prefill-path-unaffected, and missing-key fallback. `tests/test_qwen35_gguf_compact_moe_gemv_routing.py` (4 tests) covers the c=1 compact MoE routing decision tree, including missing-kernel and missing-scratch fallback. Adjacent regression bundle: 73 pass. |
| P9.C1 Q8_0 dual gate+up WMMA prefill + tile heuristic tune (PARTIAL acceptance) | `gguf_q8_0_prefill_dual_wmma_kernel<scalar_t, out_t, TM, TN>` (new) plus revised `_default_tiles` heuristic | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_prefill.{hip,py}` | `gguf_q8_0_wmma_prefill_dual_gate_up_{bf16,fp16}_{bf16,fp16}_out(...)` | New dual variant mirrors `gguf_q8_0_prefill_wmma_kernel` (`__launch_bounds__(32, 8)`, 4 wave32 reduce) but takes (`qweight_a`, `qweight_b`) and writes the row-major concatenated layout `[rows, out_features_a + out_features_b]` that `silu_mul_dual_out_*` consumes directly. Constraints: `out_features_a % tile_m == 0` and `out_features_b % tile_m == 0` so a col_tile never straddles the gate/up boundary. P9.C1 microbench (rows=512, BF16/BF16, RX 7900 XTX): single-kernel `out=4096` (shexp gate/up) `(32, 32) -> (64, 32)` is `~2x` faster (`0.643 ms -> 0.327 ms`); single-kernel `out=2048` (attn QKV/O) `(32, 32)` stays within 1% of `(16, 32)` and is left at `(32, 32)`. The default `_default_tiles(rows, out_features)` now returns `(64, 32)` for `out_features >= 4096` and keeps `(32, 32)` otherwise. Initial partial end-to-end on Qwen3.6-35B-A3B-UD-Q4_K_M 512/0 (heuristic only, before the follow-up row below wired Q8_0 dual shared-expert dispatch and selected-MoE TM/TN sweeps): dense Q8_0 WMMA prefill bucket `75.245 ms -> 65.592 ms` (`-12.8%`); combined WMMA bucket `170.410 -> 160.363 ms`. **Acceptance target `<= 110 ms` NOT met (`160 ms` actual)**; superseded by the following P9.C1 blocked row, which carries the final retained choices and artifact. Correctness fixtures from P8.4/P8.5 still pass (`113 passed` across new dual + existing single Q8_0 + Q4_K/Q5_K/Q6_K selected WMMA tests). |
| P9.C1 selected-MoE WMMA tile sweep + Q8_0 shared-expert dual wiring (BLOCKED acceptance) | Tunable selected raw-Q4_K/Q5_K/Q6_K compact WMMA wrappers plus `launch_gguf_linear_pair_concat(...)` | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_selected_prefill.{hip,py}`, `hipengine/kernels/hip_gfx1100/quant/gguf_k_selected_prefill.{hip,py}`, `hipengine/runtime/gguf_linear.py`, `hipengine/runtime/qwen35_gguf_runner.py` | Q4_K selected dual defaults to `(tile_m,tile_n)=(32,16)`; Q5_K/Q6_K selected down keep legacy `(16,16)`; Q8_0 shared gate+up uses dual concat WMMA at `(16,32)`; Q8_0 single WMMA uses shape-aware tiles (`2048x8192`/`2048x4096` -> `16x32`, `4096x2048` -> `64x32`, `2048x512` -> `16x32`) | Follow-up to the initial P9.C1 partial. Added generic TM/TN selected-MoE WMMA variants for the requested sweep space `(TM in {16,32,64}, TN in {16,32})` while preserving the original legacy 16x16 kernels as the fast default path when selected. E2E sweeps showed the generic multi-tile path increases register pressure enough that only Q4_K dual `32x16` is worth retaining; Q5_K/Q6_K multi-tile variants are slower and remain on legacy `16x16`. Added `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` as a compile-time sweep knob for selected kernels; P9.C1 launch-bounds sweep over min-blocks `{1,2,4,8}` retained the P8 hint `__launch_bounds__(32,2)` (`1435.3 tok/s` single-run vs `1434.7`, `1418.4`, `1427.2`). The Q8_0 dual concat kernel is now wired through a registry-gated `launch_gguf_linear_pair_concat(...)` path for qwen35moe shared gate+up, reusing `scratch.ffn_gate_up` and `silu_mul_dual_out_bf16`. A follow-up targeted Q8_0 shape sweep fixed the overly broad `out>=4096 -> 64x32` rule: `linear_qkv/full_q (512x2048x8192)` prefers `16x32` (`0.538 ms` synthetic vs `0.818 ms` for `64x32`), `linear_gate (512x2048x4096)` prefers `16x32`, `ssm/shared_down (512x4096x2048)` keeps `64x32`, and full-attn `k/v (512x2048x512)` uses `16x32`. 512/0 rocprof (RX 7900 XTX, Qwen3.6-35B-A3B-UD-Q4_K_M, cached builds, P9.E1 classifier updated so Q8 dual WMMA is in the Q8 bucket): dense Q8_0 WMMA `52.187 ms / 210 dispatches`, Q4_K selected dual `57.728 ms / 40`, Q5_K selected `26.833 ms / 37`, Q6_K selected `2.694 ms / 3`; combined WMMA bucket `139.442 ms` vs target `<=110 ms` (NOT met) and vs prior `170.410 ms` baseline (`-18.2%`). 512/0 wall prefill median `1678.00 tok/s` over 3 runs. Correctness: adjacent bundle passes (`tests/test_gguf_q8_0_wmma_prefill_dual.py`, `test_gguf_q8_0_wmma_prefill.py`, `test_gguf_q4_k_selected_wmma_prefill.py`, `test_gguf_k_selected_wmma_prefill.py`, `test_gguf_gemv_decode_dispatch.py`, `test_qwen35_gguf_compact_moe_gemv_routing.py`). Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c1-wmma-tile-sweep-blocked.json`. Acceptance remains blocked: residual ~29 ms requires a different selected-MoE kernel design beyond this raw generic TM/TN/Q8-shape sweep. Follow-up option tests retained in the artifact show: selected MoE is still the highest-impact bucket (`87.3 ms` vs Q8 `52.2 ms`); padding removal is bounded (`13-18%` padded-row overhead, ideal `~11 ms` max); bulk compact GEMV/no-padding is much slower (`261 tok/s`); existing pack8 sidecar without WMMA is much slower (`69.7 tok/s`); sidecar with WMMA does not change the raw selected kernels and is slightly slower; existing PARO/AWQ pack8 WMMA as a repack proxy is slower for qwen35-like shapes (`6.65 ms` Q4 dual layer proxy). Most fruitful next path is a custom raw GGUF-K selected WMMA redesign for hot experts / predecoded Q4 scale-min sidecar, with tail/no-padding as a secondary optimization. |
| P9.C2 hot-expert selected-MoE replay harness | `scripts/qwen35_gguf_moe_replay.py` | live qwen35moe compact-MoE scheduler intercept + replay timing for selected Q4_K dual and Q5_K/Q6_K down WMMA kernels | The harness runs one real qwen35moe bulk prefill, intercepts `_try_run_post_attention_moe_rows_compact_wmma`, records per-layer expert counts, compact/wmma starts, tile-expert maps, quant keys, selected/default tile decisions, padding/tail overhead, and batched replay timings for gate+up/down kernels using the resident raw GGUF-K weight pointers. Validation run: `HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt PYTHONPATH=. python3 scripts/qwen35_gguf_moe_replay.py --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --prompt-length 512 --token-id 9707 --warmup-iters 1 --replay-iters 5 --sample-groups 3 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --json /tmp/p9_c2/moe-replay-512-0-batched.json`. Result: 40 MoE layers; replay selected-MoE total `89.724 ms` vs P9.C1 rocprof reference `87.255 ms` (`+2.83%`); components Q4 dual `59.896 ms` vs `57.728 ms` (`+3.76%`), Q5 down `26.999 ms` vs `26.833 ms` (`+0.62%`), Q6 down `2.828 ms` vs `2.694 ms` (`+4.98%`). Routing summary: `163,840` compact rows, `185,872` WMMA rows (`+13.45%` padding), `19.20%` nonzero experts, p50 nonzero count `3`, p90 `404`, p99 `510`, max `512`; hot thresholds record `>=64` experts: `501` instances / `153,241` rows, `>=128`: `390` / `143,218` rows. This is the replay baseline for P9.C3/P9.C4 hot-expert kernel work. |
| P9.C3 current raw GGUF-K selected WMMA profile | `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c3-selected-moe-profile.json` | rocprof/replay/code-object profile for retained Q4_K selected dual 32x16 and Q5_K/Q6_K legacy down kernels | Diagnostic profile using P9.C2 replay harness under `rocprofv3 --kernel-trace`. Retained trace: Q4 dual `61.48 ms / 40 dispatches` (`avg 1.54 ms`, trace VGPR `72`, SGPR `128`, no scratch/LDS, grid_x `1024`), Q5 down `28.80 ms / 37` (`avg 0.78 ms`, VGPR `64`, grid_x `4096`), Q6 down `2.82 ms / 3` (`avg 0.94 ms`, VGPR `72`). Q4 legacy 16x16 comparison: `67.51 ms / 40`, trace VGPR `56`, grid_x `2048`; retained 32x16 saves `~6 ms` by halving column blocks despite higher VGPR. Code-object stats from extracted amdgcn object: Q4 legacy num_vgpr `51`, Q4 32x16 `65`, Q4 64x32 `128`; Q5 legacy `64`, Q5 64x32 `139`; Q6 legacy `65`. PMC counters on this ROCm build populated `SQ_WAVES` but instruction/busy counters returned zero, so diagnosis uses trace/code-object/static footprint evidence. Footprint/back-calc: Q4 dual executes `~5.50 TFLOP` at `~89 TFLOP/s`; executed raw-weight tile footprint estimate `~110 GB` and activation-reload envelope `~195 GB` vs unique active-expert weight footprint `~18.6 GB`, pointing at inner-loop raw GGUF-K scale/min decode + repeated weight/activation work rather than stores/scheduler. Ranked plan: prototype a hot-expert Q4 dual kernel that keeps 32x16 block reduction but reduces per-tile Q4 scale/min decode/register pressure (likely predecoded scale/min sidecar); keep trace VGPR `<=72`/no scratch; then apply to Q5 down; tail/no-padding remains secondary. |
| P9.C4 Q4_K hot/full-tile selected dual prototype v1 (rejected) | `gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_{bf16,fp16}_...` | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_selected_prefill.{hip,py}`, replay option `--q4-hot-fulltile-threshold` | Optional experimental path only; not default. It splits Q4 selected dual into a hot full-16-row `32x16` kernel plus tail/cold fallback. Correctness vs CPU selected reference passes on a synthetic hot/cold/tail fixture. Replay results: retained Q4 baseline `59.896 ms`; hot threshold `1`/`32`/`64`/`128` gives Q4 `66.08`/`65.84`/`65.28`/`65.25 ms` respectively. rocprof for threshold=1: hot full-tile kernel `49.21 ms` (trace VGPR `64`, grid_x `1024`) plus compact tail-by-expert fallback `17.47 ms` (VGPR `56`, grid_x `2048`, grid_y `256`). Rejected because fallback launch/grid overhead erases the full-tile kernel gain; not wired into default runtime. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c4-q4-hot-fulltile-v1-rejected.json`. Next path: P9.C5 sidecar/reduced-decode prototype; full-tile kernel only becomes useful if paired with compact tail lists and/or sidecar. |
| P9.C5 Q4_K predecoded scale/min sidecar v1 (rejected) | `gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_{bf16,fp16}_...` | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_selected_prefill.{hip,py}`, replay option `--q4-sidemeta-layers` | Optional experimental path only; not default. Sidecar layout: fp16 `[num_experts, out_features, blocks_per_row, 8, 2]` storing `(scale_f,min_f)` while raw q nibbles remain in GGUF Q4_K bytes. Memory estimate for qwen35moe layer: `268 MiB` per gate/up tensor, `536 MiB` for gate+up, `~15 GiB` for all 30 MoE layers, so it is expensive even before performance. Correctness passes on synthetic CPU-reference fixture. Real first-layer replay: retained raw Q4 gate+up `1.678 ms`; fp16 side metadata `2.339 ms`; total Q4 replay with one side layer `60.944 ms` vs baseline `59.896 ms`. Rejected: added side-metadata memory stream costs more than d/dmin/scale bitfield decode. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c5-q4-sidemeta-v1-rejected.json`. |
| P9.C7 Q5_K selected-down decode-hoist v1 (rejected) | `gguf_q5_k_selected_wmma_prefill_compact_opt_{bf16,fp16}_...` | `hipengine/kernels/hip_gfx1100/quant/gguf_k_selected_prefill.{hip,py}`, replay option `--q5-opt` | Optional experimental path only; not default. It preserves the compact selected-MoE ABI and legacy `16x16` shape but hoists Q5_K `d/dmin` and packed scale/min decode once per 32-value subblock instead of calling the generic element helper for every `kk`. Correctness passes against CPU selected reference for BF16/FP16. Real 512/0 replay regresses Q5 down: retained `26.999 ms`; opt `34.086 ms` (`+26.3%`), selected-MoE total `89.724 -> 96.904 ms`. rocprof opt trace: `36.65 ms / 37 dispatches`, trace VGPR `64`, SGPR `128`, no scratch. Rejected: the hoist raises scheduling/register-lifetime cost enough to lose despite the same VGPR count. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c7-q5-opt-v1-rejected.json`. |
| P9.C8 Q6_K selected-down validation (retained legacy) | existing `gguf_q6_k_selected_wmma_prefill_compact_{bf16,fp16}_...` | replay harness with `HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_{M,N}` | No new default. Q6_K is only `~2.8 ms` across 3 layers in the retained 512/0 replay. Generic tile sweep (3x5 batched replay) kept legacy `16x16` fastest: `16x16 2.839 ms`, `32x16 3.138 ms`, `16x32 3.576 ms`, `32x32 3.608 ms`, `64x16 3.248 ms`, `64x32 4.826 ms`. Since Q5 decode-hoist did not improve and Q6 is a small bucket, no dedicated Q6 hot path is warranted. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c8-q6-retain-legacy.json`. |
| P9.C9 selected-MoE tail/no-padding hybrid decision (not retained) | no new default | P9.C2 replay + P9.C4 measured tail fallback evidence | Revisited after Q4/Q5/Q6 prototypes. Retained 512/0 replay has `163,840` compact rows, `185,872` WMMA rows, `22,032` padding rows (`+13.45%`), but true residual rows are only `8,848` (`5.40%` of compact). Full-tile-only WMMA would remove `30,880` row-equivalents (`16.61%` of WMMA rows) but still needs a tail path. Measured Q4 tail-by-expert fallback from P9.C4 costs `17.47 ms` by itself and made Q4 total slower (`66.08 ms` vs retained `59.90 ms`); bulk no-padding GEMV was already rejected as much slower. A useful version would require a new compact tail-list ABI across Q4/Q5/Q6, not just toggling existing kernels. Not retained because no measured hybrid clears the required `3-5 ms` selected-MoE improvement without slowing hot experts. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c9-tail-no-padding-not-retained.json`. |
| P9.C10 combined threshold/variant tuning gap analysis (blocked target) | retained defaults only | full 512/0 + 512/128 benches, P9.E1 rocprof summary, P9.C2 replay decisions | Current commit retained defaults: Q4 selected dual `32x16`; Q5/Q6 selected down legacy `16x16`; no Q4 hot, Q4 side metadata, Q5 decode-hoist, Q6 larger tile, or tail/no-padding hybrid; Q8_0 stays shape-aware with shared-expert dual `16x32`. Full current benches (Qwen3.6-35B-A3B-UD-Q4_K_M, RX 7900 XTX, cached builds): 512/0 median prefill `1661.79 tok/s` (`0.3081 s`); 512/128 median prefill `1677.56 tok/s`, decode `62.56 tok/s`. Current 512/0 rocprof/P9.E1 bucket summary: Q4 selected `58.126 ms`, Q5 selected `27.043 ms`, Q6 selected `2.656 ms`, dense Q8_0 WMMA `52.285 ms`; combined target bucket `140.110 ms`, target `<=110 ms`, gap `30.110 ms`. Next single bottleneck: Q4 selected dual (`58.1 ms`), but shallow Q4 paths all regressed; a deeper expert-weight repack/layout or different selected-MoE design is required. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c10-combined-gap-analysis.json`. |
| P9.C11 final hot-expert path status (blocked) | retained defaults only | final artifact + correctness bundle | Final P9.C hot-expert pass did not meet acceptance. Adjacent correctness bundle passes (`143` collected tests across Q8_0 dual/single, Q4_K selected, Q5_K/Q6_K selected, dispatch/routing, replay helpers) and 512/128 bench has finite deterministic final token `220`, but 512/0 target bucket remains `140.110 ms` (`Q4 58.126`, `Q5 27.043`, `Q6 2.656`, `Q8 52.285`) vs target `<=110 ms`. Umbrella #27 remains open/blocked; do not mark complete. Next bottleneck is Q4 selected dual, but shallow variants regressed, so the next path requires deeper expert-weight repack/layout or a different selected-MoE design. Artifact: `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c11-hot-expert-final-blocked.json`. |
| P9.C12 Q4T16 selected-dual repack/layout design | future `q4_k_tile16_dual` allocation + Q4 selected dual kernel | design artifact only; implementation starts in P9.C13 | #27 is blocked because retained 512/0 target bucket is `140.110 ms` vs `<=110 ms`, with Q4 selected dual the next bottleneck (`58.126 ms`). P9.C12 selects a deeper Q4T16 tile-major replacement layout for prototype: repack raw Q4_K gate/up weights from `[E,out,row_bytes]` into `[E,out_tile16,k_block]` slabs with fp16 `d/dmin` per column, predecoded uint8 scale/min per subblock/column, and q4 nibbles arranged by `[subblock, kt, k_lane16, col]` for coalesced WMMA B-fragment loads. Actual qwen35moe Q4 shape from GGUF is `E=256`, `hidden=2048`, `expert_ffn=512`, `blocks_per_row=8`; raw gate or up tensor is `150,994,944 B`, Q4T16 is `155,189,248 B` (`+2.78%`). Final runtime must replace raw Q4 gate/up allocation rather than duplicate all layers; replay prototype may use one-layer side buffers. Go/no-go: CPU roundtrip exact, first qwen layer gate+up `<=1.35 ms`, full Q4 replay `<=45 ms` to continue and `<=35 ms` to plausibly unblock #27. Artifact: `benchmarks/results/2026-05-19-hipengine-qwen36-35b-a3b-q4km-p9_c12-q4t16-repack-design.json`. |
| P9.C13 Q4T16 materializer prototype | `repack_gguf_q4_k_tile16`, `unpack_gguf_q4_k_tile16`; replay flag `--q4-tile16-materialize-layers` | `hipengine/quant/gguf_q4_k.py`, `scripts/qwen35_gguf_moe_replay.py`, `tests/test_gguf_q4_k_tile16_repack.py` | CPU-side prototype for the P9.C12 Q4T16 layout. `tiles` shape is `[experts, out_tiles16, blocks_per_row, 2368]`; per tile stores fp16 `d`, fp16 `dmin`, uint8 scale/min vectors, and q4 nibbles packed by `[subblock,k_lane32,col_pair]`. Unit tests validate bit-exact raw GGUF Q4_K roundtrip and expected `2368/2304 = +2.78%` storage overhead. Replay smoke with `--q4-tile16-materialize-layers 1` on Qwen3.6-35B-A3B-UD-Q4_K_M builds/copies the first layer's Q4T16 gate+up buffers (`310,378,496 B`) and frees them without changing compute; this is not a perf claim, only materializer/device-copy readiness for the next HIP kernel prototype. Artifact: `benchmarks/results/2026-05-19-hipengine-qwen36-35b-a3b-q4km-p9_c13-q4t16-materializer.json`. |
| P9.C14 / GPF-3A Q4T16 selected-dual WMMA | `gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_{bf16,fp16}_...`; exact shared-activation variants `...compact32_shared_x_{bf16,fp16}_...`; replay options `--q4-tile16-wmma-layers`, `--q4-t16-shared-x` | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_t16_selected_prefill.{hip,py}`, `scripts/gguf_q4_t16_prefill_ab.py`, `scripts/qwen35_gguf_moe_replay.py`, `tests/test_gguf_q4_k_t16_selected_wmma_prefill.py` | First HIP WMMA consumer for the P9.C13 Q4T16 selected gate/up layout. It preserves the compact selected-MoE ABI and writes the same concatenated gate+up output as raw-Q4 selected WMMA, but reads `tiles[E,out_tiles16,blocks_per_row,2368]` slabs with predecoded uint8 scale/min and q4 nibbles packed by `[subblock,k_lane32,col_pair]`. GPF-3A keeps two independent 16-column WMMA accumulators live so both consume one activation load while preserving each accumulator's K/WMMA order. BF16/FP16 bytes exactly match baseline on uneven/empty/multi-block fixtures. gfx1151 trace: `44.725 -> 33.343 us` (-25.45%), 56 VGPR/128 SGPR/zero scratch/LDS; real 40-layer Q4 gate/up replay `114.633 -> 97.082 ms` (-15.31%). Clean full-model 512/1K/4K prefill improves +3.11%/+2.42%/+1.94%, all logits/trajectories are exact, and aggregate decode is -0.0031%. The 2026-08-05 controlled repeated-128K split identifies `gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_kernel<uint16_t>` as the gfx1151 trigger: conservative and exact-GDN arms pass 12/12, shared-X stalls on prefill 4, and the baseline-only production arm passes 12/12. gfx1151 automatic dispatch therefore uses baseline; explicit shared-X remains for repair/bisection, and gfx1100 policy is unchanged. Artifacts: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_c14-q4t16-selected-wmma-prototype.json`, `benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf3a-q4t16-shared-x-replay.json`, `benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf3a-full-model-ab.json`, `benchmarks/results/2026-08-05-gfx1151-q4km-shared-x-128k-fallback.json`. |
| P9.C15 Q4T16 selected-dual replay sweep (rejected) | no new default | `scripts/qwen35_gguf_moe_replay.py --q4-tile16-wmma-layers`, launch-bound sweep | Full 512/0 qwen35moe routing replay shows the compact32 Q4T16 WMMA prototype is not enough to unblock #27. Same-run raw baseline: Q4 selected gate+up `62.199 ms`, selected-MoE total `93.138 ms`. All-layer Q4T16 at launch-bound min-blocks 2: Q4 `59.680 ms`; min-blocks 1 (best): Q4 `59.395 ms`, selected-MoE total `92.478 ms`; min-blocks 4: Q4 `59.689 ms`. Best delta is only `-4.5%` on Q4 and `-0.7%` on selected-MoE total, far from the `<=35-40 ms` Q4 continuation target. Rejected as a default/runtime integration; proceed to P9.C16 alternative selected-MoE design. Artifact: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_c15-q4t16-replay-rejected.json`. |
| P9.C16 selected-MoE alternative evaluation (no design selected) | no new code/default | modeled compact tile-list plus measured `64x16`/`64x32` tile proxies | Since P9.C15 missed target, evaluated alternatives before broad runtime changes. Compact tile-list/no-padding model on real 512/0 routing: raw Q4 `62.199 ms`, compact rows `163,840`, WMMA rows `185,872`, padding `13.45%`; optimistic no-padding lower bound `54.775 ms`, still above target. Hot-threshold row distribution shows rows in experts `>=64` are `93.5%` of compact rows, so cold-tail removal cannot close the gap. Measured wider/persistent-column proxies: `64x16` Q4 `61.868 ms` (`-0.5%`), `64x32` Q4 `91.831 ms` (regression). No in-repo selected-MoE alternative is selected for #48; further Q4 selected-MoE work should move to parent kernel R&D or a new design task. Artifact: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_c16-selected-moe-alternatives.json`. |
| P9.C17 final #27 Q4 redesign gate (blocked/no-wire) | no new code/default | carried-forward P9.C11 gate plus P9.C15/P9.C16 rejection evidence | No winning Q4 selected-MoE redesign exists to wire into runtime dispatch. P9.C11 final #27 gate remains the accepted correctness/perf contract: adjacent bundle `143` tests passed, finite deterministic 512/128 token `220`, but combined target bucket is `140.110 ms` vs `<=110 ms` (`Q4 58.126`, `Q5 27.043`, `Q6 2.656`, `Q8 52.285`). P9.C15 Q4T16 improved Q4 only to `59.395 ms`, and P9.C16 no-padding/wider-tile alternatives also missed. #27 remains open/blocked; further Q4 work should happen as parent R&D/new design before hipENGINE wiring. Artifact: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_c17-no-q4-redesign-blocked.json`. |
| P9.H2 GGUF decode replacement layout | future `gguf_q4_k_t16_v1`, `gguf_q5_k_t16_v1`, `gguf_q6_k_t16_v1`, `gguf_q8_0_t16_v1` resident layouts | design only; implementation starts in P9.H3 | `docs/GGUF_DECODE_REPACK.md` | High-priority response to P9.B7: current rows=1 GGUF decode remains `~63 tok/s` because raw-GGUF `prefill_out` buckets survive even with pack8 GEMV active, and unsafe WMMA+GEMV fails P9.E2 (`KL 5.993`, top-1 `5.43%`). The retained design chooses replacement, not sidecar, tile-major `T16` slabs for qwen35moe selected Q4 gate/up, Q5/Q6 down, and Q8 dense/shared projections. Estimated persistent delta is `~+0.457 GiB` if raw covered tensors are replaced, yielding expected tracked peak `~21.34 GiB` for 512/128 under the 24 GiB-class budget; raw+packed expert duplication is explicitly rejected. Acceptance for the implementation: P9.E2 full 512/128x3 passes with `effective_* = true`, 512/128 graph decode median `>=95 tok/s`, rocprof decode trace shows T16 kernels dominate and legacy `prefill_out` absent except documented Q6_K lm-head fallback. Artifact: `benchmarks/results/2026-05-19-hipengine-qwen36-35b-a3b-q4km-p9_h2-decode-repack-design.json`. |
| P9.H3a T16 quant/materializer foundation | `gguf_q4_k_t16_v1`, `gguf_q5_k_t16_v1`, `gguf_q6_k_t16_v1`, `gguf_q8_0_t16_v1` quant keys; `repack_gguf_q{5,6}_k_tile16`, `repack_gguf_q8_0_tile16` | `hipengine/quant/gguf_q4_k.py`, `hipengine/quant/gguf_t16.py`, `tests/test_gguf_t16_repack.py` | First implementation slice for P9.H3. Registers the replacement-layout quant keys through the quant registry and adds bit-lossless CPU materializers/inverses for the missing T16 layouts selected by P9.H2: Q5T16 `[experts,out_tiles16,blocks,2880]`, Q6T16 `[experts,out_tiles16,blocks,3360]`, and Q8T16 `[out_tiles16,blocks,544]`; Q4T16 keeps the existing P9.C13 materializer and gains the `gguf_q4_k_t16_v1` key. Tests cover registry resolution, exact raw-byte roundtrip, dequant equivalence vs GGUF CPU reference, design storage overheads (`Q5 +2.27%`, Q6/Q8 byte-neutral), and shape validation. Validation: `uv run --with pytest pytest tests/test_gguf_q4_k_tile16_repack.py tests/test_gguf_t16_repack.py -q --tb=short` -> `22 passed`. This is not yet a runtime dispatch/perf row; #51 remains open for HIP kernels/materialization. |
| Laguna Q6T16 qmicro gfx1151 production | `repack_gguf_q6_k_tile16_qmicro`, `convert_gguf_q6_k_tile16_to_qmicro`, `unpack_gguf_q6_k_tile16_qmicro`; QMICRO direct/grouped/MMQ consumers; `LAGUNA_Q6_QMICRO=True` | `hipengine/quant/gguf_t16.py`, `hipengine/loading/laguna_gguf_materialize.py`, `quant/gguf_{t16_selected_gemv,q4_k_q8_1_selected_prefill}.{hip,py}`, `runtime/laguna_moe.py`, `scripts/laguna_q6_qmicro_leaf.py` | Byte-neutral Q6 selected-down payload for gfx1151 Laguna sparse expert tensors. The unchanged 288-byte `d/scales` metadata precedes `[K32][col4][K4][QL8,QH4]` 12-byte records; the tile remains 3,360 bytes and raw roundtrip is exact. Existing legacy cache payloads convert once before upload, while root lm-head, gfx1100, and unmeasured backends remain legacy. Direct, grouped-small-M, scalar/rowvec32/rowvec64 MMQ consumers match legacy BF16 bits. Actual layer-1 natural-M512 selected prefill improves **5.1564 -> 5.0714 ms (-1.65%)** and top-10 exact decode **0.0910 -> 0.0846 ms (-6.99%)**; clean pp512 improves **526.451 -> 530.447 tok/s (+0.759%)** and full tracing cuts Q6 **126.594 -> 123.473 ms (-2.465%)** with local128/VGPR88/LDS5,632B/scratch0. Retained as the gfx1151 production default; `q6_qmicro=False` remains the explicit rollback through one later selected-down checkpoint. Artifacts: `benchmarks/results/2026-07-26-gfx1151-laguna-q6-qmicro-{candidate,production}.json`. |
| P9.H3b T16 resident materialization opt-in | `HIPENGINE_GGUF_DECODE_REPACK=1`, `decode_repack=True`; layouts `gguf_q{4,5,6}_k_t16_v1`, `gguf_q8_0_t16_v1` | `hipengine/loading/qwen35_gguf_materialize.py`, `tests/test_qwen35_gguf_materialize.py` | Adds resident planning/materialization plumbing for the P9.H2 replacement layouts without changing defaults. Legacy materialization remains raw/pack8 unless `decode_repack` is explicitly enabled. In repack mode, selected qwen35moe expert tensors materialize only `tiles` allocations with T16 quant keys (Q4 gate/up, Q5/Q6 down) and no expert pack8 sidecar; layer-local Q8_0 rank-2 projections materialize Q8T16 `tiles`; P9.H3e extends the same mode to root `lm_head` as byte-neutral Q6T16. Unit tests cover default legacy plans, env-driven replacement specs, and unchanged local-quant plans. Real W7900 smoke materialized `layers.0.ffn_gate_exps` -> `(256,32,8,2368)` INT8 Q4T16 and `layers.0.ffn_gate_shexp` -> `(32,64,544)` INT8 Q8T16, then freed them. Validation: `uv run --with pytest pytest tests/test_qwen35_gguf_materialize.py tests/test_gguf_t16_repack.py -q --tb=short` -> `26 passed`. Runtime dispatch still lacks T16 kernels; #51 remains open. |
| P9.H3c Q8T16 dense/shared GEMV decode | `gguf_q8_0_t16_gemv_decode_{bf16,fp16}_{bf16,fp16}_out`, `gguf_q8_0_t16_dual_gate_up_gemv_decode_{bf16,fp16}_{bf16,fp16}_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_gemv.{hip,py}`, `hipengine/runtime/gguf_linear.py`, `tests/test_gguf_q8_0_t16_gemv_decode.py` | First HIP kernel consumer for the P9.H2 replacement layout. Consumes Q8T16 `tiles[out_tiles16,blocks_per_row,544]`: 16 fp16 scales plus `[32 K lanes,16 cols]` int8 payload per tile. One 128-thread block computes one `(row,out_tile16)` with FP32 accumulation and BF16/FP16 output; dual variant emits concatenated gate/up for shared-expert gate/up. Runtime single-dispatch supports `LAYOUT_GGUF_Q8_0_T16` via ABI `t16`; P9.H3d wires pair-concat routing to the dual T16 kernel. Correctness vs CPU `gguf_quant_gemv(..., Q8_0)` passes for BF16/FP16 single and dual synthetic shapes (`13` Q8T16 tests; adjacent linear-dispatch bundle total `47 passed`). `rocprofv3 --kernel-trace -f csv` tiny smoke shows `q8_0_t16_gemv_kernel<unsigned short,unsigned short>` launched with `End-Start=8481 ns` in `/tmp/p9_h3c_q8t16_rocprof_csv/rocm/1159928_kernel_trace.csv`. This is not a performance acceptance; #51 remains open for Q4/Q5/Q6 selected T16 kernels and E2E gates. |
| SH5-D1 gfx1151 raw Q8_0 rowvec8 pair decode leaf (runtime rejected) | `linear_pair/gguf_q8_0/rowvec8_dual_split_gemv_decode_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_pack8_gemv.{hip,py}`, `tests/test_gguf_q8_0_pack8_gemv_decode.py` | Fork-attributed raw-row output-major c1 leaf: one local64 workgroup owns one output and each lane consumes eight adjacent K values. Actual layer-0 `2048->8192 + 2048->4096` bytes with a >2x-MALL pool improve Q8T16 **0.134737 -> 0.116588 ms (1.15566x, 15/15)**; cached trace is local64, 24 VGPR, 512 B LDS, scratch0. The temporary byte-neutral model route improved decode **2.934%** but lost **13.457%** prefill and changed state. SH6-P1's exact bridge still lost **3.720%** prefill, so all model-route, materializer, dispatcher, env, scratch, and runtime-bridge surfaces are removed. Retain only the standalone leaf and source evidence; production remains Q8T16. Artifacts: `benchmarks/results/2026-08-06-gfx1151-gguf-sh5-d1-raw-rowvec8-blocked.json` and `2026-08-06-gfx1151-gguf-sh6-p1-raw-to-t16-prefill-bridge-rejected.json`. |
| SH6-P1 raw Q8_0 pair -> Q8T16 GPU bridge leaf (runtime rejected) | `layout_transform/gguf_q8_0/raw_pair_to_t16` | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_raw_to_t16.{hip,py}`, `tests/test_gguf_q8_0_raw_to_t16.py` | Byte-only combined-grid transform for the linear-attention `2048->8192 + 2048->4096` pair. It writes host-packer-identical T16 bytes for both production tensors into one exactly **26,738,688-byte** owner. Production-shape RED/GREEN passes at local64/128/256. A three-pair **80,216,064-byte** cycling screen selects local64 at **0.360914 ms/pair** versus 0.401951/0.502646 ms; cached gfx1151 trace names local64 at **40 VGPR, 128 SGPR, scratch0**. The lifecycle-correct 30-pair runtime bridge preserves complete 512 prefill state, but charged prefill regresses **1369.120 -> 1318.196 tok/s (-3.720%)**, failing the frozen 1% gate. Runtime wiring is removed; the exact transform remains only as a tested diagnostic leaf. Artifact: `benchmarks/results/2026-08-06-gfx1151-gguf-sh6-p1-raw-to-t16-prefill-bridge-rejected.json`. |
| P9.D6 Q8T16 split-output pair decode | `gguf_q8_0_t16_dual_gemv_decode_{bf16,fp16}_{bf16,fp16}_out`; diagnostics `gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out`, `gguf_q8_0_t16_dual_gemv_decode_rowtile{2,4}_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_gemv.{hip,py}`, `hipengine/runtime/gguf_linear.py`, `hipengine/runtime/qwen35_gguf_runner.py`, `tests/test_gguf_q8_0_t16_gemv_decode.py`, `tests/test_gguf_linear_dispatch.py` | Adds a split-output companion to the H3c concatenated Q8T16 dual kernel so same-input projections can share one launch without changing scratch layout. qwen35moe decode routes `attn_k+attn_v` and `ssm_alpha+ssm_beta` through this path. Synthetic BF16/FP16 CPU-oracle fixtures pass; unit rocprof smoke sees `q8_0_t16_dual_split_gemv_kernel<unsigned short,unsigned short>` with `DurationNs=13999`, `VGPR=56`, `SGPR=128`. 2026-07-01 adds a diagnostic `HIPENGINE_GGUF_Q8_T16_THREADS` launch-width override for Q8T16 single/pair/triple GEMV wrappers; default stays 128 threads. The llama-compat verifier pair shape rejected 64 threads (`rows=2/3/4`: `197.77/224.80/251.96 us` vs 128-thread `179.26/207.05/237.02 us`), and rocprof confirmed `Workgroup_Size_X=64` on the override path. The same parity pass added a callable q8_1/dp4a T16 dual-split diagnostic that matches a q8_1 CPU oracle plus KL/top-1 gate and rocprof-confirmed `q8_0_t16_dual_split_q8_1_dp4a_kernel<unsigned short>` (`Workgroup_Size_X=128`, `Grid_Size_X=1536`, `Grid_Size_Y=8`), but the qwen35 pair microbench rejects it: 128-thread exact rows 2/3/4 are `181.50/207.98/236.26 us`, while quantize+dp4a is `304.78/448.32/558.14 us` and prequantized dp4a is `303.05/452.51/566.29 us`. This proves q8_1/dp4a over the current T16 layout is not the llama.cpp win; the four adjacent K bytes for one output column are strided by 16 and must be packed before dot4. A later exact row-amortized rowtile2/rowtile4 diagnostic is bit-identical to the exact pair, and the qwen35 pair microbench likes rowtile4 at 64 threads (`rows=2/3/4/5/6`: exact 128 `179.75/207.70/236.41/265.87/298.97 us` vs rowtile4-64 `154.05/170.55/191.16/254.19/271.06 us`), with cached rocprof confirming the rowtile4 kernel launched at `Workgroup_Size_X=64`; however full-suite `llama-compat-device-chain-dp4a-q6top1dp4a` rejected runtime promotion (`59.63 -> 57.25 tok/s`, `target_block_verify_total` `13.178 -> 13.697 ms/output`). It remains default-off for the verifier under `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=1`; F3B later scopes the repaired 128-thread body to gfx1151 physical C8 packed AR. P9.E2 passes (`KL 0`, top-1 `100%`); 512/128 graph decode improves D4 `86.025 -> 86.502 tok/s` but remains below `95`. Artifacts: `benchmarks/results/2026-05-20-hipengine-qwen36-35b-a3b-q4km-p9_d6-q8t16-pair-dispatch.json`, `benchmarks/results/2026-07-01-q8-t16-pair-threads-micro.json`, `benchmarks/results/2026-07-01-q8-t16-pair-q8-1-dp4a-micro.json`, `benchmarks/results/2026-07-01-q8-t16-pair-rowtile-micro.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-rowtilepair-full.json`. |
| 2026-07-12 Q8T16 wave/block indexing promotion | production `gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_gemv.{hip,py}`, `tests/test_gguf_q8_0_t16_gemv_decode.py` | The BF16 split-output production body now traverses K explicitly as `(wave, block_idx, lane)`, preserving every thread's K sequence and FP32 accumulation order. The compile-time A/B at `8184355c` proves 64/128-thread bit identity and measures **136.415 -> 132.175 us** (**-3.108%**) on rows=1 `2048x(8192+4096)` with a >2x-MALL cycling pool. Static resources move `33 SGPR/49 VGPR -> 29/50`, retaining 16-wave occupancy and zero spills. Clean `8184355c -> e20cdc13` p512/d128 eager moves **20.5342 -> 20.4709 ms/token** (**-0.308%**); 24 marked steps move the actual leaf **4245.4 -> 4188.2 us/token** (**-1.349%**) and total GPU time **-0.296%**. State-bound graph wall also improves **-0.200%** across commits, with exact 128/128 replay. The temporary callable A/B wrapper was removed after promotion; reproduce the scalar/candidate micro at diagnostic commit `8184355c`. Artifacts: `benchmarks/results/2026-07-12-gfx1151-q8-t16-waveblock-{micro,production}.json`. |
| F3 gfx1151 packed-AR Q8T16 64-thread row amortization (rejected) | no broad backend default; explicit diagnostic `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_gemv.{hip,py}`, `hipengine/runtime/gguf_linear.py`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `tests/test_gguf_{linear_dispatch,q8_0_t16_gemv_decode}.py`, `tests/test_gfx1151_backend.py` | The clean p512/d64 screen looked exact and improved c4/c8 **+1.22%/+2.72%**, but p512/d128 changed one prompt at c2/c4/c8. A later first-transition all-layer oracle identifies the actual error: the old hardcoded 64-thread rowtile changes production reduction partition, producing one-BF16-ULP model-hidden drift first at layers 13/4 for the two c2 rows. The broad route stays rejected even after repairing thread geometry because exact 128-thread all-projection C2/C4/C8 is **77.940/107.798/133.377 tok/s** versus retained **78.552/108.050/133.251**. |
| F3B gfx1151 physical-C8 Q8T16 pair row amortization | `GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS=8`; `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` rollback | `hipengine/runtime/gguf_linear.py`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `tests/test_gguf_{linear_dispatch,q8_0_t16_gemv_decode}.py`, `tests/test_gfx1151_backend.py` | Restores the production 128-thread reduction partition while sharing qkv+gate Q8T16 weights across four rows. The c2/c8 first-step model-hidden oracles are **80/80** and **320/320** exact with token/Conv/GDN/KV/final state exact. The >2x-MALL pair leaf improves C2/C4/C8 **185.53/232.02/340.12 -> 161.94/202.15/323.99 us**; model-scale lower widths are rejected, so only physical C8 selects it on gfx1151 and gfx1100 remains disabled. Clean p512/d128 direct C8 improves **133.251 -> 133.852 tok/s (+0.452%)**, with three exact/repeatable samples and 0.033% stdev/median. Cached `rocprofv3` confirms 30 in-marker pair-rowtile launches at 128 threads, 136 VGPR, 1 KiB LDS, and zero scratch within the expected 748 packed-native/zero-copy route. The matched C1/C8 server packet is exact but mixed within noise (**-0.40% blocking, +1.63% SSE, -0.64% delayed**), so no complete-server speedup is claimed. Artifact: `benchmarks/results/2026-07-20-gfx1151-gguf-q8t16-pair-rowtile-c8-retained.json`. |
| F3G gfx1151 Q8T16 F32-input row amortization (rejected and removed) | no runtime path, flag, or wrapper retained | temporary exact rowtile2/rowtile4 F32-input instantiations at the `ssm_out` shape; `benchmarks/results/2026-07-20-gfx1151-gguf-q8t16-f32-rowtile-rejected.json` | The post-F3E profile identifies 30 `q8_0_t16_gemv_kernel<float,bf16>` launches (**3.384 ms / 6.50%** of total profile). At `rows=8, 4096->2048`, production **111.265 us** beats byte-exact rowtile2 **111.700 us (+0.39%)** and rowtile4 **130.397 us (+17.19%)**. Hardware cache reuse plus the wider production grid already wins; do not add more cross-row F32 weight reuse without a materially different scheduler. |
| F3H gfx1151 physical-C8 Q8T16 pair rowtile2/3 scheduling (rejected) | rowtile4 remains automatic; no new selector retained | existing rowtile2 plus temporary rowtile3 instantiation; same-checkout direct/server A/B; `benchmarks/results/2026-07-20-gfx1151-gguf-q8t16-pair-rowtile-scheduling-rejected.json` | Rowtile2 improves the >2x-MALL qkv+gate leaf **328.91 -> 318.60 us (-3.13%)** and exact direct C8 **151.015 -> 152.062 tok/s (+0.69%)**, but same-checkout server delayed admission regresses **68.590 -> 67.870 (-1.05%)** despite **+0.32%/+0.36%** blocking/SSE. Rowtile3 is exact and **0.79%** faster at the leaf but regresses direct **-0.10%**. Remove rowtile3/selectors and keep rowtile4; do not promote a direct-only scheduling gain that fails the matched delayed workload. |
| F3C gfx1151 physical-C8 Q4T16 selected expert-pair reuse | `selected_dual_t16_pairreuse_gemv_decode_bf16_bf16_out`; `GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS=8`; `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE=0` rollback | `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `scripts/gguf_q4_k_t16_selected_dual_dp4a_microbench.py`, `tests/test_{gguf_t16_selected_gemv_decode,gfx1151_backend,qwen35_gguf_compact_moe_gemv_routing}.py` | Two wave32 ballots find repeated dynamic expert IDs across at most 64 selected lanes. Consecutive occurrences share each Q4T16 gate/up weight tile while every row keeps the production 128-thread K partition, reduction, and BF16 output order; unique IDs take the old dual body inside the same launch. At Qwen C8 shape, paired IDs improve **363.114 -> 249.404 us (-31.32%)**, while unique/random are bounded at **+0.19%/+0.44%** and all outputs are byte-exact. The no-env combined-default model oracle is **320/320** layer outputs exact with exact token/Conv/GDN/KV/final state. Clean retained p512/d128 C8 is **133.852 -> 144.039 tok/s (+7.61%)** with 0.026% variance and all trajectories exact. A source-equivalent real-Uvicorn diagnostic improves blocking **86.185 -> 87.770 (+1.84%)**, exact SSE **84.196 -> 84.798 (+0.72%)**, and delayed **67.788 -> 68.242 tok/s (+0.67%)**, with every request exact; server speed remains diagnostic pending clean repetition. Cached `rocprofv3` confirms 128 threads, 200 VGPR, 1,032 B LDS, zero scratch. gfx1100 remains disabled pending independent W7900 transfer. |
| F3D gfx1151 Q4T16 byte-identical selected-input reuse (rejected and removed) | no runtime path retained; microbench fixture mode `--selection-pattern paired_identical` remains | temporary extension of `q4_k_t16_selected_dual_pairreuse_direct_gemv_kernel`; `scripts/gguf_q4_k_t16_selected_dual_dp4a_microbench.py`; strengthened `tests/test_gguf_t16_selected_gemv_decode.py` | The candidate cooperatively compared paired 2,048-element BF16 inputs and, when byte-identical, executed one production-order gate/up body for both lanes. It was byte-exact and improved the identical-input leaf **370.045 -> 216.955 us (-41.37%)** plus one-run direct C8 **144.039 -> 145.502 tok/s (+1.02%)**, but comparison overhead regressed the matched distinct-request server packet versus F3C by **-0.41% blocking / -1.39% exact SSE / -1.31% delayed** and put SSE/delayed **-0.69%/-0.65%** below the clean retained server row. The shortcut was removed; general dynamic expert-ID weight reuse remains. Artifact: `benchmarks/results/2026-07-20-gfx1151-gguf-selected-identical-input-reuse-rejected.json`. |
| F3E gfx1151 physical-C8 Q5T16 selected-down expert-pair reuse | `selected_t16_pairreuse_gemv_decode_bf16_bf16_out`; `GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS=8`; `HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE=0` rollback | `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `scripts/gguf_q5_k_t16_selected_down_dp4a_microbench.py`, `tests/test_{gguf_t16_selected_gemv_decode,gfx1151_backend,qwen35_gguf_compact_moe_gemv_routing}.py` | Extends dynamic expert-ID pairing to Q5T16 selected down with two independent 16-column accumulators, preserving each row's production 128-thread K/reduction/BF16 order. Real-shape unique/random/paired micros are **+5.37%/+0.35%/-28.19%**, all byte-exact. The automatic combined-default oracle is **320/320** layer outputs exact. Clean retained direct C8 improves **144.039 -> 150.756 tok/s (+4.66%)**, and **133.852 -> 150.756 (+12.63%)** versus the pre-selected-reuse row, with 0.111% variance; a matched source-equivalent distinct-request server packet is exact and moves versus F3C by **+0.80% blocking / -0.12% SSE noise / +0.11% delayed**. Automatic tracing records the expected **37** Q5 launches at 128 threads, 96 VGPR, 520 B LDS, and zero scratch. gfx1100 remains disabled pending W7900 transfer. |
| SH-D1 gfx1151 Qwen c1 Q5T16 selected-down tile8 | `selected_t16_qwen_tile8_gemv_decode_bf16_bf16_out`; `GGUF_Q5_T16_SELECTED_QWEN_TILE8=True`; direct 16-column peer/shape fallback | `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `scripts/gguf_q5_k_t16_selected_down_tile8_microbench.py`, selected-T16/routing/backend tests | Exact K512/N2048/top8 specialization splits each resident Q5T16 tile into two eight-column owners while preserving every output column's production local128 K/FMA order, wave32 tree, serial wave-0..3 reduction, and BF16 store. The >2x-MALL final leaf improves **40.815 -> 34.865 us (1.1707x)** and projects **0.2202 ms/token** over 37 layers; tracing records local128/grid256x8, **56 VGPR**, 512 B LDS, and scratch0 versus production 128 VGPR. Final-code eager 512/128 improves **52.881 -> 53.413 tok/s (+1.007%)** with all samples separated, exact IDs, zero close bytes, byte-exact complete state at 512/4K/32K/64K, and all 18 natural/heldout prompts x3 exact. Q6 tile8 reaches only **1.0803x** and tile4 regresses to **0.9325x**; both Q6 surfaces are removed. gfx1100 and all shape/quant misses retain production. Artifact: `benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-selected-down-q5-tile8-retained.json`. |
| SH-D1 Qwen c1 Q6T16 LM-head tile8 (rejected and removed) | no runtime path, wrapper, registry key, test, or microbench retained | transient exact `q6_k_t16_gemv_tile8_kernel` in `gguf_q6_k_t16_gemv.{hip,py}`; full real-weight screen and focused Q6 test | Split the K2048/N248320 producer's 16 FP32 logits into two local128 eight-column owners while preserving every logit's K/FMA/reduction/store order. Full logits/top-1 are exact and cached tracing records **48 VGPR / 512 B LDS / scratch0** versus production **72 / 512 / 0**, but the counterbalanced complete 417,177,600-byte matrix screen regresses **1.83174 -> 1.83575 ms (0.99782x, -0.218%)**. Remove the candidate and close SH-D1 row-1 weight ownership; lower accumulator pressure cannot repay doubled workgroups. Artifact: `benchmarks/results/2026-08-06-gfx1151-gguf-sh-d1-q6-lm-head-tile8-rejected.json`. |
| F3F gfx1151 physical-C8 Q6T16 selected-down expert-pair reuse | `gguf_q6_k_t16_v1/selected_t16_pairreuse_gemv_decode_bf16_bf16_out`; `GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS=8`; `HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE=0` rollback | `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `scripts/gguf_q6_k_t16_selected_down_pairreuse_microbench.py`, selected-T16/routing/backend tests | Closes the three-layer Q6 down tail with the same exact 128-thread dynamic expert pairing. Unique/random/paired real-shape micros are **+6.81%/+4.28%/-23.59%**, all byte-exact. The automatic combined-default oracle is **320/320 exact**. Clean retained direct C8 improves **150.756 -> 151.015 tok/s (+0.171%)**, and **133.852 -> 151.015 (+12.82%)** versus the pre-selected-reuse row, with 0.093% variance; a matched source-equivalent distinct-request server packet moves **-0.15% blocking noise / +0.48% SSE / +0.81% delayed**, all rows exact. Automatic tracing records the expected **3** launches at 96 VGPR, 520 B LDS, and zero scratch. gfx1100 remains disabled pending W7900 transfer. |
| Laguna LPF-2 compact expert-pair selected prefill (runtime rejected; primitives retained) | registered primitives only: `selected_dual_t16_pairreuse_gemv_decode_compact_bf16_bf16_out`; Q4/Q5/Q6 `selected_t16_pairreuse_gemv_decode_compact_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `tests/test_gguf_t16_selected_gemv_decode.py` | Exact small-M alternative to 16-row compact WMMA. The primitives pair adjacent rows inside exact expert starts while preserving each row's 128-thread K partition, four-wave reduction, and BF16 bits; irregular 132-lane Q4 dual and Q4/Q5/Q6 down fixtures remain byte-exact beyond the old 64-lane mask. Cached gfx1151 trace records Q4 dual 238.241 us / 128 VGPR, Q4 down 198.330 us / 128 VGPR, and Q6 down 204.944 us / 112 VGPR, each at 128 threads, 128 SGPR, 512 B LDS, and zero scratch. The balanced full-model route was rejected: compact-pair regressed every measured row 16..128 by **-17.07% to -10.21%**, with a weighted **0.8843x / -11.57%** result despite exact next tokens and lifecycle recovery. The selector, compact scratch, group library, benchmark harness, and runtime route were removed; direct selected GEMV remains the only Laguna route. Artifact: `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-lpf2-compact-pair-rejected.json`. |
| Laguna AR-O1 exact grouped-small-M Q4/Q6 selected down | `selected_t16_grouped_smallm_bf16_bf16_out`; gfx1151 `LAGUNA_SELECTED_DOWN_MODE=adaptive_grouped_smallm`; direct fallback below 32 rows and on unmeasured backends | `hipengine/kernels/hip_gfx1100/{moe/group_scatter,quant/gguf_t16_selected_gemv}.{hip,py}`, `hipengine/runtime/laguna_{moe,gguf_runner}.py`, `hipengine/kernels/hip_gfx1151/__init__.py`, grouped-down tests/harnesses | One deterministic device pass emits compact starts, active experts, stable lane order, and F32 routing weights without scalar D2H. C16xR4 grouped Q4/Q6 down shares each decoded T16 tile across up to four BF16 rows while preserving every row's direct K/reduction association and BF16 bits; staged count/prefix/scatter and direct selected GEMV remain unfused fallbacks. Production fixtures and the complete MoE chain are bit-exact; trace records compact metadata `10.059 us`, Q4 down `192.481 us`, Q6 down `104.636 us`, 128 threads, 1,024 B LDS, zero scratch. Clean rows 32..128 improve **2.63-6.92%**, aggregate shape wall **5.461%**, and the ten-prompt h16/h32 category gate improves weighted prefill **50.193 -> 53.178 tok/s (+5.948%)** plus E2E **3.835%/2.762%**, with `KL=0`, 320/320 teacher top-1, exact trajectories/oracle/lifecycle. Artifacts: `benchmarks/results/2026-07-23-gfx1151-laguna-prefill-grouped-down-{ab,category}.json`. |
| F3I gfx1151 physical-C8 Q8T16 pair rowtile4/col8 | `t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out`; existing `GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS=8`; `HIPENGINE_GGUF_Q8_T16_PAIR_COL8=0` restores col16 and `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` restores per-row | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_gemv.{hip,py}`, `hipengine/runtime/gguf_linear.py`, pair microbench and Q8T16/dispatch tests | Keeps rowtile4's two row groups but splits each T16 tile into two eight-column blocks. Per-thread accumulators halve from 64 to 32; tracing confirms **72 VGPR / 512 B LDS / zero scratch** versus prior **136 / 1,024 / zero**. The >2x-MALL qkv+gate leaf improves **327.91 -> 318.14 us (-2.98%)**, byte-exact. Clean retained direct C8 is **151.015 -> 152.192 tok/s (+0.779%)**, and **133.852 -> 152.192 (+13.70%)** versus the pre-selected-reuse row, with 0.069% variance and **320/320 exact** state. Same-checkout server blocking/SSE/delayed all improve **+0.38%/+0.54%/+0.46%**. gfx1100 and lower widths remain unchanged. |
| F3J gfx1151 Q8T16 single/dual/triple col8 (rejected and removed) | no runtime path, wrapper, or new env retained; F3I pair-rowtile col8 unchanged | temporary exact col8 instantiations for BF16/F32 singles, shared gate/up dual, and full-attention triple; `benchmarks/results/2026-07-20-gfx1151-gguf-q8t16-single-dual-triple-col8-rejected.json` | Long-K BF16/F32 single regresses **+16.22%/+13.12%** and full-attention triple regresses **+3.56%**. Short-K single improves **-5.68%** and direct **+0.093%**, but delayed server regresses **-0.53%**. Shared dual improves **-1.83%** at the leaf, but direct moves only **+0.019%** under **0.089%** variance. All outputs are byte-exact; remove all candidates. Dense C8 row/column scheduling is exhausted beyond F3I. |
| F3K gfx1151 physical-C8 Q6T16 lm-head 5+3 rowtile partition | `GGUF_Q6_LM_HEAD_MAX_CHUNK=5`; `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK=6` rollback | `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, chunk/backend tests | Repartitions the existing exact rowtile kernels from 6+2 to 5+3; no math/kernel body changes. Real 248,320-vocab isolated wall improves **4.865 -> 4.815 ms (-1.02%)**. Clean direct C8 is **152.192 -> 152.709 tok/s (+0.340%)** at 0.027% variance, with **320/320 exact** state. Same-checkout server blocking/SSE/delayed move **+0.15%/+0.65%/+0.02%**, all exact. Automatic trace records rowtile5/3 at 168/104 VGPR, 1,280/768 B LDS, zero scratch. A one-launch rowtile8/col8 kernel was exact and **-3.52%** isolated but regressed same-checkout direct **-0.26%**, so it was removed. gfx1100 stays 6+2. Artifact: `benchmarks/results/2026-07-20-gfx1151-gguf-q6t16-lm-head-chunk5-c8-retained.json`. |
| F3L gfx1151 physical-C8 indexed GDN paired value heads (rejected and removed) | no runtime path, wrapper, capability, or env retained | temporary 256-thread paired-head sibling of `qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_kernel`; indexed GDN exact fixture; `benchmarks/results/2026-07-20-gfx1151-gguf-gdn-pairheads-c8-rejected.json` | Pairs the two interleaved value heads sharing each Q/K head while preserving each head's 128-thread state/RMSNorm order. The >2x-MALL leaf improves **204.689 -> 202.631 us (-1.01%)** and direct C8 improves **152.709 -> 152.839 tok/s (+0.086%)**, with **320/320 exact** hidden/state. Mandatory same-checkout server blocking/SSE/delayed all regress **-0.12%/-0.66%/-0.16%**. Remove the candidate; Q/K setup is too small a share of the state-memory kernel to survive serving interference. |
| F3M gfx1151 physical-C8 grouped-query paged attention (rejected and removed) | no runtime path, wrapper, capability, env, or fixture extension retained | temporary 512-thread two-query and 1024-thread four-query grouped siblings of `qwen35_paged_full_attn_decode_context_tensor_batch_fixed256_kernel`; `benchmarks/results/2026-07-20-gfx1151-gguf-paged-attn-qhead-groups-c8-rejected.json` | Independent production-order 256-thread query groups share one block/cache footprint without changing each query's reductions. The best two-query leaf improves **314.999 -> 293.151 us (-6.94%)**, direct C8 improves **152.709 -> 153.341 tok/s (+0.414%)**, and hidden/state is **320/320 exact**, but same-checkout server blocking/delayed regress **-0.57%/-0.94%** (SSE **+0.40%**). Four-query grouping is exact but only **-5.55%** at the leaf. Explicit exact K/V-load sharing reached **-28.4%** at the leaf but regressed the model screen **-4.95%**. Remove all candidates: this family trades away serving/graph scheduling for isolated cache reuse. |
| F3N gfx1151 physical-C8 GDN state residency and Conv fusion (rejected and removed) | no runtime path, wrapper, capability, env, or fixture extension retained | temporary exact state-cache and fused indexed-Conv+paired-GDN kernels; `benchmarks/results/2026-07-20-gfx1151-gguf-gdn-state-residency-fusion-c8-rejected.json` | Caching all 128 state scalars per lane is exact but spills (**96 VGPR / 1,264 B scratch** vs production **56 / 0**) and regresses **204.831 -> 649.368 us (+217%)**; 64/32-scalar arrays also spill and regress **+221%/+232%**. A manually scalarized 8-value cache avoids the large regression but is inexact and only **-0.84%** at the leaf. Exact indexed Conv+paired-GDN fusion removes one launch and the FP32 `conv_out` round trip, improving the combined leaf **229.702 -> 224.120 us (-2.43%)**, but direct C8 regresses **152.709 -> 152.238 tok/s (-0.308%)** with identical trajectories. Remove all candidates; the current coalesced state layout and separate 128-thread schedules remain. |
| F3O gfx1151 physical-C8 paged-attention shared token offsets | `bf16_context_batch_fixed256_spans` instantiates `qwen35_paged_full_attn_decode_context_tensor_batch_kernel<true>`; no env flag | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`, existing paged-attention fixtures; `benchmarks/results/2026-07-20-gfx1151-gguf-paged-attn-token-offsets-c8-retained.json` | Precomputes each token's physical KV element offset once per query-head block in aligned dynamic shared memory and reuses it across key/value passes; no FP order changes and persistent memory is unchanged. The real C8 leaf improves **315.073 -> 181.251 us (-42.47%)**, all 655,360 compared bytes match, and clean direct p512/d128 improves **152.709 -> 158.048 tok/s (+3.496%)** at 0.092% variance with **320/320 exact** state. Same-source server blocking/SSE/delayed improve **+2.37%/+0.06%/+0.79%**. Trace: 256 threads, 40 VGPR, zero scratch, median 174.687 us. Generic adaptive/gfx1100 and long-context split-K routes are unchanged. |
| F3P gfx1151 physical-C8 paged-attention value vector 2 | `bf16_context_batch_fixed256_spans` instantiates `qwen35_paged_full_attn_decode_context_tensor_batch_kernel<true,2>`; odd head dimensions fall back to vector 1; no env flag | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`, existing paged-attention fixtures; `benchmarks/results/2026-07-20-gfx1151-gguf-paged-attn-value-vector2-c8-retained.json` | On top of F3O shared offsets, each active value thread accumulates two adjacent dimensions while each dimension retains the same 513-token FP32 order. The leaf improves **181.324 -> 134.890 us (-25.61%)**, vector 4 is weaker at **143.670 us**, and all outputs are byte-exact. Clean direct C8 improves **158.048 -> 158.804 tok/s (+0.478%)**, **+3.992%** versus F3K, with **320/320 exact** state. Server blocking/SSE/delayed improve versus F3O **+0.87%/+1.86%/+0.36%**. Trace remains 256 threads, 40 VGPR, zero scratch, median 128.441 us. Generic adaptive/gfx1100 and long-context routes keep vector 1. |
| F3Q gfx1151 physical-C8 indexed-GDN shared state cache 24 | gfx1151 overrides `gdn_recurrent_rmsnorm_gate/gguf_qwen35/bf16_indexed_singleton` with `qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16` only for physical rows >=8; lower widths and gfx1100 retain the generic indexed wrapper | `hipengine/kernels/hip_gfx1100/linear_attn/gdn.{hip,py}`, gfx1151 registry override, indexed Conv/GDN CPU-reference fixture; `benchmarks/results/2026-07-20-gfx1151-gguf-gdn-shared-statecache24-c8-retained.json` | Stores 24/128 old FP32 state rows in LDS during the first K/state pass and reuses them after the dependency barrier, preserving every recurrence/update operation. The exhaustive 0..120-row sweep selects the four-block/CU 15,360 B point: leaf **204.996 -> 197.844 us (-3.49%)**; cache-120 is equally fast locally but regresses serving from one-block occupancy. Clean direct C8 improves **158.804 -> 159.487 tok/s (+0.430%)**, with **320/320 exact** state. Final trace: GDN **3.770 -> 3.468 ms (-8.02%)**, 128 threads, 56 VGPR, 15,360 B LDS, zero scratch. Full server SSE/delayed improve **+0.94%/+0.91%**; blocking is flat within matched-run variance (**-0.18%**, 0.47–0.79% run spread). |
| F3R gfx1151 physical-C8 paged-attention softmax/value schedules (rejected and removed) | no runtime path, template parameter, wrapper, env, or fixture extension retained | temporary parallel-exp/normalization and packed-value2 siblings of F3P; `benchmarks/results/2026-07-20-gfx1151-gguf-paged-attn-softmax-value-schedules-c8-rejected.json` | Parallelizing independent exp/normalized-score writes while retaining the original eight lane-0 sum order is byte-exact and improves the leaf **134.564 -> 128.225 us (-4.71%)**; explicit packed V loads regress **+0.62%**. Direct C8 is only **+0.090%** inside variance. Serving improves blocking **+0.21%** but regresses exact SSE **-0.86%** and delayed **-0.13%**, so remove all candidates. This closes grouped-query, launch-width, packed-load, address-hoist, value-vector, shared-offset, and softmax scheduling for the short-context exact paged-attention family. |
| GPF-5A Q8T16 two-wave WMMA prefill scoped default | `wmma_prefill_2wave_bf16_bf16_out` plus gfx1151 automatic wrapper | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `tests/test_gguf_q8_0_t16_wmma_prefill.py` | One 64-thread block pairs two independent production-order 32-column waves and uses 1 KiB LDS to share one BF16-to-FP16 activation tile across 64 output columns. Tail-row/output fixtures are byte-exact to production `32x32`; profiler smoke reports 80 VGPR, 128 SGPR, zero scratch, and 1024 B LDS. Real 2048x8192 micros improve 16.17%/16.39% at 1K/4K rows. Clean full-model gates promote it on gfx1151: 512 **+8.35%**, stable 4K **+2.54%**, 82/82 state parts exact, memory unchanged. Final right-sized automatic 512-64K publishes **889.904/919.598/762.940/648.948/546.296 tok/s**, **+1.01% to +8.57%** over the prior row. Same-commit 128K rejects two-wave **382.041 vs 392.219 tok/s (-2.59%)**, so request-scoped backend metadata enables only through 65,536 prompt tokens and restores production above it; gfx1100 remains production. Explicit env `0|1` overrides for rollback/diagnosis. Artifacts: `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf5a-{candidate-focus,clean-promotion,128k-scope,right-sized-3run}.json`. |
| LCP-3 Q8T16 four-wave WMMA prefill scoped default | `wmma_prefill_4wave_bf16_bf16_out` plus gfx1151 automatic wrapper; `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE=0` rolls back to two-wave | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `tests/test_gguf_q8_0_t16_wmma_prefill.py` | One 128-thread block keeps four independent exact 32-column waves and shares one 1 KiB activation tile across 128 output columns. Tail fixtures and clean detached full-model 512/4K capture are 83/83 exact. Dominant 4K `2048x8192` and `8192x2048` micros improve **7.50%/14.08%** over GPF-5A. Clean five-pair full-model prefill improves **1214.510 -> 1220.993 (+0.53%)** at 512 and **1269.030 -> 1288.986 tok/s (+1.57%)** at 4K; all 20 timed IDs are exact. Trace: 128 threads, 80 VGPR, 1 KiB LDS, zero scratch. gfx1151 selects four-wave through the inherited 65,536-token ceiling and production above it; gfx1100 remains production. `benchmarks/results/2026-07-15-gfx1151-gguf-q8-t16-four-wave-clean-promotion.json`. |
| LCP-4A F32-weight router 256-thread gfx11 defaults | callable `qwen35_router_logits_bf16_f32w_auto_256`, registered for gfx1100 and retained as the gfx1151 alias override | `hipengine/kernels/hip_gfx1100/moe/router.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `tests/test_qwen35_router_plan.py`, `tests/test_gfx1151_backend.py` | The token-tiled HIP body previously launched with 512 threads even though `hidden_size=2048` gives useful eight-element work to only 256 lanes; its first reduction step added zero partials. On gfx1151, removing that idle half keeps bytes exact, improves the isolated 512/1024-token `2048x256` router **0.683 -> 0.380 ms (-44.32%)** and **1.354 -> 0.756 ms (-44.17%)**, and improves clean full-model 512/4K prefill **+2.76%/+3.28%** with 83/83 exact state. The independent W7900 transfer compares 1,048,576 primitive outputs bit-exact and improves balanced full-model prefill **2689.171 -> 2795.242 (+3.94%)** at 512 and **2955.867 -> 3070.905 tok/s (+3.89%)** at 4K; graph decode is **-0.022%/+0.159%**, memory unchanged, and all IDs exact. The 256-thread wrapper is now the gfx1100 registry default as well. Evidence: `benchmarks/results/2026-07-15-gfx1151-gguf-router-threads256-clean-promotion.json`, `benchmarks/results/2026-07-16-gfx1100-gguf-router-threads256-promotion.json`. |
| LCP-4B prefill router-select 128-thread gfx11 defaults | existing `qwen35_router_select_kernel`; gfx1100/gfx1151 capability `GGUF_PREFILL_ROUTER_SELECT_THREADS=128`; rollback `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS=512` | `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `tests/test_qwen35_gguf_router_select_policy.py`, `tests/test_qwen35_router_plan.py` | gfx1151 profiling leaves router select at **12.539 ms / 130 (0.41%)** and shows 128 threads reduce the family to **3.741 ms (-70.17%)**, with 24 VGPR, 128 SGPR, 512 B LDS, and zero scratch. Its clean 512/4K state is 83/83 exact and prefill improves **+0.34%/+0.36%**. The independent W7900 transfer compares 32,768 selected IDs and 32,768 routing weights bit-exact. On top of retained 256-thread logits, aggregate 512/4K median prefill improves **2789.516 -> 2798.564 (+0.32%)** and **3055.119 -> 3079.801 tok/s (+0.81%)**; paired medians are **+0.30%/+0.12%**, graph decode is **-0.068%/+0.216%**, memory unchanged, and all IDs exact. Both backends now select 128 threads for bulk prefill; decode retains its independent launch. The faster 64-thread gfx1151 primitive remains rejected for state divergence. Evidence: `benchmarks/results/2026-07-15-gfx1151-gguf-prefill-router-select-threads128-promotion.json`, `benchmarks/results/2026-07-16-gfx1100-gguf-router-select-threads128-promotion.json`. |
| LCP-M2 stream-ordered contiguous prefill metadata scoped gfx11 defaults | `prepare_prefill_chunk_metadata`; gfx1100/gfx1151 backend ceiling `GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS=4096`; rollback `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=0` | `hipengine/kernels/hip_gfx1100/runtime/state.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/kernels/hip_gfx{1100,1151}/__init__.py`, `tests/test_runtime_state_unpack_metadata.py`, `tests/test_gguf_packed_verify_layout.py` | Replaces six synchronous per-chunk H2D metadata copies with one same-stream kernel through 4K. gfx1151 clean state is 83/83 exact and five-pair 512/1K/4K prefill improves **+1.56%/+0.90%/+0.53%**. The W7900 control trace confirms the six remaining `hipMemcpy` calls cost **167.213 us** of HIP API wall at 512; the 4K candidate kernel runs in **4.160 us**, with 16 VGPR and zero LDS/scratch. Its metadata primitive is exact, and balanced full-model 512/4K prefill improves **2793.871 -> 2805.451 (+0.41%)** and **3069.782 -> 3144.263 tok/s (+2.43%)** (paired medians **+0.26%/+2.26%**), with non-regressive graph decode and unchanged memory/IDs. Both backends now select device metadata through 4K and retain synchronous preparation above it. Explicit env `0|1` remains for rollback/diagnosis. Evidence: `benchmarks/results/2026-07-15-gfx1151-gguf-prefill-device-metadata-scoped-promotion.json`, `benchmarks/results/2026-07-16-gfx1100-gguf-prefill-device-metadata-promotion.json`. |
| gfx1151 persistent prefill flight recorder | `prefill_flight_recorder_mark_i64_kernel`; `PrefillFlightRecorder` | `hipengine/kernels/hip_gfx1100/runtime/state.{hip,py}`, `hipengine/runtime/prefill_flight_recorder.py`, `hipengine/runtime/qwen35_gguf_runner.py`, `scripts/qwen35_prefill_flight_recorder.py`, `tests/test_prefill_flight_recorder.py` | Default-off diagnostic. A fixed file-backed mmap stores an 8,192-entry host submission ring; HIP registers it as mapped host memory, and the one-thread marker writes a monotonic same-stream completion cursor followed by `__threadfence_system()`. `chunk` mode records every layer submission but launches only one retirement marker per reset/4K outer chunk plus finalize/sample boundaries; `layer` is a more perturbing refinement. Fake-runtime/parser tests plus a real gfx1151 cross-process visibility test pass. Cached `rocprofv3 --kernel-trace` sees `(anonymous namespace)::prefill_flight_recorder_mark_i64_kernel(long*, long)` at **3.206 us**, one thread, 8 VGPR, zero LDS/scratch. A 512/1 full-model chunk-mode smoke finishes exact at token `9707` (**1233.080 prefill / 48.520 decode tok/s**, cursors **46/46**); diagnostic throughput is not retainable. |
| gfx1151 HIP one-hardware-queue process default | `configure_hip_process_environment`; `GPU_MAX_HW_QUEUES=1` | `hipengine/kernels/backends.py`, `hipengine/core/hip.py`, `tests/test_gfx1151_backend.py`, `tests/test_hip_runtime.py` | Process-start backend metadata is applied before `libamdhip64` loads. On clean `4d0aa281`, ROCm's default four-queue policy enters a bounded 128K first-warmup stall at 100%/2.9 GHz but only 41-43 W; changing only `GPU_MAX_HW_QUEUES=1` once completes warmup+3 at **499.755 / 500.210/500.873/500.687 prefill tok/s**, exact token `9707`, and unchanged memory. Clean 512/4K A/B is **+0.35%/+0.46% prefill** and **+0.066%/+0.072% decode**, so one queue remains the risk-reducing default. It is not lifecycle-safe: current automatic one-queue, explicit router-512/metadata-off, and `HSA_ENABLE_SDMA=0` full 128K attempts all reproduce the low-power stall. A clean full-stack matrix also reproduces it under HIP 7.13 and 7.15: 7.13 completes two exact gates before a third stalls after measured pass 1; 7.15 stalls in both controls. Existing values are preserved (`=4` is the documented ROCm default/rollback); gfx1100 and mixed recognized arches are unchanged. Evidence: `benchmarks/results/2026-07-15-gfx1151-hip-one-queue-stability-promotion.json`, `benchmarks/results/2026-07-15-gfx1151-gguf-production-refresh-512-64k-128k-blocked.json`, `benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json`; upstream follow-up: ROCm#5107 comment 4979442043. |
| 2026-07-15 gfx1151 exact-decode closure profile | retained `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel` with chunk 256 + LCP-D1 reducer; retained 128-thread Q8T16 GEMV; retained production graph replay | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.{hip,py}`, `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_gemv.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py` | Fresh exact 16-step marker profiles measure dense Q8 at **8.560/8.541/8.555 ms/token** on 512/4K/128K and 128K attention at **17.504 ms/token** (grouped-GQA body 15.509 + reducer 1.962). Chunk 128 is +2.89% on a deterministic 128K context+reduce fixture but changes one BF16 output; chunk 512 is inexact and slower. The only supported Q8 thread alternative (64) has 15.8% longer wall than 128 on the dominant rows=1 `2048x(8192+4096)` split pair. Current graph replay beats eager **+1.00%/+0.86%/+0.36%** at 512/4K/128K with exact IDs. No new kernel is promoted; another attempt requires a new exact algorithm/layout. `benchmarks/results/2026-07-15-gfx1151-gguf-decode-closure-profile.json`. |
| LCP-3A gfx1100 Q8T16 four-wave prefill widening (rejected and removed) | no new default | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.{hip,py}`, `tests/test_gguf_q8_0_t16_wmma_prefill.py` during the temporary screen | Four production-order 32-column waves shared one 1 KiB activation tile. All 83 tests passed and candidate outputs were byte-exact, but sequential W7900 60-repetition leaf A/Bs were mixed: width-8192 `-0.185%`, long-K width-2048 `-0.310%`, width-4096 `+2.24%`, and short-K width-2048 `+8.87%`. The measured pp512 mix projects about a 0.3 ms regression, so the candidate was removed before profiler/model routing. Independent FP16-WMMA wave widening is exhausted; any continuation must change the body. Artifact: `benchmarks/results/2026-07-15-gfx1100-gguf-q8t16-four-wave-rejected.json`. |
| LCP-3B gfx1100 direct Q8_1 x Q8T16 integer-WMMA prefill (rejected and removed) | no new default | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.{hip,py}`, `tests/test_gguf_q8_0_t16_wmma_prefill.py` during the temporary screen | The standalone 64-output x32-row candidate passed Q8_0 x Q8_1 CPU-oracle relative-L2 <=0.02, KL <=0.05, and top-1 >=90% on ordinary/tail fixtures. The primary W7900 cycling-pool gate rejected it: production FP16-WMMA `0.521342 ms`, prequantized integer body `0.754182 ms` (+44.66%), quantization+body `0.770183 ms` (+47.73%). Quantization is only 0.016001 ms; direct T16 packing and per-32-K scaled accumulation are the loss. Candidate code was removed before profiler/model routing. Artifact: `benchmarks/results/2026-07-15-gfx1100-gguf-q8t16-q8-1-i8-wmma-rejected.json`. |
| LCP-3C llama.cpp Q8_0 MMQ source/layout audit; LCP-3D T16-backed MMQ128 (rejected and removed) | no new default | measured llama.cpp HIP `1ebf790cda38`: `ggml/src/ggml-cuda/{mmq.cuh,mma.cuh,quantize.cu}`; temporary candidate in `gguf_q8_0_t16_prefill.{hip,py}` and its focused test | The source audit mapped llama.cpp's 256-thread, 128-output x128-token K256 shared tile, 57,856 B dynamic LDS, and two integer-WMMA K16 calls per 32-K interval. A standalone T16-backed reproduction passed D4 pack bytes and ordinary/tail relative-L2/KL/top-1 gates, but the primary W7900 leaf rejected it: production two-wave FP16 WMMA **0.523062 ms**, prequantized MMQ128 **1.144524 ms (+118.81%)**, D4 pack+body **1.151586 ms (+120.16%)**. D4 packing adds only 0.007061 ms. The blocker is T16's K-major 16-column payload: filling the output-major packed-int shared tile requires four byte gathers/packing per int32 fragment. Candidate code was removed before profiler/model routing. Source audit: `benchmarks/results/2026-07-15-gfx1100-gguf-q8-mmq-source-audit.json`; rejection: `benchmarks/results/2026-07-15-gfx1100-gguf-q8t16-mmq128-rejected.json`. |
| LCP-3E raw/output-major Q8 MMQ128 (rejected and removed) | no new default | temporary standalone `gguf_q8_0_mmq_prefill.{hip,py}` plus focused test; source lineage llama.cpp HIP `1ebf790cda38` | The source-compatible 256-thread, 128x128, K256 D4 integer-WMMA leaf passed D4 byte equality plus ordinary/tail relative-L2, KL, and top-1 gates. Implementation validation removed an invalid 418-spill unroll, matched WGP mode, emitted the source's 24 aligned `ds_load_b128` fragment loads and 32 integer-WMMA instructions, and compiled the final full-tile body at 210 VGPR with zero scratch/spills and 57,856 B dynamic LDS. The frozen W7900 primary gate still rejected it: production Q8T16 FP16 WMMA **0.521823 ms**, prequantized raw MMQ128 **0.542442 ms (+3.95%)**, D4 pack+body **0.549562 ms (+5.32%)**. Because both rows had to win and raw+T16 dual residency would add ~1.390 GiB, the candidate was removed before profiler/model routing. Artifact: `benchmarks/results/2026-07-15-gfx1100-gguf-raw-q8-mmq128-rejected.json`. |
| LCP-5A gfx1100 T16 selected-prefill spill removal | existing `gguf_q{4,5,6}_k_t16_selected_wmma_prefill_compact_{bf16,fp16}_{bf16,fp16}_out` aliases | `hipengine/kernels/hip_gfx1100/quant/gguf_k_t16_selected_prefill.hip`, `tests/test_gguf_{k,q4_k}_t16_selected_wmma_prefill.py` | HIP 7.2 over-unrolled the outer 8-subblock Q4/Q5 and 16-K-tile Q6 loops on gfx1100: Q5 BF16 compiled at 256 VGPR with 176 private bytes/thread and 75 spills, while identical HIP 7.15 source was 66 VGPR/zero spill. gfx1100 now keeps the outer loop rolled and retains unrolled K16 work; gfx1151 keeps its independently validated schedule. All 46 Q4/Q5/Q6 CPU-oracle GPU fixtures pass byte-exact. Static gfx1100 resources are Q4/Q5/Q6 BF16 70/91/73 VGPR with zero private bytes/spills; cached rocprof reports allocated 56/96/80 VGPR and zero scratch. Matched 40-layer replay moves Q5 **50.863 -> 31.725 ms (-37.63%)**, Q6 **3.475 -> 2.630 ms (-24.31%)**, and selected-MoE **98.538 -> 78.358 ms (-20.48%)**. The final pp512 trace moves Q5 **51.009 -> 29.544 ms (-42.08%)**, total peer kernels **203.808 -> 184.513 ms**, and span **215.307 -> 194.886 ms**. The follow-on peer semantic/decode and selector-unset 512/4K floors admit `chain_peer_wave32` as the gfx1100 package default while retaining explicit scalar-exact direct-LDS32 rollback. |
| LCP-D3 gfx1100 selected-Q4T16 half-sequential pressure cut (rejected and removed) | no new default | temporary change to `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.hip`; existing `tests/test_gguf_t16_selected_gemv_decode.py` | The post-router 4K system-7.2 trace made the fused c=1 Q4T16 gate/up+SiLU leaf the largest selected-MoE kernel at **975.3 us/token (11.24%)**, 200 profiler VGPR, and zero scratch. Processing columns 0-7 then 8-15 preserved every output's K/reduction/BF16/SiLU order and passed all 88 focused tests; static VGPR fell **195 -> 115** on system 7.2 and **195 -> 114** on therock 7.15, with zero private bytes/spills. The extra K/x traversal nevertheless lost both model gates: system-7.2 balanced eager wall **+1.05%**, and canonical therock-7.15 4K/128 graph decode **100.146 -> 97.348 tok/s (-2.79%)**, with exact IDs and unchanged 21.670 GiB tracked peak. Candidate source was removed. Artifact: `benchmarks/results/2026-07-15-gfx1100-gguf-decode-q4t16-halfseq-rejected.json`. |
| P9.H3d selected-MoE T16 GEMV decode | `selected_dual_t16_gemv_decode_{compact_,}{bf16,fp16}_{bf16,fp16}_out`; `selected_dual_t16_silu_gemv_decode_bf16_bf16_out`; `selected_t16_gemv_decode_{compact_,}{bf16,fp16}_{bf16,fp16}_out`; diagnostic `selected_dual_t16_q8_1_dp4a_gemv_decode_bf16_bf16_out` and `selected_t16_q8_1_dp4a_gemv_decode_bf16_bf16_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `hipengine/runtime/qwen35_gguf_runner.py`, `tests/test_gguf_t16_selected_gemv_decode.py`, `tests/test_qwen35_gguf_compact_moe_gemv_routing.py` | Adds Q4T16 selected dual gate/up plus Q5T16/Q6T16 selected down GEMV kernels for both direct selected-row ABI and compact grouped-MoE ABI. Direct variants consume `(x, selected_experts, tiles, ...)` so qwen35moe decode and row-bulk fallback can use replacement `tiles` without the compact scheduler; compact variants remain registered for scheduler-based paths. Q4 direct uses the raw row-GEMV-compatible 128-thread reduction order to keep P9.E2 logits bit-aligned. P9.D4 adds a decode-only Q4T16 dual+SiLU direct variant that BF16-round-trips gate/up accumulators before applying SiLU, removing the separate rows=1 selected-expert SiLU launch while keeping rows>1 bulk prefill on the split pair+SiLU path. 2026-06-27 adds a q8_1+sudot4 diagnostic for Q4T16 rows>1 split gate/up: isolated gfx1151 microbench at `x_rows=4, rows=32, E=256, in=2048, out=512` measured T16 split `0.198 ms` vs q8_1 quantize+dp4a `0.191 ms` (`1.04x`), gate/up `KL_mean=9.25e-05`, top-1 `1.0`; extracted AMDGPU disassembly contains `v_dot4_i32_iu8`. Production B3 split-only trace confirms `80` dp4a row-bulk calls avg `141.8 us` plus `80` q8 quantize calls avg `3.35 us`, while c1 fused stays on the exact float path because the callable fused-SiLU dp4a diagnostic regressed c1. 2026-06-27 also adds a Q5T16 selected-down q8_1+sudot4 diagnostic under `HIPENGINE_GGUF_T16_SELECTED_DP4A`: c1-shaped microbench (`rows=8, E=256, in=512, out=2048`) measured T16 `0.0335 ms` vs q8_1 quantize+dp4a `0.0306 ms` (`1.10x`), `KL_mean=0.00678`, `KL_max=0.03093`, top-1 `0.875`; `rocprofv3` shows `qk_t16_selected_q8_1_dp4a_direct_gemv_kernel<unsigned short>` around `25-26 us`, and extracted device ISA contains `v_dot4_i32_iu8`. B3 stayed exact (`15/15`) but regressed to `47.62 tok/s`, warm `48.44`, so Q5 remains default-off and Q6 dp4a is not routed after a synthetic probe missed top-1. Runtime selected-MoE routing resolves T16 quant keys to `tiles` allocations and leaves raw pack8 GEMV fallback disabled for replacement tensors. Validation: `HIPENGINE_HIP_ARCH=gfx1151 python3 -m pytest tests/test_gguf_t16_selected_gemv_decode.py -q` -> `88 passed`; `python3 -m pytest tests/test_qwen35_gguf_compact_moe_gemv_routing.py tests/test_qwen35_gguf_compact_moe_wmma_routing.py -q` -> `16 passed`; B3/C5 smoke with `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A=1` stayed exact (`15/15`) but flat (`49.31 tok/s`, warm `50.60`). Artifacts: `benchmarks/results/2026-06-27-hipengine-gguf-q4-k-t16-selected-dual-dp4a-poc.json`, `benchmarks/results/2026-06-27-hipengine-mtp-b3-q4k-t16-dp4a-verifier-diagnostic.json`, `benchmarks/results/2026-06-27-hipengine-gguf-q5-k-t16-selected-down-dp4a-poc.json`, `benchmarks/results/2026-06-27-hipengine-mtp-b3-q5-t16-dp4a-verifier-diagnostic.json`. |
| P9.H3e dense Q6T16 lm-head GEMV decode | `gguf_q6_k_t16_v1` `t16_gemv_decode_bf16_{f32,bf16}_out` | `hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_t16_gemv.{hip,py}`, `hipengine/loading/qwen35_gguf_materialize.py`, `hipengine/runtime/gguf_linear.py`, `tests/test_gguf_q6_k_t16_gemv_decode.py` | Extends decode-repack mode to root `lm_head` when it is Q6_K, materializing `repack_gguf_q6_k_tile16(raw[None,...])` as byte-neutral `tiles` and routing BF16 hidden -> FP32 logits through dense Q6T16 GEMV. This removes the previously allowed Q6_K legacy `prefill_out` lm-head fallback from current decode traces. Synthetic GPU correctness matches CPU `gguf_quant_gemv(..., Q6_K)` for rows `{1,2}`, in_features `{256,512}`, and output tiles `{16,128,256}`; dispatch/materialization tests prove `LAYOUT_GGUF_Q6_K_T16` selects the `t16` ABI. |
| P9.E1 rocprof bandwidth-utilization summary helper | `scripts/qwen35_gguf_rocprof_summary.py` | per-kernel + per-bucket rollup of `rocprofv3 --kernel-trace` CSVs | `python3 scripts/qwen35_gguf_rocprof_summary.py --csv ... --json out.json` | Read-only tooling (no kernel changes) for future P9 perf rows. Single-CSV mode rolls one CSV into a `"prefill"` phase block; paired mode (`--prefill-csv` + `--decode-csv` with optional `--strip-prefill-prefix`) emits both `"prefill"` and `"decode"` phases. Per kernel: `total_ms`, `dispatches`, `avg_dispatch_ms`, `share_of_phase`. Per bucket: same totals plus optional `effective_gb_s` back-calculated per `docs/ROOFLINE.md` 12.4 using a per-dispatch byte footprint dict. The dict defaults to Qwen3.6-35B-A3B-UD-Q4_K_M-shaped MoE/dense surfaces; override or extend via `--config-json`. Bucket classifier is GGUF-aware: distinguishes P8 WMMA prefill (`*_wmma_prefill_*`), P9.B decode GEMV (`*_pack8_gemv_decode_*`), and legacy `*_prefill_out_*` per quant template number (`<5>`/`<6>`/`<8>`), plus GDN, full-attention, router, scheduler, SiLU, combine, RMSNorm, KV-write, and runtime copy. Smoke on the post-P9.A1 512/0 CSV from `/tmp/p9_a1/rocprof-512-0/rocm/2441463_kernel_trace.csv`: total `297.265 ms / 1558 dispatches`, top buckets dense Q8_0 WMMA prefill `75.0 ms / ~14.8 GB/s`, Q4_K selected dual WMMA `65.0 ms / ~46.5 GB/s`, GDN recurrent `52.1 ms`, full attention `39.5 ms`, Q5_K selected WMMA `27.3 ms / ~62.5 GB/s`. Validation: 54 unit tests in `tests/test_qwen35_gguf_rocprof_summary.py` cover the bucket classifier (43 parametrised cases), CSV parsing edge cases, per-phase aggregation, footprint-based GB/s override, paired-mode prefix strip, and CLI validation. Adjacent dispatch + routing regression: 111 pass. |
| `moe_linear` variants `selected_wmma_prefill_compact_{bf16,fp16}_{bf16,fp16}_out` (P8.5) | `gguf_q5_k`, `gguf_q6_k` raw rank-3 expert weights, compact grouped-MoE down prefill | `hipengine/kernels/hip_gfx1100/quant/gguf_k_selected_prefill.hip` | `gguf_q5_k_selected_wmma_prefill_compact_{bf16,fp16}_{bf16,fp16}_out(...)`, `gguf_q6_k_selected_wmma_prefill_compact_{bf16,fp16}_{bf16,fp16}_out(...)` | Mirrors the P8.4 selected Q4_K compact WMMA/PARO compact ABI for single down-projection outputs: `x[compact_rows,in_features]`, `expert_start_compact`, `wmma_expert_start`, `tile_expert`, raw expert tensor `[E,out_features,row_bytes]`, output `[compact_rows,out_features]`. Inner loop dequantizes raw GGUF bytes in-register: Q5_K uses Q4_K scale/min plus high-bit bytes (`block_q5_K` 176 B = fp16 `d`, fp16 `dmin`, 12 scale/min bytes, 32 high-bit bytes, 128 q4 bytes); Q6_K uses low/high quant bytes plus per-16-K int8 scales and fp16 super-scale (`block_q6_K` 210 B = 128 `ql`, 64 `qh`, 16 int8 scales, fp16 `d`). `__launch_bounds__(32,2)`, one wave32 per 16x16 compact tile, no new compact-MoE ABI and no sidecar/repack. Wrapper import/registry smoke passes; HIP source compiles to `gguf_k_selected_prefill.so` on W7900. `tests/test_gguf_k_selected_wmma_prefill.py` covers registry/build-plan/contract checks plus BF16/FP16 compact-MoE correctness against CPU `gguf_quant_gemv(..., Q5_K/Q6_K)` across multiple experts, uneven row counts, padding, empty experts, multi-block K, and non-multiple-of-16 output widths (`22 passed` narrow; adjacent selected/GGUF-K bundle `37 passed`). `rocprofv3 --kernel-trace` on the tiny selected test confirms both `gguf_k_selected_wmma_prefill_compact_kernel<unsigned short, 5>` (`DurationNs=23598`) and `<unsigned short, 6>` (`DurationNs=24680`) launch. |
| `moe_linear` variants `selected_pack8_gemv_decode_compact_{bf16,fp16}_{bf16,fp16}_out` (P9.B2) | `gguf_q5_k`, `gguf_q6_k` raw rank-3 expert weights, compact grouped-MoE down GEMV decode | `hipengine/kernels/hip_gfx1100/quant/gguf_k_selected_pack8_gemv.hip` | `gguf_q5_k_selected_pack8_gemv_decode_compact_{bf16,fp16}_{bf16,fp16}_out(...)`, `gguf_q6_k_selected_pack8_gemv_decode_compact_{bf16,fp16}_{bf16,fp16}_out(...)` | Mirrors `paro_awq_gemv.hip::gemv_awq_selected_pack8_kernel` (PARO-style pack8 row layout, `__launch_bounds__(128, 4)`, 4 wave32 wave-level reduction across `xchg[4*8]`) but consumes the P8.5 compact-MoE ABI for single-output down projection: `x[compact_rows,in_features]`, `expert_start_compact[E+1]`, raw rank-3 expert tensors `[E,out_features,row_bytes]`, output `[compact_rows,out_features]`. One block computes 8 output channels for one compact row; expert id recovered via linear scan over `expert_start_compact` (decode rows=1 per active expert lane; `num_experts` small in practice). Per-block hoist of scale (and Q5_K min) into shared memory: Q5_K reuses Q4_K-style 6-bit packed scale/min from the same 12-byte scales block + 32 high-bit bytes + 128 packed nibbles (`s_scale[8*8]`, `s_min[8*8]`, 64 cooperative threads); Q6_K uses per-16-K int8 scales and fp16 super-scale, no min (`s_scale[8*16]`, 128 cooperative threads). Grid: `(out_features / 8, compact_rows)`. Constraints: `in_features % 256 == 0`, `out_features % 8 == 0`; no new compact-MoE ABI and no sidecar/repack. Wrapper import/registry smoke passes; HIP source compiles to `gguf_k_selected_pack8_gemv.so` on W7900. Inline GPU smoke vs CPU `gguf_quant_gemv(..., Q5_K/Q6_K)`: BF16 Q5_K max|d|=`1.604`, max_rel(eps=1)=`0.0037`; BF16 Q6_K max|d|=`0.214`, max_rel=`0.0038`; FP16 Q5_K max|d|=`0.106`, max_rel=`0.00043`; FP16 Q6_K max|d|=`0.027`, max_rel=`0.00042`. Formal compact-MoE correctness fixture (uneven counts, empty experts, padding, non-multiple-of-16 out widths) lands in task #24 (P9.B5). |
| `linear` variants `gemv_*_f32_out`, `gemv_*_fp16_out`, `gemv_bf16_bf16_out`, `prefill_*_f32_out`, `prefill_*_fp16_out`, `prefill_bf16_bf16_out` | `gguf_q4_k` raw GGUF block-q4_K weights | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip` | `gguf_q4_k_gemv_*_out(...)`, `gguf_q4_k_prefill_*_out(...)` | `python3 scripts/smoke.py --mode gguf-q4-k-gemv-hip --rows 4 --hidden-size 512 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → synthetic raw-Q4_K fixture passes (`f32/fp16/bf16/bf16_out max_abs=0.0`, `bf16_out_bit_mismatch=0`); rows>1 dispatches use the renamed row-grid `gguf_q4_k_prefill_out_kernel` device body. Historical `/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf` tensor `blk.0.attn_gate.weight` one-row smoke max abs `1.79e-07` vs CPU reference. |
| `linear` diagnostic variant `selected_dual_dp4a_gemv_bf16_bf16_out` plus prequantized helper `selected_dual_q8_1_dp4a_gemv_bf16_bf16_out` | `gguf_q4_k` raw selected MoE gate/up weights with q8_1 activation quantization | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip` | `gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out(...)`, `gguf_q4_k_quantize_bf16_q8_1(...)` | Bounded llama.cpp-parity POC for the selected-MoE Q4_K dual gate/up verifier bucket. The diagnostic wrapper preserves the selected-dual ABI but owns a temporary q8_1 buffer and synchronizes before freeing; verifier/runtime integration must use caller-owned q8_1 workspace before any default-path decision. Validation: `HIPENGINE_HIP_ARCH=gfx1151 python3 -m pytest tests/test_gguf_q4_k_selected_dual_dp4a_gemv.py tests/test_gguf_q4_k_gemv.py tests/test_gguf_q4_k_rowtile_gemv.py -q` → `57 passed`; CPU q8_1 oracle match plus KL/top-1 vs existing float-dequant path (`KL<=0.05`, top-1 `>=0.90`). Microbench on gfx1151 qwen35moe verifier shape (`x_rows=4`, `rows=32`, `E=256`, `in=2048`, `out=512`) measured raw `0.946 ms` vs q8_1 quantize+dp4a `0.357 ms` (**2.65x**), q8_1 quantize `0.0025 ms`, artifact `benchmarks/results/2026-06-27-hipengine-gguf-q4-k-selected-dual-dp4a-poc.json`. `rocprofv3 --kernel-trace` cached smoke shows `gguf_q4_k_selected_dual_q8_1_dp4a_prefill_out_kernel` avg `338 us` vs raw selected-dual avg `1007 us`; extracted AMDGPU disassembly contains `v_dot4_i32_iu8`. |
| `moe_linear` diagnostic variant `selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out` | `gguf_q4_k` raw rank-3 expert gate/up weights plus caller-owned producer-row DS4-Q8_1 activations | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}` | `gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(...)` | LAP-1 source-faithful gfx1151 packed-dot primitive based on llama.cpp Vulkan `c0bc8591e` `mul_mmq*`: one local128 workgroup maps four wave32s across a 32-column x 32-routed-row tile, stages 20-byte Q4_K columns and 36-byte DS4-Q8_1 rows for each K32 interval, and reuses them through native `sudot4`. Producer rows are quantized once and compact rows consume a source-row map. Two CPU-source fixtures, including uneven/empty experts and a nonidentity source map, pass at max softmax KL **4.745e-5**, top-1 **100%**, and max abs **0.1236** (`24 passed` focused file). Actual Laguna layer-1 K3072/N1024 natural routing-count replay, inclusive of one producer-row Q8 pack, improves M256 **26.612 -> 10.047 ms (2.649x)** and M512 **52.522 -> 12.720 ms (4.129x)** over retained direct; the T16 diagnostic is **6.297/9.307 ms**. Raw gate/up residency is **864 MiB** versus **888 MiB** T16. Cached rocprof reports local128, allocated VGPR120/SGPR128, LDS2048B, scratch0; ISA contains 64 static `v_dot4_i32_iu8` per wave. The clean all-shape replay measures inclusive speedups **0.680/0.899/0.985/1.515/1.551/2.645/4.117x** at M32/55/64/122/128/256/512. Explicit raw-layout diagnostic only; the direct T16 sibling now passes separately and LAP-2 repair remains before any default change. Artifacts: `benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-mmq32-leaf.json`, `benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-mmq32-shape-screen.json`. |
| `moe_linear` diagnostic variant `selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out` | `gguf_q4_k_x8_v1` byte-exact, byte-neutral pack-of-8 replacement weights plus caller-owned producer-row DS4-Q8_1 activations | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}` | `gguf_q4_k_x8_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(...)` | LAP-1 X8 sibling of the raw MMQ32 leaf. It shares the complete packed-dot arithmetic body and changes only the Q4_K block address to `[expert,out_pack8,k_block,col_in_pack8]`; resident bytes are unchanged. The body clamps each tile's natural live-row count once and bypasses packed-dot accumulation for padded routes while preserving the original accumulation order for every live output. Two uneven/empty-expert fixtures, including a nonidentity source-row map, are BF16-bit identical to raw MMQ32 and pass the CPU KL/top-1 gate. The focused kernel-family plus harness bundle reports **29 passed** on gfx1151. Cached post-skip tracing names raw/X8 at local128, VGPR40/48, SGPR128, LDS2048B, scratch0, with plausible X8 **6.853/16.872 us** tiny-fixture dispatches. The clean actual layer-1 K3072/N1024 live-row screen reaches **1.197/1.567/1.704/2.526/2.587/4.092/5.614x** retained direct at M32/55/64/122/128/256/512 and reduces the prior X8 time by **18.65–36.45%**, with exact checksums and unchanged **905,969,664-byte** gate/up residency. An all-full synthetic control regresses **8.34%**. Retained explicit prefill control only: the exact-decode gate rejects X8 as the sole c=1 resident layout, so a direct T16 MMQ sibling remains before any runtime default. Artifacts: `benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-layout-retained.json`, `benchmarks/results/2026-07-24-gfx1151-laguna-q4-k-x8-mmq32-live-row-retained.json`, `benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json`. |
| `moe_linear` diagnostic variant `selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out` | resident `gguf_q4_k_t16_v1` gate/up weights plus caller-owned producer-row DS4-Q8_1 activations | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}` | `gguf_q4_k_t16_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out(...)` | Direct-T16 LAP-1 consumer. It indexes the resident `[expert,out_tile16,k_block,2368]` representation and loads its existing FP16 `d/dmin`, expanded uint8 scale/min, and interleaved Q4 payload directly into the proven 20-byte MMQ cache; it never reconstructs raw/X8 tiles and allocates no weight sidecar. Two uneven/empty expert fixtures, including a nonidentity source-row map, are BF16-bit identical to raw/X8 MMQ and pass the independent CPU KL/top-1 gate; the focused kernel/harness bundle reports **31 passed**. Cached gfx1151 tracing names the intended local128 kernel at VGPR48/SGPR128/LDS2048B/scratch0 with **6.973/17.874 us** tiny-fixture dispatches, and extracted device ISA contains `v_dot4_i32_iu8` in this symbol. The clean actual layer-1 producer-pack-inclusive screen reaches **1.174/1.528/1.662/2.464/2.502/3.959/5.502x** retained at M32/55/64/122/128/256/512 and remains only **4.66%/4.05%/3.02%** behind X8 at the primary shapes. LAP-1 passes; retained explicit primitive only until the guarded residual policy is calibrated and integrated. Artifact: `benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-t16-mmq32-retained.json`. |
| `activation_quant` variant `q8_1_ds4x3/bf16`; `moe_linear` variants `selected_dual_q8_1_ds4x3_{,guarded_}mmq32_prefill_compact32_bf16_bf16_out`; `moe_linear_repair` variant `selected_dual_sparse_exact_bf16` | resident `gguf_q4_k_t16_v1` gate/up weights, raw BF16 producer rows, and three plane-major DS4-Q8_1 activation packs | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}`, `hipengine/runtime/laguna_{moe,gguf_runner}.py` | `gguf_q8_1_mmq_ds4_pack_bf16_d4x3(...)`, direct/guarded three-pass MMQ32, and `gguf_q4_k_t16_selected_dual_sparse_exact_correct_bf16(...)` | LAP-2 arithmetic primitive and explicit LAP-3 integration candidate. Plane zero remains byte-identical to the retained DS4 pack; two residual planes recursively quantize the BF16 reconstruction error. The MMQ reuses each resident T16 weight tile across all three planes, optionally ballots outputs near a BF16 rounding boundary into a bounded 16-column tile queue, and the local256 correction kernel recomputes queued tiles with the production exact T16 reduction order. Queue overflow fails safe by recomputing every output tile, with a 4,096-block grid-stride cap rather than launching one block per possible output tile. The residual pack reconstructs its BF16 source at relative L2 `<=5e-5`; on the finite Q4_K CPU fixture the three-plane projection reduces relative L2 from `0.002922` to `0.001826`, while the all-queued and forced-overflow fixtures are BF16-bit exact to production direct T16. The runtime candidate quantizes producer rows once, builds source/MMQ metadata on device, writes compact gate/up, and feeds the compact SiLU result directly to exact grouped down without a second compact/gather. The 61-test focused bundle passes, including complete synthetic Q4/Q6 MoE KL/top-1. Cached runtime tracing names D4x3 pack, local128 MMQ, dual SiLU, and exact grouped down with zero scratch. A same-session dirty-tree actual pp512 diagnostic improves **76.414 -> 127.607 tok/s (1.670x)** with the same next token. This is not a default promotion: the canonical full-model quality/category gate, real-input repair rate, and clean retained A/B remain open. |
| `activation_quant` variants `q8_1_ds4{,x2,x3}_f32/bf16`, `q8_1_ds8_f32/bf16`; `moe_linear` variants `selected_{dual_}q8_1_ds4{,x2,x3}_f32_mmq64x32_prefill_compact32_bf16_bf16_out`, `selected_dual_q8_1_ds8_f32_mmq128x32_prefill_compact32_bf16_bf16_out` | resident Q4_K/Q6_K T16 selected gate/up/down weights plus producer or compact post-SiLU BF16 rows | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}`, `hipengine/runtime/laguna_moe.py` | range-safe same-byte FP32-scale activation packs and direct Q4/Q6 T16 integer-dot consumers | LAP-3/LAP-4 gfx1151 defaults. FP32 metadata removes FP16 range exposure. The original D4-gate/D4-down route reached **355.273/355.721 tok/s** but the complete category gate rejected it at max KL **0.0767056**. The admitted gate pack stores eight FP32 scales plus 128 int8 values in the same 160-byte block, giving one scale per 16 values; down stays D4. Its local128 Q4 dual block owns 128 columns x 32 routed rows, reconstructs signed half-block quant sums for the Q4 min term, and emits native packed integer dots. Uneven/empty-expert CPU gates pass. The clean complete category gate reaches max KL **0.040724836**, **317/320** top-1, **2.615x** aggregate natural-prompt prefill, flat decode, and exact lifecycle recovery. Clean selector-unset production pp512 is **353.421/355.584/354.820 tok/s** (median **354.820**), token 2930. The cached-only final trace independently measures **354.763 tok/s** and records the 128-column gate/up specialization at local128/VGPR80/LDS6656B/scratch0; the D8 pack is local128/VGPR16/LDS512B/scratch0. The route adds no weight sidecar, leaves c=1 unchanged, and retains exact selected/grouped fallbacks. The exact Q6 compact-activation specialization removes unused Q8 sum metadata and narrows each bounded K16 quant sum to int16, reducing the 64-row body's LDS **5,632 -> 5,120 B** with unchanged VGPR/scratch. The actual leaf improves **3.36%**, the traced Q6 family improves **4.64%**, and fifteen complete-state pp512 pairs improve **0.404%** with 15/15 wins. Clean selector-unset production reaches **550.625/517.017/431.789 tok/s** at 512/1K/4K; its trace records compact Q6 at **119.384 ms**, local128/VGPR88/SGPR128/LDS5120B/scratch0. The exact half-row specialization then assigns one 16-byte activation half and one K16 sum to each of all 128 threads with the same resources. It improves **21/23** actual Q6 layers and the sum of layer medians **111.798 -> 111.490 ms (-0.276%)**, with zero BF16 mismatches and positive exact complete-model A/B. Clean publication is headline-neutral at **549.150/514.956/430.300 tok/s**, but its trace independently cuts Q6 **119.384 -> 118.568 ms (-0.684%)**, so the verified exact sub-window remains production. The final exact staging specialization omits activation-cache writes and K16 sums for never-read padded slots: **19/23** layers improve and the family sum moves **112.008 -> 111.806 ms (-0.180%)**, while exact complete pp512 moves **552.983 -> 553.559 tok/s (+0.104%)**. Clean 512/1K/4K publication improves to **551.459/517.307/432.099 tok/s** with unchanged local128/VGPR88/SGPR128/LDS5120B/scratch0; its single Q6 trace is noisy, so the repeated exact sub-window and clean medians are the retention evidence. Artifacts: `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-d8-category.json`, `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production.json`, `benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production-trace.json`, `benchmarks/results/2026-07-26-gfx1151-laguna-q6-compact-activation-{candidate,production}.json`, `benchmarks/results/2026-07-26-gfx1151-laguna-q6-half-row-activation-{candidate,production}.json`, and `benchmarks/results/2026-07-26-gfx1151-laguna-q6-skip-padded-activation-{candidate,production}.json`. |
| `silu_mul_dual+activation_quant` variant `q8_1_ds4x3_f32/bf16` | packed BF16 selected gate/up rows and caller-owned FP32-metadata DS4-Q8_1 output | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.{hip,py}`, `hipengine/runtime/laguna_{moe,gguf_runner}.py` | `gguf_q8_1_mmq_ds4_f32_pack_dual_silu_bf16_d4x3(...)` | Exact Laguna selected-down input primitive and retained gfx1151 production default. It reads packed gate/up directly, evaluates the same `gate * sigmoid(gate) * up` expression as `silu_mul_dual_out_bf16`, explicitly rounds that result to BF16, converts the rounded value back to FP32, and then executes the unchanged one/two/three-plane range-safe pack. The registered standalone SiLU plus activation-quant kernels remain its required unfused fallback. Production-shaped scratch reuse writes the 62.9-MB packed gate/up tensor into the larger 73.4-MB selected-down output allocation and packs into the existing gate/up allocation, adding no memory. RED failed on the missing registry/wrapper. Residual-pass 1/3 fixtures and production Q4_K/Q6_K complete MoE outputs are BF16-byte exact. Seven complete-state pp512 pairs are exact and win **7/7**, with paired geometric throughput **+0.651%**. Cached gfx1151 tracing removes 47 launches and cuts the target window **10.301 -> 6.377 ms (-38.09%)**; the fused production body is local128/VGPR16/LDS512B/scratch0. Clean selector-unset 512/1K/4K publication reaches **546.100/481.640/389.686 tok/s**, while cached all-family tracing records 47 fused and zero standalone selected-SiLU calls. Artifacts: `benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-candidate.json` and `benchmarks/results/2026-07-26-gfx1151-laguna-fused-silu-pack-production.json`. |
| `linear` variants `pack8_wmma_prefill_bf16_bf16_out` / `wmma_prefill_bf16_bf16_out` | resident Q4_K pack8 or raw Q6_K dense/shared weights | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.{hip,py}`, `hipengine/runtime/gguf_linear.py` | Q4 pack8 / raw-Q6 64x16 wave32 WMMA consumers | LAP-5 gfx1151 default. Both kernels reuse each decoded weight fragment across sixteen BF16 activation rows, use FP16 matrix operands and FP32 accumulation, add no weight layout, and preserve exact retained fallbacks. Q4 is BF16-bit identical to the raw-Q4 WMMA oracle; Q6 passes aligned and boundary CPU-reference fixtures. In the pre-packed-byte-reuse 320 tok/s full-model trace, Q4/Q6 dense kernels total **41.969/28.866 ms**, replacing the prior 0.6415-second dense/shared family. The clean compounded category gate admits the default. |
| `moe_linear` diagnostic variant `selected_dual_x8_exact_gemv_decode_bf16_bf16_out` | `gguf_q4_k_x8_v1` byte-exact replacement weights with raw BF16 activations | `hipengine/kernels/hip_gfx1100/quant/gguf_x8_selected_gemv.{hip,py}` | `gguf_q4_k_x8_selected_dual_exact_gemv_bf16_bf16_out(...)` | Exact selected/decode fallback for the X8 layout. One local128 block owns 16 gate and 16 up columns. For each K256 interval it converts two adjacent X8 packs into T16-shaped A/B LDS tiles—transposing nibbles and expanding scale/min metadata once—then preserves the retained T16 kernel's `k=tid+128*n`, wave32 shuffle, wave-0..3 sum, and BF16 store order. A K3072 randomized-block c=1/top-10 fixture plus a 3x4-route K512 fixture are BF16-bit identical to raw/T16 and pass the CPU-source KL/top-1 gate; the X8/repack bundle reports **14 passed**. Cached actual-weight tracing reports local128, VGPR176, SGPR128, LDS6144 B, scratch0, and warm **191.519/191.680 us** dispatches versus T16 **180.259/177.293 us**. The clean actual layer c1/c2/c4/c8 screen has zero BF16 mismatches but measures X8/T16 **1.11093/1.02987/0.99921/0.98664x**, rejecting X8 as the sole c=1 resident layout. Explicit diagnostic only; no runtime route or default changed. Artifact: `benchmarks/results/2026-07-25-gfx1151-laguna-q4-k-x8-exact-decode-rejected.json`. |
| Rejected Q4_K X16 sole-resident decode control | host-only byte-exact `[expert,out_pack16,k_block,16 raw blocks]` oracle; no quant key or device symbol retained | `hipengine/quant/gguf_x8.py`, `tests/test_gguf_x8_repack.py` | The one-pack byte-neutral control preserves exactly **905,969,664 bytes** for the actual gate/up pair and round-trips every raw byte. Its temporary local128 exact consumer beat X8 at all c1/c2/c4/c8 shapes and produced zero BF16 mismatches, but resident T16 -> X16 moved c1 **0.163258 -> 0.175753 ms (+7.654%)** and c2 **0.352933 -> 0.359698 ms (+1.917%)** before winning c4/c8 **1.167%/2.794%**. The <=2% all-shape prerequisite failed before prefill/materialization/runtime, so the device decoder was removed and no quant/runtime key was registered. Keep only the host oracle as a byte-neutral control; next screen uses T16-local Q payload plus cooperative four-column/three-byte metadata. Artifact: `benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-x16-decode-rejected.json`. |
| Q4_K T16-local qmicro exact decode primitive | host byte-exact `[expert,out_tile16,k_block,2304]` oracle plus `gguf_q4_k_qmicro_selected_dual_exact_gemv_bf16_bf16_out`; no quant key or runtime route yet | `hipengine/quant/gguf_q4_k.py`, `hipengine/kernels/hip_gfx1100/quant/gguf_x8_selected_gemv.{hip,py}`, `tests/test_gguf_{q4_k_tile16_repack,x8_selected_gemv}.py` | Byte-neutral successor to rejected T16-lite/X16. It preserves T16-local d/dmin and the 2,048-byte `[subblock,lane32,colpair]` Q payload, but packs scale/min as 64 independent `[kind,subblock,column_quartet]` 24-bit records. Each record holds four exact 6-bit coefficients; all 128 work items cooperatively expand gate/up records into LDS while retaining the direct T16 K/reduction/BF16 order. Host roundtrip and coefficient oracles plus the full adjacent X8 kernel file report **21 passed**. Balanced actual K3072/N1024 c1/c2/c4/c8 timing improves T16 **4.929%/0.781%/3.691%/4.633%** with zero BF16 mismatches and reduces pair residency **2.778%**. Cached kernel tracing confirms the expected symbol at local128/VGPR192/SGPR128/LDS1536B/scratch0. Retained exact decode primitive only until selected prefill and resident materialization/runtime gates pass. Artifact: `benchmarks/results/2026-07-26-gfx1151-laguna-q4-k-qmicro-exact-decode-retained.json`. |
| Laguna Q4T16 dual-interleaved selected-decode diagnostic | exact `[expert,out_tile16,k_block,4736]` gate/up pair plus `selected_dual_interleaved_natural_tile8_parallel_silu` direct wrapper; no registry/runtime route | `hipengine/quant/gguf_q4_k.py`, `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `scripts/laguna_selected_natural_decode_leaf.py`, `tests/test_gguf_{q4_k_tile16_repack,t16_selected_gemv_decode}.py` | Corresponding gate/up `d/dmin`, coefficient, and Q vectors are physically adjacent while total bytes equal two ordinary T16 matrices. Both inputs round-trip byte-for-byte and the natural Laguna fused-SiLU output is BF16-bit exact. Extracted gfx1151 ISA uses one `global_load_b64` for both Q streams; cached tracing reports local128/VGPR80/SGPR128/LDS512/scratch0. Actual layer-1 21x100 timing improves **0.122684 -> 0.115685 ms (-5.705%, 21/21 wins)**. Production promotion is blocked because the retained prefill D8 MMQ128x32 owner shares these resident bytes; a sidecar would add about **43.76 GB** across 47 layers. Keep diagnostic only until paired materialization and the prefill consumer pass an independent byte-neutral gate. Artifact: `benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-dual-interleaved-blocked.json`. |
| Laguna Q4T16 dual-interleaved dense/shared decode | `linear_pair_silu/gguf_q4_k/t16_dual_interleaved_sidecar_decode_bf16_bf16_out`; exact pack8 rollback | `hipengine/quant/gguf_q4_k.py`, `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.{hip,py}`, `hipengine/kernels/hip_gfx1151/__init__.py`, `hipengine/loading/laguna_gguf_materialize.py`, `hipengine/runtime/{gguf_linear,laguna_moe,laguna_gguf_runner}.py`, `tests/test_{gguf_t16_selected_gemv_decode,gguf_linear_dispatch,laguna_gguf_materialize_device,gfx1151_backend}.py` | gfx1151 replaces the 96 separate dense/shared gate/up decode-only sidecars with 48 exact `[1,out_tile16,k_block,4736]` paired payloads. Residency stays **214,597,632 bytes** and pack8 prefill is unchanged. The retained tile2/local32 owner issues one paired Q load, lowers allocated VGPR **176 -> 72**, and improves the actual one-dense-plus-47-shared leaf ledger **0.672947 -> 0.629207 ms (-6.500%)** with zero BF16 mismatches. Cached trace: local32/VGPR72/SGPR128/LDS0/scratch0. Seven resident model pairs improve **21.898558 -> 21.954474 tok/s (+0.25534%)**; selector-unset production is **21.942208 tok/s** with exact state and unchanged **79,022,520,340-byte** total residency. Artifact: `benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-dense-dual-interleaved-retained.json`. |
| Laguna Q4T16 shared-down decode | `linear/gguf_q4_k_t16_v1/dense_single_local32_bf16_bf16_out`; `linear+moe_tail+next_rmsnorm_host_batch` sibling; exact expanded-pack8 rollback | `hipengine/kernels/hip_gfx1100/{quant/gguf_t16_selected_gemv,runtime/laguna_launch_batch}.{hip,py}`, `hipengine/loading/laguna_gguf_materialize.py`, `hipengine/runtime/{gguf_linear,laguna_moe,laguna_gguf_runner}.py`, `scripts/laguna_q4_t16_dense_single_decode_leaf.py`, `tests/test_{gguf_t16_selected_gemv_decode,laguna_native_launch_batch,gguf_linear_dispatch,laguna_gguf_materialize_device}.py` | The local32 owner preserves expanded-pack8's eight-contiguous-K lane ownership, FP32 coefficient products, per-output FMA sequence, wave32 tree, and BF16 store. Across all 24 actual Q4 shared-down tensors, the 9x50 family ledger improves **0.171187 -> 0.129045 ms/token (-24.618%)**, every per-weight median is positive, **215/216** timed pairs win, and all **73,728** BF16 outputs match pack8 exactly. Each matrix streams **1,818,624 vs 2,359,296 bytes (-22.917%)**. Cached gfx1151 tracing names `q4_k_t16_dense_single_local32_gemv_kernel<unsigned short>` at grid384/local32, VGPR96/SGPR128/LDS0/scratch0. The production owner attaches 24 decode-only sidecars and preserves the retained native shared-down→D9 host boundary. All seven same-resident p512/d128 candidates win **22.377298 -> 22.563488 tok/s (+0.83205%)**; clean production reaches **22.555437 tok/s** with exact state/lifecycle and **43,646,976** added resident bytes. Tracing proves 24 T16/zero pack8 calls per token and cuts dense/shared **15.708%** plus complete kernel sum **0.353192 ms/token**. Artifacts: `benchmarks/results/2026-07-31-gfx1151-laguna-q4-t16-shared-down-{leaf,retained,production}.json`, `benchmarks/results/2026-07-31-gfx1151-laguna-post-q4-t16-shared-down-wall-reprofile.json`. |
| `linear` diagnostic variants `selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out` | `gguf_q5_k`, `gguf_q6_k` raw selected MoE down weights with q8_1 activation quantization | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.{hip,py}` | `gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out(...)`, `gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out(...)` | Bounded raw-GGUF selected-down POC for the llama.cpp/GGML vector-dot recipe. Runtime routing is default-off under `HIPENGINE_GGUF_RAW_SELECTED_DP4A=1` and is only for the no-decode-repack diagnostic bundle. Validation: `HIPENGINE_HIP_ARCH=gfx1151 python3 -m pytest tests/test_gguf_k_selected_pack8_dp4a_gemv.py -q` -> `3 passed`; routing bundle `tests/test_qwen35_gguf_compact_moe_gemv_routing.py tests/test_qwen35_gguf_compact_moe_wmma_routing.py -q` -> `18 passed`. Microbench on gfx1151 selected-down shape (`rows=8`, `E=256`, `in=512`, `out=2048`) measured Q5 raw `0.0916 ms` vs q8_1 quantize+dp4a `0.0395 ms` (**2.32x**, `KL_mean=0.00011`, top-1 `1.0`) and Q6 raw `0.0419 ms` vs `0.0259 ms` (**1.62x**, `KL_mean=0.00512`, top-1 `1.0`), artifact `benchmarks/results/2026-06-27-hipengine-gguf-raw-q5-q6-selected-pack8-dp4a-poc.json`. Cached `rocprofv3 --kernel-trace` confirms `gguf_k_selected_pack8_q8_1_dp4a_prefill_out_kernel<unsigned short,5/6>` launches with avg Q5/Q6 dot durations `~44.7 us`/`~19.5 us` and q8_1 quantization `~2.1 us`; `llvm-objdump --offloading` crashed on this ROCm/LLVM build, so retained ISA proof is the source `__builtin_amdgcn_sudot4` plus profiler-visible kernel dispatch. B3 no-decode-repack improves `31.63 -> 39.61 tok/s` exact (`15/15`) but remains below default decode-repack `51.31 tok/s`, so this is diagnostic only. |
| `linear` diagnostic variants `selected_silu_gemv_bf16_bf16_out` | `gguf_q5_k`, `gguf_q6_k` raw selected MoE down weights with BF16 gate/up inputs | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.{hip,py}` | `gguf_q5_k_selected_silu_gemv_bf16_bf16_out(...)`, `gguf_q6_k_selected_silu_gemv_bf16_bf16_out(...)` | Default-off resident MTP draft diagnostic behind `HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` / `--resident-mtp-draft-selected-silu-down-fused`. It fuses selected MoE `silu(gate) * up` into the Q5_K selected-down GEMV while BF16-rounding the intermediate to match `silu_mul_separate_out_bf16 + gguf_q5_k_selected_gemv_bf16_bf16_out`. Validation: `HIPENGINE_HIP_ARCH=gfx1151 python3 -m pytest tests/test_gguf_k_selected_pack8_dp4a_gemv.py::test_selected_pack8_dp4a_registry_and_contract tests/test_gguf_k_selected_pack8_dp4a_gemv.py::test_q5_selected_silu_down_fused_matches_unfused_chain -q` -> passed. The 2026-07-02 active llama-compat draft profile rejected it before full-suite: kernel calls/step `90.75 -> 88.5`, but kernel time `5.973 -> 6.054 ms/cycle`, host wall `7.044 -> 7.206 ms/cycle`, and selected-down family `0.325 -> 0.391 ms/cycle`. Artifacts: `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-siludown-control-fine-sync.json` and `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1-routerrow-siludown-fine-sync.json`. |
| `linear` variants `pack8_*_f32_out`, `pack8_*_fp16_out`, `pack8_bf16_bf16_out`, `pack8_prefill_*_f32_out`, `pack8_prefill_*_fp16_out`, `pack8_prefill_bf16_bf16_out` | `gguf_q4_k` lossless pack8 repack | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_gemv.hip` | `gguf_q4_k_pack8_gemv_*_out(...)`, `gguf_q4_k_pack8_prefill_*_out(...)` | `python3 scripts/gguf_prefill_projection_smoke.py --rows 4 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → Q4_K pack8 BF16→F32/FP16/BF16 rows>1 smoke passes vs CPU pack8 reference (`max_abs=0.0`, bit mismatches `0`); compatibility `python3 scripts/smoke.py --mode gguf-q4-k-pack8-gemv-hip --rows 4 --hidden-size 512 ... --require-cached-build` also passes. `rocprofv3 --kernel-trace` shows `gguf_q4_k_pack8_prefill_out_kernel<unsigned short,{float,_Float16,unsigned short}>` for the synthetic smoke and six `gguf_q4_k_pack8_prefill_out_kernel<unsigned short,unsigned short>` launches with `Grid_Size_Y=4` inside the Qwen3.5-0.8B GGUF rows=4 layer prefill profile. |
| `linear` variants `gemv_*_f32_out`, `gemv_*_fp16_out`, `gemv_bf16_bf16_out`, `prefill_*_f32_out`, `prefill_*_fp16_out`, `prefill_bf16_bf16_out` | `gguf_q8_0`, `gguf_q5_k`, `gguf_q6_k` raw GGUF weights | `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip` | `gguf_q8_0_gemv_*_out(...)`, `gguf_q8_0_prefill_*_out(...)`, same for `q5_k`/`q6_k` | `python3 scripts/gguf_prefill_projection_smoke.py --rows 4 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → synthetic Q8_0/Q5_K/Q6_K rows>1 BF16→F32/FP16/BF16 fixtures pass vs CPU reference (`worst_max_abs=0.0`, all lowp bit mismatches `0`); compatibility `python3 scripts/gguf_k_gemv_smoke.py --rows 4 --out-features 7 ... --require-cached-build` also passes. `rocprofv3 --kernel-trace` shows `gguf_k_prefill_out_kernel<unsigned short,{float,_Float16,unsigned short},8/5/6>` in the synthetic smoke and one raw Q6_K prefill projection in the Qwen3.5 GGUF rows=4 layer prefill profile. Historical `/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf` one-row BF16-output smokes vs CPU reference remain as prior evidence. **2026-07-20 exact Q8 pack8 prefill:** package dispatch now selects the existing `gguf_k_pack8_prefill_out_kernel<unsigned short,unsigned short,8>` for rows>1 raw-Q8 BF16 outputs divisible by eight when WMMA is off. Two multirow primitive shapes are BF16-bit equal to the one-output kernel; Q3 mixed-64/all-row gates remain exact. GPU1 512/4K prefill moves `218.598/211.936 -> 364.414/342.902 tok/s`; cached 4K Q8 time moves `13,687.886 -> 5,958.328 ms` (-56.47%), total kernel sum falls 38.18%, and the retained symbol is local128, VGPR 72, scratch 0. Artifact: `benchmarks/results/2026-07-20-gpu1-q3-exact-q8-pack8-prefill.json`. **2026-07-20 exact Q8 row reuse:** new four-axis variants `exact_prefill_tile8x2_bf16_bf16_out` and `exact_prefill_tile8x4_bf16_bf16_out` keep each thread's `k=tid+128*n` traversal, wave32 shuffle tree, and wave-0..3 sum while interleaving independent rows/columns. Package policy keeps pack8 below 8 rows, uses 8x2 for narrow rows 8–31, and otherwise uses 8x4; the measured inferior 4x4 candidate was removed. Both retained tiles have zero BF16 mismatches vs pack8; local128 resources are 48/72 VGPR, 512 B LDS, scratch 0. GPU1 512/mixed-4K moves pack8 `364.414/342.902 -> 573.288/523.321 tok/s`; cached Q8 `5,958.328 -> 2,216.705 ms` (-62.80%) and total kernel sum falls 34.81%. Artifact: `benchmarks/results/2026-07-20-gpu1-q3-exact-q8-row-reuse-prefill.json`. **2026-07-20 exact Q8 column reuse:** variant `exact_prefill_tile16x4_bf16_bf16_out` doubles independent output columns while retaining the 8x4 kernel's four-row reuse and exact K/reduction association. Measured row thresholds are 512/64/32 for output widths 512/2048/8192; smaller or 16-unaligned shapes retain 8x4/8x2/pack8. Production leaves improve 2.80–10.10% with zero BF16 mismatches; exact 8x8 regresses 0.81–31.34% and was removed. GPU1 512/mixed-4K moves `693.325/613.576 -> 707.420/626.077 tok/s`; cached Q8 falls `2,161.039 -> 2,056.867 ms` (-4.82%) and total kernel sum falls 1.84% with unchanged launches. The retained local128 symbol uses VGPR136, 1 KiB LDS, and zero scratch. Artifact: `benchmarks/results/2026-07-20-gpu1-q3-exact-q8-tile16x4-prefill.json`. |
| `activation_quant` variant `q8_1_d4x3/bf16`; `linear` variant `mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out`; `linear_prefill_policy` variant `raw_q8_mmq128` | `gguf_ud_q3_k_m` model policy over raw `gguf_q8_0` dense weights | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_mmq_prefill.{hip,py}` | `gguf_q8_0_mmq128_quantize_bf16_d4x3(...)`, `gguf_q8_0_mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out(...)`, `gguf_q8_0_mmq128_sparse_exact_correct_bf16(...)` | Source dataflow is pinned to clean llama.cpp HIP `1ebf790cda38` (`ggml-cuda/{quantize.cu,mmq.cuh,mma.cuh}`): a 256-thread workgroup stages raw output-major Q8_0 into a padded 128-row tile and computes a 128-output x 128-token K256 integer-WMMA tile. hipEngine adds three residual D4 activation planes, emits BF16, queues outputs within `1e-5` of a BF16 rounding boundary, and recomputes only that bounded queue with the retained exact 128-thread raw-Q8 reduction. The quant-axis policy admits only measured winning `(K,N)` shapes `(2048,8192)` from 32 rows and `(2048,4096)` from 48 rows through 4,096; every other shape keeps the exact 16x4/8x4/8x2/pack8 fallback. D4x3 reuses existing bulk scratch; the only persistent addition is a 4-byte count plus `rows*8192*4` queue (`16/128 MiB` at 512/4K). The final 18-workload x 9-position text/decode gate is logit-bit-exact (`KL=0`, top-1 `1.0`); all-queued primitive repair is BF16-bit exact. Cache-only GPU1 trace executes 250 each of quantizer/MMQ/correction; MMQ is VGPR216, dynamic LDS 57,856 B, scratch 0, while quantizer/correction are VGPR24/scratch0. Post-hardening exact matched 512/4K wall improves `760.411 -> 837.417` and `743.906 -> 831.393 tok/s`; official retained medians are `848.543/828.003 tok/s`, tracked peak `15.821/17.080 GiB`. Artifact: `benchmarks/results/2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json`. |
| `linear` variants `wmma_prefill_{bf16,fp16,f32}_{bf16,fp16,f32}_out` (P8.1) | `gguf_q8_0` raw GGUF weights, batched prefill GEMM | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_prefill.hip` | `gguf_q8_0_wmma_prefill_{bf16,fp16,f32}_{bf16,fp16,f32}_out(...)` | Real batched GEMM mirroring `awq_fusedw4_prefill_fp16_kernel`: 32 threads/block, grid `((out_features + tile_m - 1) / tile_m, (rows + tile_n - 1) / tile_n)`, `__launch_bounds__(32, 8)`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` accumulator, Q8_0 dequant in the K-loop (one 34-byte block = 1 fp16 `d` + 32 int8 `qs` → 2 K-tiles of 16). Supported tiles: (16,16), (16,32), (32,16), (32,32), (64,16), (64,32). Inline smoke vs CPU `gguf_q8_0_gemv` across rows ∈ {8,16,17,32,33,48,64}, in_features ∈ {64,96,128,192,256,512}, out_features ∈ {16,24,32,48,64,80,128} → bf16→f32 `max|d|<1.2e-6`, f32→f32 `2.4e-7`, bf16→bf16 `max_rel<4e-3` (one bf16 ULP). Replaces the decode-shaped `gguf_q8_0_prefill_*` aliases when wired through `hipengine.runtime.gguf_linear` (task P8.6); no perf row retained yet. See docs/GGUF.md "P8: real batched prefill GEMM". |
| `linear` variants `pack8_gemv_decode_{bf16,fp16}_{bf16,fp16}_out` and `pack8_dual_gate_up_gemv_decode_{bf16,fp16}_{bf16,fp16}_out` (P9.B3) | `gguf_q8_0` raw GGUF weights, dense decode-shaped pack8 GEMV (single + fused gate+up dual) | `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_pack8_gemv.hip` | `gguf_q8_0_pack8_gemv_decode_{bf16,fp16}_{bf16,fp16}_out(...)`, `gguf_q8_0_pack8_dual_gate_up_gemv_decode_{bf16,fp16}_{bf16,fp16}_out(...)` | Mirrors PARO `gemv_awq_pack8_kernel` (single) and `gemv_awq_dual_pack8_kernel` (concatenated gate+up): `__launch_bounds__(128, 4)` 4 wave32 waves per block, 8-K-per-thread `vec_stride = blockDim.x * 8` outer loop with per-iteration ``d`` hoist (Q8_0 block is 32 K's, so the 8 inner `j` lanes always share one block), wave-level `__shfl_down` reduce + cross-wave `xchg[4*8]` sum. Inner k swap: AWQ pack8 dequant -> raw Q8_0 (`d * int8`). Dual output layout: row-major `[rows, out_features_a + out_features_b]` with gate in `[0, out_features_a)` and up in `[out_features_a, out_features_total)`; matches what `silu_mul_dual_out_*` consumes. Constraints: `in_features % 32 == 0`, `out_features (or each of A/B) % 8 == 0`. Inline GPU smoke vs CPU `gguf_quant_gemv(..., Q8_0)`: BF16 single (rows=4, in=512, out=24) max|d|=`0.029`, max_rel(eps=1)=`0.0036`; BF16 dual (rows=4, in=512, out_a=16, out_b=32) max|d|=`0.058`, max_rel=`0.0035`; FP16 dual same shape max|d|=`0.006`, max_rel=`0.0005`. No new ABI and no resident weight sidecar/repack. Formal correctness fixture (multi-row, attention-shape, shared-expert-shape) lands in task #24 (P9.B5); runtime wiring in task #25 (P9.B6). |
| `linear` variants `wmma_prefill_{bf16,fp16,f32}_{bf16,fp16,f32}_out`, `wmma_prefill_dual_bf16_bf16_out` (P8.2) | `gguf_q4_k` raw GGUF block-q4_K weights, batched prefill GEMM | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_prefill.hip` | `gguf_q4_k_wmma_prefill_{bf16,fp16,f32}_{bf16,fp16,f32}_out(...)`, `gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(...)` | Real batched GEMM mirroring `awq_fusedw4_prefill_fp16_kernel` and `awq_fusedw4_prefill_dual_fp16_kernel`: 32 threads/block, grid `((out_features + tile_m - 1) / tile_m, (rows + tile_n - 1) / tile_n)`, `__launch_bounds__(32, 4)`, `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` accumulator, raw Q4_K dequant in the K-loop (`block_q4_K` 144 B = fp16 `d`, fp16 `dmin`, 12 packed scale/min bytes, 128 q4 bytes; one 256-K superblock -> 8 subblocks of 32 -> two 16-wide WMMA K-tiles per subblock). Supported tiles: (16,16), (16,32), (32,16), (32,32), (64,16), (64,32). Wrapper import/registry smoke passes; HIP source compiles to `gguf_q4_k_prefill.so` on W7900. Inline synthetic smoke vs CPU `gguf_q4_k_gemv`: f32→f32 matches the expected fp16-rounded WMMA operand reference within `max|d|=6.4e-4` (vs exact CPU raw-Q4_K GEMV `max|d|=0.299`, explained by the intentional half-operand WMMA cast); BF16 dual gate/up launches and both outputs match CPU within one BF16 output ULP budget (`max|d|=0.776` on values up to ~265). Dispatch/materialization remains opt-in/follow-up because current dense 2D Q4_K materialization uses lossless pack8 and drops raw bytes; rows==1 and pack8 fallback kernels are unchanged. |
| `embedding` variant `lookup_bf16_out` | `gguf_q5_k`, `gguf_q6_k`, `gguf_q8_0` raw GGUF token embedding rows; BF16 dense fallback rows | `hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_embedding.hip`, `hipengine/kernels/hip_gfx1100/runtime/state.hip` | `gguf_q5_k_embedding_bf16_out(...)`, `gguf_q6_k_embedding_bf16_out(...)`, `gguf_q8_0_embedding_bf16_out(...)`, `embedding_lookup_bf16_i64(...)` | `python3 scripts/gguf_q6_k_embedding_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt` → synthetic Q6_K, real Q6_K, and real Q8_0 selected-token embedding smokes pass vs CPU reference/dequant rounded to BF16 (`worst_max_abs=0.0`). `rocprofv3 --kernel-trace --output-directory /tmp/hipengine-gguf-q8-embed-rocprof-task49 --output-file q8-embed --output-format csv -- python3 scripts/gguf_q6_k_embedding_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` confirms `gguf_q8_0_embedding_bf16_out_kernel` ran with `DurationNs=8120`, `Scratch_Size=0`, `Workgroup_Size_X=256`, `Grid_Size=(1024,4,1)`, plus Q6_K embedding launches at `DurationNs=5400/8160`. The GGUF decode graph path consumes device-resident lm-head argmax token IDs through these embedding dispatch keys; `benchmarks/results/2026-05-17-hipengine-gguf-decode-graph-replay-diagnostic.json` records six Q6_K embedding launches for prompt length 3 + three graph replays, and the local-quant diagnostic records Q8_0 E2E coverage. |
| `w8a16_linear` variants `bf16_f32_out`, `bf16_lowp_out`, `fp16_lowp_out`, `shared_gate_up_silu_fp16`, `shared_gate_up_silu_fp16_token_tiled`, `shared_gate_sigmoid_fp32`, `shared_down_combine_residual_fp16`, `shared_down_combine_residual_fp16_token_tiled`, `f32_f32_out` | `w8a16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip` | `w8a16_linear_bf16_f32_out(...)`, `w8a16_linear_bf16_lowp_out(...)`, `w8a16_linear_fp16_lowp_out(...)`, `w8a16_shared_gate_up_silu_fp16(...)`, `w8a16_shared_gate_up_silu_fp16_token_tiled(...)`, `w8a16_shared_gate_sigmoid_fp32(...)`, `w8a16_shared_down_combine_residual_fp16(...)`, `w8a16_shared_down_combine_residual_fp16_token_tiled(...)`, `w8a16_linear_f32_f32_out(...)` | `python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `bf16_f32_max_abs=0.0`, `f32_f32_max_abs=4.77e-07`, `lowp_mismatch=0`, `fp16_lowp_mismatch=0`; fused shared route fixture gate passes (`max_kl=0.03406`, top-1 `1.0`); P1.2 token-tiled gate/up and P1.3 token-tiled down+combine microchecks match the original kernels (`tile2/tile4 max_abs=0`) and fixture gates pass (`max_kl=0.0396`, top-1 `1.0`); `rocprofv3 --kernel-trace` confirms `w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<2>` and `w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<2>` for legacy prompts; previous all-layer 512 prefill shows FP16 `w8a16_shared_down_combine_residual_fp16_kernel` ran 40 times (`16.047 ms` total, avg `401.166 us`, 8-row tile), `shared_gate_sigmoid_fp32_kernel` ran 40 times (`0.092 ms` total), and `w8a16_shared_gate_up_silu_fp16_kernel` ran 40 times (`15.562 ms` total) on W7900 |
| `paro_rotate1`, `paro_rotate2`, `paro_rotate3` variants `bf16`, `fp16`; `paro_rotate1` variant `bf16_gate_fp16` | `w4_paro` | `hipengine/kernels/hip_gfx1100/rotary/paro_rotate.hip` | `paro_rotate1_bf16(...)`, `paro_rotate2_bf16(...)`, `paro_rotate3_bf16(...)`, `paro_rotate1_fp16(...)`, `paro_rotate2_fp16(...)`, `paro_rotate3_fp16(...)`, `paro_rotate1_bf16_gate_fp16(...)` | `python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 rotate2/3 (`mismatches=[0, 0, 0, 0, 0]`) and FP16 rotate1/2/3 (`fp16_mismatches=[0, 0, 0, 0, 0, 0]`, `fp16_max_abs=0.0`); AOTriton gate-rotate fixture passes (`max_kl=0.0396`, top-1 100%); `rocprofv3` shows FP16 `paro_rotate1_kernel<_Float16>` (`DurationNs=11680`, `Scratch_Size=0`, `LDS_Block_Size=32`), `paro_rotate2_kernel<_Float16>` (`DurationNs=2680`), and `paro_rotate3_kernel<_Float16>` (`DurationNs=2560`) on W7900 |
| `partial_rotary`, `head_rmsnorm+partial_rotary`, `split_qgate` variants `qwen35_f32`, `qwen35_f32_bf16`, `qwen35_position_f32_bf16`, `qwen35_positions_f32_bf16`, `qwen35_positions_q_bf16_key_f32`, `bf16`, `fp16` | `w4_paro` full-attention prelude plus resident q/gate split | `hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.hip` | `qwen35_partial_rotary_f32(...)`, `qwen35_head_rmsnorm_partial_rotary*_f32_bf16(...)`, `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16(...)`, `qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32(...)`, `qwen35_split_qgate_{bf16,fp16}(...)` | `python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → `partial_max_abs=0`, `head_max_abs=2.38e-07`, `position_max_abs=2.38e-07`, `vector_position_max_abs=2.38e-07`, `split_fp16_query_max_abs=0`, `split_fp16_gate_mismatch=0`; AOTriton cast-glue fixture gate passes with `max_kl=0.0396`, top-1 `100%` after using the BF16-Q/FP32-K vector-position variant; prior `rocprofv3` shows scalar parent kernels with `Scratch_Size=0` plus FP16 `qwen35_split_qgate_fp16_kernel` (`DurationNs=3720`) on W7900 |
| `cast_f32_to_bf16`, `cast_bf16_to_f32`, `cast_f32_to_fp16`, `cast_fp16_to_f32`, `cast_fp16_to_bf16`, `cast_bf16_to_fp16` variants `fp16`/`scaled_rows`, `cast_f32_scale_rows` | `bf16`, `fp16`, `fp32`, `scaled_rows` runtime glue | `hipengine/kernels/hip_gfx1100/convert/cast.hip` | `f32_to_bf16(...)`, `bf16_to_f32(...)`, `f32_to_fp16(...)`, `fp16_to_f32(...)`, `fp16_to_bf16(...)`, `bf16_to_fp16(...)`, `bf16_to_fp16_scaled_rows(...)`, `f32_scale_rows(...)`, `f32_scale_rows_to_bf16(...)` | The general Laguna library-projection boundary computes one exact power-of-two scale per BF16 row, emits finite FP16, and restores the scale either in-place on FP32 output or while rounding FP32 to BF16. The gfx1151 default uses direct cast only at the attention-RMSNorm boundary: actual max norm weight **0.294921875** proves output bound **16.34623**, so scale is exactly one. Exact pp512 glue improves **4.434 -> 0.767 ms (-82.70%)** and full-model A/B improves **502.348 -> 505.887 tok/s (+0.704%)** with exact logits/hidden/KV/cursor. Cached trace names `bf16_to_fp16_kernel` at local256, VGPR8, SGPR128, zero LDS/scratch. |
| `token_embedding`, `decode_position`, `scalar_state`, `prefill_metadata`, `decode_metadata`, `decode_graph_commit`, `decode_graph_record`, `prefill_flight_recorder` runtime helpers | `w4_paro`, GGUF graph-friendly state, `gguf_qwen35` contiguous prefill chunks and packed c4/c8 decode metadata, and mapped-host diagnostics | `hipengine/kernels/hip_gfx1100/runtime/state.hip` | `embedding_lookup_{bf16,fp16}_i64(...)`, `embedding_lookup_batch_{bf16,fp16}_i64(...)`, `embedding_lookup_batch_mapped_{bf16,fp16}_i64(...)`, `set_i64_scalar(...)`, `set_i64_vector(...)`, `set_decode_position_i64(...)`, `set_decode_positions_i64(...)`, `prepare_prefill_chunk_metadata(...)`, `prepare_packed_decode_metadata(...)`, `prepare_packed_decode_metadata_from_positions(...)`, `commit_packed_decode_graph_step(...)`, `record_u16_rows_indexed(...)`, `flight_recorder_mark_i64(...)`, `advance_decode_position_i64(...)`, `advance_decode_positions_i64(...)`, `record_i64_scalar_indexed(...)` | `python3 -m pytest tests/test_runtime_state_plan.py tests/test_runtime_state_unpack_metadata.py -q`; GPU smokes cover embedding/state helpers and exact contiguous chunk metadata at the final 128K chunk. The default-off `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=1` candidate replaces six synchronous H2D copies with one stream-ordered kernel; its gfx1151 trace is **6.612 us**, 256 threads, 16 VGPR, zero LDS/scratch. Clean same-session 512/4K five-pair medians improve **1225.203 -> 1243.183** and **1273.720 -> 1282.003 tok/s**, with all 83 token/logit/hidden/Conv/GDN/KV parts exact. The 128K 1+3 median is **499.636 tok/s**, but a 468.801 outlier triggers escalation; its 1+5 replacement reproduces the separately tracked low-power GPU-active lifecycle state during measured pass 1, so the candidate remains default-off. GGUF graph replay also reuses `set_i64_scalar`, `set_decode_position_i64`, `advance_decode_position_i64`, and `record_i64_scalar_indexed`. C3 adds `decode_metadata / gguf_qwen35 / packed_c4_i64`: the ragged `[513,517,521,525]` analytic fixture is byte-exact for positions/contexts, disjoint block rows, singleton GDN segments, identity state indices, and reset scalars. A cached-only HIP 7.15 W7900 trace records `prepare_packed_decode_metadata_kernel` at **4.120 us**, 64 threads, 16 VGPR, and zero scratch/LDS; real p512/c4/d4 keeps **800/800** layer-hidden rows and all state/KV bytes exact while the manifest reports zero metadata H2D copies. The prefill candidate is recorded in `benchmarks/results/2026-07-15-gfx1151-gguf-prefill-device-metadata-candidate.json`; the clean C3 decode evidence is in `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c3-model-boundaries-closure.json`. C4 adds registry-resolved device-position metadata, sampled-i32→embedding-i64 commit, and indexed BF16 layer recording for fixed-width graph replay. Their cached HIP 7.15 W7900 traces are **2.360/2.720/2.080 us**, respectively, with 16/8/8 VGPR and zero scratch/LDS. Real p512 graph replay is exact for four steady c4 launches (**800/800** layer rows) and sparse c4→c3→c2→c1 recapture (**560/560**); this is correctness evidence, not a performance claim. E2 adds explicit `packed_c8_device_positions_i64` and `packed_c8_i32_i64` aliases while preserving the c4 aliases and c4-only host-scalar metadata helper. The eight-row two-step control fixture is byte-exact; cached HIP 7.15 W7900 traces show metadata at **4.520/2.400 us** (64 threads, 16 VGPR) and commit at **3.161/3.360 us** (one thread, 8 VGPR), all with zero scratch/LDS. A real p16/c8/d2 graph captures one physical c8 bucket, replays twice, and passes **960/960** all-layer comparisons plus exact tokens/Conv/GDN/live-KV with no steady metadata copies or c1/model-row fallback. Masked E2 adds a device `active_mask` pointer to both control helpers: metadata preserves `-1` block rows/zero contexts and commit skips token feedback, recording, and cursor advance for inactive lanes. The eight-row two-step masked control is byte-exact; cached-only HIP 7.15 W7900 profiling records metadata/commit at **5.560/1.720 us**, 64/1 threads, 16/8 VGPR, and zero scratch/LDS. A real fixed-physical-c8 graph sequence c8→c6→c4→c2→c1 passes **1,160/1,160** all-layer comparisons plus exact tokens and every active/retired session's Conv/GDN/live-KV state. This is partial E2 correctness evidence, not E2 closure or a performance claim. |
| `linear_attn_conv_decode` variants `f32`, `bf16`, `fp16` plus `gguf_qwen35/bf16_indexed`; `linear_attn_tree_conv_decode` variants `bf16_tloop`, `fp16_tloop`; `linear_attn_conv_prefill` variants `f32`, `f32_segments`; `gguf_qwen35` variants `f32_baseline`, `f32_tile32x128` | `w4_paro` linear-attention decode/prefill plus DFlash parent-indexed tree verify; `gguf_qwen35` exact long-token prefill and sparse batch decode | `hipengine/kernels/hip_gfx1100/linear_attn/conv.hip` | `qwen35_linear_attn_conv_decode_f32(...)`, `qwen35_linear_attn_conv_decode_bf16(...)`, `qwen35_linear_attn_conv_decode_fp16(...)`, `qwen35_linear_attn_conv_decode_indexed_bf16(...)`, `qwen35_linear_attn_tree_conv_decode_bf16_tloop(...)`, `qwen35_linear_attn_tree_conv_decode_fp16_tloop(...)`, `qwen35_linear_attn_conv_prefill_f32(...)`, `qwen35_linear_attn_conv_prefill_f32_tile32x128(...)`, `qwen35_linear_attn_conv_prefill_segments_f32(...)` | Decode: `python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `f32_out_max_abs=7.45e-09`, `bf16_out_max_abs=7.45e-09`, `fp16_out_max_abs=7.45e-09`, state max abs `0`; `rocprofv3` shows FP16 `qwen35_linear_attn_conv_decode_lowp_kernel<_Float16>` (`DurationNs=5680`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`) on W7900. Prefill: `qwen35-linear-attn-prefill-hip` → `conv_out_max_abs=1.49e-08`, `conv_state_max_abs=0`. Segment prefill: `qwen35-linear-attn-segments-hip` → `segment_conv_out_max_abs=1.86e-09`, `segment_conv_state_max_abs=0`; `rocprofv3` shows `qwen35_linear_attn_conv_prefill_segments_kernel` (`DurationNs=5800`) and segment state kernel (`2200`) on W7900. Tree DFlash smoke: `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/qwen35_linear_attn_tree_tloop_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → N=2/4/8 BF16+FP16 `max_abs=4.77e-07`; `rocprofv3 --kernel-trace` on gfx1151 shows `qwen35_linear_attn_tree_conv_decode_lowp_tloop_kernel<{unsigned short,_Float16}>` count=3 each, `DurationNs≈2966–5851`, `Scratch_Size=0`. LCP-1 adds the gfx1151-default exact `f32_tile32x128` schedule: the six-length 257-channel primitive gate and 512/4K 82-part state gate are byte-exact; cached same-stream gfx1151 `rocprofv3` records 120 output launches at **49.790 ms** total, 128 threads, 17.5 KiB LDS, and zero scratch versus **954.134 ms** for the production output body. Clean fresh-process 1+3 focus improves 512 by **1.73%** and 4K by **22.91%** with unchanged memory; gfx1100 retains `f32_baseline`. See `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-lcp1-clean-promotion.json`. Normal FP32 prefill also uses the capture-free `qwen35_linear_attn_conv_prefill_no_state_rows_kernel`: explicit `v_mul_f32_e32` plus sequential `v_add_f32_e32` preserves output/state bytes, removes 20 private bytes/thread on gfx1100, and cuts the cached pp512 body **8.496 -> 1.894 ms / 30 (-77.71%)**; state-row capture, segment, and existing scalar decode bodies are unchanged. C2 adds the sparse `gguf_qwen35/bf16_indexed` decode kernel: a c4 fixture with state indices `[4,1,5,0]` is bit-exact to four scalar BF16 launches, matches the CPU segment oracle within `2e-6`, and preserves inactive slots. Cached-only W7900 `rocprofv3` records one `qwen35_linear_attn_conv_decode_indexed_lowp_kernel<unsigned short>` launch at `2720 ns`, grid-Y 4, 16 VGPR, and zero scratch/LDS. |
| `gdn_recurrent_rmsnorm_gate` variants `bf16_lowp`, `fp16_lowp` plus `gguf_qwen35/bf16_segments`, `gguf_qwen35/bf16_indexed_singleton`; `gdn_tree_recurrent_rmsnorm_gate` variants `bf16_tloop`, `fp16_tloop`; `linear_attn_prefill_prepare` variants `f32_bf16`, `f32_fp16`; `gdn_prefill_recurrent` variants `f32`, `f32_k2`, `f32_k2_segments`; `gdn_prefill_rmsnorm_gate` variants `bf16`, `fp16`; `gdn_prefill_rmsnorm_gate_rotate` variant `fp16` | `w4_paro` linear-attention decode/prefill plus DFlash parent-indexed tree verify | `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` | `qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(...)`, `qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(...)`, `qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(...)`, `qwen35_gdn_recurrent_rmsnorm_gate_indexed_lowp_bf16(...)`, `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_bf16(...)`, `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_fp16(...)`, `qwen35_linear_attn_prefill_prepare_f32_bf16(...)`, `qwen35_linear_attn_prefill_prepare_f32_fp16(...)`, `qwen35_gdn_prefill_recurrent*_f32(...)`, `qwen35_gdn_prefill_recurrent_segments_k2_f32(...)`, `qwen35_gdn_prefill_rmsnorm_gate_{bf16,fp16}(...)`, `qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16(...)` | Decode: `python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → BF16/FP16 `out_max_abs=2.98e-08`, `state_max_abs=1.49e-08`; `rocprofv3` shows FP16 `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<_Float16>` (`DurationNs=9920`, `VGPR_Count=56`, `Scratch_Size=0`, `LDS_Block_Size=1616`). Prefill: `qwen35-linear-attn-prefill-hip` → BF16 `gated_mismatch=0`, FP16 `fp16_gated_mismatch=0`, fused FP16 gate+rotate `fused_rotate_mismatch=0`, `fp16_prepare_max_abs=5.96e-08`; `qwen35-linear-attn-segments-hip` → `segment_gdn_out_max_abs=1.86e-09`, `segment_gdn_state_max_abs=9.31e-10`; `rocprofv3` shows `qwen35_gdn_prefill_recurrent_k2_segments_kernel` (`DurationNs=5480`) on W7900; all-layer 512 prefill after the fixed two-wave k2 reduction specialization shows `qwen35_gdn_prefill_recurrent_k2_kernel` ran 30 times (`40.993 ms` total, avg `1366.4 us`) on W7900. Tree DFlash smoke: `HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/qwen35_linear_attn_tree_tloop_smoke.py --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → N=2/4/8 BF16+FP16 `max_abs=4.77e-07`; `rocprofv3 --kernel-trace` on gfx1151 shows tree GDN recurrent count=3 each, `DurationNs≈6492–18475`, `Scratch_Size=0`, and finalize count=3 each, `DurationNs≈1723–2244`. **P9.A1 (task #17)** registers `gguf_qwen35` aliases (`gdn_prefill_recurrent / f32_k2`, `f32_k2_segments`, `decode_order_bf16`; `linear_attn_prefill_prepare / f32_bf16`; `gdn_prefill_rmsnorm_gate / bf16`) that share the same HIP kernels, and the qwen35 GGUF runner now resolves the prepare + k2 (or `segments_k2` when prefill rows >= `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD`, default 256) + rmsnorm_gate chain via the registry instead of branching on `cfg.is_moe`. Qwen3.6-35B-A3B-UD-Q4_K_M 512/0 prefill bucket on RX 7900 XTX: GDN `666.9 ms / 30` (fused `decode_order_bf16`) -> prepare `2.806 ms / 30` + segments_k2 `52.057 ms / 30` + rmsnorm_gate `1.394 ms / 30` (total `56.3 ms`, ~11.9x reduction); total prefill kernel `907.8 ms -> 297.3 ms` (~3.05x). Dense Qwen3.5-0.8B 512/64 stays regression-free (`3025 vs 3066` prefill tok/s, `193.9 vs 179.0` decode tok/s, finite + deterministic). **P9.A2 (task #18)** adds `tests/test_qwen35_gguf_gdn_prefill_correctness.py` covering all three GDN paths (`decode_order_bf16` vs `prepare + k2 + rmsnorm_gate` vs `prepare + segments_k2 + rmsnorm_gate`) against a CPU oracle assembled from `kernels/cpu_reference/gdn_prefill_recurrent_segments`, plus the registry-alias smoke and the 255/256/257 segment-boundary parametrized cases (7 tests, all pass on W7900 RX 7900 XTX). E2E gate via `scripts/qwen35_gguf_bulk_parity.py` Qwen3.6-35B-A3B-UD-Q4_K_M: the GDN-chain-only path (`native_attention_bulk_ffn`) is **bit-exact** with the row-GEMV serial reference (KL `0.0`, top-1 match token `4469`); the compound `fast_bulk_attention` path remains at the P8-recorded `KL=0.707` (down from `KL=3.892` in task #16) and `top-1 mismatch` due to cumulative compact-MoE WMMA drift, which is owned by task #28/#30, not by the GDN chain. C2 exposes the existing FP32-output segment body for BF16 under exact `gguf_qwen35/bf16_segments` registration. The sparse c4 fixture is bit-exact to four scalar GDN launches, matches the CPU-reference-derived state/output and KL/top-1 gate, and leaves inactive state slots untouched. Cached-only W7900 `rocprofv3` records one `qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel<unsigned short>` launch at `10360 ns`, four segments/two V heads, 80 VGPR, and zero scratch. F3 adds the one-token-per-row `gguf_qwen35/bf16_indexed_singleton` sibling: the sparse c4 fixture is byte-exact to independent scalar GDN output/state, and the gfx1151 package selects it while gfx1100 retains segmented fallback. Cached gfx1151 c8 profiling records 30 indexed launches at **3.706 ms** total, grid Y 32, 128 threads, 56 VGPR, and zero scratch; complete Conv/GDN family time falls from the prior diagnostic **8.230 -> 4.038 ms (-50.94%)**. |
| `linear_attn_prefill_prepare` variant `f32_bf16_raw_scales`; `gdn_prefill_recurrent` variants `f32_decode_order_exact`, `f32_decode_order_exact_segments` | `gguf_qwen35` exact unfused GDN prompt prefill | `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` | `qwen35_linear_attn_prefill_prepare_raw_scales_f32_bf16(...)`, `qwen35_gdn_prefill_recurrent_decode_order_exact_f32(...)`, `qwen35_gdn_prefill_recurrent_decode_order_exact_segments_f32(...)` | SOL-G2 keeps raw Q/K plus separate FP32 normalization scales so the recurrence preserves the fused kernel's contraction order. Synthetic 17-row Qwen-shape non-segment and segment chains are byte-identical to `decode_order_bf16`. The real gfx1151 Qwen3.6-35B-A3B Q4_K_M 17-token greeting is also exact for sampled token, final hidden seed, every layer output, and every Conv/GDN state. Focused/adjacent gate: `48 passed`. A cached-build-only narrow `rocprofv3 --kernel-trace` records raw-scale prepare `30.981 us`, exact recurrence `1.677015 ms`, exact segment recurrence `1.444605 ms`, and RMSNorm gate `8.9 us`; the new recurrent kernels use 40 VGPR and `Scratch_Size=0`. These are execution/plausibility observations, not a performance claim. Current exact segment threshold default is `1025`; the older P9 paragraph's `256` value is historical. |
| UD-Q3_K_M native c=N decode families: `linear_attn_conv_decode/bf16_indexed`, `gdn_recurrent_rmsnorm_gate/bf16_segments`, exact batch paged-attention, native Q6 lm-head rewrite | `gguf_qwen35`, `gguf_ud_q3_k_m` | `hipengine/kernels/hip_gfx1100/linear_attn/{conv,gdn}.hip`, `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`, existing GGUF quant leaves | `qwen35_linear_attn_conv_decode_indexed_bf16(...)`, `qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_bf16(...)`, `qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans(...)`, existing batched GQA split/reduce and row argmax wrappers | Independent rows select persistent Conv/recurrent state with device `state_indices`, consume row-shaped `KVLiveSpans`, and keep the retained selected-row MoE ABI. The dense batch attention body treats block-table entries as global physical blocks; the long-context gated reducer uses contiguous `(head_dim,1)` gate strides. Bulk slot prefill rebases both its local block table and cache pointer, while short token-serial slot prefill retains owner-level caches with physical block ids; variable lengths `[3,4,4,1]` are full-logit exact. `tests/test_qwen35_linear_attn_decode_batch_indexed.py` is bit-exact to independent c=1 and CPU Conv/GDN oracles over sparse state slots; `tests/test_qwen35_gguf_target_rows.py` proves C=2 stateful/full-logit exactness, the C=2 1K split-attention boundary, C=4/8 full logits, variable-short prompts, and reclaim/compact/readmit state+KV exactness. Public greedy prompt lists use stable ids, finish/reclaim, downward state/KV compaction, readmission, and full-shape graph buckets. Retained scaling/profiler evidence: `benchmarks/results/2026-07-21-gpu1-q3-native-cn-retained.json`. The serial `step_rows()` and unfused primitives remain correctness fallbacks; transactional verify commit and persistent cross-call serving are not claimed. |
| UD-Q3_K_M exact fully-bulk prefill plugins: `linear_attn_prefill_prepare/f32_bf16`, `gdn_prefill_recurrent/f32_k2`, `full_attn_prefill/causal_gqa_gate_bf16` | `gguf_ud_q3_k_m` | `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip`, `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_linear_attn_prefill_prepare_decode_order_f32_bf16(...)`, `qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_f32(...)`, `qwen35_paged_full_attn_prefill_gqa_gate_bf16_decode_order_spans(...)`; internal row-batched warp/GQA BF16 split wrappers | Quant-axis plugin preserves the resident c=1 arithmetic contract without a quant/backend dispatch branch. GDN maps 32 independent value columns to one wave, keeps `[128][32]` state in 16 KiB LDS across the token loop, and preserves the ordered 128-term KV reduction, state FMA, and eight-term output grouping; the previous K2 and fused kernels remain registered fallbacks. Full attention uses the 256-thread dense schedule below context 1024, keeps per-Q-head warp split only for a singleton below context 4096, and otherwise selects grouped-GQA from two prefill rows, with 16-row workspace batches. GPU1 gates are bit-exact for mixed-64 full logits, every hidden row at layer limits 0/3/4/40, 1K/4K layer-4 boundaries, and full 4K serial-vs-bulk logits (`KL=0`, top-1 `1.0`). The current cached 4K trace confirms exact LDS32 GDN count 120, dense attention count 10, grouped-GQA count 1930, BF16 batch reduce count 1930, and no warp producer. GDN is local32/VGPR248/LDS16KiB/scratch0 and falls `1,310.186 -> 882.716 ms` (-32.63%); the grouped-GQA prefill crossover then cuts attention `936.900 -> 464.773 ms` (-50.39%) and exact prefill reaches the retained `774.185 tok/s` at 512 and `741.180 tok/s` on mixed 4K. Artifacts: `benchmarks/results/2026-07-20-gpu1-q3-exact-fully-bulk-prefill.json`, `benchmarks/results/2026-07-20-gpu1-q3-exact-gdn-lds32-prefill.json`, and `benchmarks/results/2026-07-20-gpu1-q3-exact-attn-gqa-batch-prefill.json`. Generic `gguf_qwen35` kernels and token-serial/native attention remain the unfused/correctness fallbacks. |
| `paged_kv_write` variants `mixed_bf16_spans`, `mixed_fp16_spans`, `mixed_bf16_batch_spans`, `mixed_fp16_batch_spans`, `mixed_bf16_prompt_spans`, `mixed_fp16_prompt_spans`, `f32_spans` | `w4_paro`, GGUF full-attention KV append, BF16 cache | `hipengine/kernels/hip_gfx1100/attention/paged_kv_write.hip` | `qwen35_write_paged_kv_mixed_value_{bf16,fp16}_spans(...)`, `qwen35_write_paged_kv_mixed_value_{bf16,fp16}_batch_spans(...)`, `qwen35_write_paged_kv_mixed_value_{bf16,fp16}_prompt_spans(...)`, `qwen35_write_paged_kv_f32_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-kv-write-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact KV append (`mixed_mismatch=0/0`, `mixed_fp16_mismatch=0/0`, `f32_mismatch=0/0`, `untouched_nonzero=0`); public wrapper accepts `KVLiveSpans`, where fixed-page `base_offsets` carries the parent block table and `live_counts` carries the position tensor; batched smoke validates row-major c>1 append, `qwen35-paged-attn-prefill-hip` validates single-request FP16 prompt append plus causal prefill attention, and GGUF AOTriton layer prefill validates BF16 prompt append through `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel<unsigned short>` in rocprof. |
| `paged_kv_write` variants `per_token_head_spans`, `int8_key_bf16_value/per_token_head_spans`, `int8_block16/block16_spans` | `int8_per_token_head` full-attention KV append, signed INT8 K/V cache plus separate per-token/per-KV-head K/V scales; diagnostic key-only layout stores signed INT8 K plus BF16 V and K scales only; diagnostic block16 layout stores signed INT8 K/V plus per-token/head/16-dim-block K/V scales | `hipengine/kernels/hip_gfx1100/attention/paged_kv_write.hip` | `qwen35_write_paged_kv_int8_per_token_head_spans(...)`, `qwen35_write_paged_kv_int8_per_token_head_{prompt,batch}_spans(...)`, `qwen35_write_paged_kv_int8_key_bf16_value_spans(...)`, `qwen35_write_paged_kv_int8_key_bf16_value_{prompt,batch}_spans(...)`, `qwen35_write_paged_kv_int8_block16_spans(...)`, `qwen35_write_paged_kv_int8_block16_{prompt,batch}_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-kv-write-int8-hip` → decode c=1 append with FP32/FP16 scale tensors, prompt page-boundary append, and row-major batch append match the NumPy `int8_per_token_head` oracle exactly (`key_mismatch=0`, `value_mismatch=0`, `scale_max_abs=0`, `dequant_max_abs=0`) on W7900. Key-only + block16 primitive gate: `python3 scripts/qwen35_kv_int8_accuracy.py --device hip --contexts 64,520 --block-size 256 --num-q-heads 16 --num-kv-heads 2 --head-dim 256 --scale-dtype fp32 --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build --require-int8-hip --json /tmp/hipengine-block16-hip-64-520.json` → accepted, HIP vs CPU oracle `max_abs <= 3.36e-08` for BF16/per-token/key-only/block16 paths; rocprof key-only trace shows `qwen35_write_paged_kv_int8_key_bf16_value_kernel<float>` ran 64 calls (`174801 ns` total). Wrapper signatures are torch-free raw pointers and validate `KVLiveSpans.scale_metadata`/scale pointers before loading the HIP library. |
| `paged_kv_write` variants `hadamard_group32_spans`, `hadamard_group32_prompt_spans`, `hadamard_group32_batch_spans` | `int8_hadamard_group32` selected-layer K/V append: normalized 32-wide Walsh-Hadamard transform, signed INT8 payload, FP16/FP32 per-token/head/group scales | `hipengine/kernels/hip_gfx1100/attention/paged_kv_write.hip` | `qwen35_write_paged_kv_int8_hadamard_group32_{spans,prompt_spans,batch_spans}(...)` | `python3 -m pytest tests/test_kv_hadamard_group32.py -q` matches the NumPy host-screen representation code-for-code for INT8 payloads and FP16 scales. Cached `rocprofv3 --kernel-trace` on W7900 records `qwen35_write_paged_kv_int8_hadamard_group32_kernel<_Float16>` at `6760 ns`, 16 VGPR, `Scratch_Size=0`; raw-pointer wrappers validate `KVLiveSpans.scale_metadata.granularity="hadamard_group32"`. |
| `full_attn_decode` variant `bf16_context`; `full_attn_gate_mul` variants `bf16`, `fp16` | `w4_paro` short-context full-attention decode, BF16 dense KV cache, lowp gated output | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_full_attn_decode_context_bf16(...)`, `qwen35_full_attn_gate_mul_bf16(...)`, `qwen35_full_attn_gate_mul_fp16(...)` | `python3 scripts/smoke.py --mode qwen35-full-attn-decode-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax oracle `max_abs=1.19e-07`, BF16/FP16 gate outputs bit-exact (`gated_bf16_mismatch=0`, `gated_fp16_mismatch=0`); resident Qwen3.5/PARO uses this dense parent kernel for max context <1024 before the paged path; `rocprofv3` shows FP16 `qwen35_full_attn_gate_mul_fp16_kernel` (`DurationNs=1360`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`) on W7900 |
| `paged_attn_decode` variants `bf16_context_spans`, `bf16_context_batch_spans`, `bf16_context_batch_fixed256_spans`, `bf16_context_batch_c1_exact_spans`, `bf16_context_batch_paged_c1_exact_spans` | `w4_paro` full-attention decode, BF16 KV cache | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_context_bf16_spans(...)`, `qwen35_paged_full_attn_decode_context_bf16_batch_spans(...)`, `qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans(...)`, `qwen35_paged_full_attn_decode_context_bf16_batch_c1_exact_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-decode-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax oracle `max_abs=2.98e-08`; public wrappers accept `KVLiveSpans` (`base_offsets` page table, `live_counts` context tensor); c>1 row-major smoke `batched paged kv+attn smoke OK` validates uneven context lengths; `rocprofv3` shows scalar `qwen35_paged_full_attn_decode_context_tensor_kernel` with `DurationNs=7640`, `VGPR_Count=40`, `Scratch_Size=0`, `Workgroup_Size_X=256` on W7900. PARO G2 registers both batch variants and gives decode a separate absolute physical-slot table from append's row-relative table. The optimized 1,024-thread variant closes the addressing alias but introduces a few-micro-unit FP32 reduction delta from dense c1. `PagedAttnDecodeKind.CONTEXT_BATCH` therefore selects the 256-thread dense-order variant: the 513-token physical-c2 primitive and all L4/d3 stages are bit-exact, while the full L40/d3 token/hidden/Conv/GDN/KV/NumPy-context gate is exact. Clean `32de8d08` HIP 7.15 W7900 profiling records one `qwen35_paged_full_attn_decode_context_tensor_batch_c1_exact_kernel` launch at `190841 ns`, grid Y 2, workgroup X 256, 24 VGPR, and zero reported scratch/LDS. The gfx1151 override now resolves the same logical key to `bf16_context_batch_fixed256_spans`: it keeps the generic reduction used by exact c4/c8 while bypassing only the generic rows<=2 1024-thread geometry. The p512/d128 c2 graph gate passes **10,240/10,240** all-layer comparisons plus exact tokens, Conv/GDN state, and live KV; cached `rocprofv3` records 20 `qwen35_paged_full_attn_decode_context_tensor_batch_kernel` launches at workgroup X 256/grid Y 2, median **231855 ns**, with positive duration. The GGUF-specific `bf16_context_batch_paged_c1_exact_spans` alias selects that fixed-256 paged reduction instead of PARO's dense-c1 reduction: current-system-HIP W7900 p512 sparse c4→c3→c2→c1 passes exact tokens, **560/560** layer outputs, all Conv/GDN state, and all live BF16 KV. |
| `paged_attn_decode` variant `bf16_split_k_spans` | `w4_paro` long-context full-attention decode, BF16 KV cache | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_bf16_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-split-k-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax oracle `max_abs=5.96e-08`; public wrapper runs parent split-K context kernel then reduce using caller-provided workspaces; `rocprofv3` shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel` (`DurationNs=17320`, `VGPR_Count=32`, `Scratch_Size=0`) and `qwen35_paged_full_attn_decode_split_k_reduce_kernel` (`DurationNs=6320`, `VGPR_Count=16`, `Scratch_Size=0`) on W7900 |
| `paged_attn_decode` variant `bf16_split_k_gate_f32_spans` | `w4_paro` long-context full-attention decode + gate, BF16 KV cache, FP32 gate/out | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_gate_f32_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-gate-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax+sigmoid oracle `gated_max_abs=4.47e-08`; `rocprofv3` shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel` (`DurationNs=16320`, `VGPR_Count=32`, `Scratch_Size=0`) and `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<float>` (`DurationNs=5000`, `VGPR_Count=16`, `Scratch_Size=0`) on W7900 |
| `paged_attn_decode` variants `bf16_split_k_gate_bf16_spans`, `bf16_split_k_gate_fp16_spans` | `w4_paro` long-context full-attention decode + gate, BF16 KV cache, BF16/FP16 gate/out | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_gate_bf16_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gate_fp16_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 outputs (`bf16_mismatch=0`, `fp16_mismatch=0`, max abs `0`); wrappers instantiate the parent gated reduce with `hip_bfloat16` and `_Float16`, not integer casts; `rocprofv3` shows FP16 `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<_Float16>` (`DurationNs=10040`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=24`) on W7900 |
| `paged_attn_decode` variants `bf16_split_k_warp_spans`, `bf16_split_k_gqa_spans`, `bf16_split_k_gqa_gate_bf16_spans`, `bf16_split_k_gqa_gate_bf16_parallel_reduce_spans`, `bf16_split_k_gqa_gate_bf16_batch_spans`, `bf16_split_k_gqa_gate_fp16_spans`, `bf16_split_k_gqa_gate_fp16_batch_spans` | `w4_paro` Qwen3.5 GQA-specialized long-context and small-B verifier full-attention decode | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_gate_{bf16,fp16}_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_batch_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_gate_fp16_batch_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → Qwen3.5 shape `[16,256] / 2 KV`, `ctx=512`, NumPy oracle `warp_max_abs=4.1e-08`, `gqa_max_abs=4.1e-08`, BF16 gated output bit-exact (`gqa_gate_bf16_mismatch=0`); FP16 GQA gated wrapper shares the same `_Float16` reduce instantiated by `qwen35-paged-attn-gate-bf16-hip`. `python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-state-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` drives KV append + GQA gated decode through `Qwen35ParoDecodeState` and is bit-exact (`appended_key_mismatch=0`, `appended_value_mismatch=0`, `gqa_gate_bf16_mismatch=0`). `python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-batch-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` validates row-batched BF16 and FP16 gated paths bit-exact vs independent c1 GQA decode at uneven 1017/1021/1024/1025 contexts (`gqa_gate_{bf16,fp16}_batch_vs_c1_mismatch=0`). On gfx1151, cached `rocprofv3` records the c4 producer at **262.972 us** (grid Z 4) and the new `hip_bfloat16` batch reducer at **12.143 us** (grid Y 4), both with 256-thread workgroups. MTP B=3/D32 rocprof under `chain_attn_mode=decode_batched` shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_batch_kernel<8,16,2>` + `qwen35_paged_full_attn_decode_split_k_reduce_gate_batch_kernel<_Float16>` cut full-attn attention+KV `1.91 -> 0.45 ms/pass` vs the prefill-batched verifier; artifact `benchmarks/results/2026-06-07-hipengine-mtp-small-b-decode-full-attn.json`. LCP-D2 adds the gfx1100-scoped prepare-plus-coalesced BF16 split reduction from 32K: the 8,448-token/33-split NumPy fixture is exact; clean 513-split `rocprofv3` records serial `194.881 us` versus `6.280 + 18.720 us`; clean 32K/64K/128K graph decode improves `+1.23%/+3.95%/+7.80%`, max long-context KL is `1.904e-6`, and gfx1151 retained serial fallback pending an independent gate. SH7-A1 now admits that same registered route on gfx1151 from 32K: a one-queue same-source pair improves wall **+1.560%/+2.394%** at 32K/64K (**-0.333/-0.593 ms/token**), reduces the reducer **424.162 -> 109.346** and **744.973 -> 207.485 us/token**, preserves all **1,296/1,296** semantic logits byte-exactly, and traces the 24-VGPR prepare plus 16-VGPR output at 1 KiB LDS and zero scratch. Contexts below 32K and explicit opt-out retain serial fallback. SH8-A1's exact qgroup4 producer screen lowers 72 -> 56 VGPR but doubles the K/V ownership grid and regresses complete producer+parallel-reducer wall to **0.8959x/0.8852x** at 32K/64K (**0/42 wins**); the transient sibling is removed before model routing. LCP-D1 retains the original serial gated reducer through 256 splits and parallelizes only independent work above that boundary; the 256/257 fixture and 4,096-value BF16 A/B are exact, and the clean gfx1151 128K trace moves the reducer **234.714 -> 196.466 us/call (-16.30%)** with 16 VGPR and zero scratch. |
| `paged_attn_decode` variants `gqa_splitk_spans`, `gqa_splitk_gate_bf16_spans`, `gqa_splitk_gate_fp16_spans`; key-only `int8_key_bf16_value` variants `gqa_splitk_spans`, `gqa_splitk_gate_bf16_spans`; block16 `int8_block16` variants `gqa_splitk_spans`, `gqa_splitk_gate_bf16_spans` | `int8_per_token_head` Qwen3.5 grouped-GQA split-K decode, signed INT8 K/V cache with per-token/head scales; diagnostic key-only layout stores signed INT8 K plus BF16 V; diagnostic block16 layout stores signed INT8 K/V plus per-token/head/16-dim-block scales | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_attn_decode_int8_gqa_splitk_spans(...)`, `qwen35_paged_attn_decode_int8_gqa_splitk_gate_{bf16,fp16}_spans(...)`, `qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_spans(...)`, `qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_gate_bf16_spans(...)`, `qwen35_paged_attn_decode_int8_block16_gqa_splitk_spans(...)`, `qwen35_paged_attn_decode_int8_block16_gqa_splitk_gate_bf16_spans(...)` | `python3 scripts/qwen35_kv_int8_accuracy.py --device hip --contexts 64,520 --block-size 256 --num-q-heads 16 --num-kv-heads 2 --head-dim 256 --pseudo-vocab-size 32 --require-int8-hip --max-abs-threshold 2e-3 --json /tmp/hipengine-int8-hip-ctx64-520.json` → accepted; INT8 HIP vs CPU oracle max_abs `5.22e-08` at ctx64 and `1.86e-08` at ctx520 with top-1 `1.0`. Key-only/block16 primitive gate: same script with `--scale-dtype fp32 --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build --json /tmp/hipengine-block16-hip-64-520.json` → accepted, block16 HIP vs CPU oracle `max_abs <= 2.98e-08`, top-1 `1.0`; rocprof key-only ctx64 shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_key_bf16_value_kernel<float,8,16,2>` ran once (`20520 ns`). `python3 scripts/smoke.py --mode qwen35-paged-attn-int8-gqa-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` validates direct FP32-scale ctx64 decode, unaligned split ctx384 (`chunk_size=128`), and FP16-scale ctx520 page-boundary decode plus FP16/BF16 gated outputs (`max_abs<=2.98e-08`, `gate_fp16_max_abs=1.53e-05`, `gate_bf16_max_abs=1.49e-08`). `rocprofv3` shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel<float,8,16,2>` (`DurationNs=65200`, `47363`) and `_Float16` scale launches (`88723`, `78883`, `85603` ns) plus reduce/gated reduce kernels on W7900. The GQA producer grid is `(kv_head, split)`, so each KV stream is scanned once while sharing K/V loads across the 8 Q heads in its group. The 2026-08-04 gfx1151 maintenance guard `test_int8_gqa_splitk_producer_grid_is_owned_by_kv_head` now freezes both producer ownership and the `num_kv_heads × num_splits` launch. A fresh cached-build smoke remains exact (`max_abs<=2.98e-08`, FP16/BF16 gated max abs `1.53e-05/1.49e-08`); `rocprofv3` records global grid X **512 = 2 KV heads × local256**, grid Y **1/3**, VGPR80/SGPR128/LDS0/scratch0, rather than the **4096** global-X work items a 16-Q-head producer would launch. |
| `paged_attn_decode` variants `hadamard_group32_gqa_splitk_spans`, `hadamard_group32_gqa_splitk_gate_{bf16,fp16}_spans` | `int8_hadamard_group32` Qwen3.5 grouped-GQA split-K decode; query is transformed in wave32 registers, attention consumes transformed K/V, and the self-inverse transform is fused before the existing split reducer | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_attn_decode_int8_hadamard_group32_gqa_splitk_spans(...)`, gated BF16/FP16 variants | Native writer+decode fixture matches CPU dequantized paged attention within `max_abs <= 2e-4`; cached W7900 trace records `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_groupwise_kernel<_Float16,...,32,true>` at `18760 ns`, 88 VGPR, `Scratch_Size=0`, followed by the unchanged split reducer at `1600 ns`. BF16 storage and unfused CPU dequantize+attention remain fallback boundaries. Clean `c971262f` therock-7.15 GGUF closure passes the full 512/8 and 4K/16 quality suites plus bounded 128K/16 and saves exactly 18.75% persistent K/V with no persistent BF16 shadow. It remains explicit/non-default: 4K production prefill/decode regress 0.67%/0.75%, 128K decode regresses 3.82%, and an inferred four-layer BF16 prefill transient raises allocator high water by 0.532 GiB. Artifact: `benchmarks/results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json`. |
| `paged_attn_prefill` variant `hadamard_group32_gqa_gate_fp16_spans` | `int8_hadamard_group32` streaming causal-GQA prefill over transformed K/V, with wave32 inverse transform before FP16 gate/output | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans(...)` | Compiles/registers with the shared groupwise attention body and is integrated for the explicit `tail4_hadamard_group32` policy. Clean GGUF native quality passes at 512/8, 4K/16, and bounded 128K/16, but production prefill regresses 1.20%/0.67%/0.38% at 512/4K/128K and allocates an inferred four-layer 1.002 GiB BF16 transient. It remains explicit and non-default pending transient removal plus a repeated non-regressive speed gate; BF16 prefill remains fallback. |
| `paged_attn_prefill` variant `per_token_head_gqa_gate_fp16_spans` | `int8_per_token_head` Qwen3.5 streaming causal-GQA prefill, signed INT8 K/V cache with per-token/head scales, FP16 gate/output | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_attn_prefill_int8_gqa_gate_fp16_spans(...)` | #88 removes the temporary BF16 INT8 prefill oracle by appending full-attention K/V directly into the retained INT8 cache and running online-softmax causal prefill over the INT8 cache plus scale tensors. `HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-int8-prefill-gpu1-20260615-023751/hipcc-version.txt python3 -m pytest tests/test_qwen35_int8_prefill_attention_gpu.py -q` -> `1 passed` against a NumPy causal INT8 reference. Focused host dispatch/runtime coverage `python3 -m pytest tests/test_kv_dispatch.py tests/test_qwen35_paged_attn_decode_plan.py tests/test_qwen35_decode_state.py tests/test_qwen35_resident_batch_layout.py -q` -> `195 passed`. Cached `rocprofv3 --kernel-trace` captured `qwen35_paged_full_attn_prefill_gqa_gate_int8_kernel` (`DurationNs=10440`). GPU1 262K scratch gate: `int8_oracle_bytes 536870912 -> 0`, min-free `0.664 -> 1.139 GiB`; artifact `benchmarks/results/2026-06-15-gpu1-int8-prefill-streaming-scratch-262k.json`. |
| `full_attn_prefill` variants `qwen35_causal_gqa_gate_fp16`, `qwen35_varlen_causal_gqa_gate_fp16` | `w4_paro` append-then-attend causal GQA prefill, BF16 KV cache, FP16 gate/output | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(...)`, `qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → tiny paged causal-GQA fixture vs CPU `full_attn_prefill` oracle after prompt KV append, `prefill_gate_fp16_max_abs=0`, `prefill_gate_fp16_mismatch=0`; `python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → two packed request segments with row-shaped block tables, `varlen_prefill_gate_fp16_max_abs=0`, mismatch `0`; `rocprofv3` shows prompt KV writer (`DurationNs=6880`) and `qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_kernel` (`21520`) on W7900; all-layer 512 prefill after the shared-query cache/vector key-dot update, fixed `block_size=256` address fast path, and split short-row template shows `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>` ran 10 times (`26.362 ms` total, avg `2636.2 us`) on W7900. Full single-request fixture gate accepted in `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json` (`max_kl=0.0168`, top-1 100%), and active multiloop fixture gate remains green (`max_kl=0.03406`, top-1 100%), but no throughput row promoted. |
| `moe_ffn_selected` variants `fused_dual_silu_down_{bf16,f32}_out` (M16.3 B1) | `gguf_q4_k` raw rank-3 expert weights, fused selected-expert MoE FFN megakernel | `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_moe_ffn_fused.hip` | `gguf_q4_k_selected_ffn_fused_{bf16_bf16,f32_f32}_out(...)` | First M16.3 megakernel (MEGAKERNEL.md B1). One block per selected `(token, expert)` row computes the whole expert FFN — `gate_up GEMV -> silu*mul -> down GEMV` — keeping the `ffn_len`-wide intermediate on-chip (dynamic LDS `hidden + ffn_len` f32), so the gate_up-output HBM write + down-input HBM read vanish and 3 big-grid GEMV launches collapse to 1. Thread-owns-output (no cross-thread reduction) -> **row-invariant by construction**. Q4_K dequant + selected-expert addressing forked from `gguf_q4_k_gemv.hip`. Output is per-selected-row down `[rows, hidden]`; routing-weighted combine stays a separate kernel. Grid `(rows,)`, `__launch_bounds__(256, 2)`. Constraints: `hidden % 256 == 0`, `ffn_len % 256 == 0`. Inactive/out-of-range expert lanes emit zeros. Unfused fallback = existing primitive chain (`gguf_q4_k_selected_dual_gemv` -> `silu_mul` -> `gguf_q4_k_selected_gemv`). Gated vs B0 CPU oracle `cpu_reference.gguf_moe_selected_ffn` (`tests/test_gguf_q4_k_moe_ffn_fused.py`, 5 pass): f32 `kl_mean=9.3e-12`, top-1 `1.0`, `max_rel=6.9e-6`; bf16 `kl_mean=2.5e-4`, `kl_max=8.5e-4`, top-1 `1.0` (clears KL<=0.05/top-1>=90%); GPU row-invariance bit-exact (rows=1 == in-batch per row). `rocprofv3 --kernel-trace` (W7900, hidden=2048, ffn_len=512, E=256, rows=8): `gguf_q4_k_selected_ffn_fused_kernel<unsigned short, unsigned short>` 1 dispatch, `SGPR=128`, `Workgroup=256`. B1.x block-structured Q4_K decode hoist (decode d/dmin/scale once per 256-K block) took the single-shot from `3.61 ms` -> `0.815 ms` (VGPR `24` -> `104`); hot A/B microbench `0.266 ms/call` vs unfused raw chain `0.420 ms` (**1.58x**). The bf16 variant rounds gate/up + silu(gate)*up to bf16 (`round_intermediate<out_t>`, `expf`) to match the deployed bf16 pipeline it replaces; the f32 variant stays full-fp32 (reference). **B2 wiring:** `HIPENGINE_GGUF_FUSED_MOE_FFN` (default off) routes the rows==1 raw-Q4_K decode FFN through this kernel via `runtime/qwen35_gguf_runner.py::_try_run_post_attention_moe_c1_fused_ffn` (transparent fallback for T16/non-raw). E2E raw-path decode `9.859 -> 11.343 tok/s` (+15.1%, 512/128), launches/layer 3 -> 1. Kernel certified correct: oracle-exact on real layer-0 weights (`max_rel 1.2e-8`) and fused-vs-unfused `moe_down_out` matches in situ to ~1 bf16 ULP (`max_abs 1.22e-4`); passes `KL<=0.05` vs cpu_reference. Whole-model teacher-forced KL is large (`1.09`/32 tok, `scripts/gguf_fused_moe_ffn_teacher_forced_kl.py`) purely as 40-layer + KV-drift accumulation of the ~1-ULP bf16 reduction-order difference (any kernel swap shifts the exact E2E token stream). **Not promoted:** does not apply to the deployed T16 path (raw only), and occupancy-bound at single-token decode (only `rows` blocks). Kept gated off (`...b2-fused-moe-ffn-decode-diagnostic.json`). Best megakernel targets: batched c>1 GGUF decode (`c*8` blocks) and the verify/C_B regime. |

### Laguna global single-page decode (**base and gated runtimes rejected; exact primitives retained**)

The exact gfx1100 primitive is registered under
`laguna_attention_decode/bf16/global_context_single_page_spans`; the existing
`global_context_spans` scalar control and split-exact variants remain unchanged
and callable. The sibling consumes the complete `KVLiveSpans` ABI and preserves
every FP32 attention operation while replacing only both page translations with
`base_offsets[0] * 256 + token`. A device guard leaves output untouched above
one block-256 page. It is explicitly excluded from gfx1151 and has no CUDA/CPU
alias.

Synthetic live 1/70/126/256, page permutation, causal/future/eviction, BF16 edge,
independent CPU KL/top-1, and live-257 sentinel gates pass. All **24/24** actual
outputs across the 12 global layers at live 70/126 are F32 byte-exact. The one
frozen layer-0/44 transfer improves every event/wall median **15.45-17.45%**.
Integrated Clang-22 codegen is **1,008 instructions / 4,864 B**, logical VGPR
**32**, logical SGPR **49**, private/spills0, and five barriers.

The temporary false gfx1100 owner also passed exact shared-weight full state and
cache-only **12-candidate/zero-scalar/678-kernel** tracing. The frozen clean short
gate nevertheless rejects it without rerun. Both orders improve global-attention
family time **10.97%/12.12%**, but order A regresses complete kernel sum
**0.0949%** and dispatch span **1.3203%**; favorable order B and pooled results
cannot waive either per-order failure. The gate stops before 512/1K/3968 and
categories. Capability, resolver, session/allocator/CLI option, and route
selection are removed; runtime is byte-identical to the primitive commit and the
scalar path below live 127 plus split-exact path at live>=127 remain owners.
Canonical **63.270 tok/s / 678 kernels** is unchanged. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-runtime-correctness.json),
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-rejected.json).

Post-rejection ranking selected and primitive-admitted a new composite, not a
retry of either removed owner: `laguna_attention_decode+attention_gate/bf16/
global_single_page_softplus_bf16_spans`. Its mechanically checked body preserves
the admitted page-zero attention function and appends only D15's exact F32
softplus/multiply/RNE-BF16 epilogue inside the same local256 workgroup, writing
both unchanged F32 context and BF16 gated context. The required unfused fallback
is the admitted one-page primitive plus the registered standalone softplus gate;
gfx1151, CUDA, and CPU do not alias the composite.

Synthetic live 1/70/126/256 plus live257 sentinel and all **24** actual layer/live
rows preserve both output types. Layer0/44 inclusive event/wall improves every
live70/126 row by **9.53-14.57%**. Integrated Clang-22 codegen is **1,181
instructions / 5,756 B**, logical VGPR32/SGPR54, private/spills0, five barriers,
and dynamic LDS16,928. Cache-only tracing names the distinct local256 composite
at allocated VGPR32/scratch0 with no compiler.

The separate false/default-off owner passed shared-weight bulk prefill, all **48
hidden / 47 routed** boundaries, 16 transitions through live127, K/V/spans,
reset, KL0/top-1 100%, unchanged allocation peak, and lifecycle. Cached tracing
proved live<=126 at **12 composite / zero scalar / zero standalone gate / 666
model kernels**, then live127+ at the retained **12 score + 12 gated reducer /
678 kernels** topology. SWA stayed 36+36/token, IQ3 stayed 45/token, and the
candidate stayed local256/VGPR32/dynamic-LDS16,928/scratch0.

Both frozen short process orders improved attention+gate **17.77%/18.29%**,
complete kernel sum **0.87%/1.23%**, span **1.05%/0.90%**, and profiled-child
throughput **0.33%/0.23%**. The 512/1K/3968 controls proved zero candidate and
identical retained split-gated 678-kernel ownership. The complete two-order
18-prompt gate improved aggregate h16/h32 decode **0.844%/0.809%** and every
train/heldout category at both horizons; E2E, prefill, quality, and lifecycle
checks passed. Promotion still fails without rerun because train aggregate TTFT
regresses **0.780%**, beyond +0.5%; favorable overall **+0.071%** and heldout
**+0.064%** cannot waive it. Candidate h32 **63.853 tok/s** also remains below
Vulkan **64.418**.

Capability, resolver, session/allocator/CLI option, and route selection are
removed; the five production/primitive-test surfaces are byte-identical to
primitive commit `b49250b57`. The exact composite remains registered, while
scalar below live127 and split-gated at live>=127 remain runtime owners.
Canonical **63.270 tok/s / 678 kernels** and both benchmark rollups are
unchanged. Evidence:
[`gated design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json),
[`gated primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-correctness.json),
[`gated runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-runtime-correctness.json),
[`gated rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-rejected.json).

### SOL-G2 exact GGUF GDN split evidence

The current acceptance artifact is
[`2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json`](../benchmarks/results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json).
At `332f01f8` on gfx1151, the raw-Q/K-plus-scale split passes all 6/6 greeting,
512, 1024/1025 segment-threshold, and 4095/4096 chunk-boundary cases with exact
tokens, FP32 hidden seeds, and resident Conv/GDN state; greeting and 512 also
match every captured layer output. The cached-only narrow kernel trace records
the expected prepare, non-segment recurrence, segment recurrence, and RMSNorm
gate with `Scratch_Size=0`. This closes correctness only; its one-shot wall
fields are not G3 performance evidence.

SOL-G3 then measured the production full-prefill wall from clean detached
`ad773eba` with one warmup and four balanced same-session repetitions. The
exact split is slower than fused at both required contexts: `1248.436` versus
`1186.842 ms` at 512 (+5.19%) and `10870.022` versus `10187.300 ms` at 4K
(+6.70%). All timed token pairs remain exact. Fused stays the default; the
split kernels remain the unfused fallback and should not be retuned without a
different scheduling premise.

### K1 dense INT8 KV path evidence (**hipEngine landed, diagnostic/capacity path**)

The K1 path is the dense/uniform `KVLiveSpans` path with
`storage_dtype="int8_per_token_head"`, FP16 per-token/per-KV-head K/V scales, and
no persistent BF16 KV shadow. It is registered as storage/quant-keyed kernel
families rather than backend or quant branches in the engine:

- writer: `paged_kv_write` / `per_token_head_spans`
- decode: `paged_attn_decode` / `gqa_splitk_spans` and gated BF16/FP16 variants
- policy: `FixedPagedKVPolicy(..., storage_dtype="int8_per_token_head",
  scale_dtype="fp16", scale_granularity="per_token_head")`

Correctness gate used for the retained K1 artifacts:

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py \
  tests/test_qwen35_kv_e2e_fixture_gate.py \
  tests/test_qwen35_bench_memory_audit.py -q
python3 scripts/check_fixtures.py
python3 scripts/qwen35_kv_int8_accuracy.py --device hip --contexts 64,520 \
  --block-size 256 --num-q-heads 16 --num-kv-heads 2 --head-dim 256 \
  --scale-dtype fp16 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --require-int8-hip --json /tmp/hipengine-int8-accuracy.json
python3 scripts/qwen35_kv_e2e_fixture_gate.py --max-layers 40 \
  --kv-storage int8_per_token_head \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/hipengine-int8-kv-e2e-fixture-gate.json
```

Reference results on W7900/gfx1100, model `Qwen3.5-35B-A3B-PARO`, quant
`w4_paro`:

- Layer-level INT8 HIP accuracy accepted for contexts 64 and 520. INT8 HIP vs
  CPU oracle max abs was `5.22e-08` / `1.86e-08`; quantized-vs-BF16 KL was
  `2.34e-07` / `4.46e-08`; top-1 was `1.0` for both.
- E2E fixture gate with `--kv-storage int8_per_token_head` accepted:
  `max_kl=0.015328251530778358`, mean KL `0.001639289025262575`, top-1
  agreement `1.0`, generated IDs match the BF16 reference and fixture expected
  IDs.
- 128K/128 BF16-vs-INT8 diagnostic artifact:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json).
  BF16 baseline: `1021.180` prefill / `63.299` decode tok/s, sampled/tracked
  peak `22.410/23.288 GiB`, retained KV `2.690 GB`. INT8: `1011.064` /
  `61.275` tok/s, sampled/tracked peak `21.170/24.545 GiB`, retained KV
  `1.355 GB` (`1.0 B/element` payload plus `10.506 MB` scales), no BF16 shadow.
  This is a storage/capacity diagnostic, not a speed claim.
- 128K/256K INT8 AOTriton query-reuse + q3072 artifact:
  [`benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json`](../benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json).
  The 256K run completed and correctness/no-shadow passed at `651.636` prefill /
  `40.827` decode tok/s with retained KV `2.708 GB`; sampled/tracked peaks are
  `22.013/23.766 GiB`. The previous persistent full-prompt prefill
  double-buffering blocker (`2 x [262277,4096] fp16 = 4.297 GB`) is resolved,
  decode/phase scratch overlap is lower, AOTriton BF16 query input no longer
  accumulates per full-attention layer, and q3072 chunks keep tracked high-water
  below the 24GiB-class target. The remaining follow-up is streaming/removing the
  transient INT8-prefill oracle workspace itself.

Profiler summary from the 128K INT8 selected-region traces:

- Prefill writer: `qwen35_write_paged_kv_int8_per_token_head_kernel<_Float16>`
  ran `320` calls, avg `54.848 us`, max `69.761 us`, `Scratch_Size=0`.
- Decode producer: `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel<_Float16,8,16,2>`
  ran `160` calls in the sampled 16-replay trace, avg `621.500 us`, max
  `641.850 us`, `Scratch_Size=0`; reduce-gate avg `159.271 us`; decode append
  INT8 writer avg `4.363 us`.

### gfx1151 HIP backend (**initial port landed**)

`hipengine/kernels/hip_gfx1151/` is now a peer backend key for Strix Halo / Radeon 8060S. The initial implementation reuses the current proven gfx11 kernel bodies from `hip_gfx1100` and registers them under `hip_gfx1151`; build artifacts are compiled as native `gfx1151` via `HIPENGINE_HIP_ARCH=gfx1151` / `--offload-arch=gfx1151`, which is included in the JIT cache key. Backend packages expose a refreshable registration hook so lazy GGUF embedding/linear families can recreate missing gfx1151 aliases without overwriting registry fixtures. GGUF resident models and every materialized weight retain the resolved backend, and embedding, fused/single linear, router, GDN, and compact/sidecar MoE registry resolves use that identity rather than the shared source module name. A live public Qwen3.6 GGUF smoke on Radeon 8060S retained `hip_gfx1151` across generator, runner, model, root weights, and layer weights and generated token ID `11`. This is a correctness/bring-up port, not a claim that W7900 wrapper defaults are optimal on 40-CU gfx1151.

The Qwen3.5/PARO resident generation and benchmark harness accept `backend="hip_gfx1151"` / `--backend hip_gfx1151`. The vendored AOTriton 0.11.2b payload remains the pruned `amd-gfx11xx` image set; if additional gfx1151-specific AOTriton images become necessary, vendor them with Git LFS through the existing release/fetch workflow rather than downloading them at runtime.

`smoke_add` is a build/runtime smoke, not a model-layer primitive. It proves `hipengine.core.build`, lazy `libamdhip64.so`, device allocation/copy, launch, synchronize, and copyback without torch.

`qwen35_rmsnorm` is the first real model-layer HIP family port. It is BF16-bit (`uint16_t`) at the raw pointer ABI; Qwen weights store deltas and the kernel applies `1.0 + weight_delta`. PARO `paro_out` RMSNorm variants use direct norm weights and caller-owned output buffers, matching the parent native PARO serving path.

`paro_awq_gemv` ports the selected-expert and generic pack8 GEMV bodies used by the current OPTIMAL MoE c=1 route and non-MoE projections. The fused rotate→selected-dual GEMV path is landed for the parent strided layout; generic non-MoE and selected-MoE wrappers now cover strided/transposed qweight layouts for both BF16 and parent-parity FP16 activation/scale buffers. D1.1 added a generic transposed rotate-staged dual GEMV surface for decode diagnostics, but it remains opt-in/default-off because the rotate-once barrier/staging path regressed 512/128 decode. The FP16 `awq_fusedw4_prefill_fp16` WMMA prefill projection is ported from `nano-vllm-amd@55fede9` (`paroquant_fusedw4.py`) and is used for multi-token transposed pack8 prompt projections. hipEngine also provides `awq_fusedw4_prefill_dual_fp16`, a same-math dual-output launch used for paired transposed Q/K and QKV/Z prefill projections, plus a strided-layout instantiation for V/O/linear-out prompt projections without adding transposed weight copies. The GEMV wrappers are retained as c=1/small-row fallbacks.

`paro_marlin_k` ports the parent retained Marlin-K v0 vec8 FP32-FMA rows==1 decode path documented in `docs/MARLIN.md` and `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md` (`nano-vllm-amd@7718fff` vec8 FMA and `@1522293` qweight-neutral replacement; those short SHAs are documented but not present in the current parent checkout). hipEngine materializes `qweight_mk [N/8,K/128,128]`, small `qzeros_mk/scales_mk` decode metadata, and a zero-copy `qweight_pack8_decode [N/8,K]` alias so prefill and fused pair projections keep using the existing pack8/fusedw4 paths without duplicate large W4 buffers. The env gate `HIPENGINE_PARO_MARLIN_K_REPLACE` defaults on; setting it to `0` restores the old pack8/raw-qweight materialization for diagnostics.

`paro_silu` ports the selected-expert activation and down-rotation stage, including the fused `silu_mul_dual_rotate_out_kernel` path used by the parent default and the unfused/separate-gate fallback kernels.

`paro_combine` ports the c=1 selected-weighted/shared-gate/residual combine kernels. The current hipEngine wrappers cover the parent default FP32 router-weight/gate-logit path; scalar-weight variants can be added if a future route needs them.

`gguf_q4_k_gemv` is the first native GGUF spike. It intentionally stays separate from PARO/AWQ kernels: the raw variant consumes GGUF `block_q4_K` bytes directly, while the pack8 variant uses a lossless host repack into `qweight[int32 pack8] + scales[fp32] + mins[fp32]`. Both apply GGML's `d * scale[subblock] * q - dmin * min[subblock]` math. The pack8 layout reuses PARO-style output-channel locality without re-quantizing to PARO/AWQ zeros. The selected-dual q8_1+sudot4 POC proves the llama.cpp/GGML vector-dot recipe is viable on gfx1151 for the raw selected-MoE Q4_K bucket, and the rows>1 verifier now has reusable q8_1 workspace plus default-off raw/T16 split gates (`HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A`, `HIPENGINE_GGUF_T16_SELECTED_DP4A`, `HIPENGINE_GGUF_RAW_SELECTED_DP4A`). The straight T16 ports launch but are too small or too noisy to promote; the raw Q5_K/Q6_K selected-down port is much stronger in isolation and improves no-decode-repack B3, but still trails the default T16 decode-repack verifier. The retained-performance port should therefore move production GGUF GEMVs toward a GGML-like q8_1/x4 vector-dot layout instead of adding more isolated T16 dp4a kernels. GGUF projection paths now include BF16 and FP16 output variants plus `prefill_*` rows>1 registry variants for Q4_K pack8 and raw Q8_0/Q5_K/Q6_K. The current prefill device bodies are measured-equivalent row-grid kernels (not yet WMMA/GEMM-tiled), but `launch_gguf_linear(..., rows>1)` resolves to the prefill variants and the profiler-visible kernel names are `gguf_q4_k_pack8_prefill_out_kernel` / `gguf_k_prefill_out_kernel`. The Qwen3.5-0.8B GGUF public E2E correctness gate profiles native GGUF kernel dispatches under `rocprofv3`: Q4_K pack8, Q5_K/Q6_K/Q8_0 raw GEMV/prefill projections, Q6_K/Q8_0 embedding, GGUF F32-weight RMSNorm/add-RMSNorm, linear-attention conv/GDN, casts, SiLU, GPU argmax, and graph-friendly runtime-state kernels all launch in the native path. Task #49 adds correctness-oriented dense-BF16 fallback materialization for local Q4_1/F16/IQ4_XS tensors rather than pretending those are native throughput kernels. See `benchmarks/results/2026-05-17-hipengine-gguf-local-quant-coverage-diagnostic.json`, `benchmarks/results/2026-05-17-hipengine-gguf-decode-graph-replay-diagnostic.json`, `benchmarks/results/2026-05-17-hipengine-gguf-prefill-projection-diagnostic.json`, and `benchmarks/results/2026-05-16-hipengine-gguf-qwen35-e2e-correctness-diagnostic.json` for compact trace summaries.

`gguf_x8_selected_gemv` is the first sidecar-free GGML-style q8_1+sudot4 replacement-layout slice for GGUF selected-down Q5_K/Q6_K experts. Host repack stores raw GGUF blocks byte-losslessly as `tiles[expert, out_pack8, k_block, 8 * block_bytes]`, so the resident materializer can route selected down through `gguf_q5_k_x8_v1` / `gguf_q6_k_x8_v1` without retaining duplicate raw expert tensors. Kernels provide direct selected and compact selected BF16-output wrappers under `selected_x8_q8_1_dp4a_gemv_decode_*`; wrapper defaults use 64 threads after the 2026-06-28 selected-down microbench showed Q5 dot `0.03583 -> 0.03026 ms` and Q6 dot `0.02252 -> 0.01670 ms` versus the prior 128-thread default. Runtime materialization is default-off behind `HIPENGINE_GGUF_SELECTED_X8_REPACK`, which accepts `q5`, `q6`, or `both` (`1` maps to `both`) for quant-family diagnostics; `scripts/gguf_mtp_bench.py --selected-down-x8-repack ...` is the benchmark-facing switch. The 2026-06-27/28 gfx1151 diagnostics showed exact X8-vs-raw-dp4a outputs and project-gate correctness vs production T16 float, but broad B3/C5 was slower than the default T16 decode-repack control (full X8 `49.08 tok/s`, q6-only `50.32 tok/s`, default `51.77 tok/s` on the 2026-06-28 same-tree smoke). The 2026-07-01 `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` B2 route now retains q6-only X8 selected-down for the accuracy-traded replication lane: full suite **59.63 -> 60.36 tok/s**, `cycle_wall_ms_per_output` **16.793 -> 16.587**, and `target_block_verify_total` **13.178 -> 13.023 ms/output** with `--selected-down-x8-repack q6`. q5/both remains rejected for that route (`64.81 tok/s` smoke vs q6-only `69.03 tok/s`), so Q5_K selected-down stays on T16. Q4_K selected gate/up X8 is also exposed as `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` / `--selected-gate-up-x8`, but it is rejected on the same retained route: smoke regressed **67.62 -> 59.08 tok/s** and all-sync showed selected gate/up GEMV grew **1.408 -> 3.050 ms/output** in linear-attn layers. The T16 selected q8_1/dp4a verifier kernels now also default to 64 launch threads after the 2026-07-01 llama-compat full-suite row improved **55.45 -> 58.83 tok/s** and `target_block_verify_total` **14.025 -> 13.134 ms/output**; set `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS=128` only for rollback diagnostics. The q5-only one-wave override `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS=32` improves the Q5 T16 selected-down microbench but is rejected by compat smoke (**68.14 tok/s / 14.776 ms/output** vs retained smoke around **69.06 tok/s / 14.501 ms/output**), so it remains diagnostic-only. See `docs/MTP-LLAMACPP-PARITY.md`, `benchmarks/results/2026-06-28-hipengine-gguf-x8-selected-down-t64-dp4a-poc.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-full.json`.

`gguf_q6_k_x8_gemv` is the X8-packed Q6_K draft lm-head top-1 sidecar for the accuracy-traded llama-compat lane. It materializes `output.weight[:vocab]` as contiguous groups of eight GGUF Q6_K rows (`tiles[out_pack8, k_block, 8 * block_q6_K]`) and routes the resident draft lm-head through `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1` plus the existing stage2/gather reduction when `--resident-mtp-draft-q6-top1-stage1-shape x8` is selected. Correctness is covered by the q8_1/Q6_K oracle in `tests/test_gguf_q6_k_pack8_gemv_decode.py`. The 2026-07-01 llama-compat denseq8all A/B retains this as the active replication lane: smoke moved **71.53 -> 71.76 tok/s**, draft rocprof moved stage1 **3.603 -> 3.558 ms/cycle**, and full suite moved **61.19 -> 61.31 tok/s** with unchanged acceptance/economy. The follow-up `x8_dscale` diagnostic adds an X8-aligned FP32 `d*scale` sidecar and corresponding `gguf_q6_k_x8_dscale_gemv_q8_1_dp4a_top1_stage1` wrappers; correctness matches the same q8_1/Q6_K oracle, but draft rocprof rejects it because host wall regresses **6.805 -> 8.023 ms/cycle**, kernel time **6.427 -> 7.615 ms/cycle**, and the Q6 top-1 bucket **3.648 -> 4.859 ms/cycle** versus retained X8. This is not the exact default path; it remains an accuracy-traded diagnostic lane until the parity sprint either closes the remaining draft gap or replaces the draft lm-head body/layout. Artifacts: `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-{control-smoke,smoke,full}.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1.json`, and rejected `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8dscale.json`.

`gguf_q8_0_dp4a_gemv` is the raw-Q8 sidecar q8_1+sudot4 diagnostic family for the llama-compat verifier. The original singleton wrapper consumes one raw Q8_0 sidecar matrix and a prequantized q8_1 activation buffer. The 2026-07-01 `gguf_q8_0_dp4a_dual_split_rowtile4_gemv_bf16_bf16_out` wrapper consumes two raw Q8_0 sidecar matrices and writes split outputs in one 32-thread rowtile launch, reusing one weight row across up to four verifier rows for the linear-attention `attn_qkv+attn_gate` pair. The follow-up `gguf_q8_0_dp4a_rowtile4_gemv_bf16_bf16_out` and `gguf_q8_0_dp4a_triple_split_rowtile4_gemv_bf16_bf16_out` wrappers extend the same raw-Q8/q8_1 rowtile diagnostic to singleton projections and full-attention Q/K/V triples. GPU correctness passes against the q8_1 oracle plus the KL/top-1 gate, and `rocprofv3` confirms `q8_0_dp4a_dual_split_rowtile_gemv_kernel<unsigned short, 4>`, `q8_0_dp4a_rowtile_gemv_kernel<unsigned short, 4>`, and `q8_0_dp4a_triple_split_rowtile_gemv_kernel<unsigned short, 4>` launch with `Workgroup_Size_X=32`. These remain default-off: pair-only is gated by `HIPENGINE_GGUF_DENSE_Q8_DP4A` / `--verify-dense-q8-dp4a`, while broad sidecar materialization and singleton/triple routing are gated by `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` / `--verify-dense-q8-dp4a-all`. Pair-only full-suite `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8` regressed **60.36 -> 59.42 tok/s** with lower acceptance and more target rows/output. Broad denseq8all cut the isolated block profile dense-Q8 bucket **11.420 -> 8.902 ms/block** and full-suite verifier drain **13.023 -> 12.742 ms/output**, but acceptance regressed **0.583 -> 0.567**, so it is retained only for the accuracy-traded llama-compat lane, not the exact default Q8 verifier path. Artifacts: `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-{smoke,full,rowhist-full}.json`, and the earlier `...denseq8-rowtilepair-{smoke,allsync-smoke,full}.json`.

`gguf_q8_0_dual_gemv_f32_f32_out` is the exact raw-Q8 dual-output F32 wrapper used by the resident MTP draft shared expert. It reuses the existing `launch_gguf_k_dual_gemv_out<float,float,Q8_0>` body to compute shared gate and shared up in one launch from the same FP32 post-norm row, while preserving the two single-GEMV output values bit-for-bit. The resident draft path enables this by default through `HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL=1` and falls back to two `gguf_q8_0_gemv_f32_f32_out` launches when the env is disabled. Validation: `tests/test_gguf_k_gemv.py::test_q8_0_dual_f32_matches_two_single_gemvs`, draft rocprof control/dual A/B (`gguf_k_prefill_out` 16 -> 12 calls/cycle plus `gguf_k_dual_prefill_out` 2 calls/cycle), and full-suite llama-compat B2 **60.96 -> 61.19 tok/s** with unchanged acceptance. Artifacts: `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-{control,dual}.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-full.json`, and exact default confirmation `benchmarks/results/2026-07-01-ar-mtp-stage-timing-b5-exact-q8shareddual-full.json`.

`dense_gemv` ports the parent PARO BF16 dense GEMV used by auxiliary dense paths such as linear-attention AB projections when they remain dense rather than W4/W8 quantized. GGUF now also uses this kernel as an explicitly named dense-BF16 fallback for Q4_1, F16/BF16, and IQ4_XS projection tensors in local files. `lm_head` is the temporary GPU E2E bring-up head for PARO FP16 checkpoint weights and also supplies the standalone `argmax_f32` reduction reused by GGUF after tied Q6_K/Q8_0 lm-head logits are written, so resident decode graph replay no longer depends on CPU NumPy for final-token selection.

`paro_rotate` ports the parent PARO pairwise rotation helpers used by PARO projection paths (`paro_rotate1`, `paro_rotate2`, `paro_rotate3`); rotate1 is the single-output specialization needed by projection tails such as linear-attention `out_proj`. The wrappers cover both BF16 and parent-parity FP16 activation/scales, plus `paro_rotate1_bf16_gate_fp16(...)` for the AOTriton prefill tail: it rounds `BF16 attention * sigmoid(FP16 gate)` to FP16 before applying the same PARO rotate1 math, matching the old gate-kernel + rotate1 sequence while removing one launch. `qwen35_rotary` ports the parent full-attention prelude (`partial_rotary`, fused head RMSNorm + partial rotary, table-positioned scalar fused head RMSNorm + partial rotary, and hipEngine's vector-position `(tokens, heads)` prefill variant) plus resident q/gate split helpers for BF16 and parent-mixed FP16 activation streams; the AOTriton prefill path also has a vector-position variant that writes BF16 Q directly while preserving FP32 K for the paged-KV append. `convert/cast` provides small runtime glue casts for paths where a parent kernel emits FP32 or FP16 but the next PARO/lm-head projection consumes a different lowp dtype.

`w8a16_linear` ports the parent W8A16 GEMV kernels used by the current shared-expert default (`hip_w8a16_linear_lowp_out`) and W8A16 lm-head/auxiliary dense route. Lowp output wrappers now cover both BF16 and parent-parity FP16 activation streams. The FP16 `w8a16_shared_gate_up_silu_fp16` prefill helper adapts parent `w8a16_shared_gate_up_bulk4_kernel` to the raw-pointer lowp-output path, computing four shared-expert intermediate columns per block and writing the existing `shared_intermediate` scratch. `w8a16_shared_gate_up_silu_fp16_token_tiled` is a hipEngine prefill variant that preserves W8A16 storage while sharing gate/up weights across adjacent prompt tokens; runtime defaults use `token_tile=2` for legacy shared experts only when `tokens >= 1024`, with the original helper retained as fallback/opt-out. `w8a16_shared_gate_sigmoid_fp32` precomputes the shared-expert sigmoid once per token in the router shared-gate column after top-k/routing weights are materialized. The FP16 `w8a16_shared_down_combine_residual_fp16` helper consumes that precomputed gate while fusing grouped-prefill shared down projection with selected-output/shared-gate/residual combine; its default tile computes eight hidden rows per block, preserving the already-rounded `selected_out` ABI and exact per-row accumulation order. `w8a16_shared_down_combine_residual_fp16_token_tiled` shares the same fused tail while reusing down rows across adjacent prompt tokens; runtime defaults use `token_tile=2` for legacy prefill `tokens >= 2`, with the original helper retained as fallback/opt-out. c=1 and non-grouped paths keep the unfused gate/up/down/combine fallbacks. `scripts/smoke.py --mode w8a16-shared-expert-hip` chains W8A16 gate/up → `silu_mul_dual_out` → W8A16 down and is bit-exact against the staged BF16 NumPy oracle. `scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8` is the direct synthetic c=1 decode vertical smoke; `scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8` drives the same staged fixture through `Qwen35ParoDecodeState.run_moe_c1_bf16(...)` and validates the normalized prepared-weight/runtime-workspace path.

### Current GGUF GDN scheduling diagnostics

GPF-1 adds registered `gguf_qwen35 / gdn_prefill_recurrent` variants
`f32_decode_order_exact_tile64`, `f32_decode_order_exact_segments_tile64`,
`f32_decode_order_exact_tile32`, and
`f32_decode_order_exact_segments_tile32` in
`hipengine/kernels/hip_gfx1100/linear_attn/gdn.{hip,py}`. The wrappers are
`qwen35_gdn_prefill_recurrent_decode_order_exact_{tile64,tile32}_f32(...)`
plus segment-aware peers. gfx1151 RED/GREEN extends fused-state identity across
`(128,64,32) x (plain,segments)`; all six cases are byte-exact for BF16 output
and FP32 recurrent state, and the focused correctness/routing bundle passes 36
tests. A cache-clean 512 trace confirms tile64 ran with workgroup 64, 40 VGPR,
and no scratch/LDS, but recurrence regressed `794.120 -> 862.281 ms`; full
512/128 prefill regressed `423.708 -> 388.300 tok/s`, with tile32 at
`374.206 tok/s`. These are rejected, short-lived geometry controls; see
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf1-value-tiling-rejected.json`.

GPF-2 adds exact ordered-shuffle and relaxed tree-reduced wave32 variants,
each with plain and segment-aware registry entries. One wave owns one value
column and each lane owns four of the 128 state rows. The decisive tree variant
keeps those four FP32 state values in registers across the complete serial
token loop and writes final state once. On gfx1151 512/128, the non-resident
ordered/tree controls are rejected at `128.879/129.785 tok/s`; register
residency reaches `954.063 tok/s` versus fused `423.708`. A cache-clean trace
records `qwen35_gdn_prefill_recurrent_decode_order_wave32_tree_kernel`, 30
dispatches, `61.411 ms` total, workgroup 256, 40 VGPR, zero scratch/LDS. The
six-case full-model matrix has identical sampled tokens, KL
`3.48e-6..5.39e-5`, and 100% top-1, but not byte-identical hidden/recurrent
state. The candidate remained explicit while the numerical-contract and
generated-trajectory gates ran; see
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2-register-resident-candidate.json`.
The historical gate rejected the tree as a default because only 3/10 natural
prompts preserved the complete fused 128-step trajectory. After the prospective
2026-07-15 contract change made trajectory equality diagnostic, a fresh clean
W7900 18-prompt teacher-forced gate still rejects it on product correctness:
KL max `0.068757 > 0.05`, despite `443/450` top-1 and non-regressive decode.
The existing normalized-Q/K two-wave `chain_k2` also rejects at KL `0.059031`,
with `445/450` top-1 and non-regressive decode. GPF-9C therefore combines
normalized-Q/K input materialization with the one-wave32-per-column,
four-register-rows-per-lane schedule used by llama.cpp HIP on gfx1100; neither
existing rejected route has both properties. The explicit `chain_peer_wave32`
implementation follows llama.cpp `1ebf790cda38`'s unit-Q normalization,
post-reduction output scale, XOR reduction, and post-reduction scalar-decay
placement; it launches four columns per 128-thread block and compiles at 40
VGPR with zero LDS/scratch. Plain and segmented primitive fixtures pass the
CPU-reference numerical budgets. The clean W7900 18-prompt
product gate passes at KL max `0.041737`, top-1 `445/450`, and aggregate decode
wall `-0.050%`. Its clean speed gate then rejects promotion: 512 reaches
`2210.729 tok/s`, `-8.357%` below llama.cpp HIP, while 4K reaches `2513.374
tok/s`, `+11.454%`; both floors were required. GPF-9D therefore ports the
remaining peer geometry from llama.cpp Vulkan `263cc04a5405`: eight lanes per
value column, 16 state rows per lane, and clustered reduction. The explicit
`chain_peer_cluster8` implementation maps each physical wave32 to four
contiguous eight-lane clusters and each 256-thread block to 32 value columns;
plain/segmented CPU-reference fixtures pass, and rocprof reports 96 VGPR with
zero LDS/scratch and `41.840/44.801 us` recurrence on the synthetic fixture.
Its clean quality gate passes at KL max `0.028689` and top-1 `444/450`, but the
strict decode rule rejects aggregate wall `22508.498 -> 22508.787 ms
(+0.001286%)`; no tolerance was predeclared and no speed gate follows. The
peer-schedule lane closes without promotion. See
`benchmarks/results/2026-07-15-gfx1100-gguf-gdn-peer-cluster8-rejected.json`.

GPF-2C moves the exact ordered-wave variant's four state rows per lane into
registers without changing shuffles, explicit FMA sites, token order, or the
output expression. Plain and segment-aware output plus FP32 state remain byte-
exact and the 46-test GDN correctness/routing bundle passes. Residency recovers
512 prefill from the non-resident exact wave's `128.879` to `368.702 tok/s`,
but fused remains faster at 512/1K/4K by 12.98%/14.58%/13.50%. Its cache-clean
recurrence is `928.006 ms / 30`, 16.86% slower than fused, with workgroup 256,
80 VGPR, and no scratch/LDS. Keep `chain_wave32` diagnostic-only. The next
exact candidate should retain one scalar thread per value column and keep a
32- or 64-column recurrent-state tile in LDS across the token loop; see
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2c-ordered-resident-rejected.json`.

GPF-2D adds scalar-exact LDS-resident tile32/tile64 variants, each with plain
and segment-aware registry entries. One thread continues to own one value
column and preserves the fused 0..127 contraction/update order; a row-major
128xvalue FP32 LDS tile keeps state resident across the complete serial token
loop. The tile32 production candidate uses workgroup 32 and 16 KiB LDS. Do not
force-unroll the 128-row loops: that build generated 1,880 bytes/thread scratch
and lost the fused wall. Rolled loops use 64 VGPR, zero scratch, and reduce the
gfx1151 512 recurrence to `221.873 ms / 30`; focused prefill reaches
`753.489/799.844/686.840 tok/s` at 512/1K/4K versus fused
`423.708/448.694/410.023`. The clean six-case state matrix, balanced 512/4K
wall, and ten-prompt trajectory/decode gates all pass. Backend-package
capability metadata initially selected `chain_lds32` for gfx1151 `auto`; GPF-2E
below supersedes that policy after an incremental exact gate. gfx1100 retains
fused pending an independent transfer gate. Explicit `fused` and exact `chain`
remain rollback/oracle routes; a missing automatic candidate falls back to
fused. See
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-lds32-focus-candidate.json`,
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-exact-matrix.json`,
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-balanced-ab.json`, and
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2d-trajectory-decode-gate.json`.

GPF-2E adds a compact-scale/direct-conv refinement to the GPF-2D schedule. The
registered prepare variant
`f32_bf16_compact_scales` writes beta/decay as `[token,v_head]` and Q/K scales
as `[token,k_head]`; it does not materialize Q/K/V. Registered plain/segment
`f32_decode_order_exact[_segments]_lds32_direct` recurrence variants read the
canonical raw Q/K/V slices from `conv_out`, map `v_head % num_k_heads`, and
retain the same scalar recurrence in a 16 KiB LDS tile. Do not substitute the
compact scale ABI into a materialized recurrence: their scale indexing differs.
Plain and segmented production-head fixtures are byte-exact to materialized
LDS32 and pass the CPU-reference gate. A cached trace records workgroup 32,
64 VGPR, zero scratch, and both direct kernel names. The clean current-default
A/B is `776.428/825.319/700.824 -> 823.093/889.209/744.577 tok/s` at
512/1K/4K; the six-case state matrix and 250/250 natural transitions are exact,
and aggregate decode is +0.075%. Backend capability therefore selects
`chain_lds32_direct` for gfx1151 `auto`; gfx1100 remains fused. Materialized
`chain_lds32` stays as an explicit rollback/bisection route. The clean
right-sized 1+3 rollup publishes
`819.641/893.266/752.308/640.096/540.850/387.334 tok/s` across
512/1K/4K/32K/64K/128K with at most 0.132% prefill stdev/median. See
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-exact-matrix.json`,
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-balanced-ab.json`, and
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-trajectory-decode-gate.json`,
plus
`benchmarks/results/2026-07-13-gfx1151-gguf-prefill-gpf2e-right-sized-3run.json`.

LCP-2A adds default-off plain/segment
`f32_decode_order_exact[_segments]_lds32_direct_nonvolatile` variants. They
instantiate the same rolled scalar body without a `volatile` LDS pointer,
allowing LLVM to cache legal state accesses while preserving every source
operation and the volatile GPF-2E symbol as rollback. The first gfx1151 gate is
strongly positive: the isolated 512/4K recurrence moves
`6.572 -> 1.763 ms` and `58.613 -> 19.864 ms`; `rocprofv3` names the intended
kernel with 32 VGPR, 16 KiB LDS, and zero scratch versus 64 VGPR for GPF-2E.
All six full-model token/hidden/Conv/GDN-state cases are byte-exact, including
1024/1025 and 4095/4096 boundaries. Clean detached `53928aaf` reproduces those
cases and improves balanced 512/1K/4K prefill by
`+34.76%/+36.63%/+36.58%`; all 250 natural transitions and timed decode
trajectories are exact, with decode `+0.021%`. gfx1151 `auto` therefore selects
`chain_lds32_direct_nonvolatile`; gfx1100 stays fused. Keep the volatile direct
symbol as rollback for one release. Evidence:
`benchmarks/results/2026-07-14-gfx1151-gguf-gdn-lcp2a-clean-promotion.json`.
The independent gfx1100 transfer first promoted the same exact
`chain_lds32_direct` route after byte-exact primitive/state and 250/250 natural
transition gates; W7900 512/4K moved `649.131/677.888 -> 1291.225/1401.330
tok/s`. The later LCP-5A gate superseded that gfx1100 automatic policy with
quality-admitted `chain_peer_wave32`; gfx1151 remains on compiler-cacheable
`chain_lds32_direct_nonvolatile`. A final W7900 strict-exact screen now compares
the volatile and nonvolatile direct routes byte-exact. Nonvolatile halves VGPR
**64 -> 32** and moves the 512 trace-family median **7.172 -> 1.837 ms
(-74.39%)**. Balanced full-model 512/4K prefill improves **1400.079 -> 2422.276
(+73.01%)** and **1487.611 -> 2714.284 tok/s (+82.46%)**, with flat decode,
unchanged compact-scratch memory, and exact IDs. `HIPENGINE_GGUF_GDN_PREFILL_MODE=exact`
therefore resolves through backend capability to nonvolatile direct-LDS32 on
both gfx11 backends; gfx1100 production remains peer-wave. Keep the volatile
direct symbol for one exact-route rollback release. See `docs/REFACTOR.md` and
evidence `benchmarks/results/2026-07-14-gfx1100-gguf-prefill-schedule-transfer-gate.json`,
`benchmarks/results/2026-07-16-gfx1100-gguf-gdn-nonvolatile-exact-rollback.json`.

GPF-6 screened three distinct gfx1100 register-resident/direct-input schedules
without retaining any kernel. One-wave/value and contiguous group4 reached
`2181.778/2473.972` and `2281.383/2661.671 tok/s` at 512/4K, but failed the
predeclared 18-prompt same-context gate (KL max `0.068757/0.065184 > 0.05`,
top-1 `98.444%/98.889% < 99%`). Irregular group3 passed primitive CPU-budget
fixtures but missed both speed floors at `1804.460/2019.696 tok/s`. All GPF-6
kernel bodies, registry entries, selectors, and routing were removed. The
retained `scripts/gguf_gdn_semantic_gate.py` plus frozen
`benchmarks/prompts/gdn-prefill-category-heldouts.jsonl` are the admission path
for any future non-byte-exact GDN schedule. Exact chunked/prefix algebra is the
only remaining GDN prefill research lane; do not reopen reduction-width sweeps.
See
`benchmarks/results/2026-07-14-gfx1100-gguf-gdn-register-residency-rejected.json`.

GPF-7 independently transferred Atlas `37513bf`'s scalar-column residency idea
while retaining hipEngine's exact direct-conv FP32 arithmetic. Plain and
segmented production fixtures became byte-exact after explicit FMA grouping,
but the cached gfx1100 trace compiled both 32-thread kernels at **256 VGPR** and
**1064/1060 bytes scratch per thread** (zero LDS). This fails the predeclared
<=192 VGPR/zero-scratch gate before full-model timing, so all GPF-7 code and
registry/routing surfaces were removed. Atlas remains lineage-only evidence:
SM121's register-file result does not transfer mechanically to RDNA3. GPF-7
therefore sent the next attempt to chunkwise/WY algebra rather than another
storage or reduction micro-variant. Its rejection is recorded in the GPF-6/7
artifact above; GPF-8 below records the outcome of that final lane.

GPF-8 completed the algorithmic lane and was rejected. The retained
`gdn_prefill_chunkwise_wy_segments` float64 oracle checks the direct lower-
triangular/Woodbury-Young identity against token-serial recurrence for chunk
sizes 1/2/3/8/16, packed remapping, and odd tails. The temporary C=8 HIP body
passed primitive correctness and compiled at **256 threads, 48 VGPR, zero
scratch, and 28 KiB LDS**; its synthetic 30-layer 512 GDN stage was **47.491
ms**, below the frozen 66 ms ceiling. Clean W7900 model gates then failed: KL
**0.056522 > 0.05**, top-1 **445/450 = 98.889% < 99%**, only **5/18** exact
free-running trajectories, and 512 prefill **2003.399 < 2412.320 tok/s** even
though 4K reached **2280.244 > 2255.080**. All candidate HIP/registry/runtime
test surfaces were removed; `chain_lds32_direct` remains default. Future GDN
work requires a materially different algorithm or model-wide path; do not
reopen storage/reduction or C=8 WY variants. See
`docs/GGUF-PREFILL-OPTIMIZATION.md` and
`benchmarks/results/2026-07-15-gfx1100-gguf-gdn-chunkwise-wy8-rejected.json`.

### Laguna online-qrow4 attention extension

Post-350 LAP-7 adds separately registered global/SWA online-qrow4 consumers
and M128-qualified gfx1151 selectors. The wrapped/evicted eight-row fixture,
including a seven-row tail, is byte-identical to the retained qrow2 output.
Cached gfx1151 tracing names the qrow4 templates at local32, VGPR 72/80,
SGPR128, and zero LDS/scratch. A one-load counterbalanced pp512 screen improves
qrow2 **353.836 -> 365.249 tok/s (+3.23%)**. Clean selector-unset production
is **364.839 tok/s** median with **363.944** minimum; cached tracing cuts the
attention family **274.724 -> 229.181 ms (-16.58%)**. Qrow2 remains the
short/residual-tile route. Evidence:
`benchmarks/results/2026-07-25-gfx1151-laguna-prefill-qrow4-production.json`.
The bounded qrow8 follow-up is rejected and removed. Global qrow8 is
byte-identical to qrow2 and traces local32/VGPR112/LDS0/scratch0, but its
dirty **365.471 -> 366.126 tok/s (+0.179%)** signal reverses in the clean
committed gate to **363.475 -> 361.055 (-0.666%)**. SWA qrow8 also regresses
**365.392 -> 349.177 tok/s**. Global and SWA therefore remain qrow4
(`benchmarks/results/2026-07-25-gfx1151-laguna-global-qrow8-candidate.json`).
The simpler qhead3 cooperative SWA schedule is also rejected and removed.
Grouping three qrow4 wave32 heads around one qgroup9 KV head cuts workgroups
**72 -> 24**, but exact K8/float-LDS staging regresses matched pp512
**364.738 -> 298.652 tok/s (-18.1%)**. K32/BF16-LDS staging reduces barrier
frequency 4x yet regresses further **364.943 -> 256.697 tok/s (-29.7%)**.
Cross-wave synchronous LDS reuse is closed; a future cooperative attention
body must parallelize key work without this barrier/occupancy structure
(`benchmarks/results/2026-07-25-gfx1151-laguna-swa-qhead3-rejected.json`).
Scalar qrow4 key splitting is also closed. Contiguous two/four-wave key ranges
preserve one K/V read per token and pass the wrap/eviction tolerance oracle,
but regress pp512 **386.075 -> 377.219 (-2.29%)** and
**385.998 -> 379.597 (-1.66%)**. The four-way body is
local128/VGPR88/LDS8704B/scratch0; partial-PV LDS plus two barriers and state
merge outweigh key parallelism. All candidate surfaces are removed. A future
route must tile QK/PV instead of merging complete 128-dimensional partial
outputs
(`benchmarks/results/2026-07-25-gfx1151-laguna-swa-keysplit-rejected.json`).
The follow-up source-qualified qrow4 SWA candidate stays single-wave and
barrier-free: after visibility is known it skips the unused current or cached
K/V source. Full-eight and odd-seven wrap/eviction outputs are byte-identical
to qrow4; the cached gfx1151 trace is local32/VGPR80/SGPR128/LDS0/scratch0.
The five-pair dirty same-owner screen improves **365.584 -> 368.531 tok/s
(+0.806%)**, with candidate minimum above baseline maximum. Clean committed
selector-unset production retains **364.753 -> 366.933 tok/s (+0.598%)**;
cached tracing cuts SWA **185.603 -> 173.749 ms (-6.39%)** and measures
**369.532/342.620/285.563 tok/s** at 512/1K/4K. The M128-qualified selector
uses it; qrow2 remains residual
(`benchmarks/results/2026-07-25-gfx1151-laguna-swa-sourcequal-production.json`).

### Laguna gfx1151 decode GQA3 score extension

LD-1 begins with a bit-exact, scratch-ABI-preserving score-owner screen before
the larger split-K fused-attention rewrite. The separately registered
`swa_context_split_{,tile16_}exact_gated_gqa3_scores_spans` variants load each
BF16 SWA key once for three query heads, emit the unchanged 72-head score and
physical-slot planes, and delegate to the retained wave-local value reducer.
The live 1/65/127/128/256/257/511/512 ring, wrap, eviction, finite-extreme, and
tied-score gate is byte-exact for both F32 context and gated BF16 output.
Cached gfx1151 tracing records the tile16 GQA3 producer at local256, VGPR40,
LDS0, scratch0. At live 511/512 its three samples take
**20.839/20.559/19.035 us** versus retained
**35.988/35.707/35.787 us**, a **42.1-46.8%** score-producer reduction.
Earlier exact GQA9 and GQA3 value-reducer owners were removed after losing
about **5-11%** at live 512: hot V reuse did not compensate for reduced
parallel ownership. The clean same-commit full-cycle rollback/candidate gate
moves p512/d128 **14.563678 -> 14.740486 tok/s (+1.214%)**, saves
**0.824 ms/token**, and preserves the complete trajectory, positions, and
lifecycle. The gfx1151 capability is therefore retained as the production
default; gfx1100 and ungated/fallback paths remain on the established score
owner. Evidence:
`benchmarks/results/2026-07-28-gfx1151-laguna-swa-gqa3-scores-retained.json`.

The retained saturated-512 successor specializes only the hot value reducer.
It keeps one local128 workgroup/query head and all **72 workgroups / 288
wave32s** per SWA layer, plus the exact slot-order maximum, exponent,
denominator, per-dimension FMA, divide, gate, and stores. Natural
72Q/8KV/D128 fixed bounds cut the complete score+reducer leaf
**0.108265 -> 0.081059 ms/layer (-25.13%)**. Seven exact p512/d128 pairs move
**16.386231 -> 16.833740 tok/s (+2.731%, -1.622 ms/token)**. Cache-only
tracing records all **4,572 = 36 x 127** fixed reducers and zero generic
fallback at local128/VGPR16/LDS0/scratch0. gfx1151 selects it only at saturated
512; shorter live counts, non-natural shapes, and peer backends retain the
generic exact route. Evidence:
[`retained fixed512 reducer`](../benchmarks/results/2026-07-28-gfx1151-laguna-swa-fixed512-reduce-retained.json).

The next saturated-512 successor fuses exact QK, slot-ordered softmax, PV,
gate, and stores for two adjacent query heads. Five local256 owners per KV
head retain **40 workgroups / 320 wave32s per SWA layer**, reuse each K vector
across a query pair, and remove the global score plane plus one launch without
changing a scalar, FMA, F32 context byte, or BF16 gate byte. Seven
resident-model p512/d128 pairs improve **17.013184 -> 17.065241 tok/s
(+0.306%, -0.179 ms/token)**, with every candidate faster and every
trajectory/state exact. Cached tracing records VGPR32/SGPR128/LDS6144/
scratch0. The cache-hot leaf alone regresses **2.96%**; it is not the
promotion gate because the resident model changes K-cache behavior. An exact
one-head local256 fusion makes that leaf **8.14%** faster but rereads K nine
times and regresses full production **1.038%**, so it and two other exact
losers are removed. Shorter live counts, non-natural shapes, peer backends,
and rollback retain GQA3 score plus fixed512 reduction. Evidence:
[`retained fused-GQA2 SWA`](../benchmarks/results/2026-07-28-gfx1151-laguna-swa-fused-gqa2-retained.json).

The exact local384 successor keeps **24 workgroups / 288 wave32 PV tasks**,
but stages 64 chronological V slots x D128 in LDS and reuses each BF16 load
across the three owned query heads. This is the quality-safe part of the
llama.cpp Vulkan attention lesson: tile V for grouped-query reuse without
changing softmax or PV association. A 32/64/128-slot leaf sweep is byte-exact
and improves **26.38%/26.58%/22.85%**; 64 slots wins. Seven resident-model
p512/d128 pairs improve **17.135411 -> 18.032171 tok/s (+5.233%, -2.902
ms/token)** with identical 128-token trajectories and lifecycle state.
Cached tracing records the expected template at 24 local384 blocks,
VGPR144/SGPR128/LDS24576/scratch0 and cuts the four-observation median
**184.085 -> 137.197 us (-25.47%)**. gfx1151 selects it only at the natural
saturated shape; the unstaged local384 body is the exact rollback and all
shorter/non-natural/peer paths are unchanged. Tracked-clean default production
confirms **18.026501 tok/s**; the full census cuts SWA **8.891 -> 5.844
ms/token (-34.27%)** and kernel span **58.846 -> 55.855 ms/token (-5.08%)**.
Evidence:
[`retained GQA3 V-stage64 SWA`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-retained.json) ·
[`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-production.json) ·
[`wall re-profile`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-vstage64-wall-reprofile.json).

The gfx1151 copy-width successor keeps that exact local384 compute unchanged
but maps each V-stage transaction to eight adjacent BF16 values. Aligned
16-byte global loads and LDS stores replace the scalar copy loop; QK, maximum,
denominator, PV FMA, gate, and store association are identical. The
nine-sample leaf improves **0.133491 -> 0.106533 ms (-20.19%)**. Seven
p512/d128 resident-model pairs improve **18.244607 -> 18.806305 tok/s
(+3.079%, -1.637 ms/token)** with exact trajectories/state. Cached tracing
records 24 local384 blocks, VGPR144/SGPR128/LDS30720/scratch0 and a
four-observation median **161.50 -> 112.97 us (-30.05%)**. gfx1151 selects it
only for saturated natural SWA; scalar V-stage64 is registered rollback.
Tracked-clean production confirms **18.814192 tok/s**, **+3.204%** over prior
clean 18.230064 and **+64.077%** over the 11.466687 sprint start.
Evidence:
[`retained vec16 V staging`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-retained.json) ·
[`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-production.json).

Generated ISA reveals that the first vec16 form materializes its `uint4`
aggregate as one hidden 16-byte LDS slot per thread, adding **6,144 B** and an
LDS round trip. The exact direct-store sibling branches before assignment:
valid vectors write from global memory to the real V tile, while invalid
vectors receive scalar zero stores. Wrapped/evicted F32/BF16 output is
byte-identical. The leaf improves **0.107000 -> 0.105197 ms (-1.686%)**;
seven resident-model pairs improve **19.070545 -> 19.083269 tok/s (+0.0667%,
-0.0350 ms/token)** with exact trajectories/state. Cached tracing improves
**101.590 -> 99.227 us (-2.326%)**, fixed LDS falls
**30,720 -> 24,576 B**, logical VGPR/SGPR fall **143/36 -> 138/33**, and
scratch remains zero. gfx1151 now selects the direct form only for saturated
natural SWA; the aggregate vec16 owner remains registered rollback. A
128-slot direct stage is not retained: the earlier exact stage-width screen
regresses stage64 **0.107446 -> 0.109075 ms (+1.516%)**. Evidence:
[`retained direct vec16 store`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-direct-retained.json) ·
[`clean direct-store production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage64-vec16-direct-production.json) ·
[`rejected stage128`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-vstage128-vec16-rejected.json).

The first post-ISA numerical candidate changes only the softmax exponential in
that retained direct-store body. Native `__expf` leaves QK, maximum,
slot-ordered denominator, PV FMA, gate, and store association unchanged.
Across the wrapped positions 512-519 plus explicit eviction, it remains inside
the CPU attention tolerance; one of 73,728 gated BF16 values differs from the
exact control by one rounding boundary (at most one of 9,216 per position). A
separate leaf state preserves all 9,216 gated BF16 values byte-for-byte. The
nine-sample leaf improves **0.106888 -> 0.096000 ms (-10.19%)** with maximum
F32 context error **1.86e-9**. Extracted gfx1151 ISA removes 195 static delay
slots and 64 FMA instructions while keeping logical VGPR138/SGPR33/LDS24576/
scratch0. Native tracing names both templates at local384/24 blocks and
improves **137.158 -> 112.772 us (-17.78%)**. This admits a candidate
primitive, not production. Seven resident-model pairs do confirm the wall
opportunity, **19.130955 -> 19.309790 tok/s (+0.935%)**, with 7/7 candidate
wins. The complete saturated-512 category gate rejects raw native
exponential: top-1 passes at **566/576 (98.26%)**, but max KL is **1.452698**
against 0.05. Production remains on accurate `expf`; the next numerical screen
retains accurate range reduction while deleting only generic-domain checks
that softmax inputs cannot exercise. Evidence:
[`fast-exp candidate`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-fast-exp-candidate.json) ·
[`fast-exp rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-fast-exp-rejected.json).

The bounded-domain successor keeps the accurate high/low `log2(e)` range
reduction, hardware `exp2`, and `ldexp`; it deletes only generic overflow,
NaN, and positive-domain guards that finite score-minus-max inputs cannot
exercise. Positions 512-519 plus eviction and the leaf are F32/BF16
byte-exact. Nine leaf samples improve **0.105985 -> 0.101133 ms (-4.58%)**,
and generated ISA removes 128 compares plus 128 conditional masks. The real
model disproves fixture exactness: seven production pairs improve only
**19.164777 -> 19.229973 tok/s (+0.340%)**, generated trajectories differ,
and the complete category gate reaches max KL **1.888082** with **558/576
(96.88%)** top-1. Reject and remove both manual/native exponential primitives
and their measurement seams. Production stays on compiler `expf`; further
attention work must preserve its results exactly and attack QK/PV ownership or
scheduling. Evidence:
[`bounded-exp candidate`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-bounded-exp-candidate.json) ·
[`bounded-exp rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-bounded-exp-rejected.json).

The exact successor leaves the compiler `expf` expression untouched and
exposes only the proven `score - maximum <= 0` invariant with
`__builtin_assume`. Wrapped positions 512-519 plus eviction and the leaf are
F32/BF16 byte-exact. Nine leaf samples improve **0.106007 -> 0.097387 ms
(-8.13%)**. Generated ISA preserves 65 native exponential instructions and
logical VGPR138/SGPR33/LDS24576/scratch0 while contracting **3,196 -> 2,821
instructions (-11.73%)** and **17,920 -> 16,884 B (-5.78%)**. Cached tracing
names both local384/24-block templates with allocated
VGPR144/LDS24576/scratch0 and improves the three steady observations
**126.838 -> 91.812 us (-27.61%)**. Seven tracked-clean p512/d128 pairs then
improve **19.140826 -> 19.245912 tok/s (+0.549%, -0.285 ms/token)** with 7/7
wins, complete sample separation, and exact generated IDs/state. gfx1151 now
selects the assumed-domain sibling at the saturated natural SWA shape; the
generic-domain direct-store body remains exact rollback. Evidence:
[`assume-exp candidate`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-assume-exp-candidate.json) ·
[`assume-exp retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-assume-exp-retained.json).
Three tracked-clean default runs measure
**19.231940/19.248066/19.242300 tok/s**, median **19.242300** and **+0.501%**
over the preceding clean 19.146417 packet. IDs, state, and lifecycle remain
exact. Evidence:
[`assume-exp production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-assume-exp-production.json).

Tracked-clean selector-unset production measures
**19.072126/19.085294/19.089552 tok/s**, median **19.085294**. This is
**+0.1015% / -0.0532 ms/token** versus the preceding clean 19.065940 packet
and **+66.441%** over the 11.466687 sprint start. Generated IDs, next/final
tokens, final position, and allocation lifecycle remain exact.

The global-attention sibling keeps the retained dynamic-live score ABI and all
48 local256 reducer workgroups, but specializes the production
48Q/8KV/D128/capacity-4096 geometry and scratch strides. F32 context and gated
BF16 output are byte-exact at live 257/513/576/639. Complete score+reduce
improves **0.7-2.0%** at the three production live points; seven exact
p512/d128 pairs improve **16.832097 -> 16.846689 tok/s (+0.087%, -0.051
ms/token)**. Cache-only tracing records all **1,524 = 12 x 127** fixed-shape
reducers, zero generic fallback, and local256/VGPR24/LDS512/scratch0. gfx1151
selects it only for gated natural-shape capacity-4096 global decode; peer
backends and other shapes/capacities retain the generic exact route. Evidence:
[`retained global fixed-shape reducer`](../benchmarks/results/2026-07-28-gfx1151-laguna-global-fixedshape-reduce-retained.json).

The fused global successor keeps all **48 local256 workgroups / 384 wave32s**
and the exact four-products-per-lane QK sequence, eight-wave
maximum/denominator association, token-order PV FMA, reciprocal, gate, and
stores. It removes the global score/physical round-trip and one launch; dynamic
LDS is only 8 bytes per live scan slot plus the 64-byte warp buffer. Complete
leaves improve **7.89-17.55%** at live 513/576/639. Seven resident-model
p512/d128 pairs improve **17.064962 -> 17.097044 tok/s (+0.188%, -0.110
ms/token)** with every candidate faster and every trajectory/state exact.
Cached tracing records 48 blocks, local256/VGPR24/SGPR128/scratch0. A
two-head GQA2 sibling reduces K reads but leaves only 24 workgroups, regresses
production **0.126%**, and is removed. Non-natural shapes/capacities, peer
backends, and rollback retain the exact split/fixed-shape route. Evidence:
[`retained fused one-head global`](../benchmarks/results/2026-07-28-gfx1151-laguna-global-fused-gqa1-retained.json).

The exact GQA2 global successor recovers the reuse missing from the earlier
occupancy-only screen. Twenty-four local256 workgroups each own two adjacent
GQA6 heads, share K as before, and now stage 64 V slots x D128 so each BF16 V
load feeds both exact PV chains. Softmax and per-head PV association are
unchanged. At live 513/576/639, the nine-sample leaf improves GQA1
**9.16%/12.39%/12.22%** with byte-exact F32 context and gated BF16, including
explicit eviction. Seven resident-model p512/d128 pairs improve
**18.034298 -> 18.237090 tok/s (+1.124%, -0.617 ms/token)** with every
candidate faster and all trajectories/state exact. Cached tracing records
local256/VGPR32/SGPR128/static-LDS512/scratch0; launch-time dynamic LDS is
22,540-24,052 bytes over the measured live range. gfx1151 promotes it at the
natural capacity/shape through live 4000; GQA1 remains exact rollback above
that LDS bound, and non-natural/peer paths are unchanged. Evidence:
[`retained global GQA2 V-stage64`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-retained.json).

The gfx1151 global copy-width successor pads the dynamic score/physical prefix
to a 16-byte boundary and moves eight BF16 V values per transaction. The
live4000 dynamic-LDS bound still fits below 64 KiB. QK, softmax, PV, gate, and
store order are unchanged. Nine-sample live513/576/639 leaves improve
**22.29%/25.82%/25.99%** byte-exactly. Seven p512/d128 pairs improve
**18.794424 -> 19.066920 tok/s (+1.450%, -0.760 ms/token)** with exact
trajectories/state. Tracing records 24 local256 blocks,
VGPR32/SGPR128/static-LDS512/scratch32 and **141.09 -> 103.29 us (-26.79%)**.
gfx1151 selects it through live4000; scalar GQA2 is rollback and GQA1 remains
the fallback above that bound.
Tracked-clean production confirms **19.065940 tok/s**, **+1.338%** over prior
clean 18.814192. The clean census cuts total attention **8.065 -> 5.652
ms/token (-29.92%)** and span **55.154 -> 52.814 ms (-4.24%)**. Evidence:
[`retained global vec16 V staging`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-retained.json) ·
[`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-production.json) ·
[`wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-vec16-wall-reprofile.json).

The generated global vec16 code object reveals the same aggregate-copy defect
as SWA in a different address space: one `uint4` temporary causes
`scratch_load_b128` plus two `scratch_store_b128` operations and **32 B** of
private segment per thread. The separately registered direct-store sibling
branches before assignment, sends valid global vectors directly to the real
dynamic-LDS V tile, and zeroes invalid vectors with four scalar stores.
F32/BF16 output remains byte-exact at live513/576/639 with eviction. The
nine-sample leaves improve **11.71%/11.83%/11.82%**. Cached tracing records
33 observations each at local256/24 workgroups and improves
**103.354 -> 90.530 us (-12.41%)** while scratch falls **32 -> 0 B**; logical
VGPR/SGPR stay 28/34. All seven resident-model pairs improve
**19.077502 -> 19.134537 tok/s (+0.2990%, -0.1562 ms/token)** with exact
trajectories/state. gfx1151 selects the direct form through live4000; aggregate
vec16 remains rollback and GQA1 remains fallback above the LDS bound. Evidence:
[`retained global direct vec16 store`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-direct-retained.json) ·
[`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gqa2-vstage64-vec16-direct-production.json).

The exact global exp-domain sibling applies the same proven softmax invariant
as the retained SWA specialization without changing compiler `expf` or any
QK/softmax/PV arithmetic. Live513/576/639 with explicit eviction remains
F32/BF16 byte-exact to the direct-store route and the CPU oracle. Nine-sample
leaves improve **2.34%/1.86%/1.92%**. Both code objects keep three native
exponential instructions, logical VGPR28/SGPR34, static LDS64, and no
private/spill storage; the candidate changes scheduling around the softmax
loop and contracts **830 -> 829** static instructions. Cached tracing names
both local256/24-block templates at allocated VGPR32/scratch0 and improves
aggregate median **90.490 -> 88.526 us (-2.17%)**. This is an admitted
primitive. Seven tracked-clean p512/d128 pairs then improve
**19.235596 -> 19.243968 tok/s (+0.0435%, -0.0226 ms/token)** with 7/7 wins
and exact generated IDs/state/lifecycle. gfx1151 now selects the assumed-domain
body at the qualified natural global shape through live4000; explicit false
restores the generic-domain direct-store body and peer/non-natural paths are
unchanged. Evidence:
[`global assume-exp candidate`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-assume-exp-candidate.json) ·
[`global assume-exp retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-assume-exp-retained.json).
Three tracked-clean selector-unset runs measure
**19.236922/19.250313/19.249443 tok/s**, median **19.249443** and **+0.0371%**
over the preceding clean 19.242300 packet. IDs, state, and lifecycle remain
exact. Evidence:
[`global assume-exp production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-assume-exp-production.json).

Tracked-clean selector-unset production measures
**19.136600/19.146417/19.153280 tok/s**, median **19.146417**. This is
**+0.3203% / -0.1673 ms/token** versus the preceding clean 19.085294 packet
and **+66.974%** over the 11.466687 sprint start. Generated IDs, next/final
tokens, final position, and allocation lifecycle remain exact.

The clean post-direct-store trace retains **816 dispatches/token** and measures
**50.016 ms** kernel sum / **52.567 ms** span. Attention is **5.466 ms**:
**4.152 ms SWA + 1.314 ms global**. Versus the pre-direct census, global falls
**10.70%**, total attention **3.28%**, kernel sum **0.44%**, and span **0.47%**.
The remaining Vulkan attention gap is **4.557 ms/token** and **48.5%** of the
clean wall gap. Further attention work must change the cooperative algorithm,
not copy lowering. Evidence:
[`post-direct-store wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-direct-store-wall-reprofile.json).

The final ordinary-workgroup GQA2 combination is rejected and fully removed.
Adding the later direct vec16 V transport and exact exp-domain specialization
to 40 local256 GQA2 owners is byte-exact and improves the cache-hot leaf
**0.861%**, but allocates **176 VGPR** and loses all seven resident-model
pairs: **19.249050 -> 19.182158 tok/s (-0.3475%, +0.1812 ms/token)**.
Ordinary GQA2 V staging is closed; the next exact cooperative attack must
remove the cooperative-launch/global-score boundary rather than repeat this
geometry. Evidence:
[`rejected GQA2 direct-vec16 staging`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa2-vec16-rejected.json).

The normal-launch persistent exact GQA9/K64 follow-up is also rejected and
fully removed. Thirty-two local256 workgroups produce two K64 tasks each,
rendezvous through a monotonic device counter, and replay the retained ordered
softmax/PV reduction. Wrapped/evicted F32/BF16 output is byte-exact, but the
leaf regresses **0.098299 -> 0.395157 ms (+301.99%)**. Cached tracing records
VGPR40/SGPR128/LDS0/scratch0, ruling out spills or an LDS occupancy cliff; the
full score-plane traffic and grid rendezvous are structurally dominant. The
Vulkan transfer boundary is now explicit: retain full-GQA K/V reuse only when
it stays fused with QK/softmax/PV, and do not reconstruct its split topology
with an exact global score repair. Evidence:
[`rejected persistent exact GQA9`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-persistent-gqa9-rejected.json).

The exact one-phase mixed32 successor is retained for gfx1151 saturated SWA.
Four local384 owners per KV head divide nine queries as **2+2+2+3**, raising
the grid from 24 to 32 blocks while keeping QK, compiler `expf`, ordered
softmax/PV, gate, and stores inside one fused phase. Pair-owner idle waves
still participate in each staged-V barrier. Wrapped/evicted F32/BF16 output is
byte-exact. The leaf improves **0.096586 -> 0.091360 ms (-5.41%)**; cached
tracing records the intended 32-block template at
VGPR104/SGPR128/LDS24576/scratch0 and improves
**112.931 -> 105.717 us (-6.39%)**. All seven resident p512/d128 pairs improve
**19.268862 -> 19.371717 tok/s (+0.534%, -0.276 ms/token)** with exact
trajectories/state. The architecture capability selects it only at
72Q/8KV/D128/SWA512; retained 24-block GQA3 is rollback and other shapes/
backends are unchanged. Evidence:
[`retained mixed32 SWA owner`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-retained.json).

Tracked-clean selector-unset production confirms
**19.353808/19.370310/19.368763 tok/s**, median **19.368763** and **+0.620%**
over the preceding clean 19.249443 packet. The architecture capability is
active, all three trajectories/state/lifecycle are exact, and peer/non-natural
routes remain unchanged. Evidence:
[`clean mixed32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-production.json).

The exact mixed32 exp4 sibling is retained next. Within each four-slot
softmax batch, lanes 0..3 issue one independent compiler `expf` concurrently
and shuffle the four weights back to every lane. Lane 0 retains the original
ordered denominator accumulation, every dimension retains its original PV
FMA order, and geometry/resources remain 32 local384 blocks at
VGPR104/SGPR128/LDS24576/scratch0. Wrapped/evicted output is F32/BF16
byte-exact. Nine-sample leaf timing improves
**0.091487 -> 0.089135 ms (-2.57%)** and the stable cached kernel window
improves **85.414 -> 83.584 us (-2.14%)**. All seven resident p512/d128 pairs
improve **19.368030 -> 19.432503 tok/s (+0.333%, -0.171 ms/token)** with
complete sample separation and exact trajectories/state/lifecycle. gfx1151
selects exp4 only inside the already-qualified saturated mixed32 route; the
serial-exp mixed32 sibling remains rollback. Evidence:
[`retained mixed32 exp4`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp4-retained.json).

Tracked-clean selector-unset production measures
**19.417147/19.424487/19.429963 tok/s**, median **19.424487**. This is
**+0.288% / -0.148 ms/token** over the preceding clean 19.368763 packet and
**+69.399%** over the 11.466687 sprint start. The exp4 capability is active
without a comparison selector and all three trajectories/state/lifecycle are
exact. Evidence:
[`clean mixed32 exp4 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp4-production.json).

The clean post-exp4 census keeps **816 compute dispatches/token** and measures
**49.296 ms/token** kernel sum / **51.850 ms/token** span. SWA attention falls
**3.583 -> 3.448 ms/token (-3.78%)** and total attention falls
**4.873 -> 4.745 ms (-2.64%)**. The remaining attention path is
**3.835 ms/token** slower than same-GGUF llama.cpp Vulkan and explains
**44.3%** of the clean wall gap. Evidence:
[`post-exp4 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-exp4-wall-reprofile.json).

The exact exp8 successor is retained. Lanes 0..7 issue one independent
compiler `expf` for each eight-slot batch, then shuffle the weights back to
the original ordered denominator and PV chains. There is no new LDS, barrier,
launch, or repair phase. Wrapped/evicted F32/BF16 output is byte-exact. The
leaf improves **0.089191 -> 0.083755 ms (-6.09%)** and the stable cached
kernel window improves **83.557 -> 78.667 us (-5.85%)** at unchanged
32 local384 blocks and VGPR104/SGPR128/LDS24576/scratch0. All seven resident
p512/d128 pairs improve
**19.427449 -> 19.510986 tok/s (+0.430%, -0.220 ms/token)** with complete
sample separation and exact trajectories/state/lifecycle. gfx1151 selects
exp8 only inside the qualified exp4/mixed32 route; exp4 remains rollback.
Evidence:
[`retained mixed32 exp8`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp8-retained.json).

Tracked-clean selector-unset production measures
**19.496106/19.515697/19.519033 tok/s**, median **19.515697**. This is
**+0.470% / -0.241 ms/token** over clean exp4 and **+70.195%** over the
11.466687 sprint start. The exp8 capability is active without a comparison
selector and all three trajectories/state/lifecycle are exact. Evidence:
[`clean mixed32 exp8 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp8-production.json).

The exact exp16 successor is retained. Lanes 0..15 issue one independent
compiler `expf` for each sixteen-slot batch, then return the weights to the
same ordered denominator and PV chains. Wrapped/evicted F32/BF16 output is
byte-exact. The leaf improves **0.083740 -> 0.082224 ms (-1.81%)** and the
stable cached kernel window improves **78.814 -> 77.265 us (-1.97%)** at
unchanged 32 local384 blocks and VGPR104/SGPR128/LDS24576/scratch0. All seven
resident p512/d128 pairs improve
**19.506557 -> 19.523370 tok/s (+0.0862%, -0.0441 ms/token)** with complete
sample separation and exact trajectories/state/lifecycle. gfx1151 selects
exp16 only inside the qualified exp8 route; exp8 remains rollback. Evidence:
[`retained mixed32 exp16`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp16-retained.json).

Tracked-clean selector-unset production measures
**19.514684/19.538643/19.530105 tok/s**, median **19.530105**. This is
**+0.0738% / -0.0378 ms/token** over clean exp8 and **+70.320%** over the
11.466687 sprint start. The exp16 capability is active without a comparison
selector and all three trajectories/state/lifecycle are exact. Evidence:
[`clean mixed32 exp16 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp16-production.json).

The bounded issue-width screen closes with an exact wave32 successor. Every
lane issues one compiler `expf` for each thirty-two-slot batch, then returns
the weights to the unchanged ordered denominator/PV chains. The leaf improves
**0.082313 -> 0.081551 ms (-0.93%)** and the stable cached window improves
**77.185 -> 76.838 us (-0.45%)** at unchanged
VGPR104/SGPR128/LDS24576/scratch0. All seven resident p512/d128 pairs improve
**19.524103 -> 19.538164 tok/s (+0.0720%, -0.0369 ms/token)** with complete
sample separation and exact state. gfx1151 selects exp32 only inside the
qualified exp16 route; exp16 remains rollback. Evidence:
[`retained mixed32 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp32-retained.json).

Tracked-clean selector-unset production measures
**19.521938/19.530839/19.533770 tok/s**, median **19.530839**. This is
aggregate-flat at **+0.0038% / -0.0019 ms/token** versus clean exp16; retain
on the fully separated seven-pair A/B and positive leaf/trace. Production is
**+70.327%** over sprint start and all state remains exact. Evidence:
[`clean mixed32 exp32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-mixed32-exp32-production.json).

The exact producer-maximum sibling is now separately registered. Each of the
twelve score-producing waves accumulates a partial maximum per owned query
while producing its existing score slice, then publishes those values through
the already-required score-to-softmax barrier. Each output owner reduces
twelve partials instead of rescanning all 512 scores; QK, score storage,
exp32, ordered denominator, scalar PV FMA order, gate, and stores are
unchanged. The wrapped/evicted oracle is F32/BF16 byte-exact and the
nine-sample leaf improves **0.081790 -> 0.059101 ms (-27.74%)**. Cached
tracing names the candidate at grid32/local384, VGPR104/SGPR128/scratch0, with
LDS **24,576 -> 25,088 bytes**. The matched resident gate improves all seven
p512/d128 pairs **19.684442 -> 19.996117 tok/s
(+1.583%, -0.792 ms/token)** with complete sample separation and exact
trajectories/state/lifecycle. gfx1151 therefore selects producer maxima at the
saturated natural SWA shape; mixed32/exp32 remains the exact rollback.
Evidence:
[`producer-max leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-max-leaf.json).
[`producer-max retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-max-retained.json).
Tracked-clean selector-unset production confirms
**19.983610 tok/s**, **+1.606% / -0.804 ms/token** over the prior clean
19.667705 packet, with the capability active and exact repeated
state/lifecycle:
[`producer-max production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-max-production.json).
The tracked-clean 127-transition census measures
**3.688 ms/token attention = 2.503 SWA + 1.178 global**. Relative to the
pre-producer census, SWA falls **21.36%**, total attention **15.52%**, and
kernel sum **1.46%**. The residual attention gap to same-GGUF Vulkan is
**2.778 ms/token**, **38.5%** of the complete wall gap:
[`post-producer-max census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-producer-max-wall-reprofile.json).

The same ownership rule now has a separately registered exact global
candidate. Each of the eight score waves accumulates the maximum over its
existing strided token sequence and publishes into the existing 2x8 F32 warp
buffer before the score barrier. The final wave-order maximum reduction,
scores, exp32, denominator, scalar PV, gate, and stores are unchanged; the
materialized-score reread and one workgroup barrier disappear. The
live513/576/639 eviction oracle is F32/BF16 byte-exact, and nine-sample leaves
improve **4.50%/4.89%/4.88%**. Cached tracing keeps
grid8192/local256/SGPR128/LDS512/scratch0 while VGPR falls **56 -> 48**.
All seven resident p512/d128 pairs improve
**19.978296 -> 19.993586 tok/s (+0.0765%, -0.0383 ms/token)** with complete
sample separation and exact trajectories/state/lifecycle. gfx1151 promotes
producer maxima only inside the qualified mixed32/exp32 route; mixed32/exp32
remains exact rollback:
[`global producer-max leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-max-leaf.json).
[`global producer-max retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-max-retained.json).
Tracked-clean selector-unset production is
**19.982796/19.988868/19.986371 tok/s**, median **19.986371**:
aggregate-flat-to-positive at **+0.0138% / -0.0069 ms/token** versus the
prior clean packet, with the capability active and exact repeated state:
[`global producer-max production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-producer-max-production.json).

The tracked-clean 127-transition follow-up keeps **768 compute
dispatches/token** and measures
**3.658 ms/token attention = 2.497 SWA + 1.149 global**. Relative to the
preceding census, global falls **2.50%**, total attention **0.80%**, kernel
sum **0.07%**, and span **0.06%**. The residual attention gap to same-GGUF
Vulkan is **2.749 ms/token**, **38.2%** of the complete wall gap. The global
candidate appears at grid8192/local256/VGPR48/SGPR128/LDS512/scratch0:
[`post-global-producer-max census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-global-producer-max-wall-reprofile.json).

The llama.cpp output-ownership audit exposed one smaller exact SWA
specialization. The fused gated decode runner consumes only the BF16 gated
context, so a separately registered producer-max/producer-gate sibling omits
the dead 9,216-element F32 context store. The wrapped/evicted fixture keeps a
`-123.5` sentinel untouched and produces byte-identical gated BF16 output.
All nine paired 50-launch samples improve
**0.058948 -> 0.058681 ms (-0.453%)**. Cached tracing names the final-false
template at unchanged grid32/local384/VGPR104/SGPR128/LDS25,088/scratch0.
The resident p512/d128 gate is noise-level and rejected:
**20.060575 -> 20.063738 tok/s (+0.0158%)**, but only 6/7 pairs improve and
one loses **0.0713%**, larger than the modeled saving. Its capability, cache
field, session setter, and profile switch are removed. The analogous global
specialization is also rejected and removed: live 513/576/639 regress
**0.068%/0.045%/0.043%**, winning only 3/2/1 of nine pairs. Production
remains **20.056756 tok/s**. Evidence:
[`retained SWA gated-only leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gated-only-leaf.json),
[`rejected SWA runtime`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gated-only-runtime-rejected.json),
[`rejected global gated-only leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-gated-only-rejected.json).

Plain FP64 PV substitution is also closed. A temporary mixed32
producer-max/producer-gate sibling preserves the exact QK, maximum, compiler
`expf`, ordered F32 denominator, ownership, divide, gate, and stores, while
accumulating each dimension's 512 PV terms in serial FP64 before one F32
round. The wrapped/evicted oracle stays close but changes **5/9,216** gated
BF16 values, and the leaf regresses
**0.058978 -> 0.165942 ms (+181.36%)**. The specialization, wrapper, key, and
harness seam are removed before trace or model-quality work. Higher
mathematical precision does not reproduce the recurrent scalar-F32 boundary
and is not a throughput bridge to cooperative PV:
[`rejected FP64 PV`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-fp64-pv-rejected.json).

The remaining scalar synchronization contraction is closed as well. The
final 64-slot V tile has no successor that can overwrite LDS, so a temporary
specialization omitted only its post-consume workgroup barrier and remained
F32/BF16 byte-exact through wrap and eviction. A directional 9x50 screen
improved **0.058923 -> 0.058824 ms (-0.168%)**, but the decisive 21x100 gate
reversed to **0.058735 -> 0.058774 ms (+0.066%)**, with only 11/21 pairs
positive. The specialization, wrapper, registry key, harness seam, and test
extension are removed before trace/runtime work. Do not reopen scalar
barrier-only tuning:
[`rejected terminal V barrier`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-final-vbarrier-rejected.json).

The exact scalar form of llama.cpp-style whole-GQA ownership is closed.
One local384 block per KV head stages all nine queries and reuses each K/V
tile and exp32 weight while preserving the production denominator and scalar
PV order. It is byte-exact but regresses the leaf
**0.058989 -> 0.138660 ms (+135.1%)** at grid8/VGPR224/LDS44,544/scratch0.
Constraining the fully unrolled loops worsens it to **0.173172 ms
(+192.8%)**. All candidate code is removed. Whole-GQA reuse requires
cooperative-matrix parallelism; scalar ownership underfills the device:
[`rejected scalar GQA9`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa9-shared-scalar-rejected.json).

Producer-side SWA score scaling is closed too. Although it removes four
repeated `dot * scale` evaluations per query/token, production fuses
`dot * scale - max`; storing the scaled score rounds before the subtraction.
F32 context changes by up to **2.79e-9**, gated BF16 happens to match the leaf
fixture, and timing is neutral at
**0.059183 -> 0.059172 ms (-0.018%)**. No model-quality run is warranted and
all candidate code is removed:
[`rejected producer-scaled scores`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-scaled-scores-rejected.json).

The separately registered exact producer-gate successor computes each owned
query's softplus once in the score phase and publishes it through the existing
barrier. QK, scores, producer maxima, exp32, denominator/PV order, final gate
multiply, BF16 rounding, grid, and resources remain unchanged. The
wrap/eviction oracle is F32/BF16 byte-exact and the leaf improves
**0.059058 -> 0.058680 ms (-0.641%)** at
grid32/local384/VGPR104/SGPR128/LDS25,088/scratch0:
[`producer-gate leaf`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-gate-leaf.json).
All seven counterbalanced resident p512/d128 pairs improve, with exact
generated state and lifecycle: median decode is
**19.992650 -> 20.012052 tok/s (+0.097%)**. gfx1151 therefore promotes the
producer-gate specialization while preserving producer-max as its exact
rollback:
[`producer-gate retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-gate-retained.json).
Clean selector-unset production is **20.003064 tok/s**, **+0.0835%** over the
prior packet and **+74.445%** over sprint start, with exact repeated
state/lifecycle:
[`producer-gate production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-producer-gate-production.json).
The post-promotion census attributes the exact micro-win to SWA:
**2.497126 -> 2.490833 ms/token (-0.252%)** at unchanged resources; total
kernel sum improves **0.043%**. The temporary comparison seam is removed:
[`post-producer-gate census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-producer-gate-wall-reprofile.json).

The clean post-wave32 census keeps **816 dispatches/token** and measures
**48.966 ms/token** kernel sum / **51.519 ms/token** span. Attention is
**4.478 ms/token = 3.181 SWA + 1.289 global**, down **5.62%** from post-exp4
but still **3.568 ms/token** behind same-GGUF llama.cpp Vulkan and **42.6%**
of the complete wall gap. The next exact structural gate is a fused
4+5-query local512 owner: 16 workgroups, two output dimensions per lane, no
global score plane, and roughly 2x staged-V reuse. Evidence:
[`post-exp32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-exp32-wall-reprofile.json).

The exact global-attention owner now has a mixed32 specialization for the
natural 48Q/8KV/D128/capacity-4096 route through live4000. It maps each
six-query GQA group as **2+2+1+1**, raising the grid from 24 to 32 local256
blocks while retaining the GQA2 exp32 QK/softmax/PV association. Idle waves in
singleton owners remain barrier-active for every staged-V tile. The evicted
live513/576/639 fixture is F32/BF16 byte-exact; leaves improve
**5.19%/8.39%/8.39%**, and all seven resident p512/d128 pairs improve
**19.641357 -> 19.668893 tok/s (+0.1402%, -0.0713 ms/token)** with exact
generated state. Cached tracing names grid8192/local256 at
VGPR56/SGPR128/static-LDS512/scratch0. gfx1151 selects mixed32 inside the
qualified GQA2-exp32 route; the 24-block primitive remains registered exact
rollback. Evidence:
[`retained global mixed32 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-mixed32-exp32-retained.json).
Tracked-clean selector-unset production on `ab2ea899c` measures
**19.660256/19.667705/19.670663 tok/s**, median **19.667705**:
**+0.1917% / -0.0975 ms/token** over the preceding clean packet and
**+71.520%** over sprint start, with exact repeated state. Evidence:
[`clean global mixed32 production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-mixed32-exp32-production.json).
The mixed32 producer-max owner now also has a separately registered exact
DPP-QK sibling. It preserves the scalar four-FMA QK body and the retained
**+16,+8,+4,+2,+1** F32 tree, replacing only `ds_bpermute` shuffle transport
with `permlanex16` plus DPP moves. The live513/576/639 eviction oracle is
F32/BF16 byte-exact, while cached 9x50 leaves improve
**14.51%/7.05%/6.73%**. Cache-only tracing names the intended local256/
grid8192 body at VGPR48/SGPR128/LDS512/scratch0. All seven counterbalanced
resident p512/d128 pairs improve, with complete sample separation and exact
generated state: median decode is
**20.088665 -> 20.114355 tok/s (+0.128%, -0.064 ms/token)**. gfx1151
therefore promotes DPP transport on the qualified producer-max route; the
registered shuffle sibling is exact rollback and peer backends remain
unchanged. Evidence:
[`global DPP-QK primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-dpp-qk-primitive.json) ·
[`global DPP-QK retained`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-dpp-qk-retained.json).
Tracked-clean selector-unset production is
**20.088017/20.105078/20.116745 tok/s**, median **20.105078**:
**+0.1767% / -0.0879 ms/token** over the preceding 20.069608 packet and
**+75.335%** over sprint start. The comparison-only session/profile seam is
removed; the capability selects the registered sibling directly:
[`global DPP-QK production`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-dpp-qk-production.json).
Applying the same exact transport substitution to the saturated SWA owner is
closed at the resident gate. Its wrap/eviction oracle is F32/BF16 byte-exact,
and the cached 9x50 leaf improves
**0.058897 -> 0.055084 ms (-6.47%)** at unchanged grid12288/local384,
VGPR104/SGPR128/LDS25088/scratch0. However, all seven resident p512/d128
pairs regress, with median decode
**20.103985 -> 20.093891 tok/s (-0.0502%, +0.0250 ms/token)**. The temporary
runtime selector is removed, production remains **20.105078 tok/s**, and the
registered DPP sibling remains diagnostic only. This local384 body needs a
materially different cooperative tile or resource profile, not another
lane-transport-only retry. Evidence:
[`rejected SWA DPP-QK runtime`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-dpp-qk-runtime-rejected.json).
The next exact saturated-SWA specialization transfers the comparator's
probability-tile reuse without its lower-precision cooperative arithmetic.
One output wave per active query computes the unchanged wave32 `expf` weights
and exact ordered denominator while the remaining waves load the current K64
V tile. The already-required V publication barrier also publishes a
**3 x 64** FP32 probability tile, allowing all four output-dimension waves to
reuse each weight without another barrier. Compacting V loads across the ten
or nine non-producer waves is required; with it, the byte-exact leaf improves
**0.058734 -> 0.055996 ms (-4.662%)** and all seven resident p512/d128 pairs
improve **20.097968 -> 20.282916 tok/s
(+0.9202%, -0.4537 ms/token)** with complete separation. Cached tracing keeps
grid32/local384, VGPR104, SGPR128, and scratch0; LDS rises only
**25,088 -> 25,600 bytes**. gfx1151 selects
`swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vstage64_vec16_direct_assume_exp_fixed512_spans`
only for the saturated natural shape. The producer-max/gate sibling remains
registered exact rollback; peer and non-natural routes are unchanged.
Evidence:
[`retained stage probability cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-retained.json).
Tracked-clean selector-unset production is
**20.260703/20.278430/20.270314 tok/s**, median **20.270314**:
**+0.8219% / -0.4055 ms/token** over the preceding clean packet and
**+76.776%** over sprint start. The new capability is active without a
comparison route and exact repeated state/lifecycle passes:
[`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-production.json).
The clean sorted two-queue census contains 127 one-token decode segments at
**721 dispatches/token**, **47.554 ms** kernel sum, and **49.825 ms** span.
Attention is **3.354 ms/token = 2.238 SWA + 1.107 global**, down
**0.304 ms / 8.31%** from the post-producer-gate census but still
**2.444 ms/token** behind same-GGUF llama.cpp Vulkan attention. The comparator
gets there by collapsing grouped queries into a cooperative tile, retaining
online-softmax state tile-locally, and reusing one published probability tile
for PV; its F16 K/V and lower-precision cooperative arithmetic are not an
exact drop-in for the BF16 recurrent contract. The next exact port is the
same probability/V-stage publication schedule on the 12 global layers.
Evidence:
[`post-stage-cache census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-stage-pcache-wall-reprofile.json).
The proposed global stage-cache copy is closed by source audit: global
mixed32 already caches every probability once, normalizes before PV, and
shares that plane across four output waves. SWA's unnormalized-numerator then
divide schedule would change its FP32 association. The useful exact
continuation is instead the combined SWA stage-cache plus DPP-QK primitive.
It keeps every QK product and the **+16,+8,+4,+2,+1** F32 tree while replacing
only lane transport. The wrapped/evicted oracle is byte-exact and the cached
9x50 leaf improves **0.056299 -> 0.052299 ms (-7.105%)** with complete
separation. Tracing is unchanged at grid32/local384, VGPR104, SGPR128,
LDS25,600, and scratch0. The seven-pair resident p512/d128 gate rejects the
route despite the isolated win: every candidate pair loses and median decode
moves **20.276057 -> 20.260314 tok/s
(-0.0776%, +0.0383 ms/token)** with exact state/lifecycle. Remove the
comparison capability, cache route, and profile CLI; keep the registered
primitive diagnostic-only. Production remains **20.270314 tok/s**:
[`combined stage-cache/DPP primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-dpp-qk-primitive.json),
[`runtime rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-dpp-qk-runtime-rejected.json).
Recombining the retained stage-probability schedule with exact
GQA4+5/local512 ownership is also closed. The wrapped/evicted output is
byte-exact, but halving workgroups **32 -> 16** regresses the leaf
**0.056133 -> 0.061001 ms (+8.673%)**. All candidate code is removed;
production remains the 32-block stage-cache owner. Evidence:
[`rejected GQA4+5 stage cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa45-stage-pcache-rejected.json).
The intermediate 24-block/all-active GQA3 recombination is byte-exact too,
but regresses **0.056152 -> 0.059231 ms (+5.484%)**. The retained
probability cache therefore does not move the gfx1151 scalar ownership seam:
16 and 24 blocks lose, while 32 blocks remains production. Evidence:
[`rejected GQA3 stage cache`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa3-stage-pcache-rejected.json).
The next mixed32 schedule uses the otherwise-idle waves in each pair-owner
block as exact probability/denominator producers. That frees all eight active
output waves to participate in compact K64 V staging while preserving every
QK, maximum, `expf`, denominator, PV, and BF16 operation; triple-owner blocks
keep the production schedule. Wrapped/evicted outputs are byte-exact. The
9x50 leaf improves **0.056018 -> 0.055849 ms (-0.303%)**, and a stronger
21x100 screen confirms **0.056164 -> 0.055990 ms (-0.309%)** with **20/21**
paired wins. Cache-only tracing leaves grid32/local384, VGPR104, SGPR128,
LDS25,600, and scratch0 unchanged. The resident gate is positive but not
robust: median decode moves only
**20.279694 -> 20.283354 tok/s (+0.0180%, -0.0089 ms/token)**, with six
paired wins but one **-0.0207% / +0.0102 ms/token** loss larger than the
median saving. Remove all comparison runtime plumbing and retain only the
registered diagnostic primitive; production remains **20.270314 tok/s**:
[`idle-wave producer primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-idle-producer-primitive.json),
[`runtime rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-idle-producer-runtime-rejected.json).
Replaying the exact denominator from the published K64 probability tile is
closed. The candidate removes 64 wave-shuffle transports per query/stage and
performs the identical visible-slot sequence through LDS on producer lane
zero after the existing publication barrier. Wrapped/evicted F32/BF16 output
is byte-exact, but all nine cache-hot samples regress
**0.056170 -> 0.060017 ms (+6.849%)**. Serial LDS replay delays the producer
wave more than the shuffles cost; all candidate code is removed and
production remains **20.270314 tok/s**. Evidence:
[`post-barrier denominator rejection`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-postbarrier-denom-rejected.json).
The aligned-vector successor fixes that implementation bottleneck. Producer
lane zero reads the published K64 probability row as sixteen `float4` LDS
vectors, then performs the identical 64 ordered adds while other waves
compute PV. Wrapped/evicted F32/BF16 output is byte-exact. The 9x50 and
21x100 leaves improve **19.443%/19.445%**, with complete 21-sample separation.
Cache-only tracing keeps grid32/local384, VGPR104, SGPR128, LDS25,600, and
scratch0 unchanged. Retain
`swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_vstage64_vec16_direct_assume_exp_fixed512_spans`.
All seven resident p512/d128 candidate samples beat every control, moving
**20.277561 -> 20.368173 tok/s
(+0.4469%, -0.2194 ms/token)** with exact state/lifecycle. gfx1151 promotes
the qualified gated saturated-SWA capability; shuffle replay remains
registered exact rollback and peer backends are unchanged. Remove the
comparison-only profile seam. Tracked-clean selector-unset production is
**20.351478/20.360810/20.358649 tok/s**, median **20.358649**:
**+0.4358% / -0.2141 ms/token** over the preceding clean packet and
**+77.546%** over sprint start. The normal route reports the capability
active without a comparison selector and preserves exact repeated
trajectory/state/lifecycle:
[`vec4 denominator primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-denom-primitive.json),
[`resident retention`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-denom-retained.json),
[`clean production`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-denom-production.json).
The clean 127-transition census confirms SWA
**2.237644 -> 2.018186 ms/token (-9.808%)**, total attention
**3.353534 -> 3.135648 ms/token (-6.497%)**, and kernel sum
**47.554087 -> 47.296538 ms/token (-0.542%)**. Global attention remains
effectively flat at **1.110485 ms/token**. Attention is still
**2.226225 ms/token** slower than same-GGUF Vulkan and **35.40%** of the
remaining production wall gap. Next vectorize exact contiguous probability
reads inside the retained SWA PV chain without changing its 64-FMA order:
[`post-vector-denominator census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-vec4-denom-wall-reprofile.json).
Retain the registered exact
`swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans`
primitive. It reads the contiguous K64 probability
row as sixteen aligned `float4` values, then issues x/y/z/w FMAs in the
original slot 0..63 order. Wrapped/evicted F32/BF16 output is byte-exact.
The 9x50 and 21x100 leaves improve **0.325%/0.290%**, with **21/21** paired
wins in the stronger screen. Trace resources remain grid32/local384,
VGPR104, SGPR128, LDS25,600, scratch0. All seven resident candidate runs beat
their paired controls, moving **20.366610 -> 20.379415 tok/s
(+0.06287%, -0.03085 ms/token)** with exact state/lifecycle. gfx1151
promotes the capability, scalar PV reads remain exact rollback, and the
comparison seam is removed. Tracked-clean selector-unset production is
**20.335685/20.349871/20.352342 tok/s**, median **20.349871**. The absolute
checkpoint is **0.0431%** below the preceding clean packet, inside shared-APU
variance; retention rests on the stronger **7/7** paired gate. The normal
route reports the capability active and preserves exact repeated
trajectory/state/lifecycle:
[`vectorized probability primitive`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-stage-pcache-vec4-probability-primitive.json),
[`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-stage-pcache-vec4-probability-retained.json),
[`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-stage-pcache-vec4-probability-production.json).
The same scalar-consumption seam is larger in global attention. Its retained
kernel already stores complete normalized probability planes in LDS, but PV
loads one scalar per token. Retain the separately registered exact
`global_context_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_vstage64_vec16_direct_assume_exp_fixedshape_spans`
primitive. It pads each probability plane to a four-float stride so both
query planes remain 16-byte aligned at live513/live639, then consumes one
`float4` per four original PV iterations without changing normalization,
positive tests, BF16 V conversion, or accumulation order. The explicit
eviction oracle is F32/BF16 byte-exact. Strong 21x100 leaves improve
live513/576/639 **13.006%/17.248%/10.395%**, with complete separation at
every shape. Cache-only tracing names grid8192/local256, VGPR48, SGPR128,
static-LDS512, scratch0 and no compiler. The 12-global-layer gate passes with
complete separation: all seven candidates beat every control, moving
**20.373406 -> 20.409544 tok/s
(+0.17738%, -0.08691 ms/token)** with exact state/lifecycle. gfx1151
promotes the qualified capability, scalar probability replay remains exact
rollback, and comparison plumbing is removed. Tracked-clean selector-unset
production is **20.403940/20.414792/20.418871 tok/s**, median
**20.414792**, or **+0.3190% / -0.1563 ms/token** over the preceding clean
packet and **+78.036%** over sprint start. The normal route reports the
capability active and preserves exact repeated trajectory/state/lifecycle:
[`global vector probability primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-vec4-primitive.json),
[`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-vec4-retained.json),
[`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-vec4-production.json).
The exact registered
`global_context_fused_exact_gated_mixed32_exp32_producer_max_dpp_qk_probability_vec4_prenorm_vstage64_vec16_direct_assume_exp_fixedshape_spans`
diagnostic removes another source of repeated work: once the reciprocal is
known, one cooperative pass writes each `exp * reciprocal` FP32 result back
to LDS, so all 128 output lanes consume identical normalized bits instead of
repeating that multiply. The eviction oracle is F32/BF16 byte-exact.
Strong 21x100 live513/576/639 leaves improve **1.640%/1.503%/0.128%** with
**21/21, 21/21, and 19/21** paired wins. Cache-only tracing names the distinct
final-`true` specialization at grid8192/local256, VGPR48, SGPR128,
static-LDS512, and scratch0. Seven counterbalanced p512/d128 resident pairs
preserve exact state and move decode **20.501353 -> 20.503954 tok/s
(+0.01269%)**, with **5/7** paired wins. gfx1151 selects the exact sibling by
capability; the preceding probability-vec4 key remains fallback and other
backends do not inherit it:
[`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-prenorm-primitive.json),
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-prenorm-retained.json).
Tracked-clean selector-unset production is
**20.489386/20.496816/20.498178 tok/s**, median **20.496816**, with exact
trajectory/state/lifecycle:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-probability-prenorm-production.json).
The post-promotion 127-transition census measures **46.910112 ms/token**
kernel sum, **49.119568 ms/token** dispatch span, and
**2.746352 ms/token** attention =
**1.754009 SWA + 0.992343 global**. Relative to the post-mixed40 checkpoint,
global falls another **1.197%** while SWA stays flat. Same-GGUF Vulkan remains
at **0.909423 ms/token** attention, leaving **1.836929 ms/token**, or
**30.83%** of the total production wall gap:
[`post-prenorm census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-global-prenorm-wall-reprofile.json).
An explicit one-reciprocal-per-query SWA diagnostic proves final division is
not that gap: removing **9,144** output-lane divisions improves the leaf only
**0.391%**, changes F32 context bits, and estimates just
**0.00525 ms/token** across 36 layers. It is removed before resident or
quality gating:
[`rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-reciprocal-normalize-rejected.json).
An FP16 scaled-score-plane sibling isolates another narrow part of Vulkan's
low-precision tile while retaining scalar F32 QK/PV. Halving score-plane LDS
bytes **6,144 -> 3,072** and removing score-scale replay is performance-flat
(**+0.011%**) and changes 30 gated BF16 values, so it is removed before
resident/quality work:
[`score-storage rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-fp16-scaled-scores-rejected.json).
The preceding post-vector-probability 127-transition census measured
**721 compute + 5
runtime-copy dispatches/token**, **47.174209 ms/token** kernel sum, and
**3.023432 ms/token** attention. Global falls
**1.110485 -> 1.005649 ms/token (-9.441%)** while SWA remains
**2.017783 ms/token**. Attention is still **2.114009 ms/token** above
same-GGUF Vulkan and **34.35%** of the remaining production wall gap:
[`post-global-probability census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-global-probability-vec4-wall-reprofile.json).
The separately registered exact
`swa_context_fused_exact_gated_mixed32_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans`
primitive moves the unchanged vectorized denominator replay onto otherwise-idle
waves 8/9 in the 24 pair-owner blocks while all eight active output waves
execute the unchanged PV chains; triple-owner blocks retain the production
schedule. The wrap/eviction oracle is F32/BF16 byte-exact. The stronger 21x100
leaf improves **0.045329 -> 0.045182 ms (-0.324%)** with **21/21** paired
wins at unchanged grid12288/local384, VGPR104, SGPR128, LDS25,600, and
scratch0. All seven resident p512/d128 candidates beat every control with
complete separation, moving **20.411948 -> 20.430138 tok/s
(+0.08912%, -0.04362 ms/token)** with exact state/lifecycle. gfx1151
promotes the qualified capability, active-wave replay remains exact rollback,
comparison plumbing is removed, and peer backends are unchanged. Clean
selector-unset production is **20.412363/20.425412/20.429048 tok/s**, median
**20.425412**, or **+0.05202% / -0.02547 ms/token** over the preceding clean
packet and **+78.128%** over sprint start. The normal route reports the
capability active and preserves exact repeated state/lifecycle:
[`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-idle-vector-denom-primitive.json),
[`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-idle-vector-denom-retained.json),
[`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-idle-vector-denom-production.json).
The separately registered exact
`swa_context_fused_exact_gated_mixed40_exp32_producer_max_gate_stage_pcache_idle_vec4_denom_probability_vstage64_vec16_direct_assume_exp_fixed512_spans`
primitive replaces each KV head's **2+2+2+3** query ownership with
**2+2+2+2+1** in one launch. The grid grows 32 -> 40 blocks, matching the 40
gfx1151 CUs and removing triple-query critical blocks at the cost of 25% more
K/V-owner traffic. The wrap/eviction oracle is F32/BF16 byte-exact. Strong
21x100 leaves improve **0.045322 -> 0.037599 ms (-17.039%)** with complete
separation at unchanged local384/VGPR104/SGPR128/LDS25,600/scratch0.
All seven resident p512/d128 candidates beat every control with complete
separation, moving **20.433014 -> 20.501083 tok/s
(+0.33313%, -0.16249 ms/token)** with exact state/lifecycle. gfx1151 promotes
the qualified capability, mixed32 remains exact rollback, comparison plumbing
is removed, and peer backends are unchanged. Tracked-clean selector-unset
production is **20.483884 tok/s**, **+0.28627% / -0.13975 ms/token** over the
preceding clean packet and **+78.638%** over sprint start:
[`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-primitive.json),
[`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-retained.json),
[`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-production.json).
The subsequent tracked-clean 127-transition trace measures the complete
transfer: SWA falls **2.017783 -> 1.757218 ms/token (-12.913%)**, total
attention falls **3.023432 -> 2.761582 ms/token (-8.661%)**, and kernel span
falls **2.928%**. Keep mixed40 geometry for the next body-scheduling screen:
[`post-mixed40 census`](../benchmarks/results/2026-07-30-gfx1151-laguna-post-mixed40-wall-reprofile.json).
The exact tail-producer sibling assigns exponent generation to waves 10/11,
ordered denominator replay to waves 8/9, and ordered PV to waves 0-7 without
changing geometry or arithmetic. Wrapped/evicted output is F32/BF16
byte-exact; the strong 21x100 leaf improves
**0.037001 -> 0.036896 ms (-0.285%)** with **20/21** paired wins and
unchanged resources. Six of seven resident p512/d128 pairs improve and median
decode moves **20.502555 -> 20.508345 tok/s
(+0.02824%, -0.01377 ms/token)**; the sole losing-pair magnitude is smaller
than the median paired gain. gfx1151 promotes the capability, the prior
mixed40 schedule remains exact rollback, and comparison plumbing is removed.
Tracked-clean selector-unset production is **20.494732 tok/s**,
**+0.05296% / -0.02584 ms/token** over the preceding clean packet and
**+78.733%** over sprint start:
[`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-tail-producer-primitive.json),
[`resident retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-tail-producer-retained.json),
[`clean production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-tail-producer-production.json).
The exact local512 successor keeps that 40-owner grid and every arithmetic
association while adding four score/transport wave32s per workgroup. The
wrapped/evicted oracle is F32/BF16 byte-exact; 9x50 and 21x100 leaves improve
**17.814%/18.623%**, with all 21 strong pairs positive. Seven resident
p512/d128 pairs all improve **20.472516 -> 20.542123 tok/s (+0.34000%)** and
preserve the complete generated-state/lifecycle contract. Cache-only
production tracing records exactly **4,572 = 36 x 127** candidate calls at
grid40/local512, **VGPR32/SGPR128/LDS25,600/scratch0**. gfx1151 promotes the
qualified capability; local384 remains exact rollback and peer backends are
unchanged:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-local512-retained.json).
Tracked-clean selector-unset production is
**20.550788/20.559001/20.557302 tok/s**, median **20.557302**, or
**+0.29510% / -0.14355 ms/token** over the preceding clean packet and
**+79.278%** over sprint start, with exact repeated trajectory/state/lifecycle:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-local512-production.json).
The exact local512 producer-value-tail successor reopens only the transport
ownership seam that failed at local384. Its two tail-probability waves copy
the final 64/32 staged-V vectors in pair/singleton blocks while the unchanged
14/15 loader waves copy the prefix. QK, softmax, denominator, and PV
association are unchanged. The strong 21x100 leaf improves
**0.031737 -> 0.030061 ms (-5.282%)** with all 21 pairs positive and
byte-identical F32/BF16 output. Native tracing remains grid40/local512,
**VGPR32/SGPR128/LDS25,600/scratch0**. All seven actual-model p512/d128
pairs improve **20.718104 -> 20.737481 tok/s (+0.09353%,
-0.04510 ms/token)** with exact state/lifecycle. gfx1151 promotes the
capability, the preceding local512 route remains exact rollback, and peer
backends are unchanged:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-value-tail-retained.json).
Tracked-clean selector-unset production is
**20.715636/20.731612/20.732043 tok/s**, median **20.731612**, or
**+0.06821% / -0.03290 ms/token** over the preceding clean packet and
**+80.799%** over sprint start, with exact repeated trajectory/state/lifecycle:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-value-tail-production.json).
The exact V-stage128 successor reopens the wider-stage seam only after
local512, shared probability replay, tail probability producers, and
producer-wave V-tail transport changed the synchronization balance. It keeps
all arithmetic associations and the 40-owner/local512 grid, but cuts the fixed
512-slot replay from eight stages/sixteen barriers to four stages/eight
barriers. The byte-exact 21x100 leaf improves
**0.031216 -> 0.029120 ms (-6.717%)** with all 21 pairs positive. Native
tracing records grid40/local512, **VGPR176/SGPR128/LDS43,008/scratch0**.
Seven resident p512/d128 pairs move **20.736052 -> 20.745421 tok/s
(+0.04518%, -0.02178 ms/token)**, median paired **+0.04193%**, with **6/7**
wins and exact state/lifecycle. gfx1151 promotes V128 for the saturated
natural shape; the preceding V64 symbol remains registered rollback and peer
backend selection is unchanged:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-vstage128-retained.json).
Tracked-clean selector-unset production is
**20.744351 tok/s (48.20589 ms/token)**, **+0.06145% / -0.02962 ms/token**
over the preceding clean packet and **+80.910%** over sprint start:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-vstage128-production.json).
The exact dual-tail-producer successor keeps that V128 body and all eight
pair-block PV waves, but assigns two independent exponent waves to each query.
Each producer evaluates two 32-slot chunks per stage instead of four; ordered
denominator replay and every PV FMA are unchanged. The byte-exact 21x100 leaf
improves **0.030752 -> 0.029131 ms (-5.271%)**. Native tracing remains
grid40/local512 at **VGPR176/SGPR128/LDS43,008/scratch0**. Seven resident
p512/d128 pairs move **20.806774 -> 20.809401 tok/s
(+0.01262%, -0.00607 ms/token)** with **5/7** wins and exact
trajectory/state/lifecycle. gfx1151 promotes the dual producer at the
saturated natural shape; single-producer V128 remains registered rollback and
peer backend selection is unchanged:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-dual-tail-producer-vstage128-retained.json).
Tracked-clean selector-unset production at `72ed34b08` is
**20.803739 tok/s (48.06828 ms/token)**, a noise-floor
**+0.00264% / -0.00127 ms/token** over the preceding clean packet and
**+81.428%** over sprint start, with exact repeated state/lifecycle:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-dual-tail-producer-vstage128-production.json).
The exact output-sharded-probability successor moves each V128 stage's four
32-slot probability shards onto the pair-block/singleton output waves that
consume them, keeps all eight scalar PV waves and ordered F32 association, and
uses all 16 waves for staged-V loading. The byte-exact 21x100 leaf improves
**0.030266 -> 0.028760 ms (-4.976%)** at unchanged
grid40/local512/VGPR176/SGPR128/LDS43,008/scratch0. Seven resident p512/d128
pairs move **20.803377 -> 20.816723 tok/s
(+0.06415%, -0.03082 ms/token)** with **7/7** wins and exact
trajectory/state/lifecycle. gfx1151 now selects output-sharded probability for
the saturated natural shape; dual-tail V128 remains registered rollback and
peer backend selection is unchanged:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-output-sharded-probability-vstage128-retained.json).
Tracked-clean selector-unset production at `a8a91efab` is
**20.800509 tok/s (48.07575 ms/token)**, a noise-floor
**-0.01553% / +0.00746 ms/token** versus the preceding clean packet and
**+81.399%** over sprint start. Retention rests on the exact leaf and
seven-pair evidence rather than the noisy three-run publication:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-output-sharded-probability-vstage128-production.json).
Parallel wave32 replay of the sixteen output-sharded producer maxima is
removed after the complete-model gate. It is byte-exact and improves the
9x50/21x100 leaves **2.957%/3.166%** with all pairs positive, but unchanged
grid40/local512/VGPR176/LDS43,008/scratch0 resources do not transfer to
resident decode: seven p512/d128 pairs move
**20.815600 -> 20.813188 tok/s (-0.01159%)** with only **1/7** wins. Keep
lane-0 serial maximum replay and do not retry this isolated schedule without a
larger score-production or synchronization change:
[`rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-output-sharded-parallel-max-rejected.json).
The fresh tracked-clean output-sharded wall census records
**46.214841 ms/token** device kernel sum and **48.262162 ms/token** span at
673 dispatches/token. SWA improves **2.678%** versus the post-Q4-SiLU census,
but total attention remains **1.184184 ms/token** behind same-GGUF Vulkan.
Selected Q4 gate/up is the next bounded production gate: retained T16 streams
**1.709507 GB/token** at **203.83 GB/s**, while the already-retained exact
qmicro primitive is byte-neutral at **1.663304 GB/token**. Extend qmicro to
the production tile8/parallel-tail/fused-SiLU boundary before considering a
resident route:
[`wall census`](../benchmarks/results/2026-07-30-gfx1151-laguna-output-sharded-wall-reprofile.json).
That production-shaped qmicro screen is now closed. All three implementations
are BF16-byte exact, but cooperative LDS expansion, direct per-lane record
decode, and lane-0 dword/wave broadcast regress the actual layer-1 gate/up
leaf by **27.408%/21.124%/46.396%** respectively. The 2.778% resident-byte
reduction cannot amortize scale/min unpack at c=1. The candidate and all
comparison plumbing are removed; T16 remains the production layout:
[`rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-qmicro-tile8-silu-rejected.json).
The retained T16 consumer now exploits the locality already present in its Q
plane: one packed byte supplies both adjacent output-column nibbles. This
changes neither layout nor arithmetic boundaries. The exact actual-weight
21x100 leaf improves **0.131761 -> 0.129199 ms (-1.945%)**, and seven
resident p512/d128 pairs improve **20.811539 -> 20.820664 tok/s
(+0.04385%)** with **7/7** wins. Cached tracing keeps both scalar-Q and
pair-Q specializations at local128/VGPR96/SGPR128/LDS512/scratch0. Pair-Q is
the gfx1151 production owner behind the existing variant name:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-pairq-retained.json).
Tracked-clean selector-unset production publishes
**20.823569/20.830515/20.832851 tok/s**, median **20.830515**, or
**+0.14426% / -0.06925 ms/token** versus the prior clean checkpoint with
exact repeated state:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-pairq-production.json).
The exact successor vectorizes the remaining adjacent coefficient transport.
Each pair now uses aligned 32-bit `d`/`dmin` reads and 16-bit scale/min reads,
then reconstructs the identical per-column F32 coefficients and preserves all
FMA/BF16/SiLU boundaries. The actual-weight 21x100 leaf improves
**0.129011 -> 0.114367 ms (-11.351%, 21/21 wins)**. Native tracing keeps
grid16384/local128/LDS512/scratch0 while reducing allocated VGPR
**96 -> 72**. All seven p512/d128 pairs improve
**20.818971 -> 20.986316 tok/s
(+0.80381%, -0.38302 ms/token)** with exact trajectory/state/lifecycle.
gfx1151 promotes pair-coefficient transport behind the existing production
variant, retains pair-Q as explicit rollback, and removes scalar-Q:
[`retention`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-paircoeff-retained.json).
Tracked-clean selector-unset production publishes
**20.965915/20.989580/20.976598 tok/s**, median **20.976598**, or
**+0.70129% / -0.33432 ms/token** versus the prior clean checkpoint and
**+82.935%** over sprint start, with exact repeated state and allocation
teardown:
[`production`](../benchmarks/results/2026-07-30-gfx1151-laguna-q4-t16-paircoeff-production.json).
Post-retention code-object inspection qualifies the profiler resource fields:
the AMDGPU metadata declares V64/V128 at **32/35 logical VGPR**, **32 SGPR**,
zero spills/private segment, and **25,564/42,716 B fixed LDS**. V128's trace
`VGPR_Count=176` is not 176 live logical registers. Its grid40 also exactly
matches the device's 40 CUs, so the larger LDS allocation does not strand a
second wave of workgroups in this launch:
[`code-object audit`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-vstage-codeobject-audit.json).
Limiting V128's 32-item probability/PV loop to unroll factor 16 is removed.
The byte-exact 21x100 leaf improves **0.030667 -> 0.029348 ms (-4.302%)**,
but native tracing leaves both variants at **VGPR176/SGPR128/LDS43,008** and
seven resident pairs move **20.752041 -> 20.751527 tok/s (-0.00248%)**.
Pragma-only unroll tuning is closed; a successor must structurally shorten the
live range and demonstrate lower resource allocation or complete-model wall:
[`rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-vstage128-bounded16-rejected.json).
Compounding the retained local512/value-tail body with four-vector
denominator prefetch is removed after the resident gate. The byte-exact
21x100 leaf improves **0.031099 -> 0.030302 ms (-2.563%)** with all 21 pairs
positive, but seven p512/d128 pairs move **20.734191 -> 20.731204 tok/s
(-0.01440%, +0.00695 ms/token)** with only **3/7** wins. Remove the kernel,
capability, runtime selector, and comparison CLI; production remains
**20.731612 tok/s**:
[`runtime rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-local512-denom-prefetch4-runtime-rejected.json).
The exact double-buffer successor is removed at the leaf stop. It reduces
staged-loop barriers **16 -> 9** but adds **17,152 B** dynamic LDS and narrows
pair-block next-stage loading to four waves; the byte-exact 9x50 leaf
regresses **0.037106 -> 0.040897 ms (+10.219%)**. Do not retry without a
different loader/overlap mechanism:
[`double-buffer rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-double-buffer-rejected.json).
The storage-only compact2 successor is also removed at the leaf stop. It
preserves all 12 wave roles and exact arithmetic while shrinking the mixed40
query-indexed structures from three rows to two (**-2,360 logical bytes**),
but the 9x50 and 21x100 leaves regress **0.017%/0.041%** with only **4/9** and
**9/21** paired wins. Do not retry unless a compounded change also reduces
VGPR pressure or changes occupancy:
[`compact2 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-compact2-storage-rejected.json).
The next exact mixed40 sibling is retained only as a diagnostic primitive.
Current-body
counters show a mixed latency/instruction workload: **51.264% memory-unit
busy**, **49.742% L2 hit**, **2,091.125 KiB fetched**, **1.764M VALU**,
**0.482M LDS**, and **0.641M SALU** instructions over 480 waves. ISA-directed
four-vector denominator prefetch issues four `ds_load_b128` operations before
the unchanged 16 ordered adds instead of one load/`lgkmcnt(0)` chain at a
time. The 9x50/21x100 leaves improve **0.410%/0.101%** byte-exactly with
**9/9** and **19/21 + one tie** wins. Trace resources remain
grid15,360/local384/VGPR104/LDS25,600/scratch0. Seven exact resident p512/d128
pairs then reject runtime ownership: median decode moves
**20.497384 -> 20.497114 tok/s (-0.00132%)**, median paired change is
**-0.00919%**, and only **3/7** pairs improve. Remove the comparison/runtime
route and retain production unchanged:
[`primitive`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-denom-prefetch4-primitive.json),
[`runtime rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-denom-prefetch4-runtime-rejected.json).
The packed-V-DPP successor keeps all eight output waves and scalar PV
association while one even lane loads each aligned BF16 pair and delivers it
to its odd neighbor. It is byte-exact, but the 9x50 leaf regresses
**0.036950 -> 0.045846 ms (+24.075%)**. Remove it before tracing or runtime
integration; cross-lane delivery does not repay the 16-bit LDS reads:
[`packed-V-DPP rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-packed-v-dpp-rejected.json).
Removing DPP does not rescue packed staged-V replay. Having both lanes read
the same aligned dword and select their own BF16 half is byte-exact but
regresses the 9x50 leaf **0.037053 -> 0.037558 ms (+1.363%)**. Remove the
candidate before tracing/runtime and retain 16-bit LDS value reads:
[`packed-V-broadcast rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-packed-v-broadcast-rejected.json).
A byte-neutral lane-major BF16 key diagnostic then replaces each lane's four
16-bit cache reads 32 elements apart with one aligned 64-bit load and ordered
extracts. It preserves F32/BF16 bytes under wrap and explicit eviction, but its
9x50 **0.164%** gain contracts to only **0.069%** at 21x100. Remove the
candidate before trace/runtime: key-layout-only vectorization does not justify
migrating all KV writers/readers, and the material llama.cpp target remains
cooperative K/V reuse plus tensorized QK/PV:
[`lane-major-key rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-lane-major-key-rejected.json).
Applying `__restrict__` to every independent retained mixed40 pointer is also
exact but does not improve the generated schedule measurably. Three pre/post
21x100 processes move the production-tail median
**0.037002 -> 0.037097 ms (+0.259%)**. Remove the compiler-only annotation:
[`restrict/noalias rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-mixed40-restrict-noalias-rejected.json).
Do not use LLVM's direct global-to-LDS intrinsic for gfx1151 attention.
Although the retained staged-V ISA exposes a concrete
`global_load_b128 -> s_waitcnt vmcnt(0) -> ds_store_b128` dependency chain,
the gfx1151 compile rejects `__builtin_amdgcn_global_load_lds` because the
target lacks `vmem-to-lds-load-insts`. The diagnostic candidate was removed
before benchmarking. Keep the ordinary supported instructions and pursue
multi-load issue/source prefetch if this dependency is revisited:
[`unsupported global-to-LDS load`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-global-load-lds-unsupported.json).
The supported two-load version is also closed. Issuing two ordinary
`global_load_b128` value reads before either LDS store preserves F32/BF16
bytes but regresses the paired 9x50 leaf
**0.037081 -> 0.045278 ms (+22.106%)**. The diagnostic source is removed;
do not increase staged-V source-prefetch depth:
[`value-prefetch2 rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-value-prefetch2-rejected.json).
An exact transport-ownership sibling remains registered diagnostic-only.
Finished pair/singleton tail probability producers copy the final 64/32
staged-V vectors, eliminating the longest ordinary-loader iteration. The
21x100 leaf improves **0.037575 -> 0.036346 ms (-3.270%)** at unchanged
grid15,360/local384/VGPR104/LDS25,600/scratch0, but seven actual-model pairs
move median decode **20.509962 -> 20.507264 tok/s (-0.01316%)** with only
**4/7** wins. Do not select it independently; the benchmark-only registry
swap is removed:
[`producer-value-tail runtime rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-producer-value-tail-runtime-rejected.json).
The exact 40-block **2+1+1+1+1** successor is removed at the leaf stop. It
improves live513 **4.62%** but regresses live576/live639 **0.21%/0.11%**;
the fifth K/V owner crosses the gfx1151 occupancy/reuse seam at local256.
Evidence:
[`rejected global mixed40 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-mixed40-exp32-rejected.json).
The later local512 combination reverses that result and is retained. It keeps
the production eight-wave denominator tree while all sixteen waves share
independent QK/value transport, then uses the same five-owner
`2+1+1+1+1` mapping to fill all 40 gfx1151 CUs. The byte-exact 21x100 leaf
improves live513/576/639 **1.967%/2.054%/1.824%**; all seven p512/d128 model
pairs improve **20.987128 -> 20.991542 tok/s (+0.02103%)**. gfx1151 now
selects the mixed40-local512 capability and retains mixed32-local512 as exact
rollback:
[`retained global mixed40 local512`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-mixed40-local512-retained.json).
The qualified dense-prefix successor now also overlaps exact PV and V
transport. After a cooperative V64 stage-zero fill, waves without an active
query output load the next V64 plane into a second LDS buffer while active
waves preserve the retained scalar PV sequence. This halves full barriers per
stage without changing arithmetic, grid40/local512 ownership, resident bytes,
or launch count. Live513/576/639 byte-exact 21x100 leaves improve
**5.754%/3.577%/6.057%**, and all seven p512/d128 resident pairs improve
**21.865315 -> 21.891144 tok/s (+0.11813%)**. gfx1151 selects
`LAGUNA_GLOBAL_DENSE_PREFIX_IDLE_DOUBLE_BUFFER`; single-buffer dense-prefix
remains the exact peer/non-dense fallback:
[`retained global idle-wave V ping-pong`](../benchmarks/results/2026-07-30-gfx1151-laguna-global-idle-double-buffer-retained.json).
The same exact fused body is now capacity-independent for dense, identity
prefixes. Allocated KV capacity is forwarded through the `KVLiveSpans` ABI
instead of being compiled as 4,096; the scan remains resource-bounded. gfx1151
uses the double-buffered local1024 owner through 4,000 live slots, then the
single-buffered dense-prefix local512 owner through 6,000 slots. Non-dense
metadata remains limited to the local512 4,000-slot band, and every larger
live span falls back to exact generic split attention. The bounds follow LDS
usage rather than benchmark depths: two V64 stages plus score planes fit
through 4,000, while one V64 stage plus score planes fits through 6,000.
A cached gfx1151 trace at capacity 8,192 names both fixed-shape template
instantiations at live 1,024 and 4,097 with
**VGPR48/reported-static-LDS512/scratch0**; launch-time dynamic LDS follows the
resource formulas above. Both produce bit-identical FP32 context and BF16
gated output against the generic exact route. Clean capacity-131,200
production improves d1K/d4K
**11.776%/39.987%** while the mandatory 16K/64K/128K fallback gate remains
neutral-to-positive:
[`capacity-independent production`](../benchmarks/results/2026-08-01-gfx1151-laguna-capacity-independent-short-global-decode-production.json).
The exact two-launch split32 successor is removed as well. Sixteen pair-owner
local256 blocks plus sixteen singleton-owner local128 blocks preserve four
K/V owners and byte-exact output, but regress live513/576/639
**142.26%/154.39%/154.66%**. Extra-launch and exact virtual-wave costs exceed
the saved singleton PV work. Production source is restored byte-for-byte;
future global ownership work must remain single-launch. Evidence:
[`rejected global split32 exp32`](../benchmarks/results/2026-07-29-gfx1151-laguna-global-split32-exp32-rejected.json).
The tracked-clean post-mixed32 census records **768 compute + 5 runtime-copy
dispatches/token**, **48.701 ms/token** kernel sum, and **52.205 ms/token**
span. Attention is **4.365 ms/token = 3.183 SWA + 1.182 global**; global is
**5.00%** below the prior GQA2-exp32 census while SWA is flat. The remaining
**3.456-ms/token** attention gap is **43.1%** of the complete same-GGUF Vulkan
publication-wall gap. Next attention work targets saturated SWA in one launch,
but packed BF16 dot2 is now closed: its compensated form regresses the leaf
**1.05%**, while its one-term form improves only **0.17%** and fails the
18-prompt/576-step gate at max KL **1.265727** (**25.31x** the ceiling).
All candidate code is removed. The next screen must be an exact structural
single-launch design informed by the llama.cpp audit, not another approximate
instruction substitution, split merge, or output-derived repair. Evidence:
[`post-global-mixed32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-global-mixed32-wall-reprofile.json),
[`rejected QK dot2`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-qkdot2-rejected.json).
The first audit screen also closes a fresh current-template GQA2 rebuild.
Keeping the local384 launch bound fixes the old **176 -> 104 VGPR** footprint,
but 40 local256 blocks and five K/V owners regress the exact leaf
**0.081815 -> 0.086925 ms (+6.25%)**. All code is removed before production.
Ordinary 40-block GQA2 is closed independently of register pressure:
[`rejected current-template GQA2`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-gqa2-exp32-current-template-rejected.json).
The follow-up exact ping-pong screen is also closed. Two 64-slot V buffers
halve staged-V barriers **16 -> 8**, but improve the leaf only
**0.081569 -> 0.081210 ms (-0.44%)**. Static LDS rises
**24,576 -> 40,960 bytes** and clang allocates **104 -> 224 VGPRs** under
both before-consume and after-consume copy schedules. All code is removed
before production:
[`rejected V-stage64 ping-pong`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-vstage64-pingpong-rejected.json).
The comparator-derived structural tensorized-PV screen is closed as well.
Laguna's `FULL,SWA,SWA,SWA` cycle defines three complete, non-arbitrary
12-layer SWA roles. Exact QK plus exact denominator order and tensorized F32 PV
on roles 1/2/3 reaches max KL **1.590854/1.690376/4.873391** over the same
18-prompt/576-step gate, despite **97.57%/97.40%/97.22%** top-1 and only
**3.94-3.98%** directional speedup. No historical candidate code was ported
to current main. Tensorized PV is therefore inadmissible even at one-third
structural depth; do not retry arbitrary layer subsets:
[`rejected structural-role tensorized PV`](../benchmarks/results/2026-07-29-gfx1151-laguna-swa-tensorized-pv-role-rejected.json).
A rigorous full-ring BF16-bin screen also closes certified output repair.
Using a favorable PV-only `gamma_512` interval while omitting every upstream
error still marks **7,795/9,216 components (84.581%)** and all **72** query
heads uncertain. The measured full component-replay path is already
**124.398% slower** than retained exact attention, so do not rebuild the
cooperative tensor path around this repair oracle:
[`BF16-bin bound rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-attention-bf16-bin-bound-rejected.json).
The exact head-producer fusion seam is closed too. A cooperative
local512/grid40 kernel reproduces the separate local256/grid80 head
RMSNorm/RoPE/KV write with two 256-thread cohorts, grid-synchronizes, and then
runs unchanged output-sharded V128 attention. F32 context and gated BF16 bytes
are exact, but the combined 9x50 leaf regresses
**0.034534 -> 0.039296 ms (+13.788%)**, losing all pairs. Remove the complete
candidate before runtime integration. llama.cpp's remaining advantage is in
its cooperative-matrix QK/PV tile and precision/association contract, not a
scalar producer/attention launch fusion:
[`head/KV fusion rejection`](../benchmarks/results/2026-07-30-gfx1151-laguna-swa-head-kv-attention-fusion-rejected.json).

The clean post-promotion census keeps **816 dispatches/token** and measures
**49.432 ms/token** kernel sum / **51.982 ms/token** span. Attention falls
**5.466 -> 4.873 ms/token (-10.84%)**, split as **3.583 SWA + 1.280 global**.
The remaining attention path is still **3.964 ms/token** slower than the
same-GGUF llama.cpp Vulkan comparator and explains **45.0%** of the clean wall
gap. Evidence:
[`post-mixed32 wall census`](../benchmarks/results/2026-07-29-gfx1151-laguna-post-mixed32-wall-reprofile.json).

### Laguna post-350 selected-expert screens

gfx1151 source-F16 decode now applies cache-bypassing weight loads inside the
exact projection→head/KV and output-projection→add/RMSNorm composites.
Fixed-K ownership, FP32 association, BF16 boundaries, grids, launches, and
resident bytes are unchanged. The four natural leaves are byte-exact and
improve **3.015-3.509%**; the weighted source-F16 family falls
**24.417559 -> 23.597587 ms/token (-3.358%)**. All seven same-resident
p512/d128 pairs improve **22.792512 -> 22.855773 tok/s (+0.27755%)**.
Cache-only tracing names the intended non-temporal global/SWA and
K6144/K9216 templates at local256/VGPR24/LDS512/scratch0. gfx1151 defaults
the path with constructor-false rollback; peer backends retain cached loads.
Evidence:
`benchmarks/results/2026-07-31-gfx1151-laguna-f16-nontemporal-decode-retained.json`.

The gfx1151 projection→head/KV composite additionally owns two adjacent
source-F16 output columns per local256 block. This is the exact reusable-row
idea from llama.cpp Vulkan's AMD `NUM_ROWS=2` F16 DMMV specialization at
`c0bc8591e`, without the rejected wave64/local64 geometry. Both output
accumulators preserve the one-column FMA and wave32 reduction trees while
sharing each BF16 activation load/conversion and one barrier. Global/SWA
specializations trace as `<true,true,2>`/`<false,true,2>` at
local256/VGPR32/SGPR128/LDS512/scratch0, and seven complete-model pairs improve
**22.886574 -> 22.994503 tok/s (+0.47158%, 7/7)**. The ordinary output
projection stays one-column after its tile2 screen regressed **8.598-10.489%**;
gfx1100 remains on the one-column composite:
`benchmarks/results/2026-07-31-gfx1151-laguna-f16-projection-tile2-retained.json`.
Tracked-clean production at `a4c2c5d26` advances
**22.891692 -> 23.017271 tok/s (+0.54858%, -0.238335 ms/token)** at unchanged
residency:
`benchmarks/results/2026-07-31-gfx1151-laguna-f16-projection-tile2-production.json`.

gfx1151 selected Q4T16 gate/up decode also has a quality-gated adjacent-K
FP16-dot2 consumer. The local128/tile8 owner reads the existing byte-neutral
dual-interleaved resident layout, converts each adjacent BF16 activation and
dequantized Q4 pair to FP16 in registers, executes `v_dot2_f32_f16`, and keeps
FP32 accumulation plus independent BF16 gate/up boundaries before fused SiLU.
The exact pair-coefficient consumer remains the constructor-false fallback.
The production-layout layer-1 leaf improves **0.110406 -> 0.090545 ms
(-17.989%)**; cached tracing names
`q4_k_t16_selected_dual_interleaved_natural_tile8_halfdot_silu_gemv_kernel`
at local128/VGPR96/SGPR128/LDS512/scratch0. The 32-step recurrent gate measures
max/mean KL **0.008202/0.001036** and **93.75%** teacher-forced top-1.
Seven complete-model pairs improve **22.999793 -> 23.084044 tok/s
(+0.36631%, 7/7)**, and selector-unset production reaches
**23.089693 tok/s / 43.309368 ms/token** at unchanged residency:
`benchmarks/results/2026-07-31-gfx1151-laguna-selected-halfdot-decode-retained.json`
and
`benchmarks/results/2026-07-31-gfx1151-laguna-selected-halfdot-decode-production.json`.
The final simple geometry seam is also closed: local128/tile4 preserves every
halfdot BF16 output bit and resident byte, but doubles the selected-row grid
and regresses the actual layer-1 leaf
**0.092815 -> 0.094319 ms (+1.620%)**. The candidate is removed; tile8 remains
canonical:
`benchmarks/results/2026-07-31-gfx1151-laguna-selected-halfdot-tile4-rejected.json`.
The barrier-free selector→gate contraction is also closed. Recomputing the
exact corrected top-10 inside every selected local128 workgroup preserves all
router fields and BF16 output, and removes one launch, but regresses the actual
two-kernel chain **0.104777 -> 0.125216 ms (+19.506%)**. The composite is
removed:
`benchmarks/results/2026-07-31-gfx1151-laguna-selector-selected-gate-fusion-rejected.json`.
The byte-neutral K-major router continuation is closed too. A one-launch
candidate preserved all 3,072 output producers, appended eight coalesced
router tiles, and selected top-10 in the last tile. Projection/RMSNorm and
selected IDs were exact; router error stayed below **5.96e-7**. But attaching
the continuation raised every output block from **VGPR24/LDS512** to
**VGPR64/LDS2048**, regressing K6144/K9216 complete chains
**32.506%/22.111%** and the 12-full/36-SWA model **24.072%**. Launch-bounds
8 did not repair it, so the kernel, ABI, wrapper, harness, and tests are
removed:
`benchmarks/results/2026-07-31-gfx1151-laguna-output-router-kmajor-fusion-rejected.json`.
The retained source-F16 Q8 diagnostic now isolates Q, K, V, and attention
gate roles while untouched siblings remain exact. All-layer, 12-global-layer,
and 36-SWA-layer recurrent screens reject every role: maximum KL spans
**0.481401-1.033883** versus the **0.05** ceiling. Global-Q preserves 100%
top-1 but still reaches max KL **0.609830**. No one-plane Q8 consumer or
resident role sidecar is admissible; the role-aware harness remains only for
materially higher-precision/calibrated representations:
`benchmarks/results/2026-07-31-gfx1151-laguna-f16-q8-role-isolation-rejected.json`.

gfx1151 c=1 output→router decode now uses an exact isolated any-order
continuation. The source-F16 output/add/RMSNorm producer remains an ordinary
ordered dispatch and publishes `attention_projection_counters[1]` only after
the BF16 norm row is complete. The separate
`bf16_hidden_wave0_tree_anyorder` router consumes that row through
`hipExtAnyOrderLaunch`; all 256 blocks wait on the publication and use counter
slot 2 to reset the reusable gate. The normal ordered top-10 selector remains
the downstream barrier. Do not mark the producer any-order: that precursor
bypassed preceding attention/head-KV work and failed recurrent correctness at
maximum KL **20.452903**. The corrected chain passes the live predecessor
sentinel, exact 16-step state/repeat/lifecycle, and improves the natural
K6144/K9216 leaves **3.161%/2.182%**. Both kernels remain VGPR24/scratch0;
same-resident p512/d128 moves
**23.087307 -> 23.233248 tok/s (+0.63213%, -0.272079 ms/token)**:
`benchmarks/results/2026-07-31-gfx1151-laguna-output-router-anyorder-retained.json`.
Tracked-clean default-on production reaches
**23.231783 tok/s / 43.044479 ms/token**, **+0.61538%** over the prior
headline, at unchanged residency:
`benchmarks/results/2026-07-31-gfx1151-laguna-output-router-anyorder-production.json`.

gfx1151 Q6T16 LM-head decode now emits one exact top-1 pair from each existing
16-logit producer tile and finalizes only those 6,272 pairs. This removes the
full-logit argmax stage-1 scan and one model launch while preserving all logits,
minimum-index ties, and device-owned control publication. The producer is
local128/VGPR104/LDS512/scratch0; the final reducer is
local256/VGPR16/LDS0/scratch0. The completion-counter/last-producer sibling is
performance-rejected and removed. Constructor false retains the ordinary
logits + two-stage argmax chain. Evidence:
`benchmarks/results/2026-07-31-gfx1151-laguna-q6-lm-head-tilemax-retained.json`.
Tracked-clean selector-unset production at `c882f7bd4` measures
**22.873989 tok/s / 43.717779 ms/token**, up **0.03696%** and
**0.016157 ms/token** from the preceding clean checkpoint with exact state,
unchanged residency, and a remaining **0.888256-ms/token** same-GGUF Vulkan
gap:
[`Q6 tilemax production`](../benchmarks/results/2026-07-31-gfx1151-laguna-q6-lm-head-tilemax-production.json).

gfx1151 c=1 now reuses the exact routed/shared branch-concurrency resources
previously admitted for row-batched prefill. After router correction selection,
the specialized shared Q4T16 gate/up+SiLU and shared-down kernels run on the
least-priority nonblocking stream while the caller executes the selected
Q4/Q6 T16 path; a timing-disabled event joins immediately before the exact MoE
tail. The schedule preserves **482 kernels/token**, moves **94** to the second
queue, adds no resident bytes, and keeps the serial constructor rollback.
Seven p512/d128 pairs improve **22.577646 -> 22.749657 tok/s (+0.76186%,
7/7)**, while cached tracing cuts median device span
**44.516384 -> 44.042675 ms/token**. Tracked-clean selector-unset production
reaches **22.752894 tok/s / 43.950 ms/token** with flat pp512. See
`benchmarks/results/2026-07-31-gfx1151-laguna-decode-moe-branch-concurrency-retained.json`
and
`benchmarks/results/2026-07-31-gfx1151-laguna-decode-moe-branch-concurrency-production.json`.

The retained D8 MMQ128x32 gate/up consumer now has a gfx1151 row-vector
specialization. One thread owns each routed activation row, reads its compact
source mapping once per K32 interval, and stages the row through two aligned
16-byte loads instead of reconstructing eight int32 packs byte-by-byte across
the workgroup. Resident T16 weights, D8 bytes/metadata, dot arithmetic,
accumulation order, and BF16 output are unchanged. The uneven/empty-expert
fixture is BF16 byte-identical to the prior consumer and passes the independent
CPU-reference gate. Cached tracing records local128/VGPR80/SGPR128,
6,656 B LDS, zero scratch, and **264.416 -> 226.144 us** on the comparison
fixture. A one-load five-pair pp512 screen improves **368.450 -> 379.661 tok/s
(+3.043%)**, always token 2930. The clean committed selector-unset gate
confirms **368.203 -> 379.811 tok/s (+3.153%)** with complete sample
separation. Cached all-family tracing cuts selected gate/up
**581.061 -> 537.923 ms (-7.42%)** and measures
**381.448/351.663/292.417 tok/s** at 512/1K/4K. gfx1151 selects the
row-vector body; the old D8 consumer remains explicit rollback. Evidence:
`benchmarks/results/2026-07-25-gfx1151-laguna-gate-rowvec-{candidate,production}.json`.

The exact staging transfer now covers selected Q4/Q6 down. Both 64x32
consumers assign one thread to each compact D4 activation row and replace
distributed byte assembly with two aligned 16-byte loads; D4 metadata,
resident T16 decode, packed dots, accumulation order, and BF16 stores are
unchanged. Q4 dual/single and Q6 uneven/empty-expert fixtures are BF16
byte-identical to scalar staging, as is the production-shape synthetic MoE.
The one-load five-pair actual-model screen measures old **381.211**, Q4-only
**384.594 (+0.888%)**, Q6-only **382.981 (+0.464%)**, and combined
**386.612 tok/s (+1.417%)**, with complete combined/baseline sample separation
and token 2930 throughout. Cached pp512 tracing cuts Q4
**139.554 -> 126.972 ms (-9.02%)** and Q6
**132.467 -> 122.312 ms (-7.67%)**. Both are local128/LDS4096B/scratch0;
Q4/Q6 use VGPR56/72. gfx1151 selects the combined row-vector mode, retains the
scalar-staged MMQ as explicit rollback, and removes the temporary quant-scoped
runtime selectors. Clean committed production confirms scalar-down
**379.827 -> 385.997 tok/s (+1.625%)** with complete sample separation and
token 2930 throughout. Cached tracing measures
**388.014/358.319/296.060 tok/s** at 512/1K/4K, cuts selected down
**276.556 -> 254.006 ms (-8.15%)**, and cuts total kernel sum
**1,326.263 -> 1,304.061 ms (-1.67%)**. Evidence:
`benchmarks/results/2026-07-25-gfx1151-laguna-down-rowvec-{candidate,production}.json`.

The routing-qualified hybrid keeps MMQ128x32 for experts at or below 32 rows
and tests MMQ128x64 only above 32. Frozen natural pp512 routing predicts
**5,728 -> 3,102 (-45.8%)** large-expert tiles, but the actual layer-1
K3072/N1024 D8 gate/up leaf regresses **12.332 -> 13.179 ms (+6.87%)**.
Output is BF16 byte-identical on mixed empty/small/large fixtures. The wider
accumulator body and second filtered launch are removed; do not retry 64-row
expert accumulation without a different mapping
(`benchmarks/results/2026-07-25-gfx1151-laguna-hybrid64-expert-rejected.json`).
The follow-up different mapping is also closed: a local256 128x64 body used
eight wave32 row groups so each lane retained the production **32**
accumulators while the 128-column weight tile fed 64 rows. Its CPU-reference
fixture passed, but the actual layer-1 hybrid—128x32 for experts at or below
32 rows and local256 128x64 above 32—regressed **11.437 -> 11.819 ms
(+3.34%)**, including one D8 pack and both launches. All diagnostic surfaces
were removed. Local256 scheduling/launch cost, not only the prior accumulator
growth, prevents the 64-row crossover
(`benchmarks/results/2026-07-26-gfx1151-laguna-mmq128x64-t256-rejected.json`).
The wave-transpose/direct-consume mapping is retained as the gfx1151
production default. Four wave32 groups each own 32 output columns and all 32
routed rows, so each lane still holds **32 accumulators**, each even lane
decodes one T16 column pair, and a wave shuffle distributes the high-nibble
column.
Weights stay in registers instead of a 5,120-byte shared cache; D8 staging,
packed-dot arithmetic, K order, and resident bytes are unchanged. The
uneven/empty-expert fixture is BF16 byte-identical to row-vector production.
Actual layer-1 pack-inclusive time improves **11.467 -> 8.086 ms (1.418x)**,
and the implementation-worktree pp512 screen improves **385.941 -> 433.380
tok/s (+12.29%)**. Clean selector-unset publication then improves the
row-vector rollback **385.602 -> 432.355 tok/s (+12.125%)** with complete
seven-sample separation. Direct all-exact quality remains inside the gate at
maximum KL **0.049542582** and **316/320** top-1; decode, deterministic
repeats, Poolside, allocation recovery, and lifecycle pass. Cached all-family
tracing measures **434.994 tok/s**, cuts selected gate/up to **388.719 ms**,
and names template
`<1,false,true,128,true,true>` at local128/VGPR80/LDS1536B/scratch0 versus
row-vector LDS6656B. The row-vector mode remains explicit rollback through the
next retained checkpoint
(`benchmarks/results/2026-07-26-gfx1151-laguna-wavecols-production.json`).
The 64-column selected-down transfer is retained in production for **Q4
only**. Two wave32s each own 32 output columns and all 32 routed rows;
pair decode/shuffle keeps Q4 weights in registers and reduces Q4 down from
local128/VGPR56/LDS4096B to local64/VGPR80/LDS1536B, with zero scratch and
BF16-byte-identical output. A four-mode actual-model gate separates the quant
families: row-vector production is **433.791 tok/s**, Q4-wave/Q6-row is
**448.945 (+3.493%)**, Q4-row/Q6-wave is **428.184 (-1.293%)**, and both-wave
is **442.941 (+2.109%)**. All seven repetitions per mode return token 2930.
Q6 quartet-shuffle wave consumption is therefore rejected; Q6 remains on its
row-vector body. Clean committed production confirms all-row-vector
**433.081 -> 448.203 tok/s (+3.492%)** with complete seven-sample separation.
The direct all-exact gate remains max KL **0.049542582** and **316/320**
top-1 with neutral decode and exact lifecycle. Cached tracing measures
**449.522/409.990/332.286 tok/s** at 512/1K/4K and cuts the selected-down
family to **216.616 ms**. Q4 traces at local64/VGPR80/LDS1536B/scratch0; Q6
stays local128/VGPR72/LDS4096B/scratch0
(`benchmarks/results/2026-07-26-gfx1151-laguna-down-wavecols-production.json`).

Alternate D8 gate/up wave-column widths are closed. On actual layer-1 T16
weights plus natural M512 routing, production 128x32/local128 measures
**8.048 ms**, 64x32/local64 measures **8.087 ms (+0.486%)**, and a 256x32
two-columns-per-lane body measures **9.702 ms (+20.550%)**. Outputs are
BF16-byte identical. Cached tracing shows production/local64 both allocate
VGPR80, while the 64-accumulator wide body rises to VGPR128; all use 1,536 B
LDS and zero scratch. All candidate surfaces were removed
(`benchmarks/results/2026-07-26-gfx1151-laguna-gate-wavecols-geometry-rejected.json`).

Direct per-column Q4 decode is the retained gfx1151 gate/up default.
Within the production 128x32/local128 wave-column geometry, each lane decodes
its own resident-T16 column instead of even lanes decoding pairs and shuffling
the second column. Resident bytes, D8 activation staging, packed-dot
arithmetic, K order, and BF16 output are unchanged. The nine-case
CPU-reference gate passes and output is BF16-byte identical. Actual layer-1
pack-inclusive time improves **8.107 -> 6.916 ms (-14.69%)**; a seven-repeat
integrated pp512 screen improves **447.582 -> 472.533 tok/s (+5.575%)** with
complete sample separation and token 2930. Cached tracing names
`<1,false,true,128,true,true,128,true>` at local128/VGPR88/LDS1536B/scratch0;
its text is **13,416 bytes**, 752 bytes smaller than the pair-decode body in
the same object. Clean selector-unset production confirms pair-decode
**449.020 -> 474.363 tok/s (+5.644%)** with complete seven-sample separation.
The direct all-exact gate remains max KL **0.049542582** and **316/320**
top-1 with neutral decode and exact lifecycle. Cached tracing measures
**475.267/429.785/343.453 tok/s** at 512/1K/4K and cuts selected gate/up to
**317.722 ms**. The prior pair-decode mode remains explicit rollback through
cleanup
(`benchmarks/results/2026-07-26-gfx1151-laguna-q4-direct-wavecols-production.json`).

An exact activation-double-buffer sibling is the gfx1151 package default. It
keeps the direct 128x32/local128 weight decode and arithmetic,
but ping-pongs the 1,536-byte activation cache across two LDS slots so the
trailing synchronization after each K32 can be removed. LDS rises to 3,072
bytes while barriers fall from two to one per K32; resident bytes and BF16
output are unchanged. All 11 CPU-reference/GPU parameterizations pass.
Actual layer-1 natural-M512 pack-inclusive time improves **6.995 -> 6.907 ms
(-1.258%)**. Cached tracing names
`<1,false,true,128,true,true,128,true,true>` at plausible **7.253-7.456 ms**
for the profiled leaf. Clean seven-pair pp512 then improves direct rollback
**505.970 -> 507.405 tok/s (+0.284%)**, wins **5/7** pairs, and preserves
logits, final/post-layer hidden, KV, cursor, next-token logit, and next token
exactly in all fourteen runs
(`benchmarks/results/2026-07-26-gfx1151-laguna-gate-activation-doublebuf-default.json`).
Clean selector-unset publication is **505.084 tok/s** median and
**504.984 tok/s** minimum, statistically flat (**-0.020%**) versus the prior
unmatched production packet. Cached tracing independently reaches
**509.777 tok/s**, observes the intended template 564 times across the profiled
warmup/512/1K/4K sequence at local128/VGPR88/LDS3072B/scratch0, and cuts the
pp512 gate/up family **318.559 -> 314.378 ms (-1.313%)**. The matched A/B plus
named-family trace is the retention basis
(`benchmarks/results/2026-07-26-gfx1151-laguna-gate-activation-doublebuf-production.json`).

The exact raw-nibble P8 prefetch sibling is the gfx1151 gate/up package
default. It retains the direct
128x32/local128 D8 activation-double-buffer body, but carries the next K32
interval's eight raw T16 nibble words in registers while the current packed
dots execute. It decodes those words in place on the following interval and
demand-loads `d`/scale/min; resident bytes, LDS3072B, arithmetic order, and
BF16 output remain unchanged. M256 is **0.211%** slower and therefore keeps
the previous specialization, while actual layer-1 M512 improves
**6.8727 -> 6.7389 ms (-1.948%)**. Cached tracing names the intended P8
symbol at local128/VGPR96/SGPR128/LDS3072B/scratch0 versus production VGPR88.
Seven complete pp512 pairs improve **636.367 -> 640.003 tok/s (+0.571%,
7/7 wins)** with exact complete state. Clean selector-unset 512/1K/4K
publishes **643.554/573.066/466.290 tok/s**, improving every required length
with deterministic tokens/positions and full allocation recovery. Cached
all-family tracing cuts selected Q4 gate/up
**337.395 -> 333.701 ms (-1.095%)** and confirms the P8 resource tuple
(`benchmarks/results/2026-07-27-gfx1151-laguna-q4-raw-prefetch-p8-{candidate,production}.json`).
Extending P8 with two packed next-K32 metadata registers is closed: although
BF16-identical, it restores the rejected **VGPR104** class and regresses the
actual M512 leaf **6.7265 -> 7.0330 ms (+4.556%)**. The metadata symbol,
wrapper option, harness mode, and fixture parameter were removed
(`benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-metadata-prefetch-rejected.json`).
Non-temporal P8 loads are also closed. They preserve local128/VGPR96/
LDS3072B/scratch0 and BF16 output, but regress actual M512
**6.5634 -> 6.9727 ms (+6.236%)**; the ordinary cache policy is beneficial
for this mixed weight/activation working set. All candidate surfaces were
removed
(`benchmarks/results/2026-07-27-gfx1151-laguna-q4-p8-nontemporal-rejected.json`).

The same direct-decode premise is retained in Q4-down production.
Within the 64x32/local64 wave-column body, every lane decodes its own
resident-T16 Q4 column; D4 activation staging, packed-dot arithmetic, K order,
BF16 output, and Q6 row-vector production are unchanged. The ten-case
primitive gate and production-shape Q4/Q6 runtime oracle are BF16-byte exact.
With direct Q4 gate/up fixed, seven integrated repetitions improve Q4-down
pair decode **473.774 -> 483.409 tok/s (+2.033%)** with complete sample
separation and token 2930. Cached tracing names
`<1,true,false,64,true,true,64,true>` at local64/VGPR88/LDS1536B/scratch0.
Clean selector-unset publication improves pair-decode rollback
**473.963 -> 480.629 tok/s (+1.406%)** with complete seven-sample separation.
The direct all-exact gate remains max KL **0.049542582** and **316/320**
top-1 with neutral decode and exact lifecycle. Cached all-family tracing
measures **481.997/435.961/346.675 tok/s** at 512/1K/4K and cuts the Q4-down
consumer **90.280 -> 71.378 ms (-20.94%)**. The pair-decode mode remains
explicit rollback through cleanup
(`benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-direct-wavecols-production.json`).
The raw-nibble P8 pipeline is now also retained for Q4 selected down when
producer rows are at least 512. The 64x32/local64 body carries only the next
K32 interval's eight resident-T16 nibble words; it leaves D4 activation
staging, metadata loads, packed dots, K order, BF16 output, resident bytes,
LDS1536B, and scratch unchanged. The direct single-output CPU-reference gate
is BF16-identical. At the real M512 grid, three traced pp512 arms cut the 72
Q4-down launches **217.416 -> 212.090 ms (-2.450%)** at
local64/VGPR96/SGPR128/LDS1536B/scratch0 versus VGPR88 without prefetch.
Seven complete-state pp512 pairs improve
**639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)**. The prior body remains
selected below 512 rows. Clean selector-unset publication is
**643.141/573.717/466.913 tok/s** with exact tokens, positions, lifecycle,
and allocation recovery
(`benchmarks/results/2026-07-27-gfx1151-laguna-q4-down-raw-prefetch-p8-candidate.json`).
An exact two-slot activation-cache sibling was screened and removed. Q4 down
can safely omit the trailing K32 barrier because its direct-decoded weights
live in wave registers; Q6 stayed on the existing shared-weight body.
All 12 primitive configurations and the production-shape Q4/Q6 runtime
oracles pass, but matched seven-pair pp512 regresses
**508.788 -> 508.023 tok/s (-0.150%, +1.515 ms)** with only **2/7** candidate
wins. No export, wrapper option, selector, or harness mode remains
(`benchmarks/results/2026-07-26-gfx1151-laguna-q4-down-activation-doublebuf-rejected.json`).

Direct-decode 64x32/local64 gate/up completed the same exact gate and improved
the actual layer-1 natural-M512 leaf **6.920 -> 6.839 ms (-1.17%)**, but the
one-owner full-model screen was noise at **481.323 -> 481.619 tok/s
(+0.061%)** with overlapping ranges and a worse candidate minimum. Every
candidate surface was removed; 128x32/local128 direct decode remains
production
(`benchmarks/results/2026-07-26-gfx1151-laguna-gate-direct-local64-rejected.json`).

Q6 selected down now uses an exact 64-column x 64-row/local128 gfx1151
production body.
Four wave32 groups own 16 rows each while one 5,632-byte LDS tile retains the
single shared 64-column weight decode. The new `moe_mmq_tile_map/generic/tile64`
registry leaf rewrites the reusable expert starts/tile IDs after Q4 gate/up;
Q4 down remains on its 32-row direct wave-column body. Actual layer-1
runtime-bound timing improves **5.260 -> 5.161 ms (-1.879%)**, with zero BF16
mismatches, and dirty one-owner pp512 improves **490.105 -> 491.335 tok/s
(+0.251%)**. Cached tracing names template `<1,true,false,128,64>` at
local128/VGPR88/LDS5632B/scratch0 and grid-Y 332 versus the 32-row rollback's
408. Clean committed publication improves the explicit rollback **489.110 ->
492.640 tok/s (+0.722%)**, wins all seven paired repetitions, and independently
traces **493.509 tok/s**; absolute quality transfers unchanged at maximum KL
**0.049542582** and **316/320** top-1
(`benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-rows64-production.json`).

The production-shaped planar-qmicro Q6 selected-down leaf now uses an admitted
integer-WMMA gfx1151 production consumer. Four wave32
groups own independent 16-row bands and issue two signed-int8 x unsigned-Q6
16x16x16 fragments per K32 from the existing shared weight and compact D4
activation caches. The body preserves the two K16 scales, `-32*sum(x)`
correction, ordered FP32 K32 accumulation, BF16 store, resident bytes, and
row64 metadata. The uneven/empty-expert CPU-reference case and actual
layer-1 leaf are BF16-byte exact. Twenty-one counter-rotated natural-M512
pairs improve **4.7654 -> 4.5655 ms (-4.20%, 21/21 wins)**. Cached tracing
names template `<1,true,false,128,64,true,true,true,true,false,true,true>` at
local128/VGPR96/SGPR128/LDS5120B/scratch0 versus the retained planar body's
VGPR80. The wrapper selects it by default only when the already-constrained
planar-qmicro row64 contract is active; other shapes retain packed-dot
fallbacks. Clean selector-unset 512/1K/4K improves
**573.354/530.351/446.189 -> 576.137/543.213/459.054 tok/s** with
deterministic tokens, exact positions, and complete allocation return.
Refreshed cached tracing cuts the combined pp512 selected-down family
**189.049 -> 181.583 ms (-3.95%)** and the 115-call 512/1K/4K Q6 body
**1,124.852 -> 792.625 ms (-29.54%)**; pp512 Q6 itself is **109.290 ms**
(`benchmarks/results/2026-07-26-gfx1151-laguna-q6-selected-down-integer-wmma-production.json`).
An exact follow-up hoists each wave's two K16 activation vectors outside the
four-column-fragment loop. The 21-pair actual layer-1 leaf improves
**4.5645 -> 4.5126 ms (-1.136%, 20/21 wins)** with zero BF16 mismatches.
`rocprofv3` reports unchanged local128/VGPR96/SGPR128/LDS5120B/scratch0
resources. The selector-unset gfx1151 production route enables the hoist;
clean 512/1K/4K publication improves
**576.137/543.213/459.054 -> 577.396/545.366/459.716 tok/s
(+0.218%/+0.396%/+0.144%)**, with deterministic tokens, exact positions, and
complete allocation return
(`benchmarks/results/2026-07-26-gfx1151-laguna-q6-integer-wmma-hoist-activation-production.json`).
Cached all-family tracing at the published revision cuts the 115-call
512/1K/4K Q6 window **792.625 -> 779.709 ms (-1.63%)** and pp512 Q6
**109.290 -> 108.233 ms (-0.97%)**. The intended template remains
local128/VGPR96/SGPR128/LDS5120B/scratch0; the complete pp512 kernel span is
**884.129 ms** with only **0.857 ms** outside it.
An exact zero-sidecar follow-up overlaps the next planar-qmicro K32 global
fetch with current integer-WMMA compute and then fills the same shared tile
from registers. The actual layer-1 leaf improves
**4.518 -> 4.104 ms (-9.156%, 21/21 wins)** with zero BF16 mismatches, and
seven complete-state pp512 pairs improve
**618.294 -> 623.900 tok/s (+0.907%)** with identical full state. Cached
tracing reports local128/VGPR104/SGPR128/LDS5120B/scratch0, so the overlap
costs eight VGPRs but no LDS, scratch, or resident bytes. gfx1151 enables the
candidate behind an explicit rollback. Clean selector-unset pp512 improves
**632.618 -> 636.073 tok/s (+0.546%)**; 1K/4K remain flat within 0.12% at
**568.765/464.061 tok/s**. Cached two-queue tracing cuts the exact 23-call
pp512 Q6 body **112.746 -> 101.963 ms (-9.564%)** while retaining
2,417 dispatches
(`benchmarks/results/2026-07-27-gfx1151-laguna-q6-wmma-weight-prefetch-production.json`).
A successor pipelines each next compact Q8 activation half-row beside that
weight record while the current K32 WMMA executes. It preserves the same
resident bytes, activation/weight LDS, dot order, FP32 accumulation, and BF16
boundary. The actual leaf improves **4.104 -> 4.045 ms (-1.440%, 20/21
wins)**; clean selector-unset 512/1K/4K reaches
**639.114/569.880/464.280 tok/s**. Cached tracing cuts the 23-call Q6 body
again to **100.367 ms** at local128/VGPR112/SGPR128/LDS5120B/scratch0
(`benchmarks/results/2026-07-27-gfx1151-laguna-q6-wmma-activation-prefetch-production.json`).

Direct-decode 256x32/local256 gate/up is also rejected and removed. Eight
wave32s own one output column each and all 32 routed rows, so the candidate
halves workgroups and activation staging without the rejected two-columns-per-
lane body's 64 accumulators. It remains BF16-byte identical, but actual
layer-1 natural-M512 pack-inclusive time regresses **6.868 -> 7.181 ms
(+4.559%)**, and every one of nine counter-rotated samples loses. The screen
stopped before runtime integration; 128x32/local128 remains production
(`benchmarks/results/2026-07-26-gfx1151-laguna-gate-direct-local256-rejected.json`).

Raw-Q6 dense/shared WMMA now selects 16x32 as the gfx1151 production default.
The prior 64x16 body allocates VGPR256 plus 236 B/thread scratch; 16x32 traces
at local32/VGPR136/LDS0/scratch0 without changing resident bytes or arithmetic.
The four-axis gfx1151 registry owns the override; unmeasured gfx1100 remains
64x16.
All six supported tiles are BF16-byte identical on actual model weights, and
the aligned/boundary CPU-reference fixtures are byte-identical between 16x32
and 64x16. Actual-weight M512 timing cuts the 23-call K1024/N3072 shared-down
shape **0.942 -> 0.306 ms/call (-67.50%)** and the one K12288/N3072 layer-0
down call **10.629 -> 3.616 ms (-65.98%)**. Clean seven-repeat publication
improves explicit rollback **481.950 -> 490.096 tok/s (+1.690%)** with complete
separation and token 2930. All 24 actual projection weights have zero BF16
mismatches. Cached tracing measures local32/VGPR136/LDS0/scratch0, cuts the
Q6 family **29.248 -> 11.131 ms (-61.94%)**, and reaches
**491.171/441.091/351.095 tok/s** at 512/1K/4K.
`HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE=64x16` is explicit release-window rollback
(`benchmarks/results/2026-07-26-gfx1151-laguna-q6-dense-wmma16x32-production.json`).

Resident-pack8 Q4 dense/shared WMMA now has a retained gfx1151 production
shape policy. The real pp512 mix is 94 M512/K3072/N1024 shared gate/up calls,
24 M512/K1024/N3072 shared-down calls, and two M512/K3072/N12288 layer-0
gate/up calls. Nine counter-rotated burst-three samples keep the first at
64x16, select 64x32 for shared down, and select 32x32 for layer 0, reducing the
call-weighted leaf window **34.782 -> 33.031 ms (-5.03%)**. Direct execution
over all 120 actual resident projections reports zero BF16 mismatches versus
64x16. Clean matched pp512 improves **488.692 -> 489.922 tok/s (+0.252%)**
with four of seven paired wins and token 2930. Cached tracing independently
reaches **492.717 tok/s**, cuts Q4 dense **43.702 -> 41.936 ms (-4.04%)**,
and cuts total dense/shared **54.834 -> 52.989 ms (-3.36%)**. The absolute
wall median is flat within noise versus the prior publication, so this is
retained as a family-attributed exact micro-win. gfx1100 remains unchanged;
`HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE=64x16` is release-window rollback
(`benchmarks/results/2026-07-26-gfx1151-laguna-q4-pack8-shape-policy-production.json`).

The Q6 selected-down shared-weight local64 screen is rejected and removed.
Unlike the earlier duplicate-decode row halves, this candidate retained one
4 KiB LDS weight tile and assigned two waves 16 routed rows each. It is
BF16-byte exact, but actual layer-1 natural-M512 timing regresses
**5.223 -> 5.308 ms (+1.635%)**. Doubling accumulators per lane and serializing
more cache fills outweighs the smaller workgroup; local128 remains production
(`benchmarks/results/2026-07-26-gfx1151-laguna-q6-down-shared-weight-local64-rejected.json`).

### SH2-M4 compact selected T16 metadata

`quant/gguf_t16_selected_gemv.{hip,py}` and the selected-prefill T16 families
register exact `gguf_q4_k_qmicro_t16_v1` and
`gguf_q5_k_qmicro_t16_v1` consumers. The layouts preserve FP16 `d/dmin` and
quant planes while replacing each expanded four-column scale/min group with a
24-bit record. Legacy 2,368-byte Q4 and 2,880-byte Q5 T16 keys remain registered
fallbacks.

The full Q4+Q5 route is not a production kernel policy: its actual-weight leaf
projects **+1.598%** decode, and bounded preload/wave-broadcast unpack variants
regress further. Q4 stays on current T16. The separable 2,816-byte Q5 route is
production-default for the 37 selected down tensors and removes exactly
**155,189,248 bytes / 0.14453125 GiB**. Production tracing names
`q5_k_qmicro_t16_selected_gemv_kernel<unsigned short,8>` at local128,
VGPR56, LDS512, scratch0 and
`gguf_k_t16_selected_wmma_prefill_compact_kernel<unsigned short,5,true>` at
local32, VGPR72, LDS0, scratch0. Four-depth state is byte-exact and 512
prefill/decode remain within 1%. Evidence:
[`SH2-M4 retained Q5`](../benchmarks/results/2026-08-06-gfx1151-gguf-sh2-m4-compact-q5-t16-retained.json).

## DFlash / MTP lineage map

DFlash and MTP are tracked in `docs/source_lineage.json` before any native port
so benchmark-only scaffolding is not mistaken for a production kernel source.
Use these filters before touching the verifier/drafter path:

```bash
python3 scripts/check_lineage.py --file '*DFlash*' --diff stat
python3 scripts/check_lineage.py --file '*MTP*' --diff stat
python3 scripts/check_lineage.py --file '*pack8 small-row*' --diff patch
```

### Runtime/kernel lineage

| Source | Baseline | Role | Port note |
| --- | --- | --- | --- |
| `atlas/kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu` | `37513bf` | Scalar-column register-resident GDN prefill | Scheduling/launch-bounds reference only: independently implement raw-pointer HIP against hipEngine's exact direct-conv arithmetic and byte gate; do not copy CUDA/BF16/clamp semantics. |
| `atlas/kernels/gb10/common/gated_delta_rule_wy.cu` | `8d187c7` | Small-K two-pass WY identity | Algebra/traffic reference only; its CUDA BF16, gate clamp, intermediate-state, and reduction semantics are not hipEngine's contract. |
| `atlas/kernels/gb10/common/gated_delta_rule_wy64_prefill.cu` | `37513bf` | C=32 persistent WY prefill prototype | Chunk schedule and state-residency reference only; 84.5 KiB shared memory and SM121-specific choices do not transfer to RDNA3. |
| `vllm/vllm/model_executor/layers/fla/ops/{chunk.py,chunk_scaled_dot_kkt.py,wy_fast.py}` | `ed582b6` / `adb6d96` / `cb10b7e` | FLA triangular solve and WY representation | Equation/oracle reference only. hipEngine remains Torch/Triton-free and uses direct-conv FP32 inputs with a C=8 raw-pointer HIP design. |
| `nano-vllm-amd/csrc/amd/qwen35_expert.hip` | `b95eaa5` | R1 single-launch tree Conv/GDN t-loop kernels | Kernel source for DFlash tree/chain linear-attention verification; port as raw-pointer HIP with CPU or parent oracle fixtures. |
| `nano-vllm-amd/csrc/amd/extension.cpp` | `b95eaa5` | R1 extension bindings | Binding shape only; hipEngine wrappers stay torch-free and do not copy PyBind/Tensor signatures. |
| `nano-vllm-amd/csrc/amd/smoke.hip` | `b95eaa5` | R1 smoke fixtures | Fixture/oracle reference for t-loop kernels, not an E2E runtime dependency. |
| `nano-vllm-amd/nanovllm/native/qwen35/linear_attention.py` | `69eb9d8` | R2 Python wrappers for tree Conv/GDN t-loop kernels | Dispatch/API reference for row order, parent ids, and scratch semantics. |
| `hipengine/kernels/hip_gfx1100/linear_attn/{conv,gdn}.hip` | `b95eaa5`/`69eb9d8` | hipEngine DFlash tree Conv/GDN t-loop port | Raw-pointer C ABI wrappers for `bf16_tloop`/`fp16_tloop`; registered for `hip_gfx1100` and `hip_gfx1151`, with GDN `acc_buf` supplied by fixed verifier scratch. |
| `nano-vllm-amd/nanovllm/native/qwen35/paroquant.py` | `5d8f496` | DFlash pack8 small-row and dual-pack8 threshold policy | Includes `6f0e468` (`GEMV_V8_MAX_ROWS` 8→16 for DFlash bulk verify) and `5d8f496` (`NANOVLLM_PARO_DUAL_PACK8_MAX_ROWS`). Re-audit before changing hipEngine row thresholds. |
| `nano-vllm-amd/nanovllm/native/qwen35/mtp.py` | `e7651e8` | Target-attached MTP proposal provider | Future `MtpDraftProvider` source after DFlash verifier exists. |
| `nano-vllm-amd/nanovllm/native/qwen35/weights.py` | `7b20f47` | Native weight layout plus MTP BF16 loader | Loader metadata reference only; hipEngine runtime remains torch-free. |
| `nano-vllm-amd/nanovllm/native/qwen35/spec.py` | `5bfaa85` | Qwen3.5/Qwen3.6 config/spec parsing | Config alias reference for packed PARO/DFlash metadata validation. |

### Benchmark-only / prototype lineage

These sources are useful for metric names, prompt suites, and expected JSON, but
must not import PyTorch/HF code into hipEngine's production hot path.

| Source | Baseline | Role |
| --- | --- | --- |
| `nano-vllm-amd/scripts/bench_qwen35_dflash_acceptance.py` | `bd8360e` | Qwen3.6 W8A8 DFlash acceptance harness. |
| `nano-vllm-amd/scripts/eval_qwen35_dflash_acceptance_suite.py` | `874b5ae` | AR vs DFlash chain/DDTree prompt-suite comparison. |
| `nano-vllm-amd/scripts/inspect_qwen35_mtp.py` | `6ad5aea` | MTP artifact/tensor inspector. |
| `nano-vllm-amd/scripts/make_qwen35_mtp_real_prompts.py` | `4bb2573` | Stable real-prompt fixture builder. |
| `nano-vllm-amd/scripts/sweep_qwen35_mtp_real_acceptance.py` | `4bb2573` | MTP top-1/top-k acceptance sweep. |
| `amd-gpu-tuning/scripts/bench_dense27_dflash_smoke.py` | `3d509f4` | Dense27 DFlash smoke, serial/bulk verifier prototype, DDTree flat ABI reference. |
| `amd-gpu-tuning/scripts/sweep_dense27_dflash_prediction.py` | `c09e4df` | Dense27 chain/DDTree sweep harness and row aggregation. |
| `amd-gpu-tuning/PLAN-DFLASH.md` | `3d509f4` | Parent DFlash punchlist/evidence log. |
| `amd-gpu-tuning/PLAN-MTP.md` | `ab62086` | Parent MTP punchlist/evidence log. |
| `amd-gpu-tuning/MTP-DFLASH.md` | `63a9164` | Shared-verifier notes and early DFlash/MTP root-cause analysis. |
| `amd-gpu-tuning/docs/DFLASH-FRESH-EYES.md` | `8fd89b4` | Reference implementation audit. |
| `amd-gpu-tuning/docs/SPECULATIVE-DECODE.md` | `2cd030f` | HumanEval/code prompt and speculative-decode evidence notes. |
| `amd-gpu-tuning/docs/ROOFLINE-gfx1151.md` | `82f65a3` | Strix Halo roofline for verifier economics. |

### External model/artifact references

`check_lineage.py` audits git repos only. Model artifacts are recorded in
`docs/source_lineage.json` under `external_artifacts` and must be restated in
benchmark JSON:

- Laguna architecture/oracle source: Poolside `llama.cpp` branch `laguna`,
  detached local checkout
  `/home/lhl/models/hipengine_sources/poolside-llama.cpp-laguna` at
  `04b2b72cb54048ead292884adbe11f284e3ec950`. Tracked files are
  `src/models/{laguna,dflash}.cpp`, `conversion/laguna.py`, and the S 2.1 Jinja
  template. `python3 scripts/check_lineage.py --file '*laguna*' --diff stat`,
  the exact DFlash-path filter, and the template-kind filter are clean at this
  baseline; the broad legacy scan remains blocked when old parent checkouts are
  absent.
- Laguna target: `poolside/Laguna-S-2.1-GGUF`
  `laguna-s-2.1-Q4_K_M.gguf`, local 75,173,103,200-byte file SHA-256
  `7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f`.
- Laguna target oracle binary: native gfx1151 Poolside `llama-server` from the
  pinned commit above, SHA-256
  `1a3b09cfb9a8034d44239224ac362afce4555b85da376a3a7e1f4ecaffee0419`.
  The frozen command requires `--no-mmap --no-repack -fa off --cache-ram 0`
  and one exact-token request per fresh process; see
  `tests/fixtures/laguna_poolside_v1_oracle.json`. Poolside FA faults on this
  build, and same-process sequential completions are not accepted as oracle
  evidence.
- Laguna drafter: `poolside/Laguna-S-2.1-DFlash`, local snapshot
  `b0486d1586daa0d56435c508108171fc1c8daff9`, safetensors LFS SHA-256
  `f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4`.
- Drafter: `z-lab/Qwen3.6-35B-A3B-DFlash`, local snapshot
  `42d3b34d588423cdae7ba8f53a8cf7789346a719`; observed blobs include
  `dflash.py` blob `74d3ee2a48fbb1e65e25e19ab6cd89e2b28cd120`, `config.json`
  blob `64b63098ee9f2c9e1a2c0bf5ec1a4e32eb489703`, and the safetensors LFS SHA
  `6db5c712b4f3d924026162ad1aedf7fd1fef32437690451137f967d9b7160144`.
- Target: `shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed`; local snapshots
  observed `501ef8635e5cfb5a7497d232358ca8d1afc0c66e` and
  `176e57c1a5d823bd0f41605420d04e3441465bb4`. Every benchmark row must record
  the exact snapshot path/revision used.

## Source-lineage drift check

Before porting a family, check whether the parent source moved since the last hipEngine catalog/audit baseline:

```bash
python3 scripts/check_lineage.py --diff stat
```

Useful filters:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
python3 scripts/check_lineage.py --file '*paroquant*' --diff patch
python3 scripts/check_lineage.py --fail-on-drift
```

The script is read-only. It uses `docs/source_lineage.json` to compare tracked files in `~/amd-gpu-tuning/nano-vllm-amd/` against the recorded baseline ref, then reports:

- current child-repo branch/HEAD,
- per-file dirty status,
- commits since baseline,
- diffstat or patch for the file,
- matching lines in `~/amd-gpu-tuning/WORKLOG.md` and relevant parent docs.

If a file reports **DRIFT**, inspect the listed commits/diff and read the evidence hits before copying code. Update `docs/source_lineage.json`'s baseline only after the catalog/port source is intentionally refreshed and logged in `WORKLOG.md`.

## Source-lineage kernel catalog to port

The stable source-lineage port set at the current hipEngine catalog baseline is the committed `nano-vllm-amd` Qwen3.5/PARO kernel set: **95** kernels from `csrc/amd/qwen35_expert.hip` plus **25** PARO kernels from `nanovllm/native/qwen35/paroquant_kernels.py` = **120 Qwen/PARO kernels**, plus the separate `smoke_add` build smoke. hipEngine ports these by family; bodies are preserved byte-for-byte except for includes and raw-pointer host-wrapper retyping.

### Atomic / primitive-oriented kernel families (**source-lineage status; hipEngine-landed where noted**)

- `wmma/wmma_i8_gemm.hip` (4):
  - `qwen35_wmma_i8_tile_kernel`
  - `qwen35_wmma_i8_gemm_kernel`
  - `qwen35_wmma_i8_gemm_a_row_major_kernel`
  - `qwen35_wmma_i8_gemm_grouped_a_row_major_kernel`
- `quant/w8a8_activation.hip` (2):
  - `qwen35_quantize_activation_i8_per_row_kernel`
  - `qwen35_quantize_activation_f32_i8_per_row_kernel`
- `moe/w8a8_grouped.hip` (10):
  - `qwen35_dequantize_w8a8_projection_kernel`
  - `qwen35_dequantize_w8a8_grouped_projection_kernel`
  - `qwen35_dequantize_w8a8_grouped_accumulate_kernel`
  - `qwen35_dequantize_w8a8_grouped_accumulate_deterministic_kernel`
  - `qwen35_dequantize_w8a8_c1_grouped_accumulate_kernel`
  - `qwen35_moe_grouped_accumulate_kernel`
  - `qwen35_moe_grouped_gate_up_kernel`
  - `qwen35_moe_grouped_down_kernel`
  - `qwen35_moe_grouped_down_flat_kernel`
  - `qwen35_moe_grouped_down_flat_accumulate_kernel`
- `moe/swiglu.hip` (2):
  - `qwen35_swiglu_packed_gate_up_kernel`
  - `qwen35_dequantize_swiglu_quantize_grouped_kernel`
- `quant/w8a16_moe.hip` (17):
  - `w8a16_selected_experts_kernel`
  - `w8a16_gate_up_kernel`
  - `w8a16_down_kernel`
  - `w8a16_gate_up_shared_kernel`
  - `w8a16_gate_up_shared_t_kernel`
  - `w8a16_gate_up_shared_t_decode_v2_kernel`
  - `w8a16_down_shared_kernel`
  - `w8a16_down_shared_bulk_combine_kernel`
  - `w8a16_down_shared_t_kernel`
  - `w8a16_down_shared_t_decode_v2_kernel`
  - `w8a16_down_shared_bulk_combine_t_kernel`
  - `w8a16_single_gate_up_kernel`
  - `w8a16_single_down_combine_kernel`
  - `w8a16_shared_gate_up_bulk_kernel`
  - `w8a16_shared_gate_up_bulk4_kernel`
  - `w8a16_shared_down_bulk_combine_kernel`
  - `w8a16_shared_down_bulk_combine_w8a8_c1_selected_kernel`
- `moe/group_scatter.hip` (11):
  - `qwen35_moe_group_count_kernel`
  - `qwen35_moe_group_prefix_kernel`
  - `qwen35_moe_group_scatter_kernel`
  - `qwen35_moe_group_scatter_gather_kernel`
  - `qwen35_moe_c1_group_metadata_kernel`
  - `qwen35_moe_c1_group_metadata_gather_kernel`
  - `qwen35_moe_c1_group_metadata_quantize_kernel`
  - `qwen35_moe_gather_packed_hidden_kernel`
  - `qwen35_moe_gather_quantize_packed_hidden_kernel`
  - `qwen35_build_lane_to_sorted_kernel`
  - `qwen35_moe_combine_kernel`
- `moe/router.hip` top-k subset (2) — **hipEngine landed for BF16 hidden/weight raw-pointer wrappers**:
  - `qwen35_router_logits_kernel`
  - `qwen35_router_select_kernel`
- `moe/router.hip` token-rank/top2 subset (4):
  - `qwen35_token_rank_count_partial_kernel`
  - `qwen35_token_rank_count_finalize_kernel`
  - `qwen35_token_top2_partial_kernel`
  - `qwen35_token_top2_finalize_kernel`
- `quant/w8a16_linear.hip` (5):
  - `w8a16_linear_kernel`
  - `w8a16_linear_lowp_out_kernel`
  - `w8a16_linear_f32_kernel`
  - `w8a16_linear_batched_kernel`
  - `w8a16_linear_batched_f32_kernel`
- `linear_attn/conv.hip` (8):
  - `qwen35_linear_attn_conv_decode_kernel`
  - `qwen35_linear_attn_conv_decode_lowp_kernel`
  - `qwen35_linear_attn_conv_decode_indexed_lowp_kernel`
  - `qwen35_linear_attn_tree_conv_decode_lowp_tloop_kernel`
  - `qwen35_linear_attn_conv_prefill_no_state_rows_kernel`
  - `qwen35_linear_attn_conv_prefill_kernel`
  - `qwen35_linear_attn_conv_prefill_tile32x128_kernel`
  - `qwen35_linear_attn_conv_prefill_state_kernel`
  - `qwen35_linear_attn_conv_prefill_segments_kernel`
  - `qwen35_linear_attn_conv_prefill_segments_state_kernel`
- `linear_attn/gdn.hip` (20):
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_c1_exact_tloop_kernel`
  - `qwen35_gdn_prefill_recurrent_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_decode_order_kernel`
  - `qwen35_gdn_prefill_recurrent_decode_order_exact_lds32_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_segments_kernel`
  - `qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel` no-copy state-row instantiation for verifier capture
  - `qwen35_linear_attn_prefill_prepare_kernel`
  - `qwen35_linear_attn_prefill_prepare_raw_scales_kernel`
  - `qwen35_gdn_prefill_recurrent_decode_order_exact_kernel`
  - `qwen35_gdn_prefill_recurrent_decode_order_exact_segments_kernel`
  - `qwen35_linear_attn_prefill_prepare_decode_order_kernel`
  - `qwen35_gdn_prefill_rmsnorm_gate_bf16_kernel`
  - `qwen35_gdn_prefill_rmsnorm_gate_fp16_kernel`
  - `qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16_kernel`
  - `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp_tloop_kernel`
  - `qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_kernel`
  - `qwen35_gdn_tree_rmsnorm_gate_finalize_kernel`
  - `qwen35_gdn_prefill_recurrent_rmsnorm_gate_decode_order_kernel`
  - `qwen35_gdn_recurrent_rmsnorm_gate_segments_lowp_kernel`
- `norm/rmsnorm.hip` Qwen primitive subset (4) — **hipEngine landed for BF16 raw-pointer wrappers**:
  - `qwen35_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_f32_kernel`
  - `qwen35_head_rmsnorm_kernel`
- `norm/rmsnorm.hip` PARO subset (2) — **hipEngine landed for BF16 and FP16 raw-pointer wrappers**:
  - `paro_rmsnorm_out_kernel`
  - `paro_add_rmsnorm_out_kernel`
- `rotary/rotary.hip` Qwen primitive subset (1):
  - `qwen35_partial_rotary_kernel`
- `attention/full_attn_decode.hip` (2):
  - `qwen35_full_attn_decode_kernel`
  - `qwen35_full_attn_decode_context_tensor_kernel`
- `attention/paged_attn_decode.hip` (19):
  - `qwen35_paged_full_attn_decode_kernel`
  - `qwen35_paged_full_attn_decode_context_tensor_kernel`
  - `qwen35_paged_full_attn_decode_8k_context_tensor_kernel`
  - `qwen35_paged_full_attn_decode_4k_kernel`
  - `qwen35_paged_full_attn_decode_8k_dyn_kernel`
  - `qwen35_paged_full_attn_decode_split_k_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_groupwise_kernel`
  - `qwen35_paged_full_attn_prefill_gqa_gate_int8_groupwise_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_int8_key_bf16_value_kernel`
  - `qwen35_paged_full_attn_decode_split_k_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`
  - `qwen35_paged_full_attn_decode_split_k_prepare_weights_kernel`
  - `qwen35_paged_full_attn_decode_split_k_parallel_output_gate_kernel`
- `attention/paged_kv_write.hip` (10):
  - `qwen35_write_paged_kv_kernel`
  - `qwen35_write_paged_kv_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_kernel`
  - `qwen35_write_paged_kv_mixed_value_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_batch_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel`
  - `qwen35_write_paged_kv_int8_per_token_head_kernel`
  - `qwen35_write_paged_kv_int8_block_kernel`
  - `qwen35_write_paged_kv_int8_key_bf16_value_kernel`
  - `qwen35_write_paged_kv_int8_hadamard_group32_kernel`
- `quant/paro_awq_gemv.hip` stable PARO GEMV/projection subset (7; projection-pair fused variants are called out again below):
  - `gemv_awq_v8_kernel`
  - `gemv_awq_pack8_kernel`
  - `gemv_awq_dual_pack8_kernel`
  - `gemv_awq_selected_dual_pack8_strided_kernel`
  - `gemv_awq_selected_pack8_kernel`
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`
  - `dense_gemv_out_kernel`
- `quant/paro_awq_dequant.hip` (2):
  - `dequant_awq_pack8_kernel`
  - `dequant_awq_pack8_dual_kernel`
- `rotary/rotary.hip` PARO subset (2):
  - `paro_rotate2_kernel`
  - `paro_rotate3_kernel`

### Fused / composite kernel families (**lineage green, not yet hipEngine-landed**)

Each fused kernel still requires an unfused fallback chain registered under its primitive components.

- Norm + rotary:
  - `qwen35_head_rmsnorm_partial_rotary_kernel`: `head_rmsnorm -> partial_rotary`.
  - `qwen35_head_rmsnorm_partial_rotary_position_kernel`: `head_rmsnorm -> position-indexed partial_rotary`.
- PARO selected-expert activation / rotation:
  - `silu_mul_dual_out_kernel`: `silu(gate) * up` for dual selected-expert outputs over packed `[rows, 2*features]` input.
  - `silu_mul_separate_out_kernel`: `silu(gate) * up` where gate and up live in separate `[rows, features]` buffers; used by the W4 PARO dense shared expert path.
  - `silu_mul_dual_rotate_out_kernel`: `silu(gate) * up -> PARO down-rotation`.
  - `silu_mul_pair_rotate_out_kernel`: paired `silu(gate) * up -> rotate` variant.
- Weighted routing reductions:
  - `weighted_index_add_out_kernel`: routed weighted add into output rows.
  - `weighted_index_add_atomic_float_out_kernel`: atomic-float routed weighted add variant.
  - `weighted_lanes_inverse_kernel`: lane/weight inverse helper.
  - `weighted_lanes_sum_out_kernel`: lane-group weighted sum.
  - `weighted_sum_out_kernel`: selected-expert weighted sum.
- Shared-expert + selected-expert combine:
  - `shared_gate_combine_out_kernel`: `selected_moe + sigmoid(shared_gate) * shared_expert`.
  - `shared_gate_combine_residual_out_kernel`: above plus residual add.
  - `weighted_sum_shared_gate_combine_residual_out_kernel`: selected weighted sum + shared gate combine + residual add in one c=1 decode kernel.
- Full-attention gate fusion:
  - `full_attn_gate_mul_out_kernel`: `sigmoid(attn_gate) * attention_out` plus output conversion.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`: paged split-K reduce fused with PARO full-attention gate for device-context decode.
- Projection-pair fusion routes used by the PARO path:
  - `gemv_awq_dual_pack8_kernel`: dual W4 pack8 GEMV for two projections over the same input.
  - `gemv_awq_dual_pack8_transposed_rotate_staged_kernel`: opt-in/default-off decode diagnostic that stages two input rotations once, then runs dual transposed W4 pack8 GEMV after a device barrier.
  - `gemv_awq_selected_dual_pack8_strided_kernel`: selected-expert dual W4 pack8 GEMV over compact/repacked expert weights.
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`: selected-expert dual W4 pack8 GEMV plus output rotation.

### Source catalog drift requiring refresh before PARO/WMMA ports

The last manual hipEngine catalog audit (`docs/source_lineage.json` baseline `22405a9`) counted the committed PARO embedded-HIP set at 25 kernels and observed six additional parent-worktree kernels beyond that committed set:

- `gemv_awq_mbatch_dual_pack8_kernel`
- `gemv_awq_mbatch_pack8_kernel`
- `gemv_awq_expert_seq_dual_pack8_kernel`
- `gemv_awq_expert_seq_pack8_kernel`
- `gemm_awq_selected_dual_pack8_wmma_kernel`
- `gemm_awq_selected_pack8_wmma_kernel`

`~/amd-gpu-tuning/docs/OPTIMAL.md` now promotes a compact-WMMA route, and `scripts/check_lineage.py` reports drift in `qwen35_expert.hip`, `extension.cpp`, `paroquant_kernels.py`, `paroquant.py`, and `expert.py` after `22405a9`. Therefore, treat the 120-kernel catalog above as the **baseline catalog**, not the final PARO/WMMA port inventory.

Current OPTIMAL source refresh at `nano-vllm-amd@59195ed` adds **5 kernels** over the baseline catalog: `qwen35_moe_wmma_tile_map_kernel` in `qwen35_expert.hip`, plus `gemm_awq_selected_dual_pack8_wmma_kernel`, `gemm_awq_selected_pack8_wmma_kernel`, `gemm_awq_selected_dual_pack8_wmma_compact_kernel`, and `gemm_awq_selected_pack8_wmma_compact_kernel` in `paroquant_kernels.py`. That refresh's full Qwen/PARO HIP inventory is **96** monolithic kernels + **29** PARO/WMMA kernels = **125** kernels, excluding `smoke_add`. Additional parent drift observed at `nano-vllm-amd@b95eaa5` adds five tree/speculative linear-attention kernels (`qwen35_linear_attn_tree_conv_decode_lowp*`, `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp*`, and `qwen35_gdn_tree_rmsnorm_gate_finalize_kernel`), bringing the observed parent inventory to **101** monolithic kernels + **29** PARO/WMMA kernels = **130** kernels. The dense-projection fused W4 prefill source is tracked separately in `paroquant_fusedw4.py`; hipEngine has ported its FP16 raw-pointer WMMA kernel for transposed pack8 prompt projections and added a strided-layout instantiation for existing non-transposed prompt weights. Those tree kernels are not the compact prompt-slab `cu_seqlens` ABI and were not ported for the segment-prefill task. Before porting PARO/WMMA or tree kernels, read the listed WORKLOG/OPTIMAL evidence and keep this checklist synchronized with the source commit used.

## Qwen3.5 MoE / PARO path map

This section maps the current source-lineage inference path that hipEngine should preserve when porting `z-lab/Qwen3.5-35B-A3B-PARO` (`w4_paro`, W4A16) from `nano-vllm-amd`. It is **not** an hipEngine performance claim yet; it is the target graph/kernel route to reproduce after the port.

### Current optimal route

Canonical source: `~/amd-gpu-tuning/docs/OPTIMAL.md` (2026-05-13 snapshot). Supporting design/history remains in `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md`.

The current optimal parent route is compact-WMMA prefill plus one-step graph-replay decode, with all listed parent quality gates passing. Latest retained parent sweep:

| Shape | PARO prefill tok/s | PARO decode tok/s | Peak VRAM | Validation |
| --- | ---: | ---: | ---: | --- |
| 512/128 | 2557 | 115.7 | 18.86 GiB | graph/step true |
| 1K/128 | 2876 | 112.9 | 19.34 GiB | graph/step true |
| 4K/128 | 2703 | 112.0 | 21.64 GiB | graph/step true |
| 32K/128 | 1880 | 98.8 | 21.37 GiB | graph/step true |
| 128K/128 | 914 | 62.6 | 27.42 GiB | graph/step true |

Post-sweep parent spot checks retained these defaults:

- Native weighted-lane grouped-stacked accumulation: `2642.1` vs `2561.5` prefill tok/s at 512/128, graph validation true.
- Grouped SiLU + down-rotation fusion: `2632.2` vs `2631.4` prefill tok/s at 512/128, graph validation true.
- WMMA extension load is not the Vulkan decode-gap source: graph decode was `115.56` vs `115.04` tok/s with WMMA disabled.

Correctness hierarchy for these rows: HF PARO oracle for model correctness; scalar eager pure-native as the native debug reference; tensorized eager as serving/graph ABI reference; graph replay must match tensorized eager. Long scalar-vs-tensorized greedy equality is a diagnostic, not the only promotion gate; use KL/NLL/top-k/top-1 and repetition/coherence/long-context quality gates.

### Base flags to preserve

`OPTIMAL.md` lists 23 base environment flags. hipEngine should preserve the same routing decisions as registry/plugin configuration rather than copying env-var checks into engine code:

- **MoE dispatch:** compact stacked layout, in-place selected-MoE repack replacement, GPU expert gather, grouped-stacked max tokens `4096`, native weighted lanes, grouped-stacked SiLU+rotate fusion, decode selected-MoE SiLU/down-rotate fusion, native router.
- **GEMV / WMMA:** PARO vec8 GEMV, pack8 qweight replacement, transposed pack8 disabled on W7900, WMMA GEMM enabled for prefill MoE, compact WMMA buffers, parent `WMMA_MIN_TOKENS=64` (crossover vs GEMV ~48 tokens); hipEngine P1.4 retains compact WMMA for all multi-token single-request prefill (`HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS=2`) after GEMV fallback lost at 128/256/512/4096 prompts.
- **Attention:** full-attention gate fusion, full-attention Q/K pack8 fusion, grouped-GQA paged context attention, paged max splits `512`.
- **Linear/projections:** W8A16 `lm_head`, W8A16 shared expert dense branch, fused linear-attention A/B projection, pack8 fused linear-attention QKV+Z projection.
- **Routing threshold:** native router prefill path begins at `512` tokens.

### Current OPTIMAL MoE port checklist (`nano-vllm-amd@59195ed`)

The checklist below is the active port map for reproducing the parent compact-WMMA + graph-replay route. Status values are hipEngine status, not parent status.

#### Source refresh deltas since baseline `22405a9`

| Source | Current status | Required action |
| --- | --- | --- |
| `csrc/amd/qwen35_expert.hip` | DRIFT; 96 kernels | Include new `qwen35_moe_wmma_tile_map_kernel` with grouped MoE / compact WMMA port. |
| `csrc/amd/extension.cpp` | DRIFT; + bindings for tile-map path | Retype affected launch wrapper(s), do not copy PyTorch/TORCH_LIBRARY plumbing. |
| `nanovllm/native/qwen35/paroquant_kernels.py` | DRIFT; 29 kernels, 35 `m.def` exports | Extract current V8 + WMMA embedded HIP, including four WMMA kernels and compact wrappers. |
| `nanovllm/native/qwen35/paroquant.py` | DRIFT; dispatch logic changed | Adapt routing decisions into model/quant/kernel-plan plugins, not env-var branches in engine code. |
| `nanovllm/native/qwen35/expert.py` | DRIFT; added `hip_qwen35_moe_wmma_tile_map` | Port tile-map raw-pointer wrapper with grouped MoE metadata family. |

#### MoE decode c=1 path

| Stage | Parent kernels / wrappers | hipEngine status | Notes / gate |
| --- | --- | --- | --- |
| RMSNorm / residual | `paro_rmsnorm_out_kernel`, `paro_add_rmsnorm_out_kernel`; Qwen BF16 `qwen35_*rmsnorm*` family | **Landed for BF16 and FP16 PARO raw-pointer wrappers** | PARO out-kernels multiply direct norm weights and now cover the parent FP16 activation path; Qwen kernels use `1.0 + weight_delta`. |
| Router + shared gate | `qwen35_router_logits_kernel`, `qwen35_router_select_kernel`, `hip_qwen35_router_topk_shared_out` | **Landed for BF16 and FP16 hidden raw-pointer shared-out routes; cooperative decode fold is opt-in/rejected** | Current wrappers write logits/selected/routing buffers and shared-gate logits with BF16 router weights; FP16 hidden specialization covers parent-mixed activation materialization. `HIPENGINE_PARO_ROUTER_TOPK_COOP=1` runs a diagnostic atomic last-producer fold that preserves the one-block-per-row logits grid but is not default after D1.5 regressed graph replay. |
| Selected gate/up GEMV | `gemv_awq_selected_dual_pack8_strided_kernel`, `gemv_awq_selected_dual_pack8_kernel`, optional rotate-out variant | **Landed for BF16 and FP16 raw-pointer strided/transposed dual pack8 wrappers plus fused rotate-out** | Decode path uses stacked/repacked selected-expert W4 pack8 qweights. Preserve small-K safety fix from `59195ed`. |
| Activation + down rotation | `silu_mul_dual_rotate_out_kernel` (fallback `silu_mul_dual_out_kernel` + rotate) | **Landed for BF16 and FP16 raw-pointer fused and fallback wrappers** | Default `NANOVLLM_PARO_MOE_SILU_DOWN_ROTATE_FUSED=1`; fused dual rotate plus dual/pair fallback kernels are registered for parent-mixed activations. |
| Selected down GEMV | `gemv_awq_selected_pack8_kernel` / strided wrapper | **Landed for BF16 and FP16 raw-pointer strided/transposed pack8 wrappers** | Used for selected down projection; small-K specialization applies where safe. |
| Shared expert | W8A16 shared gate/up/down (`w8a16_*shared*`, `w8a16_single_*`, `w8a16_linear*`) | **Landed for current parent lowp-linear route, including FP16 lowp wrapper, multi-token FP16 shared gate/up+SiLU, and grouped-prefill shared down+combine helper** | `w8a16-shared-expert-hip` validates W8A16 gate/up → `silu_mul_dual_out` → W8A16 down (`gate_up_mismatch=0`, `intermediate_mismatch=0`, `out_mismatch=0`); fused FP16 shared gate/up+SiLU and shared down+combine are covered by all-layer fixture gate (`max_kl=0.03406`, top-1 `1.0`) and the c=1/non-grouped fallbacks remain registered. |
| Weighted combine + residual | `weighted_sum_shared_gate_combine_residual_out_kernel`; fallback `weighted_sum_out_kernel`, `shared_gate_combine*` | **Landed for BF16 and FP16 values with FP32 weights/gate logits** | c=1 decode promoted path fuses selected sum, shared sigmoid/gate combine, and residual add; scalar-weight fallback remains unported. |
| Synthetic c=1 vertical smoke | RMSNorm → router → selected W4 gate/up/down → W8A16 shared → weighted/shared/residual combine | **Landed** | `paro-moe-c1-hip --hidden-size 8`: direct wrapper chain bit-exact; `paro-moe-c1-state-hip --hidden-size 8`: decode-state path bit-exact (`final_mismatch=0`) and uses normalized prepared weights + `RuntimeWorkspace`; full model path still needs tokenizer/model loop/attention plumbing. |

#### MoE prefill compact-WMMA path

| Stage | Parent kernels / wrappers | hipEngine status | Notes / gate |
| --- | --- | --- | --- |
| Lane grouping | `qwen35_moe_group_count_kernel`, `qwen35_moe_group_prefix_kernel`, `qwen35_moe_group_scatter[_gather]_kernel`, `qwen35_moe_gather_packed_hidden_kernel` | **Landed for metadata + lowp packed-hidden gather** | `qwen35-moe-group-scatter-hip` validates count/prefix/scatter_gather/gather; expert GEMM/WMMA still required before retained MoE prefill. |
| Compact WMMA tile map | `qwen35_moe_wmma_tile_map_kernel` | **Landed** | Maps compact expert starts to WMMA tiles without pad-multiple=16 overhead. |
| Gate/up compact WMMA | `gemm_awq_selected_dual_pack8_wmma_compact_kernel` | **Landed for BF16 and FP16** | Current grouped prefill route calls compact WMMA over packed/sorted lanes; noncompact WMMA and GEMV-only remain fallback/comparison paths. |
| Activation + down rotation | `silu_mul_dual_rotate_out_kernel` | **Landed / reused for grouped packed lanes** | `NANOVLLM_PARO_MOE_GROUPED_STACKED_SILU_ROTATE_FUSED=1` default; current grouped prefill calls the existing fused rotate over sorted lanes. |
| Down compact WMMA | `gemm_awq_selected_pack8_wmma_compact_kernel` | **Landed for BF16 and FP16** | Paired with compact tile map and compact buffers; `paro-awq-wmma-compact-hip` validates compact dual/single kernels on a tiny fixture. |
| Weighted lane accumulation | `weighted_lanes_sum_out_kernel` | **Landed for BF16 and FP16** | `paro-combine-hip` validates `weighted_lanes_sum_out_{bf16,fp16}` and batched shared-gate residual combine; `rocprofv3` shows the weighted-lane kernels and batch combine on W7900. |
| GEMV fallback/comparison | `gemv_awq_selected_dual_pack8*`, `gemv_awq_selected_pack8*` | **Available as fallback/comparison** | Single-request multi-token prefill layer orchestration defaults to grouped metadata + compact WMMA for `tokens >= 2` (`HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS`, large value forces c1 GEMV diagnostics); P1.4 found no useful GEMV crossover at 128/256/512/4096 prompts. |

#### Full-inference dependencies outside MoE

| Area | Required for reproducing parent inference | hipEngine status |
| --- | --- | --- |
| PARO quant plugin / weight layout | `w4_paro` plugin, pack8 replacement layout, compact stacked MoE weights, W8A16 shared/lm-head replacements | Missing; only `bf16` plugin landed. |
| Model plugin / scheduler | Qwen3.5 hybrid full-attn + linear-attn/GDN + MoE layer sequence, static decode buffers, one-step graph replay | Missing; `LLM.generate()` is still scaffolded. |
| Linear projections | `gemv_awq_pack8`, `gemv_awq_dual_pack8`, `awq_fusedw4_prefill_fp16`, `awq_fusedw4_prefill_strided_fp16`, `dense_gemv_out`, rotation helpers | **Partially landed**: c=1 GEMV wrappers plus FP16 fused-W4 WMMA prompt projection for transposed QKV/Z and strided linear out-proj; dense/W8 projection WMMA remains open. |
| Linear attention / GDN | `qwen35_linear_attn_conv_*`, `qwen35_gdn_*` incl. lowp recurrent RMSNorm gate | Landed for current Qwen3.5/PARO path: decode conv/GDN, single-request prefill conv/GDN, and segment-aware compact-slab conv/GDN state kernels are available; remaining c>N work is packed orchestration with varlen full-attn/final commit. |
| Full attention / KV | `awq_fusedw4_prefill_fp16`, `awq_fusedw4_prefill_strided_fp16`, `qwen35_head_rmsnorm_partial_rotary*`, `qwen35_write_paged_kv_mixed_value*`, paged/split-K full-attention decode family, `full_attn_gate_mul_out` | Partial: FP16 fused-W4 prompt projection for transposed Q/K and strided V/O, full-attention prelude, span-shaped paged KV append, span-shaped paged context-tensor decode, generic split-K reduce, FP32/BF16 gated split-K reduce, and Qwen3.5 GQA-specialized split-K context variants landed; remaining gaps are non-context/int8/8K legacy variants and engine allocation/plumbing. |
| Final head | W8A16 `lm_head` replacement path | Missing. |
| Eval harness | Parent baseline JSON capture + hipEngine JSON schema-2 artifacts + KL/top-1/sample/graph validation gates | Not yet landed. |

#### Port order for the OPTIMAL exercise

1. **Measurement harness first:** run/record the parent `512/128` and `4K/128` OPTIMAL commands as source-lineage artifacts, then create a blocked hipEngine artifact until `LLM.generate()` exists.
2. **MoE c=1 decode vertical slice:** PARO RMSNorm out-kernels → router/shared-gate → selected pack8 GEMV → fused activation/down-rotation → W8A16 shared expert → weighted shared-gate residual combine.
3. **MoE prefill compact-WMMA slice:** lane grouping/gather → compact tile map → compact dual/single WMMA → weighted-lane accumulation → GEMV fallback.
4. **Full-inference closure:** weight loader/model plugin, non-MoE projections, linear attention/GDN, full attention/KV, final head, graph replay, then end-to-end correctness/perf comparison.

### Prefill route

- Benchmark protocol: `OPTIMAL.md` uses the parent `scripts/run_moe2_baselines.py` sweep and graph-replay bench command; short/mid-context quick start targets `--prompt-len 4096 --decode-len 128 --decode-use-step-graph-replay`.
- Router/MoE:
  - Real router runs per MoE layer; no HF model execution in the pure-native path.
  - Compact WMMA prefill MoE is the current optimal grouped-stacked route for `>=64` tokens; GEMV-only only wins at ~32 tokens.
  - Grouped-stacked max tokens is now `4096`, not the older 1024 short-prefill cap.
  - Weighted-lane accumulation and grouped-stacked SiLU+down-rotation fusion are default-on parent optimizations.
- Long prefill:
  - For `>=32K`, add chunking overrides from `OPTIMAL.md`: linear chunk `1024`, MoE chunk `1024`, full-attention post/RoPE chunk `1024`, and full-attention query chunk `4096`.
  - Do **not** set long-prefill chunking overrides for `<=4K`; they change the MoE prefill path and reduce throughput.
- Projection/quant:
  - Non-expert W4 pack8 replacement uses `[out/8, in]` pack8 qweights and frees original eligible AWQ qweights.
  - `lm_head` uses the W8A16 replacement path in the optimal route.

### Decode route

The target c=1 decode path is static-buffer, graph-replay-friendly, and mostly device-resident:

1. **RMSNorm / residual:** PARO-native `rmsnorm` and `add+rmsnorm` kernels; avoid per-token framework glue.
2. **Router:** native combined router/shared-gate logits with hot BF16 cache and FP16/BF16 hidden input; reuse decode-only output buffers.
3. **Selected MoE:** compact stacked selected-expert layout plus repacked replacement qweights; selected gate/up via dual W4 pack8 GEMV; selected down uses small-K specialization where applicable.
4. **Selected activation/down rotation:** `silu(gate) * up` and PARO down-rotation fused on the stacked decode path.
5. **Shared expert:** dense shared expert c=1 branch uses W8A16 gate/up/down where enabled; grouped multi-token FP16 prefill fuses shared gate/up + SiLU into the four-column bulk helper (or token-tiled helper for legacy prompts `>=1024`) and fuses shared down with selected/shared-gate/residual combine (token-tiled for legacy prompts `>=2`).
6. **MoE combine:** selected-expert weighted sum, shared-expert sigmoid/gate combine, and residual add fuse into `weighted_sum_shared_gate_combine_residual_out_kernel` on c=1 decode.
7. **Linear attention:** native conv/GDN recurrence; lowp FP16/BF16 inputs feed kernels while recurrent state/math stay FP32. A/B projections are concatenated for c=1; QKV/Z and out-proj use fused W4→WMMA for multi-token FP16 prompt projection after rotation, with pack8 W4 GEMV retained for c=1/fallback.
8. **Full attention projections:** q/k use fused W4→WMMA for multi-token FP16 prompt projection after batched input rotation; v/o also use the strided fused-W4 prefill instantiation, and c=1 uses the dual/single pack8 W4 GEMV path.
9. **KV append:** BF16 full-attention KV cache with native mixed-input paged-KV writer; no tiny per-token framework appends.
10. **Full attention decode:** contiguous path for short contexts; paged/split-K path defaults at context `>= 1024`, with warp-cooperative context tensor QK, physical-offset address hoist, grouped-GQA reuse, split cap 512 for 128K-class rows, and gated split-K reduce where applicable.
11. **Final head:** W8A16 `lm_head` replacement path.
12. **Graph replay:** one reusable decode-step graph replay is the promoted graph shape; keep `--decode-step-graph-capture-steps=1`. Multi-step capture was tested and not promoted.

Parent decode profiling note from `OPTIMAL.md`: fused `lm_head + argmax` is not a current lever; the next decode target is the AWQ/GEMV decode family, about 40% of selected-region kernel time in the 512/128 graph profile.

### Alternative paths and caveats

- **W8A8 comparison path:** stays quality-safe and useful as a comparator; do not regress it while porting PARO.
- **40GB+ diagnostic PARO path:** stacked selected-expert diagnostics proved speed hypotheses but are not promotion candidates because 24GB W4 usability is a hard gate.
- **24GB non-stacked baseline:** green but slow; retained as a deployable-memory fallback, not the speed target.
- **Long-context decode:** contiguous full-attention decode cannot launch at 32K because dynamic LDS scales with context; long decode must use paged/split-K over the dense cache viewed as pages.
- **Tensorized paged-attention drift:** current parent docs localize long-tail scalar-vs-tensorized drift to paged context-tensor full attention. Graph replay matching tensorized eager is necessary; scalar-eager greedy equality alone is not sufficient promotion evidence.
- **Rejected standalone kernel ideas:** PARO v8 unroll-threshold 600, isolated wave32/no-LDS W4 GEMV, naive AWQ W4xQ8 dp4a, caller-owned paged workspace, and non-split-K 4K attention were tested but not promoted. Do not import them into hipEngine defaults without a fresh audit and correctness/perf evidence.

## Port = copy + partition + retype

The initial port is mechanical, not creative. Kernel bodies are preserved byte-for-byte (modulo `#include` headers). The three things that change during port:

1. **File split by family.** The monolithic `nano-vllm-amd/csrc/amd/qwen35_expert.hip` (13,769 lines, 95 `__global__`s) and the 3,766-line embedded HIP string in `nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` partition into `kernels/<backend>/<family>/*.hip` per the table in `docs/PLAN.md`. The near-duplicate `qwen35_expert_hip.hip` is dropped.
2. **Launch wrappers retyped.** Host-side wrappers go from `torch::Tensor` to raw pointer + shape/stride/dtype signatures. Scripted, ~1 day.
3. **Embedded HIP extracted.** `paroquant_kernels.py`'s `r'''...'''` block becomes real `.hip` files compiled through `hipengine.core.build` instead of `torch.utils.cpp_extension.load_inline`.

Preserve all `__launch_bounds__`, template specializations, and compiler flags (`-mllvm -amdgpu-unroll-threshold-local=600` for decode/prefill, plus `-mcumode` for decode). A port that rewrites kernel bodies is not a port.

## Port correctness gate (non-negotiable)

A kernel split / port may only land when all three of these pass on the stated fixture set:

1. **Registry resolution.** Every kernel name still resolves via the 4-axis registry (`resolve(KernelKey(...))` returns a callable for every key previously exported by the monolithic `.so`).
2. **Profiler parity.** `rocprofv3 --kernel-trace` on the target decode smoke (Qwen3.6-35B-A3B unless noted) reports the same kernel set with matching `DurationNs` distribution as the monolithic build. A new kernel name, a missing kernel name, or a >10% duration shift is a split bug.
3. **Numerical parity.** KL ≤ 0.05 AND top-1 agreement ≥ 90% vs the monolithic build on the correctness fixtures. (For a *net-new* kernel, the oracle is `kernels/cpu_reference/`, not the monolithic build.)

Never land a split that regresses any of these.

## Build layer (`hipengine.core.build`)

hipEngine uses its own build layer, not `torch.utils.cpp_extension`. It calls `hipcc` (or `nvcc` for CUDA backends) via `subprocess.run`, links with `ctypes.CDLL`, and caches `.so` files by a hash of `(source, flags, hipcc version)` under `~/.cache/hipengine/build/`. Edit → bench loop stays at ~5–10 s per kernel change.

### Three build profiles (from `nano-vllm-amd/nanovllm/native/amd/extension.py`)

| Profile | Flags | Wavefront | Used for |
| --- | --- | --- | --- |
| `decode` | `-mllvm`, `-amdgpu-unroll-threshold-local=600`, `-mcumode` | 32 | Decode-phase kernels (paged attention, W8A8 grouped MoE decode, PARO GEMV). `-mcumode` is not `-mwavefrontsize64`. |
| `prefill` | `-mllvm`, `-amdgpu-unroll-threshold-local=600` (WGP mode) | 32 | Prefill-phase kernels (GEMM, W8A16 linear prefill) |
| `baseline` | (none) | 32 | Debug / fallback |

Write device code for wave32 by default on gfx1100. Use `warpSize` for probes and dispatch metadata, but do not assume a 64-thread block is one wave. For block-wide reductions over more than 32 lanes, reduce within 32-lane waves with shuffles and exchange partials through LDS/shared memory.

### Wave32 default; wave64 experiments only

For nano-vllm-amd lineage kernels on W7900/gfx1100:

- Default to **wave32**. Current HIP build flags do not include `-mwavefrontsize64`, and
  parent probes showed `-mcumode` does not change `warpSize` by itself.
- RDNA3 wave64 is architecturally supported, but the hardware still issues through
  32-lane halves. RDNA3 can co-issue eligible wave64 halves on the dual-issue VALU path,
  while wave32 exposes VOPD pairing directly to the compiler. These scheduling features
  are orthogonal to the wavefront-size flag.
- Prefer wave32 + ILP: multiple independent accumulators, unrolled loops, fewer long
  dependent VALU chains, and low enough VGPR/scratch/LDS pressure to preserve occupancy.
- Prefer wave32-compatible collectives: `__shfl_down` within 32 lanes, then LDS/shared
  memory exchange for cross-wave reductions. Do not remove barriers on the theory that a
  64-thread block is a single wave.
- Only pursue wave64 as an isolated experiment with explicit `-mwavefrontsize64` build
  flags, `warpSize`/shuffle probes, correctness fixtures, ISA checks, and E2E benchmarks.
  There is no retained wave64 default in hipEngine.

### JIT cache gotcha

Symptom: kernel calls hang with GPU at 0% utilization and no error. This is almost always a stale cached `.so` that doesn't match the current source. Nuke the matching cache dir before re-importing:

```bash
rm -rf ~/.cache/hipengine/build/<family>-<hash>*
```

If the family is unknown, clearing the whole cache is cheap (~5 s per kernel to rebuild):

```bash
rm -rf ~/.cache/hipengine/build/
```

The hash incorporates the source file content, the flag set, and the `hipcc --version` string. If you change `hipcc` underneath an existing cache, the hash will change and old entries will be ignored — not overwritten. Prune manually when the cache grows unbounded.

## rocprofv3 smoke (port parity + new kernel check)

Minimum smoke a port or a new kernel must produce:

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  uv run python scripts/smoke.py <model> <workload>
```

If the target path uses `hipengine.core.build` JIT from Python, prebuild outside the profiler and make the profiled process cache-only. `rocprofv3` launch mode preloads into child processes; letting a profiled Python process spawn `hipcc`/clang can hang or abort in LLVM initialization.

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_smoke_add(load=False, compiler_version=version).output_path)
PY
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Expected output: a CSV with a kernel-name column (`Kernel_Name` / `KernelName`), grid/workgroup columns, `VGPR_Count`, `Scratch_Size`, and `LDS_Block_Size`. Some ROCm 7.13 traces emit `Start_Timestamp` + `End_Timestamp` instead of `DurationNs`; compute `DurationNs = End_Timestamp - Start_Timestamp` for summaries. Check:

- The expected kernel name appears.
- Duration is plausible (same order of magnitude as the reference).
- `Scratch_Size > 0` on a hot-path kernel is a red flag — escalate to `~/amd-gpu-tuning/` for audit.
- `VGPR_Count ≥ 96` may be squeezing occupancy — same.

rocprofv3 dumps are **not committed**. Store under `/tmp/` or outside the repo.

## Registering a kernel

Kernels self-register on module import:

```python
# hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py
from hipengine.kernels.registry import KernelKey, register
from hipengine.core.build import build_hip

_so = build_hip(
    sources=["paged_attn_decode.hip"],
    profile="decode",
    family="attention",
)

def paged_attn_decode_fp16(...): ...

register(
    KernelKey(backend="hip_gfx1100", layer="paged_attn_decode",
              quant="fp16", variant="split_k_warp"),
    paged_attn_decode_fp16,
)
```

The resolver does narrowest-to-broadest match: `variant` → no-variant → `quant="fp16"` fallback → `backend="cpu_reference"`. A new backend implementation or a new fused composite is a `register(...)` call, never an `if backend == "..."` branch in dispatch code.

## Per-family port checklist

When bringing up a family (`attention/`, `moe/`, `quant/`, …), follow in order:

1. Copy the relevant kernels from the monolithic source into `kernels/hip_gfx1100/<family>/*.hip`. Preserve bodies byte-for-byte.
2. Retype the host-side launch wrappers.
3. Move the `PYBIND11_MODULE` / `TORCH_LIBRARY` entries for this family from `csrc/amd/extension.cpp` into `kernels/hip_gfx1100/common/extension.cpp` (the aggregator).
4. Write `register(KernelKey(...), ...)` calls in the Python wrapper module so the kernels resolve.
5. Add a CPU-reference implementation for every new `layer` key in `kernels/cpu_reference/`.
6. Run the port correctness gate (all three checks above).
7. Commit the family as one logical unit with `port:` prefix and `nano-vllm-amd@<sha>` in the body.

Do not interleave families in one commit. A commit that touches `attention/` and `moe/` together is harder to bisect and harder to review.
