# Laguna S 2.1 Decode Gap Analysis — W7900 / UD-Q2_K_XL

Status: expanded diagnostic plan complete; P0 exact-IQ3 ownership and P2.1
exact split attention are implemented and retained as gfx1100 defaults. Both
P1 IQ3, raw-Q5, and raw-IQ2 lanes plus P2.2 online FP32 partials are rejected.
The exact SWA tile16 score producer is retained as the gfx1100 default from live
count 257; P1 and P2 are closed. P3's bit-lossless Q5 T16 replacement screen is
rejected. P4.1's exact P2-derived split-reducer+softplus-gate body and the
current-P4 exact head-RMSNorm+RoPE+BF16-KV body are retained as gfx1100
defaults. A correctness-fenced one-doorbell native-AQL owner is slower and
rejected. The device-pinned matched Vulkan reaudit still fails at both horizons.
A fresh exact SWA split-reducer max-scan synchronization contraction passed
correctness and actual-layer screens but failed the clean short dispatch-span
guard; it is rejected and removed before category measurement. A distinct
wave-local reducer that removes all remaining block barriers and reducer LDS is
retained as the gfx1100 default after exact model-state, trace, clean-context,
and complete-category gates. A subsequent exact local32 IQ2 tile2 reconstruction
is bit exact but slower on both first/last actual layers and is removed. A
separate, c=1-only wave64 code-object build of the unchanged tile2 source also
fails its clean promotion gate and is removed. A new exact c=1 sibling using a
canonical 64-bit IQ2 magnitude table passes clean context and complete-category
gates and is retained as the gfx1100 default. A subsequent exact raw-Q5
fixed-metadata sibling passes ISA, first/last actual-weight, full-state, cached
trace, both clean context orders, and both complete-category orders and is now
the gfx1100 default with an explicit role-scoped rollback. A subsequent exact
mixed Q5/Q6 plus corrected layer-47 Q6/Q8 projection quad removes 49 launches
per token and is retained as the gfx1100 c=1 default after the same complete
state, context, and category gates. Its subsequent exact fixed-Q6-metadata
sibling preserves the 723-launch structure and is retained as the gfx1100
default after improving every clean context and every train/heldout category
decode in both complete process orders. The subsequent exact BF16 shared-Q5
pair is also retained as the gfx1100 default after production-shape and actual-
weight bit identity, **26.88-27.61%** first/last event-and-wall wins, full-state/
lifecycle identity, both clean context orders, and both complete category
orders. It preserves 723 dispatches/token and moves h32 decode **59.500 ->
60.942 tok/s (+2.425%)**. A subsequent exact local64 heterogeneous IQ2/shared-
Q5 owner would remove 46 launches/token without the earlier local128/four-wave
penalties, but it regresses first/last actual-layer event and wall
**0.12-0.68%** and is rejected and removed before runtime integration. A
narrower exact Q5-output screen packs two unchanged fixed-metadata waves into
one local64 workgroup, but all first/last global/SWA boundaries regress event
**6.69-10.15%** and wall **6.51-7.66%**; it is also removed before runtime. The
subsequent exact SWA GQA3 reducer preserves every head's bytes and reduces
modeled BF16 value payload **66.67%**, but collapsing 72 independent query-head
workgroups to 24 regresses all layer-1/46/47 live-70/128/257/512 rows by
**61.33-89.58%** event and **61.16-89.95%** wall. It is removed before runtime;
GQA value reuse does not offset the lost parallelism/cache-visible reuse. The
subsequent exact all-local32 Q5/Q6 mixed projection is now retained as the
gfx1100 default. It keeps the same **131,840/181,376** global/SWA grid threads
and wave count, gives each retained Q5 pair its own workgroup, and replays each
Q6 output pair's four original local128 partitions in one wave. Production
outputs and full model state are bit-exact; first/last actual layers improve
**11.39-14.77%** in HIP events and **11.24-15.72%** in synchronized wall. Both
clean orders improve projection/kernel/span/child work, and both complete
18-prompt orders move h32 decode **60.900 -> 61.732 tok/s (+1.367%)** with every
train/heldout category positive. Cached tracing confirms 47 local32 calls plus
one retained layer-47 call at unchanged **723 dispatches/token**. Canonical h32
is now **61.732 tok/s**; Vulkan still requires another **4.35%**. Post-local32
re-ranking selected one new exact SWA reducer screen: keep all 72 query-head
workgroups, use local64 with two adjacent dimensions per thread, and halve the
retained wave-local reducer's replicated scalar softmax state and packed-load
instruction work. The primitive is byte-exact and all 12 isolated rows improve
**0.275-0.685% event** / **0.295-2.294% wall**; its default-off selector also
passes full state and tracing. The frozen clean gate nevertheless rejects it:
at context 512 the reducer regresses **0.073%** and complete SWA regresses
**0.247%**. Runtime selection is removed; the primitive remains diagnostic.
Post-dim2 re-ranking selected and primitive-admitted a materially different
exact IQ3 screen: a separately registered K1024 wave4 sibling replaces 32
compare/select sign operations with load-free IEEE-754 sign-bit insertion while
preserving all ownership, FMA, reduction, BF16, and launch boundaries.
Repository codegen contracts static instructions **527 -> 499** and logical
SGPR **28 -> 18**; layers-1/45 producer and inclusive producer+reducer event and
wall improve **6.20-9.15%** with exact bytes. Full-state and cached-trace
admission also pass, but both frozen short process orders fail end-to-end guards:
span regresses **0.571%/1.931%**, and order-A profiled-child throughput regresses
**1.124%**. Runtime selection is removed, categories are skipped, and the
primitive remains diagnostic. Post-sign-bit re-ranking selects one materially
new exact router composite: retain D11's exact projection and self-resetting
last-block election, but replace its ten block-wide selector rounds with
per-wave top-10 plus a register-resident wave-0 merge. Repository primitive
admission is now exact across hidden-17/3072 synthetic cases and every actual
router, and the fresh all-layer event/wall window improves split
**23.26%/23.23%** while beating rejected D11 **4.83%/4.84%**. The separately
registered primitive passed default-off full-state and cached 47-call/
**676-kernel** admission, but failed both frozen short clean orders: the complete
router family regresses **14.42%/13.69%** and kernel sum regresses
**0.736%/1.422%**. Runtime selection/counter ownership is removed and categories
are skipped; the primitive remains diagnostic. Post-router re-ranking selected
an exact c=1 Q4_K LM-head local32 fixed-metadata output-pair sibling. Its
repository primitive and runtime owner pass exact synthetic/production,
codegen, actual-weight, 16-transition full state, and cached one-call/
**723-kernel** tracing. Both clean process orders improve the LM head
**29.07-30.79%** and kernel sum **0.34-1.10%** at every context. Both complete
category orders move h32 decode **61.675 -> 61.992 tok/s (+0.512%)** with every
train/heldout category positive, so local32 is now the gfx1100 default and
explicit local128 remains rollback. Canonical h32 improves **61.732 -> 61.992
tok/s (+0.420%)** but still needs **3.91%** to match Vulkan.

The post-Q4 exact screen leaves a diagnostic primitive only: the already-
retained local32 Q6 output-pair helper has a standalone c=1 BF16/BF16 wrapper
and four-axis key. Synthetic boundaries and all **50** actual runtime Q6 weights
are BF16-bit exact, and repeated endpoints improve **11.42-22.32% event** and
**11.46-20.51% wall**. A temporary default-off owner passed 16-transition state
and cached **50-call/723-kernel** tracing, but failed the frozen short clean gate
on kernel-sum/span/child guards. Runtime integration is removed and categories
are skipped; the default and **61.992 tok/s** canonical row remain unchanged.
Post-Q6 re-ranking selects a genuinely different router boundary: retain all 47
registered BF16-hidden/F32-weight projection launches and replace only the
stateless correction selector with one local32 wave carrying eight experts per
lane. The separately registered repository primitive now passes RED/GREEN, CPU
and field-bit exact gates, all 47 actual projected-logit/correction rows, frozen
codegen/resource ceilings, and cache-only tracing. Its repeated repository
window improves **25.78% event / 25.77% wall**. Its temporary default-off c=1
owner passes full-state and 47+47/723 tracing, but both clean short orders
reverse the selector/router/kernel/child result. Runtime integration is removed,
categories are skipped, and the default/topline remain unchanged. Post-compact
re-ranking selects a different MoE boundary: compile-time top-10 feature-
parallel weighted+routed/hidden production followed by the unchanged exact
RMSNorm. The separately registered repository primitive now passes RED/GREEN,
synthetic/CPU/all-47-layer exactness, frozen codegen/resources, and cache-only
tracing. Its repeated repository two-call window improves **12.664% event /
12.664% wall** at unchanged launch count. A temporary default-off owner passes
exact 16-transition state and cache-only **47 producer + 47 registered RMS /
723-model-kernel** tracing, but the frozen clean short gate fails order B on
kernel-sum and span guards. Runtime integration is removed, remaining contexts
and categories are skipped, and the default/topline remain unchanged.
Post-weighted re-ranking now selects a genuinely different exact mixed-
projection dataflow: one local32 wave carries one Q5 output pair and one Q6
output pair, reusing each BF16 activation register while preserving both
retained arithmetic trees. The separately registered repository primitive now
passes RED/GREEN, K256/1024/3072/9216 synthetic and CPU-reference gates, all 47
actual-weight layers, the frozen codegen/resource ceiling, and cache-only
tracing. Its repeated repository first/last-layer screen improves **2.86-7.16%
event / 2.45-5.43% wall**. A temporary default-off owner passes exact
16-transition full state and cache-only **47 candidate + one unchanged layer-47
/ 723-model-kernel** tracing, but the frozen short clean gate rejects it: order
A regresses span **1.265%** and order B regresses child throughput **0.681%**.
Runtime integration is removed, remaining contexts/categories are skipped, and
the default/**61.992 tok/s** topline remain unchanged. Post-pair-reuse re-ranking
selects the existing 48-call add+RMSNorm boundary: stage each already-computed
unrounded F32 add in LDS and reuse it for the weighted norm output instead of
reloading both BF16 inputs. All 48 actual boundaries are bit-exact and the
complete event/wall window improves **3.46%/3.51%** out of tree. The separately
registered repository primitive now passes RED/GREEN, synthetic/CPU/all-48
exactness, codegen, cache-only trace, and **3.53%/3.70%** repository event/wall
transfer gates. A temporary default-off owner then passes exact 16-transition
state and cache-only **48-candidate/723-model-kernel** tracing, but the frozen
clean short gate rejects it: order A regresses complete kernel sum **0.340%**
and order B regresses span **0.528%**, above the 0.5% guard. Runtime integration
is removed, remaining contexts/categories are skipped, and no default or topline
has changed. Post-staged-add re-ranking returns to the retained IQ2 grid64 body
without reopening its closed geometry experiments. A fixed-local64 reduction
keeps both parallel wave32 K owners but replaces 20 LDS permutations with four
`permlanex16` plus 16 DPP transports and specializes the cross-wave tail to the
known two waves. First/last actual IQ2 layers are BF16-bit exact and improve
**0.96-1.22% event / 1.05-1.26% wall**. The separately registered repository
primitive now passes RED/GREEN, CPU, all-46-layer, codegen, repeated endpoint,
and cache-only trace gates. A temporary explicit/default-off owner also passes
exact 16-transition state and cache-only **46-candidate/723-model-kernel**
tracing, but the frozen clean short gate rejects it: both orders regress the IQ2
family **0.366%/2.742%**, order A regresses child throughput **0.634%**, and
order B regresses kernel sum **1.180%** plus span **9.707%**. Runtime integration
is removed, remaining contexts/categories are skipped, and no default or topline
has changed. Post-IQ2 re-ranking selects a materially different exact Q5
instruction contraction. Paired-output SWAR reconstruction preserves every Q5
value and FP32 boundary while sharing nibble/high-bit operations across the two
rows already owned by one wave. The separately registered repository primitive
now passes RED/GREEN, CPU-reference, every actual Q5 production boundary,
codegen, repeated endpoint, and cache-only trace gates. A temporary explicit/
default-off all-or-none owner also passes exact 16-transition state and
cache-only **47 mixed + 47 output + 46 shared = 140 candidate calls/token / 723
model kernels/token** admission, but the frozen short clean gate rejects it.
Both orders regress the combined Q5 family **0.457%/0.077%**, kernel sum
**0.280%/0.075%**, and child throughput **0.723%/0.592%**; order A also regresses
span **0.883%**. Runtime integration is removed, later contexts/categories are
skipped, and defaults/topline remain unchanged. Post-Q5 re-ranking now selects
an exact IQ3 producer/reducer boundary rather than another reconstruction or
geometry rewrite. One local320 output workgroup groups the ten retained wave32
route owners, preserves every K1024 dot/tree/BF16 route boundary, and replays the
registered weighted reducer from `+0.0` in slot order after one 20-byte LDS tuple
and barrier. The first/last actual-weight screen is BF16-bit exact and improves
inclusive event **8.31%/7.27%** and wall **7.49%/7.25%**. The separately
registered repository primitive now passes **7/7** focused
RED/GREEN, independent CPU KL/top-1, exhaustive selector/grid and BF16/routing
edges, **45/45** actual-layer byte identity, repeated endpoint, codegen, and
cache-only trace gates. Repository endpoints improve event **8.51%/7.68%** and
wall **7.18%/6.53%** at layers 1/45; integrated codegen is local320/VGPR83
logical/88 allocated/fixed-LDS20/scratch0 with 495 instructions and one barrier.
The owner removes **45 launches/token** (**723 -> 678**). It passes shared-weight
bulk prefill, all **48 hidden + 47 routed** boundaries, 16 decode transitions,
active K/V and every span field, reset/re-prefill, ownership, teardown, and a
no-argument-default versus explicit-wave4 replay at KL0/top-1 100%. Cache-only
tracing records **45 candidate + two unchanged reducers / 678 model
kernels/token**, zero candidate prefill/wave4/serial decode calls, and
local320/VGPR88/LDS512/scratch0. Every clean short/512/1K/near-4K order improves
inclusive IQ3 **9.71-11.90%**, kernel sum **0.398-1.082%**, and span
**0.813-1.998%**, with child throughput inside guard. Both complete 18-prompt
orders move h32 **62.318 -> 63.270 tok/s (+1.528%)** with every train/heldout
category positive at both horizons and all E2E/prefill/TTFT guards passing.
gfx1100 now defaults wave10-fused; explicit wave4 remains rollback. Relative to
the prior retained row, canonical h32 improves **61.992 -> 63.270 tok/s
(+2.063%)** and needs another **1.81%** to match Vulkan.

Post-wave10 re-ranking uses both retained short traces at **12.737 ms/token**.
Mixed Q5/Q6 (**1.930 ms**) and IQ2 (**1.896 ms**) remain mechanically closed
under their rejected unchanged owners. A fresh exact SWA local32/dim4 screen
keeps all 72 head workgroups and reduces softmax-state replicas from two to one,
but **9/12** required layer/live rows regress both event and wall by up to
**1.24%/1.45%**; reject it out of tree. The next selected owner instead narrows
the already-admitted Q5 paired-SWAR primitive to only the 47 attention-output
calls. In the immutable rejected all-owner traces, this role independently
improves **1.952%/2.046%** in orders A/B, while mixed projection regresses
**2.278%/1.694%** and shared gate/up regresses **0.902%/0.439%**.
A temporary false/default-off output-only owner passes fresh shared-weight full
state and cache-only **47-candidate/678-model-kernel** tracing with zero SWAR
mixed/shared/query-gate calls. The frozen clean short gate nevertheless rejects
it: attention-output and kernel sum improve **4.001%/3.705%** and
**1.452%/0.509%** in orders A/B, but profiled-child throughput regresses
**1.061%/1.035%**, outside the -0.5% guard in both orders. Runtime integration
is removed; later contexts/categories are skipped and the primitive remains
diagnostic.

The post-SWAR search rejects three independent exact contractions out of tree:
IQ3 two-output wave10 regresses endpoint wall **16.40-20.22%**, staging all SWA
softmax weights regresses every required wall row (up to **21.95%**), and a
K3072 mixed-Q5/Q6 specialization regresses the dominant SWA layers
**1.30-3.52%** wall despite smaller codegen. The selected next body instead
composes the already-proven load-free IQ3 sign reconstruction *inside* the
retained wave10 fusion. This is not the rejected standalone wave4 owner: it
keeps the retained **45 fused + two reducer / 678-kernel** topology unchanged.
All **45/45** production outputs are BF16-bit exact; layers 1/45 improve event
**8.02%/8.73%** and wall **7.15%/7.38%**. The separately registered repository
primitive now passes **7/7** RED/GREEN, exhaustive selector/grid and BF16/
routing edges, ten independent CPU cases (KL max **4.01e-5**, top-1 100%), and
a repeated **45/45** actual transfer. Repository layers 1/45 improve event
**8.95%/8.39%** and wall **8.36%/7.57%**. Integrated Clang-22 codegen contracts
**495 -> 454 instructions / 2860 -> 2700 bytes**, removes all 32 sign
compare/select pairs, and stays local320/VGPR88/LDS512/scratch0; cache-only
tracing names the distinct sibling twice with no compiler. The
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-design.json)
and [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-correctness.json)
are committed separately. A temporary [`false/default-off runtime owner`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-runtime-correctness.json)
passes shared-weight 16-transition state and exact **45 candidate + two reducer /
678-model-kernel** tracing. The frozen [`clean gate`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-rejected.json)
then rejects ownership: short passes both orders, but context-512 order B span
regresses **0.862%**, beyond +0.5%, despite IQ3 **-5.04%**, kernel sum **-0.014%**,
and child throughput **+0.408%**. Favorable pooled span **-0.736%** cannot waive
the per-order failure. The gate stops before 1K/3968/categories; runtime
capability/schedule/CLI are removed, while the primitive and retained wave10
default remain unchanged.

Post-sign-bit re-ranking selects a different, already-registered exact boundary:
the two remaining IQ4_XS expert-down layers currently execute two route-parallel
selected producers plus two weighted reducers. The existing routing-weighted
composite preserves each route's BF16 projection boundary and slot-order
weighting while contracting **4 -> 2 calls** per token. Dedicated Laguna
certification now passes **7/7** package/key/backend, synthetic edge, registered-
fallback, and independent CPU contracts (KL max **1.58e-69**, top-1 100%). The
repeated layers-46/47 gate is BF16-bit exact and improves the inclusive
producer+reducer event/wall boundary **28.59-34.22%**. Integrated codegen remains
local256/wave32, logical/allocated VGPR **78/80**, fixed/allocated LDS **32/512
B**, private/spills/scratch0, **492 instructions / 2,580 bytes**; cache-only
tracing names two grid-786,432 calls with no compiler. The [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-design.json)
and [`certification`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-correctness.json)
are committed separately. A temporary [`false/default-off runtime owner`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-runtime-correctness.json)
passes shared-weight 16-transition state and exact **2-IQ4/45-wave10/676-kernel**
tracing with zero IQ4 split decode calls/reducers. The frozen [`short clean gate`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-rejected.json)
then rejects ownership in both orders: inclusive IQ4 regresses
**25.515%/25.191%**, kernel sum regresses **0.202%/0.303%**, and span regresses
**0.681%/2.198%**. Child throughput improves but cannot waive those failures.
The gate stops before 512/1K/3968/categories; runtime integration is removed,
while the primitive and retained **63.270 tok/s / 678-kernel** default remain.

Post-IQ4 re-ranking selects global attention, the largest independent open short
surface at **0.523579 ms/token / 12 calls**. The exact source-identical sibling
is separately registered on gfx1100. Synthetic/CPU and **24/24** actual-layer
primitive gates pass; shared-weight bulk prefill, all **48 hidden / 47 routed**
boundaries, 16 transitions crossing live 127, active K/V plus every span field,
reset/re-prefill, and lifecycle are exact at KL0/top-1 100%. Cache-only tracing
records **12 candidate calls/token**, zero scalar-global, unchanged **36+36 SWA
split calls/token**, and **678 model kernels/token** at local256/VGPR32/dynamic-
LDS16928/scratch0.

The frozen clean short gate rejects runtime ownership without rerun. Both orders
improve the global family **10.97%/12.12%**, but order A regresses complete
kernel sum **0.0949%** and span **1.3203%**. Order B and pooled improvements
cannot waive those per-order failures, so 512/1K/3968 and categories are skipped.
The [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-runtime-correctness.json),
and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-rejected.json)
retain the primitive only. Runtime selection is removed; scalar below live 127,
split-exact at live>=127, and canonical **63.270 tok/s / 678 kernels** remain.

The next selected design is a materially new global-only composition: append
D15's proven exact softplus/RNE-BF16 epilogue to the admitted one-page body in
the same local256 workgroup. It does not restore D15/D17 head/KV or SWA bundles.
The unfused one-page+gate chain remains required. Current two-order scalar+gate
windows are **0.5558/0.5664 ms/token** versus **0.5008/0.5029** for one-page plus
separate gate; a zero-increment epilogue ceiling removes 12 more launches and
models h32 **63.653 tok/s (+0.605%)**, still below Vulkan. The
[`gated design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json)
is not implemented and makes no throughput/default claim.

Scope: resident batch-1 autoregressive decode of
`Laguna-S-2.1-UD-Q2_K_XL.gguf` on one AMD Radeon Pro W7900 (`gfx1100`). This
explains the measured gap between llama.cpp Vulkan and hipEngine, audits the
Qwen3.x optimization history for missed transfers, and ranks future work. It
does **not** replace the canonical benchmark or make an apples-to-apples
cross-engine throughput claim.

Compact evidence:

- [`2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json)
- [`2026-07-24-gfx1100-laguna-q2-xl-hip-vulkan-isa-attention-review.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-hip-vulkan-isa-attention-review.json)

## Executive answer

The supplied result is real and reproducible:

```text
llama.cpp Vulkan tg128: 93.67 +/- 0.16 tok/s
independent rerun:      94.513 +/- 0.141 tok/s
hipEngine D12 h32:      48.987 tok/s
```

`llama-bench tg128` and hipEngine's canonical h32 suite are not the same
protocol, so `94.513 / 48.987 = 1.929x` is diagnostic rather than a formal
product-throughput ratio. Since that frozen diagnosis, retained P0 IQ3, P2.1
exact split attention, P4.1 split-reducer+gate fusion, and current-P4 head+KV
fusion plus later exact quantized-leaf work and IQ3 wave10 fusion move the
current counterbalanced h32 row to **63.270 tok/s / 15.805 ms/token**; the
non-equivalent `llama-bench` row still represents a **49.38%** diagnostic gap.
However, the gap is not explained by prompt depth, sampling, Python, or one
missing launch flag:

1. A llama.cpp control with the same mean timed context depth as hipEngine still
   reaches **94.152 +/- 0.331 tok/s**.
2. hipEngine's clean D12 trace sums to **16.486 ms of GPU kernels/token**. That
   kernel sum alone is already **5.905 ms longer** than llama.cpp Vulkan's
   complete **10.581 ms/token wall**.
3. Even deleting every hipEngine submission gap, synchronization, argmax,
   scalar read, and Python action would cap D12 at **60.658 tok/s**.
4. Disabling **all Vulkan graph fusion** still leaves llama.cpp at
   **74.865 tok/s / 13.357 ms/token**, substantially ahead of D12. Fusion is
   important, but format-specific device work is the larger cause.
5. llama.cpp becomes faster when its Q8_1/MMVQ route is disabled:
   **98.568 tok/s**. Blindly porting integer-dot activation quantization is not
   the answer; hipEngine's inclusive IQ2 Q8_1/dp4a attempt independently
   regressed.
6. A clean same-commit llama.cpp HIP build reaches only
   **55.621 +/- 0.210 tok/s**. Its **1.699x** same-source Vulkan deficit proves
   that most of the backend gap survives inside llama.cpp, but its open HIP
   kernels expose two concrete schedules worth adapting: selected IQ3 down is
   **1.422 ms/token**, and tiled attention is **0.558 ms/token**.

The direct conclusion is:

> hipEngine is not missing one basic host optimization. It is missing a
> compound Vulkan-class implementation for Laguna's raw IQ2/IQ3/Q5 operators
> and attention schedule. The retained runtime already contains most of the
> successful Qwen tactics, while the remaining submission/fusion work is too
> small to close a 9.83-ms wall gap by itself.

Matching 94.5 tok/s while retaining D12's current non-kernel residual would
require kernel sum to fall from **16.486 to 6.653 ms/token (-59.65%)**. Even if
all non-kernel time disappeared, kernel sum still needs **-35.82%**. This is a
multi-family kernel campaign, not another launch-only sweep.

## 1. Input result and exact identity

User result:

```bash
GGML_VK_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -fa 1 \
  -m /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf
```

```text
AMD Radeon Pro W7900 (RADV NAVI31)
model size: 36.96 GiB
parameters: 117.56 B
pp512: 57.32 +/- 0.48 tok/s
tg128: 93.67 +/- 0.16 tok/s
```

The analysis pins:

| Item | Identity |
| --- | --- |
| Model | `/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf` |
| SHA-256 | `8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679` |
| File size | 39,684,584,480 bytes |
| GPU | AMD Radeon Pro W7900 / gfx1100 |
| llama.cpp | `c0bc8591e8815c63cb01dd3f051a8b0df02501c9` |
| llama binary SHA-256 | `0d466d22faf045c7bfd8c03bac28fd8f581a3a82edafacf99e51202ab5988759` |
| hipEngine retained route | D12, implementation `338d3afca`, current main includes it |
| hipEngine KV | BF16, `KVLiveSpans`, admitted capacity 4096 |
| llama.cpp KV | F16, right-sized benchmark context |

The pp512 row is useful context but is not analyzed as decode. hipEngine's
canonical D12 prefill is about 43 tok/s under a natural ten-prompt correctness
suite, which is also not protocol-equivalent to `llama-bench pp512`.

## 2. Protocol comparison

### 2.1 What `llama-bench tg128` measures

For each repetition, llama.cpp clears memory, performs one untimed generation
warmup, and times 128 one-token `llama_decode` calls. Input IDs are forced/random
benchmark tokens. Each call synchronizes, but the timed loop does not sample a
next token or run hipEngine's output checks.

Independent reproduction:

```bash
GGML_VK_VISIBLE_DEVICES=0 \
/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  -fa on \
  -m /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf \
  -p 0 -n 128 -r 3 -o json
```

| Sample | tok/s |
| ---: | ---: |
| 1 | 94.6513 |
| 2 | 94.5176 |
| 3 | 94.3701 |
| **Mean** | **94.513** |
| Stddev | 0.141 |
| Mean wall | **10.581 ms/token** |

### 2.2 What hipEngine D12 measures

The canonical retained row uses:

- ten prompts across `code`, `general_en`, `general_ja`, and `mixed_ja_en`;
- 68–122 input tokens, mean 86.4;
- natural greedy trajectories;
- h16 and h32 output horizons;
- GPU argmax, synchronization, token-ID and max-logit scalar reads;
- two complete counterbalanced process-order pairs;
- exact hidden/logit/token, full KV/`KVLiveSpans`, reset, lifecycle, and
  category non-regression gates.

The retained h32 result is **48.987 tok/s / 20.414 ms/token**.

### 2.3 Context-depth control

The h32 suite times 31 post-TTFT calls per trajectory. Its mean timed position
is 101.4. This llama.cpp control matches that call count and mean depth:

```bash
GGML_VK_VISIBLE_DEVICES=0 \
/home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  -fa on \
  -m /models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf \
  -p 0 -n 31 -d 86 -r 3 -o json
```

Result: **94.152 +/- 0.331 tok/s / 10.621 ms/token**. This is only
position-depth matching—it does not equalize prompt IDs, sampling, KV dtype, or
arithmetic—but it rules out the 2x gap being a trivial 0-vs-100 token context
effect.

### 2.4 Remaining protocol differences

| Difference | Can it explain 2x? |
| --- | --- |
| Forced-token evaluation vs natural greedy sampling | No. D12's complete kernel sum is already slower than the Vulkan whole wall. |
| hipEngine argmax and two scalar reads | No. Argmax kernels are about 0.007 ms/token; all non-kernel time together cannot cross the kernel floor. |
| F16 KV vs BF16 KV | Not isolated here; both are 16-bit storage. It may affect arithmetic/codegen but cannot be credited without a matched quality gate. |
| Right-sized context vs 4096-capacity `KVLiveSpans` | Depth control stays at 94.15 tok/s; D12 attention reads live spans rather than all capacity. |
| Three long repetitions vs ten-prompt category suite | Important for claim eligibility and variance, but not the GPU kernel floor. |

The matched completion protocol supplies the closest retained cross-engine
boundary. It runs all 18 category+heldout prompts at context 4096, natural
greedy h16/h32, and two repetitions, then counts the same post-TTFT transitions:
hipEngine `decode_forward_calls/decode_seconds` versus llama.cpp Vulkan
`sum(predicted_n - 1) / sum(predicted_ms)`. The pre-current-P4 audit measured
hipEngine **51.839/51.432 tok/s** and Vulkan **64.213/64.336 tok/s**.

The required post-current-P4 reaudit explicitly pins
`GGML_VK_VISIBLE_DEVICES=0`, reproduces Vulkan at **64.245/64.418 tok/s**
(within **0.05%/0.13%** of the original), and measures hipEngine at
**52.855/52.391 tok/s**. hipEngine remains **17.73%/18.67% slower** and needs
**21.55%/22.96%** more throughput. Two unpinned diagnostics at about
**57.1-57.6 tok/s** are excluded because they do not follow the user's or the
canonical explicit-device protocol. The matched objective still fails.

Prompt IDs, natural sampling, horizons, context length, and transition ownership
match. One unavoidable arithmetic difference remains: hipEngine uses BF16
`KVLiveSpans`, while Vulkan uses F16 KV because the reported device capability
has no BF16 support. All 72 server-native prompt/predicted timing rows are valid.
The SSE Content-only path omits one or more returned token-array entries for 18
rows even though `predicted_n` and timings are complete, so returned-array
length is reported but not used as the timing gate. Cross-engine IDs are also
reported, not substituted for hipEngine's exact within-engine correctness and
lifecycle gates. Evidence: [initial audit](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-audit.json) and [current-P4 device-pinned reaudit](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-vulkan-matched-completion-reaudit.json).

## 3. hipEngine D12 profile

The clean profile uses cached JIT objects under `rocprofv3`, excludes two
warmup rows, and averages 14 stable c=1 steps after the 69-token
`code_merge_intervals` prompt.

| Metric | D12 short profile |
| --- | ---: |
| Dispatches/token | **775** |
| Summed GPU kernels | **16.486 ms/token** |
| Median embedding-to-argmax dispatch span | **19.567 ms/token** |
| Canonical h32 wall | **20.414 ms/token** |
| Kernel-only zero-overhead ceiling | **60.658 tok/s** |

### 3.1 Hot families

| Family | ms/token | Kernel share | Calls/token | Current implementation |
| --- | ---: | ---: | ---: | --- |
| selected IQ2 gate/up + SiLU | **2.358** | **14.31%** | 46 | branchless pair16, local64, output-tile2 |
| Q5 attention output | **2.186** | **13.26%** | 47 | D12 raw wave32x2 |
| token4 SWA attention | **2.153** | **13.06%** | 36 | exact four-slot score parallelism |
| selected IQ3 weighted down | **2.131** | **12.92%** | 45 | wave-uniform, local128, routing fused |
| Q5 query + per-head gate | **1.838** | **11.15%** | 47 | D12 unequal raw wave32x2 pair |
| Q6 attention pair | 0.945 | 5.73% | 48 | exact equal/unequal pair |
| Q5 shared gate/up pair | 0.905 | 5.49% | 46 | exact same-input pair |
| Q6 BF16 projections | 0.601 | 3.65% | 50 | exact raw decode leaf |
| global attention | 0.518 | 3.14% | 12 | exact dense live-span scan |
| add + RMSNorm | 0.395 | 2.40% | 48 | fused primitive |
| router select | 0.394 | 2.39% | 47 | split exact top-10 route |
| MoE tail + next RMSNorm | 0.389 | 2.36% | 47 | D9 aggregate composite |
| Q4 lm-head | 0.383 | 2.32% | 1 | exact c=1 raw leaf |
| router projection | 0.323 | 1.96% | 47 | F32 weight projection |

The largest four families consume **8.828 ms/token (53.55%)**. Including D12's
Q5 query/gate leaf raises the concentration to **10.666 ms/token (64.70%)**.
No single small fusion can close the gap.

### 3.2 Resource audit

The hot leaves already satisfy the basic RDNA3 hygiene that fixed Qwen:

| Family | Workgroup / resources |
| --- | --- |
| D12 Q5 wave32x2 | local32, VGPR96, LDS0, scratch0 |
| IQ2 selected dual tile2 | local64, VGPR136, LDS512 B, scratch0 |
| IQ3 weighted down | local128, VGPR32, LDS512 B, scratch0 |
| SWA token4 | local128, VGPR24, dynamic LDS4120 B, scratch0 |
| D9 MoE tail | local256, VGPR16, scratch0 |

All use the decode build profile's
`-mllvm -amdgpu-unroll-threshold-local=600 -mcumode`. There is no broad spill,
wrong-wavefront, missing-unroll, or generic-row-GEMV explanation left in these
families.

### 3.3 Context scaling

| Prompt tokens | Kernel sum | Dispatch span | Dominant context work |
| ---: | ---: | ---: | --- |
| 69 | 16.486 ms | 19.567 ms | quant GEMV + short attention |
| 512 | 29.662 ms | 32.987 ms | SWA reaches its 512-token window |
| 1,024 | 32.687 ms | 36.002 ms | SWA plateau + growing global attention |
| 3,968 | 49.383 ms | 52.774 ms | global attention dominates |

The supplied `tg128` comparison is a short-context question. At 512+ tokens,
matching Vulkan also requires a fundamentally stronger attention scan; quant
kernel work alone cannot make long-context Laguna fast.

## 4. Vulkan controls and source findings

### 4.1 Ablations

All rows use the same W7900, binary, model, `-fa on`, p0/n128, one warmup token,
and three repetitions.

| Vulkan mode | tok/s | ms/token | Change vs default |
| --- | ---: | ---: | ---: |
| default | **94.513** | **10.581** | — |
| graph optimization disabled | 93.489 | 10.696 | -1.08% |
| fusion disabled | 74.865 | 13.357 | -20.79% |
| graph optimization + fusion disabled | 74.514 | 13.420 | -21.16% |
| MMVQ disabled | **98.568** | **10.145** | **+4.29%** |

The fusion ablation adds **2.777 ms/token**, only **28.24%** of the
9.833-ms D12-to-Vulkan wall gap. The other **71.76% remains even when Vulkan
fusion is disabled**. Graph reordering itself accounts for only 0.116 ms.

This is the strongest causal evidence in the analysis: Vulkan's raw operators
are much faster before its fusion advantage is counted.

### 4.2 What Vulkan fuses/groups

The reviewed source and `GGML_VK_PERF_LOGGER_CONCURRENT` trace show dependency
groups for:

- flash attention with the per-head Q5 gate projection;
- Q5 query with two Q6 K/V projections;
- selected IQ2 gate/up pairs;
- IQ3 selected-down with route weighting;
- Q5 attention output plus residual add;
- shared gate/up plus add;
- RMSNorm/multiply and RMSNorm/RoPE/KV row writes;
- top-k sigmoid, correction bias, selection, and normalization.

The warmed logger emitted **767 dependency groups/token**, close to hipEngine's
775 device dispatches, but group count is not kernel equivalence: one Vulkan
group can contain fused or concurrently submitted nodes.

The logger is intentionally not a wall decomposition. Instrumentation reduced
its run to **15.109 tok/s / 66.18 ms/token**; four timed dependency-group sums
were 15.795, 16.433, 20.873, and 20.445 ms. Group labels and relative structure
are valid; their absolute durations are not comparable with the clean
10.581-ms wall.

### 4.3 Raw quant path

The Vulkan decode shaders consume raw IQ2_XS, IQ3_XXS, Q5_K, and Q6_K layouts
with subgroup reductions. On this RADV device the reported subgroup size is 64.
D12 successfully transferred one useful geometry idea—one physical subgroup
owns two Q5 output rows—while preserving hipEngine's exact wave32 reduction
order. It improved the two Q5 families 15–18% and canonical h32 by 4.12%.

That success also exposes the remaining issue: copying only geometry while
preserving every old BF16 and reduction boundary limits how much Vulkan's
arithmetic schedule can transfer. A larger gain may require the repository's
quality-gated lane rather than bit identity to D12.

### 4.4 Active-weight throughput proxy

Laguna's active encoded-weight proxy is **4.144 GB/token**. Dividing by wall
produces the following directional—not hardware-counter—rates:

| Route | Proxy GB/s |
| --- | ---: |
| hipEngine D12 wall | 203.0 |
| hipEngine D12 kernel sum | 251.4 |
| Vulkan, fusion disabled | 310.2 |
| Vulkan default | 391.7 |
| Vulkan, MMVQ disabled | 408.5 |

This proxy excludes cache, activations, KV, and dequant traffic, but the same
model/active-row accounting makes the scale useful: Vulkan is processing the
active raw model at roughly twice D12's wall-level effective rate.

### 4.5 What the HIP-versus-Vulkan history changes

[`HIP-vs-VULKAN-HISTORY.md`](HIP-vs-VULKAN-HISTORY.md) helps mainly as an
**exclusion matrix**. Its legacy notebook ratios must not be transferred, and
the current gfx1100 timing-contract-v2 companion in
[`HIP-vs-VULKAN.md`](HIP-vs-VULKAN.md) is the relevant dashboard:

- HIP wins every retained serialized geometry, reduction, memory/waitcnt,
  VOPD, sampler, and two-stage-reduction row on W7900.
- Vulkan's serialized packed-dot lead is only **1.052-1.133x**, not the old
  gfx1151 **3-4x**.
- Vulkan's repeatable broad advantage is tiny command-buffer replay:
  **2.437-10.122x** for the bounded serialized dispatch matrix. That is runtime
  evidence, not proof of better ACO code generation.
- Production-shaped combined Q4 selected-dual, Q6 selected-down, and dense-Q8
  controls mostly favor HIP on gfx1100.

Therefore the history does not support another generic wave64, VOPD, LDS,
reduction, waitcnt, or hand-ISA sweep. It changes P0 to a much narrower
question: what is different about Laguna's exact raw-IQ3 ownership and its
attention algorithm? The answer has to come from those production slices, not
a backend-wide compiler narrative.

### 4.6 Same-source llama.cpp HIP isolation

The old reference HIP binary predates Laguna support. A clean temporary gfx1100
HIP build was therefore made from the exact llama.cpp Vulkan source commit
`c0bc8591e8815c63cb01dd3f051a8b0df02501c9`, without modifying the dirty
read-only reference tree. The resulting `llama-bench` and `libggml-hip.so`
hashes are retained in the compact artifact.

| llama.cpp HIP control | tok/s | ms/token |
| --- | ---: | ---: |
| short, FlashAttention on | **55.621 +/- 0.210** | **17.979** |
| depth 86 / n31 | 55.324 +/- 0.778 | 18.075 |
| short, FlashAttention off | 52.200 | 19.157 |
| short, HIP graphs off | 55.376 | 18.058 |

The same-source Vulkan/HIP ratio is **1.699x**. Context depth is again
negligible. FlashAttention saves about **1.18 ms/token**, while HIP graph replay
saves only about **0.08 ms/token** in this bounded run.

The depth-matched trace records **13.729 ms of GPU kernels/token** and
**1,874 dispatches/token**. It does not reveal a ready-made HIP endpoint near
94 tok/s, but it makes the implementation split concrete:

| Family | hipEngine D12 | llama.cpp HIP | Directional read |
| --- | ---: | ---: | --- |
| selected IQ2 gate/up | 2.358 ms | 2.465 ms | hipEngine is already slightly faster |
| selected IQ3 down | **2.131 ms** | **1.422 ms** | llama.cpp saves 0.709 ms before shared quantization accounting |
| Q6 projection families | 1.546 ms | 1.618 ms | near parity |
| major Q5 families | 4.929 ms | 4.199 ms | llama.cpp leads, but family boundaries differ |
| attention context bodies | **2.671 ms** | **0.558 ms** | largest direct algorithmic gap |

These are diagnostic family comparisons, not a product-throughput A/B:
llama.cpp uses forced tokens and F16 KV, while hipEngine uses natural greedy
trajectories and BF16 `KVLiveSpans`. The useful result is causal. llama.cpp HIP
pays more launches yet has lower kernel sum, so Python and graph submission do
not explain its lead. Its IQ3 and FlashAttention sources are implementation
references; the complete engine is not a performance target by itself.

### 4.7 Raw-IQ3 ISA and ownership breakdown

The exact K1024/N3072/top-10 selected-down comparison is:

| Path | Arithmetic | Ownership | Resources / static body | Observed IQ3 down |
| --- | --- | --- | --- | ---: |
| hipEngine D12 | BF16 input, FP32 scalar FMA, BF16 each route, slot-order weighted FMA | local128, one output block, **ten routes serial** | VGPR32, LDS512 B, scratch0, 343 instructions, two barriers, no dot4 | **2.131 ms/token** |
| llama.cpp HIP | Q8_1 activation + dot4, F32 output | one wave32 per `(route, output)`, routes grid-parallel | VGPR72, LDS0, 547 instructions, eight dot4 | **1.422 ms/token** |
| llama.cpp Vulkan, MMVQ off | raw IQ3 + F32 activation/FMA | subgroup64, **four adjacent output rows/workgroup**, routes grid-parallel | VGPR60, 2 KiB allocated LDS, 1,590 shaderstats instructions, 128 FMA, no barrier | warmed logger **~1.163 ms/token** |

The Vulkan timing is from the deliberately perturbed logger and is directional
only. Its source/ISA structure is still unambiguous: preload the 1-KiB IQ3
codebook, share each activation load over four output rows, process K1024 in
two eight-value iterations per lane, and subgroup-reduce without a workgroup
barrier. The shader is much larger than hipEngine's. Its advantage is
ownership/reuse, not a shorter ACO program.

hipEngine currently repeats a four-wave reduction and barrier sequence for
each of ten serial routes inside every output block. llama.cpp HIP instead
exposes route/output parallelism and uses Q8_1 dot4. That path wins despite
more static instructions and registers. The two source-backed hypotheses are
therefore:

1. move routes into the grid and tile adjacent output rows while preserving the
   current per-route BF16 and slot-order weighting boundaries; then
2. if exact topology is insufficient, test one IQ3-only Q8_1 or Vulkan-style
   raw-row4 arithmetic sibling under the quality gate.

A same-source compiler control prevents misattribution. Clang 23 changes the
hipEngine body from **343 to 329 instructions**, waitcnt-family instructions
from **24 to 20**, delays from **25 to 19**, and NOPs from **12 to 1**, but
raises logical VGPR from **32 to 33**. On actual layer weights and ten distinct
routes it is bit-exact and **4.80% slower** by paired median than Clang 22.
Compiler upgrade and static instruction count are not the next premise; Clang
22 remains the comparison baseline.

## 5. Why Qwen3.6 is competitive while Laguna is not

The current Qwen3.6-35B-A3B Q4_K_M W7900 row demonstrates that hipEngine's
runtime architecture can be competitive:

| 4K/128 decode | tok/s | hipEngine delta |
| --- | ---: | ---: |
| hipEngine GGUF Q4_K_M | **100.522** | — |
| llama.cpp HIP | 79.768 | **+26.02%** |
| llama.cpp Vulkan | 103.066 | **-2.47%** |

That result does not transfer numerically to Laguna. The workloads differ:

| Property | Qwen3.6 Q4_K_M | Laguna UD-Q2_K_XL |
| --- | --- | --- |
| Layers | 40 | 48 |
| Hidden | 2,048 | 3,072 |
| Routed top-k | 8 | 10 |
| Main quants | Q4/Q5/Q6/Q8, T16/repacked paths | IQ2/IQ3/Q5/Q6 raw paths |
| Attention | GDN + periodic full attention | 36 SWA + 12 global |
| Extra gate | model-specific GDN/router work | per-head softplus attention gate |
| Decode submission | retained one-step graph | eager; graph measured slower |
| Model file | about 22.66 GB | about 39.68 GB |

Qwen's tuned performance comes from kernels designed for its formats and shapes,
not a model-independent scheduler switch. Laguna D0 began at 19.596 tok/s and
D12 reached 48.987 tok/s precisely by replacing generic paths with
Laguna-format-specific leaves; it has already improved **2.5x**. The remaining
2x Vulkan gap means that specialization is incomplete, not absent.

## 6. Qwen3.x optimization transfer audit

Sources reviewed:

- [`OPTIMIZE.md`](OPTIMIZE.md) — Qwen3.5/PARO decode and prefill lanes;
- [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md) — dense Qwen3.6 plan and negative results;
- [`OPTIMIZE-KERNEL-IQ2_XS.md`](OPTIMIZE-KERNEL-IQ2_XS.md) — complete Laguna IQ2 campaign;
- Qwen Q4 final artifacts, Q3 direct/profile artifacts, and relevant
  `WORKLOG.md` retained/rejected entries;
- Laguna D0–D17 artifacts and the live campaign ledger in [`LAGUNA.md`](LAGUNA.md).

### 6.1 Transfer matrix

| Qwen/Vulkan lesson | Laguna state | Verdict |
| --- | --- | --- |
| Quant-format-specific rows=1 decode leaves instead of generic prefill-shaped GEMV | D1 added exact Q4/Q5/Q6/Q8 leaves; D0 kernel sum fell 44.572 -> 23.142 ms | **Transferred; largest Laguna win** |
| Fuse selected gate/up and SiLU | IQ2 selected dual+SiLU is the default | **Present** |
| Branchless raw-IQ selector decode | IQ2 branchless decode retained, 15–26% primitive wins | **Present** |
| Right-size workgroups to useful K tasks | IQ2 local64 and IQ3 K1024 local128 retained | **Present** |
| Wave-uniform raw-IQ block address | IQ3 block-base cleanup retained; VGPR and family time fell | **Present** |
| Fuse selected down with route weighting | D3 IQ3 weighted-down retained; 45 launches removed | **Transferred** |
| Fuse same-input projection pairs without widening arithmetic | D5 Q5 shared pair, D6 Q5 query/gate, D7 Q6 attention pairs retained | **Transferred** |
| Score/split parallel attention | D4 exact token4 SWA retained; SWA fell about 50–53% | **Transferred for SWA** |
| Subgroup-sized two-output raw-Q5 decode | D12 wave32x2 retained; h32 +4.12% | **Transferred** |
| Fuse MoE aggregate tail with next norm | D9 removes 94 launches/token | **Transferred** |
| One-step HIP graph replay | Qwen uses it successfully; Laguna D8 regressed 1.15–2.25% steady state and 4.77–7.15% capture-inclusive | **Tested and rejected** |
| Persistent cooperative router | Qwen Q4 gained +1.07%, then +1.65% from self-reset; Laguna D11 removed 47 dispatches but kernel sum changed +0.046% | **Tested and rejected for Laguna shape** |
| More graph-level post-op fusion | D13–D17 tested shared-SiLU, head/KV, attention/gate, and token8 bundle. D17 reached 50.668 tok/s but failed TTFT +0.795% | **Mechanically useful, not retainable under frozen gate** |
| C/C++ packetization to remove Python/ctypes transitions | D16 exact two-launch packets were event/wall neutral | **Tested and rejected** |
| Routed/shared auxiliary streams | Q3 timeline found zero kernel overlap and regressed | **Do not transfer without new concurrency evidence** |
| Q8_1 activation + integer dot | Laguna IQ2 inclusive path regressed; Vulkan `GGML_VK_DISABLE_MMVQ=1` is +4.29% | **Rejected; do not cargo-cult** |
| Spill elimination and launch-bound hygiene | Every current top Laguna family reports scratch0; decode profile already uses unroll600/mcumode | **Present; no broad basic miss** |
| Qwen GDN recurrence/Conv optimizations | Laguna has no GDN/Conv layers | **Architecturally inapplicable** |
| Qwen compact-WMMA selected prefill and metadata no-read | Multi-token prefill tactic, not c=1 decode; Laguna AR decode touches top-10 rows directly | **Out of scope for this gap** |
| Qwen LCP-D2 parallel attention output reduction | Crossed over at 32K; Laguna's admitted target is <=4K and its near-4K cost is the context scan, not only final reduction | **Not a short-decode transfer** |
| Prefix cache, c>N scheduling, serving policy | Changes TTFT/aggregate serving, not one c=1 model-step kernel floor | **Orthogonal** |

### 6.2 What the expanded audit leaves open

The review resolves the old broad-compiler question and leaves three scoped
implementation campaigns:

1. **Raw IQ3 ownership.** The missing mechanism is now identified: D12 loops
   ten routes serially and repeats reductions/barriers, while both reference
   backends expose route/output parallelism and Vulkan reuses activations across
   four output rows. Exact route-parallel and row4 screens come before any new
   arithmetic.
2. **Quality-gated raw quant throughput.** If exact IQ3 topology cannot recover
   enough time, only then admit an IQ3-scoped Q8_1/dot4 or raw-row4 sibling.
   Q5 and IQ2 follow only after independent actual-weight wins; the rejected
   IQ2 Q8_1 path is not repeated. The later raw-Q5 row4 screen won its actual
   leaves but failed the complete quality gate. Both exact and Vulkan-style
   raw-IQ2 row4 screens then failed their actual-weight performance precondition,
   closing P1 without another model-quality run.
3. **Split/online attention.** D4's token4 schedule remains one block per query
   head. llama.cpp HIP instead runs independent tile32 partials and a stable
   combine, reaching 0.558 ms/token at comparable short depth. A Laguna version
   must retain the full `KVLiveSpans` ABI and use separate global/SWA crossover
   policies.

Submission remains downstream. D12 has 3.08 ms between kernel sum and dispatch
span, but deleting all of it still caps at 60.66 tok/s. A new scheduler becomes
useful only after the IQ3 and attention bodies fall substantially.

## 7. Root-cause attribution

The measured 9.833-ms D12-to-Vulkan diagnostic wall gap is best classified as:

### Primary: raw operator/device efficiency

Evidence:

- D12 kernel sum alone exceeds Vulkan whole wall by 5.905 ms.
- Fusion-disabled Vulkan remains 7.057 ms faster than D12 wall.
- The top five HIP device families consume 10.666 ms, approximately the whole
  Vulkan wall by themselves.
- Same-model active-weight proxy rate is 203 GB/s at D12 wall versus 310 GB/s
  with Vulkan fusion disabled.

### Secondary: graph-level fusion and dependency scheduling

Evidence:

- Vulkan fusion is worth 2.777 ms/token.
- Vulkan source combines operations at boundaries that hipEngine often keeps
  separate.
- D17 proves a subset is useful: 775 -> 679 dispatches and h32
  48.971 -> 50.668 tok/s in its matched diagnostic.

Counter-evidence against making this the primary cause:

- D13 and D11 reduced launches but did not reduce kernel sum.
- D16 host packetization was neutral.
- Even D17 remains **1.865x slower** than clean Vulkan and was not retainable.

### Tertiary: host/sampling/readback

The canonical wall exceeds short profile kernel sum by 3.928 ms, but profile
span already absorbs 3.081 ms of that. This is worth eventual cleanup, not the
headline explanation. Perfect cleanup does not reach the Vulkan floor.

## 8. Ranked future experiments

This list governs the reopened AR campaign. It does not supersede retained
Laguna DFlash/MTP work, and rejected D10–D17 code must not simply be restored
unchanged.

### P0 — Exact raw-IQ3 ownership screens

The ISA/source audit is complete. P0 implementation began on 2026-07-24 in the
frozen order below so a quality-traded result cannot hide an exact alternative.
The exact local32 wave4 route/output producer plus slot-order reducer is
BF16-bit exact and improves actual layer-1/layer-45 inclusive HIP-event time
**36.34%/32.89%** versus D12's serial weighted composite. The independently
exact row4 producer improves **18.91%/14.96%**, so wave4 advances. Its cached
trace is local32/VGPR88/LDS0/scratch0 with the intended 30,720
`(route, output)` workgroups. Full logits, all 48 hidden boundaries, all 47
routed outputs, active KV/`KVLiveSpans`, reset, and lifecycle are exact through
16 model steps. Clean short/512/1K/near-4K profiles improve the inclusive IQ3
family **24.98-26.19%**, complete kernel sum **1.00-3.21%**, span
**0.63-1.82%**, and profiled-child throughput **1.09-1.52%**. The complete
counterbalanced category gate moves h32 decode **48.780 -> 50.254 tok/s
(+3.022%)** and E2E **12.103 -> 12.183 (+0.666%)**, with every category and
horizon positive and prefill/TTFT inside guard. gfx1100 now defaults wave4;
`serial_weighted` and all unmeasured backends remain exact fallback. Evidence:
[`...p0-iq3-wave4-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p0-iq3-wave4-retained.json).

A subsequent load-free sign-bit sibling remains a separately registered
runtime-unselected diagnostic primitive. It keeps the same local32
`(route, output)` grid, four K256 accumulators, shuffle trees, 0..3 partition
addition, BF16 route store, and slot-order weighted reducer. Exhaustive
all-selector/grid tests are exact; repository Clang-22 codegen removes all 32
sign compares/cndmasks, contracts **527 -> 499 instructions** and logical SGPR
**28 -> 18**, and stays in the allocated VGPR88/LDS0/scratch0 class. On actual
layers 1/45, producer event/wall improves **8.12-8.17% / 7.33-9.15%** and
inclusive producer+reducer improves **6.42-8.55% / 6.20-7.80%**, with zero
route/reduced mismatches. Its temporary default-off runtime schedule passed
shared-weight 16-transition full state and cached **45 producer + 47 unchanged
reducer / 723-kernel** topology. Clean profiling also reduces producer and
inclusive time in both short orders by **2.43-3.05% / 2.24-2.74%**, but violates
the frozen end-to-end guards: dispatch span regresses **0.571%/1.931%**, and
order-A profiled-child throughput regresses **1.124%**. The any-failure rule
stops the remaining profile matrix and all categories; runtime schedule/CLI
selection is removed while retained wave4 remains canonical. Evidence:
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq3-signbit-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq3-signbit-runtime-correctness.json),
[`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq3-signbit-rejected.json).

The experiment definitions are:

1. **Route-parallel exact producer.** Put `(route, output)` in the grid, write
   each route's BF16 result, then apply the current routing weights in original
   slot order. This isolates removal of ten serial reductions/barriers.
2. **Exact row4 producer.** Add four adjacent output rows per ownership unit,
   sharing activation/codebook work while retaining each row's current dot tree,
   BF16 route boundary, and final slot-order weighted FMA.
3. Compare both against D12 on actual
   `blk.1.ffn_down_exps.weight` K1024/N3072 with ten distinct/cold routes. Do not
   enter a full-model run unless inclusive producer+reducer time wins.

Required admission: BF16-bit equality, scratch0, no duplicated persistent
weights, Clang 22 baseline, exact registry fallback, actual code-object resource
capture, and a cached trace showing the intended route/output grid. A
0.5-ms family saving would be useful for 50 tok/s, but cannot be extrapolated
to Vulkan parity.

### P1 — Quality-gated quant family ladder

Only after P0 is adjudicated, test separately registered arithmetic variants in
this order:

1. IQ3 Q8_1/dot4, including activation quantization and reusable workspace in
   the timed window;
2. raw-IQ3 Vulkan-style row4 F32 association;
3. raw-Q5 row4;
4. raw-IQ2 row4.

The IQ3-only integer screen is justified by llama.cpp HIP's 1.422-ms family,
not by a generic MMVQ claim. Vulkan's all-MMVQ-off control and hipEngine's
rejected IQ2 Q8_1 path prohibit making activation quantization a broad default.
Each family must win independently before any bundle is tested.

The first lane is now adjudicated and rejected. A source-matched K1024
IQ3_XXS x Q8_1 signed-dot4 producer was bit-exact to its primitive oracle and
passed a synthetic exact-sibling quality check, but mandatory activation
quantization made actual layer-1/layer-45 producer+reducer HIP-event time
**16.16%/13.63% slower** than retained wave4; synchronized wall regressed
**13.61%/11.03%**. Per the frozen inclusive-win precondition, it was removed
before runtime or the 18-prompt gate. Evidence:
[`...p1-iq3-q8-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p1-iq3-q8-rejected.json).

The second IQ3 lane is also rejected. A local32 raw-row4 leaf shared each BF16
activation across four outputs, accumulated all ten routes in FP32, and applied
routing weights before the final BF16 store. It passed the synthetic CPU/source
and exact-sibling quality gates, but route serialization overwhelmed reuse:
actual layer-1/layer-45 producer+reducer HIP-event time regressed
**146.14%/93.00%**, with synchronized wall **140.04%/83.37%** slower. It was
removed before runtime and the 18-prompt gate. Evidence:
[`...p1-iq3-row4-f32-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p1-iq3-row4-f32-rejected.json).

The raw-Q5 lane is also rejected and removed. A source-backed local64 schedule
adapted llama.cpp Vulkan's four-row ownership and nested-FMA superblock
association to BF16 activations and two native wave32 reductions. On actual
layers 0/1 it approximately halves both retained Q5 families: attention-output
event windows improve **48.12-49.99%** and query/gate improves
**48.97-50.38%**, with synchronized wall agreeing. Primitive differences are
only `3.28e-7`, but the mandatory 18-prompt/576-step model gate fails at maximum
KL **0.461353** versus the `0.05` ceiling. Isolating output only and query/gate
only also fails at **0.893206/1.35822**, so no role subset is admissible despite
**98.96-99.13%** top-1 agreement. Evidence:
[`...p1-q5-row4-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p1-q5-row4-rejected.json).

The final raw-IQ2 lane is rejected and removed before runtime. An exact tile4
sibling shared each retained pair16 activation load across four gate/up columns,
but actual layer-1/layer-45 events changed **-0.08%/-1.41%** while synchronized
wall changed **+0.89%/-1.00%**: mixed noise, not an independent win. A second
local64 sibling adapted llama.cpp Vulkan's four-row ownership, four 16-lane K256
partitions, and nested-FMA selector dots. It cut allocated VGPR **136 -> 72**
and stayed scratch-free/primitive-close, but actual events regressed
**9.46%/10.90%** and wall regressed **8.38%/9.84%**. Both bodies, wrappers,
registry keys, and tests are removed; retained exact tile2 remains the only c=1
IQ2 route. P1 is closed. Evidence:
[`...p1-iq2-row4-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p1-iq2-row4-rejected.json).

Contract and gates:

- keep the exact D12 Q5 and tile2 IQ2 siblings registered and fail closed on
  shape/backend/key misses;
- primitive dual oracle: CPU/source math plus current D12 on actual weights,
  edge scales/selectors, distinct routes, and non-finite classes;
- frozen Poolside first-token gate;
- all **18 prompts**: ten train/category rows plus eight category-heldouts with
  matched token history, maximum KL `<= 0.05`, and top-1 agreement `>= 90%`;
- deterministic free-running/category runs, with complete-ID agreement reported
  but not substituted for the declared quality gate;
- exact KV/state ownership and lifecycle, no prompt/token-conditioned policy;
- scratch0 kernel bodies, bounded reusable Q8_1 activation workspace, no
  persistent weight copy, and actual-weight inclusive event/wall wins.

### P2 — Laguna split/online attention family

This is now a source-backed plan, not a request for another head remap.
llama.cpp HIP's depth-matched trace runs:

- 36 SWA tile launches at **0.357 ms/token**;
- 12 global tile launches at **0.090 ms/token**;
- 48 split combines at **0.112 ms/token**.

The source uses tile32 online softmax partials `(m, l, o)`, local `(32,2)`
workgroups, eight partial blocks at the padded-256 depth, then a local128
combine. The observed SWA/global tiles use 5,632 B LDS and VGPR136/VGPR120;
SWA reports 32 B scratch per thread while global and combine are scratch-free.
The existing in-tree Qwen split-K producer/reducer already supplies a second
reference for deterministic FP32 partial merging, but it does not implement
Laguna's ring, head shapes, or exact arithmetic.

#### P2.1 Exact split topology first

Implement a score-tile producer and a logical-slot-order reducer before online
reassociation:

- one wave computes one 128-D dot with the current global or token4 reduction
  tree;
- independent blocks write score plus physical-slot scratch;
- the reducer performs max, denominator, and value accumulation in the current
  logical-slot order;
- the final admitted reducer may also reproduce the existing softplus gate and
  write both F32 context and BF16 gated context, but the old D14-D17 head/KV
  bodies are not restored.

This stage determines whether split ownership alone is enough. It must be byte
exact to current context/gated outputs and full-model state.

Implementation status: retained on gfx1100. Independent synthetic crossover
runs select global `>=127` and SWA `>=65`: the first three buckets are positive
and no later measured bucket regresses. At live count 128, actual layer 0/44
global event windows improve **9.08%/8.53%** and layer 1/47 SWA improves
**13.31%/13.22%**, with synchronized wall agreeing and F32 outputs bit-exact.
A 16-step shared-weight gate matches full logits, all 48 hidden boundaries, all
47 sparse routed outputs, active K/V plus every `KVLiveSpans` field, reset, and
lifecycle exactly. The two reusable scratch buffers total **1,572,864 bytes**;
all four kernels are scratch-free.

Clean short/512/1K/near-4K profiles improve total attention
**15.66%/23.28%/22.98%/22.33%**, complete kernel sum
**2.67%/12.61%/13.44%/16.11%**, span
**4.65%/11.05%/11.60%/14.63%**, and profiled-child throughput
**1.19%/12.19%/12.01%/17.58%**. The complete two-order 18-prompt gate keeps
all IDs and state exact, moves h32 decode **50.093 -> 51.436 tok/s (+2.681%)**
and E2E **12.098 -> 12.158 (+0.496%)**, and improves every train/heldout
category decode/E2E row while prefill and TTFT stay within 0.5%. gfx1100 now
defaults the measured thresholds; `use_split_attention=False`, below-threshold
calls, gfx1151, and unsupported backends retain the registered readers.
Evidence:
[`...p2-split-exact-correctness.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-split-exact-correctness.json)
and
[`...p2-split-exact-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-split-exact-retained.json).

A post-P2.2 exact refinement is now retained on gfx1100. One local256 block
owns a 16-slot SWA score tile: eight waves preserve every retained 128-D dot
and write the same score/physical scratch for the unchanged reducer. Two
independent boundary screens select live `>=257`; 257/511/512 event and wall
rows are all positive and byte exact. At actual layers 1/47 and live 257, hot
counterbalanced event windows improve **0.36%/0.44%** and synchronized wall
**0.78%/0.36%**. A 150-transition shared-weight gate, including all-layer
capture after the crossover, matches complete logits, 48 hidden boundaries,
47 routed outputs, active K/V and all `KVLiveSpans`, reset, and lifecycle
exactly. The score kernel is local256/VGPR32/LDS0/scratch0 at the intended
`72 x 17` grid; no new allocation is added. The tiled global sibling has later
regressions and is removed. Two process orders at 512/1K/near-4K improve pooled SWA attention
**0.571%/0.344%/0.208%** and total attention
**0.461%/0.272%/0.056%**; complete kernel/span/child metrics remain inside the
predeclared noise guards. The two-order 18-prompt fallback gate is exact and
non-regressive. gfx1100 therefore defaults live `>=257`; explicit tile16
disable retains P2.1 and no other backend inherits it. Evidence:
[`...p2-swa-tile16-correctness.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-swa-tile16-correctness.json)
and
[`...p2-swa-tile16-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-swa-tile16-retained.json).

#### P2.2 Quality-gated online partials

If exact split is insufficient, let each tile emit FP32 `m`, `l`, and 128-wide
unnormalized `o`; merge partials in ascending split order using stable max
rescaling. Keep F32 Q and BF16 K/V first so only the softmax association changes.
Any F16 tile arithmetic is a later, separately gated candidate.

Start with one query head per tile. Then test global query-head tile2, matching
the llama.cpp GQA6 path. SWA's GQA9 starts at tile1; query-head tile3 is allowed
only after tile1 wins and a resource trace supports the extra state. Do not
jump directly to all six/nine heads.

Implementation status: rejected and removed. The tile32 producer plus ascending
stable merge was within `3.36e-8` of the retained primitive across the boundary
matrix and cut actual context-128 global/SWA event windows by **52.56-66.56%**.
It nevertheless fails the mandatory 18-prompt model gate whether enabled for
both policies, global only, or SWA only: maximum KL is
**1.77384/1.16169/1.64542** versus the `0.05` ceiling. Top-1 remains
**98.44-99.13%**, but it cannot substitute for KL. A global threshold sweep is
non-monotonic, and thresholds `124/127` only appear exact because the candidate
does not engage on the measured failing prompt; no prompt-shaped threshold is
retained. All online kernels, wrappers, workspace, selectors, and tests are
removed. Evidence:
[`...p2-online-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p2-online-rejected.json).

The mechanical result identified the exact tiled-score follow-up above. The
selected SWA tile is 16 rather than 32 tokens, and the global sibling is
removed; this preserves P2.1 arithmetic instead of attempting another online
association. Clean context-family and complete promotion gates still decide
whether the explicit SWA path becomes a gfx1100 default.

#### P2.3 `KVLiveSpans`, memory, and crossover

Every producer consumes all five fields: `base_offsets`, `live_counts`,
`token_positions`, `evict_mask`, and `row_positions`.

- Global attention retains block-size-256 page translation and absolute causal
  visibility.
- SWA retains physical ring offsets, 511/512/513 and repeated wrap, absolute
  positions, stale-slot rejection, explicit eviction, and the 512-token window.
- One session-owned reusable scratch allocation is bounded by the largest mode:
  about **3.20 MB** for global tile32 FP32 partials, or **1.57 MB** for exact
  score/slot scratch. There is no per-token allocation, duplicate KV, or
  persistent weight copy.

The runtime does not assume a crossover. Measure SWA and global independently
at live/tile boundaries. Select the lowest live count with three consecutive
positive buckets and no later regression; keep the existing D12 attention+gate
chain below that point. Thresholds belong in backend capability metadata, never
in prompt/category/token logic.

#### P2.4 RED and promotion gates

Before a model run:

- CPU-reference and current-kernel fixtures at 0/1/31/32/33, 63/64/65,
  127/128/129, 255/256/257, 511/512/513, 1K, and 4095/4096;
- reversed/permuted physical offsets, ring reuse, all/sparse eviction, stale
  positions, GQA6/GQA9, tied/extreme scores, and denominator underflow;
- exact lane byte parity; online lane keeps the existing attention primitive
  `rtol=atol=3e-4`, then passes the same 18-prompt KL/top-1 suite as P1;
- balanced actual-query/KV inclusive producer+reduce(+gate) screens at the first
  and last global/SWA layers;
- cached `rocprofv3` symbols, grid/counts, plausible duration, VGPR/SGPR/LDS,
  and zero or explicitly justified bounded scratch;
- clean short/512/1K/near-4K family, complete kernel-sum, dispatch-span, and
  child-wall wins, followed by the unchanged category/heldout promotion gate.

At a full 512-token SWA window this algorithm is mandatory; near 4K the global
scan is a separate requirement. The short **4.79x** llama-HIP attention ratio is
a mechanism/ceiling observation, not a projected Laguna speedup.

### P3 — Quant metadata sidecar/repack (rejected and closed)

Qwen Q4 benefited from T16/repacked layouts, but Laguna's 39.68-GB model leaves
limited W7900 headroom and Vulkan is already fast on raw layouts. The concrete
source-backed screen therefore reused the existing bit-lossless Q5_K T16
replacement format, whose expanded scale/min fields add only **2.2727%** over
raw Q5_K. Replacing every model Q5 tensor would add **48,710,784 bytes** rather
than retaining a second 2.14-GB copy; replacing only the 47 Q5 attention-output
tensors would add **19,021,824 bytes**.

The existing local128 T16 tile16 leaf is BF16-bit equal to retained raw Q5 but
regresses first/last global/SWA attention-output HIP events **16.46-20.51%**
and synchronized wall **13.03-17.54%**. A RED-first exact T16 wave32x2 sibling
then reproduced the retained raw kernel's accumulator and reduction tree while
coalescing each output pair's quant and metadata bytes. It is also byte exact,
local32/LDS0/scratch0, but raises VGPR **96 -> 104** and regresses the same four
actual layers by **5.28-9.07% event** and **9.48-11.73% wall**. The mandatory
actual-weight precondition therefore fails before replacement materialization,
model quality, or category gates. Candidate source/wrapper/tests are removed;
raw Q5 wave32x2 remains the default. Evidence:
[`...p3-q5-t16-repack-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p3-q5-t16-repack-rejected.json).

No smaller scale/min sidecar is justified: retained raw wave32x2 already decodes
each output/superblock's coefficients once and broadcasts them within its wave.
A precomputed FP32 scale/min sidecar would be materially larger, while the
bit-lossless expanded-byte replacement above already loses. P3 is closed until
new ISA/counter evidence identifies different repeatedly decoded metadata.

### P4 — Device scheduler/fusion (reopened by matched completion evidence)

D8 and D16 still reject unchanged capture replay and C-side packetization;
D11/D13-D17 still prove launch reduction alone is not an acceptance premise.
The final matched audit nevertheless changes the budget: Vulkan's h32 boundary
is **15.543 ms/transition**. The clean post-P4.1 short profile has **820
dispatches/token**, **15.676 ms** of GPU kernels, **18.760 ms** median dispatch
span, and **3.213 ms** span-minus-kernel. Device work alone exceeds the Vulkan
wall by **0.132 ms**. Beating Vulkan therefore requires exact device-work
reduction even under a perfect submission owner.

P4.1 supplies the required independently winning body and is retained. Its
separately registered global/SWA gated reducers preserve the P2 score producer,
logical-slot reduction order, F32 context, `KVLiveSpans` ABI, FP32 softplus, and
RNE BF16 output. The registered unfused chain remains the below-threshold,
explicit-disable, registry-miss, and non-gfx1100 fallback. First/last actual
layers at live 128/257 are bit-exact and improve inclusive event
**3.00-10.05%** and wall **2.89-9.60%**, without a new allocation. Full logits,
48 hidden boundaries, 47 routed outputs, K/V, every span field, reset, and
lifecycle are exact. The two-order 18-prompt gate moves h32 decode
**51.497 -> 51.825 tok/s (+0.637%)**; every train/heldout category improves both
decode horizons, while E2E/prefill/TTFT remain within guards. Evidence:
[`...p4-split-gate-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-split-gate-retained.json).

A genuinely new direct-AQL owner has now been screened and rejected before
runtime integration. It prefilled 820 dependent HSA kernel-dispatch packets,
rang one doorbell, and waited only on the final completion signal. Correct
barrier and agent/system fence semantics preserve all 820 dependent increments,
but AQL is **0.560-0.758% slower** than HIP across five independent
51-repetition processes even when packet construction is excluded in AQL's
favor. It cannot remove the observed 3.213-ms window. Evidence:
[`...p4-aql-submission-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-aql-submission-rejected.json).

P4 submission ownership is closed: do not restore D8 capture, D16 host packets,
or this AQL path. The current-P4 recomposition of historical D14's exact body is
now retained over P2/P4.1: pooled short plus 512/1K/near-4K kernel sum, span,
and profiled-child rows improve at **772 dispatches/token**, and the two-order
18-prompt gate moves h32 **51.872 -> 52.391 tok/s (+1.001%)** with every
train/heldout category decode positive and all E2E/prefill/TTFT guards passing.
The registered two-launch chain remains the rows/prefill, explicit-disable,
gfx1151, and unsupported fallback. Evidence:
[`...p4-head-kv-retained.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-head-kv-retained.json).

The device-pinned reaudit is complete: current-P4 measures **52.855/52.391
tok/s** versus Vulkan **64.245/64.418 tok/s** at h16/h32. It remains
**17.73%/18.67% slower** and needs **21.55%/22.96%** more throughput, so the
objective fails at both horizons.

The first fresh post-reaudit contraction audited the retained SWA exact split
reducer itself. Its maximum pass processes four logical scores at a time using
only thread 0, but the retained body executes a block-wide barrier after every
four-score group even though no other thread consumes `shared_max` until the
first weight-publication barrier. A separate gfx1100/default-off sibling omitted
only those redundant max-scan barriers; score production, logical-slot max and
denominator order, FP32 value FMA order, softplus, BF16 rounding, workspace,
and dispatch count stayed unchanged. Synthetic live 65/257, a 16-transition
shared-weight gate, full logits, all 48 hidden/47 routed boundaries, active
K/V and every span field, reset, and lifecycle were bit exact (`KL=0`, top-1
100%). First/last actual SWA layers at live 70/128/257/512 improved inclusive
event **0.18-0.50%** and synchronized wall **0.18-0.45%** in every row.

The frozen clean gate nevertheless rejects it. Two complete process orders pool
short reducer/SWA changes of only **-0.0008%/-0.0216%**, while total kernel sum
changes **+0.446%** and median dispatch span changes **+1.173%**, beyond the
0.5% guard; profiled child is **-0.167%**. At 512/1K the reducer improves
**0.557%/0.320%**, but the already-failed short gate stops near-4K and category
work. No third favorable-order rerun or waiver is used. Candidate bodies,
exports, wrappers, registry keys, runtime selector, and tests are removed;
retained current-P4 remains canonical. Evidence:
[`correctness`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-no-max-sync-correctness.json)
and [`rejection`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-no-max-sync-rejected.json).

The next screen uses a distinct premise rather than relaxing that rejection.
Each of four logical wave leaders independently replays the retained scalar
maximum and denominator order, broadcasts each four-weight batch within its
wave, and leaves every dimension's slot-order FMA chain unchanged. Duplicating
scalar score/`expf` work removes every cross-wave barrier and all reducer LDS.
The separately registered/default-off normal-score and tile16-score siblings
are F32/BF16 bit exact at live 65/257. A shared-weight gate matches full logits,
all 48 hidden and 47 routed boundaries, active K/V plus every span field, reset,
and lifecycle through 16 decode transitions (`KL=0`, top-1 100%). First/last
actual SWA layers at live 70/128/257/512 improve inclusive events
**4.87-18.91%** and synchronized wall **4.84-18.96%** in every row. Cached
tracing records 72 candidate calls and zero retained reducer calls at local128,
VGPR24, SGPR128, **LDS0**, and scratch0. Two clean process orders improve the
reducer **4.63-5.22%**, complete SWA **4.24-4.55%**, kernel sum
**0.94-1.98%**, and span **0.61-1.69%** at every context. The complete
18-prompt two-order gate moves h32 **52.211 -> 52.514 tok/s (+0.580%)**;
every train/heldout category decode improves **0.239-0.706%**, while every
E2E/prefill/TTFT guard passes. gfx1100 now defaults the wave-local siblings;
explicit `use_swa_split_wave_local=False` and unsupported backends retain the
shared-statistics reducers. Against pinned Vulkan **64.418 tok/s**, another
**22.67%** is required. Evidence:
[`correctness`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-wave-local-correctness.json)
and [`retained`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-p4-swa-wave-local-retained.json).

The next exact IQ2 screen removes the retained local64 tile2 body's cross-wave
LDS publication and barrier without changing its arithmetic. One local32 wave
replays the original two K partitions sequentially and adds their wave totals in
the retained order. The body is BF16-bit exact on first/last actual layers 1/45,
but event time regresses **5.08-5.15%** and synchronized wall regresses
**4.50-5.27%**. It fails the independent actual-weight precondition, so its
source, wrapper, registry key, and test are removed before runtime, trace, or
category work. The retained local64 tile2 body remains canonical. Evidence:
[`...iq2-tile2-wave32-rejected.json`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-tile2-wave32-rejected.json).

A narrower wave64 control then compiles the unchanged retained local64 source in
a separate `-mwavefrontsize64` code object, so both original 32-lane K
partitions remain parallel. This is not a blanket backend wave64 switch. It is
selected only through a new c=1 route; rows>1 and the default path keep the
registered wave32 library. First/last actual layers remain BF16-bit exact and
improve events **0.73%/3.49%** plus wall **1.33%/4.06%**. The shared-weight gate
matches full logits, all 48 hidden/47 routed boundaries, active K/V and spans,
reset, and lifecycle through 16 decode transitions. Cached runtime tracing
shows 92 c=1 candidate calls at local64/VGPR96/LDS512/scratch0 while all 46 bulk
prefill calls remain wave32/VGPR136.

The frozen clean gate does not preserve that actual-layer result. Two process
orders pool short/512 IQ2 changes of **-0.037%/-0.134%**, but 1K regresses
**+0.404%**; one 1K order is **+0.96%** and the reverse is **-0.15%**. The
512 profiled child also regresses **-0.562%**, beyond its 0.5% guard. The failed
1K target and 512 child gates stop near-4K and all category work. The wave64
build/wrapper, c=1 route/library owner, CLI selector, tests, and refactor debt
are removed; retained wave32 tile2 remains canonical. Evidence:
[`correctness`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-wave64-correctness.json)
and [`rejection`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-wave64-rejected.json).

The next exact IQ2 contraction targets selector reconstruction rather than
ownership or wave geometry. The retained body stores the canonical grid in
1 KiB of packed two-bit codes, then reconstructs each magnitude with shifts,
integer multiply/add, and uint-to-float conversion. The default-off sibling
stores the same eight unsigned magnitudes per selector as one 64-bit constant;
it keeps the cheaper parity `popc`, every FMA/reduction, and both BF16/SiLU
boundaries. This adds only 3 KiB of code-object constants—no duplicated weights
or persistent sidecar. The hot leaf contracts **1,246 -> 986** disassembly lines
(-20.9%), logical VGPR **132 -> 110**, uint-to-float conversions **66 -> 10**,
and multiplies **78 -> 14**, with zero spill and unchanged LDS. A separate
sign-only LUT control is rejected because its eight extra random loads regress
the retained compact-grid events **1.28-1.69%**; parity `popc` stays.

First/last actual IQ2 layers 1/45 are BF16-bit exact and improve repository-built
events **33.73%/30.78%** and synchronized wall **33.43%/30.00%**. The
shared-weight gate matches bulk prefill, full logits/top-1, all 48 hidden and 47
routed boundaries, active K/V plus every span byte, reset, and lifecycle through
16 decode transitions. Cached tracing records 92 c=1 candidate calls at
local64/VGPR112/LDS512/scratch0 with plausible **39.24-44.96 us** durations;
all 46 bulk-prefill calls remain on retained compact-grid local64/VGPR136. The
candidate is separately registered. Two clean process orders improve the IQ2
family **20.31-21.54%**, kernel sum **1.30-3.70%**, dispatch span
**1.20-3.09%**, and profiled-child throughput **1.19-2.17%** at every context.
Both complete 18-prompt orders move h32 decode **52.650 -> 54.540 tok/s
(+3.590%)** and h32 E2E **+0.730%**; every train/heldout category decode and
E2E row improves, while prefill is **-0.072%** and TTFT **+0.042%**. gfx1100
now defaults the expanded grid; explicit `use_iq2_grid64=False` /
`--disable-iq2-grid64`, rows>1, and unsupported backends retain the compact-grid
fallback. Relative to the prior
retained 52.514 row, this is **+3.858%** or **18.335 ms/token**. Pinned Vulkan
still requires another **18.11%**, so completion remains open. Evidence:
[`correctness`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-grid64-correctness.json)
and [`retained`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-iq2-grid64-retained.json).

Post-sign-bit re-ranking reopens only D11's router *composition*, not its old
selector. Current retained short traces put split projection+selection at
**0.717 ms/token / 94 launches**. The selected sibling keeps D11's exact
BF16-hidden/F32-weight projection reduction, 256 expert blocks, last-block
atomic election, and self-resetting four-byte counter, but contracts selection:
each wave computes its stable lower-ID-tie local top-10, then wave 0 selects the
same global top-10 from the sufficient 80 candidates held three per lane in
registers. Sigmoid branches, correction-only scores, selected order, denominator
sum, normalization, scaling, and all six output arrays remain unchanged. Unlike
rejected D11, this removes the old selector's repeated block barriers/shared
work array and is independently faster than that exact composite.

The out-of-tree production probe covers all **47** actual correction biases and
router weights. Current split / old D11 / selected wave-top10 medians are
**0.80112 / 0.65265 / 0.62126 ms** per 47-layer event window and
**0.80128 / 0.65305 / 0.62157 ms** synchronized wall. All logits,
routing/selection scores, selected IDs, normalized/scaled weights, and counter
replays are byte-exact. The selected body is local256/wave32, logical
VGPR64/SGPR42, static LDS680 plus the unchanged 1,024-byte projection scratch,
and spill/private/scratch0. Register-resident merge is intentional: a
logical-VGPR26 nounrolled form regresses old D11 **29.43%**, while an LDS-reload
merge regresses it **0.98%**. The isolated **0.180-ms** saving is only 26.6% of
the current **0.675-ms** Vulkan wall gap and is not a completion projection.

Repository primitive admission now passes. RED fails only on the absent wrapper;
hidden-17/3072 random, all-tie, finite-extreme, and cross-wave-tie cases match
the registered split path byte-for-byte across all six outputs and consecutive
self-reset calls. Codegen is wave32, logical VGPR64/SGPR42, static LDS680 plus
1,024-byte dynamic projection scratch, and private/spill/scratch0. The fresh
10-warmup, 15-counterbalanced-repetition, 100-window all-47-layer event medians
are **0.82081 / 0.66186 / 0.62991 ms** for split / old D11 / wave-top10; wall is
**0.82102 / 0.66243 / 0.63034 ms**. Every actual field and counter replay is
exact. A cache-only trace names the candidate at local256/VGPR64/scratch0.

The separately committed default-off runtime owner passed admission. It owned
one four-byte scalar counter and no rows counter; 16-transition full logits/IDs,
all 48 hidden/47 routed boundaries, active K/V and every `KVLiveSpans` field,
reset/re-prefill, and 20 self-reset checks were exact. Cache-only tracing
recorded **47 composite calls/token**, zero split decode-router calls, local256,
VGPR64, scratch0, and **723 -> 676 model kernels/token**, with one construction-
time fill and no fill in either decode window.

The frozen full-model gate rejects that owner. In two counterbalanced short
orders the candidate router family regresses **14.421%/13.689%** despite its
isolated actual-weight win; complete kernel sum regresses **0.736%/1.422%** and
order-B median dispatch span regresses **1.370%**. Profiled child remains inside
the 0.5% guard and all IDs/resources/counter/lifecycle checks pass, but no guard
can waive the target-family and kernel-sum failures. 512/1K/near-4K and both
18-prompt orders are therefore skipped. Runtime selector/counter integration is
removed, the split chain is again the only route, and the exact primitive stays
diagnostic. Evidence: [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-design.json),
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-runtime-correctness.json),
and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-rejected.json).

The post-router source/ISA audit closes the remaining cosmetic Q5 and mixed-Q6
metadata rewrites plus three global-reducer metadata reshapes. The next selected
surface is instead the one-call raw-Q4_K LM head, currently about **0.385
ms/token**. Its local128 output-pack body publishes 128 coefficients through
rounded **1 KiB LDS** and executes two loop barriers for each of 12 K-blocks
plus one final cross-wave barrier. The selected exact sibling gives one local32
wave an adjacent output pair. Each lane carries the four original local128 K
partitions independently, replays every `k`/`k+128` FMA in the same order,
uses the same wave tree, and adds partition totals in original 0..3 order.

This changes **12,544 local128 output-pack blocks -> 50,176 local32 output-pair
blocks** while preserving exactly **1,605,632 threads / 50,176 physical waves**
and one dispatch. Fixed-address uniform `d`/`dmin` plus three packed scale/min
words replace cooperative coefficient publication; raw Q4 bytes, BF16 input,
F32 stores, weights, and workspace are unchanged. The BF16/F32 body compiles at
logical **VGPR70/SGPR61**, local32/wave32, LDS/private/spills0; cache-only
tracing reports allocated **VGPR72/LDS0/scratch0** at the production grid.
Static instructions/text contract **520 -> 443 / 3,096 -> 2,520 bytes**, and
dynamic K3072 barriers fall **25 -> 0**. The VGPR increase is admitted only
because actual weights win; a prior local128 wave-private form raised logical
VGPR **44 -> 84** and was rejected before timing.

The selection-only actual-weight screen covers the full **K3072 x N100,352**
head after the 122-token `mixed_ja_en_review` prompt with 50 warmups, 15
counterbalanced repetitions, and 200 launches/sample. All 100,352 F32 logits
are bit-exact. Event median improves **448.65 -> 319.59 us (-28.77%)** and
synchronized wall **448.65 -> 354.59 us (-20.97%)**. This isolated **0.094
ms/token** wall saving is only 13.93% of the current Vulkan gap and is not a
throughput/default/completion claim.

Repository primitive admission now passes. RED fails only on the absent
wrapper/key. The 29-test Q4 module covers the existing CPU-reference paths plus
exact K256/512/1024/3072 output boundaries, four packed metadata patterns, and
BF16 signed-zero/subnormal/finite-edge inputs. Every candidate F32 bit matches
the retained local128 body. Repository codegen reproduces local32/wave32,
logical/allocated VGPR **70/72**, SGPR61, LDS/private/spills/scratch0, **443
instructions / 2,520 bytes**, zero barriers, and 40 shuffles. The repeated full-
head gate improves event **449.31 -> 314.34 us (-30.04%)** and synchronized wall
**449.17 -> 352.87 us (-21.44%)**, again with all 100,352 logits exact. A
require-cached trace names the sibling at grid/workgroup **1,605,632/32**,
VGPR72/LDS0/scratch0, 265.44 us; gfx1151 aliasing is explicitly excluded.

A separate runtime unit first exposed explicit
`use_q4_lm_head_local32_fixed_meta=True`; after all promotion gates pass,
gfx1100 now advertises the local32 capability by default and
`--disable-q4-lm-head-local32-fixed-meta` is the clear local128 rollback.
gfx1151, rows>1, and registry-key miss retain the registered local128 body.
Bulk-prefill and verifier projections also remain local128; only
`_project_and_sample()` decode selects the sibling. The scalar owner adds no
allocation, and its variant-keyed library entry reuses the existing Q4 pack8
code object.

The shared-weight `mixed_ja_en_review` gate covers bulk prefill, one all-layer
captured decode, 15 further transitions, reset, and eight-token re-prefill. Full
logits/IDs, all **48 hidden / 47 routed** boundaries, active K/V and every
`KVLiveSpans` field, ownership, and lifecycle are byte-exact. A non-profiled
cache warmup precedes `rocprofv3`; each decode window contains exactly one
candidate and zero retained Q4 LM-head calls at local32/grid **1,605,632**,
VGPR72/LDS0/scratch0, with unchanged **723 model kernels/token**, finite logits,
and no compiler under profiling. The trace separately names one retained
local128 bulk-prefill projection, confirming the decode-only scope.

The frozen promotion gate now passes without reruns or waivers. Across both
orders at short/512/1K/near-4K, the LM head improves **29.07-30.79%**, complete
kernel sum improves **0.34-1.10%**, dispatch span improves **0.25-1.35%**, and
profiled-child throughput changes **-0.027% to +2.413%**. All processes retain
exact IDs, finite logits, lifecycle, admitted resources, one LM-head call, and
**723 model kernels/token**. Both complete 18-prompt/two-repetition process
orders pass: paired h16/h32 decode moves **62.310/61.675 -> 62.638/61.992 tok/s
(+0.526%/+0.512%)**, every train/heldout category improves **0.247-0.804%**,
category E2E stays within **-0.389% to +0.087%**, aggregate prefill is
**-0.208%**, and TTFT is **+0.028%**. A fresh no-argument default versus explicit
local128 rollback trajectory is byte-exact through full state and teardown.
Canonical h32 is **61.992 tok/s / 16.131 ms**, still **3.91%** short of matched
Vulkan **64.418 tok/s**. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-design.json),
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-runtime-correctness.json),
and
[`retained`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-retained.json).

The post-Q4 residual audit identifies one distinct exact transfer before any
new arithmetic. Every decode token still launches **50** standalone raw-Q6
BF16/BF16 projections through the generic local128 pack8 body: one dense down,
46 shared downs, layer 47's attention output, and its shared gate/up. The
default-selected family is context-flat at **0.592-0.612 ms/token** and uses
local128/VGPR72/rounded LDS1024/scratch0. The retained heterogeneous attention
quad already contained a Q6 local32 fixed-metadata helper; the standalone
wrapper/key remains separately admitted as a runtime-unselected diagnostic.

The repository primitive gives each local32 wave two output rows while carrying
the four original local128 partitions independently. It preserves all
`k/k+128` FMAs, wave trees, partition additions, BF16 rounding, raw weights,
and total threads/waves. Codegen is logical/allocated VGPR **75/80**, logical
SGPR18, LDS/private/spills/scratch0, **451 instructions / 2,816 bytes**, zero
barriers, and 56 shuffles. Synthetic K256/1024/3072/9216/12288 and
N2/8/1024/3072 boundaries plus all **50 actual runtime weights** are BF16-bit
exact; CPU-reference KL mean is **2.96e-5** and top-1 is **100%**. Repeated
repository endpoints at K1024/3072/9216/12288 improve event **11.42-22.32%**
and synchronized wall **11.46-20.51%**. Cached tracing names
local32/grid49152/VGPR80/LDS0/scratch0 at 11.000 us with no compiler under the
profiler.

This is not the earlier rejected Q6 attention-pair lane: that candidate retained
local128 ownership, fused F32 attention pairs, and failed short child throughput
by 0.951%. This local32 sibling is separately registered only on gfx1100 and a
temporary default-off owner selected only c=1 BF16 raw single-linears. Shared-
weight 16-transition state is exact for full logits/IDs, all **48 hidden + 47
routed boundaries**, active K/V and every `KVLiveSpans` field, reset/re-prefill,
ownership, and teardown (`KL=0`, top-1 100%). Cache-only tracing records exactly
**50 candidate calls/token**, zero candidate prefill calls, local32/VGPR80/LDS0/
scratch0, and unchanged **723 model kernels/token**.

The frozen short clean gate nevertheless rejects the owner without rerun. Q6-
family time improves **12.436%/13.261%** in orders A/B, but order A regresses
complete kernel sum **0.153%** and span **2.092%**; order B regresses profiled-
child throughput **0.642%**. Pooled rows cannot waive per-order guards, so 512/
1K/near-4K and categories are skipped. Runtime capability/session/CLI/library
selection is removed; F32 mixed attention, rows/prefill, Q4/Q5/Q8, gfx1151, and
key misses retain existing routes. The **0.192-ms** isolated ceiling remains a
rejected planning model, not a throughput claim. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-design.json),
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-runtime-correctness.json),
and
[`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-rejected.json).

The post-Q6 residual audit reuses the immutable retained post-Q4 traces because
Q6 runtime integration was fully removed and changed no default or topline.
Short decode still spends **0.324 ms/token** in 47 retained router projections
and **0.402 ms/token** in 47 correction selectors. The selector is local256/
VGPR32/LDS512/scratch0 and repeats ten block-wide winner reductions with three
block barriers per call. Reopening the rejected persistent composite unchanged
is forbidden: that path fused projection, last-block election, a self-resetting
counter, and selection, then regressed full-model router time **13.69-14.42%**.

The selected candidate changes only the stateless selector. One local32 wave
owns all 256 experts as eight strided experts per lane. Each lane computes and
stores the same stable sigmoid and correction-only scores, then each of ten
rounds chooses its best remaining local expert before the existing stable
higher-score/lower-ID wave reduction. Lane zero writes the same selected order,
gathers uncorrected probabilities, sums them in retained position order,
normalizes, and applies the routed scale. There is no projection fusion,
counter, election, workspace, launch removal, or quality trade; all 47
projection and 47 selector launches plus **723 model kernels/token** remain.

The frozen out-of-tree selection probe materializes the retained projected
logits once outside timing for all 47 actual sparse layers. Every
`routing_scores`, `selection_scores`, selected-ID, routing-weight, and scaled-
weight bit matches the registered selector. With 10 warmups, 15 counterbalanced
repetitions, and 100 complete 47-call windows per sample, event time improves
**0.397111 -> 0.295086 ms (-25.692%)** and synchronized wall improves
**0.397327 -> 0.295319 ms (-25.674%)**. Local64/local128 screens improve only
about 11%, so local32 is frozen. Repository-shaped codegen is local32/wave32,
logical/allocated VGPR **70/72**, logical SGPR18, LDS/private/spills/scratch0,
zero barriers, and a deliberately unrolled **2,128-instruction / 14,080-byte**
body. Cache-only tracing names grid/workgroup **32/32**, VGPR72/LDS0/scratch0,
with no compiler under profiling.

Applying the isolated event ratio to the immutable selector family models only
**0.103 ms/token**, or **16.98%** of the current 0.608-ms Vulkan wall gap. The
modeled **62.391 tok/s** is not a performance claim and still trails Vulkan.
Repository admission now passes. The absent wrapper/key and gfx1151 alias RED
both failed before implementation; the focused suite is **7 passes** across
random, all-tie, cross-lane/cross-item tie, finite-extreme, and signed-zero
fixtures. Every candidate field is bit-exact to registered control, while the
CPU gate is KL **5.97e-16** / top-1 **100%**. Static codegen is 2,128
instructions, logical/allocated VGPR **70/72**, SGPR **18/128**, and zero
LDS/private/spill/scratch/barriers; cache-only tracing names grid/local **32/32**.
The repository all-47-layer repeat improves **0.397646 -> 0.295123 ms event
(-25.783%)** and **0.397976 -> 0.295419 ms wall (-25.770%)**, with every field
exact. The primitive is admitted under `correction_bias_compact_wave32`.

The temporary explicit/default-off c=1 owner passes its mechanical admission
gates. A shared-weight 16-transition run matches full logits/IDs, all 48 hidden
and 47 routed boundaries, active K/V plus every live-span field,
reset/re-prefill, ownership, and teardown at KL0/top-1 100%. Peak tracked
ownership remains **40,459,057,576 bytes / 1,500 allocations**. Cache-only
tracing records exactly **47 retained projections + 47 candidate
selectors/token**, zero candidate prefill calls and zero retained decode
selectors, local32/VGPR72/LDS0/scratch0, and unchanged **723 model
kernels/token**.

The frozen short clean gate nevertheless rejects runtime selection without a
rerun. Candidate versus control selector time regresses **30.58%/27.60%** in
orders A/B, complete router time regresses **16.89%/14.08%**, kernel sum
regresses **1.787%/0.591%**, and profiled-child throughput regresses
**1.587%/1.619%**. Order-A span also regresses **4.363%**; order B is within its
span guard at **+0.306%**. IDs, finite logits, 47+47 calls, resources, 723
kernels/token, and lifecycle pass. The stop-on-first-failure rule skips
512/1K/near-4K and both complete category orders. Runtime/session/CLI selection
is removed; rows and c=1 again use `correction_bias`, while the independently
exact primitive and gfx1151 exclusion remain diagnostic. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-design.json),
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-runtime-correctness.json),
and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-rejected.json).

Post-compact re-ranking selects the still-live boundary immediately after
selected down projection. Current c=1 executes 47 feature-parallel top-10
`weighted_sum` calls and then 47 one-workgroup D9 aggregate+next-RMS calls,
about **0.531 ms/token** in the immutable post-Q4 short trace. Folding the
weighted reducer directly into D9 is explicitly rejected: although all 47
actual hidden/norm rows are exact, serializing ten expert rows inside D9's one
workgroup regresses the isolated wall **0.541870 -> 0.882913 ms (+62.94%)**.

The selected schedule restores feature ownership instead. One local32 producer
uses compile-time top-10, performs the same ten ordered BF16/F32 `fmaf`s per
feature, writes the identical RNE BF16 routed row, then preserves both Laguna
BF16 adds while writing post-MoE hidden. The already registered local256
F32-weight RMSNorm consumes that hidden row with its unchanged reduction tree.
This replaces **47 weighted-sum + 47 D9 calls** with **47 weighted+routed/hidden
and 47 RMSNorm calls**: total threads, **723 model kernels/token**, buffers, and
allocation count do not change. Direct-weighted producers, rows, prefill,
registry misses, and unsupported backends retain the current route.

The all-actual screen captures all 47 expert-down rows, scaled routing weights,
shared outputs, post-attention inputs, next-norm weights, routed outputs,
hidden outputs, and normalized outputs from the retained model. Every routed,
hidden, and norm BF16 bit matches. Under 10 warmups, 15 counterbalanced
repetitions, and 100 complete windows/sample, selected local32 improves event
**0.539595 -> 0.472270 ms (-12.477%)** and synchronized wall **0.539897 ->
0.472548 ms (-12.474%)**. Local64/local128/local256 also improve
**12.34%/12.46%/12.05%** wall; local32 is frozen. Cache-only tracing reports
producer grid/local **3072/32**, VGPR24/LDS0/scratch0 followed by the unchanged
RMS grid/local **256/256**, VGPR16/LDS0/scratch0, with no compiler under the
profiler.

Repository RED fails only on the absent wrapper/oracle/key and unexcluded
gfx1151 alias; final GREEN passes **7/7**, including a hand-checked 10x3 NumPy
fixture. Random, rounding-edge, and signed-zero
routed/hidden/norm fields are byte-exact to registered weighted-sum + D9, and
the direct NumPy oracle passes KL <= 0.05/top-1 >= 90%. The mandatory all-actual
repository transfer remains exact over all 47 layers and improves event
**0.539970 -> 0.471589 ms (-12.664%)** and synchronized wall **0.540223 ->
0.471812 ms (-12.664%)**. Repository codegen is local32/wave32, logical/
allocated VGPR **23/24**, SGPR **18/128**, LDS/private/spills/scratch0, 134
instructions, 804 bytes, and zero barriers. Cache-only tracing names two
producer + two unchanged RMS calls at the frozen grid/local sizes with no
compiler under profiling.

The isolated **0.06841 ms/token** repository saving models short kernel sum
about **12.9212 -> 12.8528 ms (-0.529%)** and remains a ceiling, not a throughput
claim. A temporary gfx1100 false capability sentinel and explicit session/CLI
opt-in admitted a separate default-off owner. It deferred only the c=1 routed reducer after
the retained wave4 expert-down producer, invokes the exact split producer, and
feeds its hidden row to the registered RMSNorm. Direct-weighted schedules,
rows/prefill, key misses, gfx1151, and default execution retain control.

The shared-weight `mixed_ja_en_review` gate compares default D9 against explicit
split ownership over bulk prefill, one all-layer captured decode, 15 further
transitions, reset, and eight-token re-prefill. Full logits/IDs, all **48 hidden
+ 47 routed boundaries**, active K/V and every `KVLiveSpans` field are byte-
exact at **KL 0 / top-1 100%**; scratch remains **25 buffers / 157,128 bytes**
and teardown returns all **40,459,057,576** peak tracked bytes. Cache-only full-
model tracing records exactly **94 producer + 94 immediately adjacent registered
RMS calls** over two transitions, zero decode weighted-sum/D9 calls, unchanged
47-row prefill weighting, and a first-transition stride of **723 model kernels +
5 runtime copies**. Candidate/RMS resources remain local **32/256**, allocated
VGPR **24/16**, LDS/scratch0; no compiler runs under profiling.

The frozen clean short gate then rejects runtime selection without rerun.
Weighted-hidden boundary time improves **19.883%/12.390%** in process orders
A/B. Order A also improves kernel sum **0.701%**, span **6.980%**, and profiled-
child throughput **0.999%**. Order B, however, regresses complete kernel sum
**0.320%** and dispatch span **6.342%**, despite child throughput improving
**1.334%**. Pooled boundary/kernel/span/child changes are
**-16.156%/-0.191%/-0.373%/+1.078%**, but pooling cannot waive the every-order
contract. Exact IDs, finite logits, lifecycle, resources, 47+47 calls, and 723
kernels/token all pass.

Stop before 512/1K/near-4K and both category orders. Remove the capability,
plan/session/dispatch owner, CLI/telemetry, and refactor debt; restore production
backend/runner/MoE/benchmark files byte-identical to primitive commit
`1a10af227`. The exact primitive/key/oracle and gfx1151 exclusion remain
available only as diagnostics. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-design.json),
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-runtime-correctness.json),
and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-rejected.json).

Post-weighted re-ranking returns to the largest surviving exact family rather
than retrying the rejected owner. The latest immutable short control trace is
**12.8685 ms/token** of model kernels, including **1.93996 ms/token** across 47
retained all-local32 Q5/Q6 projection calls. Source and ISA audit closes another
local-size or fixed-metadata rewrite. It also rejects activation LDS staging:
D12's retained gain came from removing LDS/barriers, and prior packed/expanded
staging probes show no new evidence that a barrier-based replay is viable.

The selected boundary instead shares an activation **register** across
heterogeneous arithmetic. For each of the existing 1,024 Q6 output pairs, one
local32 wave simultaneously carries one retained fixed-metadata Q5 pair and one
retained fixed-metadata Q6 pair. The Q5 and Q6 accumulator chains remain
independent: every per-chain FMA order, four wave trees, and final 0..3
partition/group additions are unchanged. Excess Q5 query/gate pairs continue
through the retained helper. Raw weights, F32 outputs, one layer dispatch,
buffers, and workspace are unchanged; layer 47 remains on the registered Q6/Q8
fixed-Q6 mixed fallback.

This is a real device-work/dataflow contraction rather than launch-count-only
fusion. Global/SWA grids fall **131,840 -> 99,072** and **181,376 -> 148,608**
threads (**-24.85%/-18.07%**) while weight bytes and FMAs are unchanged. Each
layer avoids **6,291,456** logical BF16 activation bytes by loading the 3,072-
element row once rather than once per independent Q5/Q6 owner for the paired
subset. There is no LDS, barrier, counter, persistent state, sidecar, or prompt-
conditioned branch, and model topology remains **723 kernels/token**.

The frozen out-of-tree selection probe uses actual layers **0/44/1/46**, 50
warmups, 15 counterbalanced repetitions, and 200 launches/sample. All four F32
outputs are bit-exact in every layer. Event medians improve **3.08-6.77%** and
synchronized wall medians improve **3.80-5.59%**. A 12-global/35-SWA weighted
screen models the 47-call window **1.86239 -> 1.74949 ms event (-6.06%)** and
**1.84861 -> 1.76210 ms wall (-4.68%)**. Same-Clang-22 codegen is local32/
wave32, logical VGPR/SGPR **92/70**, allocated VGPR **96**, LDS/private/spills/
scratch0, zero barriers, and **1,245 instructions / 7,208 bytes**. Cache-only
tracing names grid/local **99,072/32**, VGPR96/LDS0/scratch0 with no compiler
under profiling.

Applying only the actual event ratio to the immutable projection family models
an isolated **0.11761 ms/token** ceiling: about **0.914%** of short kernel sum
and **62.447 tok/s**, still **3.16%** below matched Vulkan. This is explicitly
not a performance claim. Freeze repository RED/GREEN on the separate gfx1100
four-axis key, gfx1151 exclusion, K/output/metadata/BF16 exactness, CPU KL <=
0.05/top-1 >=90%, the same all-positive actual-layer gate, and the codegen
ceilings above. Only then may an explicit default-off owner enter the exact
16-transition state gate, cached **47 candidate + one unchanged layer-47 /
723-kernel** trace, both clean short/512/1K/near-4K orders, and both complete
18-prompt category orders. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-design.json).

Repository RED/GREEN now admits the selected body only as a separately
registered gfx1100 diagnostic primitive. The new contract rejects null pointers,
rows other than one, invalid K/output divisibility, and total Q5 width below Q6
before library load; its exact four-axis key is explicitly excluded from
gfx1151 and absent on unsupported backends. Synthetic K256/1024/3072/9216
boundaries preserve all four F32 output buffers bit-for-bit across signed Q5
metadata, signed Q6 scales, and BF16 signed-zero/subnormal/finite edges. The
independent 10x1024 CPU-reference gate measures top-1 **100%** and numerical KL
at roundoff zero.

The repository-built all-weight transfer checks all **47 layers / 188 F32
outputs** bit-exact. Under the frozen 50-warmup, 15-counterbalanced-repetition,
200-launch protocol, layers 0/44/1/46 improve event **2.86-7.16%** and
synchronized wall **2.45-5.43%** without rerun. Production codegen exactly
matches the selected ceilings: local32/wave32, logical VGPR/SGPR **92/70**,
allocated VGPR **96**, LDS/private/spills/scratch0, zero barriers, and **1,245
instructions / 7,208 bytes**. A non-profiled require-cached warmup precedes
`rocprofv3`; the trace names two candidate grid/local **99,072/32** launches at
VGPR96/LDS0/scratch0, with exact finite outputs and no compiler under profiling.
A gfx1100 false capability sentinel and explicit session/CLI opt-in now admit a
separate default-off owner. Candidate lookup precedes retained local32 and then
local128 fixed-Q6 lookup without a quant/backend branch; default/no opt-in,
rows/prefill, key misses, layer 47's Q6/Q8 tuple, gfx1151, and unsupported
backends retain registered control. There is no new allocation, workspace,
code object, launch, or default change.

The shared-weight `mixed_ja_en_review` gate compares retained local32 against
explicit pair reuse over bulk prefill, one all-layer captured decode, 15 further
transitions, reset, and eight-token re-prefill. Full logits/IDs, all **48 hidden
+ 47 routed boundaries**, active K/V and every `KVLiveSpans` field are byte-
exact at **KL 0 / top-1 100%**; candidate borrows the control's weights and
teardown returns all **40,459,057,576** peak tracked bytes. A non-profiled
require-cached child precedes `rocprofv3`. Two transitions record exactly **94
candidate calls = 47/token**, **2 unchanged layer-47 fixed-Q6 Q6/Q8 calls =
1/token**, zero candidate prefill calls, and a first-transition stride of **723
model kernels + 5 runtime copies**. Global/SWA candidates are grid/local
**99,072/32** and **148,608/32**, VGPR96/SGPR128/LDS0/scratch0; layer 47 stays
**181,376/128**, VGPR48/LDS1024/scratch0. No compiler runs under profiling.

The frozen two-order short gate then rejects runtime selection without rerun.
The target Q5/Q6 projection improves **5.306%/5.847%** and complete kernel sum
improves **0.423%/1.442%** in orders A/B. Order A nevertheless regresses median
dispatch span **1.265%**, while order B regresses profiled-child throughput
**0.681%**, both outside the 0.5% guards. Pooled projection/kernel/span/child
change **-5.577%/-0.933%/-0.848%/-0.103%**, but pooling cannot waive an order
failure. IDs, finite logits, lifecycle, resources, **47+1** calls, and 723 model
kernels/token all pass with no compiler under profiling.

Stop before 512/1K/near-4K and both complete category orders. Remove the false
capability, resolver/session/CLI/dispatch selector and the temporary refactor
entry; production backend/runner/benchmark files are byte-identical to primitive
commit `0de4f45d8`. The exact HIP body/wrapper/four-axis key/oracle and gfx1151
exclusion remain diagnostic. No default, topline, benchmark scoreboard, or
changelog changes. Evidence:
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-correctness.json),
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-runtime-correctness.json),
and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-rejected.json).

Post-pair-reuse re-ranking excludes every now-closed runtime premise and returns
to a distinct live boundary. The immutable short controls spend about **0.402
ms/token** across **48** `add_rmsnorm` calls immediately after the Q5 attention
output. The retained local256 body computes each BF16+BF16 add once for the RMS
sum, then reloads both BF16 inputs and repeats the F32 add for the normalized
output. A register-only one-wave contraction preserves every partition and tree
but is **177.13%/177.34% slower** in event/wall because it removes useful
wave-level parallelism; close it before actual-model work.

The selected body keeps all eight waves and instead stages each first-pass
**unrounded F32 add** in dynamic LDS. It writes the same RNE BF16 residual, keeps
every per-thread square accumulation and the exact local256 reduction tree, and
then reuses the staged F32 value for the unchanged `value * inv_rms * weight`
chain. This intentionally differs from the rounded DFlash add+RMS boundary:
Laguna normalizes the unrounded sum, and both selected outputs preserve that
existing contract bit-for-bit. The residual store moves earlier only within the
same kernel; its observable completion boundary does not move.

At hidden 3072, each call avoids **12,288 bytes** of second-pass global BF16
loads, or **589,824 bytes/token**, while dynamic LDS grows **1,024 -> 13,312
bytes**. Dispatches, global outputs, weights, FMA/reduction order, buffers,
workspace, and persistent allocations are unchanged; full-model topology stays
**723 kernels/token**. The separately planned gfx1100 key is
`add_rmsnorm/gguf_f32_weight/bf16_out_staged_f32_local256`; registered
`bf16_out` remains the rows/prefill/shape/key/backend/explicit-disable fallback.

The frozen synthetic hidden-3072 screen uses 50 warmups, 15 counterbalanced
repetitions, and 1,000 launches/sample. Norm and residual outputs are BF16-bit
exact; event improves **8.566652 -> 8.322392 us (-2.851%)** and synchronized
wall **8.574057 -> 8.330484 us (-2.841%)**. More importantly, one real
`code_merge_intervals` decode captures every layer boundary, replays the control
exactly, and compares the candidate: all **48/48** boundaries and all **96
combined norm/residual fields** match both ways. Under the same warmup/repetition structure
and 100 complete 48-call windows/sample, event improves **322.784042 ->
311.604481 us (-3.463%)** and wall **323.750039 -> 312.372281 us (-3.514%)**.
Tracked ownership returns to baseline.

Same-Clang-22 codegen is local256/wave32, logical/allocated VGPR **15/16** and
SGPR **18/128**, dynamic LDS **13,312 bytes**, private/spills/scratch0, **360
instructions / 1,384 bytes**. Its nine static barrier opcodes are not nine new
synchronizations: the control's two static opcodes form a loop that executes the
same nine dynamic synchronization points. Bounded fixed/static and compact-loop
variants remain exact but improve only **2.76%** or about **1.29%**, so the
unrolled dynamic-shape form is selected. A non-profiled build precedes
`rocprofv3`; two cache-only candidate calls are exact/finite at grid/local
**256/256**, VGPR16/SGPR128/scratch0, and no compiler runs under profiling.

The actual-window wall saving is only **0.01138 ms/token**. Applied to the
immutable short wall it models **62.035 tok/s (+0.071%)**, still **3.84%** below
matched Vulkan; this is explicitly not a throughput/default claim. Freeze
repository RED/GREEN, gfx1151 exclusion, synthetic/CPU/all-48 exactness, both
repository event/wall protocols, and the codegen ceilings above. Only then may
an explicit default-off c=1 owner enter 16-transition full-state, cache-only
**48-candidate/723-model-kernel** tracing, both clean short/512/1K/near-4K
orders, and both complete 18-prompt category orders. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-design.json).

Repository RED/GREEN now admits only the separately registered gfx1100
primitive. The contract requires five non-null pointers, rows exactly one,
hidden size 256..4096 divisible by 256, and local256 before library load. The
exact four-axis key is excluded from gfx1151 and unsupported backends; the
registered `bf16_out` control is unchanged. There is no capability, runtime
plan/session/CLI/selector, allocation, workspace, launch-count, or default
change.

Focused RED fails **7/7** only on the absent wrapper/key/package export and
missing gfx1151 exclusion; the first implementation passes **7/7**. Hidden
256/1024/3072/4096 random, signed-zero, subnormal, and finite BF16-edge outputs
are bit-exact to control. An independent 10x1024 CPU gate over the unrounded F32
sum measures KL mean/max **7.33e-6/1.40e-5** and top-1 **100%**, with every
candidate output also exact to the registered sibling.

The mandatory repository transfer remains positive without a measurement
rerun. Synthetic hidden 3072 improves event **8.558846 -> 8.324056 us
(-2.743%)** and wall **8.566415 -> 8.336312 us (-2.686%)**. All **48/48** actual
norm/residual boundaries again match candidate/control and control/capture; the
complete window improves event **322.540665 -> 311.143322 us (-3.534%)** and
wall **323.797970 -> 311.820321 us (-3.699%)**, with tracked ownership restored.
The first cache-only attempt stopped before allocation/capture on the missing
changed `gguf_ops` session key; a non-profiled exact-key prebuild then allowed
the unchanged harness to run once.

Integrated codegen is local256/wave32, logical/allocated VGPR **15/16**, SGPR
**18/128**, dynamic LDS **13,312 bytes**, private/spills/scratch0, **296
instructions / 1,384 bytes**, and nine static barriers. Cache-only tracing names
two expected calls at **3.88/4.80 us**, local256/VGPR16/scratch0, with exact
finite outputs and no compiler under profiling. The adjacent GGUF/runner/
registry/gfx1151 collection passes **65/65** after excluding only the documented
unchanged stale resolver call. Retain the primitive as diagnostic and proceed
only to a separate explicit/default-off owner plus full-state/**48-call/723-
kernel** admission. Evidence:
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-correctness.json).

A separate runtime unit now admits only an explicit/default-off gfx1100 c=1
owner. `LAGUNA_ADD_RMSNORM_STAGED_F32=False`,
`use_add_rmsnorm_staged_f32=True`, and
`--enable-add-rmsnorm-staged-f32` resolve one candidate key once per session;
no-argument sessions, gfx1151, registry misses, and every rows/prefill call retain
the registered control. No library, buffer, allocation, workspace, or launch-count
change is introduced.

The shared-weight gate is byte-exact over bulk prefill, all **48 norm/residual**
and hidden boundaries, all **47 routed** boundaries, 16 decode transitions, full
logits, active K/V and every `KVLiveSpans` field, reset/re-prefill, ownership,
and lifecycle (`KL=0`, top-1 **100%**). A non-profiled cache warmup precedes
`rocprofv3`; two transitions contain exactly **96 candidate calls = 48/token**,
zero candidate prefill/control decode calls, and a **728-dispatch** stride made
of five runtime copies plus **723 model kernels/token**. Candidate resources are
local256/VGPR16/SGPR128/scratch0 with **7.44/8.00/9.40 us** min/median/max;
logits are finite and no compiler runs under profiling. The adjacent bundle
passes **121/121** after excluding only the unchanged stale gfx1151 resolver
test. This admits runtime correctness only; defaults and the **61.992 tok/s**
canonical h32 topline remain unchanged. Evidence:
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-runtime-correctness.json).

The frozen two-order short gate rejects runtime selection without rerun. The
candidate family improves **2.549%/2.501%** in orders A/B, but order A regresses
complete kernel sum **0.340%** and order B regresses median dispatch span
**0.528%**, exceeding the 0.5% guard. Pooled family/kernel/span/child changes are
**-2.524%/+0.135%/+0.216%/+0.453%**; pooling cannot waive either per-order
failure. Exact IDs, finite logits, lifecycle, 48 calls, 723 kernels/token,
resources, and no-compiler checks pass. Stop before 512/1K/near-4K and all
category work. Remove the capability, resolver/plan/session/dispatch door, CLI,
and temporary refactor entry; production backend/runner/benchmark files again
match primitive commit `d2c97cacc`. The exact primitive remains diagnostic.
Evidence:
[`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-rejected.json).

Post-staged-add re-ranking uses immutable current-default traces rather than a
favorable rerun. IQ2 remains stable across short/512/1K/near-4K at about
**1.86-1.90 ms/token** and is the next independent exact leaf. The retained
local64/grid64 body already disproves the coarse “both waves reload the same
activation” premise: threads 0..63 partition K3072's 192 pair16 tasks as
`tid`, `tid+64`, and `tid+128`, and each loaded activation register already
feeds gate/up plus two adjacent output columns. Do not retry that reuse claim.

Three fresh screens close narrower alternatives before repository work. A
16-lane cohort loading each repeated FP16 super-scale once and shuffling it to
its peers is exact but regresses first/last actual layers **5.01%/2.72% event**
and **4.99%/3.01% wall**. Replacing only offsets 8/4/2/1 with DPP removes
16 of 20 `ds_bpermute`s but changes endpoint events **+0.72%/+0.31%**; mixed
wall wins do not admit it. `permlanex16` plus DPP removes all 20 permutations,
but leaving the generic 1-8-wave tail still gives layer-1 event **+0.16%**
despite wall **-1.53%**. Q5 one-wave/four-output reuse and SWA tile4 query reuse
also fail their actual-weight screens and remain out of tree.

The selected design strengthens that last form without changing ownership. A
candidate-specific fixed-local64 helper uses `permlanex16` for the offset-16
exchange and DPP `row_shl` 8/4/2/1 for each of the four accumulators. Lane 0
therefore observes the identical `+16,+8,+4,+2,+1` FP32 tree. Each wave leader
publishes one four-float tuple, the unchanged barrier remains, and thread 0
starts at `+0.0` and adds wave 0 then wave 1 exactly as the retained dynamic loop
does. Selector/sign/magnitude reconstruction, scale multiplies, every pair16
FMA, BF16 gate/up projection, SiLU, and final RNE BF16 store are unchanged.
The repository implementation must add a distinct helper/kernel/export and must
not replace the generic reducer used by retained tile2/tile4 bodies.

Same-Clang-22 codegen keeps wave32/local64, logical VGPR/SGPR **110/31**, zero
private memory/spills/scratch, and one barrier. The hot symbol contracts
**990 -> 864 disassembly lines**, **5,548 -> 4,888 bytes**, `ds_bpermute`
**20 -> 0**, and waits **49 -> 30**; fixed LDS metadata contracts **128 -> 32
bytes**. Its 50-warmup, 15-counterbalanced, 200-launch actual-weight screen is
BF16-bit exact (`KL=0`, top-1 100%). Layers 1/45 improve event **0.959%/1.215%**
and synchronized wall **1.050%/1.263%** without rerun. A non-profiled preflight
then cache-only `rocprofv3` executes two production-shape calls at grid/local
**32768x10/64**, allocated VGPR112/SGPR128/LDS512/scratch0, with finite exact
output and no compiler under profiling. ELF resources remain logical VGPR110,
SGPR31, and fixed LDS32; the trace values are allocation granularity.

Applying only the smaller endpoint event ratio to the latest **1.876674
ms/token** IQ2 family models a conservative **0.018001 ms/token** ceiling:
h32 **61.992 -> 62.061 tok/s (+0.112%)**, still **3.80%** behind matched Vulkan.
This is not a default or throughput claim. Freeze repository RED/GREEN with a
strict local64 wrapper, gfx1151 exclusion, synthetic/CPU/all-46-layer exactness,
the same all-positive endpoint transfer, and the codegen/trace ceilings above.
Only after a primitive commit may a separate explicit/default-off owner enter
16-transition full-state, cache-only **46-candidate/723-model-kernel** tracing,
two clean short/512/1K/near-4K process orders, and both complete 18-prompt
category orders. Stop on the first frozen guard failure; no third order or
pooled waiver. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-design.json).

Repository RED/GREEN now admits only the candidate-specific gfx1100 primitive.
The wrapper requires five non-null pointers, `x_rows == 1`, positive rows/
experts/output, K divisible by 256, and exactly local64 before library load. Its
separate four-axis key is package-exported and explicitly excluded from gfx1151,
CUDA, and CPU aliases. The generic quad reducer, retained grid64 kernel/symbol,
rows/prefill ownership, runtime capability, allocations, workspace, launches,
and default remain unchanged.

Focused RED fails **7/7** only on the absent wrapper/key/source/exclusion; first-
attempt GREEN passes **7/7**. K256/K1024/K3072, one/ten selected routes,
odd/even output tails, invalid experts, scale/sign/selector patterns, and BF16
signed-zero/subnormal/finite edges are bit-exact to retained grid64. The
independent 10x19x1024 IQ2 CPU gate measures KL mean/max
**6.17e-6/1.99e-5** and top-1 **100%**, while candidate and retained output
bytes are identical.

The mandatory repository actual-weight transfer runs once under the frozen
50-warmup, 15-counterbalanced, 200-launch protocol. All **46/46** IQ2 layer
pairs are BF16-bit exact. Layers 1/45 improve event **1.266%/1.641%** and
synchronized wall **1.155%/1.452%**; both timers remain positive at both
endpoints without rerun. Integrated codegen exactly meets the selected ceilings:
wave32/local64, logical VGPR/SGPR **110/31**, fixed LDS **32 bytes**, private/
spills/scratch0, **858 instructions / 4,888 bytes**, four `permlanex16`, 16 DPP
transports, zero `ds_bpermute`, 30 waits, and one barrier.

A non-profiled exact-key preflight precedes cache-only `rocprofv3`. Two calls
name the distinct `grid64_local64_reduce_kernel` at grid/local **32768x10/64**,
allocated VGPR112/SGPR128/LDS512/scratch0, with finite exact output and no
compiler under profiling; the second duration is **31.600 us**. The adjacent
bundle passes **62/62** after excluding only the documented unchanged stale
resolver test, and the CPU deterministic bundle passes **25/25**. Retain the
primitive as runtime-unselected. Only a separate explicit/default-off c=1 unit
may proceed to exact 16-transition state and cache-only **46-candidate/723-model-
kernel** admission before clean measurement. Evidence:
[`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-correctness.json).

Runtime RED freezes `LAGUNA_IQ2_LOCAL64_REDUCTION=False`, a gfx1100-only
resolver, the existing grid64 c=1 plan seam, strict retained-grid64/key-miss/
explicit-disable/rows/unsupported fallbacks, and
`laguna_target_ar_bench.py --enable-iq2-local64-reduction`. All **3/3** tests
fail only on absent capability/plan/CLI plumbing; focused GREEN passes **3/3**.
The owner adds no library, allocation, workspace, dispatch, or arithmetic. A
candidate requires both the existing grid64 c=1 owner and the distinct
registered key; otherwise resolution fails closed to retained grid64 or the
ordinary compact-grid rows/backend route.

Shared-weight `mixed_ja_en_review` (122 prompt tokens) is byte-exact over bulk
prefill, all **48 hidden** and **47 routed** boundaries, 16 decode transitions,
full logits/IDs (`KL=0`, top-1 **100%**), active K/V and every
`base_offsets/live_counts/token_positions/evict_mask` field, reset/eight-token
re-prefill, ownership, and lifecycle. Peak tracked allocation is unchanged at
**40,459,057,576 bytes / 1,500 allocations** and returns to zero.

A non-profiled require-cached preflight precedes full-model `rocprofv3`. Two
transitions record exactly **92 candidate calls = 46/token**, zero retained
`grid64_kernel` decode calls, and a 728-dispatch stride comprising five runtime
copies plus exactly **723 model kernels/token**. Every candidate is grid/local
**32768x10/64**, allocated VGPR112/SGPR128/LDS512/scratch0; durations are
**39.680/40.800/45.281 us min/median/max**. IDs are `[605, 2825, 268]`, logits
are finite, teardown passes, and no compiler runs under profiling. The adjacent
runtime/primitive/runner/registry/backend bundle passes **141/141** after
excluding only the documented unchanged stale gfx1151 resolver call. This
admits runtime correctness only; both clean process orders at short/512/1K/
near-4K remain mandatory before either complete 18-prompt category order.
Evidence:
[`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-runtime-correctness.json).

The frozen two-order short gate rejects runtime selection on its first run. The
IQ2 family regresses **0.366%/2.742%** in orders A/B. Order A improves kernel
sum/span **0.429%/2.686%** but regresses profiled-child throughput **0.634%**,
outside the -0.5% guard. Order B improves child throughput **0.013%** but
regresses kernel sum **1.180%** and median span **9.707%**. Pooled family/kernel/
span/child changes are **+1.547%/+0.372%/+6.833%/-0.362%**; pooling cannot waive
any per-order failure. Exact IDs, finite logits, lifecycle, 46 calls, 723 model
kernels/token, local64/VGPR112/LDS512/scratch0 resources, and no-compiler checks
pass.

Stop before 512/1K/near-4K and both complete 18-prompt category orders; no third
order or favorable rerun is allowed. Remove the capability, plan/session route,
benchmark CLI, and temporary refactor entry. Production backend/MoE/runner/
benchmark files again match primitive commit `2c1946c47`; retain only the exact
diagnostic primitive. Evidence:
[`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-rejected.json).

Post-IQ2 re-ranking uses the immutable current-default short controls. Across
28 stable transitions they spend **1.933541 ms/token** in 47 all-local32 mixed
Q5/Q6 projections, **1.579843 ms/token** in 47 Q5 attention-output calls, and
**0.496336 ms/token** in 46 shared-Q5 pairs, out of **12.896733 ms/token** of
model kernels. Local-size, output-ownership, T16/repack, LDS staging, Q5/Q6
activation reuse, and reassociation premises remain closed. A fresh exact
reduction-only discriminator also fails: replacing each Q5 leaf's 40
`ds_bpermute`s with eight `permlanex16` plus 32 DPP transports contracts the
BF16 symbol **499 -> 464 instructions**, but every query/gate endpoint regresses
**1.166-2.958% event / 0.956-2.120% wall** and SWA output has failed rows. It
remains out of tree.

The selected candidate changes integer reconstruction instead. Clang already
packs the ten unique per-pair `qh`/`qs` bytes into five VGPR destinations. The
new SWAR form keeps each output pair's bytes in low/high lanes of one integer,
performs each low-nibble operation once under `0x0F0F` and each high-bit
operation once under `0x0101`, then extracts the same two uint8 Q5 values. All
`d/dmin`, scale/min, `scale*q-min`, activation, FMA, four reduction trees,
0..3 partition additions, and BF16/F32 stores are unchanged. Q6 branches are
unchanged. There is no sidecar, LDS, barrier, allocation, workspace, launch,
or ownership change; full-model topology remains **723 kernels/token**.

The frozen 50-warmup/15-counterbalanced/200-launch actual-weight screens pass
without rerun. Attention-output layers 0/44/1/46 improve **7.31-9.27% event /
7.62-8.87% wall**; standalone query/gate improves **8.99-12.27% /
8.06-10.60%**. More importantly, the current production mixed Q5/Q6 symbol
improves all layers 0/44/1/46 **6.45-8.58% / 5.83-8.02%** with all four F32
outputs exact, and shared-Q5 layers 1/46 improve **5.49-5.79% / 5.49%** with
both BF16 outputs exact. Every compared field is bit-exact (`KL=0`, top-1
100% where applicable).

Same-Clang-22 codegen confirms a real operation contraction at unchanged load,
reduction, LDS, and barrier counts. The BF16 singleton changes **499 -> 470
instructions**, **2,936 -> 2,688 bytes**, logical VGPR **73 -> 71**, right
shifts **15 -> 8**, masks **26 -> 18**, and ORs **16 -> 0**. The BF16/F32 pair
symbols contract **508 -> 478 / 485 -> 461 instructions** and **2,976 -> 2,724
/ 2,892 -> 2,640 bytes**; the production mixed symbol contracts **947 -> 923
instructions** and **5,640 -> 5,388 bytes** at unchanged logical VGPR75. All
remain wave32/local32, LDS/private/spills/scratch0, and barrier-free. After a
non-profiled preflight, cache-only `rocprofv3` names direct output/F32-pair/
BF16-pair at allocated VGPR72 and the SWA mixed symbol at VGPR80, SGPR128,
LDS0/scratch0, with finite outputs on the W7900 and no compiler under profiling.

Applying only the smallest endpoint event improvement separately to the
immutable mixed/output/shared families models a conservative **0.267603
ms/token** ceiling: kernel sum **12.896733 -> 12.629130 ms/token** and h32
**61.992 -> 63.037 tok/s (+1.687%)**, still **2.19%** below matched Vulkan.
This is not a full-model/default claim. The frozen design is committed before
repository work. Evidence:
[`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q5-swar-pair-design.json).

Repository primitive admission adds only a candidate-specific SWAR helper,
direct BF16/F32 and unequal-pair BF16/F32 symbols, a mixed Q5/Q6 F32 sibling,
strict wrappers, three role-scoped four-axis keys, package exports, and gfx1151
exclusions. The retained helper, every retained key, Q6 arithmetic, allocation,
workspace, launch topology, capability/session/CLI state, and defaults remain
unchanged. Focused RED fails **19/19** solely on absent candidate pieces; GREEN
passes **19/19**. Synthetic K256/512/1024/3072/6144/9216 boundaries cover Q5
`qh`/`qs`/scale/min extremes and BF16 signed-zero/subnormal/finite edges for all
five symbol roles. The independent 10x1024 CPU gate is candidate/control
BF16-bit exact with KL mean/max **8.93e-7/8.93e-6** and top-1 **100%**.

The mandatory repository transfer runs once. All **47/47** attention-output
weights, **47/47 layers / 188/188 outputs** in the mixed Q5/Q6 quad, and
**46/46 layers / 92/92 outputs** in shared Q5 are byte-exact. The repeated
first/last gate remains all-positive: attention output improves **7.92-10.11%
event / 7.63-9.52% wall**, mixed improves **5.91-7.52% / 5.19-7.01%**, and
shared Q5 improves **5.70-5.99% / 5.75-6.01%**, with no rerun. Integrated
Clang-22 codegen exactly reproduces the frozen ceilings: BF16 direct **470
instructions / 2,688 B / VGPR71**, BF16 pair **478 / 2,724 / VGPR71**, F32
pair **461 / 2,640 / VGPR71**, and mixed **923 / 5,388 / VGPR75**. All remain
local32/wave32, LDS/private/spills/scratch0, and barrier-free with unchanged
load/reduction-permute counts.

A non-profiled require-cached preflight precedes `rocprofv3`. All five
instantiated roles execute twice: direct/pair symbols allocate VGPR72 and mixed
allocates VGPR80, all at local32, SGPR128, LDS0, scratch0, finite, with no
compiler under profiling. The adjacent filtered bundle passes **101/101** after
excluding only two documented collection-order-sensitive old gfx1151 absence
assertions and the unchanged stale resolver call; CPU registry/build passes
**25/25**. Retain the exact primitive. A temporary gfx1100 capability exposes one
explicit/default-off all-or-none owner: it selects the direct-output,
shared-pair, and mixed-projection keys only when all three keys and every
retained prerequisite are present. Default/no opt-in, any prerequisite disable,
key miss, rows/prefill, layer 47, gfx1151, and unsupported backends retain the
registered controls. Shared-weight bulk prefill plus 16 decode transitions,
reset, and re-prefill preserve full logits/IDs, all **48 hidden + 47 routed**
boundaries, active K/V and every `KVLiveSpans` field, positions, ownership,
scratch size, and lifecycle exactly (`KL=0`, top-1 100%).

A non-profiled require-cached child precedes full-model `rocprofv3`. Each token
records exactly **47 mixed + 47 attention-output + 46 shared = 140 candidate
calls**, zero candidate prefill calls, one unchanged layer-47 mixed call, five
runtime copies, and **723 model kernels**. Mixed runs local32/VGPR80 and direct/
shared run local32/VGPR72; all use SGPR128, LDS0, scratch0. The layer-47 mixed
fallback stays local128/VGPR48/LDS1024/scratch0, IDs are finite/exact, teardown
passes, and no compiler runs under profiling.

The frozen clean gate stops at short context without rerun. Combined Q5 time
regresses **0.457%/0.077%** and complete kernel sum regresses **0.280%/0.075%**
in orders A/B. The isolated attention-output benefit survives
(**-1.952%/-2.046%**), but mixed projection reverses **+2.278%/+1.694%** and
shared gate/up reverses **+0.902%/+0.439%**. Order A additionally fails span at
**+0.883%** and child throughput at **-0.723%**; order B fails child throughput
at **-0.592%**. Exact IDs, finite logits, resources, 140/723 topology,
layer-47 fallback, lifecycle, and no-compiler checks pass, but no pooled metric
can waive the per-order failures. Stop before 512/1K/near-4K and both complete
category orders. Remove capability/resolver/session/CLI ownership while keeping
the exact primitive; production backend/runner/benchmark files again match
primitive commit `f75fe94b0`. No default, scoreboard, changelog, or canonical
topline changes. Evidence: [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q5-swar-pair-correctness.json),
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-pair-runtime-correctness.json),
and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-pair-rejected.json).

### Post-sign-bit selection: exact IQ4_XS weighted composite

Re-ranking both retained short traces gives **12.736967 ms/token across 678
model kernels**. The closed mixed-Q5/Q6, IQ2, Q5-output, SWA, IQ3-sign, pair-
reuse, staged-LDS, output-tiling, and K3072-specialization owners are not
reopened. The only residual selected-down split is layers 46/47: two
`gguf_iq4_xs_selected_gemv_kernel` calls cost **0.128434 ms/token**, followed by
two `weighted_sum_out_kernel` calls at **0.006243 ms/token**. The existing
four-axis IQ4 weighted composite was introduced for Qwen top-8 and is not
currently resolved by Laguna, whose plan deliberately exposes weighted owners
only for IQ3.

Select that existing body for a new Laguna top-10/K1024 certification rather
than adding another kernel. One local256 block per output visits ten routes in
slot order. Each route executes the same IQ4 subblock dot and block reduction,
rounds the projection to BF16 exactly as the selected-single fallback does,
then applies the F32 routing weight before final BF16 rounding. No raw weight,
workspace, arithmetic, prompt, or retained IQ3-wave10 behavior changes. If
admitted and selected only for c=1 layers 46/47, topology becomes **678 -> 676
model kernels/token**; rows/prefill, key miss, unsupported shape/backend, and
all IQ3 work remain on registered controls.

The frozen W7900 production-shape discriminator uses both actual
`blk.{46,47}.ffn_down_exps.weight` tensors at E256/top-10/K1024/N3072, 50
warmups, 15 counterbalanced repetitions, and 300 complete windows/sample.
Candidate output is BF16-bit exact on both layers. Layer 46 event/wall moves
**119.121 -> 79.883 us (-32.94%) / 119.248 -> 79.828 us (-33.06%)**; layer 47
moves **99.450 -> 70.820 us (-28.79%) / 99.690 -> 70.894 us (-28.89%)**.
Tracked ownership returns to zero.

Current Clang-22 codegen is local256/wave32, **492 instructions / 2,580 bytes**,
logical/allocated VGPR **78/80**, logical/allocated SGPR **44/128**,
fixed/allocated LDS **32/512 B**, private/spills/scratch0, five
`ds_bpermute_b32`, and two static barriers. A non-profiled require-cached
preflight precedes `rocprofv3`; the trace names two candidate calls at
grid/workgroup **786,432/256**, VGPR80/LDS512/scratch0, finite output, with no
compiler under profiling. Applying only the smallest endpoint ratio to the
immutable **0.134677-ms/token** inclusive family models a **0.038771-ms/token**
ceiling and **63.426 tok/s**, still **1.56%** below Vulkan. It is planning
evidence, not a full-model/default claim.

Dedicated Laguna certification reuses the predating body rather than adding a
duplicate kernel. Its RED checkpoint passes six tests and fails only because
gfx1151 auto-aliases the unvalidated key; adding that one exclusion produces a
**7/7** GREEN with no gfx1100 device, wrapper, package, or key change. Signed-
zero/subnormal/finite BF16 inputs, signed/zero routing, repeated/invalid experts,
registered selected-single plus registered-reducer byte identity, and an
independent CPU gate all pass. The repeated production gate remains exact and
moves layer 46 event/wall **119.502 -> 78.769 us (-34.09%) / 119.678 -> 78.730
us (-34.22%)** and layer 47 **99.492 -> 71.049 us (-28.59%) / 99.726 -> 71.033
us (-28.77%)**. Codegen exactly reproduces every frozen ceiling, and cache-only
tracing records two expected local256/VGPR80/LDS512/scratch0 calls with no
compiler. Evidence: [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-design.json)
and [`certification`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-correctness.json).

A separate false/default-off gfx1100 owner included IQ4 only for the exact
E256/top-10/K1024/N3072 key; default, rows/prefill, shape/key miss, gfx1151, and
unsupported backends retained the registered split chain. Shared-weight
`mixed_ja_en_review` matched bulk prefill, full logits/IDs, all **48 hidden + 47
routed** boundaries, 16 transitions, active K/V and every `KVLiveSpans` field,
reset/re-prefill, ownership, and lifecycle byte-for-byte. A non-profiled
require-cached child preceded full-model `rocprofv3`; two transitions recorded
exactly **2 candidate calls/token**, **45 unchanged IQ3 wave10 calls/token**,
zero IQ4 selected-single or reducer decode calls, and **676 model kernels/token**
with certified resources, exact IDs, finite logits, clean teardown, and no
compiler. Evidence: [`runtime correctness`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-runtime-correctness.json).

The frozen clean contract at runtime commit `86891e008` requires each short/512/
1K/3968 process order to improve inclusive IQ4 and complete kernel sum, keep
span below +0.5%, keep child throughput above -0.5%, and preserve exact
control/candidate **678/676** topology, resources, IDs, lifecycle, and no-
compiler behavior. Both short orders fail without rerun: inclusive IQ4 moves
**+25.515%/+25.191%**, complete kernel sum **+0.202%/+0.303%**, and span
**+0.681%/+2.198%**. Profiled-child throughput improves **0.336%/1.893%** but
cannot waive three failed guards. Exactly four short roots exist; 512/1K/3968
and all category work are skipped. Remove capability/session/plan/CLI ownership
and restore the four production files byte-for-byte to certification commit
`30179e697`; retain the primitive, key, gfx1151 exclusion, and all correctness
evidence. Evidence: [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-rejected.json).

### Post-IQ4 selection: exact single-page scalar global attention

Re-ranking both immutable retained wave10 traces leaves **12.736967 ms/token
across 678 model kernels**. Closed Q5/IQ2/IQ3/IQ4/SWA/add-norm/router/Q6 owners
remain excluded. The largest independent short surface is the 12-call scalar
global-attention family at **0.523579 ms/token**. Retained dispatch selects that
body only below `global_split_min_live=127`; block size is 256, so every visible
logical slot lies in the first physical page.

Select one separately registered scalar sibling. It keeps one local256
workgroup per query head, eight wave32 score partitions, dynamic score/query
LDS, causal/eviction checks, BF16 K/V loads, every score/value FMA, shuffle/max/
denominator association, five barriers, and F32 stores source-identical. Only
both calls to the generic page translator become
`base_offsets[0] * block_size + token`, with a device guard rejecting
`live_count > block_size`. The complete `KVLiveSpans` ABI remains: page-zero
base, live count, token positions, eviction mask, and row position are all
consumed. Runtime eligibility is deliberately narrower than primitive validity:
full-attention c=1 scalar calls at live **<=126** only; live >=127 keeps retained
split-exact ownership.

The frozen W7900 actual-model discriminator uses `code_merge_intervals`, layers
0/44, live counts 70/126, 50 warmups/mode, 15 counterbalanced repetitions, and
500 launches/sample for both HIP-event and synchronized-wall timers. All four
F32 outputs are byte-exact. At live 70, layer 0 event/wall moves **24.0650 ->
20.3026 us (-15.63%) / 24.1271 -> 20.3836 us (-15.52%)** and layer 44 moves
**24.2026 -> 20.3902 us (-15.75%) / 24.2595 -> 20.4710 us (-15.62%)**. At live
126, layer 0 moves **39.0673 -> 32.1335 us (-17.75%) / 39.1333 -> 32.2628 us
(-17.56%)** and layer 44 **39.5472 -> 32.6688 us (-17.39%) / 39.6298 ->
32.7978 us (-17.24%)**. Tracked ownership peaks at 40,070,027,316 bytes / 1,159
allocations and returns to zero. Two disclosed pre-interception harness failures
(config allocation field, then current gate ABI) produced zero candidate or
timing rows; the ABI-complete process is the only measured screen and was not
favorably rerun.

Integrated Clang-22 codegen contracts baseline-to-candidate **1,705 -> 1,008
instructions (-40.88%)**, **7,896 -> 4,864 text bytes**, logical VGPR **35 ->
32**, logical SGPR **57 -> 49**, and global 32-bit loads **2 -> 1**. Both retain
wave32, fixed LDS0, private/spills0, five barriers, and the same 16,928-byte
capacity-4096 dynamic-LDS request. Applying only the smallest observed ratio to
the immutable short family models **0.081237 ms/token**, h32 **63.270 -> 63.597
tok/s (+0.517%)**, still **1.29%** below Vulkan. That is a planning ceiling, not
a full-model/default claim.

Repository primitive admission is complete. The strict wrapper, exact gfx1100
four-axis key, gfx1151/cuda/CPU exclusion, one-page no-write guard,
synthetic/CPU edge fixtures, **24/24** actual global-layer equality, repeated
endpoint timing, integrated codegen ceilings, and cache-only expected-symbol
trace all pass. No capability, selector, allocation, threshold, or default was
added. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-correctness.json).

Default-off runtime admission passed shared-weight full state through 16
transitions, the live-127 split fallback, **12-candidate/zero-scalar-global/
678-kernel** tracing, unchanged SWA owners, exact IDs/resources/lifecycle, and
no-compiler gates. Evidence:
[`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-runtime-correctness.json).

The frozen clean contract at runtime commit `2e993f822` rejects ownership at the
first context. Global-attention family time improves **10.974%/12.122%** in
orders A/B and child throughput improves **0.871%/1.960%**. Order A nevertheless
regresses complete kernel sum **0.0949%** and dispatch span **1.3203%**, beyond
the immutable strict-win/+0.5% guards; favorable order B and pooled kernel/span
**-0.0834%/+0.4783%** cannot waive it. Exactly four short roots exist, with zero
512/1K/3968/category outputs. Capability/session/CLI/route ownership is removed;
only the exact registered primitive remains. Evidence:
[`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-rejected.json).
Do not retry unchanged one-page ownership.

### Post-one-page primitive: exact gated single-page global attention

The independent residual composition is the one-page body plus the standalone
global softplus gate, not another address-only owner. Current two-order traces
measure scalar+gate at **0.555764/0.566353 ms/token** and one-page+separate-gate
at **0.500807/0.502853 ms/token**. Historical D15 independently proved that
appending `context_value * softplus(gate[q_head])` and the same RNE BF16 store
inside the generic global workgroup preserves both F32 context and BF16 gated
context; its generic global inclusive window improved **9.884%** and all clean
mechanical contexts passed, although the removed all-global/SWA/head bundle
failed its category TTFT gate.

The exact key is now separately registered:
`hip_gfx1100/laguna_attention_decode+attention_gate/bf16/
global_single_page_softplus_bf16_spans`. Its body is mechanically generated from
the admitted page-zero primitive and adds only non-null gate/gated-output
arguments, an unchanged F32 context store, and D15's exact softplus/multiply/
BF16 epilogue. Both page-zero K/V translations, the live<=256 guard, all
`KVLiveSpans` fields, local256/eight waves, five barriers, every attention
operation, and dynamic LDS16,928 remain unchanged. The primitive fallback is
one-page attention plus the registered standalone gate; scalar+gate and
live>=127 split-gated controls remain callable.

Focused synthetic/CPU admission and all **12 global layers x live70/126** preserve
both F32 context and BF16 gated bytes. The frozen layer0/44 inclusive event/wall
screen improves every row by **9.53-14.57%**. Integrated Clang-22 codegen is
**1,181 instructions / 5,756 B**, logical VGPR32/SGPR54, private/spills0, and
five barriers. Cache-only tracing names the distinct composite at local256,
allocated VGPR32, dynamic LDS16,928, and scratch0 with no compiler. This is
isolated primitive evidence, not runtime or full-model throughput evidence.

Future runtime eligibility is narrower than primitive validity: full-attention
c=1 at live<=126 after split selection is false. It replaces **12 attention +
12 gate calls with 12 composites**, contracting short topology **678 -> 666
model kernels/token** without allocation/workspace/weight changes. Assuming the
epilogue adds zero duration to the measured one-page body, the smaller order
saves **0.095009 ms/token** and models h32 **63.270 -> 63.653 tok/s (+0.605%)**,
still **1.20%** below Vulkan. This is a planning ceiling, not a measured body or
throughput claim.

Primitive admission passes mechanical source identity, synthetic/CPU and
all-actual F32/BF16 equality, all-positive endpoint timing, resource/codegen,
and distinct cache-only trace gates. Runtime ownership remains absent. Only
after the primitive commit may a separate false/default-off owner enter
shared-weight 16-transition/live127 fallback and **12-candidate/zero-gate/
666-kernel** tracing. Both clean process orders at short/512/1K/3968 and both
complete 18-prompt orders retain the same no-rerun, no-pooled-waiver, category,
E2E, prefill, and TTFT gates. Evidence:
[`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json),
[`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-correctness.json).

## 9. Do not chase without new evidence

- **Unchanged D8 graph replay:** measured regression and removed.
- **Unchanged D10/D13/D15/D17 boundaries:** all have complete rejection
  artifacts; positive diagnostic rows do not waive category/TTFT failures.
  Historical D14 remains removed; only the separately gated current-P4/18-prompt
  recomposition above is retained over the modern P2/P4.1 path.
- **More C-side packets:** D16 proved the visible gaps are queue spacing, not
  ctypes transition cost.
- **Q8_1/MMVQ/dp4a as a default premise:** Vulkan's whole-path control and
  hipEngine's IQ2 experiment reject it broadly. An IQ3-only inclusive screen is
  allowed only because the llama.cpp HIP IQ3 family supplies new direct evidence.
- **Wave64 as a blanket switch:** hipEngine's default is intentionally wave32;
  prior controlled wave64 probes were generally slower and Vulkan subgroup size
  alone is not a portable reason.
- **Broad LDS staging by default:** the successful D12 change removed LDS/
  barriers, and prior Qwen/Laguna evidence repeatedly shows occupancy and
  barrier cost can exceed reuse. The selected add+RMS boundary is the narrow
  exception backed by all-48 actual evidence; do not generalize it.
- **Unchanged IQ2 geometry/reuse repeats:** local32, wave64, tile4, grid64
  repeats, mixed IQ2/Q5 ownership, Q8_1 activation, super-scale publication,
  and partial/generic DPP reductions are closed. Only the candidate-specific
  fixed-local64 reduction above has new positive actual-weight evidence.
- **Launch-count-only fusion:** D11 and D13 are direct counterexamples.
- **Single-prompt promotion:** all future acceptance remains the complete
  category/heldout suite under the repository anti-gaming rule.

## 10. Evidence map

| Question | Evidence |
| --- | --- |
| Is 93.67 tok/s reproducible? | [`...llamacpp-vulkan-review.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-llamacpp-vulkan-review.json) |
| What is the retained hipEngine row? | [`...iq3-wave10-fused-retained.json`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-retained.json) |
| What dominates D12? | D12 clean profile plus [`...d9-residual-profile.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-d9-residual-profile.json) with D12 leaf replacements |
| What did D0–D17 retain/reject? | [`LAGUNA.md`](LAGUNA.md), “Laguna Q2 XL Decode Optimization Campaign” |
| Is IQ2 already tuned? | [`OPTIMIZE-KERNEL-IQ2_XS.md`](OPTIMIZE-KERNEL-IQ2_XS.md) |
| Why is Qwen competitive? | [`...gguf-final-optimization-sweep.json`](../benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json) |
| Which Qwen tactics were considered? | [`OPTIMIZE.md`](OPTIMIZE.md), [`OPTIMIZE-DENSE.md`](OPTIMIZE-DENSE.md), Q3/Q4 artifacts, and `WORKLOG.md` |
| Compact conclusions and logger hashes | [`...decode-gap-analysis.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-decode-gap-analysis.json) |
| What does the corrected HIP/Vulkan history transfer? | [`HIP-vs-VULKAN.md`](HIP-vs-VULKAN.md), [`HIP-vs-VULKAN-HISTORY.md`](HIP-vs-VULKAN-HISTORY.md) |
| What does same-source llama.cpp HIP isolate? | [`...hip-vulkan-isa-attention-review.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-hip-vulkan-isa-attention-review.json), `/tmp/laguna-llamacpp-hip-depth-profile-summary.json` hash therein |
| What is the raw-IQ3 ownership/ISA result? | Same review artifact plus `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.hip` and llama.cpp HIP `mmvq.cu`/`vecdotq.cuh` plus Vulkan `mul_mat_vec.comp`/`dequant_funcs_cm2.glsl` at `c0bc8591e` |
| What is the next attention algorithm? | Same review artifact; llama.cpp `fattn-tile.cuh`/`fattn-common.cuh` at `c0bc8591e`; in-tree `attention/paged_attn_decode.hip` split producer/reducer |
| Did a bit-lossless Q5 repack help? | [`...p3-q5-t16-repack-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p3-q5-t16-repack-rejected.json): exact generic/wave32x2 T16 both regress actual global/SWA layers and are not retained. |
| Does fixed-address Q5 metadata help without a repack? | [`...q5-fixed-metadata-retained.json`](../benchmarks/results/2026-07-25-gfx1100-laguna-q2-xl-q5-fixed-metadata-retained.json): yes. Two uniform 128-bit metadata loads remove 32 coefficient exchanges, reduce logical VGPR **89 -> 72**, improve clean Q5 **22.68-23.12%**, and move complete-suite h32 decode **54.476 -> 57.711 tok/s (+5.938%)** at exact full state. |
| Does one heterogeneous projection dispatch help? | [`...mixed-attention-retained.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-attention-retained.json): yes. Exact Q5/Q6 and Q6/Q8 quads remove **49 launches/token**, improve every clean context, and move complete-suite h32 decode **57.833 -> 58.425 tok/s (+1.024%)**. |
| Does cooperative Q6 metadata publication still help inside that mixed dispatch? | [`...mixed-q6-fixed-metadata-retained.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-q6-fixed-metadata-retained.json): yes. It preserves exact state and 723 dispatches/token, improves clean projection work **8.08-10.10%**, and moves complete-suite h32 decode **58.466 -> 59.211 tok/s (+1.275%)**. |
| Can the same fixed-metadata Q5 owner accelerate shared gate/up? | [`...shared-q5-fixed-metadata-retained.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-shared-q5-fixed-metadata-retained.json): yes. The BF16 pair is byte-exact, improves first/last actual pair event/wall **26.88-27.61%**, improves clean shared-pair work **45.99-47.13%**, and moves complete-suite h32 decode **59.500 -> 60.942 tok/s (+2.425%)** at unchanged 723 dispatches/token. |
| Does that one-wave Q5 body make a mixed IQ2/shared launch viable? | [`...mixed-iq2-q5-local64-rejected.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-iq2-q5-local64-rejected.json): no. All three BF16 outputs are exact and the candidate keeps the retained IQ2 VGPR/LDS ceiling, but both first/last actual layers regress event/wall **0.12-0.68%**; the candidate is removed before runtime integration. |
| Can two exact fixed-metadata Q5 output waves share one local64 workgroup? | [`...q5-output-wave32x4-rejected.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q5-output-wave32x4-rejected.json): no. K6144/K9216 N3072 outputs are byte-exact and LDS/spill-free, but logical VGPR rises **73 -> 81** and all first/last global/SWA event/wall rows regress **6.51-10.15%**; the candidate is removed before runtime integration. |
| Does exact GQA3 value reuse accelerate the SWA reducer? | [`...swa-gqa3-reducer-rejected.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-swa-gqa3-reducer-rejected.json): no. All F32 context and BF16 gated outputs are byte-exact, but reducing **72 -> 24** workgroups regresses every layer-1/46/47 live-70/128/257/512 row by **61.16-89.95%**; the candidate is removed before runtime integration. |
| Does all-local32 ownership improve the mixed Q5/Q6 projection? | [`...mixed-local32-projection-retained.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-local32-projection-retained.json): yes. Exact local32 Q5/Q6 pair owners preserve total threads/waves and full state, improve clean projection work **7.00-8.12%**, and move complete-suite h32 decode **60.900 -> 61.732 tok/s (+1.367%)** at unchanged 723 dispatches/token. |
| Can one local32 wave reuse an activation register across exact Q5 and Q6 output pairs? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-mixed-pair-reuse-rejected.json): primitive only after clean rejection. Exact 16-transition state and **47+1/723** tracing do not override order-A span **+1.265%** and order-B child throughput **-0.681%** failures. Runtime integration is removed; remaining contexts/categories are skipped and defaults/topline are unchanged. |
| Can LDS-staged unrounded F32 sums accelerate Laguna add+RMSNorm exactly? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-add-rmsnorm-staged-f32-rejected.json): primitive only after clean rejection. Actual 48-boundary, CPU, 16-transition state, and 48/723 trace gates pass; the family improves **2.549%/2.501%**, but order A kernel sum regresses **0.340%** and order B span regresses **0.528%**. Runtime integration is removed; remaining contexts/categories are skipped and defaults/topline are unchanged. |
| Can a fixed-two-wave DPP reduction accelerate the retained IQ2 grid64 body exactly? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq2-local64-reduction-rejected.json): primitive only after clean rejection. Synthetic/CPU/all-46-layer bytes, exact 16-transition state, and **46-candidate/723-kernel** tracing pass, but both short orders regress IQ2 **0.366%/2.742%**; order A also fails child and order B fails kernel/span guards. Runtime integration is removed and later contexts/categories are skipped. |
| Can paired-output SWAR contract exact Q5 reconstruction? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q5-swar-pair-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q5-swar-pair-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-pair-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-pair-rejected.json): primitive only after clean rejection. Synthetic/CPU/all-actual bytes, exact 16-transition state, and **140-candidate/723-kernel** tracing pass, but both short orders regress the combined Q5 family **0.457%/0.077%**, kernel sum **0.280%/0.075%**, and child throughput **0.723%/0.592%**; order A also fails span. Runtime integration is removed and later contexts/categories are skipped. |
| What is selected after the Q5 rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-design.json), [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-runtime-correctness.json), and [`retained`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-fused-retained.json): the exact K1024 local320 composite is the retained gfx1100 default. Focused/exhaustive/CPU and **45/45** actual-output gates pass; full/default-vs-wave4 state is exact; tracing proves **45 candidate + two reducers / 678 model kernels/token**. Every clean order and train/heldout category passes; h32 moves **62.318 -> 63.270 tok/s (+1.528%)** versus matched wave4. |
| What happened after wave10 retention? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-design.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-q5-swar-output-only-rejected.json): primitive only. Output-family and kernel-sum time improve in both short orders, but profiled-child throughput regresses **1.061%/1.035%**, beyond the frozen -0.5% guard. Runtime integration is removed; long contexts/categories are skipped. |
| What happened after closing SWAR ownership? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-design.json), [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq3-wave10-signbit-fused-rejected.json): primitive only after clean rejection. Exhaustive/CPU/**45/45** production/full-state/**45/2/678** trace gates pass. Both short and 512 orders improve IQ3 and kernel sum, but 512 order-B span regresses **0.862%**, beyond +0.5%; pooled span **-0.736%** cannot waive it. Runtime integration is removed; 1K/3968/categories are skipped and canonical **63.270 tok/s** remains unchanged. |
| What happened after sign-bit ownership rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-design.json), [`certification`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-iq4-weighted-composite-rejected.json): primitive only after clean rejection. Package/key/backend, edge, CPU, actual-layer, codegen, full-state, and exact **2-IQ4/45-wave10/676-kernel** trace gates pass. Both short orders nevertheless regress inclusive IQ4 **25.515%/25.191%**, kernel sum **0.202%/0.303%**, and span **0.681%/2.198%**. Runtime integration is removed before long contexts/categories; canonical **63.270 tok/s / 678 kernels** remains. |
| What happened after IQ4 ownership rejection? | [`design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-design.json), [`primitive`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-correctness.json), [`runtime`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-rejected.json): primitive only after clean rejection. Primitive/all-layer/full-state and **12-candidate/zero-scalar/678-kernel** trace gates pass, and both short orders improve global attention **10.97%/12.12%**. Order A still regresses kernel sum **0.0949%** and span **1.3203%**, so runtime integration is removed before longer contexts/categories; canonical **63.270 tok/s / 678 kernels** remains. |
| What is selected after one-page ownership rejection? | [`gated design`](../benchmarks/results/2026-07-27-gfx1100-laguna-q2-xl-global-single-page-gated-design.json): compose the admitted page-zero body with D15's exact softplus/RNE-BF16 epilogue inside one global-only local256 workgroup. It does not restore rejected head/KV or SWA bundles. A zero-increment epilogue ceiling contracts **678 -> 666 kernels/token** and models h32 **63.653 tok/s (+0.605%)**; no body, runtime owner, default, or throughput claim exists yet. |
| Does exact local64 dim2 ownership improve the complete clean SWA path? | [`...swa-local64-dim2-reducer-rejected.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-swa-local64-dim2-reducer-rejected.json): no. Primitive/full-state/trace gates pass and short reducer/SWA improve **0.244%/0.060%**, but context-512 reducer/SWA regress **0.073%/0.247%** across both process orders. The frozen any-context rule stops 1K/near-4K and categories; runtime selector/capability integration is removed while the exact primitive remains diagnostic. |
| Does load-free IQ3 sign-bit insertion improve complete clean decode? | [`...iq3-signbit-rejected.json`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-iq3-signbit-rejected.json): not under the frozen rule. Primitive/full-state/trace gates pass, and both short orders improve producer/inclusive/kernel-sum time, but dispatch span regresses **0.571%/1.931%** and order-A profiled-child throughput regresses **1.124%**, outside the 0.5% guards. Remaining profiles/categories stop; runtime schedule/CLI integration is removed while the exact primitive remains diagnostic. |
| Does the post-sign-bit wave-top10 router improve clean full-model decode? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-wave-top10-rejected.json): no. Primitive event/wall improve split **23.26%/23.23%** and old D11 **4.83%/4.84%**, but both clean short orders regress router-family time **14.42%/13.69%** and kernel sum **0.736%/1.422%**. Runtime integration is removed; categories are skipped and the exact primitive remains diagnostic. |
| Is the post-router Q4 LM-head local32 design implemented or retained? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-runtime-correctness.json), and [`retained`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q4-lmhead-local32-fixed-metadata-retained.json): yes. Production logits and default-vs-rollback state are bit-exact; every clean order improves LM-head and kernel-sum time; both category orders move h32 **61.675 -> 61.992 tok/s (+0.512%)** with all category decode rows positive. gfx1100 defaults local32 at unchanged 723 kernels/token. |
| Is the post-Q4 Q6 local32 standalone design implemented? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-q6-local32-standalone-rejected.json): primitive only after clean rejection. Exact state/trace and 12.44-13.26% Q6-family wins do not override order-A kernel/span or order-B child failures. Runtime integration is removed; categories are skipped and the topline is unchanged. |
| What happened to the post-Q6 compact-wave32 selector? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-router-selector-compact-wave32-rejected.json): primitive only. The isolated **25.78%/25.77%** event/wall win reverses under complete clean decode: both short orders regress selector **30.58%/27.60%**, router **16.89%/14.08%**, kernel sum **1.787%/0.591%**, and child throughput **1.587%/1.619%**. Runtime integration is removed and categories are skipped. |
| What happened to the post-compact weighted-hidden split? | [`design`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-design.json), [`primitive`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-correctness.json), [`runtime`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-runtime-correctness.json), and [`rejection`](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-weighted-hidden-split-rejected.json): primitive only after clean rejection. The repository boundary improves **12.664% event/wall**, full state and 47+47/723 tracing pass, and both short orders improve boundary time **19.883%/12.390%**. Order B nevertheless regresses kernel sum **0.320%** and span **6.342%**; runtime integration is removed and remaining contexts/categories are skipped. |
| Does retained hipEngine beat Vulkan under matched natural completion? | No. The retained wave10-fused category gate measures hipEngine **63.951/63.270 tok/s** versus device-pinned Vulkan **64.245/64.418 tok/s** h16/h32; another **0.46%/1.81%** is required. The prior [post-local32 audit](../benchmarks/results/2026-07-26-gfx1100-laguna-q2-xl-vulkan-matched-completion-post-local32.json) remains the pinned Vulkan source. |
| Can a one-doorbell native AQL owner remove the queue gap? | No. [`...p4-aql-submission-rejected.json`](../benchmarks/results/2026-07-24-gfx1100-laguna-q2-xl-p4-aql-submission-rejected.json) measures correctness-fenced direct AQL **0.560-0.758% slower** than HIP across five 820-dispatch processes. |

## Bottom line

The approximately 2x headline gap is genuine enough to guide engineering even
though the two public timing protocols differ. We are **not** losing 2x to
Python, sampling, graph replay, a missing compiler flag, or one unfused router.
The clean GPU trace proves otherwise.

hipEngine has already transferred the broad Qwen playbook and improved Laguna
from **19.596 to 63.270 tok/s**. The expanded review removes two tempting but
wrong shortcuts: neither a generic ACO/Clang upgrade nor a broad Q8_1 switch is
supported by the evidence.

The implementation loop remains ordered and falsifiable: P0 exact IQ3
route/output ownership and P2 exact attention split topology are retained; both
narrow P1 IQ3 lanes, P1 raw-Q5/raw-IQ2 row4, P2.2 online partials, and P3's
bit-lossless Q5 T16 replacement are rejected. The online-attention and raw-Q5
bodies prove tile-level ownership is mechanically valuable but violate the
frozen KL gate; IQ2 row4 and Q5 T16 instead fail the actual-weight performance
precondition. All are removed. The exact SWA tile16 score producer is retained
at live `>=257`; P1-P3 are closed. P4.1's exact gated split reducers and the
current-P4 exact head+KV body are retained after independent-body, full-state,
trace, clean-context, and complete-category gates. The device-pinned matched
reaudit replaces the non-equivalent 94.513-tok/s headline with a formal
**64.418 tok/s** h32 target; current retained hipEngine reaches **63.270 tok/s**
and still needs **1.81%** more. Direct AQL, unchanged graph capture, host
packets, and launch cleanup alone are mechanically closed. Exact Q5
fixed-metadata loads, the subsequent heterogeneous attention-projection quad,
its fixed-Q6-metadata sibling, and the shared-Q5 fixed-metadata pair are
retained after both clean context orders and both complete 18-prompt orders
pass. Exact three-query-head SWA value reuse is now also rejected: it passes the
primitive arithmetic and resource gates but loses **61-90%** on every actual
layer/live boundary when **72 -> 24** workgroups remove query-head parallelism.
The retained one-head wave-local reducer remains canonical. The subsequent
all-local32 Q5/Q6 mixed projection preserves the same total threads/waves while
removing the local128 union's LDS/barriers and resource footprint from
independent Q5 waves. It is retained after exact production/full-state/default-versus-rollback
gates, all-positive clean projection/kernel/span/child rows, and both
complete 18-prompt orders. Cached tracing records local32/VGPR80/LDS0/scratch0
at unchanged 723 model kernels/token. The later retained IQ3 wave10 composite
contracts that topology to 678 kernels/token and moves the objective to
**63.270 versus 64.418 tok/s**. The exact local64 packed-dim2 SWA path preserves all 72
query-head workgroups and passes primitive, codegen, actual-weight, full-state,
and cached-trace gates, but its complete clean context-512 reducer/SWA regresses
**0.073%/0.247%**. The frozen gate rejects runtime promotion; the selector and
capability are removed, categories are skipped, and local128 remains canonical.
The subsequent load-free exact IQ3 sign-bit insertion sibling passes
primitive, codegen, actual-weight, shared-weight 16-transition full-state, and
cached 45-producer/47-reducer/723-kernel trace gates. Its frozen clean gate does
not pass: both short orders improve the targeted producer, inclusive family,
and kernel sum, but dispatch span regresses **0.571%/1.931%** and order-A
profiled-child throughput regresses **1.124%**, outside the 0.5% guards. Per the
predeclared any-failure rule, remaining profiles and categories are skipped;
the runtime schedule/CLI route is removed, the primitive remains diagnostic,
and retained wave4 stays canonical at **61.732 tok/s**. The materially revised
D11 wave-top10 composition is now repository primitive-admitted. Hidden-17/3072
synthetic cases and every actual router are byte-exact, codegen meets the frozen
VGPR/SGPR/LDS/spill limits, and fresh all-layer event/wall improves
**23.26%/23.23%** versus split and **4.83%/4.84%** versus old D11. Its
explicit default-off runtime owner also passed full state and a 47-call/
676-model-kernel trace, but both frozen short clean orders regress the complete
router family **14.42%/13.69%** and kernel sum **0.736%/1.422%**. Runtime
selection/counter ownership is removed, categories are skipped, and the split
route plus canonical **61.732 tok/s** topline remain unchanged. The subsequent
exact local32 fixed-metadata Q4 LM-head output-pair sibling now passes repository
primitive, production actual-weight, full-state, cached
one-call/723-kernel trace, every clean-context order, both complete category
orders, and a fresh no-argument default-versus-local128 rollback gate. It is the
gfx1100 default and moves canonical h32 **61.732 -> 61.992 tok/s (+0.420%)** at
**16.131 ms/token**. Matched Vulkan remains **64.418 tok/s**, so the objective
stays open with **3.91%** additional throughput required. The subsequent
standalone-Q6 owner is clean-rejected and removed despite an isolated Q6-family
win. The next selected exact premise avoids that owner and the rejected
persistent router entirely: a stateless local32 correction selector consumes
the unchanged registered projection, assigns eight experts per lane, and finds
the same stable global top-10 in one wave. The repository primitive is now
admitted after exact synthetic/CPU/all-47-layer gates, frozen codegen and cached
trace; its actual selector window improves **25.78% event / 25.77% wall**. Its
temporary default-off owner also passes exact 16-transition state, ownership,
and **47 projection + 47 selector / 723-kernel** tracing, but both clean short
orders regress selector/router/kernel/child time. Runtime integration is removed
and categories are skipped. The primitive remains diagnostic; canonical h32
stays **61.992 tok/s** versus Vulkan **64.418 tok/s**. The next selected premise
moves to a distinct MoE boundary: a top10-unrolled local32 weighted+routed/
hidden producer followed by the unchanged registered RMSNorm. The repository
primitive now passes RED/GREEN, synthetic/CPU/all-47-layer exactness, codegen,
and cache-only trace gates; its isolated two-call window improves **12.664%
event / 12.664% wall** at unchanged launch count. Its temporary default-off
owner also passes 16-transition state and **47 producer + 47 registered RMS /
723-kernel** tracing, but clean short order B regresses kernel sum **0.320%** and
span **6.342%**. Runtime integration is removed and categories are skipped; the
primitive remains diagnostic and canonical h32 stays **61.992 tok/s**. The
subsequent heterogeneous pair-reuse primitive shares each BF16 activation
register across one exact Q5 and Q6 output pair while preserving both retained
arithmetic trees. Its temporary default-off owner passes exact 16-transition
state and cache-only **47 candidate + one unchanged layer-47 / 723-kernel**
tracing, but the frozen short gate fails order A on span **+1.265%** and order B
on child throughput **-0.681%**. Runtime integration is removed and remaining
contexts/categories are skipped; the primitive remains diagnostic and canonical
h32 stays **61.992 tok/s** versus Vulkan **64.418 tok/s**. The next selected
exact boundary reuses the already-computed unrounded F32 hidden+attention value
inside each of 48 local256 add+RMSNorm calls. All actual norm/residual boundaries
are bit-exact and the repository complete window improves **3.53%/3.70%**
event/wall. Primitive RED/GREEN, CPU, codegen, full-state, and cache-only 48/723
trace gates pass, but clean short order A regresses kernel sum **0.340%** and
order B regresses span **0.528%**. Runtime selection is removed and later
contexts/categories are skipped; the primitive remains diagnostic and no
retained topline changes. The next selected exact IQ2 design keeps the retained
two-wave local64/grid64 ownership and changes only reduction transport plus its
compile-time two-wave tail. First/last actual layers are BF16-bit exact and
improve **0.96-1.22% event / 1.05-1.26% wall**; codegen removes all 20 wave LDS
permutations at unchanged logical VGPR110 and scratch0. The separately
registered repository primitive now passes RED/GREEN, CPU, all-46-layer,
repeated-endpoint, codegen, and distinct-symbol trace gates. Its temporary
explicit/default-off owner also passes full-state and cache-only
**46-candidate/723-model-kernel** admission, but both clean short orders regress
the IQ2 family and fail additional child or kernel/span guards. Runtime
integration is removed and later contexts/categories are skipped; the primitive
remains diagnostic and canonical h32 stays **61.992 tok/s** versus Vulkan
**64.418 tok/s**. Post-IQ2 paired-output SWAR Q5 reconstruction shares
nibble/high-bit integer work across the two rows already owned by each local32
wave. Its separately registered repository primitive now passes RED/GREEN,
independent CPU, all **47 output + 47 mixed + 46 shared** actual-boundary,
repeated-endpoint, codegen, and cache-only sibling-trace gates. Its temporary
explicit/default-off all-or-none owner also passes exact 16-transition full
state and cache-only **140-candidate/723-kernel** tracing, but both frozen short
orders regress the combined Q5 family, kernel sum, and child throughput; order A
also fails span. Runtime integration is removed, later contexts/categories are
skipped, the primitive remains diagnostic, and canonical h32 stays **61.992
tok/s** versus Vulkan **64.418 tok/s**. The next selected boundary returns to
IQ3 without repeating sign insertion, Q8_1, row4, or local-size experiments: a
local320 workgroup keeps ten retained route waves parallel, stores their exact
BF16 results in a 20-byte LDS tuple, then performs the unchanged slot-order
weighted reduction. First/last actual layers improve **7.25-8.31%** in both
timers with exact output, and cache-only tracing confirms local320/VGPR88/
LDS512/scratch0. The separately registered repository primitive now also passes
focused RED/GREEN, independent CPU, exhaustive selector/grid/BF16/routing,
**45/45** actual-layer, repeated endpoint, integrated codegen, and cache-only
repository-symbol gates. Its separately admitted owner also passes shared-weight
16-transition full state and cache-only **45-candidate/2-reducer/678-kernel** tracing with no
candidate prefill or compiler. All eight clean context orders pass, improving
inclusive IQ3 **9.71-11.90%**, kernel sum **0.398-1.082%**, and span
**0.813-1.998%** with child throughput inside guard. Both complete 18-prompt
orders pass every train/heldout category at h16/h32 and move matched h32
**62.318 -> 63.270 tok/s (+1.528%)**. gfx1100 now defaults wave10-fused with
explicit wave4 rollback; canonical h32 is **63.270 tok/s**, still **1.81%**
below Vulkan **64.418**. Post-wave10 ranking closes SWA local32/dim4 after
**9/12** event-and-wall regressions. The subsequent output-only Q5 SWAR owner
passes exact state and **47-candidate/678-kernel** tracing, improves output and
kernel sum in both short orders, but fails both profiled-child guards at
**-1.061%/-1.035%**. Runtime integration is removed before long contexts or
categories; all SWAR owners remain diagnostic primitives only. Three fresh
post-SWAR screens then close IQ3 two-output wave10, one-barrier SWA full-weight
staging, and mixed-Q5/Q6 K3072 specialization on required endpoint regressions.
The selected next candidate composes load-free sign reconstruction inside the
retained wave10 body while preserving the exact **45/2/678** topology. Its
separately registered repository primitive passes RED/GREEN, exhaustive/CPU,
**45/45** production output, repeated endpoint, codegen, and cache-only trace
gates. Repository layers 1/45 improve **7.57-8.95%** across event/wall medians,
codegen contracts **495 -> 454 instructions**, and the distinct symbol remains
in allocated VGPR88/LDS512/scratch0. The conservative modeled row is **63.685
tok/s**, still **1.15%** below Vulkan and not a performance claim. A temporary
runtime owner passes exact shared-weight 16-transition state and cache-only **45
candidate + two reducer / 678-model-kernel** tracing, but is rejected at the
frozen context-512 order-B span guard (**+0.862%** versus a +0.5% ceiling).
Pooled span is favorable but cannot waive the per-order failure. Runtime
integration is removed before 1K/3968/categories; the exact primitive remains
diagnostic and retained wave10 remains canonical. Post-sign-bit re-ranking then
selects the pre-existing exact IQ4_XS routing-weighted composite. Dedicated
Laguna top-10/K1024 certification passes **7/7**, independent CPU quality, both
actual IQ4 layers, repeated event/wall, frozen codegen, and cache-only expected-
symbol gates without changing the predating gfx1100 body/wrapper/key. Both real
outputs are byte-exact and the isolated selected-plus-reducer boundary improves
**28.59-34.22%**, at local256/VGPR80/LDS512/scratch0. A temporary default-off
owner passes exact 16-transition state and **2-candidate/45-wave10/676-kernel**
tracing, but both frozen short orders regress inclusive IQ4 **25.515%/25.191%**,
kernel sum **0.202%/0.303%**, and span **0.681%/2.198%**. Runtime integration
is removed before long contexts/categories; the primitive remains diagnostic
and canonical **63.270 tok/s / 678 kernels** is unchanged. Post-IQ4 ranking then
selects an exact single-page scalar global-attention sibling without reopening
rejected head pairing, online reassociation, or fused attention boundaries.
Actual layers 0/44 at live 70/126 are F32 bit-exact and improve all event/wall
medians **15.52-17.75%**; codegen contracts **1,705 -> 1,008 instructions** and
logical VGPR **35 -> 32**. Primitive/all-layer/full-state and exact **12/678**
trace gates pass, but clean order A regresses kernel sum **0.0949%** and span
**1.3203%**. Runtime ownership is removed before long contexts/categories and
canonical **63.270 tok/s / 678 kernels** is unchanged. Post-rejection ranking
selects a materially new global-only composition: append D15's exact softplus/
RNE-BF16 epilogue to the admitted page-zero body. A zero-increment epilogue
ceiling contracts **678 -> 666 kernels/token** and models **63.653 tok/s**. The
gated primitive is now admitted after exact synthetic/CPU/all-12-layer,
all-positive endpoint, codegen, and cache-only trace gates, but runtime ownership
and the canonical **63.270 tok/s / 678-kernel** topline remain unchanged.
